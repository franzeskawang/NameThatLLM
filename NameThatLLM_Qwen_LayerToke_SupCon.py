"""
以 TaskLoRA + Queue-based Prompt-Aware SupCon（去主題監督式對比學習）微調 Qwen-3-Embedding
(NameThatLLM_Qwen_LayerToke_SupCon.py)：
1. 承襲 NameThatLLM_Qwen_LayerToken.py 的前 24 層截斷 (KEEP_LAYERS=24)、階層式雙重注意力架構
   (HierarchicalLayerTokenClassifier) 與原本穩定的 nn.Linear + CrossEntropyLoss 主分類頭：
   - 第一重【Layer Attention】：8 層樓 (3, 6, 9, 12, 15, 18, 21, 24) 獨立降維並以 torch.cat 拼接成 256 維獨立車道
   - 第二重【All-Token Attention】：Multi-Head Self-Attention + Gated Token Attention 匯聚為 512 維文章表徵
2. 新增【Queue-based Prompt-Aware SupCon 輔助對比分支】：
   - 保留原 nn.Linear(512 -> 18) 主分類頭不變，另由 512 維特徵接出輕量 Projection Head (512 -> 256 -> 128)
     並做 L2 正規化，總損失為：Total Loss = CE Loss + SUPCON_WEIGHT * Prompt-Aware SupCon Loss。
   - 在 GPU 維護 FIFO Feature Queue (預設 256 筆，顯存不到 0.2 MB)，完美解決 MICRO_BATCH_SIZE = 2 無法做
     批次內對比的問題。
   - 善用 prompt_id 設計「去主題」正負樣本：
     * 正樣本（拉近）：同 model 且「不同 prompt_id」→ 強迫模型丟掉主題內容，只保留寫作指紋。
     * 超級困難負樣本（加倍推開，權重 2.5x）：不同 model 且「同一 prompt_id」→ 內容相同，強迫放大細微文風差異。
     * 一般負樣本（正常推開，權重 1.0x）：不同 model 且不同 prompt_id。
   - 搭配 Prompt-Block 洗牌排程（每 4 道題目混洗成一個區塊），確保佇列中隨時同時存在跨題目正樣本與同題目困難負樣本。
3. 推論與驗證階段完全不執行 Projection Head，維持與原版一致的推論速度與 logits 校準特性，
   並保留完整的【8 層樓】與【關鍵 Token】雙重分析報告。
"""

from __future__ import annotations

import collections
import gc
import glob
import json
import os
import pathlib
import random
from typing import Iterable, Sequence

# 啟用 PyTorch CUDA 動態顯存區段擴展，大幅減少顯存碎片化導致的 OOM。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


# Kaggle 訓練資料可能掛載在以下任一資料路徑。
CANDIDATES = [
    pathlib.Path("/kaggle/input/competitions/name-that-llm"),
    pathlib.Path("/kaggle/input/name-that-llm"),
]

# 預設 Qwen 模型路徑（若不存在，程式會自動在 /kaggle/input 下搜尋 Qwen-3-Embedding 或 Qwen-3 模型）。
DEFAULT_MODEL_PATH = os.environ.get(
    "NTL_MODEL_PATH",
    "/kaggle/input/qwen-3-embedding/transformers/8b/1",
)

# 對齊 native_q8.py 的風格辨識指令。
INSTRUCTION = (
    "Instruct: Identify the language model that generated the text based on its writing style.\n"
    "Query: "
)

# 只保留前 24 層（第 25～36 層直接不載入、也不運算）。
KEEP_LAYERS = 24
# 在前 24 層中每 3 層設置一個監聽哨（共 8 個目標樓層）。
DEFAULT_DEPTHS = (3, 6, 9, 12, 15, 18, 21, 24)

LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# 訓練與雙重注意力模型超參數設定。
SEED = 42
VALIDATION_FRACTION = 0.20
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 512
LORA_RANK = 16
LORA_SCALE = 2.0

# 每一層樓各自降維至 LAYER_PROJ_DIM (32 維)，8 層樓以 torch.cat 拼接後總維度剛好是 32 * 8 = 256 維 (PROJ_DIM)！
LAYER_PROJ_DIM = 32
NUM_HEADS = 4
ATTN_DIM = 128
DROPOUT = 0.10

# 第一重 Layer Attention 的隨機休假與溫度平滑參數。
LAYER_DROP_PROB = 0.10
LAYER_TEMPERATURE = 2.0

# 第二重 Token Attention 的隨機休假與溫度平滑參數。
TOKEN_DROP_PROB = 0.15
TOKEN_TEMPERATURE = 1.5

# Queue-based Prompt-Aware SupCon 輔助對比學習超參數。
SUPCON_WEIGHT = 0.10                # 輔助損失權重 λ（輕量引導，避免干擾主 CE 收斂）
SUPCON_TEMPERATURE = 0.10           # 對比學習溫度參數 τ
SUPCON_PROJ_DIM = 128               # Projection Head 輸出維度
SUPCON_QUEUE_SIZE = 256             # 記憶體佇列大小（約可容納 14 道題目的 18 模型回答，僅佔 ~0.13 MB）
SUPCON_SAME_PROMPT_NEG_WEIGHT = 2.5 # 「同題目、不同模型」超級困難負樣本的加權倍率
SUPCON_PROMPT_BLOCK_SIZE = 4        # 每次將 4 道題目的樣本混洗為一個區塊，確保佇列同時具備跨題正樣本與同題困難負樣本


def _seed_everything(seed: int) -> None:
    """固定常用亂數來源，讓資料切分與模型初始化可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_data_dir() -> pathlib.Path:
    """依序尋找包含 train.csv 的 Kaggle 資料目錄。"""
    for candidate in CANDIDATES:
        if (candidate / "train.csv").exists():
            return candidate
    searched = ", ".join(str(path) for path in CANDIDATES)
    raise FileNotFoundError(f"Could not find train.csv in: {searched}")


def _find_model_dir(preferred_path: str = DEFAULT_MODEL_PATH) -> str:
    """尋找 Qwen-3-Embedding 或 Qwen-3 模型目錄。"""
    if os.path.isdir(preferred_path) and os.path.exists(
        os.path.join(preferred_path, "config.json")
    ):
        return preferred_path

    configs = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    # 優先順序：qwen-3-embedding/8b -> qwen-3-embedding -> qwen-3/8b -> 任何 qwen3
    search_patterns = [
        ("/qwen-3-embedding/", "/8b/"),
        ("qwen-3-embedding",),
        ("/qwen-3/", "/8b"),
        ("qwen3",),
    ]
    for parts in search_patterns:
        hits = [
            os.path.dirname(c)
            for c in configs
            if all(p in c.lower() for p in parts)
        ]
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"找不到 Qwen 模型目錄（已檢查 {preferred_path} 與 /kaggle/input/**/config.json）。"
        "請設定 NTL_MODEL_PATH 環境變數或掛載 Qwen-3-Embedding 模型。"
    )


def _resolve_depths(num_hidden_layers: int, target_count: int = 8) -> tuple[int, ...]:
    """若模型截斷後為 24 層則使用 (3, 6, 9, 12, 15, 18, 21, 24)，否則自動等距選出 8 個樓層。"""
    if (
        num_hidden_layers == KEEP_LAYERS
        and len(DEFAULT_DEPTHS) == target_count
        and max(DEFAULT_DEPTHS) <= num_hidden_layers
    ):
        return DEFAULT_DEPTHS
    depths = np.linspace(1, num_hidden_layers, num=target_count, dtype=int).tolist()
    return tuple(sorted(set(depths)))


def _group_split(
    groups: Sequence[object], labels: Sequence[int], fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """以 prompt_id 為單位切分訓練集與驗證集，避免同一題目洩漏並確保兩側皆含所有 18 個類別。"""
    group_to_indices: dict[object, list[int]] = {}
    for index, group in enumerate(groups):
        group_to_indices.setdefault(group, []).append(index)

    unique_groups = list(group_to_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    target = max(1, round(len(unique_groups) * fraction))
    all_classes = set(labels)
    validation_groups: list[object] = []
    validation_classes: set[int] = set()
    remaining_groups = unique_groups.copy()

    while remaining_groups and (
        len(validation_groups) < target or validation_classes != all_classes
    ):
        group = remaining_groups.pop()
        validation_groups.append(group)
        validation_classes.update(labels[i] for i in group_to_indices[group])

    validation_set = set(validation_groups)
    train_indices = np.asarray(
        [
            i
            for group, indices in group_to_indices.items()
            if group not in validation_set
            for i in indices
        ],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [i for group in validation_groups for i in group_to_indices[group]],
        dtype=np.int64,
    )
    train_classes = {labels[i] for i in train_indices.tolist()}
    validation_classes = {labels[i] for i in validation_indices.tolist()}
    if train_classes != all_classes or validation_classes != all_classes:
        raise ValueError(
            "Could not create a prompt-group split containing every class in both sets"
        )
    return train_indices, validation_indices


def _build_prompt_aware_order(
    train_indices: np.ndarray,
    prompt_codes: np.ndarray,
    rng: np.random.Generator,
    block_size: int = SUPCON_PROMPT_BLOCK_SIZE,
) -> np.ndarray:
    """
    建立 Prompt-Aware 訓練樣本順序：
    將所有訓練 prompt 洗牌後，每 block_size 個 prompt 組成一個區塊並在區塊內打散。
    如此每個 gradient accumulation 區間仍包含多題多模型，同時 Feature Queue 內永遠具備
    大量「同題目不同模型（超級困難負樣本）」與「不同題目同模型（去主題正樣本）」。
    """
    prompt_to_indices: dict[int, list[int]] = {}
    for idx in train_indices.tolist():
        p = int(prompt_codes[idx])
        prompt_to_indices.setdefault(p, []).append(idx)

    unique_prompts = list(prompt_to_indices.keys())
    rng.shuffle(unique_prompts)

    step = max(1, int(block_size))
    ordered: list[int] = []
    for start in range(0, len(unique_prompts), step):
        block_prompts = unique_prompts[start : start + step]
        block_indices = [
            idx for p in block_prompts for idx in prompt_to_indices[p]
        ]
        rng.shuffle(block_indices)
        ordered.extend(block_indices)
    return np.asarray(ordered, dtype=np.int64)


def _balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """計算 18 個類別召回率的平均值（Balanced Accuracy）。"""
    recalls = []
    for class_id in range(num_classes):
        mask = y_true == class_id
        if mask.any():
            recalls.append(float((y_pred[mask] == class_id).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _log_loss_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """以數值穩定的 log-softmax 計算多分類交叉熵（LogLoss）。"""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if len(y) == 0:
        return 0.0
    z_shift = z - z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(z_shift).sum(axis=1))
    nll = log_sum_exp - z_shift[np.arange(len(y)), y]
    return float(nll.mean())


class PromptAwareSupConQueue:
    """
    輕量級 FIFO Feature Queue + Prompt-Aware 監督式對比損失：
    - 專為 MICRO_BATCH_SIZE = 2 設計，將歷史樣本的 (z.detach(), label, prompt_id) 暫存於 GPU 佇列。
    - 正樣本：同模型 (label 相同) 且不同題目 (prompt_id 不同) -> 強迫剝離主題、聚焦寫作風格。
    - 困難負樣本：不同模型且同一題目 (prompt_id 相同) -> 在分母中給予 same_prompt_neg_weight 倍權重。
    - 一般負樣本：不同模型且不同題目 -> 權重 1.0。
    """

    def __init__(
        self,
        proj_dim: int = SUPCON_PROJ_DIM,
        queue_size: int = SUPCON_QUEUE_SIZE,
        temperature: float = SUPCON_TEMPERATURE,
        same_prompt_neg_weight: float = SUPCON_SAME_PROMPT_NEG_WEIGHT,
        device: torch.device | str = "cpu",
    ) -> None:
        self.proj_dim = int(proj_dim)
        self.queue_size = max(1, int(queue_size))
        self.temperature = max(float(temperature), 1e-4)
        self.same_prompt_neg_weight = max(1.0, float(same_prompt_neg_weight))
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """每個 Epoch 開始前清空佇列，避免舊 Epoch 的過時表徵干擾對比。"""
        self.feats = torch.zeros(
            (self.queue_size, self.proj_dim), dtype=torch.float32, device=self.device
        )
        self.labels = torch.full(
            (self.queue_size,), -1, dtype=torch.long, device=self.device
        )
        self.prompts = torch.full(
            (self.queue_size,), -1, dtype=torch.long, device=self.device
        )
        self.valid = torch.zeros(
            (self.queue_size,), dtype=torch.bool, device=self.device
        )
        self.ptr = 0

    @torch.no_grad()
    def enqueue(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        prompts: torch.Tensor,
    ) -> None:
        """將當前 micro-batch 的 detached 投影向量、模型標籤與題目編號寫入環形佇列。"""
        z = feats.detach().float().to(self.device)
        y = labels.detach().long().to(self.device)
        p = prompts.detach().long().to(self.device)
        n = int(z.shape[0])
        if n == 0:
            return
        if n >= self.queue_size:
            self.feats.copy_(z[-self.queue_size :])
            self.labels.copy_(y[-self.queue_size :])
            self.prompts.copy_(p[-self.queue_size :])
            self.valid.fill_(True)
            self.ptr = 0
            return

        end = self.ptr + n
        if end <= self.queue_size:
            self.feats[self.ptr : end] = z
            self.labels[self.ptr : end] = y
            self.prompts[self.ptr : end] = p
            self.valid[self.ptr : end] = True
        else:
            first = self.queue_size - self.ptr
            rest = n - first
            self.feats[self.ptr :] = z[:first]
            self.labels[self.ptr :] = y[:first]
            self.prompts[self.ptr :] = p[:first]
            self.valid[self.ptr :] = True
            self.feats[:rest] = z[first:]
            self.labels[:rest] = y[first:]
            self.prompts[:rest] = p[first:]
            self.valid[:rest] = True
        self.ptr = end % self.queue_size

    def compute_loss(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        prompts: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算當前 micro-batch（帶梯度）相對於「同 batch 其他樣本 + 歷史佇列」的 Prompt-Aware SupCon Loss。
        若佇列暖機初期尚無跨題正樣本或負樣本，則安全回傳 0。
        """
        z = F.normalize(feats.float(), p=2, dim=-1)
        y = labels.long()
        p = prompts.long()
        bsz = int(z.shape[0])

        if self.valid.any():
            q_feats = self.feats[self.valid]
            q_labels = self.labels[self.valid]
            q_prompts = self.prompts[self.valid]
            key_feats = torch.cat([z.detach(), q_feats], dim=0)
            key_labels = torch.cat([y, q_labels], dim=0)
            key_prompts = torch.cat([p, q_prompts], dim=0)
        else:
            key_feats = z.detach()
            key_labels = y
            key_prompts = p

        total_keys = int(key_feats.shape[0])
        if total_keys <= 1:
            return z.new_tensor(0.0)

        # 排除自身對自身 (batch 內前 bsz 個位置與查詢一一對應)
        not_self = torch.ones((bsz, total_keys), dtype=torch.bool, device=z.device)
        arange_b = torch.arange(bsz, device=z.device)
        not_self[arange_b, arange_b] = False

        same_label = y.unsqueeze(1) == key_labels.unsqueeze(0)
        same_prompt = p.unsqueeze(1) == key_prompts.unsqueeze(0)

        # 正樣本：同模型 + 不同題目（去主題！）
        pos_mask = same_label & (~same_prompt) & not_self
        # 困難負樣本：不同模型 + 同一題目（加倍推開！）
        hard_neg_mask = (~same_label) & same_prompt & not_self
        # 一般負樣本：不同模型 + 不同題目
        easy_neg_mask = (~same_label) & (~same_prompt) & not_self

        # 只有同時存在至少 1 個正樣本與至少 1 個負樣本的查詢才計入 Loss
        has_pos = pos_mask.any(dim=1)
        has_neg = (hard_neg_mask | easy_neg_mask).any(dim=1)
        valid_anchor = has_pos & has_neg
        if not valid_anchor.any():
            return z.new_tensor(0.0)

        # 組合權重矩陣：正樣本 1.0、一般負樣本 1.0、同題目困難負樣本 same_prompt_neg_weight，其餘 0.0
        weights = torch.zeros((bsz, total_keys), dtype=z.dtype, device=z.device)
        weights = weights.masked_fill(pos_mask | easy_neg_mask, 1.0)
        weights = weights.masked_fill(hard_neg_mask, self.same_prompt_neg_weight)

        sim = torch.matmul(z, key_feats.T) / self.temperature  # (B, B + Q)
        # 數值穩定：僅針對權重 > 0 的候選減去最大值
        active_mask = weights > 0.0
        sim_masked = sim.masked_fill(~active_mask, -1e9)
        sim_max = sim_masked.max(dim=1, keepdim=True).values.detach()
        sim_shifted = sim - sim_max

        exp_weighted = torch.exp(sim_shifted.clamp(max=50.0)) * weights
        log_denom = torch.log(exp_weighted.sum(dim=1, keepdim=True).clamp_min(1e-12))
        log_prob = sim_shifted - log_denom

        pos_count = pos_mask.float().sum(dim=1).clamp_min(1.0)
        anchor_loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
        return anchor_loss[valid_anchor].mean()


class TaskLoRA(nn.Module):
    """對齊 native_q8.py 的輕量級 LoRA 外掛層，凍結原始權重並訓練低秩矩陣 A 與 B。"""

    def __init__(
        self, base_layer: nn.Linear, rank: int = LORA_RANK, scale: float = LORA_SCALE
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.scale = scale
        dev = base_layer.weight.device
        self.a = nn.Linear(
            base_layer.in_features, rank, bias=False, device=dev, dtype=torch.float32
        )
        self.b = nn.Linear(
            rank, base_layer.out_features, bias=False, device=dev, dtype=torch.float32
        )
        # 標準 LoRA 初始化：A 使用 Kaiming 初始化、B 設為 0，確保初始狀態不破壞原模型輸出。
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # 顯存優化：將小型的 LoRA 權重矩陣 (rank=16) 轉為 x.dtype 進行降維與升維，梯度仍回傳至 FP32 權重。
        low_rank = F.linear(x, self.a.weight.to(x.dtype))
        lora_delta = F.linear(low_rank, self.b.weight.to(x.dtype)) * self.scale
        return base_out + lora_delta


class HierarchicalLayerTokenClassifier(nn.Module):
    """
    階層式「8 層樓 Layer Attention + 全序列 512 Token Attention + SupCon Projection 分支」分類器：
    1. 各層專屬 LayerNorm(4096) 與獨立降維 Linear(4096 -> 32)。
    2. 第一重【Layer Attention + torch.cat 獨立車道拼接】：8 層樓拼接為 256 維。
    3. 第二重【All-Token Self-Attention + Gated Token Attention】：匯聚為 512 維文章表徵。
    4. 主分類頭：保留原本穩定的 nn.Linear(512 -> 18) 輸出 18 類 logits。
    5. 輔助對比投影頭 (proj_head)：僅於訓練期將 512 維特徵投影至 128 維單位球面，供 Prompt-Aware SupCon 使用。
    """

    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        num_classes: int = 18,
        layer_proj_dim: int = LAYER_PROJ_DIM,
        num_heads: int = NUM_HEADS,
        attn_dim: int = ATTN_DIM,
        dropout: float = DROPOUT,
        layer_drop_prob: float = LAYER_DROP_PROB,
        layer_temperature: float = LAYER_TEMPERATURE,
        token_drop_prob: float = TOKEN_DROP_PROB,
        token_temperature: float = TOKEN_TEMPERATURE,
        supcon_proj_dim: int = SUPCON_PROJ_DIM,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depths = tuple(int(d) for d in depths)
        self.num_layers = len(self.depths)
        self.layer_proj_dim = int(layer_proj_dim)
        self.proj_dim = self.layer_proj_dim * self.num_layers  # 預設 32 * 8 = 256 維
        if self.proj_dim % num_heads != 0:
            raise ValueError(
                f"Total concatenated proj_dim ({self.proj_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_classes = num_classes
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.layer_drop_prob = float(layer_drop_prob)
        self.layer_temperature = max(float(layer_temperature), 1e-4)
        self.token_drop_prob = float(token_drop_prob)
        self.token_temperature = max(float(token_temperature), 1e-4)
        self.supcon_proj_dim = int(supcon_proj_dim)

        # 1. 每一層樓專屬的 LayerNorm(4096) 與專屬降維投影 Linear(4096 -> 32)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size, dtype=torch.float32) for _ in self.depths]
        )
        self.layer_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, self.layer_proj_dim, dtype=torch.float32),
                    nn.GELU(),
                    nn.LayerNorm(self.layer_proj_dim, dtype=torch.float32),
                )
                for _ in self.depths
            ]
        )

        # 2. 第一重 Layer Attention 打分網路
        self.layer_gate = nn.Sequential(
            nn.Linear(hidden_size, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )

        # 3. 第二重 Token-to-Token Multi-Head Self-Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.proj_dim,
            nhead=num_heads,
            dim_feedforward=self.proj_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            dtype=torch.float32,
        )
        self.self_attn = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

        # 4. 第二重 Gated Token Attention 打分網路
        self.token_gate = nn.Sequential(
            nn.Linear(self.proj_dim, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )

        # 5. 主分類器：保留原本穩定的 Linear(512 -> 18)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.proj_dim * 2, num_classes, dtype=torch.float32
        )

        # 6. 輔助對比學習投影頭 (僅訓練期使用，推論期不執行、不影響原本 logits 校準)
        self.proj_head = nn.Sequential(
            nn.Linear(self.proj_dim * 2, self.proj_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(self.proj_dim, self.supcon_proj_dim, dtype=torch.float32),
        )

    def forward(
        self,
        layer_seq_features: list[torch.Tensor],
        attention_mask: torch.Tensor,
        return_details: bool = False,
        return_proj: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        layer_seq_features: 長度為 8 的列表，每個元素為該樓層的完整序列 (Batch, Seq_Len, 4096)
        attention_mask: (Batch, Seq_Len)，1 代表有效 Token，0 代表 Padding
        return_details: 若為 True，回傳 (logits, layer_weights, token_weights)
        return_proj: 若為 True（訓練期），回傳 (logits, l2_normalized_proj) 供 Prompt-Aware SupCon 使用
        """
        if len(layer_seq_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer sequence features, got {len(layer_seq_features)}"
            )

        valid_mask = attention_mask.bool()  # (Batch, Seq_Len)
        batch_size = attention_mask.shape[0]
        pos = attention_mask.sum(dim=1) - 1
        rows = torch.arange(batch_size, device=attention_mask.device)

        proj_seqs: list[torch.Tensor] = []
        layer_summaries: list[torch.Tensor] = []

        for i in range(self.num_layers):
            seq_feat = layer_seq_features[i]
            normed = self.layer_norms[i](seq_feat.float())
            layer_summaries.append(normed[rows, pos])  # (Batch, 4096)

            p_seq = self.layer_projs[i](normed)  # (Batch, Seq_Len, 32)
            p_seq = p_seq.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            proj_seqs.append(p_seq)
            del seq_feat, normed

        stacked_summaries = torch.stack(layer_summaries, dim=1)  # (Batch, 8, 4096)
        layer_scores = (
            self.layer_gate(stacked_summaries) / self.layer_temperature
        )  # (Batch, 8, 1)

        if self.training and self.layer_drop_prob > 0.0:
            layer_drop_mask = (
                torch.rand_like(layer_scores) < self.layer_drop_prob
            )
            all_layers_dropped = layer_drop_mask.all(dim=1, keepdim=True)
            if all_layers_dropped.any():
                for b_idx in torch.where(
                    all_layers_dropped.squeeze(-1).squeeze(-1)
                )[0]:
                    keep_idx = int(
                        torch.randint(
                            0, self.num_layers, (1,), device=layer_scores.device
                        ).item()
                    )
                    layer_drop_mask[b_idx, keep_idx, 0] = False
            layer_scores = layer_scores.masked_fill(layer_drop_mask, -1e9)

        layer_weights = F.softmax(layer_scores, dim=1)  # (Batch, 8, 1)

        scaled_seqs = [
            proj_seqs[i]
            * (layer_weights[:, i, :].unsqueeze(1) * self.num_layers)
            for i in range(self.num_layers)
        ]
        x = torch.cat(scaled_seqs, dim=-1)  # (Batch, Seq_Len, 256)
        del proj_seqs, scaled_seqs, stacked_summaries, layer_summaries

        x = self.self_attn(x, src_key_padding_mask=~valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        raw_token_scores = (
            self.token_gate(x).squeeze(-1) / self.token_temperature
        )
        masked_token_scores = raw_token_scores.masked_fill(~valid_mask, -1e9)

        if self.training and self.token_drop_prob > 0.0:
            token_drop_mask = (
                torch.rand_like(masked_token_scores) < self.token_drop_prob
            ) & valid_mask
            remained_valid = valid_mask & (~token_drop_mask)
            all_tokens_dropped = ~remained_valid.any(dim=1)
            if all_tokens_dropped.any():
                token_drop_mask[all_tokens_dropped] = False
            masked_token_scores = masked_token_scores.masked_fill(
                token_drop_mask, -1e9
            )

        token_weights = F.softmax(masked_token_scores, dim=1)
        token_weights = token_weights.masked_fill(~valid_mask, 0.0)

        attn_pooled = (x * token_weights.unsqueeze(-1)).sum(dim=1)  # (Batch, 256)
        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pooled = x.sum(dim=1) / denom  # (Batch, 256)

        combined = torch.cat([attn_pooled, mean_pooled], dim=-1)  # (Batch, 512)
        dropped = self.dropout(combined)
        logits = self.classifier(dropped)  # (Batch, 18)

        if return_details:
            return logits, layer_weights.squeeze(-1), token_weights
        if return_proj:
            proj = F.normalize(self.proj_head(dropped), p=2, dim=-1)
            return logits, proj
        return logits


class QwenLayerTokenPipeline:
    """封裝 Qwen-3-Embedding (前 24 層截斷版) + TaskLoRA + 階層式雙重注意力 + Prompt-Aware SupCon 流程。"""

    def __init__(
        self,
        base_model_path: str,
        label_names: Sequence[str],
        batch_size: int = MICRO_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        lora_rank: int = LORA_RANK,
        lora_scale: float = LORA_SCALE,
        layer_proj_dim: int = LAYER_PROJ_DIM,
        num_heads: int = NUM_HEADS,
        attn_dim: int = ATTN_DIM,
        keep_layers: int = KEEP_LAYERS,
        layer_drop_prob: float = LAYER_DROP_PROB,
        layer_temperature: float = LAYER_TEMPERATURE,
        token_drop_prob: float = TOKEN_DROP_PROB,
        token_temperature: float = TOKEN_TEMPERATURE,
        supcon_proj_dim: int = SUPCON_PROJ_DIM,
        supcon_weight: float = SUPCON_WEIGHT,
        supcon_temperature: float = SUPCON_TEMPERATURE,
        supcon_queue_size: int = SUPCON_QUEUE_SIZE,
        supcon_same_prompt_neg_weight: float = SUPCON_SAME_PROMPT_NEG_WEIGHT,
        supcon_prompt_block_size: int = SUPCON_PROMPT_BLOCK_SIZE,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.label_names = list(label_names)
        self.batch_size = batch_size
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.layer_proj_dim = int(layer_proj_dim)
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.keep_layers = int(keep_layers)
        self.layer_drop_prob = float(layer_drop_prob)
        self.layer_temperature = float(layer_temperature)
        self.token_drop_prob = float(token_drop_prob)
        self.token_temperature = float(token_temperature)
        self.supcon_proj_dim = int(supcon_proj_dim)
        self.supcon_weight = float(supcon_weight)
        self.supcon_temperature = float(supcon_temperature)
        self.supcon_queue_size = int(supcon_queue_size)
        self.supcon_same_prompt_neg_weight = float(supcon_same_prompt_neg_weight)
        self.supcon_prompt_block_size = int(supcon_prompt_block_size)

        # 對齊 native_q8.py 的數值精確度設定
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, local_files_only=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 固定使用 Qwen-3-Embedding 預訓練與 0-683 規定的句尾結束符 <|endoftext|> (151643)
        self.terminal_id = 151643
        self.prefix_len = len(
            self.tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
        )

        # 透過 AutoConfig 在載入模型權重前直接將 num_hidden_layers 截斷為 24 層
        cfg = AutoConfig.from_pretrained(self.base_model_path, local_files_only=True)
        tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
        orig_layers = int(tc.num_hidden_layers)
        self.num_hidden_layers = min(self.keep_layers, orig_layers)
        tc.num_hidden_layers = self.num_hidden_layers
        tc.use_cache = False
        if getattr(tc, "layer_types", None) is not None:
            tc.layer_types = tc.layer_types[: self.num_hidden_layers]
        self.hidden_size = int(tc.hidden_size)
        self.depths = _resolve_depths(self.num_hidden_layers, target_count=8)
        print(
            f"✂️ 已截斷 Qwen 模型至前 {self.num_hidden_layers} 層 "
            f"(原 {orig_layers} 層，第 {self.num_hidden_layers + 1}~{orig_layers} 層不載入亦不運算)"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            cut = int(round(self.num_hidden_layers * 0.40))
            device_map: dict[str, int] | str = {
                "embed_tokens": 0,
                "norm": n_gpus - 1,
                "rotary_emb": n_gpus - 1,
            }
            device_map.update(
                {
                    f"layers.{i}": 0 if i < cut else 1
                    for i in range(self.num_hidden_layers)
                }
            )
            self.input_device = torch.device("cuda:0")
            self.head_device = torch.device(f"cuda:{n_gpus - 1}")
        elif n_gpus == 1:
            device_map = {"": 0}
            self.input_device = torch.device("cuda:0")
            self.head_device = torch.device("cuda:0")
        else:
            device_map = "cpu"
            self.input_device = torch.device("cpu")
            self.head_device = torch.device("cpu")

        dtype = torch.float16 if n_gpus >= 1 else torch.float32
        self.model = AutoModel.from_pretrained(
            self.base_model_path,
            config=cfg,
            local_files_only=True,
            dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        if self.num_hidden_layers < orig_layers and hasattr(self.model, "norm"):
            self.model.norm = nn.Identity()

        self.model.config.use_cache = False
        self.model.requires_grad_(False)

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # 在前 24 層的每層 7 個指定線性模組掛上可訓練的 TaskLoRA
        self.loras: dict[str, TaskLoRA] = {}
        for name, mod in list(self.model.named_modules()):
            if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in LORA_SUFFIXES:
                wrapped = TaskLoRA(mod, rank=self.lora_rank, scale=self.lora_scale)
                parent, attr = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent), attr, wrapped)
                self.loras[name] = wrapped

        # 建立結合 8 層樓 Layer Attention + 全序列 Token Attention + SupCon Projection 分支的分類頭
        self.head = HierarchicalLayerTokenClassifier(
            hidden_size=self.hidden_size,
            depths=self.depths,
            num_classes=len(self.label_names),
            layer_proj_dim=self.layer_proj_dim,
            num_heads=self.num_heads,
            attn_dim=self.attn_dim,
            layer_drop_prob=self.layer_drop_prob,
            layer_temperature=self.layer_temperature,
            token_drop_prob=self.token_drop_prob,
            token_temperature=self.token_temperature,
            supcon_proj_dim=self.supcon_proj_dim,
        ).to(self.head_device)

    def _tokenize_batch(
        self, texts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """加上風格辨識指令並補上句尾 TERMINAL token，回傳 input_ids 與 attention_mask。"""
        prompted = [INSTRUCTION + ("" if pd.isna(t) else str(t)) for t in texts]
        raw_ids = self.tokenizer(
            prompted,
            truncation=True,
            max_length=self.max_length - 1,
            add_special_tokens=False,
        )["input_ids"]
        seqs = [
            (ids + [self.terminal_id]) if ids else [self.terminal_id]
            for ids in raw_ids
        ]
        padded = self.tokenizer.pad(
            {"input_ids": seqs}, padding=True, return_tensors="pt"
        )
        return (
            padded["input_ids"].to(self.input_device),
            padded["attention_mask"].to(self.input_device),
        )

    def forward_batch(
        self,
        texts: Sequence[str],
        return_details: bool = False,
        return_proj: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """
        前向傳播：
        1. 抽取 8 個目標樓層 (3, 6, 9, 12, 15, 18, 21, 24) 的完整 Token 序列 (Batch, Seq_Len, 4096)。
        2. 送入 HierarchicalLayerTokenClassifier 輸出 18 類 logits（訓練期可同時回傳 128 維對比投影向量）。
        """
        input_ids, attention_mask = self._tokenize_batch(texts)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        layer_seq_features: list[torch.Tensor] = [
            out.hidden_states[depth].to(self.head_device)
            for depth in self.depths
        ]
        head_mask = attention_mask.to(self.head_device)
        del out

        if return_details:
            logits, layer_weights, token_weights = self.head(
                layer_seq_features,
                head_mask,
                return_details=True,
                return_proj=False,
            )
            return (
                logits,
                layer_weights,
                token_weights,
                input_ids,
                attention_mask,
            )

        return self.head(
            layer_seq_features,
            head_mask,
            return_details=False,
            return_proj=return_proj,
        )

    @torch.inference_mode()
    def predict_with_dual_importance(
        self,
        texts: Iterable[object],
        progress_desc: str = "Inference",
        max_samples_to_detail: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray]:
        """
        批次預測類別索引，並同時收集：
        1. 每篇文章在 8 個樓層的 Layer Attention 權重矩陣 (N, 8)。
        2. 單篇 Top-10 與全域 Top-20 的 Token Attention 重要性統計報告。
        3. 原始 logits 矩陣 (N, 18)，供驗證集 LogLoss 評估使用。
        """
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        if not clean_texts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, len(self.depths)), dtype=np.float32),
                {"sample_inspections": [], "global_top_tokens": []},
                np.empty((0, len(self.label_names)), dtype=np.float32),
            )

        self.model.eval()
        self.head.eval()
        preds: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        all_layer_weights: list[np.ndarray] = []
        sample_inspections: list[dict[str, object]] = []

        token_rel_weight_sum: dict[int, float] = collections.defaultdict(float)
        token_abs_weight_sum: dict[int, float] = collections.defaultdict(float)
        token_counts: dict[int, int] = collections.defaultdict(int)

        batch_starts = range(0, len(clean_texts), self.batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(clean_texts) + self.batch_size - 1) // self.batch_size,
            desc=progress_desc,
            unit="batch",
        ):
            batch = clean_texts[start : start + self.batch_size]
            logits, layer_w, token_w, inp_ids, att_mask = self.forward_batch(
                batch, return_details=True, return_proj=False
            )
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.append(batch_preds)
            all_logits.append(logits.float().cpu().numpy())
            all_layer_weights.append(layer_w.float().cpu().numpy())

            w_np = token_w.float().cpu().numpy()
            ids_np = inp_ids.cpu().numpy()
            mask_np = att_mask.cpu().numpy().astype(bool)

            for r in range(len(batch)):
                valid_len = int(mask_np[r].sum())
                if valid_len <= 0:
                    continue
                row_ids = ids_np[r, :valid_len]
                row_w = w_np[r, :valid_len]

                content_start = min(self.prefix_len, max(valid_len - 1, 0))
                for pos in range(content_start, valid_len):
                    tid = int(row_ids[pos])
                    w_val = float(row_w[pos])
                    token_rel_weight_sum[tid] += w_val * valid_len
                    token_abs_weight_sum[tid] += w_val
                    token_counts[tid] += 1

                if len(sample_inspections) < max_samples_to_detail:
                    content_positions = np.arange(content_start, valid_len)
                    if len(content_positions) == 0:
                        content_positions = np.arange(valid_len)
                    top_order = content_positions[
                        np.argsort(-row_w[content_positions])[:10]
                    ]
                    top_tokens_info = []
                    for pos in top_order:
                        tid = int(row_ids[pos])
                        tok_str = repr(self.tokenizer.decode([tid]))
                        top_tokens_info.append(
                            {
                                "position": int(pos),
                                "token_id": tid,
                                "token": tok_str,
                                "weight": round(float(row_w[pos]), 6),
                                "relative_multiplier": round(
                                    float(row_w[pos] * valid_len), 2
                                ),
                            }
                        )
                    sample_inspections.append(
                        {
                            "sample_index": int(start + r),
                            "predicted_model": self.label_names[int(batch_preds[r])],
                            "valid_tokens": valid_len,
                            "text_preview": batch[r][:100].replace("\n", " "),
                            "top_tokens": top_tokens_info,
                        }
                    )

        min_occurrences = 5 if len(clean_texts) >= 20 else 1
        ranked_tids = [
            tid
            for tid, cnt in token_counts.items()
            if cnt >= min_occurrences
        ]
        ranked_tids.sort(
            key=lambda tid: token_rel_weight_sum[tid] / token_counts[tid],
            reverse=True,
        )

        global_top_tokens = []
        for tid in ranked_tids[:20]:
            cnt = token_counts[tid]
            avg_rel = token_rel_weight_sum[tid] / cnt
            avg_abs = token_abs_weight_sum[tid] / cnt
            global_top_tokens.append(
                {
                    "token_id": int(tid),
                    "token": repr(self.tokenizer.decode([int(tid)])),
                    "occurrences": int(cnt),
                    "avg_weight_pct": round(float(avg_abs * 100.0), 4),
                    "avg_relative_multiplier": round(float(avg_rel), 3),
                }
            )

        token_report = {
            "total_samples": len(clean_texts),
            "sample_inspections": sample_inspections,
            "global_top_tokens": global_top_tokens,
        }
        return (
            np.concatenate(preds),
            np.concatenate(all_layer_weights, axis=0),
            token_report,
            np.concatenate(all_logits, axis=0),
        )

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """預測模型名稱，並同時回傳 8 層樓權重矩陣與 Token 重要性報告。"""
        indices, layer_weights, token_report, _ = self.predict_with_dual_importance(
            texts, progress_desc=progress_desc
        )
        names = np.asarray(
            [self.label_names[idx] for idx in indices], dtype=object
        )
        return names, layer_weights, token_report


def print_layer_importance_report(
    depths: Sequence[int], weights: np.ndarray, title: str = "8 層樓 Attention 重要性分析"
) -> dict[str, object]:
    """統計並印出 8 個樓層的平均 Attention 權重、標準差與最常主導的樓層比例。"""
    mean_w = weights.mean(axis=0)
    std_w = weights.std(axis=0)
    top1_counts = np.bincount(weights.argmax(axis=1), minlength=len(depths))
    top1_ratio = top1_counts / max(len(weights), 1)
    best_idx = int(np.argmax(mean_w))

    print(f"\n{'=' * 68}")
    print(f"📊 {title} (樣本數: {len(weights)})")
    print(f"{'-' * 68}")
    print(f"{'樓層 (Layer)':<14} | {'平均權重':<10} | {'標準差':<10} | {'奪冠比例':<10} | 視覺化長條圖")
    print(f"{'-' * 68}")
    for i, depth in enumerate(depths):
        bar = "█" * int(round(mean_w[i] * 50))
        mark = " ⭐ 最重要" if i == best_idx else ""
        print(
            f"Layer {depth:2d}       | {mean_w[i] * 100:6.2f}%   | ±{std_w[i] * 100:5.2f}%  | "
            f"{top1_ratio[i] * 100:6.2f}%   | {bar}{mark}"
        )
    print(f"{'=' * 68}\n")

    return {
        "depths": [int(d) for d in depths],
        "mean_weights": [round(float(x), 6) for x in mean_w],
        "std_weights": [round(float(x), 6) for x in std_w],
        "top1_win_ratio": [round(float(x), 6) for x in top1_ratio],
        "most_important_layer": int(depths[best_idx]),
    }


def print_token_importance_report(
    report: dict[str, object], title: str = "全 Token 注意力重要性分析"
) -> None:
    """在終端機印出單篇文章 Top-10 關鍵 Token 以及全域 Top-20 高權重關鍵字/標點排行榜。"""
    print(f"\n{'=' * 78}")
    print(f"🔍 {title} (總樣本數: {report.get('total_samples', 0)})")
    print(f"{'-' * 78}")

    samples = report.get("sample_inspections", [])
    if samples:
        print("📌 [單篇文章透視] 模型最關注的前 10 個關鍵 Token：")
        for sample in samples:
            print(
                f"\n  ▶ 樣本 #{sample['sample_index']} | 預測模型: {sample['predicted_model']} "
                f"| 有效 Token 數: {sample['valid_tokens']}"
            )
            print(f"    內文預覽: {sample['text_preview']}...")
            for rank, item in enumerate(sample["top_tokens"], start=1):
                w_pct = item["weight"] * 100.0
                bar = "█" * max(1, int(round(w_pct * 2)))
                print(
                    f"      {rank:2d}. Pos {item['position']:3d} | Token: {item['token']:<18} "
                    f"| 權重: {w_pct:6.2f}% ({item['relative_multiplier']:5.1f}x 平均) | {bar}"
                )

    global_top = report.get("global_top_tokens", [])
    if global_top:
        print(f"\n{'-' * 78}")
        print("🏆 [全域關鍵 Token 排行榜] 出現時最常獲得高注意力倍率的 Top-20 字詞/標點：")
        print(
            f"{'排名':<5} | {'Token 內容':<20} | {'出現次數':<8} | {'平均權重':<10} | {'相對平均倍率':<12}"
        )
        print(f"{'-' * 78}")
        for rank, item in enumerate(global_top, start=1):
            print(
                f"#{rank:<4d} | {item['token']:<20} | {item['occurrences']:<8d} | "
                f"{item['avg_weight_pct']:6.3f}%    | {item['avg_relative_multiplier']:6.2f}x"
            )
    print(f"{'=' * 78}\n")


def _restore_best_weights(
    pipeline: QwenLayerTokenPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重載回記憶體中的 pipeline，確保最終推論使用最佳模型。"""
    save_dir = pathlib.Path(save_dir)
    lora_path = save_dir / "lora_weights.pt"
    head_path = save_dir / "layer_token_head.pt"
    if not lora_path.exists() or not head_path.exists():
        return

    lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, mod in pipeline.loras.items():
            mod.a.weight.copy_(
                lora_state[f"{name}.lora_A.weight"].to(mod.a.weight.device)
            )
            mod.b.weight.copy_(
                lora_state[f"{name}.lora_B.weight"].to(mod.b.weight.device)
            )

    head_state = torch.load(
        head_path, map_location=pipeline.head_device, weights_only=True
    )
    pipeline.head.load_state_dict(head_state)


def _train_pipeline(
    pipeline: QwenLayerTokenPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    prompt_codes: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_supcon",
) -> tuple[float, dict[str, object], dict[str, object]]:
    """
    聯合訓練 TaskLoRA 與 HierarchicalLayerTokenClassifier (CE + Queue-based Prompt-Aware SupCon)：
    每個 Epoch 使用 Prompt-Block 洗牌排程並維護 GPU Feature Queue，同時監控 CE Loss 與 SupCon Loss；
    驗證期檢查 balanced_accuracy 與 val_log_loss，僅在創新高時儲存模型，最後載回最佳權重。
    """
    lora_params = [
        p
        for mod in pipeline.loras.values()
        for p in (mod.a.weight, mod.b.weight)
        if p.requires_grad
    ]
    head_params = [p for p in pipeline.head.parameters() if p.requires_grad]
    trainable_params = lora_params + head_params
    if not trainable_params:
        raise ValueError("No trainable parameters found in TaskLoRA or Head")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LEARNING_RATE},
            {"params": head_params, "lr": LEARNING_RATE * 2.0},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    supcon_queue = PromptAwareSupConQueue(
        proj_dim=pipeline.supcon_proj_dim,
        queue_size=pipeline.supcon_queue_size,
        temperature=pipeline.supcon_temperature,
        same_prompt_neg_weight=pipeline.supcon_same_prompt_neg_weight,
        device=pipeline.head_device,
    )
    rng = np.random.default_rng(SEED)

    best_score = -1.0
    best_epoch = 0
    best_layer_report: dict[str, object] = {}
    best_token_report: dict[str, object] = {}

    for epoch in range(EPOCHS):
        pipeline.model.train()
        pipeline.head.train()
        supcon_queue.reset()

        order = _build_prompt_aware_order(
            train_indices,
            prompt_codes,
            rng,
            block_size=pipeline.supcon_prompt_block_size,
        )
        total_loss = 0.0
        total_ce = 0.0
        total_sc = 0.0
        seen = 0

        micro_batches = [
            order[start : start + pipeline.batch_size]
            for start in range(0, len(order), pipeline.batch_size)
        ]
        progress = tqdm(
            range(len(micro_batches)),
            total=len(micro_batches),
            desc=f"Training epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
        )

        for batch_number in progress:
            indices = micro_batches[batch_number]
            group_start = (
                batch_number // GRADIENT_ACCUMULATION_STEPS
            ) * GRADIENT_ACCUMULATION_STEPS
            group_end = min(
                group_start + GRADIENT_ACCUMULATION_STEPS, len(micro_batches)
            )
            group_examples = sum(
                len(micro_batches[i]) for i in range(group_start, group_end)
            )

            if batch_number == group_start:
                optimizer.zero_grad(set_to_none=True)

            batch_texts = [texts[i] for i in indices]
            logits, proj = pipeline.forward_batch(
                batch_texts, return_details=False, return_proj=True
            )
            targets = torch.as_tensor(
                labels[indices], dtype=torch.long, device=logits.device
            )
            prompts_t = torch.as_tensor(
                prompt_codes[indices], dtype=torch.long, device=logits.device
            )

            ce_loss = criterion(logits, targets)
            sc_loss = supcon_queue.compute_loss(proj, targets, prompts_t)
            loss = ce_loss + pipeline.supcon_weight * sc_loss
            (loss * (len(indices) / group_examples)).backward()

            # 算完對比損失後再將當前 micro-batch 寫入歷史佇列
            supcon_queue.enqueue(proj, targets, prompts_t)

            if (batch_number + 1) == group_end:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            n_cur = len(indices)
            total_loss += float(loss.item()) * n_cur
            total_ce += float(ce_loss.item()) * n_cur
            total_sc += float(sc_loss.item()) * n_cur
            seen += n_cur
            progress.set_postfix(
                loss=f"{total_loss / seen:.4f}",
                ce=f"{total_ce / seen:.4f}",
                sc=f"{total_sc / seen:.4f}",
            )
            del logits, proj, targets, prompts_t, ce_loss, sc_loss, loss

        # 進入驗證集推論前先清空訓練階段的顯存快取
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_preds, val_layer_weights, val_token_report, val_logits = (
            pipeline.predict_with_dual_importance(
                (texts[i] for i in validation_indices.tolist()),
                progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
            )
        )
        val_true = labels[validation_indices]
        score = _balanced_accuracy(val_true, val_preds, len(pipeline.label_names))
        val_ll = _log_loss_from_logits(val_logits, val_true)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / max(seen, 1):.4f} "
            f"(ce: {total_ce / max(seen, 1):.4f}, supcon: {total_sc / max(seen, 1):.4f}) "
            f"- validation balanced_accuracy: {score:.6f} - val_log_loss: {val_ll:.4f}"
        )

        val_layer_report = print_layer_importance_report(
            pipeline.depths,
            val_layer_weights,
            title=(
                f"Epoch {epoch + 1} 驗證集【第一重：8 層樓 Layer Attention】權重統計 "
                f"(T={pipeline.layer_temperature}, LayerDrop={pipeline.layer_drop_prob:.0%})"
            ),
        )
        print_token_importance_report(
            val_token_report,
            title=(
                f"Epoch {epoch + 1} 驗證集【第二重：全 Token Attention】重要性分析 "
                f"(T={pipeline.token_temperature}, TokenDrop={pipeline.token_drop_prob:.0%})"
            ),
        )

        if score > best_score:
            prev_best = best_score
            best_score = score
            best_epoch = epoch + 1
            best_layer_report = val_layer_report
            best_token_report = val_token_report
            save_model(
                pipeline,
                save_dir,
                layer_report=best_layer_report,
                token_report=best_token_report,
            )
            if prev_best < 0:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: 初次建立最佳模型 "
                    f"(balanced_accuracy = {best_score:.6f}, val_log_loss = {val_ll:.4f})，已儲存至 {save_dir}！"
                )
            else:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: validation balanced_accuracy 創新高 "
                    f"({prev_best:.6f} -> {best_score:.6f}, val_log_loss = {val_ll:.4f})，已更新最佳模型至 {save_dir}！"
                )
        else:
            print(
                f"⚠️ Epoch {epoch + 1}/{EPOCHS}: validation balanced_accuracy ({score:.6f}) "
                f"未超越最佳紀錄 ({best_score:.6f} @ Epoch {best_epoch})，不儲存新模型，繼續訓練..."
            )

    if best_epoch > 0 and best_epoch != EPOCHS:
        print(
            f"🔄 訓練結束：載回 Epoch {best_epoch} 的最佳模型權重 "
            f"(best validation balanced_accuracy = {best_score:.6f})"
        )
        _restore_best_weights(pipeline, save_dir)

    return best_score, best_layer_report, best_token_report


def save_model(
    pipeline: QwenLayerTokenPipeline,
    save_dir: str | pathlib.Path,
    layer_report: dict[str, object] | None = None,
    token_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、HierarchicalLayerTokenClassifier (含 SupCon 投影頭) 權重以及分析報告。"""
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for name, mod in pipeline.loras.items():
        lora_state[f"{name}.lora_A.weight"] = mod.a.weight.detach().cpu()
        lora_state[f"{name}.lora_B.weight"] = mod.b.weight.detach().cpu()

    torch.save(lora_state, save_dir / "lora_weights.pt")
    torch.save(pipeline.head.state_dict(), save_dir / "layer_token_head.pt")

    metadata = {
        "label_names": pipeline.label_names,
        "base_model_path": pipeline.base_model_path,
        "keep_layers": pipeline.keep_layers,
        "depths": list(pipeline.depths),
        "hidden_size": pipeline.hidden_size,
        "num_classes": len(pipeline.label_names),
        "batch_size": pipeline.batch_size,
        "max_length": pipeline.max_length,
        "lora_rank": pipeline.lora_rank,
        "lora_scale": pipeline.lora_scale,
        "layer_proj_dim": pipeline.layer_proj_dim,
        "num_heads": pipeline.num_heads,
        "attn_dim": pipeline.attn_dim,
        "layer_drop_prob": pipeline.layer_drop_prob,
        "layer_temperature": pipeline.layer_temperature,
        "token_drop_prob": pipeline.token_drop_prob,
        "token_temperature": pipeline.token_temperature,
        "supcon_proj_dim": pipeline.supcon_proj_dim,
        "supcon_weight": pipeline.supcon_weight,
        "supcon_temperature": pipeline.supcon_temperature,
        "supcon_queue_size": pipeline.supcon_queue_size,
        "supcon_same_prompt_neg_weight": pipeline.supcon_same_prompt_neg_weight,
        "supcon_prompt_block_size": pipeline.supcon_prompt_block_size,
    }
    (save_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if layer_report is not None:
        (save_dir / "layer_importance.json").write_text(
            json.dumps(layer_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if token_report is not None:
        (save_dir / "token_importance.json").write_text(
            json.dumps(token_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Saved model, layer importance, and token importance reports to {save_dir}")


def load_model(
    save_dir: str | pathlib.Path,
    base_model_path: str | None = None,
    trainable: bool = False,
) -> QwenLayerTokenPipeline:
    """載入已保存的 TaskLoRA 與 HierarchicalLayerTokenClassifier (Prompt-Aware SupCon)。"""
    save_dir = pathlib.Path(save_dir)
    metadata = json.loads((save_dir / "metadata.json").read_text(encoding="utf-8"))
    resolved_base = base_model_path or _find_model_dir(metadata["base_model_path"])

    pipeline = QwenLayerTokenPipeline(
        base_model_path=resolved_base,
        label_names=metadata["label_names"],
        batch_size=int(metadata["batch_size"]),
        max_length=int(metadata["max_length"]),
        lora_rank=int(metadata["lora_rank"]),
        lora_scale=float(metadata["lora_scale"]),
        layer_proj_dim=int(metadata.get("layer_proj_dim", LAYER_PROJ_DIM)),
        num_heads=int(metadata.get("num_heads", NUM_HEADS)),
        attn_dim=int(metadata.get("attn_dim", ATTN_DIM)),
        keep_layers=int(metadata.get("keep_layers", KEEP_LAYERS)),
        layer_drop_prob=float(metadata.get("layer_drop_prob", LAYER_DROP_PROB)),
        layer_temperature=float(
            metadata.get("layer_temperature", LAYER_TEMPERATURE)
        ),
        token_drop_prob=float(metadata.get("token_drop_prob", TOKEN_DROP_PROB)),
        token_temperature=float(
            metadata.get("token_temperature", TOKEN_TEMPERATURE)
        ),
        supcon_proj_dim=int(metadata.get("supcon_proj_dim", SUPCON_PROJ_DIM)),
        supcon_weight=float(metadata.get("supcon_weight", SUPCON_WEIGHT)),
        supcon_temperature=float(
            metadata.get("supcon_temperature", SUPCON_TEMPERATURE)
        ),
        supcon_queue_size=int(
            metadata.get("supcon_queue_size", SUPCON_QUEUE_SIZE)
        ),
        supcon_same_prompt_neg_weight=float(
            metadata.get(
                "supcon_same_prompt_neg_weight", SUPCON_SAME_PROMPT_NEG_WEIGHT
            )
        ),
        supcon_prompt_block_size=int(
            metadata.get("supcon_prompt_block_size", SUPCON_PROMPT_BLOCK_SIZE)
        ),
    )

    lora_state = torch.load(
        save_dir / "lora_weights.pt", map_location="cpu", weights_only=True
    )
    with torch.no_grad():
        for name, mod in pipeline.loras.items():
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key not in lora_state or b_key not in lora_state:
                raise KeyError(f"Missing LoRA weights for {name}")
            mod.a.weight.copy_(lora_state[a_key].to(mod.a.weight.device))
            mod.b.weight.copy_(lora_state[b_key].to(mod.b.weight.device))
            mod.a.weight.requires_grad_(trainable)
            mod.b.weight.requires_grad_(trainable)

    head_state = torch.load(
        save_dir / "layer_token_head.pt",
        map_location=pipeline.head_device,
        weights_only=True,
    )
    pipeline.head.load_state_dict(head_state)
    pipeline.head.requires_grad_(trainable)
    pipeline.model.eval()
    pipeline.head.eval()
    return pipeline


def Trainer(
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_supcon",
    load_weights: bool = False,
) -> QwenLayerTokenPipeline:
    """讀取資料、建立 Qwen (前 24 層) + TaskLoRA + Queue-based Prompt-Aware SupCon 並執行訓練。"""
    _seed_everything(SEED)
    data_dir = _find_data_dir()
    train_df = pd.read_csv(data_dir / "train.csv")
    required_train = {"text", "model", "prompt_id"}
    if not required_train.issubset(train_df.columns):
        raise ValueError(f"train.csv must contain columns {sorted(required_train)}")

    texts = train_df["text"].fillna("").astype(str).tolist()
    label_names = sorted(train_df["model"].astype(str).unique().tolist())
    if len(label_names) != 18:
        raise ValueError(f"Expected 18 model classes, found {len(label_names)}")
    label_to_index = {name: index for index, name in enumerate(label_names)}
    labels = train_df["model"].astype(str).map(label_to_index).to_numpy(dtype=np.int64)

    raw_prompts = train_df["prompt_id"].fillna("").astype(str).tolist()
    prompt_to_code = {p: idx for idx, p in enumerate(sorted(set(raw_prompts)))}
    prompt_codes = np.asarray(
        [prompt_to_code[p] for p in raw_prompts], dtype=np.int64
    )

    train_indices, validation_indices = _group_split(
        raw_prompts,
        labels.tolist(),
        VALIDATION_FRACTION,
        SEED,
    )

    if load_weights:
        print(f"Loading existing weights from {save_dir} to continue training...")
        pipeline = load_model(save_dir, trainable=True)
    else:
        model_dir = _find_model_dir()
        print(f"Using Qwen base model: {model_dir}")
        pipeline = QwenLayerTokenPipeline(
            base_model_path=model_dir,
            label_names=label_names,
        )

    lora_count = sum(
        p.numel()
        for mod in pipeline.loras.values()
        for p in (mod.a.weight, mod.b.weight)
        if p.requires_grad
    )
    head_count = sum(
        p.numel() for p in pipeline.head.parameters() if p.requires_grad
    )
    print(
        f"監聽的 8 個樓層 (DEPTHS): {pipeline.depths} (各層獨立投影 {pipeline.layer_proj_dim} 維 -> torch.cat 拼接 {pipeline.head.proj_dim} 維) | "
        f"Prompt-Aware SupCon (λ={pipeline.supcon_weight}, τ={pipeline.supcon_temperature}, "
        f"Q={pipeline.supcon_queue_size}, hard_neg={pipeline.supcon_same_prompt_neg_weight}x, block={pipeline.supcon_prompt_block_size}) | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 Head 參數: {head_count:,}"
    )

    _train_pipeline(
        pipeline,
        texts,
        labels,
        prompt_codes,
        train_indices,
        validation_indices,
        save_dir=save_dir,
    )
    return pipeline


def main() -> pd.DataFrame:
    """執行完整訓練、分析 8 層樓與每個 Token 的雙重重要性，並對 test.csv 產生預測提交檔。"""
    save_dir = "saved_qwen_layertoken_supcon"
    pipeline = Trainer(save_dir, load_weights=False)

    # 釋放訓練階段多餘顯存後直接進行推論
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_dir = _find_data_dir()
    test_df = pd.read_csv(data_dir / "test.csv")
    if not {"id", "text"}.issubset(test_df.columns):
        raise ValueError("test.csv must contain id and text columns")

    test_preds, test_layer_weights, test_token_report = pipeline.predict(
        test_df["text"].fillna(""), progress_desc="Test Inference"
    )
    print_layer_importance_report(
        pipeline.depths,
        test_layer_weights,
        title="測試集 (test.csv)【第一重：8 層樓 Layer Attention】權重統計",
    )
    print_token_importance_report(
        test_token_report,
        title="測試集 (test.csv)【第二重：全 Token Attention】重要性分析",
    )

    submission = pd.DataFrame({"id": test_df["id"], "model": test_preds})
    output_path = pathlib.Path("submission.parquet")
    submission.to_parquet(output_path, index=False)
    print(f"Saved {output_path} with {len(submission)} rows.")
    return submission


if __name__ == "__main__":
    submission = main()
    os.makedirs("/kaggle/working", exist_ok=True)
    submission.to_parquet("/kaggle/working/submission.parquet", index=False)
    submission.to_csv("/kaggle/working/submission.csv", index=False)
