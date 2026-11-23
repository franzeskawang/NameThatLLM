"""
以 TaskLoRA + 階層式雙重注意力 + 同題目去主題雙頭 (Plain + Centered Twin Head) 微調 Qwen-3-Embedding (NameThatLLM_Qwen_LayerToken_v3.py)：
1. 拿掉 CosFace，回歸保留特徵強度資訊的標準線性分類頭，並升級為【雙頭架構】：
   - plain_classifier：單篇文章獨立預測頭。
   - centered_classifier：同題目去主題對比頭（將同一道 prompt_id 的文章表徵減去組平均，消除題目主題雜訊）。
2. 零額外顯存與零額外時間的「同題目配對去主題訓練」。
3. 驗證集同時印出 Plain、Centered 與雙頭融合 (Blend) 分數與校準 LogLoss；測試集自動透過 TF-IDF + capped_agglo 聚類。
4. 內建 Linear Warmup + Cosine Annealing 學習率排程、Label Smoothing (0.05) 與雙重注意力分析報告。
"""

from __future__ import annotations

import json
import math
import os
import pathlib
from typing import Iterable, Sequence

# 啟用 PyTorch CUDA 動態顯存區段擴展，大幅減少顯存碎片化導致的 OOM。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from data_utils import (
    CANDIDATES,
    clear_cuda_cache,
    find_data_dir,
    group_split,
    load_test_data,
    load_train_data,
    save_kaggle_submission,
    save_submission,
    seed_everything,
)
from metrics_utils import (
    balanced_accuracy,
    fit_temperature_and_blend,
    log_softmax_np,
)
from qwen_utils import (
    DEFAULT_MODEL_PATH,
    INSTRUCTION,
    LORA_SUFFIXES,
    TaskLoRA,
    build_cosine_warmup_scheduler,
    count_qwen_trainable_params,
    find_qwen_model_dir,
    load_qwen_backbone,
    load_task_lora_and_head,
    resolve_depths,
    save_task_lora_and_head,
    tokenize_qwen_batch,
)
from report_utils import (
    TokenImportanceTracker,
    print_layer_importance_report,
    print_token_importance_report,
)


# 只保留前 24 層，每 3 層設置一個監聽哨（共 8 個目標樓層）。
KEEP_LAYERS = 24
DEFAULT_DEPTHS = (3, 6, 9, 12, 15, 18, 21, 24)

# 訓練與雙重注意力雙頭模型超參數設定。
SEED = 42
VALIDATION_FRACTION = 0.20
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 20
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LABEL_SMOOTHING = 0.05
MAX_LENGTH = 512
LORA_RANK = 16
LORA_SCALE = 2.0

LAYER_PROJ_DIM = 32
NUM_HEADS = 4
ATTN_DIM = 128
DROPOUT = 0.10

LAYER_DROP_PROB = 0.10
LAYER_TEMPERATURE = 2.0

TOKEN_DROP_PROB = 0.15
TOKEN_TEMPERATURE = 1.5

# 雙頭訓練 Loss 權重：總 Loss = plain_loss + CENTERED_LOSS_WEIGHT * centered_loss
CENTERED_LOSS_WEIGHT = 1.0
# 測試集 TF-IDF 內容分群門檻
TEST_GROUP_THR = 0.15


def _build_prompt_paired_batches(
    train_indices: np.ndarray,
    groups: Sequence[object],
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, bool]]:
    """將訓練集依照 prompt_id 分組並兩兩配對成 micro-batches，回傳 (batch_indices, is_same_prompt_group)。"""
    group_to_indices: dict[object, list[int]] = {}
    for idx in train_indices.tolist():
        group_to_indices.setdefault(groups[idx], []).append(idx)

    group_keys = list(group_to_indices.keys())
    rng.shuffle(group_keys)

    batches: list[tuple[np.ndarray, bool]] = []
    leftovers: list[int] = []

    for g in group_keys:
        members = group_to_indices[g].copy()
        rng.shuffle(members)
        pos = 0
        while pos + batch_size <= len(members):
            batches.append(
                (np.asarray(members[pos : pos + batch_size], dtype=np.int64), True)
            )
            pos += batch_size
        rem = members[pos:]
        if len(rem) >= 2:
            batches.append((np.asarray(rem, dtype=np.int64), True))
        elif len(rem) == 1:
            leftovers.append(rem[0])

    rng.shuffle(leftovers)
    for pos in range(0, len(leftovers), batch_size):
        batches.append(
            (np.asarray(leftovers[pos : pos + batch_size], dtype=np.int64), False)
        )

    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def _capped_agglo(
    sim_matrix: np.ndarray, cap: int = 18, thr: float = TEST_GROUP_THR
) -> np.ndarray:
    """受限階層式分群，將相似文章聚為最多 18 篇的同題目小組。"""
    n = len(sim_matrix)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    A = sim_matrix.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, dtype=np.int32)
    alive = np.ones(n, dtype=bool)
    lab = np.arange(n, dtype=np.int64)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab


def _infer_test_groups(texts: Sequence[str], thr: float = TEST_GROUP_THR) -> np.ndarray:
    """當測試集沒有 prompt_id 時，使用 TF-IDF 內容相似度自動推估同題目群組。"""
    n = len(texts)
    if n < 2:
        return np.arange(n, dtype=np.int64)
    try:
        vec = TfidfVectorizer(
            sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05
        )
        X = vec.fit_transform(texts)
        return _capped_agglo((X @ X.T).toarray(), cap=18, thr=thr)
    except ValueError:
        return np.arange(n, dtype=np.int64)


class HierarchicalLayerTokenTwinClassifier(nn.Module):
    """階層式「8 層樓 Layer Attention + 全序列 512 Token Attention + 同題目去主題雙頭 (Plain + Centered)」分類器。"""

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depths = tuple(int(d) for d in depths)
        self.num_layers = len(self.depths)
        self.layer_proj_dim = int(layer_proj_dim)
        self.proj_dim = self.layer_proj_dim * self.num_layers
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

        self.layer_gate = nn.Sequential(
            nn.Linear(hidden_size, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )

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

        self.token_gate = nn.Sequential(
            nn.Linear(self.proj_dim, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )

        self.dropout = nn.Dropout(dropout)
        self.plain_classifier = nn.Linear(
            self.proj_dim * 2, num_classes, dtype=torch.float32
        )
        self.centered_classifier = nn.Linear(
            self.proj_dim * 2, num_classes, dtype=torch.float32
        )

    def extract_features(
        self,
        layer_seq_features: list[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(layer_seq_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer sequence features, got {len(layer_seq_features)}"
            )

        valid_mask = attention_mask.bool()
        batch_size = attention_mask.shape[0]
        pos = attention_mask.sum(dim=1) - 1
        rows = torch.arange(batch_size, device=attention_mask.device)

        proj_seqs: list[torch.Tensor] = []
        layer_summaries: list[torch.Tensor] = []

        for i in range(self.num_layers):
            seq_feat = layer_seq_features[i]
            normed = self.layer_norms[i](seq_feat.float())
            layer_summaries.append(normed[rows, pos])

            p_seq = self.layer_projs[i](normed)
            p_seq = p_seq.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
            proj_seqs.append(p_seq)
            del seq_feat, normed

        stacked_summaries = torch.stack(layer_summaries, dim=1)
        layer_scores = self.layer_gate(stacked_summaries) / self.layer_temperature

        if self.training and self.layer_drop_prob > 0.0:
            layer_drop_mask = torch.rand_like(layer_scores) < self.layer_drop_prob
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

        layer_weights = F.softmax(layer_scores, dim=1)

        scaled_seqs = [
            proj_seqs[i] * (layer_weights[:, i, :].unsqueeze(1) * self.num_layers)
            for i in range(self.num_layers)
        ]
        x = torch.cat(scaled_seqs, dim=-1)
        del proj_seqs, scaled_seqs, stacked_summaries, layer_summaries

        x = self.self_attn(x, src_key_padding_mask=~valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        raw_token_scores = self.token_gate(x).squeeze(-1) / self.token_temperature
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

        attn_pooled = (x * token_weights.unsqueeze(-1)).sum(dim=1)
        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pooled = x.sum(dim=1) / denom

        combined = torch.cat([attn_pooled, mean_pooled], dim=-1)
        return combined, layer_weights.squeeze(-1), token_weights

    def forward(
        self,
        layer_seq_features: list[torch.Tensor],
        attention_mask: torch.Tensor,
        same_prompt_batch: bool = False,
        return_details: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        combined, layer_weights, token_weights = self.extract_features(
            layer_seq_features, attention_mask
        )
        plain_logits = self.plain_classifier(self.dropout(combined))

        if return_details:
            return plain_logits, combined, layer_weights, token_weights

        centered_logits: torch.Tensor | None = None
        n = combined.shape[0]
        if same_prompt_batch and n >= 2:
            scale = math.sqrt(float(n) / float(n - 1))
            centered_feat = (combined - combined.mean(dim=0, keepdim=True)) * scale
            centered_logits = self.centered_classifier(self.dropout(centered_feat))

        return plain_logits, centered_logits


class QwenLayerTokenTwinPipeline:
    """封裝 Qwen-3-Embedding (前 24 層截斷版) + TaskLoRA + 階層式雙重注意力雙頭模型的訓練與推論流程。"""

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
        plain_temperature: float = 1.0,
        centered_temperature: float = 1.0,
        centered_blend_weight: float = 0.5,
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
        self.plain_temperature = float(plain_temperature)
        self.centered_temperature = float(centered_temperature)
        self.centered_blend_weight = float(centered_blend_weight)

        backbone = load_qwen_backbone(
            base_model_path=self.base_model_path,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            keep_layers=self.keep_layers,
            multi_gpu_cut_ratio=0.40,
            lora_suffixes=LORA_SUFFIXES,
            instruction=INSTRUCTION,
        )
        self.tokenizer = backbone.tokenizer
        self.terminal_id = backbone.terminal_id
        self.prefix_len = backbone.prefix_len
        self.model = backbone.model
        self.loras = backbone.loras
        self.num_hidden_layers = backbone.num_hidden_layers
        self.hidden_size = backbone.hidden_size
        self.input_device = backbone.input_device
        self.head_device = backbone.head_device
        self.depths = resolve_depths(
            self.num_hidden_layers,
            default_depths=DEFAULT_DEPTHS,
            target_count=8,
            expected_layers=KEEP_LAYERS,
        )

        self.head = HierarchicalLayerTokenTwinClassifier(
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
        ).to(self.head_device)

    def _tokenize_batch(
        self, texts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tokenize_qwen_batch(
            self.tokenizer,
            texts,
            max_length=self.max_length,
            terminal_id=self.terminal_id,
            input_device=self.input_device,
            instruction=INSTRUCTION,
        )

    def forward_batch(
        self,
        texts: Sequence[str],
        same_prompt_batch: bool = False,
        return_details: bool = False,
    ):
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
            plain_logits, combined, layer_weights, token_weights = self.head(
                layer_seq_features,
                head_mask,
                same_prompt_batch=False,
                return_details=True,
            )
            return (
                plain_logits,
                combined,
                layer_weights,
                token_weights,
                input_ids,
                attention_mask,
            )

        return self.head(
            layer_seq_features,
            head_mask,
            same_prompt_batch=same_prompt_batch,
            return_details=False,
        )

    @torch.inference_mode()
    def compute_group_centered_logits(
        self, features: np.ndarray, groups: Sequence[object]
    ) -> tuple[np.ndarray, np.ndarray]:
        """根據群組編號計算同題目去主題特徵與 centered_logits，並回傳 eligible_mask。"""
        n = len(features)
        num_classes = len(self.label_names)
        centered_logits = np.zeros((n, num_classes), dtype=np.float32)
        eligible_mask = np.zeros(n, dtype=bool)
        if n == 0:
            return centered_logits, eligible_mask

        group_to_rows: dict[object, list[int]] = {}
        for row_idx, g in enumerate(groups):
            group_to_rows.setdefault(g, []).append(row_idx)

        centered_feats = np.zeros_like(features, dtype=np.float32)
        for rows in group_to_rows.values():
            g_size = len(rows)
            if 2 <= g_size <= 18:
                idx = np.asarray(rows, dtype=np.int64)
                sub = features[idx]
                scale = math.sqrt(float(g_size) / float(g_size - 1))
                centered_feats[idx] = (sub - sub.mean(axis=0, keepdims=True)) * scale
                eligible_mask[idx] = True

        if eligible_mask.any():
            elig_idx = np.where(eligible_mask)[0]
            feat_tensor = torch.as_tensor(
                centered_feats[elig_idx],
                dtype=torch.float32,
                device=self.head_device,
            )
            cent_out = self.head.centered_classifier(feat_tensor)
            centered_logits[elig_idx] = cent_out.float().cpu().numpy()

        return centered_logits, eligible_mask

    @torch.inference_mode()
    def predict_with_dual_importance(
        self,
        texts: Iterable[object],
        groups: Sequence[object] | None = None,
        progress_desc: str = "Inference",
        max_samples_to_detail: int = 3,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, object],
    ]:
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        n = len(clean_texts)
        if n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, len(self.label_names)), dtype=np.float32),
                np.empty((0, len(self.label_names)), dtype=np.float32),
                np.empty(0, dtype=bool),
                np.empty((0, len(self.depths)), dtype=np.float32),
                {"sample_inspections": [], "global_top_tokens": []},
            )

        if groups is None:
            groups = _infer_test_groups(clean_texts, thr=TEST_GROUP_THR).tolist()

        self.model.eval()
        self.head.eval()
        all_plain_logits: list[np.ndarray] = []
        all_features: list[np.ndarray] = []
        all_layer_weights: list[np.ndarray] = []
        tracker = TokenImportanceTracker(
            tokenizer=self.tokenizer,
            label_names=self.label_names,
            prefix_len=self.prefix_len,
            max_samples_to_detail=max_samples_to_detail,
            large_dataset_min_occurrences=30,
        )

        batch_starts = range(0, n, self.batch_size)
        for start in tqdm(
            batch_starts,
            total=(n + self.batch_size - 1) // self.batch_size,
            desc=progress_desc,
            unit="batch",
        ):
            batch = clean_texts[start : start + self.batch_size]
            plain_l, feat, layer_w, token_w, inp_ids, att_mask = self.forward_batch(
                batch, return_details=True
            )
            plain_np = plain_l.float().cpu().numpy()
            all_plain_logits.append(plain_np)
            all_features.append(feat.float().cpu().numpy())
            all_layer_weights.append(layer_w.float().cpu().numpy())
            tracker.update_batch(
                batch_texts=batch,
                batch_preds=np.argmax(plain_np, axis=1),
                token_weights=token_w.float().cpu().numpy(),
                input_ids=inp_ids.cpu().numpy(),
                attention_mask=att_mask.cpu().numpy(),
                start_index=start,
            )

        plain_logits = np.concatenate(all_plain_logits, axis=0)
        features = np.concatenate(all_features, axis=0)
        layer_weights = np.concatenate(all_layer_weights, axis=0)

        centered_logits, eligible_mask = self.compute_group_centered_logits(
            features, groups
        )

        lp_plain = log_softmax_np(plain_logits / max(self.plain_temperature, 1e-4))
        lp_cent = log_softmax_np(
            centered_logits / max(self.centered_temperature, 1e-4)
        )
        w = eligible_mask.astype(np.float64)[:, None] * self.centered_blend_weight
        blend_lp = log_softmax_np((1.0 - w) * lp_plain + w * lp_cent)
        blend_preds = blend_lp.argmax(axis=1)

        tracker.update_sample_predictions(blend_preds)
        token_report = tracker.build_report(n)
        return (
            blend_preds,
            plain_logits,
            centered_logits,
            eligible_mask,
            layer_weights,
            token_report,
        )

    def predict(
        self,
        texts: Iterable[object],
        groups: Sequence[object] | None = None,
        progress_desc: str = "Inference",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        preds, _, _, _, layer_weights, token_report = (
            self.predict_with_dual_importance(
                texts, groups=groups, progress_desc=progress_desc
            )
        )
        names = np.asarray([self.label_names[idx] for idx in preds], dtype=object)
        return names, layer_weights, token_report


def _restore_best_weights(
    pipeline: QwenLayerTokenTwinPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重與雙頭融合校準參數載回記憶體中的 pipeline。"""
    loaded = load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_dir,
        head_filename="layer_token_twin_head.pt",
        trainable=None,
        missing_ok=True,
    )
    if not loaded:
        return
    meta_path = pathlib.Path(save_dir) / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        pipeline.plain_temperature = float(metadata.get("plain_temperature", 1.0))
        pipeline.centered_temperature = float(
            metadata.get("centered_temperature", 1.0)
        )
        pipeline.centered_blend_weight = float(
            metadata.get("centered_blend_weight", 0.5)
        )


def _train_pipeline(
    pipeline: QwenLayerTokenTwinPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    groups: Sequence[object],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_v3",
) -> tuple[float, dict[str, object], dict[str, object]]:
    """聯合訓練 TaskLoRA 與 HierarchicalLayerTokenTwinClassifier (Plain + Centered 雙頭)。"""
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

    approx_batches = (len(train_indices) + pipeline.batch_size - 1) // pipeline.batch_size
    steps_per_epoch = max(
        1,
        (approx_batches + GRADIENT_ACCUMULATION_STEPS - 1)
        // GRADIENT_ACCUMULATION_STEPS,
    )
    total_steps = max(1, steps_per_epoch * EPOCHS)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=WARMUP_RATIO
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    rng = np.random.default_rng(SEED)

    val_texts = [texts[i] for i in validation_indices.tolist()]
    val_groups = [groups[i] for i in validation_indices.tolist()]
    val_true = labels[validation_indices]

    best_score = -1.0
    best_epoch = 0
    best_layer_report: dict[str, object] = {}
    best_token_report: dict[str, object] = {}

    for epoch in range(EPOCHS):
        pipeline.model.train()
        pipeline.head.train()

        micro_batches = _build_prompt_paired_batches(
            train_indices, groups, pipeline.batch_size, rng
        )
        paired_count = sum(1 for _, is_same in micro_batches if is_same)
        print(
            f"🎯 Epoch {epoch + 1}/{EPOCHS} 啟動: "
            f"共 {len(micro_batches)} batches（其中 {paired_count} 組為同題目去主題配對）| "
            f"LoRA lr = {optimizer.param_groups[0]['lr']:.2e}"
        )

        total_loss = 0.0
        total_plain_loss = 0.0
        total_cent_loss = 0.0
        seen = 0
        seen_cent = 0

        progress = tqdm(
            range(len(micro_batches)),
            total=len(micro_batches),
            desc=f"Training epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
        )

        for batch_number in progress:
            indices, is_same_prompt = micro_batches[batch_number]
            group_start = (
                batch_number // GRADIENT_ACCUMULATION_STEPS
            ) * GRADIENT_ACCUMULATION_STEPS
            group_end = min(
                group_start + GRADIENT_ACCUMULATION_STEPS, len(micro_batches)
            )
            group_examples = sum(
                len(micro_batches[i][0]) for i in range(group_start, group_end)
            )

            if batch_number == group_start:
                optimizer.zero_grad(set_to_none=True)

            batch_texts = [texts[i] for i in indices]
            targets = torch.as_tensor(
                labels[indices], dtype=torch.long, device=pipeline.head_device
            )
            plain_logits, centered_logits = pipeline.forward_batch(
                batch_texts,
                same_prompt_batch=is_same_prompt,
                return_details=False,
            )

            loss_plain = criterion(plain_logits, targets)
            if centered_logits is not None:
                loss_cent = criterion(centered_logits, targets)
                loss = loss_plain + CENTERED_LOSS_WEIGHT * loss_cent
                total_cent_loss += float(loss_cent.item()) * len(indices)
                seen_cent += len(indices)
            else:
                loss = loss_plain

            (loss * (len(indices) / group_examples)).backward()

            if (batch_number + 1) == group_end:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * len(indices)
            total_plain_loss += float(loss_plain.item()) * len(indices)
            seen += len(indices)
            progress.set_postfix(
                plain=f"{total_plain_loss / seen:.3f}",
                cent=f"{total_cent_loss / max(seen_cent, 1):.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )
            del plain_logits, centered_logits, targets, loss_plain, loss

        clear_cuda_cache()

        (
            _,
            val_plain_logits,
            val_cent_logits,
            val_eligible,
            val_layer_weights,
            val_token_report,
        ) = pipeline.predict_with_dual_importance(
            val_texts,
            groups=val_groups,
            progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
        )

        t_plain, t_cent, best_alpha, raw_ll, blend_ll, blend_lp = (
            fit_temperature_and_blend(
                val_plain_logits, val_cent_logits, val_eligible, val_true
            )
        )

        plain_preds = val_plain_logits.argmax(axis=1)
        cent_preds = np.where(
            val_eligible, val_cent_logits.argmax(axis=1), plain_preds
        )
        blend_preds = blend_lp.argmax(axis=1)

        plain_acc = balanced_accuracy(val_true, plain_preds, len(pipeline.label_names))
        cent_acc = balanced_accuracy(val_true, cent_preds, len(pipeline.label_names))
        blend_acc = balanced_accuracy(val_true, blend_preds, len(pipeline.label_names))

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - train_loss: {total_loss / max(seen, 1):.4f} "
            f"(plain={total_plain_loss / max(seen, 1):.4f}, cent={total_cent_loss / max(seen_cent, 1):.4f})\n"
            f"  📈 Validation Balanced Accuracy -> "
            f"Plain: {plain_acc:.6f} | Centered: {cent_acc:.6f} | ⭐ Twin-Blend (alpha={best_alpha:.2f}): {blend_acc:.6f}\n"
            f"  📉 Val LogLoss -> Raw Plain: {raw_ll:.4f} | Calibrated Twin-Blend (Tp={t_plain:.2f}, Tc={t_cent:.2f}): {blend_ll:.4f}"
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

        if blend_acc > best_score:
            prev_best = best_score
            best_score = blend_acc
            best_epoch = epoch + 1
            best_layer_report = val_layer_report
            best_token_report = val_token_report
            pipeline.plain_temperature = t_plain
            pipeline.centered_temperature = t_cent
            pipeline.centered_blend_weight = best_alpha
            save_model(
                pipeline,
                save_dir,
                layer_report=best_layer_report,
                token_report=best_token_report,
            )
            if prev_best < 0:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: 初次建立最佳模型 "
                    f"(Twin-Blend balanced_accuracy = {best_score:.6f}, alpha={best_alpha:.2f})，已儲存至 {save_dir}！"
                )
            else:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: Twin-Blend balanced_accuracy 創新高 "
                    f"({prev_best:.6f} -> {best_score:.6f}, alpha={best_alpha:.2f})，已更新最佳模型至 {save_dir}！"
                )
        else:
            print(
                f"⚠️ Epoch {epoch + 1}/{EPOCHS}: Twin-Blend balanced_accuracy ({blend_acc:.6f}) "
                f"未超越最佳紀錄 ({best_score:.6f} @ Epoch {best_epoch})，不儲存新模型，繼續訓練..."
            )

    if best_epoch > 0 and best_epoch != EPOCHS:
        print(
            f"🔄 訓練結束：載回 Epoch {best_epoch} 的最佳模型權重 "
            f"(best Twin-Blend balanced_accuracy = {best_score:.6f})"
        )
        _restore_best_weights(pipeline, save_dir)

    return best_score, best_layer_report, best_token_report


def save_model(
    pipeline: QwenLayerTokenTwinPipeline,
    save_dir: str | pathlib.Path,
    layer_report: dict[str, object] | None = None,
    token_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、雙重注意力雙頭權重與分析報告。"""
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
        "plain_temperature": pipeline.plain_temperature,
        "centered_temperature": pipeline.centered_temperature,
        "centered_blend_weight": pipeline.centered_blend_weight,
    }
    save_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        save_dir=save_dir,
        head_filename="layer_token_twin_head.pt",
        metadata=metadata,
        layer_report=layer_report,
        token_report=token_report,
    )
    print(f"Saved model, twin-head calibration, and importance reports to {save_dir}")


def load_model(
    save_dir: str | pathlib.Path,
    base_model_path: str | None = None,
    trainable: bool = False,
) -> QwenLayerTokenTwinPipeline:
    """載入已保存的 TaskLoRA 與 HierarchicalLayerTokenTwinClassifier。"""
    save_path = pathlib.Path(save_dir)
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
    resolved_base = base_model_path or find_qwen_model_dir(metadata["base_model_path"])

    pipeline = QwenLayerTokenTwinPipeline(
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
        plain_temperature=float(metadata.get("plain_temperature", 1.0)),
        centered_temperature=float(metadata.get("centered_temperature", 1.0)),
        centered_blend_weight=float(metadata.get("centered_blend_weight", 0.5)),
    )
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_path,
        head_filename="layer_token_twin_head.pt",
        trainable=trainable,
    )
    pipeline.model.eval()
    pipeline.head.eval()
    return pipeline


def Trainer(
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_v3",
    load_weights: bool = False,
) -> QwenLayerTokenTwinPipeline:
    """讀取資料、建立 Qwen (前 24 層) + TaskLoRA + 同題目去主題雙頭分類器並執行訓練。"""
    seed_everything(SEED)
    data_dir = find_data_dir(CANDIDATES)
    _, texts, labels, label_names, groups = load_train_data(data_dir)
    train_indices, validation_indices = group_split(
        groups, labels.tolist(), VALIDATION_FRACTION, SEED
    )

    if load_weights:
        print(f"Loading existing weights from {save_dir} to continue training...")
        pipeline = load_model(save_dir, trainable=True)
    else:
        model_dir = find_qwen_model_dir(DEFAULT_MODEL_PATH)
        print(f"Using Qwen base model: {model_dir}")
        pipeline = QwenLayerTokenTwinPipeline(
            base_model_path=model_dir,
            label_names=label_names,
        )

    lora_count, head_count = count_qwen_trainable_params(pipeline.loras, pipeline.head)
    print(
        f"監聽的 8 個樓層 (DEPTHS): {pipeline.depths} (各層獨立投影 {pipeline.layer_proj_dim} 維 -> torch.cat 拼接 {pipeline.head.proj_dim} 維) | "
        f"雙頭架構: Plain + Prompt-Centered Twin Head (無 CosFace) | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 Head 參數: {head_count:,}"
    )

    _train_pipeline(
        pipeline,
        texts,
        labels,
        groups,
        train_indices,
        validation_indices,
        save_dir=save_dir,
    )
    return pipeline


def main() -> pd.DataFrame:
    """執行完整訓練、分析 8 層樓與每個 Token 的雙重重要性，並對 test.csv 產生雙頭融合提交檔。"""
    save_dir = "saved_qwen_layertoken_v3"
    pipeline = Trainer(save_dir, load_weights=False)

    clear_cuda_cache()
    test_df = load_test_data(find_data_dir(CANDIDATES))

    test_texts = test_df["text"].fillna("").astype(str).tolist()
    test_groups = _infer_test_groups(test_texts, thr=TEST_GROUP_THR)
    grp_sizes = np.bincount(np.unique(test_groups, return_inverse=True)[1])
    grouped_rows = int(grp_sizes[grp_sizes >= 2].sum()) if len(grp_sizes) else 0
    print(
        f"🔎 測試集自動聚類結果：共 {len(test_texts)} 篇，分成 {len(grp_sizes)} 個題目群組，"
        f"其中 {grouped_rows} 篇啟用 Centered 去主題雙頭聯合預測 (alpha={pipeline.centered_blend_weight:.2f})"
    )

    test_preds, test_layer_weights, test_token_report = pipeline.predict(
        test_texts, groups=test_groups.tolist(), progress_desc="Test Inference"
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
    return save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    submission = main()
    save_kaggle_submission(submission)
