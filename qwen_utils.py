# qwen_utils.py
# 用途：提供 Qwen-3-Embedding 骨幹模型與 TaskLoRA 的共用元件與工具函式，包含
#       模型目錄自動搜尋、樓層選擇、TaskLoRA 外掛層、單/雙 GPU 載入與層數截斷、
#       批次斷詞、權重存檔/載入/還原、可訓練參數量統計與 Cosine Warmup 學習率排程。

from __future__ import annotations

import glob
import json
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data_utils import clear_cuda_cache


# 預設 Qwen 模型路徑（若不存在，程式會自動在 /kaggle/input 下搜尋 Qwen-3-Embedding 或 Qwen-3 模型）。
DEFAULT_MODEL_PATH = os.environ.get(
    "NTL_MODEL_PATH",
    "/kaggle/input/qwen-3-embedding/transformers/8b/1",
)

# 風格辨識指令與句尾結束符 <|endoftext|> (151643)。
INSTRUCTION = (
    "Instruct: Identify the language model that generated the text based on its writing style.\n"
    "Query: "
)
TERMINAL_TOKEN_ID = 151643

LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def find_qwen_model_dir(preferred_path: str = DEFAULT_MODEL_PATH) -> str:
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


def resolve_depths(
    num_hidden_layers: int,
    default_depths: Sequence[int],
    target_count: int = 8,
    expected_layers: int | None = None,
) -> tuple[int, ...]:
    """若模型層數符合預期則使用預設監聽樓層，否則自動等距選出 target_count 個樓層。"""
    depths_tuple = tuple(int(d) for d in default_depths)
    layer_match = expected_layers is None or num_hidden_layers == expected_layers
    if (
        layer_match
        and len(depths_tuple) == target_count
        and depths_tuple
        and max(depths_tuple) <= num_hidden_layers
    ):
        return depths_tuple
    depths = np.linspace(1, num_hidden_layers, num=target_count, dtype=int).tolist()
    return tuple(sorted(set(depths)))


class TaskLoRA(nn.Module):
    """輕量級 LoRA 外掛層，凍結原始權重並訓練低秩矩陣 A 與 B。"""

    def __init__(
        self, base_layer: nn.Linear, rank: int = 16, scale: float = 2.0
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
        # 顯存優化：將小型 LoRA 權重轉為 x.dtype 計算，梯度仍會自動回傳至 FP32 權重。
        low_rank = F.linear(x, self.a.weight.to(x.dtype))
        lora_delta = F.linear(low_rank, self.b.weight.to(x.dtype)) * self.scale
        return base_out + lora_delta


@dataclass
class QwenBackboneBundle:
    """封裝載入完成的 Qwen 骨幹模型、Tokenizer、TaskLoRA 與裝置配置。"""

    tokenizer: object
    terminal_id: int
    prefix_len: int
    model: nn.Module
    loras: dict[str, TaskLoRA]
    num_hidden_layers: int
    hidden_size: int
    input_device: torch.device
    head_device: torch.device


def load_qwen_backbone(
    base_model_path: str,
    lora_rank: int = 16,
    lora_scale: float = 2.0,
    keep_layers: int | None = None,
    multi_gpu_cut_ratio: float | None = None,
    lora_suffixes: Sequence[str] = LORA_SUFFIXES,
    instruction: str = INSTRUCTION,
) -> QwenBackboneBundle:
    """
    載入 Qwen-3-Embedding 骨幹模型（支援截斷前 keep_layers 層）、配置單/雙 GPU
    device_map、啟用 Gradient Checkpointing，並掛上 TaskLoRA。
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 固定使用 Qwen-3-Embedding 預訓練規定的句尾結束符 <|endoftext|> (151643)。
    terminal_id = TERMINAL_TOKEN_ID
    prefix_len = len(tokenizer(instruction, add_special_tokens=False)["input_ids"])

    cfg = None
    orig_layers: int | None = None
    if keep_layers is not None:
        cfg = AutoConfig.from_pretrained(base_model_path, local_files_only=True)
        tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
        orig_layers = int(tc.num_hidden_layers)
        num_hidden_layers = min(int(keep_layers), orig_layers)
        tc.num_hidden_layers = num_hidden_layers
        tc.use_cache = False
        if getattr(tc, "layer_types", None) is not None:
            tc.layer_types = tc.layer_types[:num_hidden_layers]
        hidden_size = int(tc.hidden_size)
        print(
            f"✂️ 已截斷 Qwen 模型至前 {num_hidden_layers} 層 "
            f"(原 {orig_layers} 層，第 {num_hidden_layers + 1}~{orig_layers} 層不載入亦不運算)"
        )
    else:
        cfg_path = os.path.join(base_model_path, "config.json")
        cfg_json = json.loads(pathlib.Path(cfg_path).read_text(encoding="utf-8"))
        num_hidden_layers = int(cfg_json["num_hidden_layers"])
        hidden_size = int(cfg_json["hidden_size"])

    clear_cuda_cache()

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        ratio = (
            multi_gpu_cut_ratio
            if multi_gpu_cut_ratio is not None
            else (0.40 if keep_layers is not None else 0.38)
        )
        cut = int(round(num_hidden_layers * ratio))
        device_map: dict[str, int] | str = {
            "embed_tokens": 0,
            "norm": n_gpus - 1,
            "rotary_emb": n_gpus - 1,
        }
        device_map.update(
            {f"layers.{i}": 0 if i < cut else 1 for i in range(num_hidden_layers)}
        )
        input_device = torch.device("cuda:0")
        head_device = torch.device(f"cuda:{n_gpus - 1}")
    elif n_gpus == 1:
        device_map = {"": 0}
        input_device = torch.device("cuda:0")
        head_device = torch.device("cuda:0")
    else:
        device_map = "cpu"
        input_device = torch.device("cpu")
        head_device = torch.device("cpu")

    dtype = torch.float16 if n_gpus >= 1 else torch.float32
    from_pretrained_kwargs: dict[str, object] = {
        "local_files_only": True,
        "dtype": dtype,
        "device_map": device_map,
        "attn_implementation": "sdpa",
    }
    if cfg is not None:
        from_pretrained_kwargs["config"] = cfg

    model = AutoModel.from_pretrained(base_model_path, **from_pretrained_kwargs)
    if (
        orig_layers is not None
        and num_hidden_layers < orig_layers
        and hasattr(model, "norm")
    ):
        model.norm = nn.Identity()

    model.config.use_cache = False
    model.requires_grad_(False)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    loras: dict[str, TaskLoRA] = {}
    suffix_set = set(lora_suffixes)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in suffix_set:
            wrapped = TaskLoRA(mod, rank=lora_rank, scale=lora_scale)
            parent, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), attr, wrapped)
            loras[name] = wrapped

    return QwenBackboneBundle(
        tokenizer=tokenizer,
        terminal_id=terminal_id,
        prefix_len=prefix_len,
        model=model,
        loras=loras,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        input_device=input_device,
        head_device=head_device,
    )


def tokenize_qwen_batch(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    terminal_id: int,
    input_device: torch.device,
    instruction: str = INSTRUCTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """加上風格辨識指令並補上句尾 TERMINAL token，回傳 input_ids 與 attention_mask。"""
    prompted = [instruction + ("" if pd.isna(t) else str(t)) for t in texts]
    raw_ids = tokenizer(
        prompted,
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )["input_ids"]
    seqs = [(ids + [terminal_id]) if ids else [terminal_id] for ids in raw_ids]
    padded = tokenizer.pad({"input_ids": seqs}, padding=True, return_tensors="pt")
    return (
        padded["input_ids"].to(input_device),
        padded["attention_mask"].to(input_device),
    )


def save_task_lora_and_head(
    loras: dict[str, TaskLoRA],
    head: nn.Module,
    save_dir: str | pathlib.Path,
    head_filename: str,
    metadata: dict[str, object],
    layer_report: dict[str, object] | None = None,
    token_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、分類頭權重、metadata.json 與可選的樓層/Token 分析報告。"""
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    lora_state = {}
    for name, mod in loras.items():
        lora_state[f"{name}.lora_A.weight"] = mod.a.weight.detach().cpu()
        lora_state[f"{name}.lora_B.weight"] = mod.b.weight.detach().cpu()

    torch.save(lora_state, save_path / "lora_weights.pt")
    torch.save(head.state_dict(), save_path / head_filename)

    (save_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if layer_report is not None:
        (save_path / "layer_importance.json").write_text(
            json.dumps(layer_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if token_report is not None:
        (save_path / "token_importance.json").write_text(
            json.dumps(token_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_task_lora_and_head(
    loras: dict[str, TaskLoRA],
    head: nn.Module,
    head_device: torch.device,
    save_dir: str | pathlib.Path,
    head_filename: str,
    trainable: bool | None = None,
    missing_ok: bool = False,
) -> bool:
    """
    載入已保存的 TaskLoRA 與分類頭權重。
    - trainable 為 bool 時同步設定 requires_grad；為 None 時保留原本狀態（適用於還原最佳權重）。
    - missing_ok=True 時若檔案不存在直接回傳 False，否則正常載入並回傳 True。
    """
    save_path = pathlib.Path(save_dir)
    lora_path = save_path / "lora_weights.pt"
    head_path = save_path / head_filename
    if missing_ok and (not lora_path.exists() or not head_path.exists()):
        return False

    lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, mod in loras.items():
            a_key = f"{name}.lora_A.weight"
            b_key = f"{name}.lora_B.weight"
            if a_key not in lora_state or b_key not in lora_state:
                raise KeyError(f"Missing LoRA weights for {name}")
            mod.a.weight.copy_(lora_state[a_key].to(mod.a.weight.device))
            mod.b.weight.copy_(lora_state[b_key].to(mod.b.weight.device))
            if trainable is not None:
                mod.a.weight.requires_grad_(trainable)
                mod.b.weight.requires_grad_(trainable)

    head_state = torch.load(head_path, map_location=head_device, weights_only=True)
    head.load_state_dict(head_state)
    if trainable is not None:
        head.requires_grad_(trainable)
    return True


def count_qwen_trainable_params(
    loras: dict[str, TaskLoRA], head: nn.Module
) -> tuple[int, int]:
    """統計 TaskLoRA 與分類頭各自的可訓練參數總數。"""
    lora_count = sum(
        p.numel()
        for mod in loras.values()
        for p in (mod.a.weight, mod.b.weight)
        if p.requires_grad
    )
    head_count = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return lora_count, head_count


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """建立 Linear Warmup + Cosine Annealing 學習率排程器。"""
    safe_total = max(1, int(total_steps))
    warmup_steps = max(1, int(safe_total * warmup_ratio))

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress_ratio = float(current_step - warmup_steps) / float(
            max(1, safe_total - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress_ratio)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
