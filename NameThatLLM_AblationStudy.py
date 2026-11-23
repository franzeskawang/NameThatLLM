# NameThatLLM_AblationStudy.py
# 用途：Kaggle Notebook 一鍵打包腳本。
#       將本檔內容整份貼入 Kaggle Notebook 的單一 Cell 執行（或執行 python NameThatLLM_AblationStudy.py），
#       即可在目前工作目錄自動產生全部 5 個共用模組：
#       data_utils.py、metrics_utils.py、report_utils.py、qwen_utils.py、sentence_lora_utils.py。

from __future__ import annotations

import pathlib


UTILS_FILES: dict[str, str] = {
    'data_utils.py': '''\
# data_utils.py
# 用途：提供資料路徑搜尋、亂數種子固定、GPU 顯存清理、訓練/測試集讀取、
#       以 prompt_id 分組切分資料（防止資料洩漏）與 Kaggle 提交檔輸出等通用資料工具函式。

from __future__ import annotations

import gc
import os
import pathlib
import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch


# Kaggle 訓練資料可能掛載在以下任一資料路徑。
CANDIDATES = [
    pathlib.Path("/kaggle/input/competitions/name-that-llm"),
    pathlib.Path("/kaggle/input/name-that-llm"),
]


def seed_everything(seed: int) -> None:
    """固定常用亂數來源，讓資料切分與模型初始化可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda_cache() -> None:
    """執行垃圾回收並釋放未使用的 CUDA 顯存快取。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_data_dir(candidates: Sequence[pathlib.Path] = CANDIDATES) -> pathlib.Path:
    """依序尋找包含 train.csv 的 Kaggle 資料目錄。"""
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find train.csv in: {searched}")


def group_split(
    groups: Sequence[object], labels: Sequence[int], fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """以 prompt_id 為單位切分訓練集與驗證集，避免同一題目洩漏並確保兩側皆含所有類別。"""
    # 建立 prompt 到資料列索引的對照，確保同一 prompt 不會跨集合。
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

    # 隨機加入整組 prompt，直到達到目標比例且驗證集涵蓋所有類別。
    while remaining_groups and (
        len(validation_groups) < target or validation_classes != all_classes
    ):
        group = remaining_groups.pop()
        validation_groups.append(group)
        validation_classes.update(labels[i] for i in group_to_indices[group])

    # 由選出的 prompt 組合訓練與驗證資料列索引。
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
    # 確認訓練集與驗證集都含有全部類別，否則無法可靠計算指標。
    train_classes = {labels[i] for i in train_indices.tolist()}
    validation_classes = {labels[i] for i in validation_indices.tolist()}
    if train_classes != all_classes or validation_classes != all_classes:
        raise ValueError(
            "Could not create a prompt-group split containing every class in both sets"
        )
    return train_indices, validation_indices


def load_train_data(
    data_dir: pathlib.Path | None = None, expected_classes: int = 18
) -> tuple[pd.DataFrame, list[str], np.ndarray, list[str], list[object]]:
    """讀取並驗證 train.csv，回傳 (train_df, texts, labels, label_names, groups)。"""
    resolved_dir = data_dir or find_data_dir()
    train_df = pd.read_csv(resolved_dir / "train.csv")
    required_train = {"text", "model", "prompt_id"}
    if not required_train.issubset(train_df.columns):
        raise ValueError(f"train.csv must contain columns {sorted(required_train)}")

    texts = train_df["text"].fillna("").astype(str).tolist()
    groups = train_df["prompt_id"].fillna("").tolist()
    if train_df["model"].isna().any():
        raise ValueError("Training data contains missing model labels")

    label_names = sorted(train_df["model"].astype(str).unique().tolist())
    if expected_classes is not None and len(label_names) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} model classes, found {len(label_names)}"
        )

    label_to_index = {name: index for index, name in enumerate(label_names)}
    mapped_labels = train_df["model"].astype(str).map(label_to_index)
    if mapped_labels.isna().any():
        raise ValueError("Training data contains missing or unmapped model labels")
    labels = mapped_labels.to_numpy(dtype=np.int64)
    return train_df, texts, labels, label_names, groups


def load_test_data(data_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """讀取並驗證 test.csv，確認包含 id 與 text 欄位。"""
    resolved_dir = data_dir or find_data_dir()
    test_df = pd.read_csv(resolved_dir / "test.csv")
    if not {"id", "text"}.issubset(test_df.columns):
        raise ValueError("test.csv must contain id and text columns")
    return test_df


def save_submission(
    test_ids: pd.Series | Sequence[object],
    test_preds: np.ndarray | Sequence[object],
    output_path: str | pathlib.Path = "submission.parquet",
) -> pd.DataFrame:
    """建立 id/model 提交 DataFrame 並輸出為 Parquet 檔案。"""
    submission = pd.DataFrame({"id": test_ids, "model": test_preds})
    out_path = pathlib.Path(output_path)
    submission.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(submission)} rows.")
    return submission


def save_kaggle_submission(
    submission: pd.DataFrame, working_dir: str | pathlib.Path = "/kaggle/working"
) -> None:
    """將提交結果同時輸出為 Kaggle 工作目錄下的 Parquet 與 CSV 檔案。"""
    os.makedirs(working_dir, exist_ok=True)
    work_path = pathlib.Path(working_dir)
    submission.to_parquet(work_path / "submission.parquet", index=False)
    submission.to_csv(work_path / "submission.csv", index=False)
''',
    'metrics_utils.py': '''\
# metrics_utils.py
# 用途：提供模型評估指標與機率校準函式，包含 Balanced Accuracy、數值穩定的
#       NumPy Log-Softmax / LogLoss 計算，以及單頭與雙頭（Plain + Centered）
#       的溫度縮放（Temperature Scaling）與最佳融合權重搜尋。

from __future__ import annotations

import numpy as np


def balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """計算各類別召回率的平均值（Balanced Accuracy），避免類別不平衡影響評估。"""
    recalls = []
    for class_id in range(num_classes):
        mask = y_true == class_id
        if mask.any():
            recalls.append(float((y_pred[mask] == class_id).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    """數值穩定的 NumPy log-softmax。"""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def log_loss_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    """以數值穩定的 log-softmax 計算多分類交叉熵（LogLoss）。"""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if len(y) == 0:
        return 0.0
    z_shift = z - z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(z_shift).sum(axis=1))
    nll = log_sum_exp - z_shift[np.arange(len(y)), y]
    return float(nll.mean())


def log_loss_from_logprobs(logprobs: np.ndarray, y_true: np.ndarray) -> float:
    """由 log-probabilities 計算多分類交叉熵（LogLoss）。"""
    if len(y_true) == 0:
        return 0.0
    return float(-logprobs[np.arange(len(y_true)), y_true].mean())


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray
) -> tuple[float, float, float]:
    """
    針對驗證集 logits 搜尋最佳溫度縮放參數 T*（使 logits / T* 的 LogLoss 最小化）。
    回傳 (原始 LogLoss, 溫度校準後 LogLoss, 最佳溫度 T*)。
    """
    raw_ll = log_loss_from_logits(logits, y_true)
    if len(y_true) == 0:
        return raw_ll, raw_ll, 1.0

    # 先在對數空間粗搜，再於最佳點附近進行兩輪細搜。
    grid = np.geomspace(0.1, 10.0, num=81)
    losses = np.array(
        [log_loss_from_logits(logits / t, y_true) for t in grid],
        dtype=np.float64,
    )
    best_idx = int(np.argmin(losses))
    best_t = float(grid[best_idx])
    best_ll = float(losses[best_idx])

    for half_width in (0.25, 0.05):
        lo = max(0.05, best_t * (1.0 - half_width))
        hi = min(20.0, best_t * (1.0 + half_width))
        fine_grid = np.linspace(lo, hi, num=41)
        for t in fine_grid:
            ll = log_loss_from_logits(logits / float(t), y_true)
            if ll < best_ll:
                best_ll = ll
                best_t = float(t)

    return raw_ll, best_ll, best_t


def fit_temperature_and_blend(
    plain_logits: np.ndarray,
    centered_logits: np.ndarray,
    eligible_mask: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """
    雙頭溫度校準與融合搜尋流程：
    1. 各自搜尋 plain_logits 與 centered_logits 的最佳溫度 T_plain, T_centered。
    2. 在驗證集搜尋最佳雙頭融合比例 alpha (0.0 ~ 0.85)，最小化校準後 LogLoss。
    3. 對於群組大小為 1 的孤兒樣本 (eligible_mask=False)，自動只用 plain 預測。
    回傳 (T_plain, T_centered, best_alpha, raw_plain_ll, blend_cal_ll, blend_logprobs)。
    """

    def _best_temp(z: np.ndarray, mask: np.ndarray | None = None) -> float:
        if mask is not None:
            z = z[mask]
            yt = y_true[mask]
        else:
            yt = y_true
        if len(yt) == 0:
            return 1.0
        grid = np.geomspace(0.2, 5.0, num=41)
        best_t, best_ll = 1.0, float("inf")
        for t in grid:
            ll = log_loss_from_logprobs(log_softmax_np(z / float(t)), yt)
            if ll < best_ll:
                best_ll = ll
                best_t = float(t)
        return best_t

    t_plain = _best_temp(plain_logits)
    t_cent = _best_temp(centered_logits, eligible_mask if eligible_mask.any() else None)

    lp_plain = log_softmax_np(plain_logits / t_plain)
    lp_cent = log_softmax_np(centered_logits / t_cent)
    raw_plain_ll = log_loss_from_logprobs(log_softmax_np(plain_logits), y_true)

    best_alpha = 0.0
    best_ll = log_loss_from_logprobs(lp_plain, y_true)
    best_blend = lp_plain

    if eligible_mask.any():
        elig = eligible_mask.astype(np.float64)[:, None]
        for alpha in np.linspace(0.0, 0.85, num=35):
            w = elig * float(alpha)
            mixed = log_softmax_np((1.0 - w) * lp_plain + w * lp_cent)
            ll = log_loss_from_logprobs(mixed, y_true)
            if ll < best_ll:
                best_ll = ll
                best_alpha = float(alpha)
                best_blend = mixed

    return t_plain, t_cent, best_alpha, raw_plain_ll, best_ll, best_blend
''',
    'report_utils.py': '''\
# report_utils.py
# 用途：提供注意力權重分析與終端機報表輸出工具，包含 8 層樓 Layer Attention 權重
#       統計表、全序列 Token 注意力重要性統計累積器（TokenImportanceTracker）與
#       單篇/全域關鍵 Token 排行榜報表。

from __future__ import annotations

import collections
from typing import Sequence

import numpy as np


class TokenImportanceTracker:
    """累積批次推論過程中的 Token Attention 權重，產生單篇 Top-10 與全域 Top-20 報告。"""

    def __init__(
        self,
        tokenizer,
        label_names: Sequence[str],
        prefix_len: int,
        max_samples_to_detail: int = 3,
        large_dataset_min_occurrences: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_names = list(label_names)
        self.prefix_len = int(prefix_len)
        self.max_samples_to_detail = int(max_samples_to_detail)
        self.large_dataset_min_occurrences = int(large_dataset_min_occurrences)

        self.sample_inspections: list[dict[str, object]] = []
        self.token_rel_weight_sum: dict[int, float] = collections.defaultdict(float)
        self.token_abs_weight_sum: dict[int, float] = collections.defaultdict(float)
        self.token_counts: dict[int, int] = collections.defaultdict(int)

    def update_batch(
        self,
        batch_texts: Sequence[str],
        batch_preds: np.ndarray,
        token_weights: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        start_index: int,
    ) -> None:
        """累積單一 batch 的 Token 權重統計與範例文章 Top-10 透視資訊。"""
        mask_bool = attention_mask.astype(bool)
        for r in range(len(batch_texts)):
            valid_len = int(mask_bool[r].sum())
            if valid_len <= 0:
                continue
            row_ids = input_ids[r, :valid_len]
            row_w = token_weights[r, :valid_len]

            # 忽略開頭固定的 INSTRUCTION prefix，專注統計文章本身的 Token（包含句尾 TERMINAL）。
            content_start = min(self.prefix_len, max(valid_len - 1, 0))
            for pos in range(content_start, valid_len):
                tid = int(row_ids[pos])
                w_val = float(row_w[pos])
                self.token_rel_weight_sum[tid] += w_val * valid_len
                self.token_abs_weight_sum[tid] += w_val
                self.token_counts[tid] += 1

            # 收集前 max_samples_to_detail 篇文章的單篇 Top-10 Token 詳情。
            if len(self.sample_inspections) < self.max_samples_to_detail:
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
                self.sample_inspections.append(
                    {
                        "sample_index": int(start_index + r),
                        "predicted_model": self.label_names[int(batch_preds[r])],
                        "valid_tokens": valid_len,
                        "text_preview": batch_texts[r][:100].replace("\\n", " "),
                        "top_tokens": top_tokens_info,
                    }
                )

    def update_sample_predictions(self, final_preds: np.ndarray) -> None:
        """在後處理（例如雙頭融合）完成後，更新範例透視的最終預測模型名稱。"""
        for item in self.sample_inspections:
            s_idx = int(item["sample_index"])
            item["predicted_model"] = self.label_names[int(final_preds[s_idx])]

    def build_report(self, total_samples: int) -> dict[str, object]:
        """整理全域最具影響力的 Top-20 Token 排行榜並回傳完整報告字典。"""
        if total_samples >= 100:
            min_occurrences = self.large_dataset_min_occurrences
        elif total_samples >= 20:
            min_occurrences = 5
        else:
            min_occurrences = 1

        ranked_tids = [
            tid
            for tid, cnt in self.token_counts.items()
            if cnt >= min_occurrences
        ]
        ranked_tids.sort(
            key=lambda tid: self.token_rel_weight_sum[tid] / self.token_counts[tid],
            reverse=True,
        )

        global_top_tokens = []
        for tid in ranked_tids[:20]:
            cnt = self.token_counts[tid]
            avg_rel = self.token_rel_weight_sum[tid] / cnt
            avg_abs = self.token_abs_weight_sum[tid] / cnt
            global_top_tokens.append(
                {
                    "token_id": int(tid),
                    "token": repr(self.tokenizer.decode([int(tid)])),
                    "occurrences": int(cnt),
                    "avg_weight_pct": round(float(avg_abs * 100.0), 4),
                    "avg_relative_multiplier": round(float(avg_rel), 3),
                }
            )

        return {
            "total_samples": int(total_samples),
            "sample_inspections": self.sample_inspections,
            "global_top_tokens": global_top_tokens,
        }


def print_layer_importance_report(
    depths: Sequence[int], weights: np.ndarray, title: str = "8 層樓 Attention 重要性分析"
) -> dict[str, object]:
    """統計並印出各樓層的平均 Attention 權重、標準差與最常主導的樓層比例。"""
    mean_w = weights.mean(axis=0)
    std_w = weights.std(axis=0)
    top1_counts = np.bincount(weights.argmax(axis=1), minlength=len(depths))
    top1_ratio = top1_counts / max(len(weights), 1)
    best_idx = int(np.argmax(mean_w))

    print(f"\\n{'=' * 68}")
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
    print(f"{'=' * 68}\\n")

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
    print(f"\\n{'=' * 78}")
    print(f"🔍 {title} (總樣本數: {report.get('total_samples', 0)})")
    print(f"{'-' * 78}")

    samples = report.get("sample_inspections", [])
    if samples:
        print("📌 [單篇文章透視] 模型最關注的前 10 個關鍵 Token：")
        for sample in samples:
            print(
                f"\\n  ▶ 樣本 #{sample['sample_index']} | 預測模型: {sample['predicted_model']} "
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
        print(f"\\n{'-' * 78}")
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
    print(f"{'=' * 78}\\n")
''',
    'qwen_utils.py': '''\
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
    "Instruct: Identify the language model that generated the text based on its writing style.\\n"
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
''',
    'sentence_lora_utils.py': '''\
# sentence_lora_utils.py
# 用途：提供 SentenceTransformer（如 BGE、all-mpnet-base-v2）搭配 PEFT LoRA 與
#       Transformer 節奏分類器的共用元件，包含 torchao 相容性修補、LoRA 目標模組
#       自動解析與掛載、RhythmTransformerClassifier、SentenceLoRAPipeline、梯度累積
#       訓練迴圈，以及模型權重的存檔與載入。

from __future__ import annotations

import json
import pathlib
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from metrics_utils import balanced_accuracy


def disable_incompatible_torchao_dispatch(
    model_label: str = "SentenceTransformer",
) -> None:
    """停用舊版 torchao 的 PEFT optional dispatcher，避免阻擋一般 Linear LoRA。"""
    try:
        torchao_version = version("torchao")
    except PackageNotFoundError:
        return

    from packaging.version import Version

    if Version(torchao_version) >= Version("0.16.0"):
        return

    try:
        import peft.tuners.lora.torchao as peft_torchao
    except ImportError as error:
        raise RuntimeError(
            "偵測到不相容的 torchao 版本，且無法停用 PEFT 的 torchao dispatcher；"
            "請升級 torchao 至 0.16.0 以上，或移除舊版 torchao。"
        ) from error

    # 此模型使用一般浮點數 Linear 權重，不是 torchao 量化權重；略過可選 dispatcher 即可。
    peft_torchao.is_torchao_available = lambda: False
    print(
        f"Info: 偵測到 torchao {torchao_version}；略過其量化 dispatcher，"
        f"一般 {model_label} Linear 層的 LoRA 訓練不受影響。"
    )


class RhythmTransformerClassifier(nn.Module):
    """以單層 Transformer 編碼 token 節奏序列並分類語言模型類別。"""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        """建立 Transformer 編碼層與最終分類層。"""
        super().__init__()
        if embedding_dim % 4 != 0:
            raise ValueError(
                f"Embedding dimension {embedding_dim} must be divisible by 4"
            )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        # 關閉 nested tensor 最佳化，避免 PyTorch 原型 API 警告；此模型不依賴該功能。
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, rhythm: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """以有效 token 遮罩編碼節奏序列，masked mean pooling 後輸出 logits。"""
        mask = attention_mask.bool()
        encoded = self.transformer(rhythm, src_key_padding_mask=~mask)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(self.dropout(pooled))


class SentenceLoRAPipeline:
    """封裝可訓練 LoRA SentenceTransformer 編碼器與 Transformer 分類器，提供批次預測介面。"""

    def __init__(
        self,
        encoder: SentenceTransformer,
        classifier: RhythmTransformerClassifier,
        label_names: Sequence[str],
        device: torch.device,
        batch_size: int = 4,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.label_names = list(label_names)
        self.device = device
        self.batch_size = batch_size

    def _encode_batch(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """以可微分的前向傳播產生正規化 token 節奏向量與有效 token 遮罩。"""
        features = self.encoder.preprocess(list(texts))
        features = {
            key: value
            for key, value in features.items()
            if isinstance(value, torch.Tensor)
        }
        features = {key: value.to(self.device) for key, value in features.items()}
        # 不停用梯度：分類 loss 必須能回傳到編碼器中的 LoRA adapter。
        output = self.encoder(features)
        token_embeddings = output["token_embeddings"]
        rhythm = F.normalize(token_embeddings, p=2, dim=-1)
        mask = features["attention_mask"].bool()
        rhythm = rhythm.masked_fill(~mask.unsqueeze(-1), 0.0)
        return rhythm, mask

    @torch.inference_mode()
    def predict_indices(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> np.ndarray:
        """分批預測類別索引，並顯示推論進度。"""
        clean_texts = ["" if pd.isna(text) else str(text) for text in texts]
        if not clean_texts:
            return np.empty(0, dtype=np.int64)
        self.encoder.eval()
        self.classifier.eval()
        predictions: list[np.ndarray] = []
        batch_starts = range(0, len(clean_texts), self.batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(clean_texts) + self.batch_size - 1) // self.batch_size,
            desc=progress_desc,
            unit="batch",
        ):
            batch = clean_texts[start : start + self.batch_size]
            rhythm, mask = self._encode_batch(batch)
            logits = self.classifier(rhythm, mask)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions)

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> np.ndarray:
        """將類別索引轉回原始模型名稱。"""
        indices = self.predict_indices(texts, progress_desc=progress_desc)
        return np.asarray([self.label_names[index] for index in indices], dtype=object)


def load_sentence_encoder(
    model_path: str,
    device: torch.device,
    max_seq_length: int | None = None,
) -> SentenceTransformer:
    """載入 SentenceTransformer，必要時限制最大序列長度。"""
    encoder = SentenceTransformer(model_path, device=str(device))
    if max_seq_length is not None and (
        encoder.max_seq_length is None or int(encoder.max_seq_length) > max_seq_length
    ):
        encoder.max_seq_length = max_seq_length
    return encoder


def resolve_lora_target_modules(
    auto_model: nn.Module, preferred: Sequence[str]
) -> list[str]:
    """檢查模型內的注意力線性層名稱，支援 MPNet (q, v)、BERT (query, value) 等命名。"""
    linear_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in auto_model.named_modules()
        if isinstance(module, nn.Linear)
    }
    preferred_list = list(preferred)
    if set(preferred_list).issubset(linear_suffixes):
        return preferred_list
    for candidate in (["q", "v"], ["query", "value"], ["q_proj", "v_proj"]):
        if set(candidate).issubset(linear_suffixes):
            return candidate
    raise ValueError(
        f"Could not find LoRA target modules {preferred_list} in model linear layers: {sorted(linear_suffixes)}"
    )


def attach_peft_lora(
    auto_model: nn.Module,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str],
    metadata: dict[str, object] | None = None,
    auto_resolve: bool = True,
    model_label: str = "SentenceTransformer",
) -> nn.Module:
    """將 PEFT LoRA 附加到指定的注意力線性層。"""
    disable_incompatible_torchao_dispatch(model_label=model_label)
    preferred_targets = (
        list(metadata.get("target_modules", target_modules))
        if metadata
        else list(target_modules)
    )
    resolved_targets = (
        resolve_lora_target_modules(auto_model, preferred_targets)
        if auto_resolve
        else preferred_targets
    )
    lora_config = LoraConfig(
        r=int(metadata.get("lora_rank", lora_rank)) if metadata else lora_rank,
        lora_alpha=(
            int(metadata.get("lora_alpha", lora_alpha)) if metadata else lora_alpha
        ),
        target_modules=resolved_targets,
        lora_dropout=(
            float(metadata.get("lora_dropout", lora_dropout))
            if metadata
            else lora_dropout
        ),
        bias="none",
    )
    return get_peft_model(auto_model, lora_config)


def train_sentence_lora_classifier(
    pipeline: SentenceLoRAPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
) -> float:
    """以梯度累積訓練 PEFT LoRA 與 Transformer 分類器，並回傳最後一輪驗證 balanced accuracy。"""
    classifier = pipeline.classifier
    trainable_parameters = [
        parameter
        for parameter in pipeline.encoder.parameters()
        if parameter.requires_grad
    ] + list(classifier.parameters())
    if not trainable_parameters:
        raise ValueError("No trainable LoRA or classifier parameters were found")
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    score = 0.0
    for epoch in range(epochs):
        pipeline.encoder.train()
        classifier.train()
        order = train_indices.copy()
        rng.shuffle(order)
        total_loss = 0.0
        seen = 0
        micro_batches = [
            order[start : start + micro_batch_size]
            for start in range(0, len(order), micro_batch_size)
        ]
        progress = tqdm(
            range(len(micro_batches)),
            total=len(micro_batches),
            desc=f"Training epoch {epoch + 1}/{epochs}",
            unit="batch",
        )
        for batch_number in progress:
            indices = micro_batches[batch_number]
            group_start = (
                batch_number // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_end = min(
                group_start + gradient_accumulation_steps, len(micro_batches)
            )
            group_examples = sum(
                len(micro_batches[i]) for i in range(group_start, group_end)
            )
            if batch_number == group_start:
                optimizer.zero_grad(set_to_none=True)
            batch_texts = [texts[i] for i in indices]
            rhythm, mask = pipeline._encode_batch(batch_texts)
            targets = torch.as_tensor(
                labels[indices], dtype=torch.long, device=pipeline.device
            )
            logits = classifier(rhythm, mask)
            loss = criterion(logits, targets)
            (loss * (len(indices) / group_examples)).backward()
            if (batch_number + 1) == group_end:
                optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")

        validation_predictions = pipeline.predict_indices(
            (texts[i] for i in validation_indices.tolist()),
            progress_desc=f"Validation epoch {epoch + 1}/{epochs}",
        )
        score = balanced_accuracy(
            labels[validation_indices],
            validation_predictions,
            len(pipeline.label_names),
        )
        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / max(seen, 1):.4f} "
            f"- validation balanced_accuracy: {score:.6f}"
        )
    return score


def save_sentence_lora_model(
    pipeline: SentenceLoRAPipeline,
    save_dir: str | pathlib.Path,
    base_model_path: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: Sequence[str],
    lora_dropout: float,
) -> None:
    """保存 PEFT LoRA 權重、分類器權重及重建 pipeline 所需設定。"""
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    auto_model = pipeline.encoder[0].auto_model
    lora_state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in auto_model.state_dict().items()
        if "lora_" in name.lower()
    }
    if not lora_state_dict:
        raise RuntimeError("No LoRA parameters found in encoder when saving model")
    torch.save(lora_state_dict, save_path / "lora_weights.pt")
    torch.save(pipeline.classifier.state_dict(), save_path / "classifier.pt")
    metadata = {
        "label_names": pipeline.label_names,
        "embedding_dim": pipeline.classifier.classifier.in_features,
        "num_classes": pipeline.classifier.classifier.out_features,
        "batch_size": pipeline.batch_size,
        "base_model_path": base_model_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
        "lora_dropout": lora_dropout,
    }
    (save_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model to {save_path}")


def load_sentence_lora_model(
    save_dir: str | pathlib.Path,
    default_lora_rank: int,
    default_lora_alpha: int,
    default_target_modules: Sequence[str],
    default_lora_dropout: float,
    device: torch.device | str | None = None,
    base_model_path: str | pathlib.Path | None = None,
    trainable: bool = False,
    max_seq_length: int | None = None,
    model_label: str = "SentenceTransformer",
) -> SentenceLoRAPipeline:
    """載入已保存的 PEFT LoRA pipeline；trainable=True 時保留可訓練狀態以接續訓練。"""
    save_path = pathlib.Path(save_dir)
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base_path = str(base_model_path or metadata["base_model_path"])
    disable_incompatible_torchao_dispatch(model_label=model_label)
    encoder = load_sentence_encoder(
        base_path, resolved_device, max_seq_length=max_seq_length
    )
    transformer_module = encoder[0]
    if not hasattr(transformer_module, "auto_model"):
        raise TypeError("SentenceTransformer first module does not expose auto_model")
    for parameter in transformer_module.auto_model.parameters():
        parameter.requires_grad_(False)

    transformer_module.auto_model = attach_peft_lora(
        transformer_module.auto_model,
        lora_rank=default_lora_rank,
        lora_alpha=default_lora_alpha,
        lora_dropout=default_lora_dropout,
        target_modules=default_target_modules,
        metadata=metadata,
        auto_resolve=True,
        model_label=model_label,
    )

    lora_weights_path = save_path / "lora_weights.pt"
    adapter_dir = save_path / "lora_adapter"
    if lora_weights_path.exists():
        lora_state_dict = torch.load(
            lora_weights_path, map_location=resolved_device, weights_only=True
        )
        incompatible = transformer_module.auto_model.load_state_dict(
            lora_state_dict, strict=False
        )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected LoRA keys when loading model: {incompatible.unexpected_keys}"
            )
    elif adapter_dir.exists():
        # 相容舊版 save_pretrained 輸出的權重檔（safetensors / bin）。
        safetensors_files = [
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "model.safetensors",
        ]
        bin_files = [
            adapter_dir / "adapter_model.bin",
            adapter_dir / "pytorch_model.bin",
        ]
        loaded_state: dict[str, torch.Tensor] | None = None
        for sf_path in safetensors_files:
            if sf_path.exists():
                from safetensors.torch import load_file

                loaded_state = load_file(str(sf_path), device=str(resolved_device))
                break
        if loaded_state is None:
            for bin_path in bin_files:
                if bin_path.exists():
                    loaded_state = torch.load(
                        bin_path, map_location=resolved_device, weights_only=True
                    )
                    break
        if loaded_state is None:
            raise FileNotFoundError(
                f"Could not find lora_weights.pt in {save_path} or weight files in {adapter_dir}"
            )
        lora_only = {k: v for k, v in loaded_state.items() if "lora_" in k.lower()}
        transformer_module.auto_model.load_state_dict(
            lora_only or loaded_state, strict=False
        )
    else:
        raise FileNotFoundError(
            f"Could not find LoRA weights at {lora_weights_path} or {adapter_dir}"
        )

    for name, parameter in transformer_module.auto_model.named_parameters():
        parameter.requires_grad_(trainable and "lora_" in name.lower())

    classifier = RhythmTransformerClassifier(
        int(metadata["embedding_dim"]), int(metadata["num_classes"])
    ).to(resolved_device)
    classifier.load_state_dict(
        torch.load(
            save_path / "classifier.pt",
            map_location=resolved_device,
            weights_only=True,
        )
    )
    for parameter in classifier.parameters():
        parameter.requires_grad_(trainable)
    encoder.eval()
    classifier.eval()
    return SentenceLoRAPipeline(
        encoder,
        classifier,
        metadata["label_names"],
        resolved_device,
        batch_size=int(metadata["batch_size"]),
    )


def log_peft_lora_params(encoder: nn.Module) -> None:
    """驗證並印出 SentenceTransformer encoder 中可訓練的 LoRA 參數量與比例。"""
    lora_parameters = [
        (name, parameter)
        for name, parameter in encoder.named_parameters()
        if "lora_" in name.lower() and parameter.requires_grad
    ]
    if not lora_parameters:
        raise RuntimeError(
            "PEFT 未在 SentenceTransformer encoder 中建立可訓練的 LoRA 參數；"
            "請確認 PEFT、Transformers 與 Sentence Transformers 版本相容。"
        )
    trainable_count = sum(
        parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in encoder.parameters())
    print(
        f"LoRA trainable parameters: {trainable_count:,} / {total_count:,} "
        f"({100 * trainable_count / total_count:.4f}%)"
    )
''',
}


def write_utils_files(output_dir: str | pathlib.Path = ".") -> list[pathlib.Path]:
    """將打包的 5 個 *_utils.py 共用模組寫入指定目錄（預設為目前工作目錄）。"""
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for filename, content in UTILS_FILES.items():
        target = out_dir / filename
        target.write_text(content, encoding="utf-8")
        written.append(target)
        print(f"Writing {target} ({len(content):,} chars)")
    return written


if __name__ == "__main__":
    write_utils_files()
