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
