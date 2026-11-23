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
