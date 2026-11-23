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
                        "text_preview": batch_texts[r][:100].replace("\n", " "),
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
