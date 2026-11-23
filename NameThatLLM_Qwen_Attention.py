"""以 TaskLoRA 微調 Qwen-3-Embedding，並外接 8 層樓 Gated Layer Attention 判斷各層重要性與分類 18 種 LLM。"""

from __future__ import annotations

import json
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
from metrics_utils import balanced_accuracy
from qwen_utils import (
    DEFAULT_MODEL_PATH,
    INSTRUCTION,
    LORA_SUFFIXES,
    TaskLoRA,
    count_qwen_trainable_params,
    find_qwen_model_dir,
    load_qwen_backbone,
    load_task_lora_and_head,
    resolve_depths,
    save_task_lora_and_head,
    tokenize_qwen_batch,
)
from report_utils import print_layer_importance_report


DEFAULT_DEPTHS = (3, 6, 9, 12, 18, 24, 30, 36)

# 訓練超參數設定。
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
ATTN_DIM = 128
DROPOUT = 0.10


class GatedLayerAttentionHead(nn.Module):
    """
    外接於 8 層樓輸出的 Gated Layer Attention 分類頭：
    1. 每層樓各自經過獨立的 LayerNorm 標準化。
    2. 透過 Gated Attention 網路 (Linear -> Tanh -> Linear -> Softmax) 計算 8 層樓各自的重要性權重。
    3. 將 8 層樓依權重加權融合為單一特徵向量，再經由分類層輸出 18 個模型的分數。
    """

    def __init__(
        self,
        hidden_size: int,
        depths: Sequence[int],
        num_classes: int = 18,
        attn_dim: int = ATTN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depths = tuple(int(d) for d in depths)
        self.num_layers = len(self.depths)

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size, dtype=torch.float32) for _ in self.depths]
        )
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_size, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes, dtype=torch.float32)

    def forward(
        self, layer_features: list[torch.Tensor], return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(layer_features) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layer features, got {len(layer_features)}"
            )

        normed = [
            norm(feat.float()) for norm, feat in zip(self.norms, layer_features)
        ]
        stacked = torch.stack(normed, dim=1)
        scores = self.attn_gate(stacked)
        weights = F.softmax(scores, dim=1)
        fused = (stacked * weights).sum(dim=1)
        logits = self.classifier(self.dropout(fused))

        if return_weights:
            return logits, weights.squeeze(-1)
        return logits


class QwenLayerAttentionPipeline:
    """封裝 Qwen-3-Embedding + TaskLoRA + 8 層樓 Gated Layer Attention 的訓練與推論流程。"""

    def __init__(
        self,
        base_model_path: str,
        label_names: Sequence[str],
        batch_size: int = MICRO_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        lora_rank: int = LORA_RANK,
        lora_scale: float = LORA_SCALE,
        attn_dim: int = ATTN_DIM,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.label_names = list(label_names)
        self.batch_size = batch_size
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.attn_dim = attn_dim

        backbone = load_qwen_backbone(
            base_model_path=self.base_model_path,
            lora_rank=self.lora_rank,
            lora_scale=self.lora_scale,
            keep_layers=None,
            multi_gpu_cut_ratio=0.38,
            lora_suffixes=LORA_SUFFIXES,
            instruction=INSTRUCTION,
        )
        self.tokenizer = backbone.tokenizer
        self.terminal_id = backbone.terminal_id
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
            expected_layers=36,
        )

        self.head = GatedLayerAttentionHead(
            hidden_size=self.hidden_size,
            depths=self.depths,
            num_classes=len(self.label_names),
            attn_dim=self.attn_dim,
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
        self, texts: Sequence[str], return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._tokenize_batch(texts)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        pos = attention_mask.sum(dim=1) - 1
        rows = torch.arange(len(input_ids))

        layer_features: list[torch.Tensor] = []
        for depth in self.depths:
            hs = (
                out.last_hidden_state
                if depth == self.num_hidden_layers
                else out.hidden_states[depth]
            )
            pooled = hs[rows.to(hs.device), pos.to(hs.device)].float()
            if depth == self.num_hidden_layers:
                pooled = F.normalize(pooled, dim=-1)
            layer_features.append(pooled.to(self.head_device))

        del out, hs
        return self.head(layer_features, return_weights=return_weights)

    @torch.inference_mode()
    def predict_with_layer_weights(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, np.ndarray]:
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        if not clean_texts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, len(self.depths)), dtype=np.float32),
            )

        self.model.eval()
        self.head.eval()
        preds: list[np.ndarray] = []
        all_weights: list[np.ndarray] = []
        batch_starts = range(0, len(clean_texts), self.batch_size)

        for start in tqdm(
            batch_starts,
            total=(len(clean_texts) + self.batch_size - 1) // self.batch_size,
            desc=progress_desc,
            unit="batch",
        ):
            batch = clean_texts[start : start + self.batch_size]
            logits, layer_w = self.forward_batch(batch, return_weights=True)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            all_weights.append(layer_w.float().cpu().numpy())

        return np.concatenate(preds), np.concatenate(all_weights, axis=0)

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, weights = self.predict_with_layer_weights(
            texts, progress_desc=progress_desc
        )
        names = np.asarray([self.label_names[idx] for idx in indices], dtype=object)
        return names, weights


def _restore_best_weights(
    pipeline: QwenLayerAttentionPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重載回記憶體中的 pipeline。"""
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_dir,
        head_filename="layer_attention_head.pt",
        trainable=None,
        missing_ok=True,
    )


def _train_pipeline(
    pipeline: QwenLayerAttentionPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_attention",
) -> tuple[float, dict[str, object]]:
    """聯合訓練 TaskLoRA 與 8 層樓 Gated Layer Attention，保留最佳驗證模型。"""
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
    rng = np.random.default_rng(SEED)

    best_score = -1.0
    best_epoch = 0
    best_report: dict[str, object] = {}

    for epoch in range(EPOCHS):
        pipeline.model.train()
        pipeline.head.train()
        order = train_indices.copy()
        rng.shuffle(order)
        total_loss = 0.0
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
            logits = pipeline.forward_batch(batch_texts, return_weights=False)
            targets = torch.as_tensor(
                labels[indices], dtype=torch.long, device=logits.device
            )
            loss = criterion(logits, targets)
            (loss * (len(indices) / group_examples)).backward()

            if (batch_number + 1) == group_end:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")
            del logits, targets, loss

        clear_cuda_cache()

        val_preds, val_weights = pipeline.predict_with_layer_weights(
            (texts[i] for i in validation_indices.tolist()),
            progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
        )
        score = balanced_accuracy(
            labels[validation_indices], val_preds, len(pipeline.label_names)
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / max(seen, 1):.4f} "
            f"- validation balanced_accuracy: {score:.6f}"
        )
        val_report = print_layer_importance_report(
            pipeline.depths,
            val_weights,
            title=f"Epoch {epoch + 1} 驗證集 8 層樓 Attention 權重統計",
        )

        if score > best_score:
            prev_best = best_score
            best_score = score
            best_epoch = epoch + 1
            best_report = val_report
            save_model(pipeline, save_dir, layer_report=best_report)
            if prev_best < 0:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: 初次建立最佳模型 "
                    f"(balanced_accuracy = {best_score:.6f})，已儲存至 {save_dir}！"
                )
            else:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: validation balanced_accuracy 創新高 "
                    f"({prev_best:.6f} -> {best_score:.6f})，已更新最佳模型至 {save_dir}！"
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

    return best_score, best_report


def save_model(
    pipeline: QwenLayerAttentionPipeline,
    save_dir: str | pathlib.Path,
    layer_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、Gated Layer Attention 分類頭權重與樓層重要性實驗報告。"""
    metadata = {
        "label_names": pipeline.label_names,
        "base_model_path": pipeline.base_model_path,
        "depths": list(pipeline.depths),
        "hidden_size": pipeline.hidden_size,
        "num_classes": len(pipeline.label_names),
        "batch_size": pipeline.batch_size,
        "max_length": pipeline.max_length,
        "lora_rank": pipeline.lora_rank,
        "lora_scale": pipeline.lora_scale,
        "attn_dim": pipeline.attn_dim,
    }
    save_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        save_dir=save_dir,
        head_filename="layer_attention_head.pt",
        metadata=metadata,
        layer_report=layer_report,
    )
    print(f"Saved model and layer importance report to {save_dir}")


def load_model(
    save_dir: str | pathlib.Path,
    base_model_path: str | None = None,
    trainable: bool = False,
) -> QwenLayerAttentionPipeline:
    """載入已保存的 TaskLoRA 與 Gated Layer Attention 分類頭。"""
    save_path = pathlib.Path(save_dir)
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
    resolved_base = base_model_path or find_qwen_model_dir(metadata["base_model_path"])

    pipeline = QwenLayerAttentionPipeline(
        base_model_path=resolved_base,
        label_names=metadata["label_names"],
        batch_size=int(metadata["batch_size"]),
        max_length=int(metadata["max_length"]),
        lora_rank=int(metadata["lora_rank"]),
        lora_scale=float(metadata["lora_scale"]),
        attn_dim=int(metadata["attn_dim"]),
    )
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_path,
        head_filename="layer_attention_head.pt",
        trainable=trainable,
    )
    pipeline.model.eval()
    pipeline.head.eval()
    return pipeline


def Trainer(
    save_dir: str | pathlib.Path = "saved_qwen_attention",
    load_weights: bool = False,
) -> QwenLayerAttentionPipeline:
    """讀取資料、建立 Qwen + TaskLoRA + 8 層樓 Gated Layer Attention 並執行訓練。"""
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
        pipeline = QwenLayerAttentionPipeline(
            base_model_path=model_dir,
            label_names=label_names,
        )

    lora_count, head_count = count_qwen_trainable_params(pipeline.loras, pipeline.head)
    print(
        f"監聽的 8 個樓層 (DEPTHS): {pipeline.depths} | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 Gated Layer Attention 頭參數: {head_count:,}"
    )

    _train_pipeline(
        pipeline,
        texts,
        labels,
        train_indices,
        validation_indices,
        save_dir=save_dir,
    )
    return pipeline


def main() -> pd.DataFrame:
    """執行完整訓練、分析 8 層樓重要性，並對 test.csv 產生預測提交檔。"""
    save_dir = "saved_qwen_attention"
    pipeline = Trainer(save_dir, load_weights=False)

    clear_cuda_cache()
    test_df = load_test_data(find_data_dir(CANDIDATES))

    test_preds, test_layer_weights = pipeline.predict(
        test_df["text"].fillna(""), progress_desc="Test Inference"
    )
    print_layer_importance_report(
        pipeline.depths,
        test_layer_weights,
        title="測試集 (test.csv) 8 層樓 Attention 權重統計",
    )
    return save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    submission = main()
    save_kaggle_submission(submission)
