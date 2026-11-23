"""以 TaskLoRA 微調 Qwen-3-Embedding，取最後一層完整 Token 序列 (Batch, 512, 4096) 做注意力機制並分類 18 種 LLM。"""

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
    save_task_lora_and_head,
    tokenize_qwen_batch,
)
from report_utils import TokenImportanceTracker, print_token_importance_report


# 訓練與模型超參數設定。
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
PROJ_DIM = 256
NUM_HEADS = 4
ATTN_DIM = 128
DROPOUT = 0.10


class AllTokenAttentionClassifier(nn.Module):
    """
    接收最後一層完整 Token 序列的 hidden_state (Batch, Seq_Len, 4096) 與 attention_mask (Batch, Seq_Len)：
    1. 對每個 Token 做 LayerNorm 後投影至精煉維度 proj_dim (預設 256)。
    2. 透過單層 Multi-Head Self-Attention (TransformerEncoderLayer) 讓序列中所有 Token 進行上下文注意力互動。
    3. 透過 Gated Token Attention 計算每個有效 Token 的重要性權重 (將 Padding 遮罩為 -1e9)。
    4. 結合「Token Attention 加權特徵」與「Masked Mean Pooling 全文平均特徵」送入分類層輸出 18 類 logits。
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_classes: int = 18,
        proj_dim: int = PROJ_DIM,
        num_heads: int = NUM_HEADS,
        attn_dim: int = ATTN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        if proj_dim % num_heads != 0:
            raise ValueError(
                f"proj_dim ({proj_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.attn_dim = attn_dim

        self.input_norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, proj_dim, dtype=torch.float32),
            nn.GELU(),
            nn.LayerNorm(proj_dim, dtype=torch.float32),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 2,
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
            nn.Linear(proj_dim, attn_dim, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False, dtype=torch.float32),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim * 2, num_classes, dtype=torch.float32)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        return_token_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        valid_mask = attention_mask.bool()
        x = self.proj(self.input_norm(hidden_state.float()))
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        x = self.self_attn(x, src_key_padding_mask=~valid_mask)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        raw_scores = self.token_gate(x).squeeze(-1)
        masked_scores = raw_scores.masked_fill(~valid_mask, -1e9)
        token_weights = F.softmax(masked_scores, dim=1)
        token_weights = token_weights.masked_fill(~valid_mask, 0.0)

        attn_pooled = (x * token_weights.unsqueeze(-1)).sum(dim=1)
        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pooled = x.sum(dim=1) / denom

        combined = torch.cat([attn_pooled, mean_pooled], dim=-1)
        logits = self.classifier(self.dropout(combined))

        if return_token_weights:
            return logits, token_weights
        return logits


class QwenAllTokenPipeline:
    """封裝 Qwen-3-Embedding + TaskLoRA + 最後一層全 Token 注意力分類器的訓練與推論流程。"""

    def __init__(
        self,
        base_model_path: str,
        label_names: Sequence[str],
        batch_size: int = MICRO_BATCH_SIZE,
        max_length: int = MAX_LENGTH,
        lora_rank: int = LORA_RANK,
        lora_scale: float = LORA_SCALE,
        proj_dim: int = PROJ_DIM,
        num_heads: int = NUM_HEADS,
        attn_dim: int = ATTN_DIM,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.label_names = list(label_names)
        self.batch_size = batch_size
        self.max_length = max_length
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.proj_dim = proj_dim
        self.num_heads = num_heads
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
        self.prefix_len = backbone.prefix_len
        self.model = backbone.model
        self.loras = backbone.loras
        self.num_hidden_layers = backbone.num_hidden_layers
        self.hidden_size = backbone.hidden_size
        self.input_device = backbone.input_device
        self.head_device = backbone.head_device

        self.head = AllTokenAttentionClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.label_names),
            proj_dim=self.proj_dim,
            num_heads=self.num_heads,
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
        self,
        texts: Sequence[str],
        return_token_details: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        input_ids, attention_mask = self._tokenize_batch(texts)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        last_hidden_state = out.last_hidden_state.to(self.head_device)
        head_mask = attention_mask.to(self.head_device)

        if return_token_details:
            logits, token_weights = self.head(
                last_hidden_state, head_mask, return_token_weights=True
            )
            return logits, token_weights, input_ids, attention_mask

        return self.head(last_hidden_state, head_mask, return_token_weights=False)

    @torch.inference_mode()
    def predict_with_token_importance(
        self,
        texts: Iterable[object],
        progress_desc: str = "Inference",
        max_samples_to_detail: int = 3,
    ) -> tuple[np.ndarray, dict[str, object]]:
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        if not clean_texts:
            return np.empty(0, dtype=np.int64), {
                "sample_inspections": [],
                "global_top_tokens": [],
            }

        self.model.eval()
        self.head.eval()
        preds: list[np.ndarray] = []
        tracker = TokenImportanceTracker(
            tokenizer=self.tokenizer,
            label_names=self.label_names,
            prefix_len=self.prefix_len,
            max_samples_to_detail=max_samples_to_detail,
            large_dataset_min_occurrences=5,
        )

        batch_starts = range(0, len(clean_texts), self.batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(clean_texts) + self.batch_size - 1) // self.batch_size,
            desc=progress_desc,
            unit="batch",
        ):
            batch = clean_texts[start : start + self.batch_size]
            logits, token_w, inp_ids, att_mask = self.forward_batch(
                batch, return_token_details=True
            )
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.append(batch_preds)
            tracker.update_batch(
                batch_texts=batch,
                batch_preds=batch_preds,
                token_weights=token_w.float().cpu().numpy(),
                input_ids=inp_ids.cpu().numpy(),
                attention_mask=att_mask.cpu().numpy(),
                start_index=start,
            )

        return np.concatenate(preds), tracker.build_report(len(clean_texts))

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, dict[str, object]]:
        indices, report = self.predict_with_token_importance(
            texts, progress_desc=progress_desc
        )
        names = np.asarray([self.label_names[idx] for idx in indices], dtype=object)
        return names, report


def _restore_best_weights(
    pipeline: QwenAllTokenPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重載回記憶體中的 pipeline。"""
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_dir,
        head_filename="all_token_head.pt",
        trainable=None,
        missing_ok=True,
    )


def _train_pipeline(
    pipeline: QwenAllTokenPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_alltoken",
) -> tuple[float, dict[str, object]]:
    """聯合訓練 TaskLoRA 與 AllTokenAttentionClassifier，保留最佳驗證模型。"""
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
            logits = pipeline.forward_batch(batch_texts, return_token_details=False)
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

        val_preds, val_report = pipeline.predict_with_token_importance(
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
        print_token_importance_report(
            val_report,
            title=f"Epoch {epoch + 1} 驗證集全 Token 注意力重要性分析",
        )

        if score > best_score:
            prev_best = best_score
            best_score = score
            best_epoch = epoch + 1
            best_report = val_report
            save_model(pipeline, save_dir, token_report=best_report)
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
    pipeline: QwenAllTokenPipeline,
    save_dir: str | pathlib.Path,
    token_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、AllTokenAttentionClassifier 權重與 Token 重要性分析報告。"""
    metadata = {
        "label_names": pipeline.label_names,
        "base_model_path": pipeline.base_model_path,
        "hidden_size": pipeline.hidden_size,
        "num_classes": len(pipeline.label_names),
        "batch_size": pipeline.batch_size,
        "max_length": pipeline.max_length,
        "lora_rank": pipeline.lora_rank,
        "lora_scale": pipeline.lora_scale,
        "proj_dim": pipeline.proj_dim,
        "num_heads": pipeline.num_heads,
        "attn_dim": pipeline.attn_dim,
    }
    save_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        save_dir=save_dir,
        head_filename="all_token_head.pt",
        metadata=metadata,
        token_report=token_report,
    )
    print(f"Saved model and token importance report to {save_dir}")


def load_model(
    save_dir: str | pathlib.Path,
    base_model_path: str | None = None,
    trainable: bool = False,
) -> QwenAllTokenPipeline:
    """載入已保存的 TaskLoRA 與 AllTokenAttentionClassifier。"""
    save_path = pathlib.Path(save_dir)
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
    resolved_base = base_model_path or find_qwen_model_dir(metadata["base_model_path"])

    pipeline = QwenAllTokenPipeline(
        base_model_path=resolved_base,
        label_names=metadata["label_names"],
        batch_size=int(metadata["batch_size"]),
        max_length=int(metadata["max_length"]),
        lora_rank=int(metadata["lora_rank"]),
        lora_scale=float(metadata["lora_scale"]),
        proj_dim=int(metadata.get("proj_dim", PROJ_DIM)),
        num_heads=int(metadata.get("num_heads", NUM_HEADS)),
        attn_dim=int(metadata.get("attn_dim", ATTN_DIM)),
    )
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_path,
        head_filename="all_token_head.pt",
        trainable=trainable,
    )
    pipeline.model.eval()
    pipeline.head.eval()
    return pipeline


def Trainer(
    save_dir: str | pathlib.Path = "saved_qwen_alltoken",
    load_weights: bool = False,
) -> QwenAllTokenPipeline:
    """讀取資料、建立 Qwen + TaskLoRA + AllTokenAttentionClassifier 並執行訓練。"""
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
        pipeline = QwenAllTokenPipeline(
            base_model_path=model_dir,
            label_names=label_names,
        )

    lora_count, head_count = count_qwen_trainable_params(pipeline.loras, pipeline.head)
    print(
        f"輸入特徵形狀: (Batch, {pipeline.max_length}, {pipeline.hidden_size}) | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 AllTokenAttentionClassifier 參數: {head_count:,}"
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
    """執行完整訓練、分析每個 Token 的重要性，並對 test.csv 產生預測提交檔。"""
    save_dir = "saved_qwen_alltoken"
    pipeline = Trainer(save_dir, load_weights=False)

    clear_cuda_cache()
    test_df = load_test_data(find_data_dir(CANDIDATES))

    test_preds, test_token_report = pipeline.predict(
        test_df["text"].fillna(""), progress_desc="Test Inference"
    )
    print_token_importance_report(
        test_token_report,
        title="測試集 (test.csv) 全 Token 注意力重要性分析",
    )
    return save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    submission = main()
    save_kaggle_submission(submission)
