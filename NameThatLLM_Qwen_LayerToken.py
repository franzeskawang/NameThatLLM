"""
以 TaskLoRA 微調 Qwen-3-Embedding (雙重注意力結合版 NameThatLLM_Qwen_LayerToken.py)：
1. 直接截斷並移除第 25～36 層（只載入並運算前 24 層 KEEP_LAYERS=24），省下 1/3 顯存與訓練時間。
2. 結合「8 層樓 Layer Attention」與「全序列 512 Token Attention」的階層式雙重注意力架構 (HierarchicalLayerTokenClassifier)。
3. 每個 Epoch 驗證與最終測試推論時，同時印出【8 層樓 Attention 權重統計表】與【關鍵 Token 排行榜】雙重分析報告。
"""

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
from report_utils import (
    TokenImportanceTracker,
    print_layer_importance_report,
    print_token_importance_report,
)


# 只保留前 24 層，每 3 層設置一個監聽哨（共 8 個目標樓層）。
KEEP_LAYERS = 24
DEFAULT_DEPTHS = (3, 6, 9, 12, 15, 18, 21, 24)

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

# 每一層樓各自降維至 LAYER_PROJ_DIM (32 維)，8 層樓以 torch.cat 拼接後總維度為 256 維。
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


class HierarchicalLayerTokenClassifier(nn.Module):
    """階層式「8 層樓 Layer Attention + 全序列 512 Token Attention」雙重注意力分類器。"""

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
        self.classifier = nn.Linear(
            self.proj_dim * 2, num_classes, dtype=torch.float32
        )

    def forward(
        self,
        layer_seq_features: list[torch.Tensor],
        attention_mask: torch.Tensor,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        logits = self.classifier(self.dropout(combined))

        if return_details:
            return logits, layer_weights.squeeze(-1), token_weights
        return logits


class QwenLayerTokenPipeline:
    """封裝 Qwen-3-Embedding (前 24 層截斷版) + TaskLoRA + 階層式雙重注意力分類器的訓練與推論流程。"""

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
        return_details: bool = False,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
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
            logits, layer_weights, token_weights = self.head(
                layer_seq_features, head_mask, return_details=True
            )
            return logits, layer_weights, token_weights, input_ids, attention_mask

        return self.head(layer_seq_features, head_mask, return_details=False)

    @torch.inference_mode()
    def predict_with_dual_importance(
        self,
        texts: Iterable[object],
        progress_desc: str = "Inference",
        max_samples_to_detail: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        clean_texts = ["" if pd.isna(t) else str(t) for t in texts]
        if not clean_texts:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, len(self.depths)), dtype=np.float32),
                {"sample_inspections": [], "global_top_tokens": []},
            )

        self.model.eval()
        self.head.eval()
        preds: list[np.ndarray] = []
        all_layer_weights: list[np.ndarray] = []
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
            logits, layer_w, token_w, inp_ids, att_mask = self.forward_batch(
                batch, return_details=True
            )
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.append(batch_preds)
            all_layer_weights.append(layer_w.float().cpu().numpy())
            tracker.update_batch(
                batch_texts=batch,
                batch_preds=batch_preds,
                token_weights=token_w.float().cpu().numpy(),
                input_ids=inp_ids.cpu().numpy(),
                attention_mask=att_mask.cpu().numpy(),
                start_index=start,
            )

        return (
            np.concatenate(preds),
            np.concatenate(all_layer_weights, axis=0),
            tracker.build_report(len(clean_texts)),
        )

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        indices, layer_weights, token_report = self.predict_with_dual_importance(
            texts, progress_desc=progress_desc
        )
        names = np.asarray([self.label_names[idx] for idx in indices], dtype=object)
        return names, layer_weights, token_report


def _restore_best_weights(
    pipeline: QwenLayerTokenPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重載回記憶體中的 pipeline。"""
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_dir,
        head_filename="layer_token_head.pt",
        trainable=None,
        missing_ok=True,
    )


def _train_pipeline(
    pipeline: QwenLayerTokenPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_v2",
) -> tuple[float, dict[str, object], dict[str, object]]:
    """聯合訓練 TaskLoRA 與 HierarchicalLayerTokenClassifier，保留最佳驗證模型。"""
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
    best_layer_report: dict[str, object] = {}
    best_token_report: dict[str, object] = {}

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
            logits = pipeline.forward_batch(batch_texts, return_details=False)
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

        val_preds, val_layer_weights, val_token_report = (
            pipeline.predict_with_dual_importance(
                (texts[i] for i in validation_indices.tolist()),
                progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
            )
        )
        score = balanced_accuracy(
            labels[validation_indices], val_preds, len(pipeline.label_names)
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / max(seen, 1):.4f} "
            f"- validation balanced_accuracy: {score:.6f}"
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

    return best_score, best_layer_report, best_token_report


def save_model(
    pipeline: QwenLayerTokenPipeline,
    save_dir: str | pathlib.Path,
    layer_report: dict[str, object] | None = None,
    token_report: dict[str, object] | None = None,
) -> None:
    """保存 TaskLoRA 權重、HierarchicalLayerTokenClassifier 權重以及樓層與 Token 雙重分析報告。"""
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
    }
    save_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        save_dir=save_dir,
        head_filename="layer_token_head.pt",
        metadata=metadata,
        layer_report=layer_report,
        token_report=token_report,
    )
    print(f"Saved model, layer importance, and token importance reports to {save_dir}")


def load_model(
    save_dir: str | pathlib.Path,
    base_model_path: str | None = None,
    trainable: bool = False,
) -> QwenLayerTokenPipeline:
    """載入已保存的 TaskLoRA 與 HierarchicalLayerTokenClassifier。"""
    save_path = pathlib.Path(save_dir)
    metadata = json.loads((save_path / "metadata.json").read_text(encoding="utf-8"))
    resolved_base = base_model_path or find_qwen_model_dir(metadata["base_model_path"])

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
    )
    load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_path,
        head_filename="layer_token_head.pt",
        trainable=trainable,
    )
    pipeline.model.eval()
    pipeline.head.eval()
    return pipeline


def Trainer(
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_v2",
    load_weights: bool = False,
) -> QwenLayerTokenPipeline:
    """讀取資料、建立 Qwen (前 24 層) + TaskLoRA + HierarchicalLayerTokenClassifier 並執行訓練。"""
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
        pipeline = QwenLayerTokenPipeline(
            base_model_path=model_dir,
            label_names=label_names,
        )

    lora_count, head_count = count_qwen_trainable_params(pipeline.loras, pipeline.head)
    print(
        f"監聽的 8 個樓層 (DEPTHS): {pipeline.depths} (各層獨立投影 {pipeline.layer_proj_dim} 維 -> torch.cat 拼接 {pipeline.head.proj_dim} 維) | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 HierarchicalLayerTokenClassifier 參數: {head_count:,}"
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
    """執行完整訓練、分析 8 層樓與每個 Token 的雙重重要性，並對 test.csv 產生預測提交檔。"""
    save_dir = "saved_qwen_layertoken_v2"
    pipeline = Trainer(save_dir, load_weights=False)

    clear_cuda_cache()
    test_df = load_test_data(find_data_dir(CANDIDATES))

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
    return save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    submission = main()
    save_kaggle_submission(submission)
