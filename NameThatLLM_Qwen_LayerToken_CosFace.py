"""
以 TaskLoRA + Sub-center CosFace 對比分類頭微調 Qwen-3-Embedding (NameThatLLM_Qwen_LayerToken_CosFace.py)：
1. 承襲 NameThatLLM_Qwen_LayerToken.py 的前 24 層截斷 (KEEP_LAYERS=24) 與階層式雙重注意力架構。
2. 引入【Sub-center CosFace 餘弦邊界對比分類頭 (SubCenterCosFaceHead)】與 Margin Warmup + Cosine LR。
3. 驗證階段同時計算 Balanced Accuracy、原始 LogLoss 與最佳 Temperature Scaling (T*) 校準後 LogLoss，
   並保留完整的【8 層樓】與【關鍵 Token】雙重分析報告。
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
from metrics_utils import balanced_accuracy, fit_temperature
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

LAYER_PROJ_DIM = 32
NUM_HEADS = 4
ATTN_DIM = 128
DROPOUT = 0.10

LAYER_DROP_PROB = 0.10
LAYER_TEMPERATURE = 2.0

TOKEN_DROP_PROB = 0.15
TOKEN_TEMPERATURE = 1.5

# 溫和版 CosFace + Margin Warmup + Cosine LR 超參數。
COSFACE_SCALE = 12.0
COSFACE_MARGIN = 0.08
COSFACE_SUB_CENTERS = 1
COSFACE_WARMUP_EPOCHS = 1
WARMUP_RATIO = 0.10
LABEL_SMOOTHING = 0.05


class SubCenterCosFaceHead(nn.Module):
    """Sub-center CosFace 餘弦邊界對比分類頭。"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = COSFACE_SCALE,
        margin: float = COSFACE_MARGIN,
        num_sub_centers: int = COSFACE_SUB_CENTERS,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.num_sub_centers = max(1, int(num_sub_centers))

        self.weight = nn.Parameter(
            torch.empty(
                self.num_classes * self.num_sub_centers,
                self.in_features,
                dtype=torch.float32,
            )
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_norm = F.normalize(x.float(), p=2, dim=-1)
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1)
        cosine_all = F.linear(x_norm, w_norm)
        cosine_all = cosine_all.view(-1, self.num_classes, self.num_sub_centers)
        cosine = cosine_all.max(dim=-1).values.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        if self.training and targets is not None:
            one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).to(
                dtype=cosine.dtype
            )
            cosine = cosine - one_hot * self.margin

        return cosine * self.scale


class HierarchicalLayerTokenClassifier(nn.Module):
    """階層式「8 層樓 Layer Attention + 全序列 512 Token Attention + Sub-center CosFace」分類器。"""

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
        cosface_scale: float = COSFACE_SCALE,
        cosface_margin: float = COSFACE_MARGIN,
        cosface_sub_centers: int = COSFACE_SUB_CENTERS,
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
        self.cosface_scale = float(cosface_scale)
        self.cosface_margin = float(cosface_margin)
        self.cosface_sub_centers = max(1, int(cosface_sub_centers))

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
        self.classifier = SubCenterCosFaceHead(
            in_features=self.proj_dim * 2,
            num_classes=num_classes,
            scale=self.cosface_scale,
            margin=self.cosface_margin,
            num_sub_centers=self.cosface_sub_centers,
        )

    def forward(
        self,
        layer_seq_features: list[torch.Tensor],
        attention_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
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
        logits = self.classifier(self.dropout(combined), targets=targets)

        if return_details:
            return logits, layer_weights.squeeze(-1), token_weights
        return logits


class QwenLayerTokenPipeline:
    """封裝 Qwen-3-Embedding (前 24 層截斷版) + TaskLoRA + 階層式雙重注意力 + Sub-center CosFace 流程。"""

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
        cosface_scale: float = COSFACE_SCALE,
        cosface_margin: float = COSFACE_MARGIN,
        cosface_sub_centers: int = COSFACE_SUB_CENTERS,
        optimal_temperature: float = 1.0,
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
        self.cosface_scale = float(cosface_scale)
        self.cosface_margin = float(cosface_margin)
        self.cosface_sub_centers = max(1, int(cosface_sub_centers))
        self.optimal_temperature = float(optimal_temperature)

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
            cosface_scale=self.cosface_scale,
            cosface_margin=self.cosface_margin,
            cosface_sub_centers=self.cosface_sub_centers,
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
        targets: torch.Tensor | None = None,
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
        head_targets = (
            targets.to(self.head_device) if targets is not None else None
        )
        del out

        if return_details:
            logits, layer_weights, token_weights = self.head(
                layer_seq_features,
                head_mask,
                targets=head_targets,
                return_details=True,
            )
            return logits, layer_weights, token_weights, input_ids, attention_mask

        return self.head(
            layer_seq_features,
            head_mask,
            targets=head_targets,
            return_details=False,
        )

    @torch.inference_mode()
    def predict_with_dual_importance(
        self,
        texts: Iterable[object],
        progress_desc: str = "Inference",
        max_samples_to_detail: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray]:
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
        tracker = TokenImportanceTracker(
            tokenizer=self.tokenizer,
            label_names=self.label_names,
            prefix_len=self.prefix_len,
            max_samples_to_detail=max_samples_to_detail,
            large_dataset_min_occurrences=30,
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
                batch, targets=None, return_details=True
            )
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.append(batch_preds)
            all_logits.append(logits.float().cpu().numpy())
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
            np.concatenate(all_logits, axis=0),
        )

    def predict(
        self, texts: Iterable[object], progress_desc: str = "Inference"
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        indices, layer_weights, token_report, _ = self.predict_with_dual_importance(
            texts, progress_desc=progress_desc
        )
        names = np.asarray([self.label_names[idx] for idx in indices], dtype=object)
        return names, layer_weights, token_report


def _restore_best_weights(
    pipeline: QwenLayerTokenPipeline, save_dir: str | pathlib.Path
) -> None:
    """將已保存在磁碟上的最佳 Epoch 權重載回記憶體中的 pipeline。"""
    loaded = load_task_lora_and_head(
        loras=pipeline.loras,
        head=pipeline.head,
        head_device=pipeline.head_device,
        save_dir=save_dir,
        head_filename="layer_token_head.pt",
        trainable=None,
        missing_ok=True,
    )
    if not loaded:
        return
    meta_path = pathlib.Path(save_dir) / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        pipeline.optimal_temperature = float(metadata.get("optimal_temperature", 1.0))


def _train_pipeline(
    pipeline: QwenLayerTokenPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_cosface",
) -> tuple[float, dict[str, object], dict[str, object]]:
    """聯合訓練 TaskLoRA 與 HierarchicalLayerTokenClassifier (Sub-center CosFace)。"""
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

    num_micro_batches = (len(train_indices) + pipeline.batch_size - 1) // pipeline.batch_size
    steps_per_epoch = max(
        1,
        (num_micro_batches + GRADIENT_ACCUMULATION_STEPS - 1)
        // GRADIENT_ACCUMULATION_STEPS,
    )
    total_steps = max(1, steps_per_epoch * EPOCHS)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=WARMUP_RATIO
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    rng = np.random.default_rng(SEED)

    best_score = -1.0
    best_epoch = 0
    best_layer_report: dict[str, object] = {}
    best_token_report: dict[str, object] = {}

    for epoch in range(EPOCHS):
        active_margin = (
            0.0 if epoch < COSFACE_WARMUP_EPOCHS else pipeline.cosface_margin
        )
        pipeline.head.classifier.margin = active_margin
        print(
            f"🎯 Epoch {epoch + 1}/{EPOCHS} 啟動設定: "
            f"CosFace active_margin = {active_margin:.3f} (scale={pipeline.cosface_scale}, K={pipeline.cosface_sub_centers}) | "
            f"current LoRA lr = {optimizer.param_groups[0]['lr']:.2e}"
        )

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
            targets = torch.as_tensor(
                labels[indices], dtype=torch.long, device=pipeline.head_device
            )
            logits = pipeline.forward_batch(
                batch_texts, targets=targets, return_details=False
            )
            loss = criterion(logits, targets)
            (loss * (len(indices) / group_examples)).backward()

            if (batch_number + 1) == group_end:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
            progress.set_postfix(
                loss=f"{total_loss / seen:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )
            del logits, targets, loss

        clear_cuda_cache()

        val_preds, val_layer_weights, val_token_report, val_logits = (
            pipeline.predict_with_dual_importance(
                (texts[i] for i in validation_indices.tolist()),
                progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
            )
        )
        val_true = labels[validation_indices]
        score = balanced_accuracy(val_true, val_preds, len(pipeline.label_names))
        raw_ll, cal_ll, opt_temp = fit_temperature(val_logits, val_true)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / max(seen, 1):.4f} "
            f"- validation balanced_accuracy: {score:.6f} "
            f"- val_log_loss: {raw_ll:.4f} (calibrated @ T={opt_temp:.3f}: {cal_ll:.4f})"
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
            pipeline.optimal_temperature = opt_temp
            save_model(
                pipeline,
                save_dir,
                layer_report=best_layer_report,
                token_report=best_token_report,
            )
            if prev_best < 0:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: 初次建立最佳模型 "
                    f"(balanced_accuracy = {best_score:.6f}, T* = {opt_temp:.3f})，已儲存至 {save_dir}！"
                )
            else:
                print(
                    f"✅ Epoch {epoch + 1}/{EPOCHS}: validation balanced_accuracy 創新高 "
                    f"({prev_best:.6f} -> {best_score:.6f}, T* = {opt_temp:.3f})，已更新最佳模型至 {save_dir}！"
                )
        else:
            print(
                f"⚠️ Epoch {epoch + 1}/{EPOCHS}: validation balanced_accuracy ({score:.6f}) "
                f"未超越最佳紀錄 ({best_score:.6f} @ Epoch {best_epoch})，不儲存新模型，繼續訓練..."
            )

    pipeline.head.classifier.margin = pipeline.cosface_margin

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
    """保存 TaskLoRA 權重、HierarchicalLayerTokenClassifier (Sub-center CosFace) 權重以及分析報告。"""
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
        "cosface_scale": pipeline.cosface_scale,
        "cosface_margin": pipeline.cosface_margin,
        "cosface_sub_centers": pipeline.cosface_sub_centers,
        "optimal_temperature": pipeline.optimal_temperature,
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
    """載入已保存的 TaskLoRA 與 HierarchicalLayerTokenClassifier (Sub-center CosFace)。"""
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
        cosface_scale=float(metadata.get("cosface_scale", COSFACE_SCALE)),
        cosface_margin=float(metadata.get("cosface_margin", COSFACE_MARGIN)),
        cosface_sub_centers=int(
            metadata.get("cosface_sub_centers", COSFACE_SUB_CENTERS)
        ),
        optimal_temperature=float(metadata.get("optimal_temperature", 1.0)),
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
    save_dir: str | pathlib.Path = "saved_qwen_layertoken_cosface",
    load_weights: bool = False,
) -> QwenLayerTokenPipeline:
    """讀取資料、建立 Qwen (前 24 層) + TaskLoRA + HierarchicalLayerTokenClassifier (CosFace) 並執行訓練。"""
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
        f"Sub-center CosFace (s={pipeline.cosface_scale}, m={pipeline.cosface_margin}, K={pipeline.cosface_sub_centers}) | "
        f"可訓練 TaskLoRA 參數: {lora_count:,} | "
        f"可訓練 Head 參數: {head_count:,}"
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
    save_dir = "saved_qwen_layertoken_cosface"
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
