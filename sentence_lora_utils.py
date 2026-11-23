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
