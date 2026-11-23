"""訓練節奏特徵 CNN，根據文字內容辨識回答所屬的語言模型。"""

from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from data_utils import (
    CANDIDATES,
    find_data_dir,
    group_split,
    load_test_data,
    load_train_data,
    save_submission,
    seed_everything,
)
from metrics_utils import balanced_accuracy


MODEL_PATH = (
    "/kaggle/input/models/levantaokkz/bge-large-en-v1.5/transformers/"
    "default/1/bge-large-en-v1.5"
)
# 訓練設定：固定亂數種子、保留 20% prompt 做驗證，並訓練一個 epoch。
SEED = 42
VALIDATION_FRACTION = 0.20
BATCH_SIZE = 32
EPOCHS = 1


class RhythmCNN(nn.Module):
    """將 token 節奏序列分類為 18 種語言模型之一。"""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        """建立兩層一維卷積與最終分類層。"""
        super().__init__()
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding="same")
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, rhythm: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """卷積處理 token 序列，遮罩補齊位置後輸出各類別分數。"""
        # 輸入 rhythm 為 (批次, 序列, 向量維度)，卷積需要通道維在中間。
        x = rhythm.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.masked_fill(~attention_mask.unsqueeze(1).bool(), torch.finfo(x.dtype).min)
        # 取有效 token 位置的最大特徵，作為整句表示並進行分類。
        pooled = x.max(dim=-1).values
        return self.classifier(pooled)


class ModelPipeline:
    """封裝凍結的句子編碼器與 CNN，提供批次預測介面。"""

    def __init__(
        self,
        encoder: SentenceTransformer,
        classifier: RhythmCNN,
        label_names: Sequence[str],
        device: torch.device,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """保存編碼器、分類器、標籤順序與執行裝置。"""
        self.encoder = encoder
        self.classifier = classifier
        self.label_names = list(label_names)
        self.device = device
        self.batch_size = batch_size

    def _encode_batch(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """取得 token/text embeddings，轉成節奏向量並回傳有效位置遮罩。"""
        features = self.encoder.preprocess(list(texts))
        features = {
            key: value for key, value in features.items() if isinstance(value, torch.Tensor)
        }
        # 補齊到編碼器最大長度，超長輸入則已由 tokenizer 截斷。
        max_length = int(self.encoder.max_seq_length)
        current_length = int(features["input_ids"].shape[1])
        if current_length < max_length:
            padding = max_length - current_length
            for key, value in features.items():
                pad_value = self.encoder.tokenizer.pad_token_id if key == "input_ids" else 0
                features[key] = F.pad(value, (0, padding), value=int(pad_value or 0))
        features = {key: value.to(self.device) for key, value in features.items()}
        # 編碼器凍結且不需梯度；使用 no_grad，讓輸出可安全供 CNN 反向傳播。
        with torch.no_grad():
            output = self.encoder(features)
        token_embeddings = output["token_embeddings"]
        text_embedding = output["sentence_embedding"]
        rhythm = F.normalize(token_embeddings, p=2, dim=-1) - text_embedding.unsqueeze(1)
        mask = features["attention_mask"].bool()
        # 將補齊位置設為零，避免其 embedding 差值干擾卷積特徵。
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


def _train_classifier(
    pipeline: ModelPipeline,
    texts: Sequence[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> float:
    """以 CrossEntropyLoss 與 Adam 訓練 CNN，並回傳驗證 balanced accuracy。"""
    classifier = pipeline.classifier
    optimizer = torch.optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)

    score = 0.0
    for epoch in range(EPOCHS):
        classifier.train()
        order = train_indices.copy()
        rng.shuffle(order)
        total_loss = 0.0
        seen = 0
        batch_starts = range(0, len(order), BATCH_SIZE)
        progress = tqdm(
            batch_starts,
            total=(len(order) + BATCH_SIZE - 1) // BATCH_SIZE,
            desc=f"Training epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
        )
        for start in progress:
            indices = order[start : start + BATCH_SIZE]
            batch_texts = [texts[i] for i in indices]
            rhythm, mask = pipeline._encode_batch(batch_texts)
            targets = torch.as_tensor(labels[indices], dtype=torch.long, device=pipeline.device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(rhythm, mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")

        validation_predictions = pipeline.predict_indices(
            (texts[i] for i in validation_indices.tolist()),
            progress_desc=f"Validation epoch {epoch + 1}/{EPOCHS}",
        )
        score = balanced_accuracy(
            labels[validation_indices], validation_predictions, len(pipeline.label_names)
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss / max(seen, 1):.4f} "
            f"- validation balanced_accuracy: {score:.6f}"
        )
    return score


def main() -> None:
    """載入資料、建立並訓練模型，最後產生 Kaggle 提交檔。"""
    seed_everything(SEED)
    data_dir = find_data_dir(CANDIDATES)
    train_df, texts, labels, label_names, groups = load_train_data(data_dir)
    test_df = load_test_data(data_dir)

    # 以 prompt 分組切分，並額外確認兩集合沒有 prompt 洩漏。
    train_indices, validation_indices = group_split(
        groups, labels.tolist(), VALIDATION_FRACTION, SEED
    )
    train_groups = set(train_df.iloc[train_indices]["prompt_id"].tolist())
    validation_groups = set(train_df.iloc[validation_indices]["prompt_id"].tolist())
    if train_groups & validation_groups:
        raise AssertionError("prompt_id leakage between training and validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device}; train rows: {len(train_indices)}; "
        f"validation rows: {len(validation_indices)}"
    )
    encoder = SentenceTransformer(MODEL_PATH, device=str(device))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    embedding_dim = encoder.get_embedding_dimension()
    if embedding_dim is None:
        raise ValueError("Could not determine the sentence embedding dimension")

    classifier = RhythmCNN(embedding_dim, len(label_names)).to(device)
    model_pipeline = ModelPipeline(encoder, classifier, label_names, device)
    _train_classifier(model_pipeline, texts, labels, train_indices, validation_indices)

    test_preds = model_pipeline.predict(test_df["text"].fillna(""))
    save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    main()
