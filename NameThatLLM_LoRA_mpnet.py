"""以 LoRA 微調 all-mpnet-base-v2，並用 Transformer 分類文字的節奏特徵。"""

from __future__ import annotations

import pathlib

import pandas as pd
import torch

from data_utils import (
    CANDIDATES,
    find_data_dir,
    group_split,
    load_test_data,
    load_train_data,
    save_kaggle_submission,
    save_submission,
    seed_everything,
)
from sentence_lora_utils import (
    RhythmTransformerClassifier,
    SentenceLoRAPipeline as ModelPipeline,
    attach_peft_lora,
    load_sentence_encoder,
    load_sentence_lora_model,
    log_peft_lora_params,
    save_sentence_lora_model,
    train_sentence_lora_classifier,
)


MODEL_PATH = (
    "/kaggle/input/models/anrenk/all-mpnet-base-v2/transformers/"
    "all-mpnet-base-v2/1"
)
# 訓練設定：固定亂數種子、保留 20% prompt 驗證。
SEED = 42
VALIDATION_FRACTION = 0.20
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 10
LEARNING_RATE = 2e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# MPNetSelfAttention 的注意力線性層命名為 q, k, v, o（不同於 BERT/BGE 的 query, key, value）。
LORA_TARGET_MODULES = ["q", "v"]
# all-mpnet-base-v2 預訓練最大序列長度為 384（底層 MPNet position embeddings 上限為 512）。
MAX_SEQ_LENGTH = 384


def save_model(pipeline: ModelPipeline, save_dir: str | pathlib.Path) -> None:
    """保存 LoRA 權重、分類器權重及重建 pipeline 所需設定。"""
    save_sentence_lora_model(
        pipeline=pipeline,
        save_dir=save_dir,
        base_model_path=MODEL_PATH,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
    )


def load_model(
    save_dir: str | pathlib.Path,
    device: torch.device | str | None = None,
    base_model_path: str | pathlib.Path | None = None,
    trainable: bool = False,
) -> ModelPipeline:
    """載入已保存的 LoRA pipeline；trainable=True 時保留 LoRA 與分類器可訓練狀態以接續訓練。"""
    return load_sentence_lora_model(
        save_dir=save_dir,
        default_lora_rank=LORA_RANK,
        default_lora_alpha=LORA_ALPHA,
        default_target_modules=LORA_TARGET_MODULES,
        default_lora_dropout=LORA_DROPOUT,
        device=device,
        base_model_path=base_model_path,
        trainable=trainable,
        max_seq_length=MAX_SEQ_LENGTH,
        model_label="MPNet",
    )


def Trainer(
    save_dir: str | pathlib.Path = "saved_model",
    load_weights: bool = False,
) -> ModelPipeline:
    """載入訓練資料、建立或讀取 LoRA 與分類器權重進行訓練，最後保存模型。"""
    seed_everything(SEED)
    data_dir = find_data_dir(CANDIDATES)
    train_df, texts, labels, label_names, groups = load_train_data(data_dir)

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

    if load_weights:
        print(f"Loading existing weights from {save_dir} to continue training...")
        model_pipeline = load_model(save_dir, device=device, trainable=True)
        if model_pipeline.label_names != label_names:
            raise ValueError("Loaded model label_names do not match training data labels")
        encoder = model_pipeline.encoder
    else:
        encoder = load_sentence_encoder(
            MODEL_PATH, device, max_seq_length=MAX_SEQ_LENGTH
        )
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        transformer_module = encoder[0]
        if not hasattr(transformer_module, "auto_model"):
            raise TypeError("SentenceTransformer first module does not expose auto_model")
        transformer_module.auto_model = attach_peft_lora(
            transformer_module.auto_model,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            model_label="MPNet",
        )
        embedding_dim = encoder.get_embedding_dimension()
        if embedding_dim is None:
            raise ValueError("Could not determine the sentence embedding dimension")
        classifier = RhythmTransformerClassifier(embedding_dim, len(label_names)).to(device)
        model_pipeline = ModelPipeline(
            encoder, classifier, label_names, device, batch_size=MICRO_BATCH_SIZE
        )

    log_peft_lora_params(encoder)
    train_sentence_lora_classifier(
        pipeline=model_pipeline,
        texts=texts,
        labels=labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epochs=EPOCHS,
        micro_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )
    save_model(model_pipeline, save_dir)
    return model_pipeline


def main() -> pd.DataFrame:
    """執行模型訓練、載入保存的模型並產生 Kaggle 提交檔。"""
    Trainer("saved_model", load_weights=False)

    data_dir = find_data_dir(CANDIDATES)
    test_df = load_test_data(data_dir)

    model_pipeline = load_model("saved_model")
    test_preds = model_pipeline.predict(test_df["text"].fillna(""))
    return save_submission(test_df["id"], test_preds, pathlib.Path("submission.parquet"))


if __name__ == "__main__":
    submission = main()
    save_kaggle_submission(submission)
