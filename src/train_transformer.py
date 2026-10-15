"""Fine-tune DistilBERT on the 6-class GoEmotions data with the HuggingFace Trainer API.

Usage:  python -m src.train_transformer
"""
import math
import time

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from . import config as C
from .data import class_weights, load_goemotions
from .evaluate import report_and_save

OUT_DIR = C.MODELS_DIR / "distilbert"


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (same weights as the RNN models)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def main():
    set_seed(C.SEED)
    hp = C.TRANSFORMER_HPARAMS
    splits = load_goemotions()
    tokenizer = AutoTokenizer.from_pretrained(C.TRANSFORMER_NAME)

    ds = DatasetDict({s: Dataset.from_pandas(df[["text", "label"]]) for s, df in splits.items()})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=hp["max_length"]),
                batched=True, remove_columns=["text"]).rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        C.TRANSFORMER_NAME, num_labels=len(C.EMOTIONS), id2label=C.ID2LABEL, label2id=C.LABEL2ID)

    args = TrainingArguments(
        output_dir=str(C.MODELS_DIR / "distilbert_ckpt"),
        learning_rate=hp["lr"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"] * 2,
        num_train_epochs=hp["epochs"],
        weight_decay=hp["weight_decay"],
        warmup_steps=int(hp["warmup_ratio"] * hp["epochs"] * math.ceil(len(ds["train"]) / hp["batch_size"])),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=C.SEED,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics_fn, class_weights=class_weights(splits["train"]["label"].values),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    pred = trainer.predict(ds["test"])
    y_pred = pred.predictions.argmax(-1)
    history = [r for r in trainer.state.log_history if "eval_macro_f1" in r]
    report_and_save("distilbert", pred.label_ids, y_pred, extra={
        "best_val_macro_f1": trainer.state.best_metric, "train_seconds": secs,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history})


if __name__ == "__main__":
    main()
