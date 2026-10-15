"""Train and evaluate the LSTM, GRU and BiLSTM+Attention models.

Usage:  python -m src.train_rnn [--models lstm gru bilstm_attention]
"""
import argparse
import json
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from . import config as C
from .data import EmotionDataset, Vocab, class_weights, load_goemotions, tokenize
from .embeddings import build_embedding_matrix
from .evaluate import report_and_save
from .models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for ids, lengths, y in loader:
        logits, _ = model(ids.to(DEVICE), lengths.to(DEVICE))
        preds.append(logits.argmax(-1).cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def train_one(name, emb_matrix, loaders, weights, hp):
    set_seed()
    model = build_model(name, emb_matrix, len(C.EMOTIONS), **hp).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n##### {name}  ({n_params:,} trainable params) on {DEVICE}")

    history, best_f1, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for ids, lengths, y in loaders["train"]:
            ids, lengths, y = ids.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(ids, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, loaders["validation"])
        val_f1 = f1_score(yv, pv, average="macro")
        val_acc = (yv == pv).mean()
        sched.step(val_f1)
        history.append({"epoch": epoch, "train_loss": total / n, "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {epoch:2d}  loss {total / n:.4f}  val_acc {val_acc:.4f}  val_macro_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                print("early stopping")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    torch.save({"name": name, "state_dict": best_state, "hparams": hp,
                "embedding_shape": list(emb_matrix.shape)}, C.MODELS_DIR / f"{name}.pt")
    return model, history, best_f1, train_time, n_params


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], marker="o", label=name)
        axes[1].plot(ep, [r["val_macro_f1"] for r in h], marker="o", label=name)
    axes[0].set(title="Training loss", xlabel="epoch")
    axes[1].set(title="Validation macro-F1", xlabel="epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "rnn_training_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "gru", "bilstm_attention"])
    args = ap.parse_args()

    splits = load_goemotions()
    vocab = Vocab.build([tokenize(t) for t in splits["train"]["text"]])
    vocab.save(C.MODELS_DIR / "vocab.txt")
    emb_matrix, coverage = build_embedding_matrix(vocab)
    print(f"vocab size {len(vocab):,}  GloVe coverage {coverage:.1%}")

    datasets = {s: EmotionDataset(df["text"].tolist(), df["label"].values, vocab) for s, df in splits.items()}
    loaders = {s: DataLoader(d, batch_size=C.RNN_HPARAMS["batch_size"], shuffle=(s == "train"))
               for s, d in datasets.items()}
    weights = class_weights(splits["train"]["label"].values)

    histories = {}
    for name in args.models:
        model, hist, best_val, secs, n_params = train_one(name, emb_matrix, loaders, weights, C.RNN_HPARAMS)
        histories[name] = hist
        y, p = predict(model, loaders["test"])
        report_and_save(name, y, p, extra={"best_val_macro_f1": best_val, "train_seconds": secs,
                                           "trainable_params": n_params, "history": hist})
    plot_history(histories)
    (C.METRICS_DIR / "glove_coverage.json").write_text(json.dumps({"vocab": len(vocab), "coverage": coverage}))


if __name__ == "__main__":
    main()
