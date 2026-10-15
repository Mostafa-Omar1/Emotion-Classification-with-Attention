"""Plot BiLSTM+Attention weights per input token to explain predictions.

Produces:
  reports/figures/attention_examples.png   – heatmap, one row per sentence
  reports/figures/attention_per_class.png  – a correctly classified test example for every emotion
  reports/figures/attention_top_words.png  – words receiving the most attention per predicted class

Usage:  python -m src.visualize_attention
"""
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .data import load_goemotions
from .predict import RNNPredictor

CUSTOM = [
    "I can't believe you did this, I'm so angry right now",
    "Thank you so much, this made my day!",
    "I miss her so much, it hurts every single day",
    "I'm really scared about the surgery tomorrow",
    "Wait, what? I had no idea that was possible!",
    "That is absolutely disgusting, gross",
]


def heatmap(predictor, texts, path, title, true_labels=None):
    probs, toks, attn = predictor.predict(texts)
    width = max(len(t) for t in toks)
    fig, axes = plt.subplots(len(texts), 1, figsize=(max(10, 0.75 * width), 1.15 * len(texts)))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        row = np.full((1, width), np.nan)
        row[0, : len(attn[i])] = attn[i]
        ax.imshow(row, cmap="Reds", aspect="auto", vmin=0, vmax=max(attn[i].max(), 1e-6))
        for j, tok in enumerate(toks[i]):
            ax.text(j, 0, tok, ha="center", va="center", fontsize=9,
                    color="white" if attn[i][j] > 0.6 * attn[i].max() else "black")
        pred = C.EMOTIONS[probs[i].argmax()]
        label = f"pred: {pred} ({probs[i].max():.0%})"
        if true_labels is not None:
            label = f"true: {true_labels[i]} | " + label
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(label, fontsize=9, loc="left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    pred = RNNPredictor("bilstm_attention")
    heatmap(pred, CUSTOM, C.FIG_DIR / "attention_examples.png", "BiLSTM + Attention — attention weights")

    test = load_goemotions()["test"]
    probs, toks, attn = pred.predict(test["text"].tolist())
    y_pred = probs.argmax(1)
    y_true = test["label"].values

    # one confident, correctly classified example (4-20 tokens) per class
    picks = []
    for k, emo in enumerate(C.EMOTIONS):
        idx = [i for i in range(len(test)) if y_true[i] == k and y_pred[i] == k and 4 <= len(toks[i]) <= 20]
        if idx:
            picks.append(max(idx, key=lambda i: probs[i, k]))
    heatmap(pred, test["text"].iloc[picks].tolist(), C.FIG_DIR / "attention_per_class.png",
            "Attention on correctly classified test examples",
            true_labels=[C.EMOTIONS[y_true[i]] for i in picks])

    # aggregate: average attention a word gets when the model predicts each class
    agg = defaultdict(lambda: defaultdict(list))
    for i in range(len(test)):
        for tok, a in zip(toks[i], attn[i]):
            agg[y_pred[i]][tok].append(a)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for k, ax in enumerate(axes.flat):
        scores = {w: np.sum(v) for w, v in agg[k].items() if len(v) >= 5}
        top = sorted(scores.items(), key=lambda x: -x[1])[:12][::-1]
        ax.barh([w for w, _ in top], [s for _, s in top], color="#C44E52")
        ax.set_title(f"{C.EMOTIONS[k]}")
        ax.set_xlabel("total attention mass (test set)")
    fig.suptitle("Words receiving most attention per predicted emotion", fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "attention_top_words.png", dpi=150)
    plt.close(fig)
    print("saved attention figures to", C.FIG_DIR)


if __name__ == "__main__":
    main()
