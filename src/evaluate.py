"""Shared evaluation: accuracy, macro-F1, per-class precision/recall, confusion matrix."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import config as C


def compute_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=list(range(len(C.EMOTIONS))),
                                   target_names=C.EMOTIONS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "per_class": {e: {k: report[e][k] for k in ("precision", "recall", "f1-score", "support")}
                      for e in C.EMOTIONS},
    }


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(C.EMOTIONS))))
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{p:.2f}\n({n})" for p, n in zip(rp, rn)] for rp, rn in zip(cm_norm, cm)])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=C.EMOTIONS, yticklabels=C.EMOTIONS, ax=ax, cbar_kws={"label": "recall"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_and_save(name, y_true, y_pred, extra=None):
    """Print, save metrics JSON and confusion matrix for one model on the test set."""
    m = compute_metrics(y_true, y_pred)
    if extra:
        m.update(extra)
    print(f"\n=== {name} — test ===")
    print(classification_report(y_true, y_pred, target_names=C.EMOTIONS, digits=4, zero_division=0))
    print(f"accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
    (C.METRICS_DIR / f"{name}.json").write_text(json.dumps(m, indent=2))
    plot_confusion_matrix(y_true, y_pred, f"{name} — confusion matrix (test)",
                          C.FIG_DIR / f"confusion_{name}.png")
    return m
