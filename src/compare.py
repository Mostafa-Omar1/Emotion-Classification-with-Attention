"""Collect test metrics of all four models into a comparison table and chart.

Usage:  python -m src.compare
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

MODELS = {"lstm": "LSTM", "gru": "GRU", "bilstm_attention": "BiLSTM + Attention", "distilbert": "DistilBERT"}


def main():
    rows, per_class = [], {}
    for key, label in MODELS.items():
        f = C.METRICS_DIR / f"{key}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        rows.append({"model": label, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "weighted_f1": m["weighted_f1"], "params": m.get("trainable_params"),
                     "train_min": m.get("train_seconds", 0) / 60})
        per_class[label] = {e: m["per_class"][e]["f1-score"] for e in C.EMOTIONS}
    df = pd.DataFrame(rows).set_index("model")
    pc = pd.DataFrame(per_class).T[C.EMOTIONS]

    md = ["| Model | Accuracy | Macro F1 | Weighted F1 | Trainable params | Train time (min) |",
          "|---|---|---|---|---|---|"]
    for name, r in df.iterrows():
        md.append(f"| {name} | {r.accuracy:.4f} | {r.macro_f1:.4f} | {r.weighted_f1:.4f} | "
                  f"{int(r.params):,} | {r.train_min:.1f} |")
    md += ["", "Per-class F1 (test):", "", "| Model | " + " | ".join(C.EMOTIONS) + " |",
           "|---|" + "---|" * len(C.EMOTIONS)]
    for name, r in pc.iterrows():
        md.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    (C.REPORTS_DIR / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    df[["accuracy", "macro_f1"]].plot.bar(ax=axes[0], rot=15, color=["#4C72B0", "#DD8452"])
    axes[0].set(title="Overall (test)", ylim=(0, 1))
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.3f", fontsize=8)
    x = np.arange(len(C.EMOTIONS))
    w = 0.8 / len(pc)
    for i, (name, r) in enumerate(pc.iterrows()):
        axes[1].bar(x + i * w - 0.4 + w / 2, r.values, w, label=name)
    axes[1].set_xticks(x, C.EMOTIONS)
    axes[1].set(title="Per-class F1 (test)", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
