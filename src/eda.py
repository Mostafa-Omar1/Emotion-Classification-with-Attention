"""Exploratory analysis of GoEmotions before and after mapping to the 6 Ekman classes.

Usage:  python -m src.eda
"""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from . import config as C
from .data import load_goemotions, tokenize


def main():
    raw = load_dataset(C.DATASET_NAME, C.DATASET_CONFIG)
    names = raw["train"].features["labels"].feature.names
    train = raw["train"].to_pandas()

    # 1. raw 28-label distribution
    raw_counts = Counter(names[i] for ls in train["labels"] for i in ls)
    s = pd.Series(raw_counts).sort_values()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = [("#999999" if n == "neutral" else "#4C72B0") for n in s.index]
    s.plot.barh(ax=ax, color=colors)
    ax.set(title="GoEmotions train — raw label counts (28 labels)", xlabel="count")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_raw_labels.png", dpi=150)
    plt.close(fig)

    # 2. what happens during mapping
    outcome = Counter()
    for ls in train["labels"]:
        ek = {C.GO_TO_EKMAN[names[i]] for i in ls if names[i] in C.GO_TO_EKMAN}
        outcome["kept (single Ekman class)" if len(ek) == 1 else
                "dropped: neutral only" if len(ek) == 0 else "dropped: conflicting classes"] += 1

    # 3. mapped 6-class distribution per split
    splits = load_goemotions()
    dist = pd.DataFrame({s: df["label"].map(C.ID2LABEL).value_counts() for s, df in splits.items()})
    dist = dist.loc[C.EMOTIONS]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dist[["train", "validation", "test"]].plot.bar(ax=ax, rot=0)
    ax.set(title="6-class distribution after Ekman mapping", ylabel="examples")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_ekman_distribution.png", dpi=150)
    plt.close(fig)

    # 4. sequence lengths (to choose MAX_LEN)
    lengths = np.array([len(tokenize(t)) for t in splits["train"]["text"]])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=range(0, lengths.max() + 2), color="#55A868")
    ax.axvline(C.MAX_LEN, color="red", ls="--", label=f"MAX_LEN={C.MAX_LEN}")
    ax.set(title="Tokens per comment (train)", xlabel="tokens", ylabel="comments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_lengths.png", dpi=150)
    plt.close(fig)

    summary = {
        "raw_sizes": {s: len(raw[s]) for s in raw},
        "mapped_sizes": {s: len(df) for s, df in splits.items()},
        "train_mapping_outcome": dict(outcome),
        "class_distribution": dist.to_dict(),
        "length_percentiles": {p: int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)},
        "pct_truncated": float((lengths > C.MAX_LEN).mean()),
        "examples": {e: splits["train"][splits["train"]["label"] == C.LABEL2ID[e]]["text"].head(3).tolist()
                     for e in C.EMOTIONS},
    }
    (C.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
