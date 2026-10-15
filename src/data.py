"""Loading GoEmotions, mapping to 6 Ekman emotions, cleaning, tokenising and padding."""
import html
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C

PAD, UNK = "<pad>", "<unk>"

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"/?u/\w+")
_SUB = re.compile(r"/?r/\w+")
_REPEAT = re.compile(r"(.)\1{2,}")           # "soooo" -> "soo"
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]")


# ---------------------------------------------------------------- loading
def map_labels(go_label_ids, go_names):
    """Map a list of GoEmotions label ids to one Ekman class id, or None.

    Neutral is dropped; examples whose labels map to more than one Ekman class
    are ambiguous for single-label classification and are dropped too.
    """
    ekman = {C.GO_TO_EKMAN[go_names[i]] for i in go_label_ids if go_names[i] in C.GO_TO_EKMAN}
    if len(ekman) != 1:
        return None
    return C.LABEL2ID[ekman.pop()]


def load_goemotions(cache=True):
    """Return {"train"|"validation"|"test": DataFrame[text, label]} with 6-class labels."""
    cache_file = C.DATA_DIR / "goemotions_ekman6.parquet"
    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return {s: g.drop(columns="split").reset_index(drop=True) for s, g in df.groupby("split")}

    from datasets import load_dataset

    ds = load_dataset(C.DATASET_NAME, C.DATASET_CONFIG)
    go_names = ds["train"].features["labels"].feature.names
    frames = []
    for split in ("train", "validation", "test"):
        d = ds[split].to_pandas()
        d["label"] = d["labels"].apply(lambda ls: map_labels(ls, go_names))
        d = d.dropna(subset=["label"]).astype({"label": int})
        d["split"] = split
        frames.append(d[["text", "label", "split"]])
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache_file, index=False)
    return {s: g.drop(columns="split").reset_index(drop=True) for s, g in df.groupby("split")}


# ---------------------------------------------------------------- cleaning / tokenising
def clean_text(text: str) -> str:
    text = html.unescape(text).lower()
    text = _URL.sub(" url ", text)
    text = _USER.sub(" user ", text)
    text = _SUB.sub(" subreddit ", text)
    text = text.replace("[name]", " name ").replace("[religion]", " religion ")
    text = text.replace("’", "'").replace("‘", "'")
    text = _REPEAT.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str):
    return _TOKEN.findall(clean_text(text))


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_freq=C.MIN_FREQ):
        counter = Counter(t for toks in token_lists for t in toks)
        words = sorted((w for w, c in counter.items() if c >= min_freq), key=lambda w: (-counter[w], w))
        return cls([PAD, UNK] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, max_len=C.MAX_LEN):
        ids = [self.stoi.get(t, 1) for t in tokens[:max_len]]
        length = max(len(ids), 1)
        ids = ids + [0] * (max_len - len(ids))
        return ids, length

    def save(self, path):
        path.write_text("\n".join(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(path.read_text(encoding="utf-8").split("\n"))


class EmotionDataset(Dataset):
    """Pre-tokenised, padded tensors: ids [N, MAX_LEN], lengths [N], labels [N]."""

    def __init__(self, texts, labels, vocab, max_len=C.MAX_LEN):
        enc = [vocab.encode(tokenize(t), max_len) for t in texts]
        self.ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in enc], dtype=torch.long)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.lengths[i], self.labels[i]


def class_weights(labels, n_classes=len(C.EMOTIONS)):
    """Inverse-sqrt-frequency weights (normalised to mean 1) to soften the heavy joy imbalance."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float)
