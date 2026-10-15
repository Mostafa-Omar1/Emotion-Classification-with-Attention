"""Inference helpers used by the attention visualisation and the web app."""
import numpy as np
import torch

from . import config as C
from .data import Vocab, tokenize
from .models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNPredictor:
    """Loads a trained LSTM / GRU / BiLSTM+Attention checkpoint from models/."""

    def __init__(self, name="bilstm_attention"):
        ckpt = torch.load(C.MODELS_DIR / f"{name}.pt", map_location="cpu")
        self.vocab = Vocab.load(C.MODELS_DIR / "vocab.txt")
        dummy = np.zeros(ckpt["embedding_shape"], dtype=np.float32)
        self.model = build_model(name, dummy, len(C.EMOTIONS), **ckpt["hparams"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, texts):
        """Returns (probs [N, 6], tokens list, attention list or None)."""
        toks = [tokenize(t)[: C.MAX_LEN] or [""] for t in texts]
        enc = [self.vocab.encode(t) for t in toks]
        ids = torch.tensor([e[0] for e in enc], device=DEVICE)
        lengths = torch.tensor([e[1] for e in enc], device=DEVICE)
        logits, alpha = self.model(ids, lengths)
        probs = torch.softmax(logits, -1).cpu().numpy()
        attn = None if alpha is None else [a[: len(t)].cpu().numpy() for a, t in zip(alpha, toks)]
        return probs, toks, attn


class TransformerPredictor:
    def __init__(self, path=C.MODELS_DIR / "distilbert"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=C.TRANSFORMER_HPARAMS["max_length"],
                             padding=True, return_tensors="pt").to(DEVICE)
        return torch.softmax(self.model(**enc).logits, -1).float().cpu().numpy()
