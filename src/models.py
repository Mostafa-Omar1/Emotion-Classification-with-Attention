"""Sequence models: LSTM, GRU and BiLSTM + additive attention."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _embedding(embedding_matrix, freeze=False):
    weights = torch.as_tensor(embedding_matrix, dtype=torch.float)
    return nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=0)


class RNNClassifier(nn.Module):
    """Unidirectional LSTM/GRU; the final hidden state of the last layer feeds the classifier."""

    def __init__(self, embedding_matrix, n_classes, hidden_dim=128, num_layers=1,
                 dropout=0.4, cell="lstm", freeze_embeddings=False):
        super().__init__()
        self.embedding = _embedding(embedding_matrix, freeze_embeddings)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(self.embedding.embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if isinstance(h, tuple):         # LSTM returns (h_n, c_n)
            h = h[0]
        return self.fc(self.dropout(h[-1])), None


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention: score_t = vᵀ tanh(W h_t + b); padding positions are masked."""

    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.proj(h))).squeeze(-1)          # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)                            # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)            # [B, H]
        return context, alpha


class BiLSTMAttention(nn.Module):
    def __init__(self, embedding_matrix, n_classes, hidden_dim=128, num_layers=1,
                 dropout=0.4, freeze_embeddings=False, **_):
        super().__init__()
        self.embedding = _embedding(embedding_matrix, freeze_embeddings)
        self.rnn = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AdditiveAttention(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        mask = torch.arange(ids.size(1), device=ids.device)[None, :] < lengths[:, None].to(ids.device)
        context, alpha = self.attention(out, mask)
        return self.fc(self.dropout(context)), alpha


def build_model(name, embedding_matrix, n_classes, **hp):
    keys = ("hidden_dim", "num_layers", "dropout")
    kw = {k: hp[k] for k in keys if k in hp}
    if name == "lstm":
        return RNNClassifier(embedding_matrix, n_classes, cell="lstm", **kw)
    if name == "gru":
        return RNNClassifier(embedding_matrix, n_classes, cell="gru", **kw)
    if name == "bilstm_attention":
        return BiLSTMAttention(embedding_matrix, n_classes, **kw)
    raise ValueError(f"unknown model {name!r}")
