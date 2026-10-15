"""Build an embedding matrix initialised from pre-trained GloVe vectors."""
import gzip

import numpy as np

from . import config as C


def load_glove(path=C.GLOVE_PATH, vocab_words=None):
    """Read GloVe (word2vec text format, optionally gzipped). Only keeps words in `vocab_words`."""
    vectors = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) != 2:              # plain GloVe .txt has no header line
            f.seek(0)
        for line in f:
            word, *vals = line.rstrip().split(" ")
            if vocab_words is None or word in vocab_words:
                vectors[word] = np.asarray(vals, dtype=np.float32)
    return vectors


def build_embedding_matrix(vocab, dim=C.EMBED_DIM, path=C.GLOVE_PATH, seed=C.SEED):
    """Rows for known words come from GloVe; unknown words get N(0, 0.6^2·σ) noise; <pad> is zero."""
    glove = load_glove(path, set(vocab.itos))
    rng = np.random.default_rng(seed)
    std = np.std(np.stack(list(glove.values()))) if glove else 0.5
    matrix = rng.normal(0, std, size=(len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0
    hits = 0
    for i, w in enumerate(vocab.itos):
        if w in glove:
            matrix[i] = glove[w]
            hits += 1
    coverage = hits / max(len(vocab) - 2, 1)
    return matrix, coverage
