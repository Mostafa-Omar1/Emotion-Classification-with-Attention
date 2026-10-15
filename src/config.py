"""Project-wide configuration: paths, label mapping and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GLOVE_PATH = DATA_DIR / "glove" / "glove-wiki-gigaword-100.gz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (DATA_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"

# The 6 target classes (Ekman's basic emotions).
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

# Official GoEmotions -> Ekman mapping (google-research/goemotions/data/ekman_mapping.json).
# "neutral" is not one of the 6 targets and is dropped.
EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
}
GO_TO_EKMAN = {go: ek for ek, gos in EKMAN_MAPPING.items() for go in gos}

# Text preprocessing
MAX_LEN = 40          # tokens per comment after padding/truncation (no train comment is truncated; p99 = 27)
MIN_FREQ = 2          # minimum token frequency to enter the vocabulary
EMBED_DIM = 100       # GloVe dimension

# RNN training
RNN_HPARAMS = dict(
    hidden_dim=128,
    num_layers=1,
    dropout=0.5,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    patience=4,
    grad_clip=1.0,
)

# Transformer fine-tuning
TRANSFORMER_NAME = "distilbert-base-uncased"
TRANSFORMER_HPARAMS = dict(
    max_length=64,
    batch_size=32,
    lr=3e-5,
    epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
)

SEED = 42
