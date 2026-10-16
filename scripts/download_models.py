"""Rebuild the fine-tuned DistilBERT weights from the split zip in models/distilbert_parts/.

The weights are stored in fp16 as a zip split into 20 MB parts (each under GitHub's file-size limit).
If the parts are not present locally (e.g. a shallow/sparse checkout) they are downloaded from GitHub.

Usage:  python scripts/download_models.py
"""
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/Mostafa-Omar1/Emotion-Classification-with-Attention/raw/main/models/distilbert_parts/"
N_PARTS = 6
DEST = Path(__file__).resolve().parent.parent / "models"


def main():
    if (DEST / "distilbert" / "config.json").exists():
        print("models/distilbert already present")
        return
    buf = bytearray()
    for i in range(N_PARTS):
        local = DEST / "distilbert_parts" / f"distilbert.zip.part{i:02d}"
        if local.exists():
            buf += local.read_bytes()
            continue
        url = f"{BASE}distilbert.zip.part{i:02d}"
        print(f"downloading part {i + 1}/{N_PARTS} ...")
        try:
            buf += urllib.request.urlopen(url).read()
        except urllib.error.URLError as e:
            sys.exit(f"download failed for {url}: {e}")
    zipfile.ZipFile(io.BytesIO(bytes(buf))).extractall(DEST)
    print("extracted to", DEST / "distilbert")


if __name__ == "__main__":
    main()
