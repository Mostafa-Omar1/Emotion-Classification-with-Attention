"""Download the fine-tuned DistilBERT weights (too large for git) from the GitHub release.

Usage:  python scripts/download_models.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/Mostafa-Omar1/Emotion-Classification-with-Attention/releases/download/v1.0/distilbert.zip"
DEST = Path(__file__).resolve().parent.parent / "models"


def main():
    if (DEST / "distilbert" / "config.json").exists():
        print("models/distilbert already present")
        return
    print(f"downloading {URL} ...")
    try:
        data = urllib.request.urlopen(URL).read()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"download failed: {e}")
    zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
    print("extracted to", DEST / "distilbert")


if __name__ == "__main__":
    main()
