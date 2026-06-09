"""
Upload VocalStars runtime assets to Hugging Face.

Assets uploaded:
  - data/reference_catalog/   (pre-computed pitch vectors + index, derived from VocalSet/MIR-1K/GTSinger)
  - data/reference_melodies/  (built-in melody pack, MIDI, JSON)
  - data/manifests/            (processed_manifest.csv, raw_manifest.csv)
  - weights/                   (model_a_unified.pt, nanopitch_best.pth)
  - samples/                   (WAV/MP3/OGG test clips)

Raw third-party audio (data/external/) is NOT uploaded — users must obtain
those datasets from their original sources.

Usage:
    pip install huggingface_hub python-dotenv
    python scripts/upload_to_hf.py

Token resolution order:
    1. HF_TOKEN in .env (repo root)
    2. HF_TOKEN environment variable
    3. huggingface-cli login cached credentials
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Install huggingface_hub first:  pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "punumbed/vocalstars-data"
REPO_TYPE = "dataset"

ROOT = Path(__file__).resolve().parent.parent

_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)
    except ImportError:
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

ALLOW_PATTERNS = [
    "data/reference_catalog/**",
    "data/reference_melodies/**",
    "data/manifests/**",
    "weights/**",
    "samples/**",
]

IGNORE_PATTERNS = [".DS_Store", "__pycache__", ".gitkeep"]


def upload():
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    print(f"Creating repo {REPO_ID} (skips if already exists)...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        exist_ok=True,
        private=False,
        token=token,
    )

    total = sum(
        1 for p in ROOT.rglob("*")
        if p.is_file() and any(
            p.is_relative_to(ROOT / pat.split("/")[0])
            for pat in ALLOW_PATTERNS
        )
    )
    print(f"\nUploading {total} files from repo root (batched)...")

    api.upload_large_folder(
        repo_id=REPO_ID,
        folder_path=str(ROOT),
        repo_type=REPO_TYPE,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    print("\nDone. View at: https://huggingface.co/datasets/" + REPO_ID)


if __name__ == "__main__":
    upload()
