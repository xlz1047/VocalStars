"""
Download VocalStars runtime assets from Hugging Face.

What this downloads:
  - data/reference_catalog/   (5196 pre-computed pitch vectors + index)
  - data/reference_melodies/  (built-in melody pack)
  - data/manifests/            (processed and raw manifests)
  - weights/                   (model_a_unified.pt, nanopitch_best.pth)
  - samples/                   (test WAV/MP3/OGG clips)

What this does NOT download (third-party datasets — obtain separately):
  - VocalSet:   https://zenodo.org/record/1193957
  - MIR-1K:     https://sites.google.com/site/unvoicedsoundseparation/mir-1k
  - GTSinger:   https://github.com/GTSinger/GTSinger

Usage:
    pip install huggingface_hub python-dotenv
    python scripts/download_data.py

Optional flags:
    --dest <path>   Root directory to place files (default: repo root)
    --token <tok>   HF token (overrides .env and environment)

Token resolution order:
    1. --token flag
    2. HF_TOKEN in .env (repo root)
    3. HF_TOKEN environment variable
    4. huggingface-cli login cached credentials
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Install huggingface_hub first:  pip install huggingface_hub")
    sys.exit(1)

REPO_ID = "punumbed/vocalstars-data"

_root = Path(__file__).resolve().parent.parent
_env_file = _root / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=False)
    except ImportError:
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

DOWNLOAD_PATTERNS = [
    "data/reference_catalog/**",
    "data/reference_melodies/**",
    "data/manifests/**",
    "weights/**",
    "samples/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VocalStars runtime data from HuggingFace.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root directory to place downloaded files (default: repo root)",
    )
    parser.add_argument("--token", default=None, help="HuggingFace API token (overrides .env)")
    return parser.parse_args()


def download(dest: Path, token: str | None) -> None:
    resolved_token = token or os.environ.get("HF_TOKEN") or None
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading runtime assets from {REPO_ID}...")
    print(f"Destination: {dest}\n")

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=DOWNLOAD_PATTERNS,
        token=resolved_token,
    )

    print(f"\nAssets written to: {local_dir}")
    _print_summary(dest)


def _print_summary(root: Path) -> None:
    dirs = [
        "weights",
        "data/reference_catalog",
        "data/reference_melodies",
        "data/manifests",
        "samples",
    ]
    print("\nDownload summary:")
    for d in dirs:
        path = root / d
        if path.exists():
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
            print(f"  {d:<30} {count:>5} files  {size_mb:>7.1f} MB")
        else:
            print(f"  {d:<30}  (not found)")

    print(
        "\nNote: raw training datasets (VocalSet, MIR-1K, GTSinger) are NOT included.\n"
        "      The app runs without them — they are only needed to retrain models."
    )


if __name__ == "__main__":
    args = parse_args()
    download(args.dest, args.token)
