"""Build a manifest.csv by scanning an existing extraction directory.

Used to recover a manifest when extraction died before writing one (e.g., disk-full crash).
"""

import csv
import pathlib
import sys

import numpy as np


def build(out_dir: pathlib.Path, dataset_name: str) -> None:
    npz_dir = out_dir / dataset_name
    if not npz_dir.exists():
        print(f"ERROR: {npz_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Match schema used by extract_all.py
    fieldnames = ["npz_path", "audio_path", "dataset", "singer_id", "technique", "n_frames"]
    rows, skipped = [], 0
    for npz in sorted(npz_dir.glob("*.npz")):
        try:
            d = np.load(npz)
            assert "f0_hz" in d and "hcqt" in d and d["f0_hz"].shape[0] > 0
            rows.append({
                "npz_path": str(npz),
                "audio_path": "",
                "dataset": dataset_name,
                "singer_id": "",
                "technique": "",
                "n_frames": int(d["f0_hz"].shape[0]),
            })
        except Exception:
            skipped += 1

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {manifest}  ({skipped} corrupt files skipped)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="extraction output directory")
    p.add_argument("--dataset", required=True, help="dataset subdirectory name (e.g. popbutfy)")
    args = p.parse_args()
    build(pathlib.Path(args.out_dir), args.dataset)
