"""Merge a pyin-extracted manifest with a yin-extracted manifest.

Rows from ``--pyin`` take precedence (used for VocalSet+GTSinger).
Rows from ``--yin`` that belong to datasets NOT in the pyin manifest
are appended (used for PopBuTFy which was not re-extracted with pyin).

Usage::

    python ml_new/data/merge_manifests.py \\
        --pyin ml_new/data/extracted_pyin/manifest.csv \\
        --yin  ml_new/data/extracted_hires/manifest.csv \\
        --out  ml_new/data/extracted_merged/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def merge(pyin_csv: Path, yin_csv: Path, out_csv: Path) -> None:
    def _read(p: Path) -> list[dict]:
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))

    pyin_rows = _read(pyin_csv)
    yin_rows = _read(yin_csv)

    pyin_datasets = {r["dataset"] for r in pyin_rows}

    # Keep yin rows only for datasets not covered by the pyin extraction
    yin_extra = [r for r in yin_rows if r["dataset"] not in pyin_datasets]

    merged = pyin_rows + yin_extra

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged[0].keys())
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"Merged manifest: {len(merged)} rows → {out_csv}")
    print(f"  pyin datasets : {sorted(pyin_datasets)}  ({len(pyin_rows)} rows)")
    print(f"  yin extra     : {sorted({r['dataset'] for r in yin_extra})}  ({len(yin_extra)} rows)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge pyin+yin manifests.")
    p.add_argument("--pyin", type=Path, required=True, help="Manifest with pyin-extracted clips")
    p.add_argument("--yin", type=Path, required=True, help="Manifest with yin-extracted clips (fallback)")
    p.add_argument("--out", type=Path, required=True, help="Output merged manifest path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge(args.pyin, args.yin, args.out)
