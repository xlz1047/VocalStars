#!/usr/bin/env python3
"""Offline vectorization pipeline for human reference tracks.

Reads raw_manifest.csv (produced by scan_reference_assets.py), extracts F0
for each asset, and writes per-asset NPZ files + a processed_manifest.csv.

No production inference code is imported. The F0 extractor is selected per
dataset and is swappable via --extractor.

Usage:
    python scripts/data/preprocess_human_references.py \\
        --manifest  data/reference_catalog/raw_manifest.csv \\
        --out-dir   data/reference_catalog/vectors \\
        --workers   4 \\
        --extractor pyin
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.pitch_extractor import (
    GroundTruthExtractor,
    PyinExtractor,
    make_extractor,
    resample_to_10ms,
)

TARGET_HOP_S = 0.01  # canonical 10 ms hop for all output vectors
TARGET_SR = 16_000

_PROCESSED_FIELDS = [
    "asset_id", "dataset", "exercise_type", "singer_id", "technique", "note_name",
    "audio_path_rel", "vector_path_rel", "duration_s", "f0_mean_hz", "f0_std_hz",
    "voiced_fraction", "hop_s", "extractor_name", "processed_at",
]


@dataclass
class ProcessResult:
    asset_id: str
    success: bool
    error: str = ""
    row: dict | None = None


# ---------------------------------------------------------------------------
# Per-asset worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_one(
    row: dict,
    out_dir: Path,
    extractor_name: str,
) -> ProcessResult:
    """Extract F0 for one asset and write an NPZ file."""
    asset_id = row["asset_id"]
    try:
        import librosa

        audio_path = Path(row["audio_path_abs"])
        if not audio_path.exists():
            return ProcessResult(asset_id, False, f"audio not found: {audio_path}")

        channels = int(row["channels"])

        # Load audio, preserving channels for stereo (MIR-1K)
        if channels == 2:
            audio_raw, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=False)
            if audio_raw.ndim == 1:
                # librosa collapsed to mono despite mono=False (single-channel edge case)
                audio_mono = audio_raw
            else:
                # Right channel (index 1) is vocals; left (index 0) is backing mix
                audio_mono = audio_raw[1, :]
        else:
            audio_mono, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)

        audio_mono = np.asarray(audio_mono, dtype=np.float32)

        # Select extractor
        dataset = row["dataset"]
        if dataset == "mir1k" and row.get("has_ground_truth_f0") == "True":
            pv_path = REPO_ROOT / row["ground_truth_f0_path"]
            extractor = GroundTruthExtractor(pv_path)
        else:
            extractor = make_extractor(extractor_name)

        f0_raw, voiced_raw = extractor.extract(audio_mono, TARGET_SR)

        # Resample to canonical 10 ms hop
        f0_hz, voiced = resample_to_10ms(f0_raw, voiced_raw, extractor.hop_s, TARGET_HOP_S)

        duration_s = len(audio_mono) / TARGET_SR
        voiced_fraction = float(voiced.sum()) / max(1, len(voiced))
        voiced_f0 = f0_hz[voiced]
        f0_mean_hz = float(voiced_f0.mean()) if len(voiced_f0) > 0 else 0.0
        f0_std_hz  = float(voiced_f0.std())  if len(voiced_f0) > 0 else 0.0

        # Write NPZ — asset_id may contain colons/slashes; sanitize for filename
        safe_id = asset_id.replace(":", "_").replace("/", "_").replace(" ", "_")
        npz_path = out_dir / f"{safe_id}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz_path),
            f0_hz=f0_hz,
            voiced=voiced,
            hop_s=np.float32(TARGET_HOP_S),
            duration_s=np.float32(duration_s),
        )

        vector_path_rel: str
        try:
            vector_path_rel = npz_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            vector_path_rel = npz_path.as_posix()

        processed_row = {
            "asset_id":        asset_id,
            "dataset":         dataset,
            "exercise_type":   row["exercise_type"],
            "singer_id":       row["singer_id"],
            "technique":       row["technique"],
            "note_name":       row["note_name"],
            "audio_path_rel":  row["audio_path_rel"],
            "vector_path_rel": vector_path_rel,
            "duration_s":      round(duration_s, 4),
            "f0_mean_hz":      round(f0_mean_hz, 3),
            "f0_std_hz":       round(f0_std_hz, 3),
            "voiced_fraction": round(voiced_fraction, 4),
            "hop_s":           TARGET_HOP_S,
            "extractor_name":  extractor.__class__.__name__,
            "processed_at":    datetime.now(timezone.utc).isoformat(),
        }
        return ProcessResult(asset_id, True, row=processed_row)

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return ProcessResult(asset_id, False, error=f"{type(exc).__name__}: {exc}\n{tb}")


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

def run_pipeline(
    manifest_path: Path,
    out_dir: Path,
    processed_manifest_path: Path,
    extractor_name: str,
    workers: int,
    limit: int | None,
) -> None:
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if limit:
        rows = rows[:limit]

    print(f"[preprocess] Processing {len(rows)} assets with {workers} workers, extractor={extractor_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    processed_rows: list[dict] = []
    errors: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, row, out_dir, extractor_name): row["asset_id"]
            for row in rows
        }
        for i, future in enumerate(as_completed(futures), 1):
            result: ProcessResult = future.result()
            if result.success and result.row:
                processed_rows.append(result.row)
            else:
                errors.append((result.asset_id, result.error))
            if i % 50 == 0 or i == len(rows):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                print(f"  {i}/{len(rows)} ({rate:.1f}/s) — {len(errors)} errors so far")

    with processed_manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PROCESSED_FIELDS)
        writer.writeheader()
        writer.writerows(processed_rows)

    elapsed = time.perf_counter() - t0
    print(f"\n[preprocess] Done in {elapsed:.1f}s")
    print(f"  Succeeded: {len(processed_rows)}")
    print(f"  Failed:    {len(errors)}")
    if errors:
        print("\nFailed assets (first 10):")
        for asset_id, err in errors[:10]:
            print(f"  {asset_id}: {err[:200]}")
    print(f"\n  Processed manifest → {processed_manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/raw_manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/vectors",
    )
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/processed_manifest.csv",
    )
    parser.add_argument("--extractor", default="pyin", choices=["pyin"])
    parser.add_argument("--workers",   type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--limit",     type=int, default=None, help="Process only first N rows (debugging).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"[preprocess] Manifest not found: {args.manifest}")
        print("Run scan_reference_assets.py first.")
        return 1
    run_pipeline(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        processed_manifest_path=args.processed_manifest,
        extractor_name=args.extractor,
        workers=args.workers,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
