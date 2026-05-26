"""Feature extraction runner for GTSinger, VocalSet, and PopBuTFy datasets.

For each audio clip this script extracts:
  - HCQT  (6, 60, T) float32  — log-magnitude harmonic CQT
  - vad_features  (3, T) float32  — [rms, spectral_flatness, zcr]
  - f0_hz  (T,) float32  — fundamental frequency (0.0 = unvoiced)
  - vad  (T,) uint8  — voice activity label (0 or 1)

Results are saved as per-clip ``.npz`` files under
``<output_dir>/<dataset>/``.  A ``manifest.csv`` index is written at
``<output_dir>/manifest.csv``.

Extraction is parallelised across ``--workers`` processes (default: all CPUs).
F0 uses the fast YIN backend by default; pass ``--f0-method accurate`` to use
pyin (~100x slower but more robust for vibrato/falsetto).

Usage::

    python ml_new/data/extract_all.py \\
        --data-dir ml/data/raw \\
        --output-dir ml_new/data/extracted

    # Test 5 clips per dataset, no writes
    python ml_new/data/extract_all.py --dry-run --n-samples 5
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.feature_extraction.hcqt import HCQTExtractor
from ml_new.feature_extraction.vad_features import VADFeatureExtractor
from ml_new.feature_extraction.labels import (
    extract_f0,
    extract_vad_gtsinger,
    extract_vad_energy,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SR = 16000
HOP_LENGTH = 160


# ---------------------------------------------------------------------------
# Dataset walkers
# ---------------------------------------------------------------------------

def _walk_gtsinger(root: Path) -> list[dict]:
    records = []
    english_dir = root / "English"
    if not english_dir.exists():
        log.warning("GTSinger English dir not found: %s", english_dir)
        return records
    for wav_path in sorted(english_dir.glob("**/*.wav")):
        parts = wav_path.parts
        if len(parts) < 6:
            continue
        singer_id = parts[-5]
        technique = parts[-4].lower().replace(" ", "_")
        json_path = wav_path.with_suffix(".json")
        records.append({
            "audio_path": wav_path,
            "singer_id": singer_id,
            "technique": technique,
            "json_path": json_path if json_path.exists() else None,
            "dataset": "gtsinger",
        })
    return records


def _walk_vocalset(root: Path) -> list[dict]:
    records = []
    full_dir = root / "FULL"
    if not full_dir.exists():
        log.warning("VocalSet FULL dir not found: %s", full_dir)
        return records
    for wav_path in sorted(full_dir.glob("**/*.wav")):
        parts = wav_path.parts
        if len(parts) < 5:
            continue
        singer_id = parts[-4]
        technique = parts[-2]
        records.append({
            "audio_path": wav_path,
            "singer_id": singer_id,
            "technique": technique,
            "json_path": None,
            "dataset": "vocalset",
        })
    return records


def _walk_popbutfy(root: Path) -> list[dict]:
    records = []
    data_dir = root / "data"
    if not data_dir.exists():
        log.warning("PopBuTFy data dir not found: %s", data_dir)
        return records
    for mp3_path in sorted(data_dir.glob("**/*.mp3")):
        group_name = mp3_path.parent.name
        parts_hash = group_name.split("#")
        if len(parts_hash) < 3:
            continue
        singer_id = parts_hash[0]
        song_quality = parts_hash[2]
        tokens = song_quality.rsplit("_", 1)
        technique = tokens[-1].lower() if len(tokens) == 2 else "unknown"
        records.append({
            "audio_path": mp3_path,
            "singer_id": singer_id,
            "technique": technique,
            "json_path": None,
            "dataset": "popbutfy",
        })
    return records


# ---------------------------------------------------------------------------
# Per-clip extraction (runs in a worker process)
# ---------------------------------------------------------------------------

def _worker_extract(args: tuple) -> dict | None:
    """Extract features for one clip.  Designed for ProcessPoolExecutor."""
    record, out_subdir, dry_run, clip_idx, f0_method, bins_per_octave, n_bins = args

    # Instantiate extractors inside the worker (not picklable across processes).
    hcqt_ext = HCQTExtractor(sr=SR, hop_length=HOP_LENGTH,
                             bins_per_octave=bins_per_octave, n_bins=n_bins)
    vad_feat_ext = VADFeatureExtractor(sr=SR, hop_length=HOP_LENGTH)

    audio_path = record["audio_path"]
    try:
        audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    except Exception as exc:
        return {"status": "error", "path": str(audio_path), "msg": str(exc)}

    if len(audio) < HOP_LENGTH * 4:
        return {"status": "skip", "path": str(audio_path), "msg": "too short"}

    try:
        hcqt = hcqt_ext.compute(audio)
        vad_feats = vad_feat_ext.compute(audio)
        f0_hz, voiced_flag, voiced_probs = extract_f0(audio, sr=SR, hop_length=HOP_LENGTH, method=f0_method)

        if record["json_path"] is not None:
            n_frames = hcqt.shape[2]
            vad = extract_vad_gtsinger(
                record["json_path"], n_frames=n_frames, sr=SR, hop_length=HOP_LENGTH
            )
        else:
            vad = extract_vad_energy(audio, sr=SR, hop_length=HOP_LENGTH)

        T = min(hcqt.shape[2], vad_feats.shape[1], len(f0_hz), len(vad), len(voiced_probs))
        hcqt = hcqt[:, :, :T]
        vad_feats = vad_feats[:, :T]
        f0_hz = f0_hz[:T]
        vad = vad[:T]
        voiced_probs = voiced_probs[:T]

    except Exception as exc:
        return {"status": "error", "path": str(audio_path), "msg": str(exc)}

    if dry_run:
        return {
            "status": "ok",
            "dry_run": True,
            "path": str(audio_path),
            "singer_id": record["singer_id"],
            "technique": record["technique"],
            "dataset": record["dataset"],
            "T": T,
            "hcqt_shape": hcqt.shape,
            "f0_min": float(f0_hz[f0_hz > 0].min()) if (f0_hz > 0).any() else 0.0,
            "f0_max": float(f0_hz.max()),
            "vad_ratio": float(vad.mean()),
        }

    # Save NPZ
    dataset_name = record["dataset"]
    npz_path = Path(out_subdir) / f"{dataset_name}_{clip_idx:06d}.npz"
    np.savez_compressed(
        str(npz_path),
        hcqt=hcqt.astype(np.float32),
        vad_features=vad_feats.astype(np.float32),
        f0_hz=f0_hz.astype(np.float32),
        vad=vad.astype(np.uint8),
        voiced_probs=voiced_probs.astype(np.float32),
    )

    return {
        "status": "ok",
        "npz_path": str(npz_path),
        "audio_path": str(audio_path),
        "dataset": dataset_name,
        "singer_id": record["singer_id"],
        "technique": record["technique"],
        "n_frames": T,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    n_samples: int | None = None,
    workers: int | None = None,
    f0_method: str = "fast",
    bins_per_octave: int = 12,
    n_bins: int = 60,
    datasets: list[str] | None = None,
) -> None:
    """Extract features for all three datasets (or a subset via ``datasets``).

    Args:
        data_dir: Root containing ``gtsinger/``, ``vocalset/``, ``popbutfy/``.
        output_dir: Destination for ``.npz`` files and ``manifest.csv``.
        dry_run: Validate shapes without writing files.
        n_samples: Limit clips per dataset (for quick tests).
        workers: Worker processes (default: all CPU cores).
        f0_method: ``"fast"`` (yin) or ``"accurate"`` (pyin).
        bins_per_octave: CQT frequency resolution (12=semitone, 36=third-tone).
        n_bins: Total CQT bins per harmonic (must keep all harmonics < Nyquist).
        datasets: If given, only process these dataset names (e.g. ["vocalset", "gtsinger"]).
    """
    if workers is None:
        workers = os.cpu_count() or 1

    walkers = {
        "gtsinger": (_walk_gtsinger, data_dir / "gtsinger"),
        "vocalset": (_walk_vocalset, data_dir / "vocalset"),
        "popbutfy": (_walk_popbutfy, data_dir / "popbutfy"),
    }

    if datasets is not None:
        walkers = {k: v for k, v in walkers.items() if k in datasets}

    manifest_rows: list[dict] = []

    for dataset_name, (walker_fn, dataset_root) in walkers.items():
        records = walker_fn(dataset_root)
        if not records:
            log.info("No records for %s — skipping.", dataset_name)
            continue

        if n_samples is not None:
            records = records[:n_samples]

        log.info(
            "Processing %s: %d clips | workers=%d | f0=%s",
            dataset_name, len(records), workers, f0_method,
        )

        out_subdir = output_dir / dataset_name
        if not dry_run:
            out_subdir.mkdir(parents=True, exist_ok=True)

        # Build task list
        tasks = [
            (record, str(out_subdir), dry_run, i, f0_method, bins_per_octave, n_bins)
            for i, record in enumerate(records)
        ]

        ok = 0
        err = 0
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futs = {pool.submit(_worker_extract, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                done += 1
                result = fut.result()
                if result is None or result["status"] == "error":
                    err += 1
                    msg = result.get("msg", "unknown") if result else "None"
                    log.debug("Error on %s: %s", futs[fut][0]["audio_path"], msg)
                elif result["status"] == "skip":
                    pass
                else:
                    ok += 1
                    if dry_run:
                        log.info(
                            "[dry-run] %s | %s | T=%d | f0=[%.1f,%.1f] | vad=%.2f",
                            Path(result["path"]).name,
                            result["singer_id"],
                            result["T"],
                            result["f0_min"], result["f0_max"],
                            result["vad_ratio"],
                        )
                    else:
                        manifest_rows.append({
                            "npz_path": result["npz_path"],
                            "audio_path": result["audio_path"],
                            "dataset": result["dataset"],
                            "singer_id": result["singer_id"],
                            "technique": result["technique"],
                            "n_frames": result["n_frames"],
                        })
                if done % 500 == 0 or done == len(tasks):
                    log.info("  %d/%d done (%d ok, %d errors)", done, len(tasks), ok, err)

        log.info("  → %s: %d ok, %d errors/skipped", dataset_name, ok, err)

    if not dry_run and manifest_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        fieldnames = ["npz_path", "audio_path", "dataset", "singer_id", "technique", "n_frames"]
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        log.info("Manifest: %s (%d rows)", manifest_path, len(manifest_rows))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract HCQT+VAD features from singing datasets.")
    p.add_argument("--data-dir", type=Path, default=Path("ml/data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("ml_new/data/extracted"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default: all CPU cores)")
    p.add_argument("--f0-method", choices=["fast", "accurate"], default="fast",
                   help="fast=yin (~100x faster), accurate=pyin (better for vibrato)")
    p.add_argument("--bins-per-octave", type=int, default=12,
                   help="CQT frequency resolution (12=semitone, 36=third-tone)")
    p.add_argument("--n-bins", type=int, default=60,
                   help="CQT bins per harmonic layer")
    p.add_argument("--datasets", nargs="+", choices=["gtsinger", "vocalset", "popbutfy"],
                   default=None, metavar="DS",
                   help="Process only these datasets (default: all three)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        n_samples=args.n_samples,
        workers=args.workers,
        f0_method=args.f0_method,
        bins_per_octave=args.bins_per_octave,
        n_bins=args.n_bins,
        datasets=args.datasets,
    )
