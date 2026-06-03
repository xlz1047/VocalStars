#!/usr/bin/env python3
"""Aggregate processed_manifest.csv into data/reference_catalog/index.json.

The index is loaded once at API startup (no per-request DB queries). Each entry
embeds the pre-computed target_pitch_vector so the frontend never triggers
on-the-fly F0 computation.

Usage:
    python scripts/data/build_reference_index.py \\
        --manifest  data/reference_catalog/processed_manifest.csv \\
        --vectors   data/reference_catalog/vectors \\
        --output    data/reference_catalog/index.json \\
        [--max-vector-len 6000]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.pitch_extractor import note_name_to_hz

SCHEMA_VERSION = "vocalstars.reference_catalog.v2"

# Infer voice type from singer_id strings
_VOICE_TYPE_MAP: dict[str, str] = {
    "EN-Alto":   "female_alto",
    "EN-Tenor":  "male_tenor",
    "EN-Bass":   "male_bass",
    "EN-Soprano": "female_soprano",
    "f":         "female",
    "m":         "male",
}

_TECHNIQUE_SKILL_MAP: dict[str, list[str]] = {
    "vibrato":       ["vibrato", "pitch_stability"],
    "pitch_slide":   ["pitch_slide", "slide_smoothness"],
    "sustained_note": ["pitch_stability", "tone_consistency"],
    "long_note":     ["pitch_stability", "breath_control"],
}

_EXERCISE_TASK_TYPE: dict[str, str] = {
    "vibrato":       "reference_song",
    "pitch_slide":   "pitch_slide",
    "sustained_note": "sustained_note",
    "long_note":     "sustained_note",
}


def _voice_type(singer_id: str) -> str:
    for prefix, label in _VOICE_TYPE_MAP.items():
        if singer_id.startswith(prefix):
            return label
    return "unknown"


def _audio_url(audio_path_rel: str) -> str:
    return f"/api/audio/file?path={audio_path_rel}"


def _vector_url(asset_id: str) -> str:
    safe = asset_id.replace(":", "_").replace("/", "_").replace(" ", "_")
    return f"/api/reference/vector/{safe}"


def _build_task_config(row: dict) -> dict:
    exercise_type = row["exercise_type"]
    task_type = _EXERCISE_TASK_TYPE.get(exercise_type, "reference_song")
    skills = _TECHNIQUE_SKILL_MAP.get(exercise_type, ["pitch_stability"])
    reference: dict = {
        "type": "human_vocal_reference",
        "source": row["dataset"],
        "technique": row["technique"],
        "singer": row["singer_id"],
        "hop_s": float(row["hop_s"]),
    }
    if row.get("note_name"):
        reference["note"] = row["note_name"]
    return {
        "task_type": task_type,
        "target": None,
        "reference": reference,
        "skill_focus": skills,
        "scoring_mode": "diagnostic",
        "strictness": "beginner",
    }


def _load_vector(vector_path_rel: str, max_len: int) -> tuple[list[float], list[bool]]:
    """Load NPZ and return (f0_hz list, voiced list), truncated to max_len frames."""
    npz_path = REPO_ROOT / vector_path_rel
    if not npz_path.exists():
        return [], []
    data = np.load(str(npz_path))
    f0: np.ndarray = data["f0_hz"]
    voiced: np.ndarray = data["voiced"].astype(bool)
    if len(f0) > max_len:
        f0 = f0[:max_len]
        voiced = voiced[:max_len]
    return [round(float(v), 3) for v in f0], [bool(v) for v in voiced]


def build_index(
    manifest_path: Path,
    vectors_dir: Path,
    output_path: Path,
    max_vector_len: int,
) -> None:
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    print(f"[index] Building index from {len(rows)} processed assets …")
    entries: list[dict] = []
    skipped = 0

    for row in rows:
        vector_path_rel = row.get("vector_path_rel", "")
        if not vector_path_rel:
            skipped += 1
            continue

        f0_list, voiced_list = _load_vector(vector_path_rel, max_vector_len)
        if not f0_list:
            skipped += 1
            continue

        exercise_type = row["exercise_type"]
        note_name = row.get("note_name", "") or ""
        f0_mean = float(row.get("f0_mean_hz") or 0)
        f0_std  = float(row.get("f0_std_hz") or 0)
        voiced_fraction = float(row.get("voiced_fraction") or 0)

        # Derive f0_target_hz: for VocalSet sustained notes, encode from note name
        f0_target_hz: float | None = None
        if note_name:
            f0_target_hz = note_name_to_hz(note_name)
        if f0_target_hz is None and f0_mean > 0:
            f0_target_hz = round(f0_mean, 2)

        entry: dict = {
            "asset_id":            row["asset_id"],
            "dataset":             row["dataset"],
            "exercise_type":       exercise_type,
            "exercise_type_tags":  [exercise_type],
            "singer_id":           row["singer_id"],
            "voice_type":          _voice_type(row["singer_id"]),
            "technique":           row["technique"],
            "note_name":           note_name or None,
            "f0_target_hz":        f0_target_hz,
            "duration_s":          float(row.get("duration_s") or 0),
            "hop_s":               float(row.get("hop_s") or 0.01),
            "audio_url":           _audio_url(row["audio_path_rel"]),
            "vector_url":          _vector_url(row["asset_id"]),
            "target_pitch_vector": f0_list,
            "voiced_vector":       voiced_list,
            "f0_summary": {
                "mean_hz":          round(f0_mean, 3),
                "std_hz":           round(f0_std, 3),
                "voiced_fraction":  round(voiced_fraction, 4),
            },
            "task_config":         _build_task_config(row),
        }
        entries.append(entry)

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "entry_count":    len(entries),
        "entries":        entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    print(f"[index] Wrote {len(entries)} entries ({skipped} skipped) → {output_path}")
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"[index] Index file size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/processed_manifest.csv",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/vectors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/reference_catalog/index.json",
    )
    parser.add_argument(
        "--max-vector-len",
        type=int,
        default=6000,
        help="Truncate F0 vectors to this many frames (6000 = 60 s at 10 ms hop).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"[index] Processed manifest not found: {args.manifest}")
        print("Run preprocess_human_references.py first.")
        return 1
    build_index(
        manifest_path=args.manifest,
        vectors_dir=args.vectors,
        output_path=args.output,
        max_vector_len=args.max_vector_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
