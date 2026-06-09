#!/usr/bin/env python3
"""Scan VocalSet, MIR-1K, and GTSinger and emit a raw asset manifest CSV.

This script is read-only — it produces no audio output and modifies no files.
Run it first to validate dataset structure before running preprocess_human_references.py.

Usage:
    python scripts/data/scan_reference_assets.py \\
        --vocalset  ml/data/raw/vocalset/FULL \\
        --mir1k     data/external/mir1k/extracted/MIR-1K \\
        --gtsinger  ml/data/raw/gtsinger/English \\
        --output    data/reference_catalog/raw_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Exercise-type classification rules
# ---------------------------------------------------------------------------

# VocalSet: classify based on exercise directory and technique subdirectory.
# File naming: {singer_abbrev}_{exercise_abbrev}_{technique}_{vowel}.wav — no pitch in filename.
_VOCALSET_RULES: list[tuple[str, re.Pattern]] = [
    ("sustained_note", re.compile(r"^long_tones/", re.I)),
    ("vibrato",        re.compile(r"/vibrato/", re.I)),
    ("pitch_slide",    re.compile(r"/(portamento|glissando)/", re.I)),
]
_VOCALSET_EXCLUDE = re.compile(r"^(scales|excerpts|songs)/", re.I)

# GTSinger: technique directory name
_GTSINGER_TECHNIQUE_MAP: dict[str, str] = {
    "Vibrato":                    "vibrato",
    "Glissando":                  "pitch_slide",
    "Portamento":                 "pitch_slide",
    "Mixed_Voice_and_Falsetto":   "vibrato",
    "Normal":                     "sustained_note",
}
_GTSINGER_EXCLUDE = {"Breathy", "Pharyngeal", "Belt"}

# MIR-1K: all clips become "sustained_note" (phrase-level, varied pitches)
_MIR1K_EXERCISE_TYPE = "sustained_note"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssetRecord:
    asset_id: str
    dataset: str
    exercise_type: str
    singer_id: str
    technique: str
    note_name: str
    audio_path_abs: str
    audio_path_rel: str
    channels: int
    sample_rate: int
    duration_s: float
    has_ground_truth_f0: bool
    ground_truth_f0_path: str


_CSV_FIELDS = [f.name for f in AssetRecord.__dataclass_fields__.values()]


# ---------------------------------------------------------------------------
# Audio metadata helpers
# ---------------------------------------------------------------------------

def _wav_meta(path: Path) -> tuple[int, int, float] | None:
    """Return (channels, sample_rate, duration_s) or None on failure."""
    try:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            frames = wf.getnframes()
            dur = frames / float(sr) if sr > 0 else 0.0
            return ch, sr, round(dur, 4)
    except Exception:
        return None


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# VocalSet scanner
# ---------------------------------------------------------------------------

def scan_vocalset(root: Path) -> list[AssetRecord]:
    """Scan ml/data/raw/vocalset/FULL/ and classify files.

    Directory layout:
        FULL/{singer_id}/{exercise_dir}/{technique}/{note}.wav
    """
    records: list[AssetRecord] = []
    if not root.exists():
        print(f"[scan] VocalSet root not found, skipping: {root}")
        return records

    for singer_dir in sorted(root.iterdir()):
        if not singer_dir.is_dir():
            continue
        singer_id = singer_dir.name

        for exercise_dir in sorted(singer_dir.iterdir()):
            if not exercise_dir.is_dir():
                continue

            for technique_dir in sorted(exercise_dir.iterdir()):
                if not technique_dir.is_dir():
                    continue

                technique = technique_dir.name
                for wav_path in sorted(technique_dir.glob("*.wav")):
                    rel_path = _rel(wav_path)
                    # Build the path fragment used for classification.
                    # Format: "{exercise_dir}/{technique}/{filename}"
                    path_fragment = f"{exercise_dir.name}/{technique}/{wav_path.name}"

                    if _VOCALSET_EXCLUDE.match(path_fragment):
                        continue

                    exercise_type: str | None = None
                    for et, pattern in _VOCALSET_RULES:
                        if pattern.search(path_fragment):
                            exercise_type = et
                            break
                    if exercise_type is None:
                        continue

                    meta = _wav_meta(wav_path)
                    if meta is None:
                        continue
                    channels, sr, dur = meta

                    # VocalSet filenames encode vowel sounds, not pitch names.
                    # note_name is left empty; pYIN will determine the actual F0.
                    note_name = ""
                    # Use file stem as the unique clip identifier (includes vowel/abbrev).
                    clip_id = wav_path.stem

                    asset_id = f"vocalset:{singer_id}:{exercise_dir.name}:{technique}:{clip_id}"
                    records.append(
                        AssetRecord(
                            asset_id=asset_id,
                            dataset="vocalset",
                            exercise_type=exercise_type,
                            singer_id=singer_id,
                            technique=technique,
                            note_name=note_name,
                            audio_path_abs=str(wav_path.resolve()),
                            audio_path_rel=rel_path,
                            channels=channels,
                            sample_rate=sr,
                            duration_s=dur,
                            has_ground_truth_f0=False,
                            ground_truth_f0_path="",
                        )
                    )

    print(f"[scan] VocalSet: {len(records)} assets")
    return records


# ---------------------------------------------------------------------------
# MIR-1K scanner
# ---------------------------------------------------------------------------

def scan_mir1k(root: Path) -> list[AssetRecord]:
    """Scan data/external/mir1k/extracted/MIR-1K/ and classify files.

    Directory layout:
        MIR-1K/
          Wavfile/{stem}.wav    (stereo: ch0=mix, ch1=vocals)
          PitchLabel/{stem}.pv  (ground-truth F0 at 20 ms hop)
    """
    records: list[AssetRecord] = []
    if not root.exists():
        print(f"[scan] MIR-1K root not found, skipping: {root}")
        return records

    wav_dir = root / "Wavfile"
    pv_dir = root / "PitchLabel"

    if not wav_dir.exists():
        print(f"[scan] MIR-1K Wavfile directory not found: {wav_dir}")
        return records

    for wav_path in sorted(wav_dir.glob("*.wav")):
        pv_path = pv_dir / f"{wav_path.stem}.pv"
        has_gt = pv_path.exists()

        meta = _wav_meta(wav_path)
        if meta is None:
            continue
        channels, sr, dur = meta

        # Derive singer_id from filename pattern {singer}_{session}_{phrase}.wav
        parts = wav_path.stem.split("_")
        singer_id = parts[0] if parts else wav_path.stem

        asset_id = f"mir1k:{wav_path.stem}"
        records.append(
            AssetRecord(
                asset_id=asset_id,
                dataset="mir1k",
                exercise_type=_MIR1K_EXERCISE_TYPE,
                singer_id=singer_id,
                technique="mixed_voice",
                note_name="",
                audio_path_abs=str(wav_path.resolve()),
                audio_path_rel=_rel(wav_path),
                channels=channels,
                sample_rate=sr,
                duration_s=dur,
                has_ground_truth_f0=has_gt,
                ground_truth_f0_path=_rel(pv_path) if has_gt else "",
            )
        )

    print(f"[scan] MIR-1K: {len(records)} assets")
    return records


# ---------------------------------------------------------------------------
# GTSinger scanner
# ---------------------------------------------------------------------------

def scan_gtsinger(root: Path) -> list[AssetRecord]:
    """Scan ml/data/raw/gtsinger/English/ and classify files.

    Directory layout:
        English/{singer_id}/{technique}/{song_title}/{group}/{index}.wav
    """
    records: list[AssetRecord] = []
    if not root.exists():
        print(f"[scan] GTSinger root not found, skipping: {root}")
        return records

    for singer_dir in sorted(root.iterdir()):
        if not singer_dir.is_dir():
            continue
        singer_id = singer_dir.name

        for technique_dir in sorted(singer_dir.iterdir()):
            if not technique_dir.is_dir():
                continue
            technique = technique_dir.name

            if technique in _GTSINGER_EXCLUDE:
                continue
            exercise_type = _GTSINGER_TECHNIQUE_MAP.get(technique)
            if exercise_type is None:
                continue

            for song_dir in sorted(technique_dir.iterdir()):
                if not song_dir.is_dir():
                    continue

                for group_dir in sorted(song_dir.iterdir()):
                    if not group_dir.is_dir():
                        continue
                    group_name = group_dir.name
                    # Skip speech groups by default
                    if "speech" in group_name.lower():
                        continue

                    for wav_path in sorted(group_dir.glob("*.wav")):
                        meta = _wav_meta(wav_path)
                        if meta is None:
                            continue
                        channels, sr, dur = meta

                        asset_id = (
                            f"gtsinger:{singer_id}:{technique}:{song_dir.name}:{group_name}:{wav_path.stem}"
                        )
                        records.append(
                            AssetRecord(
                                asset_id=asset_id,
                                dataset="gtsinger",
                                exercise_type=exercise_type,
                                singer_id=singer_id,
                                technique=technique,
                                note_name="",
                                audio_path_abs=str(wav_path.resolve()),
                                audio_path_rel=_rel(wav_path),
                                channels=channels,
                                sample_rate=sr,
                                duration_s=dur,
                                has_ground_truth_f0=False,
                                ground_truth_f0_path="",
                            )
                        )

    print(f"[scan] GTSinger: {len(records)} assets")
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocalset",  type=Path, default=REPO_ROOT / "ml/data/raw/vocalset/FULL")
    parser.add_argument("--mir1k",     type=Path, default=REPO_ROOT / "data/external/mir1k/extracted/MIR-1K")
    parser.add_argument("--gtsinger",  type=Path, default=REPO_ROOT / "ml/data/raw/gtsinger/English")
    parser.add_argument("--output",    type=Path, default=REPO_ROOT / "data/reference_catalog/raw_manifest.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[AssetRecord] = []
    records.extend(scan_vocalset(args.vocalset))
    records.extend(scan_mir1k(args.mir1k))
    records.extend(scan_gtsinger(args.gtsinger))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    print(f"\n[scan] Total: {len(records)} assets → {args.output}")

    # Print summary by dataset and exercise type
    from collections import Counter
    by_type: Counter = Counter()
    by_dataset: Counter = Counter()
    for rec in records:
        by_type[rec.exercise_type] += 1
        by_dataset[rec.dataset] += 1
    print("\nBy exercise type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")
    print("\nBy dataset:")
    for k, v in sorted(by_dataset.items()):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
