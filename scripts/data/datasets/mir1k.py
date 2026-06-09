"""MIR-1K annotation adapter for normalized human-singing manifests.

The adapter expects a locally downloaded/extracted MIR-1K root. It does not
download files. It normalizes the frame-level pitch/voicing labels used for
report-only VAD/f0 evaluation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any


PITCH_HOP_S = 0.02
PITCH_WINDOW_S = 0.04


class MIR1KAdapterError(RuntimeError):
    """Raised when MIR-1K annotations are missing or unsupported."""


def enrich_record(record: Any, root: Path, audio_path: Path) -> Any:
    """Return a manifest record with MIR-1K frame annotations populated."""
    pitch_path = find_sidecar(
        root,
        audio_path,
        ("PitchLabel", "pitchlabel", "pitch"),
        (".pv", ".txt", ".csv", ".f0", ".pitch"),
    )
    if pitch_path is None:
        raise MIR1KAdapterError(
            f"MIR-1K pitch label not found for {audio_path}. Expected a matching file under PitchLabel/."
        )

    pitch_info = load_pitch_label(pitch_path)
    lyric_path = find_sidecar(root, audio_path, ("Lyrics", "lyrics", "Lyric"), (".txt", ".lab", ".csv"))
    lyrics = load_text(lyric_path) if lyric_path else None
    singer_id, song_id, clip_id = parse_clip_id(audio_path.stem)
    sample_rate = read_sample_rate(audio_path)

    source_metadata = dict(record.source_metadata)
    source_metadata.update(
        {
            "mir1k_pitch_label_path": str(pitch_path),
            "mir1k_pitch_label_unit": pitch_info["unit"],
            "mir1k_pitch_frame_hop_s": PITCH_HOP_S,
            "mir1k_pitch_frame_window_s": PITCH_WINDOW_S,
            "mir1k_lyrics_path": str(lyric_path) if lyric_path else None,
            "mir1k_clip_id": clip_id,
        }
    )
    if lyrics:
        source_metadata["lyrics_text"] = lyrics

    return replace(
        record,
        sample_rate=sample_rate,
        task_tags=sorted(set(record.task_tags + ["vad", "f0_ground_truth", "mir1k"])),
        singer_id=singer_id or record.singer_id,
        song_or_exercise_id=song_id or record.song_or_exercise_id,
        frame_annotations={
            "time_s": pitch_info["time_s"],
            "f0_hz": pitch_info["f0_hz"],
            "voiced": pitch_info["voiced"],
            "phoneme_or_lyric": None,
            "status": "loaded_from_mir1k_pitch_label",
            "source_path": str(pitch_path),
        },
        reference={
            "type": "lyrics_alignment",
            "path": str(lyric_path) if lyric_path else None,
            "status": "lyrics_loaded_without_note_timing" if lyric_path else "pitch_only_no_lyrics_sidecar",
        },
        license_review={
            "status": "restricted",
            "notes": (
                "MIR-1K is used here only as a local, report-only evaluation dataset. "
                "Confirm download terms and citation requirements before any product or training use."
            ),
        },
        source_metadata=source_metadata,
    )


def find_sidecar(root: Path, audio_path: Path, preferred_dirs: tuple[str, ...], suffixes: tuple[str, ...]) -> Path | None:
    stem = audio_path.stem
    candidates: list[Path] = []
    for directory in preferred_dirs:
        base = root / directory
        if base.exists():
            for suffix in suffixes:
                candidates.append(base / f"{stem}{suffix}")
    for suffix in suffixes:
        candidates.append(audio_path.with_suffix(suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    escaped = re.escape(stem).lower()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if re.fullmatch(escaped, path.stem.lower()):
            return path
    return None


def load_pitch_label(path: Path) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(item) for item in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", line)]
        if values:
            rows.append(values)
    if not rows:
        raise MIR1KAdapterError(f"Pitch label has no numeric frames: {path}")

    if all(len(row) >= 2 for row in rows):
        times = [row[0] for row in rows]
        pitch_values = [row[-1] for row in rows]
        if not looks_like_time_axis(times):
            times = [i * PITCH_HOP_S for i in range(len(rows))]
    else:
        times = [i * PITCH_HOP_S for i in range(len(rows))]
        pitch_values = [row[0] for row in rows]

    f0_hz, unit = pitch_values_to_hz(pitch_values)
    voiced = [bool(value and math.isfinite(value) and value > 0.0) for value in f0_hz]
    return {
        "time_s": [round(float(value), 6) for value in times],
        "f0_hz": [round(float(value), 6) if value > 0.0 else 0.0 for value in f0_hz],
        "voiced": voiced,
        "unit": unit,
    }


def pitch_values_to_hz(values: list[float]) -> tuple[list[float], str]:
    finite_positive = [value for value in values if math.isfinite(value) and value > 0.0]
    if not finite_positive:
        return [0.0 for _ in values], "unknown_all_unvoiced"
    median = sorted(finite_positive)[len(finite_positive) // 2]
    if 20.0 <= median <= 140.0:
        return [
            440.0 * (2.0 ** ((value - 69.0) / 12.0)) if math.isfinite(value) and value > 0.0 else 0.0
            for value in values
        ], "semitone_midi_like"
    return [value if math.isfinite(value) and value > 0.0 else 0.0 for value in values], "hz"


def looks_like_time_axis(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    diffs = [b - a for a, b in zip(values[:-1], values[1:])]
    positive = [value for value in diffs if value > 0.0]
    if len(positive) < len(diffs) * 0.8:
        return False
    median = sorted(positive)[len(positive) // 2]
    return 0.001 <= median <= 0.2


def load_text(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or None


def parse_clip_id(stem: str) -> tuple[str | None, str | None, str | None]:
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], None
    return None, stem, None


def read_sample_rate(audio_path: Path) -> int | None:
    try:
        import soundfile as sf

        return int(sf.info(str(audio_path)).samplerate)
    except Exception:
        try:
            import librosa

            return int(librosa.get_samplerate(str(audio_path)))
        except Exception:
            return None


def write_setup_template(path: Path) -> None:
    """Write a local config template for MIR-1K dataset roots."""
    template = {
        "mir1k": {
            "root": "/absolute/path/to/MIR-1K",
            "license_acknowledged": False,
            "notes": "Set license_acknowledged to true after reviewing MIR-1K terms and citation requirements.",
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
