#!/usr/bin/env python3
"""Build normalized manifest stubs for human-singing datasets.

This script does not download data and does not train models. It scans a local
dataset directory and emits JSONL records in the normalized schema documented in
docs/ai-coach/DATASET_INGESTION_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.data.datasets import mir1k
except Exception:  # pragma: no cover - keeps generic manifest scans usable
    try:
        from datasets import mir1k  # type: ignore[no-redef]
    except Exception:
        mir1k = None  # type: ignore[assignment]


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".aif", ".aiff"}


DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "annotated_vocalset": {
        "split": "external_eval",
        "task_tags": ["sustained_note", "scale", "note_match", "tone_consistency"],
        "reference_type": "note_sequence",
        "license_status": "required",
        "notes": "Use f0/note annotations when annotation sidecars are available.",
    },
    "vocalset": {
        "split": "external_eval",
        "task_tags": ["sustained_note", "scale", "tone_consistency"],
        "reference_type": "none",
        "license_status": "required",
        "notes": "Raw VocalSet metadata can identify singer, vowel, and technique.",
    },
    "mir1k": {
        "split": "external_eval",
        "task_tags": ["free_singing", "vad"],
        "reference_type": "lyrics_alignment",
        "license_status": "required",
        "notes": "Normalize pitch tracks, lyrics, and vocal activity labels.",
    },
    "damp_amazing_grace": {
        "split": "external_eval",
        "task_tags": ["reference_song", "free_singing", "score_calibration"],
        "reference_type": "midi",
        "license_status": "required",
        "notes": "Use Amazing Grace MIDI/backing reference and amateur performance metadata.",
    },
    "medleydb": {
        "split": "external_eval",
        "task_tags": ["free_singing", "reference_song"],
        "reference_type": "melody_f0",
        "license_status": "restricted",
        "notes": "Use vocal stems and melody annotations; review non-commercial terms.",
    },
    "dali": {
        "split": "external_eval",
        "task_tags": ["reference_song", "phrase_practice"],
        "reference_type": "lyrics_alignment",
        "license_status": "restricted",
        "notes": "Use melody notes and lyric timing; audio linkage/license requires review.",
    },
    "tonas": {
        "split": "external_eval",
        "task_tags": ["pitch_slide", "free_singing", "note_segmentation"],
        "reference_type": "manual_melody_transcription",
        "license_status": "required",
        "notes": "Useful for a cappella expressive contours and ornaments.",
    },
    "vocadito": {
        "split": "external_eval",
        "task_tags": ["note_match", "phrase_practice", "reference_song"],
        "reference_type": "note_sequence",
        "license_status": "required",
        "notes": "Normalize solo-vocal f0, note, and lyric annotations.",
    },
    "opencpop": {
        "split": "external_eval",
        "task_tags": ["reference_song", "phrase_practice", "tone_consistency"],
        "reference_type": "note_sequence",
        "license_status": "required",
        "notes": "Normalize note, syllable, and phoneme boundary labels for future phoneme-aware evaluators.",
    },
    "generic": {
        "split": "external_eval",
        "task_tags": ["free_singing"],
        "reference_type": "none",
        "license_status": "required",
        "notes": "Generic local human singing collection.",
    },
}


@dataclass
class ManifestRecord:
    dataset: str
    split: str
    audio_path: str
    sample_rate: int | None
    task_tags: list[str]
    singer_id: str | None
    song_or_exercise_id: str | None
    frame_annotations: dict[str, Any]
    note_annotations: list[dict[str, Any]]
    reference: dict[str, Any]
    negative_labels: dict[str, bool]
    license_review: dict[str, str]
    source_metadata: dict[str, Any]


def infer_task_tags(dataset: str, path: Path, defaults: list[str]) -> list[str]:
    text = " ".join(path.parts).lower()
    tags = set(defaults)
    if any(token in text for token in ("sustain", "long", "held", "vowel")):
        tags.add("sustained_note")
    if any(token in text for token in ("scale", "arpeggio")):
        tags.add("scale")
    if any(token in text for token in ("slide", "siren", "glide", "portamento")):
        tags.add("pitch_slide")
    if any(token in text for token in ("speech", "speaking", "spoken", "talk")):
        tags.add("speech_like_or_non_singing")
    if any(token in text for token in ("noise", "silence", "backing", "instrumental")):
        tags.add("negative")
    if dataset in {"damp_amazing_grace", "dali", "medleydb", "vocadito", "opencpop"}:
        tags.add("reference_song")
    return sorted(tags)


def infer_singer_id(path: Path) -> str | None:
    for part in path.parts:
        lowered = part.lower()
        if re.fullmatch(r"(singer|vocalist|subject|user|p)\w*[-_]?\d+", lowered):
            return part
        if re.fullmatch(r"(male|female|m|f)[-_]?\d+", lowered):
            return part
    return None


def infer_negative_labels(path: Path) -> dict[str, bool]:
    text = " ".join(path.parts).lower()
    return {
        "speech": any(token in text for token in ("speech", "speaking", "spoken", "talk")),
        "noise_only": any(token in text for token in ("noise", "silence", "fan", "hum")),
        "backing_track_only": any(token in text for token in ("backing", "instrumental", "karaoke")),
    }


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def sidecar_candidates(audio_path: Path) -> list[str]:
    candidates: list[str] = []
    for suffix in (".json", ".csv", ".tsv", ".f0", ".pitch", ".mid", ".midi", ".lab", ".TextGrid"):
        sidecar = audio_path.with_suffix(suffix)
        if sidecar.exists():
            candidates.append(str(sidecar))
    return candidates


def build_record(dataset: str, root: Path, audio_path: Path) -> ManifestRecord:
    profile = DATASET_PROFILES.get(dataset, DATASET_PROFILES["generic"])
    relative_path = audio_path.relative_to(root) if audio_path.is_relative_to(root) else audio_path
    task_tags = infer_task_tags(dataset, audio_path, list(profile["task_tags"]))
    negative_labels = infer_negative_labels(audio_path)
    record = ManifestRecord(
        dataset=dataset,
        split=str(profile["split"]),
        audio_path=str(audio_path),
        sample_rate=None,
        task_tags=task_tags,
        singer_id=infer_singer_id(relative_path),
        song_or_exercise_id=audio_path.stem,
        frame_annotations={
            "time_s": None,
            "f0_hz": None,
            "voiced": None,
            "phoneme_or_lyric": None,
            "status": "not_loaded_by_manifest_builder",
        },
        note_annotations=[],
        reference={
            "type": str(profile["reference_type"]),
            "path": None,
            "status": "placeholder_until_dataset_adapter_loads_annotations",
        },
        negative_labels=negative_labels,
        license_review={
            "status": str(profile["license_status"]),
            "notes": str(profile["notes"]),
        },
        source_metadata={
            "relative_path": str(relative_path),
            "extension": audio_path.suffix.lower(),
            "sidecar_candidates": sidecar_candidates(audio_path),
        },
    )
    if dataset == "mir1k":
        if mir1k is None:
            raise RuntimeError("MIR-1K adapter could not be imported.")
        return mir1k.enrich_record(record, root, audio_path)
    return record


def write_manifest(records: list[ManifestRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_summary(records: list[ManifestRecord], output: Path) -> None:
    summary_path = output.with_suffix(".summary.json")
    by_tag: dict[str, int] = {}
    by_extension: dict[str, int] = {}
    negative_counts = {"speech": 0, "noise_only": 0, "backing_track_only": 0}
    for record in records:
        for tag in record.task_tags:
            by_tag[tag] = by_tag.get(tag, 0) + 1
        extension = str(record.source_metadata.get("extension") or "")
        by_extension[extension] = by_extension.get(extension, 0) + 1
        for key in negative_counts:
            if record.negative_labels.get(key):
                negative_counts[key] += 1
    summary = {
        "manifest": str(output),
        "record_count": len(records),
        "by_task_tag": dict(sorted(by_tag.items())),
        "by_extension": dict(sorted(by_extension.items())),
        "negative_counts": negative_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PROFILES),
        required=True,
        help="Dataset profile to use for manifest defaults.",
    )
    parser.add_argument("--root", type=Path, required=True, help="Local dataset root to scan.")
    parser.add_argument(
        "--license-acknowledged",
        action="store_true",
        help="Required for restricted dataset profiles such as MIR-1K. Confirms local research/eval use only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to data/manifests/human_singing/<dataset>.jsonl.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Skip writing summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.dataset == "mir1k" and not args.license_acknowledged:
        raise SystemExit(
            "MIR-1K manifest generation is license-gated. Review dataset terms, "
            "download/request the dataset yourself, then rerun with --license-acknowledged."
        )
    if not root.exists():
        raise SystemExit(
            f"Dataset root does not exist: {root}\n"
            "Download/request the dataset locally first, then rerun with --root pointing at the extracted directory."
        )
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")
    output = args.output or Path("data") / "manifests" / "human_singing" / f"{args.dataset}.jsonl"
    try:
        records = [build_record(args.dataset, root, path) for path in iter_audio_files(root)]
    except Exception as exc:
        raise SystemExit(f"Dataset adapter failed: {type(exc).__name__}: {exc}") from exc
    write_manifest(records, output)
    if not args.no_summary:
        write_summary(records, output)
    print(f"Wrote {len(records)} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
