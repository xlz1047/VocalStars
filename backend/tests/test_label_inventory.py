from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.eval.label_inventory import build_inventory


def test_label_inventory_counts_missing_files_and_label_support(tmp_path: Path) -> None:
    existing_npz = tmp_path / "clip.npz"
    existing_audio = tmp_path / "clip.wav"
    existing_nano = tmp_path / "clip_nanopitch.npz"
    label_json = tmp_path / "clip.json"
    for path in (existing_npz, existing_audio, existing_nano):
        path.write_bytes(b"ok")
    label_json.write_text(
        json.dumps(
            [
                {
                    "ph": ["AA", "EH"],
                    "ph_start": [0.0, 0.5],
                    "ph_end": [0.5, 1.0],
                    "vibrato": ["1", "0"],
                    "glissando": ["0", "1"],
                    "mix": ["0", "0"],
                    "falsetto": ["0", "0"],
                    "breathy": ["0", "0"],
                    "pharyngeal": ["0", "0"],
                }
            ]
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "npz_path",
                "audio_path",
                "dataset",
                "singer_id",
                "split",
                "source_technique",
                "supervision",
                "json_path",
                "song_key",
                "n_frames",
                "nanopitch_npz",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "npz_path": str(existing_npz),
                "audio_path": str(existing_audio),
                "dataset": "gtsinger",
                "singer_id": "singer_a",
                "split": "train",
                "source_technique": "vibrato",
                "supervision": "phoneme_multilabel_technique",
                "json_path": str(label_json),
                "song_key": "song_a",
                "n_frames": "100",
                "nanopitch_npz": str(existing_nano),
            }
        )
        writer.writerow(
            {
                "npz_path": "missing.npz",
                "audio_path": "missing.wav",
                "dataset": "gtsinger",
                "singer_id": "singer_b",
                "split": "test_alto1",
                "source_technique": "glissando",
                "supervision": "phoneme_multilabel_technique",
                "json_path": "missing.json",
                "song_key": "song_b",
                "n_frames": "50",
                "nanopitch_npz": "missing_nano.npz",
            }
        )

    payload = build_inventory(manifest, root=tmp_path)
    assert payload["rows"] == 2
    assert payload["counts"]["split"]["train"] == 1
    assert payload["counts"]["split"]["test_alto1"] == 1
    assert payload["missing_files"]["npz_path"] == 1
    assert payload["missing_files"]["audio_path"] == 1
    assert payload["missing_files"]["json_path"] == 1
    assert payload["missing_files"]["nanopitch_npz"] == 1
    assert payload["label_support"]["techniques"]["vibrato"]["positive_duration_s"] == 0.5
    assert payload["label_support"]["techniques"]["glissando"]["positive_duration_s"] == 0.5
