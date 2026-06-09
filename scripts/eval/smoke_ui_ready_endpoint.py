#!/usr/bin/env python3
"""Smoke-test /api/audio/analyze-with-ml UI-ready task-aware responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app  # noqa: E402


CASES: list[dict[str, Any]] = [
    {
        "path": REPO_ROOT / "samples" / "00_silence.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
        "input_type": "no_voice_or_noise",
        "status": "no_voice_or_noise_no_task_score",
        "full_song_score": None,
        "diagnostic_score": None,
    },
    {
        "path": REPO_ROOT / "samples" / "01_speaking_voice.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
        "input_type": "speech_like_or_non_singing",
        "status": "speech_like_or_non_singing_no_task_score",
        "full_song_score": None,
        "diagnostic_score": None,
    },
    {
        "path": REPO_ROOT / "samples" / "03_sustained_aaa.wav",
        "task_config": {
            "task_type": "sustained_note",
            "target": {"note": "C4", "f0_hz": 261.63},
            "expected_duration": 5,
            "skill_focus": ["pitch_stability"],
        },
        "input_type": "diagnostic_sustained_tone",
        "status": "diagnostic_sustained_note_complete",
        "full_song_score": None,
        "diagnostic_score": "present",
    },
    {
        "path": REPO_ROOT / "samples" / "04_pitch_slide.wav",
        "task_config": {
            "task_type": "pitch_slide",
            "expected_direction": "up",
            "expected_duration": 5,
            "skill_focus": ["slide_smoothness"],
        },
        "input_type": "diagnostic_pitch_slide",
        "status": "diagnostic_pitch_slide_complete",
        "full_song_score": None,
        "diagnostic_score": "present",
    },
    {
        "path": REPO_ROOT / "samples" / "05_twinkle_twinkle.wav",
        "task_config": {"task_type": "free_singing", "scoring_mode": "no_reference"},
        "input_type": "analyzable_singing",
        "status": "free_singing_general_feedback",
        "full_song_score": "present",
        "diagnostic_score": None,
    },
    {
        "path": REPO_ROOT / "samples" / "05_twinkle_twinkle.wav",
        "task_config": {
            "task_type": "reference_song",
            "reference": {
                "type": "midi_note_sequence",
                "title": "Twinkle opening smoke",
                "notes": ["C4", "C4", "G4", "G4", "A4", "A4", "G4"],
                "f0_hz": [261.63, 261.63, 392.0, 392.0, 440.0, 440.0, 392.0],
                "durations_s": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
            },
            "strictness": "beginner",
        },
        "input_type": "analyzable_singing",
        "status": "provisional_reference_contour_complete",
        "full_song_score": None,
        "diagnostic_score": "present",
        "requires_reference_targets": True,
    },
]


def assert_score(value: Any, expectation: Any, label: str) -> None:
    if expectation == "present":
        assert isinstance(value, (int, float)), f"{label} should be numeric, got {value!r}"
        return
    assert value == expectation, f"{label} expected {expectation!r}, got {value!r}"


def main() -> int:
    client = TestClient(app)
    for case in CASES:
        path = Path(case["path"])
        with path.open("rb") as fh:
            response = client.post(
                "/api/audio/analyze-with-ml?response_mode=ui_ready",
                files={"file": (path.name, fh, "audio/wav")},
                data={
                    "song_title": "Endpoint Smoke",
                    "artist": "VocalStars",
                    "task_config": json.dumps(case["task_config"]),
                    "include_frames": "true",
                    "debug": "false",
                },
            )
        assert response.status_code == 200, f"{path.name}: HTTP {response.status_code}"
        body = response.json()
        assert body.get("status") == "success", f"{path.name}: {body}"
        analysis = body.get("uiReadyAnalysis") or (body.get("data") or {}).get("uiReadyAnalysis")
        assert analysis, f"{path.name}: missing uiReadyAnalysis"
        validity = analysis.get("analysis_validity") or {}
        task_result = analysis.get("task_result") or {}
        feedback_policy = analysis.get("feedback_policy") or {}

        assert validity.get("input_type") == case["input_type"], path.name
        assert task_result.get("status") == case["status"], path.name
        assert_score(task_result.get("full_song_score"), case["full_song_score"], f"{path.name} full_song_score")
        assert_score(task_result.get("diagnostic_score"), case["diagnostic_score"], f"{path.name} diagnostic_score")
        assert "blocked_feedback" in feedback_policy, f"{path.name}: missing blocked_feedback"
        assert "caveats" in feedback_policy, f"{path.name}: missing caveats"
        assert analysis.get("frames"), f"{path.name}: missing frames"
        assert analysis.get("segments") is not None, f"{path.name}: missing segments"
        source_strategy = analysis.get("source_strategy") or {}
        expected_task_type = case["task_config"].get("task_type")
        assert source_strategy.get("task_type") == expected_task_type, (
            f"{path.name}: source strategy task_type expected {expected_task_type!r}, "
            f"got {source_strategy.get('task_type')!r}"
        )
        if case.get("requires_reference_targets"):
            frames = analysis.get("frames") or []
            assert any(frame.get("target_f0_hz") for frame in frames), f"{path.name}: missing target_f0_hz frames"
            assert any(frame.get("cents_error") is not None for frame in frames), f"{path.name}: missing cents_error frames"
            alignment = analysis.get("reference_alignment") or (analysis.get("subscores") or {}).get("reference_alignment") or {}
            assert alignment.get("method") == "voiced_span_linear_time_warp", f"{path.name}: missing reference alignment"
            assert isinstance(alignment.get("tempo_scale"), (int, float)), f"{path.name}: missing tempo_scale"
            reference_regions = ((analysis.get("segments") or {}).get("reference_pitch_error_regions") or [])
            assert isinstance(reference_regions, list), f"{path.name}: reference pitch error regions should be a list"
        visualizations = analysis.get("visualizations") or {}
        spectrogram = visualizations.get("spectrogram") or {}
        posteriorgram = visualizations.get("posteriorgram") or {}
        assert spectrogram.get("kind") == "log_mel_spectrogram", f"{path.name}: missing log-mel spectrogram"
        assert spectrogram.get("values"), f"{path.name}: missing spectrogram values"
        assert posteriorgram.get("kind") == "posterior_confidence_summary", f"{path.name}: missing posteriorgram"
        assert posteriorgram.get("values"), f"{path.name}: missing posteriorgram values"
        print(
            "PASS",
            path.name,
            validity.get("input_type"),
            task_result.get("status"),
            task_result.get("full_song_score"),
            task_result.get("diagnostic_score"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
