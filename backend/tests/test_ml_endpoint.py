"""Integration tests for POST /api/audio/analyze-with-ml."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path so ml_new is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.main import app  # noqa: E402

TEST_WAV = Path(__file__).parent.parent / "audio_uploads" / "3fb429c74d3248a2b3b80571cef355e6.wav"

client = TestClient(app)


def _fake_coaching_result():
    return SimpleNamespace(
        score=75,
        full_song_score=75,
        diagnostic_score=None,
        score_status="full_song_score_available_no_reference_melody",
        score_caveat="test caveat",
        summary="fake summary",
        issues=[],
        exercises=[],
        pitch_accuracy=0.75,
        pitch_drift_cents=0.0,
        phrase_lengths_s=[1.0],
        breath_count=0,
        onset_count=0,
        onset_clarity=0.0,
        technique="unknown",
        technique_confidence=0.0,
        all_technique_scores={},
        notes=[],
        voice_quality=None,
        vibrato_stats={},
        diagnostics={"source": "test"},
        analysis_validity={
            "is_analyzable": True,
            "input_type": "analyzable_singing",
            "confidence": 1.0,
            "reason_codes": ["test"],
            "summary_metrics": {},
        },
        task_config={
            "task_type": "free_singing",
            "skill_focus": None,
            "target": None,
            "reference": None,
            "scoring_mode": "auto",
            "strictness": None,
        },
        task_analysis={
            "task_type": "free_singing",
            "status": "free_singing_general_feedback",
            "caveats": ["test caveat"],
        },
        pitch_hz=np.array([0.0], dtype=np.float32),
        voiced=np.array([False]),
        breath_frames=np.array([False]),
        onset_frames=np.array([False]),
        hop_s=0.01,
    )


def test_wrong_content_type_rejected():
    """Non-audio uploads must be rejected with 400."""
    response = client.post(
        "/api/audio/analyze-with-ml",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_analyze_reports_fallback_debug_by_default(monkeypatch):
    """Default analyze-with-ml path should disclose fallback mode."""
    monkeypatch.setattr(
        "app.services.ml_inference.analyse_recording",
        lambda *args, **kwargs: _fake_coaching_result(),
    )

    response = client.post(
        "/api/audio/analyze-with-ml",
        files={"file": ("test.wav", b"fake wav bytes", "audio/wav")},
        data={"song_title": "Debug Song", "artist": "Debug Artist"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["analysisValidity"]["input_type"] == "analyzable_singing"
    assert body["data"]["fullSongScore"] == 75
    assert body["data"]["diagnosticScore"] is None
    assert body["data"]["scoreStatus"] == "full_song_score_available_no_reference_melody"
    assert body["data"]["taskConfig"]["task_type"] == "free_singing"
    assert body["data"]["taskAnalysis"]["status"] == "free_singing_general_feedback"
    assert body["debug"] == {
        "inference_mode": "fallback",
        "checkpoint_path_used": None,
        "device_used": "cpu",
        "model_stack_used": "ml_new",
    }


def test_analyze_reports_checkpoint_debug_when_checkpoint_exists(monkeypatch, tmp_path):
    """Supplying an existing checkpoint path should disclose checkpoint mode."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not used because inference is mocked")
    seen = {}

    def fake_analyse_recording(*args, **kwargs):
        seen["checkpoint"] = kwargs.get("checkpoint")
        seen["task_config"] = kwargs.get("task_config")
        return _fake_coaching_result()

    monkeypatch.setattr(
        "app.services.ml_inference.analyse_recording",
        fake_analyse_recording,
    )

    response = client.post(
        f"/api/audio/analyze-with-ml?checkpoint_path={checkpoint}",
        files={"file": ("test.wav", b"fake wav bytes", "audio/wav")},
        data={
            "song_title": "Debug Song",
            "artist": "Debug Artist",
            "task_config": json.dumps({"task_type": "pitch_slide"}),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["analysisValidity"]["input_type"] == "analyzable_singing"
    assert body["data"]["fullSongScore"] == 75
    assert body["data"]["diagnosticScore"] is None
    assert body["data"]["scoreStatus"] == "full_song_score_available_no_reference_melody"
    assert body["data"]["taskConfig"]["task_type"] == "free_singing"
    assert body["debug"] == {
        "inference_mode": "checkpoint",
        "checkpoint_path_used": str(checkpoint),
        "device_used": "cpu",
        "model_stack_used": "ml_new",
    }
    assert seen["checkpoint"] == checkpoint
    assert seen["task_config"] == {"task_type": "pitch_slide"}


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test WAV file not present in audio_uploads")
def test_analyze_returns_success_shape():
    """Successful analysis returns status=success with expected fields."""
    with open(TEST_WAV, "rb") as f:
        response = client.post(
            "/api/audio/analyze-with-ml",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"song_title": "Test Song", "artist": "Test Artist"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success", f"ML inference failed: {body.get('error')}"
    data = body["data"]
    for key in ("pitchAccuracy", "score", "issues", "exercises", "technique"):
        assert key in data, f"Missing field: {key}"


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test WAV file not present in audio_uploads")
def test_song_title_comes_from_form_not_query():
    """song_title and artist must be read from FormData, not URL query params."""
    with open(TEST_WAV, "rb") as f:
        response = client.post(
            "/api/audio/analyze-with-ml",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"song_title": "My Song", "artist": "Me"},
        )
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["songTitle"] == "My Song"
    assert body["data"]["artist"] == "Me"


@pytest.mark.skipif(not TEST_WAV.exists(), reason="test WAV file not present in audio_uploads")
def test_score_in_valid_range():
    """Overall score must be 0–100."""
    with open(TEST_WAV, "rb") as f:
        response = client.post(
            "/api/audio/analyze-with-ml",
            files={"file": ("test.wav", f, "audio/wav")},
            data={"song_title": "Range Check", "artist": "Tester"},
        )
    data = response.json()["data"]
    if data["score"] is not None:
        assert 0 <= data["score"] <= 100
    if data["fullSongScore"] is not None:
        assert 0 <= data["fullSongScore"] <= 100
    if data["diagnosticScore"] is not None:
        assert 0 <= data["diagnosticScore"] <= 100
    assert data["scoreStatus"]
    assert 0.0 <= data["pitchAccuracy"] <= 100.0
