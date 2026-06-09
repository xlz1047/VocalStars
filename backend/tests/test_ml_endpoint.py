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
GTSINGER_ROOT = _REPO_ROOT / "ml" / "data" / "raw" / "gtsinger" / "English"


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


def test_analyze_ui_ready_response_mode_includes_contract(monkeypatch, tmp_path):
    """UI-ready mode should include predictable uiReadyAnalysis fields."""
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not used because inference is mocked")
    seen = {}

    def fake_analyse_recording(*args, **kwargs):
        seen["checkpoint"] = kwargs.get("checkpoint")
        seen["task_config"] = kwargs.get("task_config")
        return _fake_coaching_result()

    def fake_build_ui_ready_response(*args, **kwargs):
        seen["ui_task_config"] = kwargs.get("task_config")
        seen["include_frames"] = kwargs.get("include_frames")
        seen["debug"] = kwargs.get("debug")
        return {
            "schema_version": "test.ui_ready.v1",
            "task_config": kwargs.get("task_config"),
            "analysis_validity": {
                "is_analyzable": True,
                "input_type": "analyzable_singing",
            },
            "frames": [{"time_s": 0.0, "frame_index": 0, "f0_hz": 440.0, "voiced": True}],
            "segments": {"notes": [], "phrases": [], "dropouts": []},
            "task_result": {
                "task_type": "sustained_note",
                "status": "diagnostic_sustained_note_complete",
                "full_song_score": None,
                "diagnostic_score": 80,
                "summary": "ok",
            },
            "subscores": {"pitch_stability": 0.8},
            "feedback_policy": {
                "allowed_feedback": ["pitch_stability"],
                "blocked_feedback": [{"type": "full_song_score", "reason": "diagnostic"}],
                "caveats": ["Diagnostic only."],
            },
            "source_strategy": {"selected_f0_source_recommendation": "pyin"},
            "caveats": ["Diagnostic only."],
        }

    monkeypatch.setattr(
        "app.services.ml_inference.analyse_recording",
        fake_analyse_recording,
    )
    monkeypatch.setattr(
        "app.api.routers.audio_processing.build_ui_ready_response",
        fake_build_ui_ready_response,
    )

    task_config = {"task_type": "sustained_note", "target": {"note": "C4", "f0_hz": 261.63}}
    response = client.post(
        f"/api/audio/analyze-with-ml?response_mode=ui_ready&checkpoint_path={checkpoint}",
        files={"file": ("test.wav", b"fake wav bytes", "audio/wav")},
        data={
            "song_title": "Debug Song",
            "artist": "Debug Artist",
            "task_config": json.dumps(task_config),
            "include_frames": "true",
            "debug": "false",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["uiReadyAnalysis"]["task_result"]["status"] == "diagnostic_sustained_note_complete"
    assert body["data"]["uiReadyAnalysis"]["feedback_policy"]["blocked_feedback"][0]["type"] == "full_song_score"
    assert body["data"]["ui_ready_analysis"]["feedback_policy"]["caveats"] == ["Diagnostic only."]
    assert seen["checkpoint"] == checkpoint
    assert seen["task_config"] == task_config
    assert seen["ui_task_config"] == task_config
    assert seen["include_frames"] is True
    assert seen["debug"] is False


def test_include_ui_ready_analysis_form_flag_is_supported(monkeypatch):
    """Frontend form flag should trigger UI-ready response without query params."""
    monkeypatch.setattr(
        "app.services.ml_inference.analyse_recording",
        lambda *args, **kwargs: _fake_coaching_result(),
    )
    monkeypatch.setattr(
        "app.api.routers.audio_processing.build_ui_ready_response",
        lambda *args, **kwargs: {
            "schema_version": "test.ui_ready.v1",
            "task_config": kwargs.get("task_config"),
            "analysis_validity": {"input_type": "analyzable_singing"},
            "frames": [],
            "segments": {},
            "task_result": {"task_type": "free_singing", "status": "free_singing_general_feedback"},
            "subscores": {},
            "feedback_policy": {"allowed_feedback": [], "blocked_feedback": [], "caveats": []},
            "source_strategy": {},
            "caveats": [],
        },
    )

    response = client.post(
        "/api/audio/analyze-with-ml",
        files={"file": ("test.wav", b"fake wav bytes", "audio/wav")},
        data={
            "include_ui_ready_analysis": "true",
            "task_config": json.dumps({"task_type": "free_singing"}),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["uiReadyAnalysis"]["schema_version"] == "test.ui_ready.v1"
    assert body["data"]["uiReadyAnalysis"]["task_config"] == {"task_type": "free_singing"}


@pytest.mark.skipif(not GTSINGER_ROOT.exists(), reason="GTSinger dataset is not present")
def test_gtsinger_catalog_hides_paired_speech_by_default():
    response = client.get("/api/audio/gtsinger-catalog?limit=20")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "gtsinger_catalog.v1"
    assert body["default_group_policy"] == "sung_only"
    assert body["songs"], "Expected at least one discovered GTSinger song"

    for song in body["songs"]:
        assert song["phrases"], f"Expected phrases for {song['id']}"
        assert song["default_group"] != "Paired_Speech_Group"
        assert not any(phrase["is_speech"] for phrase in song["phrases"])
        assert song["default_audio_url"].startswith("/api/audio/file?path=ml/data/raw/gtsinger/")


@pytest.mark.skipif(not GTSINGER_ROOT.exists(), reason="GTSinger dataset is not present")
def test_gtsinger_catalog_can_include_paired_speech_for_dev_use():
    response = client.get("/api/audio/gtsinger-catalog?include_speech=true&limit=50")

    assert response.status_code == 200
    body = response.json()
    assert body["songs"], "Expected at least one discovered GTSinger song"
    assert any(
        phrase["is_speech"]
        for song in body["songs"]
        for phrase in song["phrases"]
    )


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
