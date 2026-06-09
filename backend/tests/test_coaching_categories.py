from __future__ import annotations

import math

from scripts.eval.coaching_categories import build_coaching_categories


def _frames_from_f0(values: list[float | None]) -> list[dict]:
    frames = []
    for idx, value in enumerate(values):
        frames.append(
            {
                "time_s": round(idx * 0.01, 4),
                "frame_index": idx,
                "f0_hz": value,
                "voiced": value is not None,
                "pitch_confidence": 0.92 if value is not None else 0.0,
                "voice_confidence": 0.90 if value is not None else 0.0,
                "signal_quality": {"near_silence": value is None, "low_confidence": False},
            }
        )
    return frames


def test_vibrato_category_detects_synthetic_regular_vibrato() -> None:
    sr = 100
    base = 220.0
    values = []
    for idx in range(180):
        t = idx / sr
        cents = 55.0 * math.sin(2.0 * math.pi * 5.8 * t)
        values.append(base * (2.0 ** (cents / 1200.0)))
    categories = build_coaching_categories({"frames": _frames_from_f0(values), "analysis_validity": {"status": "valid"}})
    vibrato = categories["vibrato"]
    assert vibrato["status"] == "complete"
    assert vibrato["score"] >= 60
    assert 5.0 <= vibrato["metrics"]["mean_rate_hz"] <= 6.5
    assert vibrato["metrics"]["mean_extent_cents"] >= 25


def test_slide_category_detects_synthetic_up_slide() -> None:
    values = [220.0 * (2.0 ** ((idx * 500.0 / 119.0) / 1200.0)) for idx in range(120)]
    categories = build_coaching_categories({"frames": _frames_from_f0(values), "analysis_validity": {"status": "valid"}})
    slide = categories["slide"]
    assert slide["status"] == "complete"
    assert slide["metrics"]["direction"] == "up"
    assert slide["metrics"]["pitch_range_cents"] >= 400
    assert slide["metrics"]["smoothness"] >= 0.8


def test_categories_abstain_on_invalid_input() -> None:
    categories = build_coaching_categories(
        {
            "frames": _frames_from_f0([None] * 100),
            "analysis_validity": {"status": "invalid", "invalid_type": "no_voice_or_noise"},
        }
    )
    assert categories["vibrato"]["status"] == "not_enough_evidence"
    assert categories["slide"]["status"] == "not_enough_evidence"
    assert categories["vibrato"]["score"] is None
    assert categories["slide"]["score"] is None
