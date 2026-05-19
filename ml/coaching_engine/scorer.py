"""Scoring logic: converts raw feature dicts into normalised 0-1 skill scores."""

from typing import Any


def score_features(
    pitch_result: dict[str, Any],
    rhythm_result: dict[str, Any],
    breath_result: dict[str, Any],
) -> dict[str, float]:
    """Convert raw feature dicts into normalised 0–1 skill scores.

    Args:
        pitch_result: Output of PitchDetector.analyze or extract_pitch_features.
        rhythm_result: Output of analyze_rhythm.
        breath_result: Output of BreathDetector.analyze or analyze_breath.

    Returns:
        Dict with keys: pitch_score, rhythm_score, breath_score, overall_score.
        All values are floats in [0, 1].
    """
    pitch_score = _score_pitch(pitch_result)
    rhythm_score = _score_rhythm(rhythm_result)
    breath_score = _score_breath(breath_result)
    overall_score = round((pitch_score + rhythm_score + breath_score) / 3.0, 4)

    return {
        "pitch_score": pitch_score,
        "rhythm_score": rhythm_score,
        "breath_score": breath_score,
        "overall_score": overall_score,
    }


def _score_pitch(pitch_result: dict[str, Any]) -> float:
    raw = pitch_result.get("stability_score")
    if raw is None:
        return 0.5
    return round(float(max(0.0, min(1.0, raw))), 4)


def _score_rhythm(rhythm_result: dict[str, Any]) -> float:
    # rhythm_detector.py returns: tempo, timing_variance, beat_alignment
    # Derive a proxy score from timing_variance when available (lower = better)
    variance = rhythm_result.get("timing_variance")
    if variance is None:
        return 0.5
    # Clamp variance to [0, 1] range; invert so low variance = high score
    return round(float(max(0.0, min(1.0, 1.0 - variance))), 4)


def _score_breath(breath_result: dict[str, Any]) -> float:
    raw = breath_result.get("support_score")
    if raw is None:
        return 0.5
    return round(float(max(0.0, min(1.0, raw))), 4)
