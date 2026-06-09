"""Feedback generator: maps skill scores to actionable coaching strings."""

from typing import Any


def build_feedback(scores: dict[str, float]) -> dict[str, Any]:
    """Map skill scores to an actionable coaching outline.

    Args:
        scores: Output of score_features — dict with pitch_score, rhythm_score,
            breath_score, overall_score (all floats in [0, 1]).

    Returns:
        Dict with keys: summary, focus_areas, suggested_exercises.
    """
    focus_areas: list[str] = []
    exercises: list[str] = []

    if scores.get("pitch_score", 1.0) < 0.7:
        focus_areas.append("pitch stability")
        exercises.append("Slow sustained scales to improve pitch stability.")

    if scores.get("rhythm_score", 1.0) < 0.7:
        focus_areas.append("rhythm support")
        exercises.append("Practice with a metronome for rhythm consistency.")

    if scores.get("breath_score", 1.0) < 0.7:
        focus_areas.append("breath control")
        exercises.append("Diaphragmatic breathing — inhale 4 beats, exhale 8 beats.")

    focus_areas.append("note transitions")

    if not exercises:
        exercises.append("Maintain current practice — focus on consistency.")

    overall = scores.get("overall_score", 0.5)
    summary = (
        f"Overall performance score: {overall:.0%}. "
        + ("Great foundation — keep refining." if overall >= 0.7 else "Focus on the areas below to improve.")
    )

    return {
        "summary": summary,
        "focus_areas": focus_areas,
        "suggested_exercises": exercises,
    }


def build_coaching_outline(
    pitch: dict[str, Any],
    rhythm: dict[str, Any],
    breath: dict[str, Any],
    spectral: dict[str, Any],
) -> dict[str, Any]:
    """Compatibility shim for pipeline.py — scores features then delegates to build_feedback.

    Args:
        pitch: Output of extract_pitch_features.
        rhythm: Output of analyze_rhythm.
        breath: Output of analyze_breath.
        spectral: Output of extract_spectral_features (currently unused in scoring).

    Returns:
        Dict with keys: summary, focus_areas, suggested_exercises.
    """
    from ml_new.legacy_ml.coaching_engine.scorer import score_features

    scores = score_features(pitch, rhythm, breath)
    return build_feedback(scores)
