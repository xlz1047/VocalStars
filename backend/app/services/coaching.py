from typing import Any
from ml.coaching_engine.coach import build_coaching_outline
from app.core.database import SessionLocal
from app.models.analysis import AnalysisResult


def generate_recommendations(session_id: int | str, analysis_summary: dict[str, Any] | None = None) -> list[dict]:
    """Generate coaching recommendations. If `analysis_summary` is provided, use it; otherwise try to load from DB."""
    summary = analysis_summary
    if summary is None:
        db = SessionLocal()
        try:
            existing = db.query(AnalysisResult).filter(AnalysisResult.session_id == int(session_id)).first()
            if existing:
                summary = existing.summary
        finally:
            db.close()

    # If we still don't have a summary, return conservative generic recommendations.
    if not summary:
        return [
            {
                "category": "Pitch Control",
                "details": {
                    "focus": "Work on maintaining steady pitch through held notes.",
                    "exercise": "Try a 5-note siren with a slow sustain, focusing on pitch consistency.",
                },
            },
            {
                "category": "Rhythm",
                "details": {
                    "focus": "Sync your phrases with a simple metronome pattern.",
                    "exercise": "Practice singing short melodies on quarter notes at 80 BPM.",
                },
            },
        ]

    # Map analysis summary into coaching outline via the ML coaching engine
    outline = build_coaching_outline(
        pitch=summary.get("pitch", {}),
        rhythm=summary.get("rhythm", {}),
        breath=summary.get("breath", {}),
        spectral=summary.get("spectral", {}),
    )

    # Convert outline into list of recommendation dicts
    recs = []
    for area, ex in zip(outline.get("focus_areas", []), outline.get("suggested_exercises", [])):
        recs.append({"category": area, "details": {"exercise": ex}})
    return recs
