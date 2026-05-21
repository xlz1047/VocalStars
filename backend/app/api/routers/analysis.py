from fastapi import APIRouter
from app.services.analysis import run_analysis_pipeline
from app.core.database import SessionLocal
from app.models.analysis import AnalysisResult

router = APIRouter()


@router.post("/run")
async def run_analysis(session_id: int):
    # Try to return a persisted analysis if available
    db = SessionLocal()
    try:
        existing = db.query(AnalysisResult).filter(AnalysisResult.session_id == session_id).first()
        if existing:
            return {"session_id": session_id, "report": existing.summary}
    finally:
        db.close()

    # Fallback to on-demand analysis (may be slow)
    report = run_analysis_pipeline(session_id)
    return {"session_id": session_id, "report": report}
