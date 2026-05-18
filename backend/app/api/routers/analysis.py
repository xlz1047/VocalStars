from fastapi import APIRouter
from app.services.analysis import run_analysis_pipeline

router = APIRouter()


@router.post("/run")
async def run_analysis(session_id: str):
    report = run_analysis_pipeline(session_id)
    return {"session_id": session_id, "report": report}
