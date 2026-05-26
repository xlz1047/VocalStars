import asyncio
import uuid
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.schemas.coaching_result import CoachingAnalysisResponse
from app.services.coaching import generate_recommendations
from app.services.ml_serialiser import coaching_result_to_dict

UPLOAD_DIR = Path("audio_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()


@router.get("/recommendations/{session_id}")
async def coaching_recommendations(session_id: str):
    recommendations = generate_recommendations(session_id)
    return {"session_id": session_id, "recommendations": recommendations}


@router.post("/analyse", response_model=CoachingAnalysisResponse)
async def analyse_audio(file: UploadFile = File(...)):
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only audio files are accepted.")

    suffix = Path(file.filename or "recording").suffix or ".wav"
    audio_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    audio_path.write_bytes(await file.read())

    try:
        from ml_new.inference.coach_inference import analyse_recording

        loop = asyncio.get_event_loop()
        fn = partial(
            analyse_recording,
            str(audio_path),
            checkpoint=settings.ML_CHECKPOINT,
            device=settings.ML_DEVICE,
        )
        result = await loop.run_in_executor(None, fn)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ML inference failed: {exc}") from exc

    return CoachingAnalysisResponse(**coaching_result_to_dict(result))
