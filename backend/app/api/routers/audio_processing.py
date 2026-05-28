from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.audio_processing import process_audio_file
from app.services.ml_inference import get_ml_service
import tempfile
from pathlib import Path

router = APIRouter()


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    content = await file.read()
    result = process_audio_file(file.filename, content)
    return {"status": "accepted", "analysis_session": result}


@router.post("/analyze-with-ml")
async def analyze_audio_with_ml(
    file: UploadFile = File(...),
    song_title: str = Query("Unknown Song"),
    artist: str = Query("Unknown Artist"),
    checkpoint_path: str = Query(None),
):
    """Analyze audio file using ml_new models.
    
    Args:
        file: Audio file upload
        song_title: Title of the song
        artist: Artist name
        checkpoint_path: Optional path to checkpoint (defaults to fallback)
        
    Returns:
        Analysis results with coaching feedback
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    try:
        # Save audio to temporary file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Run ML inference
        service = get_ml_service(
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
        )
        result = service.analyze_audio(tmp_path, song_title, artist)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": None,
        }
