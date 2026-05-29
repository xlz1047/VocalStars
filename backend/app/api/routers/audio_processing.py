from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from app.services.audio_processing import process_audio_file
from app.services.ml_inference import get_ml_service
import tempfile
import json
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
    song_title: str = Form("Unknown Song"),
    artist: str = Form("Unknown Artist"),
    task_config: str = Form(None),
    checkpoint_path: str = Query(None),
):
    """Analyze audio file using ml_new models.

    Args:
        file: Audio file upload
        song_title: Title of the song (sent as FormData field)
        artist: Artist name (sent as FormData field)
        checkpoint_path: Optional path to checkpoint (defaults to fallback)

    Returns:
        Analysis results with coaching feedback
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    try:
        # Preserve original file extension so librosa/soundfile can detect format
        suffix = Path(file.filename).suffix if file.filename else ".webm"
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Run ML inference
        service = get_ml_service(
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None
        )
        parsed_task_config = None
        if task_config:
            try:
                parsed_task_config = json.loads(task_config)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid task_config JSON: {exc}") from exc
        result = service.analyze_audio(tmp_path, song_title, artist, parsed_task_config)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": None,
        }
