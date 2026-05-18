from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.audio_processing import process_audio_file

router = APIRouter()


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    content = await file.read()
    result = process_audio_file(file.filename, content)
    return {"status": "accepted", "analysis_session": result}
