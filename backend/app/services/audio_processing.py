import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ml.pipeline import analyze_audio_file
from app.core.database import SessionLocal
from app.models.user import User
from app.models.session import SingingSession
from app.models.analysis import AnalysisResult

BASE_STORAGE = Path("./audio_uploads")
BASE_STORAGE.mkdir(parents=True, exist_ok=True)


def _get_or_create_guest(db):
    guest = db.query(User).filter(User.email == "guest@local").first()
    if guest:
        return guest
    guest = User(email="guest@local", display_name="Guest")
    db.add(guest)
    db.commit()
    db.refresh(guest)
    return guest


def process_audio_file(filename: str, file_bytes: bytes) -> dict:
    """Save uploaded audio, run the ML pipeline, and persist a session + analysis.

    Returns a small summary for immediate frontend consumption.
    """
    audio_path = BASE_STORAGE / filename
    audio_path.write_bytes(file_bytes)

    # Run analysis (may be slow depending on ML libs)
    analysis = analyze_audio_file(str(audio_path))

    # Persist to the database using a simple local session user
    db = SessionLocal()
    try:
        user = _get_or_create_guest(db)
        session = SingingSession(user_id=user.id, file_name=filename, metadata={"path": str(audio_path)})
        db.add(session)
        db.commit()
        db.refresh(session)

        result = AnalysisResult(session_id=session.id, summary=analysis)
        db.add(result)
        db.commit()
        db.refresh(result)
    finally:
        db.close()

    return {
        "audio_path": str(audio_path),
        "session_id": session.id,
        "analysis_preview": analysis,
        "note": "Audio analyzed and results persisted",
    }
