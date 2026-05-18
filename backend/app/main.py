from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import auth, audio_processing, analysis, coaching, progress
from app.core.config import settings
from app.core.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="VocalStars API",
    description="Modular API for beginner vocal coaching analysis and recommendations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(audio_processing.router, prefix="/api/audio", tags=["audio_processing"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(coaching.router, prefix="/api/coaching", tags=["coaching"])
app.include_router(progress.router, prefix="/api/progress", tags=["progress"])
