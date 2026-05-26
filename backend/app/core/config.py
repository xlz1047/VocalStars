from pathlib import Path
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    DEBUG: bool = True
    SECRET_KEY: str = "supersecret"
    DATABASE_URL: str
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    ML_CHECKPOINT: str = str(ROOT_DIR / "ml_new" / "checkpoints" / "unified" / "best.pt")
    ML_DEVICE: str = "cpu"

    model_config = {
        "extra": "ignore",
        "env_file": ROOT_DIR / ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
