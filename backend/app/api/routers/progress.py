from fastapi import APIRouter
from app.services.progress import get_progress_metrics

router = APIRouter()


@router.get("/user/{user_id}")
async def progress_for_user(user_id: int):
    metrics = get_progress_metrics(user_id)
    return {"user_id": user_id, "progress": metrics}
