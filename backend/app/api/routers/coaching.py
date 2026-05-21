from fastapi import APIRouter
from app.services.coaching import generate_recommendations

router = APIRouter()


@router.get("/recommendations/{session_id}")
async def coaching_recommendations(session_id: int):
    recommendations = generate_recommendations(session_id)
    return {"session_id": session_id, "recommendations": recommendations}
