from typing import Any
from app.schemas.base import BaseSchema


class RecommendationItem(BaseSchema):
    category: str
    details: dict[str, Any]


class RecommendationResponse(BaseSchema):
    session_id: int
    recommendations: list[RecommendationItem]
