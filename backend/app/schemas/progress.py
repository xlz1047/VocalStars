from typing import Any
from app.schemas.base import BaseSchema


class ProgressMetricRead(BaseSchema):
    metric_name: str
    values: dict[str, Any]


class ProgressResponse(BaseSchema):
    user_id: int
    progress: list[ProgressMetricRead]
