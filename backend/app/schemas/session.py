from typing import Any
from app.schemas.base import BaseSchema


class SingingSessionCreate(BaseSchema):
    user_id: int
    file_name: str
    metadata: dict[str, Any] | None = None


class SingingSessionRead(BaseSchema):
    id: int
    user_id: int
    file_name: str
    metadata: dict[str, Any] | None = None
