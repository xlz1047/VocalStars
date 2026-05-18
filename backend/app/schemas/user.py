from pydantic import EmailStr
from app.schemas.base import BaseSchema


class UserCreate(BaseSchema):
    email: EmailStr
    display_name: str | None = None


class UserRead(BaseSchema):
    id: int
    email: EmailStr
    display_name: str | None = None
