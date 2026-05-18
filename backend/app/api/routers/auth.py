from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.get("/status")
async def auth_status():
    return {"status": "ok", "message": "Auth service placeholder"}
