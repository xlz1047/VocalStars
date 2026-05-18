from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(120), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sessions = relationship("SingingSession", back_populates="user")
