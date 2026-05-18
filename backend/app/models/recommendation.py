from sqlalchemy import Column, Integer, ForeignKey, String, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class PracticeRecommendation(Base):
    __tablename__ = "practice_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("singing_sessions.id"), nullable=False)
    category = Column(String(120), nullable=False)
    details = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("SingingSession")
