from sqlalchemy import Column, Integer, ForeignKey, JSON, DateTime, String
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class ProgressMetric(Base):
    __tablename__ = "progress_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    metric_name = Column(String(120), nullable=False)
    values = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
