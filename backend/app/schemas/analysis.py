from typing import Any
from app.schemas.base import BaseSchema


class AnalysisSummary(BaseSchema):
    pitch_stability: str
    rhythm_timing: str
    breath_consistency: str
    vocal_stability: str
    transition_quality: str
    strain_indicators: str
    raw_features: dict[str, Any] | None = None
