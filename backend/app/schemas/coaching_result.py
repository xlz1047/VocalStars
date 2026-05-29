from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class VibratoInfoSchema(BaseModel):
    rate_hz: float
    depth_cents: float
    regularity: float


class NoteSegmentSchema(BaseModel):
    start_s: float
    duration_s: float
    pitch_hz: float
    note_name: str
    cents_error: float
    stability_cents: float
    vibrato: Optional[VibratoInfoSchema] = None


class VoiceQualitySchema(BaseModel):
    hnr_db: float
    jitter_pct: float
    shimmer_pct: float
    breathiness: str       # "clear" | "mild" | "breathy"
    is_unstable: bool


class CoachingAnalysisResponse(BaseModel):
    # Per-frame arrays (serialised as plain lists)
    pitch_hz: list[float]
    voiced: list[bool]
    breath_frames: list[bool]
    onset_frames: list[bool]
    hop_s: float

    # Clip-level technique
    technique: str
    technique_confidence: float
    all_technique_scores: dict[str, float]

    # Coaching metrics
    pitch_accuracy: float
    pitch_drift_cents: float
    phrase_lengths_s: list[float]
    breath_count: int
    onset_count: int
    onset_clarity: float

    # Algorithmic enrichments
    notes: list[NoteSegmentSchema]
    voice_quality: Optional[VoiceQualitySchema] = None
    vibrato_stats: dict
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    analysis_validity: dict[str, Any] = Field(default_factory=dict)
    task_config: dict[str, Any] = Field(default_factory=dict)
    task_analysis: dict[str, Any] = Field(default_factory=dict)

    # Human-readable coaching
    score: Optional[int] = None
    full_song_score: Optional[int] = None
    diagnostic_score: Optional[int] = None
    score_status: str
    score_caveat: Optional[str] = None
    summary: str
    issues: list[str]
    exercises: list[str]
