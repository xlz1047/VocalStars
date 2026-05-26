"""Convert a CoachingResult dataclass to the Pydantic response schema.

dataclasses.asdict() handles nested dataclasses recursively but:
  1. Returns numpy arrays/scalars as-is (not JSON-serialisable)
  2. Skips @property methods (breathiness, is_unstable on VoiceQuality)

_clean() fixes (1); we inject the properties manually for (2).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np


def _clean(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def coaching_result_to_dict(result: object) -> dict:
    """Serialise a CoachingResult to a JSON-safe dict for the Pydantic response."""
    raw = _clean(asdict(result))  # type: ignore[arg-type]

    # Inject @property values that asdict() misses
    vq = getattr(result, "voice_quality", None)
    if vq is not None and raw.get("voice_quality") is not None:
        raw["voice_quality"]["breathiness"] = vq.breathiness
        raw["voice_quality"]["is_unstable"] = vq.is_unstable

    return raw
