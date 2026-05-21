from typing import Any


def build_coaching_outline(pitch: dict[str, Any], rhythm: dict[str, Any], breath: dict[str, Any], spectral: dict[str, Any]) -> dict[str, Any]:
    """Translate simple feature heuristics into beginner-friendly coaching notes.

    This function keeps recommendations conservative and actionable; it
    avoids any judgement and maps technical observations to exercises.
    """
    focus = []
    exercises = []

    # Pitch heuristics
    stability = pitch.get("stability_score") if isinstance(pitch, dict) else None
    voiced_ratio = pitch.get("voiced_ratio") if isinstance(pitch, dict) else None
    if stability is not None and stability > 0.0:
        if stability > 40:
            focus.append("pitch stability")
            exercises.append("Practice slow descending scales sustaining each note for 3-5 seconds to reduce pitch jitter.")
        else:
            focus.append("pitch tracking")
            exercises.append("Sing simple stepwise melodies on a single vowel, focusing on steady pitch.")

    # Rhythm heuristics
    timing_variance = rhythm.get("timing_variance") if isinstance(rhythm, dict) else None
    if timing_variance is not None:
        if timing_variance > 0.05:
            focus.append("rhythm/timing")
            exercises.append("Practice short phrases with a metronome at a comfortable tempo to improve timing consistency.")

    # Breath heuristics
    support = breath.get("support_score") if isinstance(breath, dict) else None
    if support is not None and support < 0.8:
        focus.append("breath support")
        exercises.append("Try 4-5 breath control exercises: inhale for 3, exhale on a hiss for 6, repeat to build steady support.")

    # Spectral / energy cues
    energy = spectral.get("energy_contour") if isinstance(spectral, dict) else None
    if energy:
        focus.append("vocal balance")
        exercises.append("Work on even dynamics by singing scales while keeping RMS energy steady across notes.")

    if not focus:
        focus = ["general technique"]
        exercises = ["Start with gentle warmups: lip trills, humming, and short sustain notes to find a comfortable placement."]

    return {"summary": "Coaching outline (starter)", "focus_areas": focus, "suggested_exercises": exercises}
