from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from app.services.audio_processing import process_audio_file
from app.services.ml_inference import get_ml_service
from app.services.ui_ready_response import build_ui_ready_response
from app.core.config import settings
import tempfile
import json
import time
import wave
from pathlib import Path

# Repo root: four levels up from this file
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Allowed path prefixes (relative to repo root) — prevents directory traversal
_AUDIO_WHITELIST = (
    "data/",
    "samples/",
    "ml/data/raw/gtsinger/",
    "ml/data/raw/vocalset/",
    "ml/data/raw/mir1k/",   # isolated vocal mono WAVs written by preprocess_human_references.py
)

router = APIRouter()

_GTSINGER_ENGLISH_ROOT = _REPO_ROOT / "ml" / "data" / "raw" / "gtsinger" / "English"
_SPEECH_GROUPS = {"Paired_Speech_Group"}
_CONTROL_GROUPS = {"Control_Group"}


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(_REPO_ROOT).as_posix()


def _audio_url_for_relative_path(path: str) -> str:
    return f"/api/audio/file?path={path}"


def _group_is_speech(group_name: str) -> bool:
    return group_name in _SPEECH_GROUPS or "speech" in group_name.lower()


def _group_rank(group_name: str) -> int:
    """Prefer sung technique examples, then natural singing, then speech."""
    if _group_is_speech(group_name):
        return 2
    if group_name in _CONTROL_GROUPS:
        return 1
    return 0


def _wav_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate > 0:
                return round(frames / float(rate), 3)
    except Exception:
        return None
    return None


def _sidecar_for(wav_path: Path, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        candidate = wav_path.with_name(f"{wav_path.stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _gtsinger_phrase_payload(wav_path: Path, *, group_name: str, include_duration: bool) -> dict:
    rel = _repo_relative(wav_path)
    metadata = _sidecar_for(wav_path, (".json",))
    textgrid = _sidecar_for(wav_path, (".TextGrid", "_TextGrid"))
    payload = {
        "id": wav_path.stem,
        "index": int(wav_path.stem) if wav_path.stem.isdigit() else None,
        "group": group_name,
        "is_speech": _group_is_speech(group_name),
        "audio_path": rel,
        "audio_url": _audio_url_for_relative_path(rel),
        "metadata_path": _repo_relative(metadata) if metadata else None,
        "textgrid_path": _repo_relative(textgrid) if textgrid else None,
        "status": "ok",
        "warnings": [],
    }
    if include_duration:
        payload["duration_s"] = _wav_duration_s(wav_path)
        if payload["duration_s"] is None:
            payload["warnings"].append("duration_unavailable")
    return payload


def _discover_gtsinger_catalog(
    *,
    singer_filter: str | None,
    technique_filter: str | None,
    include_speech: bool,
    include_duration: bool,
    limit: int | None,
) -> dict:
    warnings: list[str] = []
    if not _GTSINGER_ENGLISH_ROOT.exists():
        return {
            "schema_version": "gtsinger_catalog.v1",
            "root": "ml/data/raw/gtsinger/English",
            "default_group_policy": "sung_only",
            "songs": [],
            "warnings": ["gtsinger_english_root_missing"],
        }

    songs: list[dict] = []
    for singer_dir in sorted(path for path in _GTSINGER_ENGLISH_ROOT.iterdir() if path.is_dir()):
        singer = singer_dir.name
        if singer_filter and singer_filter.lower() not in singer.lower():
            continue
        for technique_dir in sorted(path for path in singer_dir.iterdir() if path.is_dir()):
            technique = technique_dir.name
            if technique_filter and technique_filter.lower() not in technique.lower():
                continue
            for song_dir in sorted(path for path in technique_dir.iterdir() if path.is_dir()):
                groups = []
                phrase_payloads = []
                song_warnings = []
                for group_dir in sorted(path for path in song_dir.iterdir() if path.is_dir()):
                    group_name = group_dir.name
                    is_speech = _group_is_speech(group_name)
                    groups.append({"name": group_name, "is_speech": is_speech})
                    if is_speech and not include_speech:
                        continue
                    wavs = sorted(group_dir.glob("*.wav"), key=lambda p: (not p.stem.isdigit(), p.stem))
                    for wav_path in wavs:
                        phrase_payloads.append(
                            _gtsinger_phrase_payload(
                                wav_path,
                                group_name=group_name,
                                include_duration=include_duration,
                            )
                        )

                phrase_payloads.sort(
                    key=lambda item: (
                        _group_rank(item["group"]),
                        item["index"] is None,
                        item["index"] if item["index"] is not None else 10**9,
                        item["id"],
                    )
                )
                groups.sort(key=lambda item: (_group_rank(item["name"]), item["name"]))
                if not phrase_payloads:
                    song_warnings.append("no_beginner_practice_audio_found")
                    if not include_speech and any(group["is_speech"] for group in groups):
                        song_warnings.append("paired_speech_hidden_by_default")
                    continue

                default_phrase = phrase_payloads[0]
                songs.append(
                    {
                        "id": f"gtsinger:{singer}:{technique}:{song_dir.name}",
                        "title": song_dir.name,
                        "singer": singer,
                        "technique": technique,
                        "groups": groups,
                        "default_group": default_phrase["group"],
                        "default_audio_path": default_phrase["audio_path"],
                        "default_audio_url": default_phrase["audio_url"],
                        "phrase_count": len(phrase_payloads),
                        "phrases": phrase_payloads,
                        "warnings": song_warnings,
                    }
                )
                if limit and len(songs) >= limit:
                    break
            if limit and len(songs) >= limit:
                break
        if limit and len(songs) >= limit:
            break

    return {
        "schema_version": "gtsinger_catalog.v1",
        "root": "ml/data/raw/gtsinger/English",
        "default_group_policy": "sung_only",
        "group_policy_notes": [
            "Technique groups are preferred for practice.",
            "Control_Group is natural singing and is used as fallback.",
            "Paired_Speech_Group is spoken lyrics and is hidden unless include_speech=true.",
        ],
        "songs": songs,
        "warnings": warnings,
    }

# ---------------------------------------------------------------------------
# Pre-selected dev / reference audio samples
# ---------------------------------------------------------------------------

_DEV_SAMPLES = [
    {"label": "Silence (5 s)",              "path": "samples/00_silence.wav"},
    {"label": "Speaking voice",             "path": "samples/01_speaking_voice.wav"},
    {"label": "Sustained aaa",              "path": "samples/03_sustained_aaa.wav"},
    {"label": "Pitch slide 220→440 Hz",     "path": "samples/04_pitch_slide.wav"},
    {"label": "Twinkle Twinkle (melody)",   "path": "samples/05_twinkle_twinkle.wav"},
    {
        "label": "GTSinger – A Thousand Years (Vibrato, Alto)",
        "path": "ml/data/raw/gtsinger/English/EN-Alto-2/Vibrato/A Thousand Years/Paired_Speech_Group/0001.wav",
    },
    {
        "label": "GTSinger – Shallow (Breathy, Tenor)",
        "path": "ml/data/raw/gtsinger/English/EN-Tenor-1/Breathy/Shallow/Paired_Speech_Group/0001.wav",
    },
    {
        "label": "GTSinger – Rolling in the Deep (Glissando, Alto)",
        "path": "ml/data/raw/gtsinger/English/EN-Alto-1/Glissando/rolling in the deep/Paired_Speech_Group/0001.wav",
    },
    {
        "label": "GTSinger – For the First Time in Forever (Glissando, Alto)",
        "path": "ml/data/raw/gtsinger/English/EN-Alto-1/Glissando/for the first time in forever/Control_Group/0000.wav",
    },
    {
        "label": "GTSinger – Someone Like You (Vibrato, Tenor)",
        "path": "ml/data/raw/gtsinger/English/EN-Tenor-1/Vibrato/Someone Like You/Paired_Speech_Group/0001.wav",
    },
    {
        "label": "GTSinger – Hello (Mixed Voice, Alto)",
        "path": "ml/data/raw/gtsinger/English/EN-Alto-2/Mixed_Voice_and_Falsetto/Hello/Paired_Speech_Group/0001.wav",
    },
]


@router.get("/dev-samples")
async def list_dev_samples():
    """Return the pre-selected list of dev/reference audio clips."""
    return [s for s in _DEV_SAMPLES if (_REPO_ROOT / s["path"]).exists()]


@router.get("/gtsinger-catalog")
async def list_gtsinger_catalog(
    singer: str | None = Query(None),
    technique: str | None = Query(None),
    include_speech: bool = Query(False),
    include_duration: bool = Query(False),
    limit: int | None = Query(None, ge=1, le=500),
):
    """Return dynamically discovered GTSinger reference clips.

    Beginner practice defaults to sung audio only. Paired_Speech_Group is useful
    for training/debugging speech-to-singing tasks, but it is not exposed unless
    explicitly requested with include_speech=true.
    """
    return _discover_gtsinger_catalog(
        singer_filter=singer,
        technique_filter=technique,
        include_speech=include_speech,
        include_duration=include_duration,
        limit=limit,
    )


@router.get("/file")
async def serve_audio_file(path: str):
    """Serve a local audio file by relative path from repo root.

    Only paths inside data/, samples/, or ml/data/raw/gtsinger/ are allowed.
    """
    # Normalise and check whitelist
    norm = path.replace("\\", "/").lstrip("/")
    if not any(norm.startswith(prefix) for prefix in _AUDIO_WHITELIST):
        raise HTTPException(status_code=400, detail="Path not in allowed directories.")
    # Block directory traversal by rejecting any ".." path component.
    # The whitelist prefix check above is the primary security boundary; this
    # guard catches explicit traversal patterns without following macOS aliases.
    from pathlib import PurePosixPath
    if ".." in PurePosixPath(norm).parts:
        raise HTTPException(status_code=400, detail="Path traversal not allowed.")
    abs_path = _REPO_ROOT / norm
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {norm}")
    media_type = "audio/wav" if abs_path.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(str(abs_path), media_type=media_type)


def _truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    content = await file.read()
    result = process_audio_file(file.filename, content)
    return {"status": "accepted", "analysis_session": result}


@router.post("/analyze-with-ml")
async def analyze_audio_with_ml(
    file: UploadFile = File(...),
    song_title: str = Form("Unknown Song"),
    artist: str = Form("Unknown Artist"),
    task_config: str = Form(None),
    response_mode_form: str = Form(None, alias="response_mode"),
    include_ui_ready_analysis_form: str = Form(None, alias="include_ui_ready_analysis"),
    include_frames_form: str = Form(None, alias="include_frames"),
    debug_form: str = Form(None, alias="debug"),
    response_mode_query: str = Query(None, alias="response_mode"),
    include_ui_ready_analysis_query: str = Query(None, alias="include_ui_ready_analysis"),
    include_frames_query: str = Query(None, alias="include_frames"),
    debug_query: str = Query(None, alias="debug"),
    checkpoint_path: str = Query(None),
):
    """Analyze audio file using ml_new models.

    Args:
        file: Audio file upload
        song_title: Title of the song (sent as FormData field)
        artist: Artist name (sent as FormData field)
        checkpoint_path: Optional path to checkpoint (defaults to fallback)

    Returns:
        Analysis results with coaching feedback
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio uploads are supported.")

    tmp_path = None
    route_start = time.perf_counter()
    try:
        # Preserve original file extension so librosa/soundfile can detect format
        suffix = Path(file.filename).suffix if file.filename else ".webm"
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        parsed_task_config = None
        if task_config:
            try:
                parsed_task_config = json.loads(task_config)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid task_config JSON: {exc}") from exc

        response_mode = response_mode_form or response_mode_query
        include_ui_ready_analysis = _truthy(
            include_ui_ready_analysis_form or include_ui_ready_analysis_query
        )
        wants_ui_ready = response_mode == "ui_ready" or include_ui_ready_analysis
        include_frames = _truthy(include_frames_form or include_frames_query)
        debug = _truthy(debug_form or debug_query)

        # Preserve legacy behavior unless UI-ready mode is explicitly requested.
        resolved_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path
            else Path(settings.ML_CHECKPOINT)
            if wants_ui_ready
            else None
        )
        service = get_ml_service(checkpoint_path=resolved_checkpoint)

        if not wants_ui_ready:
            return service.analyze_audio(tmp_path, song_title, artist, parsed_task_config)

        coaching_start = time.perf_counter()
        coaching_result = service.run_coaching(tmp_path, parsed_task_config)
        coaching_elapsed = time.perf_counter() - coaching_start
        legacy_data = service.format_coaching_result(coaching_result, song_title, artist)
        ui_ready_start = time.perf_counter()
        ui_ready_analysis = build_ui_ready_response(
            tmp_path,
            coaching_result=coaching_result,
            task_config=parsed_task_config,
            checkpoint=resolved_checkpoint,
            device=service.device,
            include_frames=include_frames,
            debug=debug,
        )
        ui_ready_elapsed = time.perf_counter() - ui_ready_start
        route_elapsed = time.perf_counter() - route_start
        ui_ready_analysis["performance"] = {
            **(ui_ready_analysis.get("performance") or {}),
            "checkpoint_coaching_s": round(coaching_elapsed, 4),
            "build_ui_ready_response_s": round(ui_ready_elapsed, 4),
            "endpoint_total_s": round(route_elapsed, 4),
        }
        legacy_data["uiReadyAnalysis"] = ui_ready_analysis
        legacy_data["ui_ready_analysis"] = ui_ready_analysis

        return {
            "status": "success",
            "data": legacy_data,
            "uiReadyAnalysis": ui_ready_analysis,
            "ui_ready_analysis": ui_ready_analysis,
            "debug": service._debug_info(),
        }


    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": None,
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
