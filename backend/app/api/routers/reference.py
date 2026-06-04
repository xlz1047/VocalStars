"""Reference catalog API — human vocal exercise tracks.

The index.json is loaded once at startup from data/reference_catalog/index.json
and held in module-level state. All filtering happens in Python over the
in-memory list — no database round-trip per request.

Endpoints
---------
GET /api/reference/catalog          — filtered list of catalog entries
GET /api/reference/exercise/{id}    — single fully-populated entry by asset_id
GET /api/reference/vector/{safe_id} — raw NPZ bytes for direct browser fetch
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INDEX_PATH = _REPO_ROOT / "data" / "reference_catalog" / "index.json"
_VECTORS_DIR = _REPO_ROOT / "data" / "reference_catalog" / "vectors"

router = APIRouter()

# ---------------------------------------------------------------------------
# Startup: load index into memory once
# ---------------------------------------------------------------------------

_catalog_entries: list[dict[str, Any]] = []
_catalog_by_asset_id: dict[str, dict[str, Any]] = {}
_catalog_loaded: bool = False


def _ensure_catalog_loaded() -> None:
    global _catalog_entries, _catalog_by_asset_id, _catalog_loaded
    if _catalog_loaded:
        return
    if not _INDEX_PATH.exists():
        _catalog_loaded = True
        return
    try:
        raw = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        _catalog_entries = raw.get("entries", [])
        _catalog_by_asset_id = {e["asset_id"]: e for e in _catalog_entries}
    except Exception:  # noqa: BLE001
        _catalog_entries = []
        _catalog_by_asset_id = {}
    _catalog_loaded = True


# Force load on import so the first request is not slow.
_ensure_catalog_loaded()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id_to_asset_id(safe_id: str) -> str:
    """Undo the asset_id → filename sanitization used by build_reference_index."""
    # The index stores asset_ids in the original colon-delimited form.
    # The safe filename replaces ":" and "/" with "_". We can't reverse
    # this losslessly, so we look up by iterating the index.
    # This is O(n) but n is small (thousands) and it is a dev endpoint.
    return safe_id  # callers should pass the raw asset_id URL-encoded


def _match_str(value: str | None, query: str | None) -> bool:
    if query is None:
        return True
    if value is None:
        return False
    return query.lower() in value.lower()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/catalog")
async def get_catalog(
    exercise_type: str | None = Query(None, description="e.g. sustained_note | vibrato | pitch_slide"),
    dataset: str | None = Query(None, description="vocalset | mir1k | gtsinger"),
    technique: str | None = Query(None),
    note: str | None = Query(None, description="e.g. C4"),
    singer: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    include_vectors: bool = Query(False, description="Embed target_pitch_vector in each result (large)."),
):
    """Return filtered catalog entries.

    By default, target_pitch_vector and voiced_vector are omitted to keep
    the list response small. Pass include_vectors=true to embed them.
    """
    _ensure_catalog_loaded()
    if not _catalog_entries:
        return {
            "schema_version": "vocalstars.reference_catalog.v2",
            "warning": "Reference catalog not yet built. Run scripts/data/build_reference_index.py.",
            "entries": [],
        }

    results: list[dict] = []
    for entry in _catalog_entries:
        if not _match_str(entry.get("exercise_type"), exercise_type):
            continue
        if not _match_str(entry.get("dataset"), dataset):
            continue
        if not _match_str(entry.get("technique"), technique):
            continue
        if not _match_str(entry.get("note_name"), note):
            continue
        if not _match_str(entry.get("singer_id"), singer):
            continue
        row = dict(entry)
        if not include_vectors:
            row.pop("target_pitch_vector", None)
            row.pop("voiced_vector", None)
        results.append(row)
        if len(results) >= limit:
            break

    return {
        "schema_version": "vocalstars.reference_catalog.v2",
        "total_matched": len(results),
        "limit": limit,
        "entries": results,
    }


@router.get("/exercise/{asset_id:path}")
async def get_exercise(asset_id: str):
    """Return one fully-populated entry including target_pitch_vector."""
    _ensure_catalog_loaded()
    # asset_id may be URL-encoded (colons are safe in paths but just in case)
    decoded = urllib.parse.unquote(asset_id)
    entry = _catalog_by_asset_id.get(decoded)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Asset not found: {decoded}")
    return {
        "schema_version": "vocalstars.exercise_reference.v2",
        **entry,
    }


@router.get("/vector/{safe_id:path}")
async def get_vector_npz(safe_id: str):
    """Serve the raw NPZ for a reference track (application/octet-stream).

    The safe_id is the sanitized filename stem (colons and slashes replaced
    with underscores), as embedded in the vector_url field of each catalog entry.
    """
    npz_path = _VECTORS_DIR / f"{safe_id}.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail=f"Vector file not found: {safe_id}.npz")
    # Resolve and verify it is inside the vectors directory
    try:
        npz_path.resolve().relative_to(_VECTORS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal not allowed.")
    data = npz_path.read_bytes()
    return Response(content=data, media_type="application/octet-stream")


@router.get("/stats")
async def get_catalog_stats():
    """Return a summary of what is in the reference catalog."""
    _ensure_catalog_loaded()
    if not _catalog_entries:
        return {"status": "empty", "entry_count": 0}

    by_dataset: dict[str, int] = {}
    by_exercise_type: dict[str, int] = {}
    by_technique: dict[str, int] = {}
    for entry in _catalog_entries:
        d = entry.get("dataset", "unknown")
        et = entry.get("exercise_type", "unknown")
        t = entry.get("technique", "unknown")
        by_dataset[d] = by_dataset.get(d, 0) + 1
        by_exercise_type[et] = by_exercise_type.get(et, 0) + 1
        by_technique[t] = by_technique.get(t, 0) + 1

    return {
        "status": "ok",
        "entry_count": len(_catalog_entries),
        "index_path": str(_INDEX_PATH),
        "index_exists": _INDEX_PATH.exists(),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_exercise_type": dict(sorted(by_exercise_type.items())),
        "by_technique": dict(sorted(by_technique.items(), key=lambda kv: -kv[1])),
    }
