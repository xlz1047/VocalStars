"""WebSocket endpoint for real-time streaming vocal analysis.

Protocol
--------
Client → Server  (binary, every 10 ms):
    Raw Float32 PCM samples at 16 kHz, little-endian.
    Each message should contain exactly 160 samples (one 10 ms frame).
    Smaller messages are zero-padded; larger are truncated.

Server → Client  (JSON, every 10 ms):
    {
      "t_ms":               1230.0,
      "pitch_hz":           247.5,     // 0 = unvoiced
      "voiced":             true,
      "loudness_db":        -18.3,
      "breath":             false,
      "onset":              false,
      "vibrato_rate_hz":    5.2,       // 0 = no vibrato
      "vibrato_depth_cents": 38.0,
      "tempo_bpm":          112.0,     // 0 = not enough data yet
      "technique":          "vibrato",
      "technique_conf":     0.62
    }

Special control messages (JSON text):
    Client → Server: {"cmd": "reset"}   — start a new session
    Server → Client: {"error": "..."}   — fatal error, connection closes
    Server → Client: {"ready": true}    — model loaded, ready for audio

Usage from the browser (TypeScript sketch)::

    const ws = new WebSocket("ws://localhost:8000/ws/live-analysis");
    ws.binaryType = "arraybuffer";

    // Send reset before recording starts
    ws.send(JSON.stringify({ cmd: "reset" }));

    // In AudioWorkletProcessor.process():
    ws.send(float32Buffer.buffer);

    ws.onmessage = (ev) => {
        const frame = JSON.parse(ev.data);
        renderPitchDot(frame.pitch_hz, frame.t_ms);
    };
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_new.inference.streaming_inference import StreamingEngine

log = logging.getLogger(__name__)
router = APIRouter()

_CHECKPOINT = _REPO_ROOT / "ml_new" / "checkpoints" / "unified_v2" / "best.pt"

# Module-level singleton — loaded once at backend startup so the first
# WebSocket connection never waits ~2s for model initialisation.
# Single-user dev setup; add a per-connection pool for multi-user production.
_engine: StreamingEngine | None = None


def _get_engine() -> StreamingEngine:
    global _engine
    if _engine is None:
        log.info("Pre-warming streaming engine from %s …", _CHECKPOINT)
        _engine = StreamingEngine(checkpoint_path=_CHECKPOINT, device="auto")
        log.info("Streaming engine ready.")
    return _engine


# Trigger load at import time so uvicorn startup warms the model.
try:
    _get_engine()
except Exception as _e:
    log.warning("Could not pre-warm streaming engine: %s", _e)


@router.websocket("/ws/live-analysis")
async def live_analysis(websocket: WebSocket) -> None:
    """Stream per-frame vocal analysis back to the browser."""
    await websocket.accept()
    log.info("Live analysis WS connected: %s", websocket.client)

    engine = _get_engine()
    engine.reset()

    try:
        await websocket.send_text(json.dumps({"ready": True}))

        while True:
            message = await websocket.receive()

            # ── Text control messages ──────────────────────────────────────
            if "text" in message:
                try:
                    cmd = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if cmd.get("cmd") == "reset":
                    engine.reset()
                    await websocket.send_text(json.dumps({"ready": True}))
                continue

            # ── Binary audio frame ─────────────────────────────────────────
            if "bytes" not in message or not message["bytes"]:
                continue

            raw = message["bytes"]
            pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32)

            try:
                batch = engine.push_frame(pcm)
                if batch:
                    for frame in batch:
                        await websocket.send_text(json.dumps(frame))
            except Exception as exc:
                log.warning("Frame error: %s", exc)
                await websocket.send_text(json.dumps({"error": str(exc)}))

    except WebSocketDisconnect:
        log.info("Live analysis WS disconnected — flushing remaining frames")
        try:
            remainder = engine.flush()
            if remainder:
                for frame in remainder:
                    await websocket.send_text(json.dumps(frame))
        except Exception:
            pass
    except RuntimeError as exc:
        # Suppress the common "Cannot call receive once disconnect received" noise
        # that occurs when the browser closes the tab without a clean WS handshake.
        log.debug("Live analysis WS closed abruptly: %s", exc)
    except Exception as exc:
        log.exception("Live analysis WS error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
