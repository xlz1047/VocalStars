/**
 * useLiveAnalysis — WebSocket + audio pipeline for real-time vocal analysis.
 *
 * Opens a WebSocket to /ws/live-analysis when isActive=true.
 * Creates a 16kHz AudioContext, taps the mic stream, and sends 160-sample
 * Float32 PCM chunks to the server every 10ms.
 * Returns the last HISTORY_FRAMES LiveFrames for the visualisation.
 *
 * Server batches 20 frames (200ms), runs inference, returns frames one-by-one.
 * Total end-to-end latency ≈ 200ms buffering + 144ms processing = 344ms.
 * RTF = proc_ms / (BATCH_FRAMES * 10) where proc_ms comes from the server.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { LiveFrame } from "../types";

const WS_URL      = "ws://localhost:8000/ws/live-analysis";
const SR_MODEL    = 16_000;
const CHUNK       = 160;          // one 10ms model frame
const BATCH_FRAMES = 20;          // must match server BATCH_FRAMES
const HISTORY_FRAMES = 1000;      // 10 seconds of history at 10ms/frame

export interface LiveAnalysisState {
  frames:           LiveFrame[];
  isConnected:      boolean;
  currentTechnique: string;
  currentTechConf:  number;
  currentTempo:     number;
  latestFrame:      LiveFrame | null;
  latencyMs:        number;        // estimated end-to-end latency in ms
  rtf:              number;        // Real-Time Factor = proc_ms / audio_ms (< 1 = faster than RT)
  connectionError:  string | null; // set after 6s if server never responds
}

export function useLiveAnalysis(
  stream: MediaStream | null,
  isActive: boolean,
): LiveAnalysisState {
  const wsRef           = useRef<WebSocket | null>(null);
  const audioCtxRef     = useRef<AudioContext | null>(null);
  const processorRef    = useRef<ScriptProcessorNode | null>(null);
  const accumRef        = useRef<Float32Array>(new Float32Array(0));
  const sessionStartRef = useRef<number>(0);   // Date.now() when "reset" was sent

  const isConnectedRef = useRef(false);  // ref copy for use inside timeout closure

  const [frames, setFrames]               = useState<LiveFrame[]>([]);
  const [isConnected, setIsConnected]     = useState(false);
  const [latencyMs, setLatencyMs]         = useState(0);
  const [rtf, setRtf]                     = useState(0);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const cleanup = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      void audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    accumRef.current = new Float32Array(0);
    setIsConnected(false);
  }, []);

  useEffect(() => {
    if (!isActive || !stream) {
      cleanup();
      setFrames([]);
      setLatencyMs(0);
      setRtf(0);
      setConnectionError(null);
      isConnectedRef.current = false;
      return;
    }

    // ── WebSocket ─────────────────────────────────────────────────────
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    // 6-second timeout: if server never sends {"ready":true}, show a clear error
    isConnectedRef.current = false;
    setConnectionError(null);
    const connectTimeoutId = setTimeout(() => {
      if (!isConnectedRef.current) {
        setConnectionError(
          "Cannot reach analysis server. " +
          "Run: cd VocalStars && uvicorn app.main:app --app-dir backend --reload-dir backend --port 8000"
        );
      }
    }, 6000);

    ws.onopen = () => {
      sessionStartRef.current = Date.now();
      ws.send(JSON.stringify({ cmd: "reset" }));
    };

    ws.onerror = () => {
      clearTimeout(connectTimeoutId);
      setConnectionError(
        "WebSocket connection refused. Is the backend running on port 8000?"
      );
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string);
        if (msg.ready) {
          clearTimeout(connectTimeoutId);
          isConnectedRef.current = true;
          setIsConnected(true);
          setConnectionError(null);
          return;
        }
        if (msg.error) { console.warn("live-analysis error:", msg.error); return; }

        const frame = msg as LiveFrame;

        // Compute latency: wall-clock elapsed since session start minus
        // how far into the recording this frame sits.
        const elapsed = Date.now() - sessionStartRef.current;
        const lag = elapsed - frame.t_ms;
        setLatencyMs(lag > 0 ? Math.round(lag) : 0);

        // RTF: how long it took to process one batch vs the audio it represents
        if (frame.proc_ms > 0) {
          setRtf(frame.proc_ms / (BATCH_FRAMES * 10));
        }

        setFrames(prev => {
          const next = [...prev, frame];
          return next.length > HISTORY_FRAMES
            ? next.slice(next.length - HISTORY_FRAMES)
            : next;
        });
      } catch { /* ignore parse errors */ }
    };

    ws.onclose = () => {
      clearTimeout(connectTimeoutId);
      isConnectedRef.current = false;
      setIsConnected(false);
    };

    // ── Audio pipeline at 16kHz ───────────────────────────────────────
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    let ctx: AudioContext;
    try {
      ctx = new AudioCtx({ sampleRate: SR_MODEL });
    } catch {
      ctx = new AudioCtx();
    }
    audioCtxRef.current = ctx;

    if (ctx.state === "suspended") void ctx.resume();

    const source    = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(1024, 1, 1);
    processorRef.current = processor;

    const nativeSR = ctx.sampleRate;

    processor.onaudioprocess = (ev) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const input   = ev.inputBuffer.getChannelData(0);
      const samples = nativeSR === SR_MODEL
        ? input
        : _downsample(input, nativeSR, SR_MODEL);

      const joined = new Float32Array(accumRef.current.length + samples.length);
      joined.set(accumRef.current);
      joined.set(samples, accumRef.current.length);

      let offset = 0;
      while (offset + CHUNK <= joined.length) {
        const chunk = joined.subarray(offset, offset + CHUNK);
        wsRef.current?.send(
          chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength),
        );
        offset += CHUNK;
      }
      accumRef.current = joined.slice(offset);
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    return () => {
      clearTimeout(connectTimeoutId);
      cleanup();
    };
  }, [isActive, stream, cleanup]);

  const latest = frames.length > 0 ? frames[frames.length - 1] : null;

  return {
    frames,
    isConnected,
    currentTechnique: latest?.technique ?? "—",
    currentTechConf:  latest?.technique_conf ?? 0,
    currentTempo:     latest?.tempo_bpm ?? 0,
    latestFrame:      latest,
    latencyMs,
    rtf,
    connectionError,
  };
}

function _downsample(input: Float32Array, inRate: number, outRate: number): Float32Array {
  const ratio  = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const output = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo     = Math.floor(srcIdx);
    const hi     = Math.min(lo + 1, input.length - 1);
    const frac   = srcIdx - lo;
    output[i]    = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return output;
}
