/**
 * LivePitchCanvas — scrolling real-time pitch + signal visualization.
 *
 * Layout (top → bottom):
 *   ┌─────────────────────────────────────┬──────┐
 *   │  8-second scrolling pitch lane      │ vol  │
 *   │  • coloured pitch dots              │ bar  │
 *   │  • breath ▼ / onset ▲ markers       │      │
 *   │  • vibrato glow + badge             │      │
 *   ├─────────────────────────────────────┴──────┤
 *   │  RTF bar  [■■■■░░░░]  0.47x  ~344ms       │
 *   └────────────────────────────────────────────┘
 *
 * Width is read from the DOM container via ResizeObserver so it fills
 * any container without overflow.
 */

import { useRef, useEffect } from "react";
import { LiveFrame } from "../types";
import { hzToNoteName } from "../utils/noteUtils";

const WINDOW_S    = 8;          // seconds of history shown
const HOP_S       = 0.01;       // 10ms per frame
const FMIN_HZ     = 80;
const FMAX_HZ     = 880;
const LOG_FMIN    = Math.log2(FMIN_HZ);
const LOG_FMAX    = Math.log2(FMAX_HZ);

const PAD_LEFT    = 36;
const PAD_RIGHT   = 52;
const PAD_TOP     = 28;
const PAD_BOTTOM  = 4;
const RTF_BAR_H   = 22;        // height of the RTF status bar at the bottom

const SEMITONE_NOTES: [number, string][] = (() => {
  const names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
  const result: [number, string][] = [];
  for (let midi = 40; midi <= 84; midi++) {
    const hz = 440 * 2 ** ((midi - 69) / 12);
    if (hz >= FMIN_HZ && hz <= FMAX_HZ) {
      result.push([hz, names[midi % 12] + String(Math.floor(midi / 12) - 1)]);
    }
  }
  return result;
})();

function hzToY(hz: number, pitchH: number): number {
  const t = (Math.log2(Math.max(hz, 1)) - LOG_FMIN) / (LOG_FMAX - LOG_FMIN);
  return PAD_TOP + (1 - t) * pitchH;
}

function pitchColor(hz: number): string {
  if (hz <= 0) return "rgba(255,255,255,0.15)";
  const midi = 69 + 12 * Math.log2(hz / 440);
  const cents = Math.abs(midi - Math.round(midi)) * 100;
  if (cents <= 25) return "#3cddc7";
  if (cents <= 50) return "#ffd166";
  return "#ff6b6b";
}

export interface TargetContourPoint {
  /** Absolute time in seconds from the start of the exercise. */
  t_s: number;
  f0_hz: number;
  note_name: string;
}

interface Props {
  frames:           LiveFrame[];
  isConnected:      boolean;
  latencyMs:        number;
  rtf:              number;
  height?:          number;
  connectionError?: string | null;
  /** Optional time-series of target pitch points to overlay as a guide line. */
  targetContour?:   TargetContourPoint[] | null;
  /** Total expected duration of the exercise in seconds (used for contour scroll). */
  exerciseDuration?: number;
  /**
   * When true the canvas expands to fit ALL frames (no sliding window) and
   * the live-only overlays (RTF bar, connection badge, playhead) are hidden.
   * The parent is responsible for wrapping this component in a horizontally
   * scrollable container.
   */
  frozen?: boolean;
}

export default function LivePitchCanvas({
  frames,
  isConnected,
  latencyMs,
  rtf,
  height = 260,
  connectionError = null,
  targetContour = null,
  exerciseDuration,
  frozen = false,
}: Props) {
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const widthRef     = useRef<number>(760);

  // ── Responsive width via ResizeObserver ────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      widthRef.current = Math.floor(entry.contentRect.width) || 760;
      // Force a redraw on the next animation frame
      if (canvasRef.current) {
        canvasRef.current.style.width = `${widthRef.current}px`;
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // ── Canvas draw ────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const viewportW = widthRef.current;
    const dpr       = window.devicePixelRatio || 1;

    // In frozen mode the canvas expands to hold every recorded frame at the
    // same pixel density as the live 8-second window.
    const totalFrames = WINDOW_S / HOP_S;              // frames that fit in viewport
    const innerViewW  = viewportW - PAD_LEFT - PAD_RIGHT;
    const pxPerFrame  = innerViewW / totalFrames;

    const frozenHeight = frozen ? height - RTF_BAR_H : height;  // no RTF bar when frozen
    const width = frozen
      ? Math.max(viewportW, PAD_LEFT + frames.length * pxPerFrame + PAD_RIGHT)
      : viewportW;

    canvas.width  = width       * dpr;
    canvas.height = frozenHeight * dpr;
    canvas.style.width  = `${width}px`;
    canvas.style.height = `${frozenHeight}px`;
    ctx.scale(dpr, dpr);

    const pitchH = frozenHeight - PAD_TOP - PAD_BOTTOM - (frozen ? 0 : RTF_BAR_H);
    const innerW = width - PAD_LEFT - PAD_RIGHT;
    const latest = frames.length > 0 ? frames[frames.length - 1] : null;

    // Background
    ctx.fillStyle = "#0a0c16";
    ctx.fillRect(0, 0, width, frozenHeight);

    // ── Semitone grid ─────────────────────────────────────────────────
    for (const [hz, label] of SEMITONE_NOTES) {
      const y    = hzToY(hz, pitchH);
      const isC  = label.startsWith("C") && !label.startsWith("C#");
      ctx.strokeStyle = isC ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.05)";
      ctx.lineWidth   = isC ? 0.8 : 0.4;
      ctx.setLineDash(isC ? [] : [2, 4]);
      ctx.beginPath();
      ctx.moveTo(PAD_LEFT, y);
      ctx.lineTo(width - PAD_RIGHT, y);
      ctx.stroke();
      ctx.setLineDash([]);
      if (isC || label.match(/^[ACEG]\d$/)) {
        ctx.fillStyle  = "rgba(255,255,255,0.30)";
        ctx.font       = "9px monospace";
        ctx.textAlign  = "right";
        ctx.fillText(label, PAD_LEFT - 4, y + 3);
      }
    }

    // ── Target contour overlay ────────────────────────────────────────
    if (targetContour && targetContour.length >= 2) {
      // Determine time offset: the rightmost visible edge is the latest frame time
      const latestT = frames.length > 0
        ? (frames[frames.length - 1].t_ms / 1000)
        : (exerciseDuration ?? WINDOW_S);
      const windowStart = latestT - WINDOW_S;

      ctx.save();
      ctx.strokeStyle = "rgba(100,220,200,0.55)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      let first = true;
      for (const pt of targetContour) {
        if (pt.f0_hz < FMIN_HZ || pt.f0_hz > FMAX_HZ) continue;
        const relT = pt.t_s - windowStart;
        const px = PAD_LEFT + (relT / WINDOW_S) * innerW;
        const py = hzToY(pt.f0_hz, pitchH);
        if (first) { ctx.moveTo(px, py); first = false; }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();

      // Label at right edge showing current target note
      const nowTarget = (() => {
        const nowT = latestT;
        for (let i = targetContour.length - 1; i >= 0; i--) {
          if (targetContour[i].t_s <= nowT) return targetContour[i];
        }
        return targetContour[0];
      })();
      if (nowTarget && nowTarget.f0_hz >= FMIN_HZ && nowTarget.f0_hz <= FMAX_HZ) {
        const ty = hzToY(nowTarget.f0_hz, pitchH);
        ctx.fillStyle = "rgba(100,220,200,0.80)";
        ctx.font = "bold 10px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`→ ${nowTarget.note_name}`, width - PAD_RIGHT + 4, ty + 3);
      }
    }

    // ── Pitch dots + markers ──────────────────────────────────────────
    // Live: show only the last 8 s (sliding window, newest frame at right edge).
    // Frozen: show all frames from the beginning, left-to-right.
    const startIdx      = frozen ? 0 : Math.max(0, frames.length - totalFrames);
    const displayFrames = frames.slice(startIdx);
    const xOffset       = frozen ? 0 : (totalFrames - displayFrames.length);

    for (let i = 0; i < displayFrames.length; i++) {
      const frame = displayFrames[i];
      const x     = PAD_LEFT + (xOffset + i) * pxPerFrame;

      if (frame.breath) {
        ctx.fillStyle = "#ff9f43";
        ctx.beginPath();
        ctx.moveTo(x,     PAD_TOP + pitchH + 4);
        ctx.lineTo(x - 3, PAD_TOP + pitchH + 10);
        ctx.lineTo(x + 3, PAD_TOP + pitchH + 10);
        ctx.closePath();
        ctx.fill();
      }
      if (frame.onset) {
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.lineWidth   = 1;
        ctx.beginPath();
        ctx.moveTo(x, PAD_TOP - 7);
        ctx.lineTo(x, PAD_TOP - 2);
        ctx.stroke();
      }

      if (!frame.voiced || frame.pitch_hz <= 0) continue;
      const y = hzToY(frame.pitch_hz, pitchH);
      const r = frame.vibrato_rate_hz > 0 ? 3 : 2;

      if (frame.vibrato_rate_hz > 0) {
        ctx.shadowColor = "#ff9f43";
        ctx.shadowBlur  = 8;
      }
      ctx.fillStyle = pitchColor(frame.pitch_hz);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // ── Current sung note label (right edge, near playhead) ──────────
    if (latest?.voiced && latest.pitch_hz > 0) {
      const noteY = hzToY(latest.pitch_hz, pitchH);
      const noteName = hzToNoteName(latest.pitch_hz);
      const labelX = width - PAD_RIGHT - 36;
      ctx.fillStyle = "rgba(255,177,192,0.90)";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(noteName, labelX, noteY - 7);
    }

    // ── Loudness bar (right edge) ─────────────────────────────────────
    if (latest) {
      const dbNorm = Math.max(0, Math.min(1, (latest.loudness_db + 60) / 60));
      const barH   = dbNorm * pitchH;
      const barX   = width - PAD_RIGHT + 8;
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(barX, PAD_TOP, 10, pitchH);
      const lColor  = dbNorm > 0.8 ? "#ff6b6b" : dbNorm > 0.5 ? "#ffd166" : "#3cddc7";
      ctx.fillStyle = lColor;
      ctx.fillRect(barX, PAD_TOP + pitchH - barH, 10, barH);
      ctx.fillStyle  = "rgba(255,255,255,0.25)";
      ctx.font       = "8px monospace";
      ctx.textAlign  = "center";
      ctx.fillText("vol", barX + 5, PAD_TOP + pitchH + 12);
    }

    if (!frozen) {
      // ── Playhead (live only) ────────────────────────────────────────
      ctx.strokeStyle = "rgba(255,255,255,0.20)";
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(width - PAD_RIGHT, PAD_TOP);
      ctx.lineTo(width - PAD_RIGHT, PAD_TOP + pitchH);
      ctx.stroke();
      ctx.setLineDash([]);

      // ── Top badges (live only) ──────────────────────────────────────
      const badgeY = 15;
      ctx.fillStyle = isConnected ? "#3cddc7" : "#ff6b6b";
      ctx.beginPath();
      ctx.arc(PAD_LEFT + 7, badgeY - 2, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle  = "rgba(255,255,255,0.40)";
      ctx.font       = "10px sans-serif";
      ctx.textAlign  = "left";
      ctx.fillText(isConnected ? "live" : "connecting…", PAD_LEFT + 16, badgeY);

      if (latest?.tempo_bpm && latest.tempo_bpm > 0) {
        ctx.font      = "bold 10px sans-serif";
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255,255,255,0.50)";
        ctx.fillText(`♩= ${latest.tempo_bpm.toFixed(0)}`, width - PAD_RIGHT - 4, badgeY);
      }
      if (latest?.technique && latest.technique !== "—" && latest.technique !== "unknown") {
        ctx.font      = "10px sans-serif";
        ctx.textAlign = "right";
        ctx.fillStyle = "#ff9f43";
        const tempoW  = (latest?.tempo_bpm > 0) ? 54 : 0;
        ctx.fillText(
          latest.technique.replace(/_/g, " "),
          width - PAD_RIGHT - 4 - tempoW - (tempoW > 0 ? 8 : 0),
          badgeY,
        );
      }
      if (latest?.vibrato_rate_hz && latest.vibrato_rate_hz > 0) {
        ctx.font      = "9px monospace";
        ctx.textAlign = "left";
        ctx.fillStyle = "#ff9f43";
        ctx.fillText(
          `vib ${latest.vibrato_rate_hz.toFixed(1)}Hz ${latest.vibrato_depth_cents.toFixed(0)}¢`,
          PAD_LEFT + 58,
          badgeY,
        );
      }

      // ── RTF / latency bar (live only) ───────────────────────────────
      const barY    = height - RTF_BAR_H + 2;
      const barFillW = innerW * 0.40;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, height - RTF_BAR_H);
      ctx.lineTo(width, height - RTF_BAR_H);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(PAD_LEFT, barY + 2, barFillW, 8);
      const rtfClamped = Math.min(rtf, 1.5);
      const rtfFill    = (rtfClamped / 1.5) * barFillW;
      const rtfColor   = rtf <= 0 ? "rgba(255,255,255,0.15)"
                       : rtf < 0.8 ? "#3cddc7"
                       : rtf < 1.0 ? "#ffd166"
                       : "#ff6b6b";
      ctx.fillStyle = rtfColor;
      ctx.fillRect(PAD_LEFT, barY + 2, rtfFill, 8);
      ctx.font      = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fillText(rtf > 0 ? `RTF ${rtf.toFixed(2)}x` : "RTF —", PAD_LEFT + barFillW + 8, barY + 9);
      if (latencyMs > 0) {
        ctx.textAlign = "right";
        ctx.fillStyle = "rgba(255,255,255,0.40)";
        ctx.fillText(`~${latencyMs}ms`, width - PAD_RIGHT, barY + 9);
      }
    } else {
      // ── Frozen label ────────────────────────────────────────────────
      const durationS = frames.length * HOP_S;
      ctx.fillStyle  = "rgba(255,255,255,0.30)";
      ctx.font       = "9px monospace";
      ctx.textAlign  = "left";
      ctx.fillText(`${durationS.toFixed(1)} s  ·  scroll to review`, PAD_LEFT + 4, PAD_TOP - 10);
    }

  }, [frames, isConnected, latencyMs, rtf, height, targetContour, exerciseDuration, frozen]);

  return (
    <div
      ref={containerRef}
      className={`relative rounded-xl border border-white/8 bg-surface-container-lowest/70 w-full${frozen ? "" : " overflow-hidden"}`}
    >
      <canvas ref={canvasRef} style={{ display: "block" }} />
      {connectionError ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 pointer-events-none px-6 text-center">
          <span className="text-sm font-bold text-red-400">⚠ Analysis server unreachable</span>
          <span className="text-[11px] text-on-surface-variant/60 font-mono leading-relaxed">
            {connectionError}
          </span>
        </div>
      ) : !isConnected && frames.length === 0 ? (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-xs text-on-surface-variant/50">
            Connecting to analysis server…
          </span>
        </div>
      ) : null}
    </div>
  );
}
