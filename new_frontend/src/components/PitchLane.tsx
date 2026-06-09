import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { UiFrame, UiReadyAnalysis } from "../types";
import { hzToNoteName, midiToHz } from "../utils/noteUtils";

interface PitchLaneProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
}

function frameDuration(frames: UiFrame[], fallback?: number): number {
  const last = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(fallback || 0, last, 0.1);
}

function targetFromAnalysis(analysis: UiReadyAnalysis): number | null {
  const target = analysis.task_config?.target as { f0_hz?: number } | null | undefined;
  if (target?.f0_hz && target.f0_hz > 0) return target.f0_hz;
  const uniqueTargets = new Set(
    (analysis.frames || [])
      .map((frame) => frame.target_f0_hz)
      .filter((value): value is number => typeof value === "number" && value > 0)
      .map((value) => Math.round(value * 100) / 100)
  );
  if (uniqueTargets.size === 1) {
    return analysis.frames?.find((frame) => frame.target_f0_hz && frame.target_f0_hz > 0)?.target_f0_hz || null;
  }
  return null;
}

function confidenceForFrame(frame: UiFrame): number {
  return Math.min(frame.pitch_confidence ?? 1, frame.voice_confidence ?? 1);
}

function dotColor(frame: UiFrame, hasTarget: boolean): string {
  if (!hasTarget || frame.cents_error === null || frame.cents_error === undefined) return "#ffd9df";
  const error = Math.abs(frame.cents_error);
  if (error <= 25) return "#3cddc7";
  if (error <= 75) return "#ffd166";
  return "#ffb4ab";
}

const MIN_LABEL_PX = 12; // minimum vertical gap between rendered labels

/** Build Y-axis note labels within the visible Hz range, skipping overlaps. */
function buildNoteGridLines(
  minF0: number,
  maxF0: number,
  y: (f0: number) => number
): { hz: number; label: string; yy: number; isC: boolean }[] {
  const rangeOctaves = Math.log2(maxF0 / minF0);
  // When range exceeds 2 octaves only label C notes to prevent crowding.
  const cNotesOnly = rangeOctaves > 2;

  const candidates: { hz: number; label: string; yy: number; isC: boolean }[] = [];
  const midiMin = Math.floor(12 * Math.log2(minF0 / 440) + 69) - 1;
  const midiMax = Math.ceil(12 * Math.log2(maxF0 / 440) + 69) + 1;
  for (let midi = midiMin; midi <= midiMax; midi++) {
    const hz = midiToHz(midi);
    if (hz < minF0 || hz > maxF0) continue;
    const label = hzToNoteName(hz);
    const isC = label.startsWith("C") && !label.startsWith("C#");
    if (cNotesOnly && !isC) continue;
    candidates.push({ hz, label, yy: y(hz), isC });
  }

  // Filter by minimum pixel spacing (candidates are sorted by hz ascending = yy descending).
  const out: { hz: number; label: string; yy: number; isC: boolean }[] = [];
  let lastYy = Infinity;
  for (const c of candidates) {
    if (lastYy - c.yy >= MIN_LABEL_PX) {
      out.push(c);
      lastYy = c.yy;
    }
  }
  return out;
}

export default function PitchLane({ analysis, compact = false }: PitchLaneProps) {
  const frames = analysis.frames || [];
  const duration = frameDuration(frames, analysis.audio?.duration_s);
  const target = targetFromAnalysis(analysis);
  const values = frames
    .map((frame) => frame.f0_hz)
    .filter((value): value is number => typeof value === "number" && value > 0);
  if (target) values.push(target);

  const minF0 = Math.max(40, (values.length ? Math.min(...values) : 180) * 0.85);
  const maxF0 = Math.max(260, (values.length ? Math.max(...values) : 440) * 1.15);
  const width = 960;
  const height = compact ? 190 : 250;
  const padLeft = 48; // wider left pad for note labels
  const padRight = 28;
  const padTop = 20;
  const padBottom = 22;

  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  const x = (time: number) => padLeft + (time / duration) * innerW;
  const y = (f0: number) => height - padBottom - ((f0 - minF0) / Math.max(maxF0 - minF0, 1)) * innerH;

  const voicedFrames = frames.filter((frame) => typeof frame.f0_hz === "number" && frame.f0_hz > 0);
  const targetFrames = frames.filter((frame) => typeof frame.target_f0_hz === "number" && frame.target_f0_hz > 0);
  const targetY = target ? y(target) : null;
  const hasReferenceContour = targetFrames.length > 1;

  // Build note grid lines
  const noteLines = buildNoteGridLines(minF0, maxF0, y);

  // Task-type specific info
  const taskType = analysis.task_config?.task_type;
  const isPitchSlide = taskType === "pitch_slide";
  const slideStartHz = isPitchSlide
    ? (analysis.task_config?.target as any)?.start_f0_hz ?? null
    : null;
  const slideEndHz = isPitchSlide
    ? (analysis.task_config?.target as any)?.end_f0_hz ?? null
    : null;

  // Compute accuracy of actual start/end vs target for slide
  const getSlideEndpointColor = (targetHz: number | null, actualHz: number | null): string => {
    if (targetHz == null || actualHz == null) return "#ffd166";
    const centsDiff = Math.abs(1200 * Math.log2(actualHz / targetHz));
    if (centsDiff <= 50) return "#3cddc7";
    if (centsDiff <= 150) return "#ffd166";
    return "#ff6b6b";
  };

  // Actual first/last voiced pitch for slide endpoint markers
  const firstVoicedHz = voicedFrames[0]?.f0_hz ?? null;
  const lastVoicedHz = voicedFrames[voicedFrames.length - 1]?.f0_hz ?? null;
  const startMarkerColor = getSlideEndpointColor(slideStartHz, firstVoicedHz as number | null);
  const endMarkerColor = getSlideEndpointColor(slideEndHz, lastVoicedHz as number | null);

  // Target contour note change labels: find frames where target_note changes
  const targetNoteLabels: { time_s: number; note: string; f0: number }[] = [];
  if (hasReferenceContour) {
    let lastNote = "";
    for (const frame of targetFrames) {
      const note = frame.target_note || hzToNoteName(frame.target_f0_hz as number);
      if (note !== lastNote) {
        targetNoteLabels.push({ time_s: frame.time_s, note, f0: frame.target_f0_hz as number });
        lastNote = note;
      }
    }
  }

  const [legendOpen, setLegendOpen] = useState(false);

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="font-display font-bold text-lg text-white">Pitch Lane</h3>
          <p className="text-xs text-on-surface-variant">
            F0 trace{target ? " with target reference" : ""}
            {isPitchSlide && slideStartHz != null && slideEndHz != null
              ? ` · slide ${hzToNoteName(slideStartHz)} → ${hzToNoteName(slideEndHz)}`
              : ""}
          </p>
        </div>
        <button
          onClick={() => setLegendOpen((o) => !o)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-semibold text-on-surface-variant transition-all"
        >
          ⓘ How to read this
          {legendOpen ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </button>
      </div>

      {legendOpen && (
        <div className="rounded-xl bg-white/4 border border-white/8 p-4 space-y-3 text-xs">
          <p className="font-semibold text-on-surface/80 text-[11px] uppercase tracking-wider">What you're looking at</p>
          <div className="grid sm:grid-cols-2 gap-x-6 gap-y-2">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 w-10 h-1.5 rounded-full bg-[#ffb1c0] flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Pink line</strong> — your actual singing pitch, traced over time</span>
            </div>
            {(target || hasReferenceContour) && (
              <div className="flex items-start gap-2">
                <span className="mt-0.5 w-10 border-t-2 border-dashed border-[#3cddc7] flex-shrink-0" />
                <span className="text-on-surface-variant"><strong className="text-on-surface">Cyan dashed line</strong> — the ideal/target pitch you should be hitting</span>
              </div>
            )}
            <div className="flex items-start gap-2">
              <span className="mt-0.5 w-3 h-3 rounded-full bg-[#3cddc7] flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Cyan dot</strong> — right on pitch (within 25 cents — pro-level accuracy)</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="mt-0.5 w-3 h-3 rounded-full bg-[#ffd166] flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Yellow dot</strong> — close but slightly off (25–75 cents — within half a semitone)</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="mt-0.5 w-3 h-3 rounded-full bg-[#ffb4ab] flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Pink/red dot</strong> — off-key (beyond 75 cents — needs practice)</span>
            </div>
            <div className="flex items-start gap-2">
              <span className="mt-0.5 w-3 h-1.5 rounded-full bg-white/30 flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Faded/dashed line</strong> — low confidence detection (background noise or unclear pitch)</span>
            </div>
            <div className="flex items-start gap-2 sm:col-span-2">
              <span className="mt-1 w-3 h-2 rounded-sm bg-white/40 flex-shrink-0" />
              <span className="text-on-surface-variant"><strong className="text-on-surface">Bottom bar</strong> — white = your voice detected, dark = silence or breath</span>
            </div>
          </div>
          <p className="text-on-surface-variant/60 text-[10px]">Tip: 100 cents = 1 semitone. Professional singers typically stay within ±25 cents.</p>
        </div>
      )}

      <div className="rounded-xl bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className={`w-full min-w-[720px] ${compact ? "h-[190px]" : "h-[250px]"}`}
          role="img"
          aria-label="Pitch lane"
        >
          <rect x="0" y="0" width={width} height={height} rx="12" fill="#0c0e17" />

          {/* Note grid lines + left-axis labels */}
          {noteLines.map(({ hz, label, yy, isC }) => (
            <g key={`grid-${hz}`}>
              <line
                x1={padLeft}
                x2={width - padRight}
                y1={yy}
                y2={yy}
                stroke={isC ? "rgba(255,255,255,0.12)" : "rgba(255,255,255,0.05)"}
                strokeWidth={isC ? 0.8 : 0.4}
                strokeDasharray={isC ? undefined : "2 4"}
              />
              <text
                x={padLeft - 4}
                y={yy + 3}
                fill={isC ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.25)"}
                fontSize={isC ? "11" : "9"}
                fontWeight={isC ? "700" : "400"}
                textAnchor="end"
              >
                {label}
              </text>
            </g>
          ))}

          {/* Single-target horizontal line */}
          {targetY !== null && (
            <>
              <line x1={padLeft} x2={width - padRight} y1={targetY} y2={targetY} stroke="#3cddc7" strokeDasharray="7 7" strokeWidth="2" />
              <text x={width - padRight - 4} y={Math.max(padTop + 8, targetY - 6)} fill="#3cddc7" fontSize="11" fontWeight="700" textAnchor="end">
                {target ? hzToNoteName(target) : ""} {target?.toFixed(0)} Hz
              </text>
            </>
          )}

          {/* Multi-target reference contour */}
          {targetY === null && hasReferenceContour && targetFrames.slice(1).map((frame, index) => {
            const prev = targetFrames[index];
            const contiguous = frame.frame_index === prev.frame_index + 1;
            return (
              <line
                key={`target-trace-${frame.frame_index}`}
                x1={x(prev.time_s)}
                y1={y(prev.target_f0_hz as number)}
                x2={x(frame.time_s)}
                y2={y(frame.target_f0_hz as number)}
                stroke="#3cddc7"
                strokeWidth="2"
                strokeDasharray={contiguous ? "8 5" : "2 8"}
                opacity={contiguous ? 0.9 : 0.45}
                strokeLinecap="round"
              />
            );
          })}

          {/* Target contour note change labels */}
          {targetNoteLabels.map((item, i) => (
            <text
              key={`tnote-${i}`}
              x={x(item.time_s) + 3}
              y={y(item.f0) - 6}
              fill="#3cddc7"
              fontSize="9"
              fontWeight="600"
              opacity="0.85"
            >
              {item.note}
            </text>
          ))}

          {/* User pitch trace lines */}
          {voicedFrames.slice(1).map((frame, index) => {
            const prev = voicedFrames[index];
            const lowConfidence = confidenceForFrame(frame) < 0.45 || confidenceForFrame(prev) < 0.45;
            return (
              <line
                key={`trace-${frame.frame_index}`}
                x1={x(prev.time_s)}
                y1={y(prev.f0_hz as number)}
                x2={x(frame.time_s)}
                y2={y(frame.f0_hz as number)}
                stroke="#ffb1c0"
                strokeWidth={lowConfidence ? 2 : 3}
                strokeDasharray={lowConfidence ? "6 6" : undefined}
                opacity={lowConfidence ? 0.42 : 0.95}
                strokeLinecap="round"
              />
            );
          })}

          {/* User pitch dots */}
          {voicedFrames.map((frame) => (
            <circle
              key={`dot-${frame.frame_index}`}
              cx={x(frame.time_s)}
              cy={y(frame.f0_hz as number)}
              r={confidenceForFrame(frame) < 0.45 ? 3 : 4}
              fill={dotColor(frame, Boolean(target || hasReferenceContour))}
              opacity={confidenceForFrame(frame) < 0.45 ? 0.45 : 0.95}
            />
          ))}

          {/* Voiced/unvoiced bar at bottom */}
          {frames.map((frame) => {
            const cx = x(frame.time_s);
            return (
              <rect
                key={`voice-${frame.frame_index}`}
                x={cx - 2}
                y={height - padBottom + 2}
                width="4"
                height="7"
                rx="2"
                fill={frame.voiced ? "#3cddc7" : "#5b3f44"}
                opacity={frame.voiced ? 0.95 : 0.42}
              />
            );
          })}

          {/* Pitch slide start/end markers */}
          {isPitchSlide && slideStartHz != null && (
            <>
              <line
                x1={padLeft + 1}
                x2={padLeft + 1}
                y1={padTop}
                y2={height - padBottom}
                stroke={startMarkerColor}
                strokeWidth="1.5"
                strokeDasharray="4 3"
                opacity="0.7"
              />
              <text x={padLeft + 5} y={y(slideStartHz) - 5} fill={startMarkerColor} fontSize="10" fontWeight="700">
                Start: {hzToNoteName(slideStartHz)}
              </text>
            </>
          )}
          {isPitchSlide && slideEndHz != null && (
            <>
              <line
                x1={width - padRight - 1}
                x2={width - padRight - 1}
                y1={padTop}
                y2={height - padBottom}
                stroke={endMarkerColor}
                strokeWidth="1.5"
                strokeDasharray="4 3"
                opacity="0.7"
              />
              <text x={width - padRight - 6} y={y(slideEndHz) - 5} fill={endMarkerColor} fontSize="10" fontWeight="700" textAnchor="end">
                Target: {hzToNoteName(slideEndHz)}
              </text>
            </>
          )}
        </svg>
      </div>
    </section>
  );
}
