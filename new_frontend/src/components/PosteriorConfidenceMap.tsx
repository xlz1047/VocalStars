import { UiFrame, UiReadyAnalysis } from "../types";

interface PosteriorConfidenceMapProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
}

function durationFor(frames: UiFrame[], fallback?: number): number {
  const last = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(fallback || 0, last, 0.1);
}

function clamp01(value: number | null | undefined, fallback = 0): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  return Math.max(0, Math.min(1, value));
}

function confidenceColor(value: number, hue: "voice" | "pitch" | "quality"): string {
  const alpha = 0.2 + value * 0.78;
  if (hue === "voice") return `rgba(60, 221, 199, ${alpha})`;
  if (hue === "pitch") return `rgba(255, 177, 192, ${alpha})`;
  return value > 0.66
    ? `rgba(255, 180, 171, ${alpha})`
    : `rgba(255, 209, 102, ${0.22 + value * 0.55})`;
}

function qualityRisk(frame: UiFrame): number {
  const quality = frame.signal_quality || {};
  let risk = 0;
  if (quality.near_silence) risk += 0.35;
  if (quality.noisy) risk += 0.3;
  if (quality.low_confidence) risk += 0.25;
  if (quality.source_disagreement) risk += 0.2;
  if (quality.clipped) risk += 0.35;
  return clamp01(risk);
}

export default function PosteriorConfidenceMap({ analysis, compact = false }: PosteriorConfidenceMapProps) {
  const posteriorgram = analysis.visualizations?.posteriorgram;
  const frames = analysis.frames || [];
  const duration = Math.max(
    durationFor(frames, analysis.audio?.duration_s),
    posteriorgram?.time_s?.length ? posteriorgram.time_s[posteriorgram.time_s.length - 1] : 0
  );
  const width = 960;
  const rowHeight = compact ? 24 : 30;
  const top = 42;
  const height = top + rowHeight * 3 + 30;
  const pad = 20;
  const plotWidth = width - pad * 2;
  const posteriorValues = posteriorgram?.values || [];
  const posteriorTimes = posteriorgram?.time_s || [];
  const usePosteriorgram = posteriorValues.length > 0;
  const displayCount = usePosteriorgram ? posteriorValues.length : frames.length;
  const frameWidth = Math.max(2, plotWidth / Math.max(displayCount, 1));

  const rows = [
    { key: "voice", label: "Voice posterior", value: (frame: UiFrame) => clamp01(frame.voice_confidence), hue: "voice" as const },
    { key: "pitch", label: "Pitch posterior", value: (frame: UiFrame) => clamp01(frame.pitch_confidence), hue: "pitch" as const },
    { key: "quality", label: "Quality risk", value: qualityRisk, hue: "quality" as const },
  ];

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="font-display font-bold text-lg text-white">Posterior Confidence Map</h3>
          <p className="text-xs text-on-surface-variant">
            Model/DSP confidence over time for voice, pitch, and low-quality regions.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
          <span className="inline-flex items-center gap-1"><span className="w-3 h-2 rounded-sm bg-tertiary" /> voice</span>
          <span className="inline-flex items-center gap-1"><span className="w-3 h-2 rounded-sm bg-primary" /> pitch</span>
          <span className="inline-flex items-center gap-1"><span className="w-3 h-2 rounded-sm bg-error" /> risk</span>
        </div>
      </div>

      <div className="rounded-xl bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className={`w-full min-w-[720px] ${compact ? "h-[150px]" : "h-[180px]"}`} role="img" aria-label="Posterior confidence map">
          <rect x="0" y="0" width={width} height={height} rx="12" fill="#0c0e17" />
          {rows.map((row, rowIndex) => {
            const y = top + rowIndex * rowHeight;
            return (
              <g key={row.key}>
                <text x={pad} y={y - 8} fill="#e4bdc3" fontSize="11" fontWeight="700">{row.label}</text>
                <rect x={pad} y={y} width={plotWidth} height={rowHeight - 7} rx="6" fill="#171a24" />
                {(usePosteriorgram ? posteriorValues : frames).map((item, index) => {
                  const value = usePosteriorgram
                    ? clamp01((item as number[])[rowIndex])
                    : row.value(item as UiFrame);
                  const time = usePosteriorgram ? posteriorTimes[index] || 0 : (item as UiFrame).time_s;
                  const x = pad + (time / duration) * plotWidth;
                  return (
                    <rect
                      key={`${row.key}-${index}`}
                      x={x}
                      y={y}
                      width={Math.max(2, frameWidth)}
                      height={rowHeight - 7}
                      rx="1"
                      fill={confidenceColor(value, row.hue)}
                    />
                  );
                })}
              </g>
            );
          })}
          <text x={pad} y={height - 10} fill="#8f7379" fontSize="10">0s</text>
          <text x={width - pad - 42} y={height - 10} fill="#8f7379" fontSize="10">{duration.toFixed(1)}s</text>
        </svg>
      </div>
    </section>
  );
}
