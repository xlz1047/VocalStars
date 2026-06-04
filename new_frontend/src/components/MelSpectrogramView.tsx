import { UiReadyAnalysis } from "../types";

interface MelSpectrogramViewProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
}

function clamp01(value: number | null | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function spectrogramColor(value: number): string {
  const v = clamp01(value);
  const r = Math.round(24 + v * 231);
  const g = Math.round(16 + Math.max(0, v - 0.25) * 210);
  const b = Math.round(42 + (1 - Math.abs(v - 0.55)) * 90);
  return `rgb(${r}, ${g}, ${b})`;
}

function frameDuration(analysis: UiReadyAnalysis): number {
  const spec = analysis.visualizations?.spectrogram;
  const times = spec?.time_s || [];
  const lastSpecTime = times.length ? times[times.length - 1] : 0;
  const frameLast = (analysis.frames || []).reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(analysis.audio?.duration_s || 0, lastSpecTime, frameLast, 0.1);
}

export default function MelSpectrogramView({ analysis, compact = false }: MelSpectrogramViewProps) {
  const spectrogram = analysis.visualizations?.spectrogram;
  const values = spectrogram?.values || [];
  const nFrames = values.length;
  const nMels = spectrogram?.n_mels || values[0]?.length || 0;
  const duration = frameDuration(analysis);
  const width = 960;
  const height = compact ? 190 : 240;
  const pad = 26;
  const plotWidth = width - pad * 2;
  const plotHeight = height - pad * 2;
  const cellWidth = Math.max(1.5, plotWidth / Math.max(nFrames, 1));
  const cellHeight = Math.max(1.5, plotHeight / Math.max(nMels, 1));
  const frames = analysis.frames || [];
  const voicedFrames = frames.filter((frame) => typeof frame.f0_hz === "number" && frame.f0_hz > 0);
  const minFreq = spectrogram?.frequency_min_hz || 50;
  const maxFreq = spectrogram?.frequency_max_hz || 8000;
  const x = (time: number) => pad + (time / duration) * plotWidth;
  const yFreq = (freq: number) => {
    const logMin = Math.log2(Math.max(minFreq, 1));
    const logMax = Math.log2(Math.max(maxFreq, minFreq + 1));
    const normalized = (Math.log2(Math.max(freq, minFreq)) - logMin) / Math.max(logMax - logMin, 1e-9);
    return pad + (1 - Math.max(0, Math.min(1, normalized))) * plotHeight;
  };

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="font-display font-bold text-lg text-white">Log-Mel Spectrogram</h3>
          <p className="text-xs text-on-surface-variant">
            Compact acoustic energy view with selected f0 overlay.
          </p>
        </div>
        <div className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
          {spectrogram?.kind || "not available"}
        </div>
      </div>

      <div className="rounded-xl bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        {spectrogram?.error ? (
          <p className="text-sm text-on-surface-variant p-4">{spectrogram.error}</p>
        ) : nFrames && nMels ? (
          <svg viewBox={`0 0 ${width} ${height}`} className={`w-full min-w-[720px] ${compact ? "h-[190px]" : "h-[240px]"}`} role="img" aria-label="Log-mel spectrogram with f0 overlay">
            <rect x="0" y="0" width={width} height={height} rx="12" fill="#0c0e17" />
            {values.map((row, frameIndex) =>
              row.map((value, melIndex) => (
                <rect
                  key={`${frameIndex}-${melIndex}`}
                  x={pad + frameIndex * cellWidth}
                  y={pad + (nMels - 1 - melIndex) * cellHeight}
                  width={cellWidth + 0.4}
                  height={cellHeight + 0.4}
                  fill={spectrogramColor(value)}
                />
              ))
            )}
            {voicedFrames.slice(1).map((frame, index) => {
              const prev = voicedFrames[index];
              return (
                <line
                  key={`f0-${frame.frame_index}`}
                  x1={x(prev.time_s)}
                  y1={yFreq(prev.f0_hz as number)}
                  x2={x(frame.time_s)}
                  y2={yFreq(frame.f0_hz as number)}
                  stroke="#3cddc7"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  opacity={frame.pitch_confidence !== undefined && frame.pitch_confidence < 0.35 ? 0.45 : 0.95}
                />
              );
            })}
            <text x={pad} y={17} fill="#e4bdc3" fontSize="11">{Math.round(maxFreq)} Hz</text>
            <text x={pad} y={height - 10} fill="#e4bdc3" fontSize="11">{Math.round(minFreq)} Hz</text>
            <text x={width - pad - 52} y={height - 10} fill="#8f7379" fontSize="10">{duration.toFixed(1)}s</text>
          </svg>
        ) : (
          <p className="text-sm text-on-surface-variant p-4">Spectrogram data was not included in this response.</p>
        )}
      </div>
      {spectrogram?.caveat && (
        <p className="text-xs text-on-surface-variant leading-relaxed">{spectrogram.caveat}</p>
      )}
    </section>
  );
}
