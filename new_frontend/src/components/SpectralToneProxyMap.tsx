import { UiFrame, UiReadyAnalysis } from "../types";

interface SpectralToneProxyMapProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
}

function durationFor(frames: UiFrame[], fallback?: number): number {
  const last = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(fallback || 0, last, 0.1);
}

function numericMetric(frame: UiFrame, key: string): number | null {
  const value = frame.spectral_tone_proxy?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricRange(frames: UiFrame[], key: string, fallback: [number, number]): [number, number] {
  const values = frames
    .map((frame) => numericMetric(frame, key))
    .filter((value): value is number => typeof value === "number");
  if (!values.length) return fallback;
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 1e-9) return [Math.max(0, min - 1), max + 1];
  return [min, max];
}

function normalize(value: number | null, range: [number, number]): number {
  if (value === null) return 0;
  return Math.max(0, Math.min(1, (value - range[0]) / Math.max(range[1] - range[0], 1e-9)));
}

function heatColor(value: number, palette: "centroid" | "flatness" | "harmonic"): string {
  const alpha = 0.22 + value * 0.76;
  if (palette === "centroid") return `rgba(128, 203, 255, ${alpha})`;
  if (palette === "flatness") return `rgba(255, 209, 102, ${alpha})`;
  return `rgba(60, 221, 199, ${alpha})`;
}

export default function SpectralToneProxyMap({ analysis, compact = false }: SpectralToneProxyMapProps) {
  const frames = analysis.frames || [];
  const duration = durationFor(frames, analysis.audio?.duration_s);
  const width = 960;
  const rowHeight = compact ? 24 : 30;
  const top = 44;
  const height = top + rowHeight * 4 + 40;
  const pad = 20;
  const plotWidth = width - pad * 2;
  const frameWidth = Math.max(2, plotWidth / Math.max(frames.length, 1));

  const centroidRange = metricRange(frames, "spectral_centroid_hz", [0, 4000]);
  const flatnessRange = metricRange(frames, "spectral_flatness", [0, 1]);
  const lowRange = metricRange(frames, "low_frequency_ratio", [0, 1]);
  const harmonicRange = metricRange(frames, "harmonicity_noise_proxy", [0, 1]);

  const rows = [
    {
      key: "spectral_centroid_hz",
      label: "Brightness proxy",
      range: centroidRange,
      palette: "centroid" as const,
    },
    {
      key: "spectral_flatness",
      label: "Noise/flatness proxy",
      range: flatnessRange,
      palette: "flatness" as const,
    },
    {
      key: "low_frequency_ratio",
      label: "Low-frequency energy",
      range: lowRange,
      palette: "flatness" as const,
    },
    {
      key: "harmonicity_noise_proxy",
      label: "Harmonicity proxy",
      range: harmonicRange,
      palette: "harmonic" as const,
    },
  ];

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div>
        <h3 className="font-display font-bold text-lg text-white">Spectral Tone Proxy Map</h3>
        <p className="text-xs text-on-surface-variant leading-relaxed">
          A spectrogram-inspired map from frame-level spectral proxies. These are not timbre, strain, or technique diagnoses.
        </p>
      </div>

      <div className="rounded-xl bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className={`w-full min-w-[720px] ${compact ? "h-[170px]" : "h-[210px]"}`} role="img" aria-label="Spectral tone proxy map">
          <rect x="0" y="0" width={width} height={height} rx="12" fill="#0c0e17" />
          {rows.map((row, rowIndex) => {
            const y = top + rowIndex * rowHeight;
            return (
              <g key={row.key}>
                <text x={pad} y={y - 8} fill="#e4bdc3" fontSize="11" fontWeight="700">{row.label}</text>
                <rect x={pad} y={y} width={plotWidth} height={rowHeight - 7} rx="6" fill="#171a24" />
                {frames.map((frame) => {
                  const value = normalize(numericMetric(frame, row.key), row.range);
                  const x = pad + (frame.time_s / duration) * plotWidth;
                  return (
                    <rect
                      key={`${row.key}-${frame.frame_index}`}
                      x={x}
                      y={y}
                      width={Math.max(2, frameWidth)}
                      height={rowHeight - 7}
                      rx="1"
                      fill={heatColor(value, row.palette)}
                      opacity={frame.voiced ? 1 : 0.45}
                    />
                  );
                })}
              </g>
            );
          })}
          <text x={pad} y={height - 12} fill="#8f7379" fontSize="10">Proxy visualization. Use for exploration, not diagnosis.</text>
          <text x={width - pad - 42} y={height - 12} fill="#8f7379" fontSize="10">{duration.toFixed(1)}s</text>
        </svg>
      </div>
    </section>
  );
}
