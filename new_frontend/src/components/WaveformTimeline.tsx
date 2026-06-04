import { UiReadyAnalysis } from "../types";

interface WaveformTimelineProps {
  analysis: UiReadyAnalysis;
  compact?: boolean;
}

function frameDuration(analysis: UiReadyAnalysis): number {
  const frames = analysis.frames || [];
  const last = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(analysis.audio?.duration_s || 0, last, 0.1);
}

export default function WaveformTimeline({ analysis, compact = false }: WaveformTimelineProps) {
  const frames = analysis.frames || [];
  const duration = frameDuration(analysis);
  const width = 960;
  const height = compact ? 120 : 150;
  const pad = 18;
  const rmsValues = frames.map((frame) => {
    if (typeof frame.volume?.rms === "number") return frame.volume.rms;
    if (typeof frame.volume?.rms_db === "number") return Math.max(0, 1 + frame.volume.rms_db / 80);
    return 0;
  });
  const maxRms = Math.max(...rmsValues, 0.001);
  const x = (time: number) => pad + (time / duration) * (width - pad * 2);

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="font-display font-bold text-lg text-white">Volume And Voicing</h3>
          <p className="text-xs text-on-surface-variant">RMS-style timeline with voiced and unvoiced frame overlay.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
          <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-tertiary" /> voiced</span>
          <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary/50" /> unvoiced</span>
        </div>
      </div>

      <div className="rounded-xl bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className={`w-full min-w-[720px] ${compact ? "h-[120px]" : "h-[150px]"}`} role="img" aria-label="Volume timeline">
          <rect x="0" y="0" width={width} height={height} rx="12" fill="#0c0e17" />
          {frames.map((frame, index) => {
            const cx = x(frame.time_s);
            const normalized = rmsValues[index] / maxRms;
            const barHeight = normalized * (height - pad * 2);
            return (
              <g key={frame.frame_index}>
                <rect
                  x={cx - 4}
                  y={height - pad - barHeight}
                  width="8"
                  height={barHeight}
                  rx="4"
                  fill={frame.voiced ? "#3cddc7" : "#ab888e"}
                  opacity={frame.voiced ? 0.9 : 0.35}
                />
                <rect
                  x={cx - 4}
                  y={height - 11}
                  width="8"
                  height="5"
                  rx="2.5"
                  fill={frame.voiced ? "#3cddc7" : "#5b3f44"}
                />
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}
