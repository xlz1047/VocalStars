import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  FileJson,
  Info,
  ListChecks,
  Lock,
  Music2,
  Shield,
  SlidersHorizontal,
  Volume2,
} from "lucide-react";
import MelSpectrogramView from "./MelSpectrogramView";
import PosteriorConfidenceMap from "./PosteriorConfidenceMap";
import SpectralToneProxyMap from "./SpectralToneProxyMap";

type GoldenIndex = {
  examples: string[];
};

type Frame = {
  time_s: number;
  frame_index: number;
  f0_hz: number | null;
  target_f0_hz?: number | null;
  voiced: boolean;
  voice_confidence?: number | null;
  pitch_confidence?: number | null;
  selected_f0_source?: string;
  selected_vad_source?: string;
  volume?: {
    rms?: number | null;
    rms_db?: number | null;
  };
  signal_quality?: Record<string, boolean>;
  caveats?: string[];
};

type Segment = {
  id?: string;
  type?: string;
  start_s?: number;
  end_s?: number;
  duration_s?: number;
  median_f0_hz?: number | null;
  ui_severity?: "info" | "warning" | "error" | string;
  summary?: string;
  source?: string;
};

type GoldenAnalysis = {
  schema_version?: string;
  golden_case: string;
  source_audio?: string;
  audio?: {
    duration_s?: number;
    sample_rate?: number;
    hop_s?: number;
  };
  task_config?: {
    task_type?: string;
    target?: {
      f0_hz?: number;
      note?: string;
      label?: string;
      direction?: string;
    } | null;
    skill_focus?: string | string[] | null;
    strictness?: string;
  };
  analysis_validity?: {
    input_type?: string;
    is_analyzable?: boolean;
    reason_codes?: string[];
  };
  task_result?: {
    status?: string;
    score_status?: string;
    full_song_score?: number | null;
    diagnostic_score?: number | null;
    summary?: string;
    next_exercise_suggestion?: Record<string, string> | null;
  };
  feedback_policy?: {
    allowed_feedback?: string[];
    blocked_feedback?: Array<{ type?: string; reason?: string }>;
    caveats?: string[];
  };
  proxy_metrics?: Record<string, unknown>;
  subscores?: Record<string, unknown>;
  frames?: Frame[];
  segments?: Record<string, Segment[]>;
  display_hints?: {
    debug_fields_included?: boolean;
    user_visible_warning?: string;
  };
  payload?: {
    frame_count?: number;
    example_frame_count?: number;
    payload_strategy?: string;
  };
};

const EXAMPLE_BASE = "/golden-ui-examples/";

const segmentStyles: Record<string, string> = {
  notes: "#ffb1c0",
  phrases: "#3cddc7",
  dropouts: "#ffb4ab",
  unstable_pitch_regions: "#ddb7ff",
  low_confidence_regions: "#ffd166",
  breath_phrase_proxy_regions: "#77ddaa",
  tone_consistency_proxy_regions: "#8ec5ff",
};

function labelFromPath(path: string): string {
  return path
    .split("/")
    .pop()
    ?.replace(".json", "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (value) => value.toUpperCase()) || path;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "none";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.join(", ");
  return String(value).replaceAll("_", " ");
}

function percent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "none";
  return `${Math.round(value * 100)}%`;
}

function selectedExampleUrl(example: string): string {
  return `${EXAMPLE_BASE}${example.split("/").pop()}`;
}

function targetF0(analysis: GoldenAnalysis): number | null {
  const target = analysis.task_config?.target;
  if (target?.f0_hz && target.f0_hz > 0) return target.f0_hz;
  const frameTarget = analysis.frames?.find((frame) => frame.target_f0_hz && frame.target_f0_hz > 0)?.target_f0_hz;
  return frameTarget || null;
}

function frameExtent(frames: Frame[], audioDuration?: number): number {
  const maxFrameTime = frames.reduce((max, frame) => Math.max(max, frame.time_s || 0), 0);
  return Math.max(audioDuration || 0, maxFrameTime, 0.1);
}

function MetricPill({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-lg bg-surface-container-high/70 border border-white/5 px-3 py-2 min-w-0">
      <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">{label}</p>
      <p className="text-sm font-semibold text-white truncate">{formatValue(value)}</p>
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="glass-card rounded-xl p-5 border border-white/5 space-y-4">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">{icon}</div>
        <h3 className="font-display font-bold text-lg text-white">{title}</h3>
      </div>
      {children}
    </section>
  );
}

function TaskResultCard({ analysis }: { analysis: GoldenAnalysis }) {
  const result = analysis.task_result || {};
  const validity = analysis.analysis_validity || {};
  const score = result.full_song_score ?? result.diagnostic_score;
  const scoreLabel = result.full_song_score !== null && result.full_song_score !== undefined ? "Full-song score" : "Diagnostic score";
  const invalid = ["no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"].includes(
    validity.input_type || "",
  );

  return (
    <section className="glass-card rounded-xl p-5 border border-white/5">
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5">
        <div className="space-y-3 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-widest font-bold px-2.5 py-1 rounded-full bg-secondary/10 text-secondary">
              {analysis.task_config?.task_type || "unknown task"}
            </span>
            <span
              className={`text-[10px] uppercase tracking-widest font-bold px-2.5 py-1 rounded-full ${
                invalid ? "bg-error/10 text-error" : "bg-tertiary/10 text-tertiary"
              }`}
            >
              {validity.input_type || "unknown input"}
            </span>
          </div>
          <div>
            <h2 className="font-display text-2xl font-extrabold text-white">AI coach contract preview</h2>
            <p className="text-sm text-on-surface-variant mt-2 leading-relaxed max-w-3xl">
              {result.summary || "No task summary provided."}
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-on-surface-variant">
            <span>Case: {analysis.golden_case?.replaceAll("_", " ")}</span>
            <span className="text-on-surface-variant/40">/</span>
            <span>Source: {analysis.source_audio || "golden fixture"}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 min-w-[260px]">
          <MetricPill label={scoreLabel} value={score ?? "skipped"} />
          <MetricPill label="Score status" value={result.score_status || result.status} />
          <MetricPill label="Analyzable" value={validity.is_analyzable} />
          <MetricPill label="Frames" value={analysis.payload?.frame_count ?? analysis.frames?.length ?? 0} />
        </div>
      </div>
    </section>
  );
}

function PitchLane({ analysis }: { analysis: GoldenAnalysis }) {
  const frames = analysis.frames || [];
  const duration = frameExtent(frames, analysis.audio?.duration_s);
  const target = targetF0(analysis);
  const values = frames
    .map((frame) => frame.f0_hz)
    .filter((value): value is number => typeof value === "number" && value > 0);
  if (target) values.push(target);
  const minF0 = Math.max(40, Math.min(...values, 220) * 0.85);
  const maxF0 = Math.max(260, Math.max(...values, 440) * 1.15);
  const width = 960;
  const height = 260;
  const pad = 28;

  const x = (time: number) => pad + (time / duration) * (width - pad * 2);
  const y = (f0: number) => height - pad - ((f0 - minF0) / Math.max(maxF0 - minF0, 1)) * (height - pad * 2);
  const path = frames
    .filter((frame) => typeof frame.f0_hz === "number" && frame.f0_hz > 0)
    .map((frame, index) => `${index === 0 ? "M" : "L"} ${x(frame.time_s).toFixed(2)} ${y(frame.f0_hz as number).toFixed(2)}`)
    .join(" ");
  const targetY = target ? y(target) : null;

  return (
    <Section title="Pitch Lane" icon={<Music2 className="w-4 h-4" />}>
      <div className="rounded-lg bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[720px] h-[260px]" role="img" aria-label="Pitch lane">
          <rect x="0" y="0" width={width} height={height} rx="10" fill="#0c0e17" />
          {[0, 1, 2, 3].map((line) => {
            const yy = pad + (line / 3) * (height - pad * 2);
            return <line key={line} x1={pad} x2={width - pad} y1={yy} y2={yy} stroke="#32343e" strokeWidth="1" />;
          })}
          {targetY !== null && (
            <>
              <line x1={pad} x2={width - pad} y1={targetY} y2={targetY} stroke="#3cddc7" strokeDasharray="7 7" strokeWidth="2" />
              <text x={width - pad - 120} y={targetY - 8} fill="#3cddc7" fontSize="12" fontWeight="700">
                target {target?.toFixed(1)} Hz
              </text>
            </>
          )}
          {frames.map((frame) => {
            const cx = x(frame.time_s);
            return (
              <rect
                key={`voice-${frame.frame_index}`}
                x={cx - 2}
                y={height - 18}
                width="4"
                height="8"
                rx="2"
                fill={frame.voiced ? "#3cddc7" : "#5b3f44"}
                opacity={frame.voiced ? 0.95 : 0.45}
              />
            );
          })}
          {path && <path d={path} fill="none" stroke="#ffb1c0" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />}
          {frames
            .filter((frame) => typeof frame.f0_hz === "number" && frame.f0_hz > 0)
            .map((frame) => (
              <circle key={`dot-${frame.frame_index}`} cx={x(frame.time_s)} cy={y(frame.f0_hz as number)} r="4" fill="#ffd9df" />
            ))}
          <text x={pad} y={18} fill="#e4bdc3" fontSize="12">
            {Math.round(maxF0)} Hz
          </text>
          <text x={pad} y={height - 28} fill="#e4bdc3" fontSize="12">
            {Math.round(minF0)} Hz
          </text>
        </svg>
      </div>
      <div className="flex flex-wrap gap-2 text-xs text-on-surface-variant">
        <span className="inline-flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-primary" /> selected f0</span>
        <span className="inline-flex items-center gap-1"><span className="w-3 h-1 rounded-full bg-tertiary" /> target f0</span>
        <span className="inline-flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-tertiary" /> voiced</span>
      </div>
    </Section>
  );
}

function VolumeTimeline({ analysis }: { analysis: GoldenAnalysis }) {
  const frames = analysis.frames || [];
  const duration = frameExtent(frames, analysis.audio?.duration_s);
  const width = 960;
  const height = 150;
  const pad = 20;
  const rmsValues = frames.map((frame) => frame.volume?.rms || 0);
  const maxRms = Math.max(...rmsValues, 0.001);
  const x = (time: number) => pad + (time / duration) * (width - pad * 2);

  return (
    <Section title="Volume And Voicing" icon={<Volume2 className="w-4 h-4" />}>
      <div className="rounded-lg bg-surface-container-lowest/70 border border-white/5 p-3 overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[720px] h-[150px]" role="img" aria-label="Volume timeline">
          <rect x="0" y="0" width={width} height={height} rx="10" fill="#0c0e17" />
          {frames.map((frame) => {
            const cx = x(frame.time_s);
            const barHeight = ((frame.volume?.rms || 0) / maxRms) * (height - pad * 2);
            return (
              <g key={frame.frame_index}>
                <rect
                  x={cx - 5}
                  y={height - pad - barHeight}
                  width="10"
                  height={barHeight}
                  rx="5"
                  fill={frame.voiced ? "#3cddc7" : "#ab888e"}
                  opacity={frame.voiced ? 0.9 : 0.35}
                />
                <rect
                  x={cx - 5}
                  y={height - 11}
                  width="10"
                  height="5"
                  rx="2.5"
                  fill={frame.voiced ? "#3cddc7" : "#5b3f44"}
                />
              </g>
            );
          })}
        </svg>
      </div>
    </Section>
  );
}

function SegmentMarkers({ analysis }: { analysis: GoldenAnalysis }) {
  const segments = analysis.segments || {};
  const duration = frameExtent(analysis.frames || [], analysis.audio?.duration_s);
  const groups = Object.entries(segments);

  return (
    <Section title="Segment Markers" icon={<Activity className="w-4 h-4" />}>
      <div className="space-y-3">
        {groups.map(([key, items]) => (
          <div key={key} className="space-y-1.5">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-bold uppercase tracking-wider text-on-surface-variant">{key.replaceAll("_", " ")}</p>
              <span className="text-[11px] text-on-surface-variant">{items.length} markers</span>
            </div>
            <div className="relative h-8 rounded-lg bg-surface-container-lowest border border-white/5 overflow-hidden">
              {items.map((segment, index) => {
                const start = Math.max(0, Number(segment.start_s || 0));
                const duration = Number(segment.duration_s || 0.04);
                const end = Math.max(start + 0.04, Number(segment.end_s || start + duration));
                const left = (start / duration) * 100;
                const width = Math.max(((end - start) / duration) * 100, 1);
                return (
                  <div
                    key={`${segment.id || key}-${index}`}
                    title={`${segment.summary || segment.type || key} ${start.toFixed(2)}-${end.toFixed(2)}s`}
                    className="absolute top-1 bottom-1 rounded"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: segmentStyles[key] || "#ffb1c0",
                      opacity: segment.ui_severity === "warning" ? 0.95 : 0.72,
                    }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function JsonList({ title, values, tone = "info" }: { title: string; values: string[]; tone?: "info" | "blocked" | "allowed" }) {
  const color = tone === "allowed" ? "text-tertiary" : tone === "blocked" ? "text-error" : "text-secondary";
  return (
    <div className="rounded-lg bg-surface-container-high/60 border border-white/5 p-4">
      <h4 className={`text-xs uppercase tracking-wider font-bold ${color}`}>{title}</h4>
      {values.length ? (
        <ul className="mt-3 space-y-2">
          {values.map((value, index) => (
            <li key={`${value}-${index}`} className="text-sm text-on-surface-variant leading-relaxed">
              {value.replaceAll("_", " ")}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-sm text-on-surface-variant">none</p>
      )}
    </div>
  );
}

function FeedbackAndScores({ analysis }: { analysis: GoldenAnalysis }) {
  const feedback = analysis.feedback_policy || {};
  const blocked = (feedback.blocked_feedback || []).map((item) => `${item.type || "blocked"}: ${item.reason || ""}`);
  const subscores = Object.entries(analysis.subscores || {}).filter(([, value]) => typeof value !== "object" || value === null);

  return (
    <div className="grid xl:grid-cols-2 gap-5">
      <Section title="Subscores" icon={<BarChart3 className="w-4 h-4" />}>
        {subscores.length ? (
          <div className="grid sm:grid-cols-2 gap-3">
            {subscores.map(([key, value]) => (
              <div key={key}>
                <MetricPill label={key.replaceAll("_", " ")} value={value as string | number | null} />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-on-surface-variant">No numeric subscores for this example.</p>
        )}
      </Section>

      <Section title="Feedback Policy" icon={<Shield className="w-4 h-4" />}>
        <div className="grid md:grid-cols-2 gap-3">
          <JsonList title="Allowed" values={feedback.allowed_feedback || []} tone="allowed" />
          <JsonList title="Blocked" values={blocked} tone="blocked" />
        </div>
      </Section>

      <div className="xl:col-span-2">
        <Section title="Caveats" icon={<AlertTriangle className="w-4 h-4" />}>
          <div className="grid md:grid-cols-2 gap-3">
            {(feedback.caveats || []).map((caveat, index) => (
              <div key={`${caveat}-${index}`} className="flex gap-3 rounded-lg bg-surface-container-high/60 border border-white/5 p-4">
                <Info className="w-4 h-4 text-secondary mt-0.5 flex-shrink-0" />
                <p className="text-sm text-on-surface-variant leading-relaxed">{caveat}</p>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </div>
  );
}

function DebugPanel({ analysis }: { analysis: GoldenAnalysis }) {
  const [open, setOpen] = useState(false);
  const frames = analysis.frames || [];
  const sourceRows = frames.map((frame) => ({
    t: frame.time_s,
    f0: frame.f0_hz,
    voiced: frame.voiced,
    f0Source: frame.selected_f0_source,
    vadSource: frame.selected_vad_source,
    pitchConfidence: frame.pitch_confidence,
    voiceConfidence: frame.voice_confidence,
  }));

  return (
    <Section title="Selected Source Debug" icon={<SlidersHorizontal className="w-4 h-4" />}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="inline-flex items-center gap-2 rounded-lg bg-surface-container-high border border-white/5 px-4 py-2 text-sm font-bold text-on-surface hover:text-primary transition-colors"
      >
        <Lock className="w-4 h-4" />
        {open ? "Hide source details" : "Show source details"}
      </button>

      {open && (
        <div className="mt-4 overflow-x-auto rounded-lg border border-white/5">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-container-high text-on-surface-variant uppercase tracking-wider">
              <tr>
                <th className="p-3">Time</th>
                <th className="p-3">F0</th>
                <th className="p-3">Voiced</th>
                <th className="p-3">F0 source</th>
                <th className="p-3">VAD source</th>
                <th className="p-3">Pitch conf</th>
                <th className="p-3">Voice conf</th>
              </tr>
            </thead>
            <tbody>
              {sourceRows.map((row, index) => (
                <tr key={`${row.t}-${index}`} className="border-t border-white/5 text-on-surface-variant">
                  <td className="p-3">{row.t.toFixed(2)}s</td>
                  <td className="p-3">{row.f0 ? `${row.f0.toFixed(1)} Hz` : "none"}</td>
                  <td className="p-3">{row.voiced ? "yes" : "no"}</td>
                  <td className="p-3">{row.f0Source || "none"}</td>
                  <td className="p-3">{row.vadSource || "none"}</td>
                  <td className="p-3">{percent(row.pitchConfidence)}</td>
                  <td className="p-3">{percent(row.voiceConfidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

export default function AICoachDebugView() {
  const [examples, setExamples] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [analysis, setAnalysis] = useState<GoldenAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(`${EXAMPLE_BASE}index.json`)
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load golden example index (${response.status})`);
        return response.json() as Promise<GoldenIndex>;
      })
      .then((index) => {
        if (cancelled) return;
        const items = index.examples || [];
        setExamples(items);
        setSelected(items[0] || "");
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(selectedExampleUrl(selected))
      .then((response) => {
        if (!response.ok) throw new Error(`Unable to load ${selected} (${response.status})`);
        return response.json() as Promise<GoldenAnalysis>;
      })
      .then((payload) => {
        if (!cancelled) setAnalysis(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const statusIcon = useMemo(() => {
    const inputType = analysis?.analysis_validity?.input_type || "";
    if (["no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"].includes(inputType)) {
      return <AlertTriangle className="w-5 h-5 text-error" />;
    }
    return <CheckCircle2 className="w-5 h-5 text-tertiary" />;
  }, [analysis]);

  return (
    <div className="space-y-6 animate-fade-in max-w-7xl">
      <div className="flex flex-col xl:flex-row xl:items-end xl:justify-between gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-primary">
            <FileJson className="w-5 h-5" />
            <span className="text-[10px] uppercase tracking-widest font-bold">H5a Debug Viewer</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl text-white">UI-ready analysis viewer</h1>
          <p className="text-sm text-on-surface-variant max-w-3xl leading-relaxed">
            Loads golden AI coach contract JSON and renders task results, frame timelines, segment markers, policy, and gated source diagnostics.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
          <label className="text-xs uppercase tracking-wider font-bold text-on-surface-variant" htmlFor="golden-example">
            Golden example
          </label>
          <select
            id="golden-example"
            value={selected}
            onChange={(event) => setSelected(event.target.value)}
            className="bg-surface-container-high border border-white/10 rounded-lg px-3 py-2 text-sm text-on-surface min-w-[280px] outline-none focus:ring-2 focus:ring-primary/40"
          >
            {examples.map((example) => (
              <option key={example} value={example} className="bg-surface text-on-surface">
                {labelFromPath(example)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-error/20 bg-error/10 p-4 text-error text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="glass-card rounded-xl p-8 text-center text-on-surface-variant">
          Loading golden example...
        </div>
      )}

      {analysis && !loading && (
        <>
          <div className="flex flex-wrap items-center gap-3 text-xs text-on-surface-variant">
            {statusIcon}
            <span>Schema: {analysis.schema_version || "unknown"}</span>
            <span className="text-on-surface-variant/40">/</span>
            <span>Sample rate: {analysis.audio?.sample_rate || "unknown"} Hz</span>
            <span className="text-on-surface-variant/40">/</span>
            <span>Duration: {analysis.audio?.duration_s?.toFixed(2) || "unknown"}s</span>
          </div>

          <TaskResultCard analysis={analysis} />
          <MelSpectrogramView analysis={analysis as any} />
          <div className="grid 2xl:grid-cols-[1.3fr_0.9fr] gap-5">
            <PitchLane analysis={analysis} />
            <VolumeTimeline analysis={analysis} />
            <PosteriorConfidenceMap analysis={analysis as any} />
            <SpectralToneProxyMap analysis={analysis as any} />
          </div>
          <SegmentMarkers analysis={analysis} />
          <FeedbackAndScores analysis={analysis} />
          <DebugPanel analysis={analysis} />

          <Section title="Payload Notes" icon={<ListChecks className="w-4 h-4" />}>
            <div className="grid md:grid-cols-3 gap-3">
              <MetricPill label="Full frame count" value={analysis.payload?.frame_count ?? analysis.frames?.length ?? 0} />
              <MetricPill label="Fixture frames" value={analysis.payload?.example_frame_count ?? analysis.frames?.length ?? 0} />
              <MetricPill label="Debug fields" value={analysis.display_hints?.debug_fields_included ? "included" : "hidden"} />
            </div>
            <p className="text-sm text-on-surface-variant leading-relaxed">{analysis.payload?.payload_strategy}</p>
            {analysis.display_hints?.user_visible_warning && (
              <p className="text-sm text-secondary leading-relaxed">{analysis.display_hints.user_visible_warning}</p>
            )}
          </Section>
        </>
      )}
    </div>
  );
}
