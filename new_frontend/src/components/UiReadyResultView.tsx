import { useState } from "react";
import { ArrowLeft, CheckCircle2, Info, Lock, Music2, RotateCcw, Shield, SlidersHorizontal, AlertTriangle } from "lucide-react";
import { CoachingCategories, CoachingCategoryResult, FeedbackPolicy, PerformanceResult, PitchSlideScoreBreakdown, ReferenceAlignment, TaskConfig, TaskResult, UiFrame, UiReadyAnalysis, UiSegment } from "../types";
import { buildImprovementPath, resolvePreset } from "../utils/improvementPath";
import AnalysisCoachingSummary from "./AnalysisCoachingSummary";
import MelSpectrogramView from "./MelSpectrogramView";
import PitchLane from "./PitchLane";
import PosteriorConfidenceMap from "./PosteriorConfidenceMap";
import RecordingPlaybackControls from "./RecordingPlaybackControls";
import RecordingQualityGuide from "./RecordingQualityGuide";
import SegmentMarkerTrack from "./SegmentMarkerTrack";
import SpectralToneProxyMap from "./SpectralToneProxyMap";
import WaveformTimeline from "./WaveformTimeline";

interface UiReadyResultViewProps {
  result: PerformanceResult;
  analysis: UiReadyAnalysis;
  recordingUrl?: string | null;
  recordingLabel?: string | null;
  onReview: () => void;
  onTryAgainSameTask?: () => void;
  onBackToTaskSetup?: () => void;
  onPracticeTask?: (taskConfig: TaskConfig, presetId?: string) => void;
  onBackToDashboard: () => void;
}

function blockedTypes(policy?: FeedbackPolicy): Set<string> {
  return new Set((policy?.blocked_feedback || []).map((item) => String(item.type || "").toLowerCase()));
}

function scoreLabel(taskResult?: TaskResult, policy?: FeedbackPolicy): { label: string; value: number | null } {
  const blocked = blockedTypes(policy);
  if (!blocked.has("full_song_score") && typeof taskResult?.full_song_score === "number") {
    return { label: "Full-song score", value: taskResult.full_song_score };
  }
  if (typeof taskResult?.diagnostic_score === "number") {
    return { label: "Diagnostic score", value: taskResult.diagnostic_score };
  }
  return { label: "Score", value: null };
}

function formatToken(value?: string | null): string {
  if (!value) return "Not available";
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function nextExerciseLabel(next?: Record<string, string> | null): string | null {
  if (!next) return null;
  return next.title || next.name || next.summary || next.description || Object.values(next).find(Boolean) || null;
}

function numericFrameValues(frames: UiFrame[], getter: (frame: UiFrame) => number | null | undefined): number[] {
  return frames
    .map(getter)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function pct(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "n/a";
}

function seconds(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}s` : "n/a";
}

function ReferenceContourSummary({
  alignment,
  frames,
  subscores,
}: {
  alignment?: ReferenceAlignment;
  frames: UiFrame[];
  subscores?: Record<string, any>;
}) {
  const targetFrames = frames.filter((frame) => typeof frame.target_f0_hz === "number" && frame.target_f0_hz > 0);
  const absErrors = numericFrameValues(targetFrames, (frame) => (
    typeof frame.cents_error === "number" ? Math.abs(frame.cents_error) : null
  ));
  const medianAbs = median(absErrors);
  const within50 = absErrors.length ? absErrors.filter((value) => value <= 50).length / absErrors.length : null;
  const within100 = absErrors.length ? absErrors.filter((value) => value <= 100).length / absErrors.length : null;
  const noteResults = Array.isArray(subscores?.note_results) ? subscores.note_results : [];
  const shownNotes = noteResults.slice(0, 8);

  if (!alignment && !targetFrames.length) return null;

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-5">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-tertiary/10 text-tertiary flex items-center justify-center">
          <Music2 className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-display font-bold text-lg text-white">Reference Contour</h3>
          <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
            The target melody is aligned to your detected singing span before comparing f0. This is pitch-contour practice, not full rhythm or song scoring.
          </p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="rounded-xl bg-white/5 border border-white/5 p-4">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Alignment</p>
          <p className="text-sm font-bold text-white mt-1">{formatToken(alignment?.status)}</p>
          <p className="text-[11px] text-on-surface-variant mt-1">{alignment?.method || "reference contour"}</p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-4">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Tempo scale</p>
          <p className="text-sm font-bold text-white mt-1">
            {typeof alignment?.tempo_scale === "number" ? `${alignment.tempo_scale.toFixed(2)}x` : "n/a"}
          </p>
          <p className="text-[11px] text-on-surface-variant mt-1">
            {seconds(alignment?.aligned_duration_s)} sung span
          </p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-4">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Pitch error</p>
          <p className="text-sm font-bold text-white mt-1">
            {medianAbs === null ? "n/a" : `${Math.round(medianAbs)} cents median`}
          </p>
          <p className="text-[11px] text-on-surface-variant mt-1">{pct(within50)} within 50 cents</p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-4">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Reference coverage</p>
          <p className="text-sm font-bold text-white mt-1">
            {pct(typeof subscores?.reference_f0_coverage === "number" ? subscores.reference_f0_coverage : null)}
          </p>
          <p className="text-[11px] text-on-surface-variant mt-1">{pct(within100)} within 100 cents</p>
        </div>
      </div>

      {shownNotes.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-white/5">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-container-high text-on-surface-variant uppercase tracking-wider">
              <tr>
                <th className="p-3">Note</th>
                <th className="p-3">Accuracy</th>
                <th className="p-3">Error</th>
                <th className="p-3">Coverage</th>
                <th className="p-3">Coaching</th>
              </tr>
            </thead>
            <tbody>
              {shownNotes.map((note: Record<string, any>) => {
                const absError = typeof note.median_cents_error === "number" ? Math.abs(note.median_cents_error) : null;
                const accuracy = typeof note.accuracy === "number" ? note.accuracy : null;
                const badge =
                  absError === null ? null :
                  absError <= 25  ? { label: "✓ On pitch",    cls: "bg-emerald-500/15 text-emerald-300 border-emerald-500/25" } :
                  absError <= 75  ? { label: "~ Slightly off", cls: "bg-amber-500/15 text-amber-300 border-amber-500/25" } :
                                   { label: "✗ Off-key",     cls: "bg-rose-500/15 text-rose-300 border-rose-500/25" };
                const errorDir = typeof note.median_cents_error === "number"
                  ? note.median_cents_error > 10 ? " (sharp)" : note.median_cents_error < -10 ? " (flat)" : ""
                  : "";
                return (
                  <tr key={`${note.index}-${note.start_s}`} className="border-t border-white/5 text-on-surface-variant">
                    <td className="p-3 text-white font-bold">{note.note || `#${Number(note.index) + 1}`}</td>
                    <td className="p-3">
                      {badge && (
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold border ${badge.cls}`}>
                          {badge.label}
                        </span>
                      )}
                      {accuracy !== null && <span className="ml-2 text-[10px] text-on-surface-variant">{Math.round(accuracy * 100)}%</span>}
                    </td>
                    <td className="p-3">
                      {absError !== null ? <>{Math.round(absError)}¢{errorDir}<span className="ml-1 text-[10px] text-on-surface-variant/50">({(absError / 100).toFixed(1)} st)</span></> : "n/a"}
                    </td>
                    <td className="p-3">{pct(note.f0_coverage)}</td>
                    <td className="p-3 text-[11px] text-on-surface-variant/70 max-w-[180px]">
                      {note.actionable_hint ||
                        (absError !== null && absError > 75 ? "Try humming this note slowly to find the centre" :
                         absError !== null && absError > 40 ? "Close — focus on landing the note first, then sustain" :
                         "Good — keep the support to stay consistent")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {alignment?.caveat && (
        <p className="rounded-xl bg-secondary/10 border border-secondary/20 p-4 text-sm text-on-surface-variant leading-relaxed">
          <Info className="w-4 h-4 text-secondary inline mr-2" />
          {alignment.caveat}
        </p>
      )}
    </section>
  );
}

function SubScoreBar({ label, value, description }: { label: string; value: number | null | undefined; description?: string }) {
  const pctVal = typeof value === "number" ? Math.round(value * 100) : null;
  const color = pctVal === null ? "bg-white/15"
    : pctVal >= 80 ? "bg-tertiary"
    : pctVal >= 50 ? "bg-amber-400"
    : "bg-red-400";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-bold text-on-surface-variant uppercase tracking-wider">{label}</span>
        <span className={`text-xs font-mono font-bold ${pctVal === null ? "text-on-surface-variant/40" : pctVal >= 80 ? "text-tertiary" : pctVal >= 50 ? "text-amber-300" : "text-red-400"}`}>
          {pctVal === null ? "n/a" : `${pctVal}%`}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pctVal ?? 0}%` }} />
      </div>
      {description && <p className="text-[10px] text-on-surface-variant/60 leading-snug">{description}</p>}
    </div>
  );
}

function PitchSlideBreakdownCard({ breakdown }: { breakdown: PitchSlideScoreBreakdown }) {
  const fb = breakdown.feedback || {};
  const feedbackEntries = Object.entries(fb);
  return (
    <section className="glass-card rounded-2xl p-6 border border-white/5 space-y-5">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center flex-shrink-0">
          <SlidersHorizontal className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-display font-bold text-lg text-white">Pitch Slide Breakdown</h3>
          <p className="text-xs text-on-surface-variant mt-1">Five sub-scores for your slide attempt.</p>
          {breakdown.score_capped && (
            <div className="mt-2 flex items-center gap-1.5 bg-red-900/20 border border-red-400/30 rounded-xl px-3 py-2">
              <AlertTriangle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
              <p className="text-[11px] text-red-300 leading-snug">Score capped at 70 — direction or target note significantly missed.</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <SubScoreBar
          label="Direction"
          value={breakdown.direction_correct ? 1.0 : 0.0}
          description={fb.direction}
        />
        <SubScoreBar
          label="Smoothness"
          value={breakdown.smoothness_score}
          description={fb.smoothness}
        />
        <SubScoreBar
          label="Start Note"
          value={breakdown.start_note_accuracy}
          description={fb.start_note}
        />
        <SubScoreBar
          label="End Note"
          value={breakdown.end_note_accuracy}
          description={fb.end_note}
        />
        {breakdown.contour_deviation_score !== null && (
          <SubScoreBar
            label="Contour Tracking"
            value={breakdown.contour_deviation_score}
            description={fb.contour ?? (breakdown.contour_deviation_cents != null ? `${Math.round(breakdown.contour_deviation_cents)} cents RMS deviation from ideal path` : undefined)}
          />
        )}
      </div>

      {feedbackEntries.length > 0 && (
        <div className="space-y-2">
          {feedbackEntries
            .filter(([, text]) => text)
            .map(([key, text]) => {
              const isGood = text.includes("correct") || text.includes("well matched") || text.includes("fluid") || text.includes("closely");
              return (
                <div key={key} className={`flex gap-2 rounded-xl p-3 text-xs leading-relaxed ${isGood ? "bg-tertiary/8 border border-tertiary/15 text-on-surface-variant" : "bg-amber-900/15 border border-amber-400/20 text-on-surface-variant"}`}>
                  {isGood
                    ? <CheckCircle2 className="w-3.5 h-3.5 text-tertiary flex-shrink-0 mt-0.5" />
                    : <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                  }
                  {text}
                </div>
              );
            })}
        </div>
      )}
    </section>
  );
}

function metricLabel(key: string): string {
  return key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricValue(value: any): string {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return Number.isFinite(value) ? (Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2)) : "n/a";
  return String(value);
}

function CoachingCategoryPanel({
  title,
  result,
}: {
  title: string;
  result?: CoachingCategoryResult;
}) {
  if (!result) return null;
  const complete = result.status === "complete";
  const metrics = Object.entries(result.metrics || {}).slice(0, 6);
  const confidence = typeof result.confidence === "number" ? result.confidence : null;
  return (
    <div className={`rounded-2xl border p-5 space-y-4 ${complete ? "bg-tertiary/8 border-tertiary/20" : "bg-white/5 border-white/5"}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="font-display font-bold text-base text-white">{title}</h4>
          <p className="text-xs text-on-surface-variant mt-1">{complete ? "Evidence-based coaching available" : "Not enough evidence"}</p>
        </div>
        <span className={`rounded-full px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${complete ? "bg-tertiary/15 text-tertiary" : "bg-white/8 text-on-surface-variant"}`}>
          {complete ? "Ready" : "Abstained"}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-black/15 border border-white/5 p-3">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Score</p>
          <p className="text-lg font-black text-white mt-1">{typeof result.score === "number" ? Math.round(result.score) : "n/a"}</p>
        </div>
        <div className="rounded-xl bg-black/15 border border-white/5 p-3">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Confidence</p>
          <p className="text-lg font-black text-white mt-1">{confidence === null ? "n/a" : `${Math.round(confidence * 100)}%`}</p>
        </div>
      </div>

      {metrics.length > 0 && (
        <div className="grid sm:grid-cols-2 gap-2">
          {metrics.map(([key, value]) => (
            <div key={key} className="rounded-xl bg-white/5 border border-white/5 p-3">
              <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">{metricLabel(key)}</p>
              <p className="text-sm font-bold text-white mt-1">{metricValue(value)}</p>
            </div>
          ))}
        </div>
      )}

      {result.recommended_exercise && complete && (
        <p className="rounded-xl bg-primary/10 border border-primary/20 p-3 text-xs text-on-surface-variant leading-relaxed">
          {result.recommended_exercise}
        </p>
      )}
      {result.caveats?.[0] && (
        <p className="text-[11px] text-on-surface-variant/70 leading-relaxed">{result.caveats[0]}</p>
      )}
    </div>
  );
}

function CoachingCategoriesCard({ categories }: { categories?: CoachingCategories }) {
  if (!categories?.vibrato && !categories?.slide) return null;
  return (
    <section className="glass-card rounded-2xl p-6 border border-white/5 space-y-5">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-tertiary/10 text-tertiary flex items-center justify-center flex-shrink-0">
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-display font-bold text-lg text-white">Reliable Coaching Categories</h3>
          <p className="text-xs text-on-surface-variant mt-1">
            Vibrato and slide feedback use interpretable pitch/VAD evidence and abstain when confidence is low.
          </p>
        </div>
      </div>
      <div className="grid lg:grid-cols-2 gap-4">
        <CoachingCategoryPanel title="Vibrato control" result={categories.vibrato} />
        <CoachingCategoryPanel title="Slide control" result={categories.slide} />
      </div>
    </section>
  );
}

function DeveloperSourcePanel({ analysis }: { analysis: UiReadyAnalysis }) {
  const [open, setOpen] = useState(false);
  const frames = analysis.frames || [];
  const sampleRows = frames.slice(0, 120);

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-4">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="inline-flex items-center gap-2 rounded-xl bg-surface-container-high border border-white/5 px-4 py-2 text-sm font-bold text-on-surface hover:text-primary transition-colors"
      >
        <Lock className="w-4 h-4" />
        {open ? "Hide developer source details" : "Show developer source details"}
      </button>

      {open && (
        <div className="space-y-4">
          {analysis.debug && (
            <pre className="max-h-72 overflow-auto rounded-xl bg-[#0c0e17] border border-white/5 p-4 text-[11px] text-on-surface-variant">
              {JSON.stringify(analysis.debug, null, 2)}
            </pre>
          )}

          <div className="overflow-x-auto rounded-xl border border-white/5">
            <table className="w-full text-left text-xs">
              <thead className="bg-surface-container-high text-on-surface-variant uppercase tracking-wider">
                <tr>
                  <th className="p-3">Time</th>
                  <th className="p-3">F0 source</th>
                  <th className="p-3">VAD source</th>
                  <th className="p-3">Pitch conf</th>
                  <th className="p-3">Voice conf</th>
                  <th className="p-3">Debug flags</th>
                  <th className="p-3">Source values</th>
                </tr>
              </thead>
              <tbody>
                {sampleRows.map((frame) => (
                  <tr key={frame.frame_index} className="border-t border-white/5 text-on-surface-variant">
                    <td className="p-3">{frame.time_s.toFixed(2)}s</td>
                    <td className="p-3">{frame.selected_f0_source || "none"}</td>
                    <td className="p-3">{frame.selected_vad_source || "none"}</td>
                    <td className="p-3">{frame.pitch_confidence === null || frame.pitch_confidence === undefined ? "none" : `${Math.round(frame.pitch_confidence * 100)}%`}</td>
                    <td className="p-3">{frame.voice_confidence === null || frame.voice_confidence === undefined ? "none" : `${Math.round(frame.voice_confidence * 100)}%`}</td>
                    <td className="p-3">{frame.debug_flags?.join(", ") || "none"}</td>
                    <td className="p-3 max-w-[280px] truncate">{frame.source_values ? JSON.stringify(frame.source_values) : "hidden"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

export default function UiReadyResultView({
  result,
  analysis,
  recordingUrl,
  recordingLabel,
  onReview,
  onTryAgainSameTask,
  onBackToTaskSetup,
  onPracticeTask,
  onBackToDashboard,
}: UiReadyResultViewProps) {
  const taskResult = analysis.task_result || {};
  const effectiveTaskConfig = analysis.task_config || result.taskConfig;
  const policy = analysis.feedback_policy || {};
  const score = scoreLabel(taskResult, policy);
  const caveats = policy.caveats || [];
  const allowed = policy.allowed_feedback || [];
  const nextExercise = nextExerciseLabel(taskResult.next_exercise_suggestion);
  const improvementPath = buildImprovementPath(analysis);
  const primaryFocus = improvementPath.primaryFocus;
  const primaryPreset = primaryFocus?.presetId ? resolvePreset(primaryFocus.presetId) : undefined;
  const hasFrames = Boolean(analysis.frames?.length);
  const referenceAlignment = analysis.reference_alignment || analysis.subscores?.reference_alignment;
  const [selectedRegion, setSelectedRegion] = useState<UiSegment | null>(null);
  const pitchSlideBreakdown: PitchSlideScoreBreakdown | null =
    (taskResult.task_type === "pitch_slide" || effectiveTaskConfig?.task_type === "pitch_slide")
      ? (analysis.subscores?.pitch_slide_breakdown ?? null)
      : null;

  return (
    <div className="space-y-8 pb-12 animate-fade-in relative z-10">
      <section className="text-center md:text-left flex flex-col md:flex-row justify-between items-center gap-6 bg-surface-container/20 p-6 rounded-2xl border border-white/5">
        <div>
          <h1 className="font-display font-extrabold text-3xl md:text-4xl text-white mb-2">
            Session Complete
          </h1>
          <p className="text-on-surface-variant font-sans text-sm md:text-base">
            Task: <span className="text-secondary font-bold">{formatToken(taskResult.task_type || effectiveTaskConfig?.task_type)}</span>
          </p>
          <div className="mt-2.5 flex items-center justify-center md:justify-start gap-1.5 text-on-surface-variant/50 text-[10px] font-bold tracking-wider uppercase">
            <CheckCircle2 className="w-4.5 h-4.5 text-tertiary" />
            <span>UI-READY TASK ANALYSIS</span>
          </div>
        </div>
        <button
          onClick={onBackToDashboard}
          className="flex items-center gap-2 px-5 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all font-semibold text-xs"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </button>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <section className="lg:col-span-5 glass-card rounded-[24px] p-8 md:p-10 relative overflow-hidden flex flex-col items-center justify-center text-center">
          <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-transparent to-secondary/5 pointer-events-none" />
          <div className="relative w-56 h-56 rounded-full border-[12px] border-surface-container-highest flex flex-col items-center justify-center mb-6">
            <span className="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">{score.label}</span>
            <span className="font-display font-black text-6xl text-primary mt-1">
              {score.value === null ? "—" : Math.round(score.value)}
            </span>
          </div>
          <h2 className="font-display font-extrabold text-xl md:text-2xl text-secondary">
            {formatToken(taskResult.score_status || taskResult.status)}
          </h2>
          <p className="text-xs md:text-sm text-on-surface-variant/90 leading-relaxed max-w-sm mt-3">
            {taskResult.summary || "Task-specific analysis completed."}
          </p>
          {nextExercise && (
            <div className="mt-5 rounded-xl bg-tertiary/10 border border-tertiary/20 p-4 text-left w-full">
              <p className="text-[10px] uppercase tracking-wider font-bold text-tertiary">Next exercise</p>
              <p className="text-sm text-on-surface-variant mt-1 leading-relaxed">{nextExercise}</p>
            </div>
          )}
          {primaryFocus && (
            <div className="mt-4 rounded-xl bg-primary/10 border border-primary/20 p-4 text-left w-full">
              <p className="text-[10px] uppercase tracking-wider font-bold text-primary">Recommended next</p>
              {/* Show preset badge when it's a GTSinger human-vocal recommendation */}
              {primaryPreset?.song.referenceType === "human_vocal" && primaryPreset.song.referenceStyle && (
                <span className="inline-flex items-center gap-1 mt-1 text-[10px] font-bold border rounded-full px-2 py-0.5 bg-tertiary/15 border-tertiary/30 text-tertiary">
                  <span className="w-1.5 h-1.5 rounded-full bg-current" />
                  Human vocal · {primaryPreset.song.referenceStyle}
                </span>
              )}
              <p className="text-sm font-bold text-white mt-1.5">{primaryFocus.label}</p>
              <p className="text-xs text-on-surface-variant mt-1 leading-relaxed">{primaryFocus.summary}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {primaryFocus.evidence.slice(0, 2).map((item) => (
                  <span key={item} className="px-2.5 py-1 rounded-full bg-white/5 border border-white/5 text-[10px] font-bold text-on-surface-variant">
                    {item}
                  </span>
                ))}
              </div>
              {primaryFocus.caveat && (
                <p className="text-[11px] text-on-surface-variant/70 mt-3 leading-relaxed">{primaryFocus.caveat}</p>
              )}
              {onPracticeTask && (
                <button
                  onClick={() => onPracticeTask(primaryFocus.taskConfig, primaryFocus.presetId)}
                  className="mt-4 w-full px-4 py-2.5 rounded-xl bg-primary text-on-primary text-xs font-bold hover:brightness-110 active:scale-95 transition-all flex items-center justify-center gap-2"
                >
                  <Music2 className="w-4 h-4 text-on-primary" />
                  {primaryFocus.practiceLabel}
                </button>
              )}
            </div>
          )}
        </section>

        <section className="lg:col-span-7 glass-card rounded-[24px] p-6 md:p-8 space-y-6">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-tertiary" />
            <h3 className="font-display text-lg font-bold text-on-surface">Feedback Policy</h3>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant mb-3">Allowed feedback</p>
            <div className="flex flex-wrap gap-2">
              {allowed.length ? allowed.map((item) => (
                <span key={item} className="px-3 py-1.5 rounded-full bg-tertiary/10 text-tertiary text-xs font-bold">
                  {formatToken(item)}
                </span>
              )) : (
                <span className="text-sm text-on-surface-variant">No feedback categories were allowed.</span>
              )}
            </div>
          </div>
          {caveats.length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant mb-3">Caveats</p>
              <div className="grid md:grid-cols-2 gap-3">
                {caveats.map((caveat, index) => (
                  <p key={`${caveat}-${index}`} className="rounded-xl bg-secondary/10 border border-secondary/20 p-4 text-sm text-on-surface-variant leading-relaxed">
                    <Info className="w-4 h-4 text-secondary inline mr-2" />
                    {caveat}
                  </p>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      {hasFrames ? (
        <div className="grid xl:grid-cols-2 gap-6">
          <div className="xl:col-span-2">
            <MelSpectrogramView analysis={analysis} />
          </div>
          <PitchLane analysis={analysis} />
          <WaveformTimeline analysis={analysis} />
          <PosteriorConfidenceMap analysis={analysis} />
          <SpectralToneProxyMap analysis={analysis} />
          <div className="xl:col-span-2">
            <SegmentMarkerTrack
              analysis={analysis}
              selectedSegmentId={selectedRegion?.id || null}
              onSelectSegment={(segment) => setSelectedRegion(segment)}
            />
          </div>
        </div>
      ) : (
        <section className="glass-card rounded-2xl p-6 border border-white/5">
          <p className="text-sm text-on-surface-variant">Frame-level analysis was not included in this response.</p>
        </section>
      )}

      <ReferenceContourSummary
        alignment={referenceAlignment}
        frames={analysis.frames || []}
        subscores={analysis.subscores}
      />

      {pitchSlideBreakdown && (
        <PitchSlideBreakdownCard breakdown={pitchSlideBreakdown} />
      )}

      <AnalysisCoachingSummary analysis={analysis} />

      <CoachingCategoriesCard categories={analysis.coaching_categories} />

      <RecordingQualityGuide analysis={analysis} />

      <RecordingPlaybackControls
        audioUrl={recordingUrl}
        label={recordingLabel || "My Recording"}
        selectedRegion={selectedRegion}
        onTryAgainSameTask={onTryAgainSameTask}
        onBackToTaskSetup={onBackToTaskSetup}
        recommendedFocus={primaryFocus}
        onPracticeTask={onPracticeTask}
      />

      <DeveloperSourcePanel analysis={analysis} />

      <section className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
        <button
          onClick={onReview}
          className="w-full sm:w-auto px-8 py-3.5 rounded-full border-2 border-primary text-primary text-sm font-bold hover:bg-primary/10 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
        >
          <SlidersHorizontal className="w-4 h-4" />
          Frame Review
        </button>
        <button
          onClick={onTryAgainSameTask}
          className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary text-sm font-bold hover:brightness-110 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2 glow-pink"
        >
          <RotateCcw className="w-4 h-4 text-on-primary" />
          Try Again
        </button>
        {onBackToTaskSetup && (
          <button
            onClick={onBackToTaskSetup}
            className="w-full sm:w-auto px-8 py-3.5 rounded-full border border-white/15 text-on-surface-variant text-sm font-bold hover:bg-white/5 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
          >
            Try Different Task
            <ArrowLeft className="w-4 h-4 rotate-180" />
          </button>
        )}
      </section>
    </div>
  );
}
