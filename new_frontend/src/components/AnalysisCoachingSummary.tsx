import { CheckCircle2, AlertTriangle, TrendingUp, Volume2, Wind, Info } from "lucide-react";
import { UiFrame, UiReadyAnalysis } from "../types";

interface Props {
  analysis: UiReadyAnalysis;
}

function pct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function centsToPlain(cents: number): string {
  const semitones = cents / 100;
  if (semitones < 0.5) return `${Math.round(cents)} cents (tiny drift)`;
  if (semitones < 1) return `${Math.round(cents)} cents (about half a semitone)`;
  return `${semitones.toFixed(1)} semitones`;
}

function deriveCoachingAdvice(
  within50Frac: number | null,
  medianAbsError: number | null,
  f0Coverage: number | null,
  stabilityMax: number | null,
): { icon: React.ReactNode; title: string; body: string }[] {
  const tips: { icon: React.ReactNode; title: string; body: string }[] = [];

  if (within50Frac !== null) {
    if (within50Frac >= 0.8) {
      tips.push({
        icon: <CheckCircle2 className="w-4 h-4 text-emerald-400" />,
        title: "Great pitch accuracy!",
        body: "You're hitting the target pitch most of the time. Keep using breath support to stay consistent.",
      });
    } else if (within50Frac >= 0.5) {
      tips.push({
        icon: <TrendingUp className="w-4 h-4 text-amber-400" />,
        title: "Pitch is close — tighten it up",
        body: "Try singing the phrase slowly with a piano or reference track, focusing on landing each note centre-first before adding expression.",
      });
    } else {
      tips.push({
        icon: <AlertTriangle className="w-4 h-4 text-rose-400" />,
        title: "Pitch needs work",
        body: "Start with sustained single notes — hold each one for 5 seconds while watching the pitch lane. Accuracy builds before speed.",
      });
    }
  }

  if (medianAbsError !== null && medianAbsError > 50) {
    const dir = medianAbsError > 0 ? "sharp (too high)" : "flat (too low)";
    tips.push({
      icon: <Info className="w-4 h-4 text-sky-400" />,
      title: `You tend to sing ${dir}`,
      body: `Your pitch was off by about ${centsToPlain(Math.abs(medianAbsError))} on average. Try humming the note quietly first, then open into full voice.`,
    });
  }

  if (f0Coverage !== null && f0Coverage < 0.6) {
    tips.push({
      icon: <Wind className="w-4 h-4 text-violet-400" />,
      title: "Hold your notes longer",
      body: "The system detected your voice for less than 60% of the target phrase. Take a deeper breath before each phrase and sustain through to the end.",
    });
  }

  if (stabilityMax !== null && stabilityMax > 40) {
    tips.push({
      icon: <TrendingUp className="w-4 h-4 text-amber-400" />,
      title: "Pitch wobbled on some notes",
      body: "Instability usually comes from tension. Drop your shoulders, relax your jaw, and practise sustained tones (see Exercises → Straight Tone).",
    });
  }

  return tips.slice(0, 3);
}

export default function AnalysisCoachingSummary({ analysis }: Props) {
  const frames = analysis.frames ?? [];
  const subscores = (analysis as any).subscores ?? (analysis as any).task_result?.subscores ?? {};
  const signalQuality = (analysis as any).signal_quality ?? {};
  const noteResults: any[] = Array.isArray(subscores?.note_results) ? subscores.note_results : [];

  const targetFrames = frames.filter(
    (f: UiFrame) => typeof f.target_f0_hz === "number" && f.target_f0_hz > 0,
  );

  if (!targetFrames.length && !noteResults.length) return null;

  const absErrors = targetFrames
    .map((f: UiFrame) => (typeof f.cents_error === "number" ? Math.abs(f.cents_error) : null))
    .filter((v): v is number => v !== null);

  const within50Frac = absErrors.length
    ? absErrors.filter((v) => v <= 50).length / absErrors.length
    : null;
  const within100Frac = absErrors.length
    ? absErrors.filter((v) => v <= 100).length / absErrors.length
    : null;

  const sorted = [...absErrors].sort((a, b) => a - b);
  const medianAbsError =
    sorted.length
      ? sorted.length % 2 === 0
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
        : sorted[Math.floor(sorted.length / 2)]
      : null;

  const notesTotal = noteResults.length;
  const notesAccurate = noteResults.filter(
    (n: any) => typeof n.accuracy === "number" && n.accuracy >= 0.5,
  ).length;

  const coverageValues = noteResults
    .map((n: any) => n.f0_coverage)
    .filter((v): v is number => typeof v === "number");
  const avgCoverage = coverageValues.length
    ? coverageValues.reduce((a, b) => a + b, 0) / coverageValues.length
    : null;

  const stabilityValues = noteResults
    .map((n: any) => n.stability_cents)
    .filter((v): v is number => typeof v === "number");
  const maxStability = stabilityValues.length ? Math.max(...stabilityValues) : null;

  const tips = deriveCoachingAdvice(within50Frac, medianAbsError, avgCoverage, maxStability);

  const accuracyColor =
    within50Frac === null
      ? "text-on-surface-variant"
      : within50Frac >= 0.8
      ? "text-emerald-400"
      : within50Frac >= 0.5
      ? "text-amber-400"
      : "text-rose-400";

  return (
    <section className="glass-card rounded-2xl p-5 border border-white/5 space-y-5">
      {/* header */}
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0">
          <CheckCircle2 className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-display font-bold text-lg text-white">How did I do?</h3>
          <p className="text-xs text-on-surface-variant mt-0.5">Plain-English breakdown of your performance</p>
        </div>
      </div>

      {/* stat row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {within50Frac !== null && (
          <div className="rounded-xl bg-white/5 border border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">On-pitch frames</p>
            <p className={`text-2xl font-bold mt-1 ${accuracyColor}`}>{pct(within50Frac)}</p>
            <p className="text-[10px] text-on-surface-variant mt-1">within ±50 cents of target</p>
          </div>
        )}
        {within100Frac !== null && (
          <div className="rounded-xl bg-white/5 border border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Close enough</p>
            <p className="text-2xl font-bold mt-1 text-sky-400">{pct(within100Frac)}</p>
            <p className="text-[10px] text-on-surface-variant mt-1">within ±100 cents (1 semitone)</p>
          </div>
        )}
        {notesTotal > 0 && (
          <div className="rounded-xl bg-white/5 border border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Notes accurate</p>
            <p className="text-2xl font-bold mt-1 text-violet-400">
              {notesAccurate}<span className="text-sm font-normal text-on-surface-variant"> / {notesTotal}</span>
            </p>
            <p className="text-[10px] text-on-surface-variant mt-1">notes hit at ≥50% accuracy</p>
          </div>
        )}
        {medianAbsError !== null && (
          <div className="rounded-xl bg-white/5 border border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Typical error</p>
            <p className="text-2xl font-bold mt-1 text-on-surface">{Math.round(medianAbsError)}¢</p>
            <p className="text-[10px] text-on-surface-variant mt-1">
              {medianAbsError <= 25 ? "pro level" : medianAbsError <= 50 ? "good" : medianAbsError <= 100 ? "needs work" : "off-key"} · 100¢ = 1 semitone
            </p>
          </div>
        )}
      </div>

      {/* volume / signal flags */}
      {(signalQuality.clipped || signalQuality.near_silence || signalQuality.noisy) && (
        <div className="flex flex-wrap gap-2">
          {signalQuality.clipped && (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300">
              <Volume2 className="w-3 h-3" /> Too loud — mic was clipping. Back away slightly next time.
            </span>
          )}
          {signalQuality.near_silence && (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300">
              <Volume2 className="w-3 h-3" /> Very quiet recording. Sing closer to the mic or project more.
            </span>
          )}
          {signalQuality.noisy && (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500/10 border border-sky-500/20 text-xs text-sky-300">
              <Wind className="w-3 h-3" /> Background noise detected — find a quieter space for best results.
            </span>
          )}
        </div>
      )}

      {/* coaching tips */}
      {tips.length > 0 && (
        <div className="space-y-2">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">What to work on</p>
          {tips.map((tip, i) => (
            <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-white/4 border border-white/6">
              <span className="mt-0.5 flex-shrink-0">{tip.icon}</span>
              <div>
                <p className="text-sm font-semibold text-on-surface">{tip.title}</p>
                <p className="text-xs text-on-surface-variant mt-0.5 leading-relaxed">{tip.body}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
