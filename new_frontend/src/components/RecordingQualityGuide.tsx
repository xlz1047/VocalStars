import { AlertTriangle, Mic2, ShieldCheck } from "lucide-react";
import { UiReadyAnalysis } from "../types";

interface RecordingQualityGuideProps {
  analysis?: UiReadyAnalysis | null;
  compact?: boolean;
}

function pct(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "n/a";
}

function ratioFromFrames(analysis: UiReadyAnalysis | null | undefined, flag: string): number | null {
  const frames = analysis?.frames || [];
  if (!frames.length) return null;
  const count = frames.filter((frame) => Boolean(frame.signal_quality?.[flag])).length;
  return count / frames.length;
}

function metricNumber(analysis: UiReadyAnalysis | null | undefined, key: string): number | null {
  const metrics = analysis?.analysis_validity?.summary_metrics || {};
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function buildTips(analysis: UiReadyAnalysis | null | undefined): string[] {
  const reasonCodes = new Set(analysis?.analysis_validity?.reason_codes || []);
  const inputType = analysis?.analysis_validity?.input_type;
  const noisy = ratioFromFrames(analysis, "noisy") ?? 0;
  const clipped = ratioFromFrames(analysis, "clipped") ?? 0;
  const nearSilence = ratioFromFrames(analysis, "near_silence") ?? 0;
  const lowConfidence = ratioFromFrames(analysis, "low_confidence") ?? 0;
  const audioRms = metricNumber(analysis, "audio_rms");

  const tips: string[] = [];
  if (inputType === "no_voice_or_noise" || nearSilence > 0.35 || reasonCodes.has("very_low_audio_rms") || (audioRms !== null && audioRms < 0.003)) {
    tips.push("Move the phone closer and sing one clear vowel or phrase before trying a full melody.");
  }
  if (inputType === "speech_like_or_non_singing") {
    tips.push("Use a sustained sung vowel first, then try the phrase. Speaking-like takes are intentionally not scored as singing.");
  }
  if (noisy > 0.15 || reasonCodes.has("high_pitch_entropy")) {
    tips.push("Reduce fan, road, water, or speaker noise. In a car or bathroom, aim the mic toward your mouth and away from reflective noise.");
  }
  if (clipped > 0.02) {
    tips.push("Sing a little farther from the mic or lower input gain; clipped audio can hide pitch detail.");
  }
  if (lowConfidence > 0.2 || reasonCodes.has("low_pitch_confidence")) {
    tips.push("Try a slower, simpler task such as sustained note or note match so the model can lock onto a stable f0.");
  }
  if (!tips.length) {
    tips.push("Recording quality looks usable. Keep the mic distance steady and avoid backing tracks during analysis.");
  }
  return [...new Set(tips)].slice(0, 4);
}

export default function RecordingQualityGuide({ analysis, compact = false }: RecordingQualityGuideProps) {
  if (!analysis) return null;

  const noisy = ratioFromFrames(analysis, "noisy");
  const clipped = ratioFromFrames(analysis, "clipped");
  const nearSilence = ratioFromFrames(analysis, "near_silence");
  const lowConfidence = ratioFromFrames(analysis, "low_confidence");
  const metrics = analysis.analysis_validity?.summary_metrics || {};
  const tips = buildTips(analysis);
  const hasRisk = (noisy || 0) > 0.15 || (clipped || 0) > 0.02 || (nearSilence || 0) > 0.25 || (lowConfidence || 0) > 0.2;

  return (
    <section className={`glass-card rounded-2xl border border-white/5 ${compact ? "p-4" : "p-5 md:p-6"} space-y-4`}>
      <div className="flex items-start gap-3">
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
          hasRisk ? "bg-primary/10 text-primary" : "bg-tertiary/10 text-tertiary"
        }`}>
          {hasRisk ? <AlertTriangle className="w-5 h-5" /> : <ShieldCheck className="w-5 h-5" />}
        </div>
        <div>
          <h3 className="font-display font-bold text-lg text-white">Recording Quality</h3>
          <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
            Practical checks for noisy rooms, cars, bathrooms, and phone recordings.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <div className="rounded-xl bg-white/5 border border-white/5 p-3">
          <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Noise risk</p>
          <p className="text-sm font-bold text-white mt-1">{pct(noisy)}</p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-3">
          <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Too quiet</p>
          <p className="text-sm font-bold text-white mt-1">{pct(nearSilence)}</p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-3">
          <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Clipping</p>
          <p className="text-sm font-bold text-white mt-1">{pct(clipped)}</p>
        </div>
        <div className="rounded-xl bg-white/5 border border-white/5 p-3">
          <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Low confidence</p>
          <p className="text-sm font-bold text-white mt-1">{pct(lowConfidence)}</p>
        </div>
      </div>

      <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Mic2 className="w-4 h-4 text-secondary" />
          <p className="text-[10px] uppercase tracking-wider font-bold text-secondary">Try this next</p>
        </div>
        <ul className="space-y-1.5">
          {tips.map((tip) => (
            <li key={tip} className="text-xs text-on-surface-variant leading-relaxed">
              {tip}
            </li>
          ))}
        </ul>
      </div>

      {typeof metrics.audio_rms === "number" && (
        <p className="text-[11px] text-on-surface-variant">
          Audio RMS: <span className="text-white font-mono">{metrics.audio_rms.toFixed(5)}</span>
        </p>
      )}
    </section>
  );
}
