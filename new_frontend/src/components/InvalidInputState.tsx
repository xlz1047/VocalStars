import { AlertCircle, ArrowLeft, RotateCcw, Shield } from "lucide-react";
import { PerformanceResult } from "../types";
import RecordingPlaybackControls from "./RecordingPlaybackControls";
import RecordingQualityGuide from "./RecordingQualityGuide";

interface InvalidInputStateProps {
  result: PerformanceResult;
  recordingUrl?: string | null;
  recordingLabel?: string | null;
  onRetake: () => void;
  onTryAgainSameTask?: () => void;
  onBackToTaskSetup?: () => void;
  onBackToDashboard: () => void;
}

function messageForInputType(inputType?: string): { title: string; body: string } {
  if (inputType === "no_voice_or_noise") {
    return {
      title: "No analyzable singing detected",
      body: "This take looked like silence or background noise, so no singing score or coaching was generated.",
    };
  }
  if (inputType === "speech_like_or_non_singing") {
    return {
      title: "Speech or non-singing voice detected",
      body: "This sounds like speech or non-singing voice, so singing coaching was not generated.",
    };
  }
  if (inputType === "low_confidence_or_unreliable") {
    return {
      title: "Analysis confidence was too low",
      body: "The audio was too noisy or unreliable to score confidently.",
    };
  }
  return {
    title: "Analysis unavailable",
    body: "We could not analyze this take. No singing score or coaching was generated.",
  };
}

export default function InvalidInputState({
  result,
  recordingUrl,
  recordingLabel,
  onRetake,
  onTryAgainSameTask,
  onBackToTaskSetup,
  onBackToDashboard,
}: InvalidInputStateProps) {
  const validity = result.uiReadyAnalysis?.analysis_validity;
  const taskResult = result.uiReadyAnalysis?.task_result;
  const feedback = result.uiReadyAnalysis?.feedback_policy;
  const taskType = result.uiReadyAnalysis?.task_config?.task_type || result.taskConfig?.task_type;
  const message = result.analysisUnavailable
    ? { title: "Analysis unavailable", body: result.analysisError || "No singing score or coaching was generated." }
    : messageForInputType(validity?.input_type);

  return (
    <div className="space-y-8 pb-12 animate-fade-in">
      <section className="glass-card rounded-2xl border border-white/5 p-8 md:p-10 max-w-4xl mx-auto">
        <div className="flex flex-col md:flex-row gap-6 md:items-start">
          <div className="w-14 h-14 rounded-2xl bg-primary/10 text-primary flex items-center justify-center flex-shrink-0">
            <AlertCircle className="w-7 h-7" />
          </div>
          <div className="space-y-4 flex-1">
            <div>
              <p className="text-[10px] uppercase tracking-widest font-bold text-on-surface-variant">
                Safe analysis state
              </p>
              <h1 className="font-display font-extrabold text-3xl text-white mt-2">{message.title}</h1>
            </div>
            <p className="text-sm text-on-surface-variant leading-relaxed max-w-2xl">
              {taskResult?.summary || message.body}
            </p>
            <div className="grid sm:grid-cols-2 gap-3">
              {taskType && (
                <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
                  <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Task</p>
                  <p className="text-lg font-bold text-white mt-1">{taskType.replaceAll("_", " ")}</p>
                </div>
              )}
              <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
                <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Full-song score</p>
                <p className="text-lg font-bold text-white mt-1">Not produced</p>
              </div>
              <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
                <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Input type</p>
                <p className="text-lg font-bold text-white mt-1">{validity?.input_type?.replaceAll("_", " ") || "unavailable"}</p>
              </div>
            </div>

            {(feedback?.caveats || []).length > 0 && (
              <div className="rounded-xl bg-secondary/10 border border-secondary/20 p-4">
                <p className="text-[10px] uppercase tracking-wider font-bold text-secondary mb-2">Caveats</p>
                <ul className="space-y-1.5">
                  {feedback?.caveats?.map((caveat, index) => (
                    <li key={`${caveat}-${index}`} className="text-xs text-on-surface-variant leading-relaxed">
                      {caveat}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(feedback?.blocked_feedback || []).length > 0 && (
              <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Shield className="w-4 h-4 text-tertiary" />
                  <p className="text-[10px] uppercase tracking-wider font-bold text-tertiary">Blocked feedback</p>
                </div>
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  Singing score, note-specific advice, exercises, and diagnosis-style feedback are blocked for this take.
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      <RecordingPlaybackControls
        audioUrl={recordingUrl}
        label={recordingLabel || "My Recording"}
        onTryAgainSameTask={onTryAgainSameTask}
        onBackToTaskSetup={onBackToTaskSetup}
      />

      {result.uiReadyAnalysis && (
        <div className="max-w-4xl mx-auto">
          <RecordingQualityGuide analysis={result.uiReadyAnalysis} />
        </div>
      )}

      <section className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={onTryAgainSameTask || onRetake}
          className="w-full sm:w-auto px-8 py-3.5 rounded-full border-2 border-primary text-primary text-sm font-bold hover:bg-primary/10 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Try Again
        </button>
        <button
          onClick={onBackToDashboard}
          className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-on-surface text-sm font-bold active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </button>
      </section>
    </div>
  );
}
