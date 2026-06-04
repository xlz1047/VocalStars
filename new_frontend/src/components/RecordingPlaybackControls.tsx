import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Pause, Play, RotateCcw, Square, Target } from "lucide-react";
import { TaskConfig, UiSegment } from "../types";
import { ImprovementFocus } from "../utils/improvementPath";

interface RecordingPlaybackControlsProps {
  audioUrl?: string | null;
  label?: string;
  onTryAgainSameTask?: () => void;
  onBackToTaskSetup?: () => void;
  selectedRegion?: UiSegment | null;
  recommendedFocus?: ImprovementFocus | null;
  onPracticeTask?: (taskConfig: TaskConfig, presetId?: string) => void;
  compact?: boolean;
}

function formatTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "00:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

function formatHz(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)} Hz` : "n/a";
}

function formatCents(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  return `${value > 0 ? "+" : ""}${Math.round(value)} cents`;
}

function formatPercent(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "n/a";
}

export default function RecordingPlaybackControls({
  audioUrl,
  label = "My Recording",
  onTryAgainSameTask,
  onBackToTaskSetup,
  selectedRegion,
  recommendedFocus,
  onPracticeTask,
  compact = false,
}: RecordingPlaybackControlsProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [regionEnd, setRegionEnd] = useState<number | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
      if (regionEnd !== null && audio.currentTime >= regionEnd) {
        audio.pause();
        setIsPlaying(false);
        setRegionEnd(null);
      }
    };
    const handleLoaded = () => setDuration(audio.duration || 0);
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      audio.currentTime = 0;
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("loadedmetadata", handleLoaded);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.pause();
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("loadedmetadata", handleLoaded);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [audioUrl, regionEnd]);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setRegionEnd(null);
  }, [audioUrl]);

  useEffect(() => {
    if (!audioRef.current || !selectedRegion) return;
    const start = Number(selectedRegion.start_s || 0);
    if (!Number.isFinite(start) || start < 0) return;
    audioRef.current.currentTime = start;
    setCurrentTime(start);
    setRegionEnd(null);
  }, [selectedRegion?.id, selectedRegion?.start_s]);

  const play = async () => {
    if (!audioRef.current) return;
    setRegionEnd(null);
    await audioRef.current.play();
    setIsPlaying(true);
  };

  const playSelectedRegion = async () => {
    if (!audioRef.current || !selectedRegion) return;
    const start = Number(selectedRegion.start_s || 0);
    const end = Number(selectedRegion.end_s || start + Math.max(Number(selectedRegion.duration_s || 1), 0.25));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    audioRef.current.currentTime = start;
    setCurrentTime(start);
    setRegionEnd(end);
    await audioRef.current.play();
    setIsPlaying(true);
  };

  const pause = () => {
    audioRef.current?.pause();
    setIsPlaying(false);
  };

  const stop = () => {
    if (!audioRef.current) return;
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
    setIsPlaying(false);
    setRegionEnd(null);
  };

  return (
    <section className={`glass-card rounded-2xl border border-white/5 ${compact ? "p-4" : "p-5 md:p-6"}`}>
      <audio ref={audioRef} src={audioUrl || undefined} preload="metadata" />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-widest font-bold text-on-surface-variant">Playback</p>
          <h3 className="font-display font-bold text-lg text-white mt-1">{label}</h3>
          <p className="text-xs text-on-surface-variant mt-1">
            {audioUrl ? `${formatTime(currentTime)} / ${formatTime(duration)}` : "No recorded take is available yet."}
          </p>
          {selectedRegion && (
            <p className="text-xs text-primary mt-2 leading-relaxed">
              Selected: {selectedRegion.summary || selectedRegion.type || "review region"} ({formatTime(Number(selectedRegion.start_s || 0))}-{formatTime(Number(selectedRegion.end_s || 0))})
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {selectedRegion && (
            <button
              onClick={playSelectedRegion}
              disabled={!audioUrl}
              className="px-4 py-2.5 rounded-xl bg-tertiary/10 border border-tertiary/20 text-xs font-bold text-tertiary hover:bg-tertiary/15 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Play selected region
            </button>
          )}
          <button
            onClick={isPlaying ? pause : play}
            disabled={!audioUrl}
            className="px-4 py-2.5 rounded-xl bg-primary text-on-primary text-xs font-bold hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? "Pause" : "Play my recording"}
          </button>
          <button
            onClick={stop}
            disabled={!audioUrl}
            className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
          {onTryAgainSameTask && (
            <button
              onClick={onTryAgainSameTask}
              className="px-4 py-2.5 rounded-xl bg-tertiary/10 border border-tertiary/20 text-xs font-bold text-tertiary hover:bg-tertiary/15 flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Try again same task
            </button>
          )}
          {onBackToTaskSetup && (
            <button
              onClick={onBackToTaskSetup}
              className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 flex items-center gap-2"
            >
              Try Different Task
              <ArrowLeft className="w-4 h-4 rotate-180" />
            </button>
          )}
          {recommendedFocus && onPracticeTask && (
            <button
              onClick={() => onPracticeTask(recommendedFocus.taskConfig, recommendedFocus.presetId)}
              className="px-4 py-2.5 rounded-xl bg-primary/10 border border-primary/25 text-xs font-bold text-primary hover:bg-primary/15 flex items-center gap-2"
            >
              <Target className="w-4 h-4" />
              Practice this skill
            </button>
          )}
        </div>
      </div>

      {selectedRegion && (
        <div className="mt-4 rounded-xl bg-primary/8 border border-primary/20 p-4">
          <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
            <div className="space-y-2">
              <p className="text-[10px] uppercase tracking-widest font-bold text-primary">
                Selected practice region
              </p>
              <p className="text-sm font-bold text-white">
                {selectedRegion.summary || selectedRegion.type || "Review this region"}
              </p>
              {selectedRegion.actionable_hint && (
                <p className="text-xs text-on-surface-variant leading-relaxed max-w-2xl">
                  {selectedRegion.actionable_hint}
                </p>
              )}
            </div>

            {(selectedRegion.target_f0_hz || selectedRegion.sung_median_f0_hz || selectedRegion.median_cents_error || selectedRegion.f0_coverage) && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 min-w-full lg:min-w-[460px]">
                <div className="rounded-lg bg-white/5 border border-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Target</p>
                  <p className="text-xs font-bold text-white mt-1">{formatHz(selectedRegion.target_f0_hz)}</p>
                </div>
                <div className="rounded-lg bg-white/5 border border-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">You sang</p>
                  <p className="text-xs font-bold text-white mt-1">{formatHz(selectedRegion.sung_median_f0_hz)}</p>
                </div>
                <div className="rounded-lg bg-white/5 border border-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Error</p>
                  <p className="text-xs font-bold text-white mt-1">{formatCents(selectedRegion.median_cents_error)}</p>
                </div>
                <div className="rounded-lg bg-white/5 border border-white/5 p-3">
                  <p className="text-[9px] uppercase tracking-wider font-bold text-on-surface-variant">Coverage</p>
                  <p className="text-xs font-bold text-white mt-1">{formatPercent(selectedRegion.f0_coverage)}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
