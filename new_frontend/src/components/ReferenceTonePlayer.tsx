import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw, Volume2, VolumeX, Mic2 } from "lucide-react";
import { PracticeSessionState, ReferenceAudioType, TaskConfig } from "../types";
import { getReferenceTonePlan, playReferenceTone, ReferenceTonePlayback } from "../utils/referenceTone";
import type { GtsingerPhrase } from "../utils/useGtsingerCatalog";

const _API_BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** Ensure backend-relative audio paths include the full origin. */
function resolveAudioUrl(url: string | undefined): string | undefined {
  if (!url) return url;
  if (url.startsWith("/api/")) return `${_API_BASE}${url}`;
  return url;
}

const REFERENCE_TYPE_BADGE: Record<ReferenceAudioType, { label: string; color: string }> = {
  human_vocal:    { label: "Human vocal recording",   color: "bg-tertiary/20 border-tertiary/40 text-tertiary" },
  synth_melody:   { label: "Melody guide (synth)",     color: "bg-amber-500/15 border-amber-500/35 text-amber-300" },
  generated_tone: { label: "Reference tone (synth)",   color: "bg-white/8 border-white/15 text-on-surface-variant" },
  none:           { label: "No reference",             color: "bg-white/5 border-white/10 text-on-surface-variant/60" },
};

interface ReferenceTonePlayerProps {
  taskConfig?: TaskConfig | null;
  disabled?: boolean;
  onPlaybackStateChange?: (state: PracticeSessionState) => void;
  /** Real vocal reference audio URL (served by backend). When set, renders an
   *  audio element instead of synthesising tones. */
  referenceAudioUrl?: string;
  /** Short label like "Vibrato – Female Alto" shown next to the player. */
  referenceStyle?: string;
  /** How the reference audio was produced — used to show a source badge. */
  referenceType?: ReferenceAudioType;
  /** All available phrase clips for this song from the catalog.
   *  When provided and length > 1, a phrase-picker row is shown. */
  catalogClips?: GtsingerPhrase[];
}

export default function ReferenceTonePlayer({
  taskConfig,
  disabled = false,
  onPlaybackStateChange,
  referenceAudioUrl,
  referenceStyle,
  referenceType,
  catalogClips,
}: ReferenceTonePlayerProps) {
  const [selectedClipUrl, setSelectedClipUrl] = useState<string | null>(null);
  const activeAudioUrl = resolveAudioUrl(selectedClipUrl ?? referenceAudioUrl);

  // Reset phrase selection whenever the song changes so stale clip URLs
  // from a previous song don't carry over to the new reference audio.
  useEffect(() => {
    setSelectedClipUrl(null);
  }, [referenceAudioUrl]);
  const [volume, setVolume] = useState(0.45);
  const [isPlaying, setIsPlaying] = useState(false);
  const playbackRef = useRef<ReferenceTonePlayback | null>(null);
  const plan = useMemo(() => getReferenceTonePlan(taskConfig), [taskConfig]);

  const stop = () => {
    playbackRef.current?.stop();
    playbackRef.current = null;
    setIsPlaying(false);
    onPlaybackStateChange?.("ready");
  };

  useEffect(() => {
    return () => {
      playbackRef.current?.stop();
      playbackRef.current = null;
    };
  }, []);

  useEffect(() => {
    stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan?.label, plan?.startF0Hz, plan?.endF0Hz, plan?.durationSeconds, plan?.notes?.length]);

  const effectiveType: ReferenceAudioType = referenceType ?? (referenceAudioUrl ? "human_vocal" : "generated_tone");
  const badge = REFERENCE_TYPE_BADGE[effectiveType];

  // ── Real vocal reference (GTSinger audio) ────────────────────────────────
  if (activeAudioUrl) {
    return (
      <section className={`glass-card rounded-2xl p-5 border border-tertiary/20 bg-tertiary/5 ${disabled ? "opacity-70" : ""}`}>
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-xl bg-tertiary/15 text-tertiary flex items-center justify-center flex-shrink-0">
              <Mic2 className="w-5 h-5" />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-widest font-bold text-tertiary">Reference Prompt</p>
              <p className="text-sm text-white font-bold mt-0.5">Listen first, then record yourself singing along</p>
              {referenceStyle && (
                <p className="text-[11px] text-on-surface-variant mt-1">Style: {referenceStyle}</p>
              )}
            </div>
          </div>
          <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider border rounded-full px-2.5 py-1 whitespace-nowrap ${badge.color}`}>
            {effectiveType === "human_vocal" && <span className="w-1.5 h-1.5 rounded-full bg-current" />}
            {badge.label}
          </span>
        </div>
        {/* Phrase picker — only shown when multiple clips are available */}
        {catalogClips && catalogClips.length > 1 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            <span className="text-[10px] font-bold uppercase tracking-wider text-on-surface-variant self-center mr-1">Phrase:</span>
            {catalogClips.map((clip, i) => {
              const isSelected = (selectedClipUrl ?? referenceAudioUrl) === clip.audio_url;
              return (
                <button
                  key={clip.id}
                  onClick={() => setSelectedClipUrl(resolveAudioUrl(clip.audio_url) ?? clip.audio_url)}
                  className={`px-2.5 py-1 rounded-lg text-[10px] font-bold border transition-all ${
                    isSelected
                      ? "bg-tertiary/20 border-tertiary/40 text-tertiary"
                      : "bg-white/5 border-white/10 text-on-surface-variant hover:bg-white/10"
                  }`}
                >
                  {i + 1}
                </button>
              );
            })}
          </div>
        )}
        <audio
          key={activeAudioUrl}
          src={activeAudioUrl}
          controls
          className="w-full rounded-xl accent-tertiary"
          style={{ colorScheme: "dark" }}
          onPlay={() => onPlaybackStateChange?.("listening_to_reference")}
          onPause={() => onPlaybackStateChange?.("ready")}
          onEnded={() => onPlaybackStateChange?.("ready")}
        />
        {disabled && (
          <p className="text-[11px] text-primary mt-2">
            Pause the reference before recording so it does not bleed into your microphone.
          </p>
        )}
      </section>
    );
  }

  if (!plan) {
    return (
      <section className="glass-card rounded-2xl p-5 border border-secondary/15 bg-secondary/5">
        <p className="text-[10px] uppercase tracking-widest font-bold text-secondary">Reference Prompt</p>
        <h3 className="font-display font-bold text-lg text-white mt-1">No reference tone for this task</h3>
        <p className="text-xs text-on-surface-variant leading-relaxed mt-2">
          Free singing is evaluated as general practice and does not claim reference-melody accuracy.
        </p>
      </section>
    );
  }

  const play = () => {
    if (disabled) return;
    stop();
    const playback = playReferenceTone(plan, volume);
    playbackRef.current = playback;
    setIsPlaying(true);
    onPlaybackStateChange?.("listening_to_reference");
    void playback.finished.then(() => {
      if (playbackRef.current === playback) {
        playbackRef.current = null;
        setIsPlaying(false);
        onPlaybackStateChange?.("ready");
      }
    });
  };

  return (
    <section className={`glass-card rounded-2xl p-5 border border-white/5 ${disabled ? "opacity-70" : ""}`}>
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <p className="text-[10px] uppercase tracking-widest font-bold text-tertiary">Reference Prompt</p>
            <span className={`inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider border rounded-full px-2.5 py-0.5 ${badge.color}`}>
              {badge.label}
            </span>
          </div>
          <h3 className="font-display font-bold text-lg text-white mt-1">{plan.label}</h3>
          <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
            {plan.kind === "note_sequence"
              ? `${plan.notes?.length || 0} notes over ${plan.durationSeconds.toFixed(1)}s.`
              : plan.kind === "pitch_slide"
              ? `${plan.startF0Hz.toFixed(0)} Hz to ${plan.endF0Hz?.toFixed(0)} Hz over ${plan.durationSeconds}s.`
              : `${plan.startF0Hz.toFixed(2)} Hz for ${plan.durationSeconds}s.`}
          </p>
          {plan.kind === "note_sequence" && Boolean(plan.notes?.length) && (
            <p className="text-[11px] text-on-surface-variant mt-2">
              {(plan.notes || []).map((note) => note.label || `${note.f0Hz.toFixed(0)} Hz`).join(" · ")}
            </p>
          )}
          {disabled && (
            <p className="text-[11px] text-primary mt-2">
              Reference tones are stopped before recording by default. Return to setup to replay the prompt.
            </p>
          )}
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <div className="flex items-center gap-2 min-w-[180px]">
            {volume <= 0 ? <VolumeX className="w-4 h-4 text-on-surface-variant" /> : <Volume2 className="w-4 h-4 text-on-surface-variant" />}
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(volume * 100)}
              onChange={(event) => setVolume(Number(event.target.value) / 100)}
              disabled={disabled}
              className="w-full accent-tertiary h-1 bg-white/10 rounded-full appearance-none cursor-pointer disabled:opacity-50"
              aria-label="Reference tone volume"
            />
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={isPlaying ? stop : play}
              disabled={disabled}
              className="px-4 py-2.5 rounded-xl bg-tertiary/15 border border-tertiary/25 text-tertiary text-xs font-bold hover:bg-tertiary/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? "Stop" : "Play"}
            </button>
            <button
              onClick={play}
              disabled={disabled}
              className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-on-surface text-xs font-bold hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Replay
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
