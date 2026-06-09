import { useState, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Play,
  Pause,
  Mic,
  Music,
  Wind,
  Zap,
  Volume2,
  ChevronRight,
  Loader2,
  AlertCircle,
  Check,
} from "lucide-react";
import { PracticePreset, Song, TaskConfig } from "../types";
import { PRACTICE_PRESETS } from "../utils/musicDb";
import { useGtsingerCatalog, findAudioUrlForSong } from "../utils/useGtsingerCatalog";

interface ExercisesViewProps {
  onStartPracticePreset: (preset: PracticePreset) => void;
  onSelectSong: (song: Song) => void;
}

// ── helpers ──────────────────────────────────────────────────────────────────

type Category = "all" | "pitch" | "slide" | "technique" | "gtsinger";

const CATEGORY_LABELS: Record<Category, string> = {
  all: "All",
  pitch: "Pitch & Tone",
  slide: "Slides",
  technique: "Technique",
  gtsinger: "Reference Tracks",
};

const TECHNIQUE_COLORS: Record<string, string> = {
  vibrato: "text-pink-400 bg-pink-400/10 border-pink-400/20",
  breathy: "text-sky-400 bg-sky-400/10 border-sky-400/20",
  glissando: "text-violet-400 bg-violet-400/10 border-violet-400/20",
  mixed_voice: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  mixed_voice_and_falsetto: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  pharyngeal: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  pitch: "text-teal-400 bg-teal-400/10 border-teal-400/20",
  slide: "text-indigo-400 bg-indigo-400/10 border-indigo-400/20",
};

const DIFFICULTY_COLORS: Record<string, string> = {
  EASY: "text-emerald-400",
  MEDIUM: "text-amber-400",
  HARD: "text-rose-400",
};

function techniqueLabel(raw: string): string {
  return raw
    .replace(/_and_/gi, " & ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function singerLabel(raw: string): string {
  return raw
    .replace("EN-Alto-1", "Alto I")
    .replace("EN-Alto-2", "Alto II")
    .replace("EN-Tenor-1", "Tenor I");
}

// ── small audio preview player ───────────────────────────────────────────────

function AudioPreview({ url, label }: { url: string; label: string }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState(false);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (playing) {
      el.pause();
      setPlaying(false);
    } else {
      el.currentTime = 0;
      el.play().then(() => setPlaying(true)).catch(() => setError(true));
    }
  };

  return (
    <div className="flex items-center gap-2">
      <audio
        ref={audioRef}
        src={url}
        onEnded={() => setPlaying(false)}
        onError={() => { setError(true); setPlaying(false); }}
        className="hidden"
      />
      <button
        onClick={toggle}
        disabled={error}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
          error
            ? "opacity-40 cursor-not-allowed bg-white/5 text-on-surface-variant"
            : playing
            ? "bg-primary/20 text-primary border border-primary/30"
            : "bg-white/8 hover:bg-white/12 text-on-surface-variant hover:text-on-surface border border-white/10"
        }`}
        title={error ? "Audio unavailable" : playing ? "Pause preview" : "Preview reference audio"}
      >
        {error ? (
          <AlertCircle className="w-3 h-3" />
        ) : playing ? (
          <Pause className="w-3 h-3" />
        ) : (
          <Play className="w-3 h-3" />
        )}
        {playing ? "Playing…" : error ? "Unavailable" : label}
      </button>
    </div>
  );
}

// ── preset card ───────────────────────────────────────────────────────────────

function PresetCard({
  preset,
  onStart,
  isGtSinger = false,
}: {
  preset: PracticePreset;
  onStart: () => void;
  isGtSinger?: boolean;
}) {
  const technique = preset.taskConfig?.skill_focus?.[0] ?? preset.category;
  const colorClass = TECHNIQUE_COLORS[technique.toLowerCase()] ?? TECHNIQUE_COLORS.pitch;
  const audioUrl = preset.song.referenceAudioUrl;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97 }}
      className="group relative glass-card rounded-2xl border border-white/8 hover:border-primary/30 transition-all duration-300 overflow-hidden"
    >
      {/* accent strip */}
      <div className={`absolute left-0 top-0 bottom-0 w-0.5 rounded-l-2xl ${colorClass.split(" ")[2].replace("border-", "bg-").replace("/20", "/60")}`} />

      <div className="p-5 pl-6">
        {/* header row */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1.5 flex-wrap">
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider border ${colorClass}`}>
                {techniqueLabel(technique)}
              </span>
              <span className={`text-[10px] font-bold uppercase ${DIFFICULTY_COLORS[preset.difficulty] ?? "text-on-surface-variant"}`}>
                {preset.difficulty}
              </span>
              {isGtSinger && (
                <span className="inline-flex items-center gap-1 text-[10px] text-on-surface-variant/60">
                  <Music className="w-2.5 h-2.5" />
                  Human vocal
                </span>
              )}
            </div>
            <h3 className="font-semibold text-sm text-on-surface leading-snug">{preset.title}</h3>
            {preset.song.artist && (
              <p className="text-[11px] text-on-surface-variant/60 mt-0.5 truncate">{preset.song.artist}</p>
            )}
          </div>
          <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-gradient-to-br from-secondary/20 to-primary/20 flex items-center justify-center">
            {isGtSinger ? <Volume2 className="w-4 h-4 text-primary" /> : <Mic className="w-4 h-4 text-secondary" />}
          </div>
        </div>

        {/* description */}
        <p className="text-[11px] text-on-surface-variant/70 leading-relaxed mb-4 line-clamp-2">
          {preset.description}
        </p>

        {/* audio preview + practice button */}
        <div className="flex items-center justify-between gap-2">
          {audioUrl ? (
            <AudioPreview
              url={`${(import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000"}${audioUrl}`}
              label="Listen"
            />
          ) : (
            <div />
          )}
          <button
            onClick={onStart}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-secondary to-primary text-on-primary text-xs font-bold hover:brightness-110 hover:shadow-[0_0_16px_rgba(255,177,192,0.4)] active:scale-95 transition-all duration-200"
          >
            <Mic className="w-3 h-3" />
            Practice
            <ChevronRight className="w-3 h-3" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}

// ── GTSinger dynamic catalog section ─────────────────────────────────────────

function GtSingerCatalogSection({
  catalog,
  onSelectSong,
}: {
  catalog: ReturnType<typeof useGtsingerCatalog>;
  onSelectSong: (song: Song) => void;
}) {
  const [playingUrl, setPlayingUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [filter, setFilter] = useState<string>("all");
  const apiBase = (import.meta as any).env?.VITE_API_URL ?? "http://localhost:8000";

  if (!catalog) {
    return (
      <div className="flex items-center gap-3 py-8 justify-center text-on-surface-variant/50">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">Loading GTSinger catalog…</span>
      </div>
    );
  }

  const techniques = Array.from(new Set(catalog.songs.map((s) => s.technique)));
  const filtered = filter === "all" ? catalog.songs : catalog.songs.filter((s) => s.technique === filter);

  const togglePlay = (url: string) => {
    const el = audioRef.current;
    if (!el) return;
    if (playingUrl === url) {
      el.pause();
      setPlayingUrl(null);
    } else {
      el.src = `${apiBase}${url}`;
      el.play().then(() => setPlayingUrl(url)).catch(() => {});
    }
  };

  return (
    <div className="space-y-5">
      <audio ref={audioRef} onEnded={() => setPlayingUrl(null)} className="hidden" />

      {/* technique filter pills */}
      <div className="flex flex-wrap gap-2">
        {["all", ...techniques].map((t) => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
              filter === t
                ? "bg-primary/20 text-primary border border-primary/40"
                : "bg-white/5 text-on-surface-variant hover:bg-white/10 border border-white/8"
            }`}
          >
            {t === "all" ? "All Techniques" : techniqueLabel(t)}
          </button>
        ))}
      </div>

      {/* song grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <AnimatePresence mode="popLayout">
          {filtered.map((song) => {
            const isPlaying = playingUrl === song.default_audio_url;
            const colorKey = song.technique.toLowerCase();
            const colorClass = TECHNIQUE_COLORS[colorKey] ?? TECHNIQUE_COLORS.pitch;

            // Build a synthetic Song for navigation
            const syntheticSong: Song = {
              id: song.id,
              title: song.title,
              artist: `GTSinger · ${singerLabel(song.singer)} · ${techniqueLabel(song.technique)}`,
              genre: "Technique Exercise",
              difficulty: "MEDIUM" as const,
              duration: song.phrases[0]?.duration_s
                ? `${song.phrases[0].duration_s.toFixed(1)}s`
                : "—",
              bpm: 90,
              imageUrl: "",
              lyrics: [],
              referencePitchSeq: [],
              referenceAudioUrl: song.default_audio_url,
              referenceStyle: `${techniqueLabel(song.technique)} · ${singerLabel(song.singer)}`,
              referenceType: "human_vocal" as const,
            };

            return (
              <motion.div
                key={song.id}
                layout
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96 }}
                className="glass-card rounded-2xl border border-white/8 hover:border-primary/25 transition-all overflow-hidden"
              >
                <div className="p-4">
                  {/* top row */}
                  <div className="flex items-start gap-3 mb-3">
                    <button
                      onClick={() => togglePlay(song.default_audio_url)}
                      className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
                        isPlaying
                          ? "bg-primary/30 border border-primary/50"
                          : "bg-white/8 hover:bg-primary/15 border border-white/10"
                      }`}
                    >
                      {isPlaying ? (
                        <Pause className="w-4 h-4 text-primary" />
                      ) : (
                        <Play className="w-4 h-4 text-on-surface-variant" />
                      )}
                    </button>
                    <div className="flex-1 min-w-0">
                      <p className="font-semibold text-sm text-on-surface truncate capitalize">{song.title}</p>
                      <p className="text-[10px] text-on-surface-variant/60 mt-0.5">
                        {singerLabel(song.singer)}
                      </p>
                    </div>
                  </div>

                  {/* badges */}
                  <div className="flex items-center gap-2 mb-3 flex-wrap">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold border ${colorClass}`}>
                      {techniqueLabel(song.technique)}
                    </span>
                    <span className="text-[10px] text-on-surface-variant/50">
                      {song.phrase_count} phrase{song.phrase_count !== 1 ? "s" : ""}
                    </span>
                  </div>

                  {/* phrases list */}
                  {song.phrases.length > 1 && (
                    <div className="space-y-1 mb-3 max-h-24 overflow-y-auto pr-1">
                      {song.phrases.slice(0, 4).map((phrase) => {
                        const pUrl = phrase.audio_url;
                        const pPlaying = playingUrl === pUrl;
                        return (
                          <div
                            key={phrase.id}
                            className="flex items-center gap-2 py-1 px-2 rounded-lg bg-white/4 hover:bg-white/8 transition-all"
                          >
                            <button
                              onClick={() => togglePlay(pUrl)}
                              className="flex-shrink-0 w-5 h-5 flex items-center justify-center"
                            >
                              {pPlaying ? (
                                <Pause className="w-3 h-3 text-primary" />
                              ) : (
                                <Play className="w-3 h-3 text-on-surface-variant/60" />
                              )}
                            </button>
                            <span className="text-[10px] text-on-surface-variant/70">
                              Phrase {phrase.index ?? phrase.id}
                            </span>
                            {phrase.duration_s && (
                              <span className="text-[10px] text-on-surface-variant/40 ml-auto">
                                {phrase.duration_s.toFixed(1)}s
                              </span>
                            )}
                          </div>
                        );
                      })}
                      {song.phrases.length > 4 && (
                        <p className="text-[10px] text-on-surface-variant/40 text-center py-0.5">
                          + {song.phrases.length - 4} more phrases
                        </p>
                      )}
                    </div>
                  )}

                  {/* practice button */}
                  <button
                    onClick={() => onSelectSong(syntheticSong)}
                    className="w-full py-2.5 rounded-xl bg-gradient-to-r from-secondary/80 to-primary/80 text-on-primary text-xs font-bold hover:brightness-110 hover:from-secondary hover:to-primary active:scale-95 transition-all flex items-center justify-center gap-2"
                  >
                    <Mic className="w-3 h-3" />
                    Sing Along
                  </button>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ── main view ─────────────────────────────────────────────────────────────────

export default function ExercisesView({
  onStartPracticePreset,
  onSelectSong,
}: ExercisesViewProps) {
  const [activeCategory, setActiveCategory] = useState<Category>("all");
  const catalog = useGtsingerCatalog();

  const filteredPresets = PRACTICE_PRESETS.filter((p) => {
    if (activeCategory === "all") return true;
    if (activeCategory === "gtsinger") return p.source === "dataset_reference" || (p.source as string) === "gtsinger";
    if (activeCategory === "technique") return p.category === "scale" || p.category === "free";
    return p.category === activeCategory;
  });

  const gtsingerPresets = PRACTICE_PRESETS.filter(
    (p) => p.source === "dataset_reference" || (p.source as string) === "gtsinger"
  );
  const otherPresets = filteredPresets.filter(
    (p) => p.source !== "dataset_reference" && (p.source as string) !== "gtsinger"
  );

  const showGtSection =
    activeCategory === "all" || activeCategory === "gtsinger" || activeCategory === "technique";

  return (
    <div className="space-y-10 animate-fade-in max-w-6xl">
      {/* page header */}
      <div>
        <h1 className="font-display font-bold text-3xl text-white mb-1">Exercises & Practice</h1>
        <p className="text-on-surface-variant text-sm leading-relaxed">
          Guided vocal exercises with reference audio from the GTSinger dataset. Listen first, then
          record your attempt.
        </p>
      </div>

      {/* category filter */}
      <div className="flex flex-wrap gap-2">
        {(Object.keys(CATEGORY_LABELS) as Category[]).map((cat) => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
              activeCategory === cat
                ? "bg-gradient-to-r from-secondary/20 to-primary/20 text-primary border border-primary/30"
                : "bg-white/5 text-on-surface-variant hover:bg-white/10 border border-white/8"
            }`}
          >
            {CATEGORY_LABELS[cat]}
          </button>
        ))}
      </div>

      {/* preset cards grid (non-gtsinger) */}
      {(activeCategory !== "gtsinger") && otherPresets.length > 0 && (
        <section className="space-y-4">
          <h2 className="font-semibold text-base text-on-surface/80">
            {activeCategory === "all" ? "Guided Exercises" : CATEGORY_LABELS[activeCategory]}
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {(activeCategory === "gtsinger" ? gtsingerPresets : otherPresets).map((preset) => (
              <div key={preset.id}>
                <PresetCard
                  preset={preset}
                  onStart={() => onStartPracticePreset(preset)}
                  isGtSinger={preset.source === "dataset_reference" || (preset.source as string) === "gtsinger"}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* GTSinger reference tracks section */}
      {showGtSection && (
        <section className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex-1">
              <h2 className="font-semibold text-base text-on-surface/80">
                GTSinger Reference Tracks
              </h2>
              <p className="text-xs text-on-surface-variant/60 mt-0.5">
                Real human vocal recordings demonstrating specific singing techniques. Listen and
                sing along.
              </p>
            </div>
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 border border-white/8 text-[10px] font-medium text-on-surface-variant">
              <Check className="w-3 h-3 text-emerald-400" />
              Audio from GTSinger dataset
            </span>
          </div>

          {/* preset cards for gtsinger */}
          {(activeCategory === "gtsinger" || activeCategory === "all") && gtsingerPresets.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {gtsingerPresets.map((preset) => {
                const catalogUrl = findAudioUrlForSong(catalog, preset.song.title, preset.song.singer);
                const enrichedPreset = catalogUrl
                  ? { ...preset, song: { ...preset.song, referenceAudioUrl: catalogUrl } }
                  : preset;
                return (
                  <div key={preset.id}>
                    <PresetCard
                      preset={enrichedPreset}
                      onStart={() => onStartPracticePreset(preset)}
                      isGtSinger
                    />
                  </div>
                );
              })}
            </div>
          )}

          <GtSingerCatalogSection catalog={catalog} onSelectSong={onSelectSong} />
        </section>
      )}

      {filteredPresets.length === 0 && !showGtSection && (
        <div className="glass-card rounded-2xl border border-white/8 p-12 text-center">
          <Wind className="w-10 h-10 mx-auto text-on-surface-variant/30 mb-3" />
          <p className="text-sm text-on-surface-variant/60">No exercises in this category yet.</p>
        </div>
      )}
    </div>
  );
}
