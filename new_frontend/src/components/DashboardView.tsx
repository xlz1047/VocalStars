import { motion } from "motion/react";
import {
  Play,
  Wind,
  Activity,
  TrendingUp,
  Sparkle,
  ChevronRight,
  Music,
  User,
  CheckCircle2,
  Target,
  Waves,
  Mic,
  Lock,
} from "lucide-react";
import { Song, Exercise, PracticePreset } from "../types";
import { SONGS, EXERCISES, PRACTICE_PRESETS } from "../utils/musicDb";
import { LEARNING_STEPS, useLearningPath } from "../utils/learningPath";
import { useHumanReferencePresets } from "../hooks/useHumanReferencePresets";

interface DashboardViewProps {
  onSelectSong: (song: Song) => void;
  onStartPracticePreset?: (preset: PracticePreset) => void;
  onActiveWarmup: (exercise: Exercise) => void;
  onQuickStart?: () => void;
  searchQuery: string;
}

export default function DashboardView({
  onSelectSong,
  onStartPracticePreset,
  onActiveWarmup,
  onQuickStart,
  searchQuery
}: DashboardViewProps) {
  
  const { completed: pathCompleted, nextStep } = useLearningPath();

  // Fetch human reference presets from the catalog API.
  // These replace hardcoded synthetic entries for sustained_note / vibrato / pitch_slide.
  const { presets: humanPresets, loading: humanPresetsLoading } = useHumanReferencePresets(
    ["sustained_note", "vibrato", "pitch_slide"],
    8,
  );

  // Merge: human reference presets first (they carry real F0 vectors), then
  // retain static presets for exercise types NOT covered by the catalog
  // (scale, interval, breath_control, free singing).
  const STATIC_FALLBACK_TYPES = new Set(["scale", "interval", "free", "song"]);
  const staticOnlyPresets = PRACTICE_PRESETS.filter(
    (p) => STATIC_FALLBACK_TYPES.has(p.category) || p.source !== "dataset_reference" && p.source !== "generated_reference",
  );
  const allPresets: PracticePreset[] = humanPresets.length
    ? [...humanPresets, ...staticOnlyPresets]
    : PRACTICE_PRESETS;

  // Filter songs based on search
  const filteredSongs = SONGS.filter(song =>
    song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    song.artist.toLowerCase().includes(searchQuery.toLowerCase()) ||
    song.genre.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const featuredSong = SONGS.find(s => s.featured) || SONGS[0];
  const filteredPracticePresets = allPresets.filter((preset) =>
    preset.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    preset.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    preset.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getExerciseIcon = (type: string) => {
    switch (type) {
      case "breath": return <Wind className="w-5 h-5 text-tertiary" />;
      case "pitch": return <Activity className="w-5 h-5 text-secondary" />;
      case "agility": return <TrendingUp className="w-5 h-5 text-primary" />;
      default: return <Sparkle className="w-5 h-5 text-primary" />;
    }
  };

  const getDifficultyColor = (diff: string) => {
    switch (diff) {
      case "EASY": return "bg-tertiary/10 text-tertiary border border-tertiary/25";
      case "MEDIUM": return "bg-secondary/10 text-secondary border border-secondary/25";
      default: return "bg-error/10 text-error border border-error/25";
    }
  };

  return (
    <div className="space-y-12 animate-fade-in">
      
      {/* Learning path progression strip */}
      <section className="glass-card rounded-2xl p-5 border border-white/5">
        <div className="flex items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="font-display font-bold text-lg text-white">Your Practice Path</h2>
            <p className="text-xs text-on-surface-variant mt-0.5">
              {pathCompleted.size === 0
                ? "Start at step 1 and work through each skill in order."
                : pathCompleted.size >= LEARNING_STEPS.length
                ? "All steps complete — keep exploring more exercises below."
                : `${pathCompleted.size} of ${LEARNING_STEPS.length} steps done.`}
            </p>
          </div>
          {nextStep && (
            <button
              onClick={() => onStartPracticePreset?.(nextStep.preset)}
              className="flex-shrink-0 px-4 py-2 rounded-xl bg-tertiary/15 border border-tertiary/25 text-tertiary text-xs font-bold hover:bg-tertiary/20 transition-all"
            >
              Start Next
            </button>
          )}
        </div>

        <div className="flex gap-2 overflow-x-auto pb-1 no-scrollbar">
          {(() => {
            // Compute once, not inside the map
            const nextStepIndex = nextStep
              ? LEARNING_STEPS.findIndex((s) => s.presetId === nextStep.presetId)
              : LEARNING_STEPS.length; // all done → no locked steps
            return LEARNING_STEPS.map((step, index) => {
            const done = pathCompleted.has(step.presetId);
            const isNext = nextStep?.presetId === step.presetId;
            const locked = !done && !isNext && index > nextStepIndex;
            return (
              <button
                key={step.presetId}
                onClick={() => !locked && onStartPracticePreset?.(step.preset)}
                disabled={locked}
                title={locked ? "Complete the previous step first" : step.preset.description}
                className={`flex-shrink-0 flex flex-col items-center gap-1.5 px-3 py-2.5 rounded-xl border text-center min-w-[88px] transition-all ${
                  done
                    ? "bg-tertiary/10 border-tertiary/25 text-tertiary"
                    : isNext
                    ? "bg-primary/10 border-primary/35 text-primary ring-1 ring-primary/30"
                    : "bg-white/3 border-white/8 text-on-surface-variant/50 cursor-not-allowed"
                }`}
              >
                <span className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-black border border-current/30">
                  {done ? <CheckCircle2 className="w-4 h-4" /> : locked ? <Lock className="w-3 h-3" /> : index + 1}
                </span>
                <span className="text-[10px] font-bold leading-tight">{step.milestone}</span>
              </button>
            );
            });
          })()}
        </div>
      </section>

      {/* 1. Human vocal practice block */}
      <section className="relative rounded-[32px] overflow-hidden shadow-lg border border-white/5 bg-surface-container/60 p-6 md:p-8">
        <img 
          alt="Singer practicing into a microphone" 
          className="absolute inset-0 w-full h-full object-cover opacity-30" 
          src="https://images.unsplash.com/photo-1516280440614-37939bbacd81?w=1600&auto=format&fit=crop&q=80"
          referrerPolicy="no-referrer"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-background via-background/85 to-background/30" />

        <div className="relative z-10 grid lg:grid-cols-12 gap-8 items-center min-h-[500px]">
          <div className="lg:col-span-5 space-y-5">
            <span className="inline-flex px-4 py-1.5 rounded-full bg-tertiary text-on-tertiary font-semibold tracking-widest text-[11px] uppercase shadow-[0_0_20px_rgba(60,221,199,0.18)]">
              Guided human vocal practice
            </span>
            <h1 className="font-display font-extrabold text-white text-4xl md:text-6xl drop-shadow-2xl">
              Train notes your voice actually sings
            </h1>
            <p className="text-on-surface-variant max-w-2xl text-lg leading-relaxed">
              Start with singer-first drills: sustained vowels, note matching, glides, scale fragments, and simple reference melodies.
            </p>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => onStartPracticePreset?.(
                  humanPresets.find(p => p.category === "pitch") ?? allPresets[0]
                )}
                className="inline-flex items-center gap-2.5 px-8 py-3.5 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary text-sm font-bold hover:scale-105 hover:brightness-110 hover:shadow-[0_0_25px_rgba(255,177,192,0.45)] active:scale-95 transition-all duration-300 glow-pink"
              >
                <Play className="w-4 h-4 fill-current" />
                Start Sustained Note
              </button>
              {onQuickStart && (
                <button
                  onClick={onQuickStart}
                  className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-full border border-white/20 bg-white/5 text-white text-sm font-bold hover:bg-white/10 hover:border-white/35 active:scale-95 transition-all duration-200"
                >
                  <Mic className="w-4 h-4" />
                  Quick Record
                </button>
              )}
            </div>
          </div>

          <div className="lg:col-span-7 grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
            {humanPresetsLoading && !humanPresets.length && (
              <div className="col-span-full text-center text-on-surface-variant text-xs py-4 opacity-60">
                Loading human reference exercises…
              </div>
            )}
            {filteredPracticePresets.slice(0, 8).map((preset) => {
              const Icon = preset.category === "slide" ? Waves : preset.category === "song" ? Music : Target;
              return (
                <button
                  key={preset.id}
                  onClick={() => onStartPracticePreset?.(preset)}
                  className="text-left rounded-2xl border border-white/10 bg-background/45 hover:bg-background/65 hover:border-primary/30 transition-all p-5"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-11 h-11 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                      <Icon className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-display font-bold text-lg text-white">{preset.title}</h3>
                        <span className="text-[9px] uppercase tracking-wider px-2 py-0.5 rounded-full bg-white/5 text-on-surface-variant">
                          {preset.duration}
                        </span>
                      </div>
                      <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
                        {preset.description}
                      </p>
                      <p className="text-[10px] text-tertiary uppercase tracking-wider font-bold mt-3">
                        {preset.source.replaceAll("_", " ")}
                      </p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </section>

      {/* 2. Vocal Fitness & Warmups grid */}
      <section className="space-y-6">
        <div className="flex justify-between items-center pr-2">
          <div className="flex items-center gap-3">
            <h3 className="font-display text-2xl font-bold text-on-surface">Vocal Fitness &amp; Warmups</h3>
            <span className="w-2 h-2 rounded-full bg-tertiary glow-teal" />
          </div>
          <span className="text-xs font-semibold text-tertiary uppercase tracking-wider hidden sm:inline-block">
            Recommended Daily Routine
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {EXERCISES.map((ex) => (
            <div 
              key={ex.id}
              onClick={() => onActiveWarmup(ex)}
              className={`glass-card p-6 rounded-2xl border-l-4 cursor-pointer hover:translate-y-[-4px] transition-all duration-300 group flex flex-col justify-between ${
                ex.type === "breath" ? "border-tertiary" : 
                ex.type === "pitch" ? "border-secondary" : 
                ex.type === "agility" ? "border-primary" : "border-primary-container"
              }`}
            >
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div className="p-3 rounded-xl bg-white/5 group-hover:bg-white/10 transition-colors duration-300">
                    {getExerciseIcon(ex.type)}
                  </div>
                  <span className="text-[10px] font-bold text-on-surface-variant/50 uppercase tracking-widest mt-1">
                    {ex.duration}
                  </span>
                </div>
                <h4 className="font-bold text-base text-on-surface group-hover:text-primary transition-colors mb-2">
                  {ex.title}
                </h4>
                <p className="text-xs text-on-surface-variant/80 leading-relaxed mb-4">
                  {ex.description}
                </p>
              </div>
              <div>
                <div className="flex justify-between items-center mb-1 text-[10px] font-bold tracking-wider text-on-surface-variant/60 uppercase">
                  <span>Current Skill</span>
                  <span>{ex.progress}%</span>
                </div>
                <div className="w-full bg-surface-container-high h-1.5 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-700 ${
                      ex.type === "breath" ? "bg-tertiary" : 
                      ex.type === "pitch" ? "bg-secondary" : "bg-primary"
                    }`}
                    style={{ width: `${ex.progress}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Trending & New Song Lists */}
      <div className="space-y-12">
        
        {/* Trending Hits Catalog */}
        <section className="space-y-6">
          <div className="flex justify-between items-center pr-2">
            <div className="flex items-center gap-3">
              <h3 className="font-display text-2xl font-extrabold text-white tracking-tight">Trending Hits to Practice</h3>
              <span className="w-2.5 h-2.5 rounded-full bg-primary glow-pink" />
            </div>
            <span className="text-xs font-semibold text-primary uppercase tracking-wider cursor-pointer hover:underline">
              View All
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {filteredSongs.slice(0, 3).map((song) => (
              <div 
                key={song.id}
                onClick={() => onSelectSong(song)}
                className="bg-[#131520] hover:bg-[#191b2b] p-5 rounded-[28px] border border-white/5 transition-all duration-300 hover:shadow-2xl hover:shadow-primary/5 hover:-translate-y-1 group flex flex-col justify-between cursor-pointer"
              >
                <div>
                  <div className="relative w-full aspect-[4/3] rounded-[20px] overflow-hidden mb-5 border border-white/5 shadow-inner">
                    <img 
                      src={song.imageUrl} 
                      alt={song.title} 
                      className="w-full h-full object-cover group-hover:scale-105 duration-700 ease-out" 
                    />
                    <span className={`absolute top-4 left-4 px-3 py-1 text-[10px] font-extrabold tracking-widest rounded-lg uppercase ${
                      song.difficulty === "EASY" ? "bg-[rgba(60,221,199,0.95)] text-[#0e2d27] shadow-[0_4px_12px_rgba(60,221,199,0.2)]" :
                      song.difficulty === "MEDIUM" ? "bg-[rgba(221,183,255,0.95)] text-[#2a104c] shadow-[0_4px_12px_rgba(221,183,255,0.2)]" :
                      "bg-[rgba(255,177,192,0.95)] text-[#4c1020] shadow-[0_4px_12px_rgba(255,177,192,0.2)]"
                    }`}>
                      {song.difficulty}
                    </span>
                    <div className="absolute bottom-4 right-4 bg-black/65 backdrop-blur-sm shadow-md py-1 px-3 rounded-lg text-[10px] font-mono tracking-wider font-bold text-white/90 uppercase">
                      {song.duration}
                    </div>
                  </div>

                  <h4 className="font-display font-bold text-lg text-white group-hover:text-primary transition-colors pr-1">
                    {song.title}
                  </h4>
                  <p className="text-xs text-on-surface-variant flex items-center gap-1.5 mt-1 opacity-80">
                    <User className="w-3.5 h-3.5 text-on-surface-variant/40" />
                    <span>{song.artist}</span>
                  </p>
                </div>

                <div className="w-full py-3 mt-5 rounded-2xl text-center flex items-center justify-center gap-1.5 text-xs font-bold transition-all bg-white/5 border border-white/10 text-primary group-hover:bg-primary group-hover:text-on-primary group-hover:border-primary shadow-md">
                  <span>Practice</span>
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* New Releases Catalog */}
        <section className="space-y-6">
          <div className="flex justify-between items-center pr-2">
            <div className="flex items-center gap-3">
              <h3 className="font-display text-2xl font-extrabold text-white tracking-tight">New for You</h3>
              <span className="w-2.5 h-2.5 rounded-full bg-secondary glow-purple" />
            </div>
            <span className="text-xs font-semibold text-secondary uppercase tracking-wider cursor-pointer hover:underline">
              View All
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {filteredSongs.slice(3, 6).map((song) => (
              <div 
                key={song.id}
                onClick={() => onSelectSong(song)}
                className="bg-[#131520] hover:bg-[#191b2b] p-5 rounded-[28px] border border-white/5 transition-all duration-300 hover:shadow-2xl hover:shadow-secondary/5 hover:-translate-y-1 group flex flex-col justify-between cursor-pointer"
              >
                <div>
                  <div className="relative w-full aspect-[4/3] rounded-[20px] overflow-hidden mb-5 border border-white/5 shadow-inner">
                    <img 
                      src={song.imageUrl} 
                      alt={song.title} 
                      className="w-full h-full object-cover group-hover:scale-105 duration-700 ease-out" 
                    />
                    <span className={`absolute top-4 left-4 px-3 py-1 text-[10px] font-extrabold tracking-widest rounded-lg uppercase ${
                      song.difficulty === "EASY" ? "bg-[rgba(60,221,199,0.95)] text-[#0e2d27] shadow-[0_4px_12px_rgba(60,221,199,0.2)]" :
                      song.difficulty === "MEDIUM" ? "bg-[rgba(221,183,255,0.95)] text-[#2a104c] shadow-[0_4px_12px_rgba(221,183,255,0.2)]" :
                      "bg-[rgba(255,177,192,0.95)] text-[#4c1020] shadow-[0_4px_12px_rgba(255,177,192,0.2)]"
                    }`}>
                      {song.difficulty}
                    </span>
                    <div className="absolute bottom-4 right-4 bg-black/65 backdrop-blur-sm shadow-md py-1 px-3 rounded-lg text-[10px] font-mono tracking-wider font-bold text-white/90 uppercase">
                      {song.duration}
                    </div>
                  </div>

                  <h4 className="font-display font-bold text-lg text-white group-hover:text-secondary transition-colors pr-1">
                    {song.title}
                  </h4>
                  <p className="text-xs text-on-surface-variant flex items-center gap-1.5 mt-1 opacity-80">
                    <User className="w-3.5 h-3.5 text-on-surface-variant/40" />
                    <span>{song.artist}</span>
                  </p>
                </div>

                <div className="w-full py-3 mt-5 rounded-2xl text-center flex items-center justify-center gap-1.5 text-xs font-bold transition-all bg-white/5 border border-white/10 text-secondary group-hover:bg-secondary group-hover:text-on-secondary group-hover:border-secondary shadow-md">
                  <span>Practice</span>
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </section>

      </div>

    </div>
  );
}
