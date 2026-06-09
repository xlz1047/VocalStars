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
  CheckCircle2
} from "lucide-react";
import { Song, Exercise } from "../types";
import { SONGS, EXERCISES } from "../utils/musicDb";

interface DashboardViewProps {
  onSelectSong: (song: Song) => void;
  onActiveWarmup: (exercise: Exercise) => void;
  searchQuery: string;
}

export default function DashboardView({ 
  onSelectSong, 
  onActiveWarmup,
  searchQuery 
}: DashboardViewProps) {
  
  // Filter songs based on search
  const filteredSongs = SONGS.filter(song => 
    song.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    song.artist.toLowerCase().includes(searchQuery.toLowerCase()) ||
    song.genre.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const featuredSong = SONGS.find(s => s.featured) || SONGS[0];

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
      
      {/* 1. Featured Song block */}
      <section className="relative rounded-[32px] overflow-hidden group h-[520px] cursor-pointer shadow-lg border border-white/5">
        <img 
          alt="Featured Background" 
          className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-105" 
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuBjF0HjXlfhqx_tY3aaxi46xoYrmN5jYLTj5vjQej8B4-bLiLn82wQLj9cvgHni7pEpWz9TOvqgGjXc4bGb9XVOgUtzlfiDX0nVW6sSqSpKFd67cu6qtQdZ5tzbpNdyN8ngdgPdQlb26YVXqPZJgD0IUS-PETbPGg2A_8jvaMrnfXFBBl1qluy4pg_gya-2_HoBVNCugnEKAxD7qGk0_AasjS-6vHqVr9lG_Eq9RVi9dGO9nz9lYJqurlE3RIzajQ22b39_0hTrYcV_"
          referrerPolicy="no-referrer"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-background/60 to-background"></div>
        
        {/* Ambient neon flares inside image container */}
        <div className="absolute top-24 left-1/4 w-72 h-72 rounded-full bg-primary/20 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-12 right-1/4 w-72 h-72 rounded-full bg-secondary/20 blur-[100px] pointer-events-none" />

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 md:p-12 space-y-6 z-10">
          <span className="px-5 py-1.5 rounded-full bg-tertiary text-on-tertiary font-semibold tracking-widest text-[11px] uppercase shadow-[0_0_20px_rgba(60,221,199,0.3)] animate-pulse">
            SONG OF THE DAY
          </span>
          <h1 className="font-display font-extrabold text-white text-4xl md:text-6xl drop-shadow-2xl transition-all duration-500 group-hover:scale-[1.02]">
            {featuredSong.title}
          </h1>
          <p class="font-body-lg text-on-surface-variant max-w-2xl text-xl leading-relaxed">Unleash your soulful depth with this synth-heavy masterpiece. Perfect for mastering mid-range control and expressive vibrato.</p>
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <button 
              onClick={() => onSelectSong(featuredSong)}
              className="flex items-center gap-2.5 px-8 py-3.5 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary text-sm font-bold hover:scale-105 hover:brightness-110 hover:shadow-[0_0_25px_rgba(255,177,192,0.6)] active:scale-95 transition-all duration-300 glow-pink"
            >
              <Play className="w-4 h-4 fill-current" />
              Start Practice
            </button>
            <button 
              onClick={() => onSelectSong(featuredSong)}
              className="px-8 py-3.5 rounded-full bg-white/5 backdrop-blur-md border border-white/10 text-sm font-bold hover:bg-white/15 hover:border-white/20 hover:scale-105 active:scale-95 transition-all duration-300"
            >
              Preview Track
            </button>
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
