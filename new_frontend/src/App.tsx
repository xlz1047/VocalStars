import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import TopAppBar from "./components/TopAppBar";
import DashboardView from "./components/DashboardView";
import StudioView from "./components/StudioView";
import ResultsView from "./components/ResultsView";
import ReviewView from "./components/ReviewView";
import AICoachDebugView from "./components/AICoachDebugView";
import TaskPracticeSetup from "./components/TaskPracticeSetup";
import ExercisesView from "./components/ExercisesView";

import { Song, Exercise, PerformanceResult, PracticePreset, RecordedAttempt, TaskConfig } from "./types";
import { SONGS } from "./utils/musicDb";
import { useLearningPath } from "./utils/learningPath";
import { 
  Play, 
  X, 
  Wind, 
  Gauge, 
  Volume2, 
  CheckCircle2, 
  Mic, 
  VolumeX, 
  Compass, 
  Activity, 
  Calendar,
  ChevronRight
} from "lucide-react";

export default function App() {
  const [currentView, setCurrentView] = useState("dashboard");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(true); // Default to collapsed as per mockups
  const [searchQuery, setSearchQuery] = useState("");

  // Core active states
  const [activeSong, setActiveSong] = useState<Song>(SONGS[0]);
  const [activeTaskConfig, setActiveTaskConfig] = useState<TaskConfig | null>(null);
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [activeWarmup, setActiveWarmup] = useState<Exercise | null>(null);
  const [warmupActive, setWarmupActive] = useState(false);
  const [warmupTimer, setWarmupTimer] = useState(0);
  const [warmupFinished, setWarmupFinished] = useState(false);

  const { markCompleted } = useLearningPath();

  // Results & History trackers
  const [currentResult, setCurrentResult] = useState<PerformanceResult | null>(null);
  const [currentRecording, setCurrentRecording] = useState<RecordedAttempt | null>(null);
  const [previousTakes, setPreviousTakes] = useState<PerformanceResult[]>([]);

  useEffect(() => {
    return () => {
      if (currentRecording?.audioUrl) {
        URL.revokeObjectURL(currentRecording.audioUrl);
      }
    };
  }, [currentRecording?.audioUrl]);

  // Navigation controller
  const handleNavigate = (view: string) => {
    if (view === "studio") {
      // "Live Studio" from sidebar: go straight to studio if a song is already
      // selected, otherwise send to task-setup to pick one first.
      if (activeSong) {
        setActiveTaskConfig(prev => prev ?? { task_type: "free_singing", scoring_mode: "no_reference" });
        setCurrentView("studio");
      } else {
        setCurrentView("task-setup");
      }
    } else {
      setCurrentView(view);
    }
  };

  // Quick-start: skip task-setup entirely and go straight to free singing
  const handleQuickStart = () => {
    setActiveTaskConfig({ task_type: "free_singing", scoring_mode: "no_reference" });
    setCurrentView("studio");
  };

  // Launch vocal warmup overlay
  const handleStartWarmup = (ex: Exercise) => {
    setActiveWarmup(ex);
    setWarmupActive(true);
    setWarmupTimer(ex.id === "breath-support" ? 15 : 10); // length of warmup
    setWarmupFinished(false);

    // Warmup timer count countdown
    const interval = setInterval(() => {
      setWarmupTimer((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setWarmupFinished(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleSelectSongFromCatalog = (song: Song) => {
    setActiveSong(song);
    setActiveTaskConfig(null);
    setCurrentView("task-setup");
  };

  const handleStartPracticePreset = (preset: PracticePreset) => {
    setActiveSong(preset.song);
    setActiveTaskConfig(preset.taskConfig);
    setActivePresetId(preset.id);
    setCurrentView("studio");
  };

  const handleStartTaskPractice = (taskConfig: TaskConfig) => {
    setActiveTaskConfig(taskConfig);
    setCurrentView("studio");
  };

  const handlePracticeRecommendedTask = (taskConfig: TaskConfig, presetId?: string) => {
    setCurrentResult(null);
    setCurrentRecording((previous) => {
      if (previous?.audioUrl) {
        URL.revokeObjectURL(previous.audioUrl);
      }
      return null;
    });
    setActiveTaskConfig(taskConfig);
    setActivePresetId(presetId ?? null);
    setCurrentView("studio");
  };

  const handleSessionComplete = (result: PerformanceResult, attempt?: Omit<RecordedAttempt, "audioUrl">) => {
    const inputType = result.uiReadyAnalysis?.analysis_validity?.input_type;
    const invalidInputTypes = new Set(["no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"]);
    const sessionState = result.analysisUnavailable
      ? "error"
      : inputType && invalidInputTypes.has(inputType)
      ? "invalid_input"
      : "review";
    const resultWithTask: PerformanceResult = {
      ...result,
      taskConfig: result.taskConfig || activeTaskConfig || undefined,
      sessionState,
    };
    if (attempt) {
      setCurrentRecording((previous) => {
        if (previous?.audioUrl) {
          URL.revokeObjectURL(previous.audioUrl);
        }
        return {
          ...attempt,
          audioUrl: URL.createObjectURL(attempt.audioBlob),
        };
      });
    }
    setCurrentResult(resultWithTask);
    setPreviousTakes((prev) => [resultWithTask, ...prev]);
    // Mark the completed preset in the learning path (only for valid sessions).
    if (sessionState === "review" && activePresetId) {
      markCompleted(activePresetId);
    }
    setCurrentView("results");
  };

  const handleRetrySameTask = () => {
    setCurrentResult(null);
    setCurrentRecording((previous) => {
      if (previous?.audioUrl) {
        URL.revokeObjectURL(previous.audioUrl);
      }
      return null;
    });
    setCurrentView("studio");
  };

  const handleBackToTaskSetup = () => {
    setCurrentView("task-setup");
  };

  return (
    <div className="bg-background min-h-screen text-on-surface select-none pb-12">
      {/* Top Application Bar holds global Search input filtering */}
      <TopAppBar 
        isSidebarCollapsed={isSidebarCollapsed}
        onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        currentView={currentView}
      />

      {/* Main Container handles sidebar dynamic left-margin sizing */}
      <div className="flex pt-16">
        <Sidebar 
          currentView={currentView}
          onNavigate={handleNavigate}
          isCollapsed={isSidebarCollapsed}
          setIsCollapsed={setIsSidebarCollapsed}
        />

        {/* Dynamic Inner Main Panel */}
        <main 
          className={`flex-1 min-h-[calc(100vh-4rem)] p-6 md:p-10 transition-all duration-300 ${
            isSidebarCollapsed ? "ml-20" : "ml-64"
          }`}
        >
          {/* Dashboard Panel View */}
          {currentView === "dashboard" && (
            <DashboardView
              onSelectSong={handleSelectSongFromCatalog}
              onStartPracticePreset={handleStartPracticePreset}
              onActiveWarmup={handleStartWarmup}
              onQuickStart={handleQuickStart}
              searchQuery={searchQuery}
            />
          )}

          {/* Guided task setup before recording */}
          {currentView === "task-setup" && (
            <TaskPracticeSetup
              song={activeSong}
              initialTaskConfig={activeTaskConfig}
              onStartPractice={handleStartTaskPractice}
              onBack={() => setCurrentView("dashboard")}
            />
          )}

          {/* Practice/Recording Studio Studio Session HUD View */}
          {currentView === "studio" && (
            <StudioView 
              song={activeSong}
              taskConfig={activeTaskConfig}
              onSessionComplete={handleSessionComplete}
              onExit={() => setCurrentView("task-setup")}
            />
          )}

          {/* Dynamic Evaluation Results Dashboard Take View */}
          {currentView === "results" && currentResult && (
            <ResultsView 
              result={currentResult}
              recordingUrl={currentRecording?.audioUrl}
              recordingLabel={currentRecording?.sourceLabel}
              onOpenReview={() => setCurrentView("review")}
              onTryAgainSameTask={handleRetrySameTask}
              onBackToTaskSetup={handleBackToTaskSetup}
              onPracticeTask={handlePracticeRecommendedTask}
              onBackToDashboard={() => setCurrentView("dashboard")}
            />
          )}

          {/* Interactive Vintage Playback Review Scrubber View */}
          {currentView === "review" && currentResult && (
            <ReviewView 
              song={activeSong}
              result={currentResult}
              recordingUrl={currentRecording?.audioUrl}
              recordingLabel={currentRecording?.sourceLabel}
              onTryAgainSameTask={handleRetrySameTask}
              onBackToTaskSetup={handleBackToTaskSetup}
              onPracticeTask={handlePracticeRecommendedTask}
              onClose={() => setCurrentView("results")}
            />
          )}

          {/* Vocal Takes History Panel View */}
          {currentView === "history" && (
            <div className="space-y-8 animate-fade-in max-w-4xl">
              <div>
                <h2 className="font-display font-bold text-2xl text-white">Vocal Session History</h2>
                <p className="text-on-surface-variant text-sm mt-1">
                  Track and inspect your previous karaoke performance takes and AI bio-feedback ratings.
                </p>
              </div>

              {previousTakes.length === 0 ? (
                <div className="glass-card p-12 text-center rounded-2xl border border-white/5 space-y-4">
                  <Calendar className="w-12 h-12 mx-auto text-on-surface-variant/40" />
                  <p className="text-on-surface-variant leading-relaxed text-sm">
                    No session records saved yet. Launch your microphone in the Recording Studio to record your first vocal take!
                  </p>
                  <button 
                    onClick={() => handleSelectSongFromCatalog(SONGS[0])}
                    className="px-6 py-2.5 bg-gradient-to-r from-secondary to-primary text-on-primary font-bold text-xs rounded-lg hover:brightness-110 shadow-lg glow-pink"
                  >
                    Start Session
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {previousTakes.map((take, idx) => (
                    <div 
                      key={idx}
                      onClick={() => {
                        const targetSong = SONGS.find(s => s.title === take.songTitle) || SONGS[0];
                        setActiveSong(targetSong);
                        setCurrentResult(take);
                        setCurrentView("results");
                      }}
                      className="glass-card p-6 rounded-xl hover:translate-x-1 duration-200 cursor-pointer flex items-center justify-between border border-white/5"
                    >
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                          <Mic className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-bold text-sm text-white">{take.songTitle}</h4>
                          <p className="text-xs text-on-surface-variant">{take.artist}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6">
                        <div className="text-right">
                          <p className="text-xs font-bold text-on-surface-variant uppercase">OVERALL SCORE</p>
                          <p className="font-display font-extrabold text-lg text-primary">{take.overallScore}%</p>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center text-on-surface-variant hover:text-white">
                          <ChevronRight className="w-4 h-4" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Exercises & Practice Page */}
          {currentView === "exercises" && (
            <ExercisesView
              onStartPracticePreset={handleStartPracticePreset}
              onSelectSong={handleSelectSongFromCatalog}
            />
          )}

          {currentView === "ai-debug" && (
            <AICoachDebugView />
          )}

        </main>
      </div>

      {/* Live Warmup Micro-Interaction Modal/Overlay */}
      {warmupActive && activeWarmup && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-md flex items-center justify-center z-[100] p-4">
          <div className="glass-card rounded-[28px] max-w-md w-full p-8 border border-tertiary/20 relative shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-tertiary/5 rounded-full blur-3xl pointer-events-none" />
            
            <button 
              onClick={() => setWarmupActive(false)}
              className="absolute top-4 right-4 text-on-surface-variant hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="space-y-6 text-center">
              <div className="w-14 h-14 rounded-2xl bg-tertiary/10 text-tertiary flex items-center justify-center mx-auto glow-teal">
                <Wind className="w-6 h-6 text-tertiary animate-pulse" />
              </div>

              <div>
                <span className="text-[10px] font-bold tracking-widest text-[#ddb7ff] bg-[#ddb7ff]/10 px-3 py-1 rounded-full uppercase">
                  ACTIVE VOCAL CO-PILOT
                </span>
                <h3 className="font-display font-bold text-xl text-white mt-3">
                  {activeWarmup.title} Practice
                </h3>
                <p className="text-xs text-on-surface-variant mt-2 leading-relaxed">
                  {activeWarmup.description}
                </p>
              </div>

              {/* Live interactive visualization area depending on timer */}
              <div className="bg-surface-container-high/60 border border-white/5 p-6 rounded-2xl flex flex-col items-center justify-center py-8">
                {warmupFinished ? (
                  <div className="space-y-3">
                    <CheckCircle2 className="w-10 h-10 text-tertiary mx-auto animate-bounce" />
                    <p className="text-sm font-bold text-white">Warmup Sequence Completed!</p>
                    <p className="text-[11px] text-on-surface-variant">Your voice chords are warm and ready for the recording stage.</p>
                  </div>
                ) : (
                  <div className="space-y-4 w-full">
                    <p className="text-3xl font-display font-extrabold text-primary animate-pulse">{warmupTimer}s</p>
                    <p className="text-[10px] font-bold tracking-widest text-on-surface-variant uppercase">
                      {activeWarmup.id === "breath-support" ? "HOLD STEADY EXHALE EXERCISE" : "HUM STEADY RESONATING SCALE"}
                    </p>
                    
                    {/* Pulsing micro timeline bar mapping */}
                    <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-gradient-to-r from-tertiary to-primary h-full transition-all duration-1000"
                        style={{ width: `${(warmupTimer / (activeWarmup.id === "breath-support" ? 15 : 10)) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="flex gap-4">
                <button 
                  onClick={() => setWarmupActive(false)}
                  className="flex-1 py-3 rounded-xl border border-white/10 text-xs font-bold hover:bg-white/5 text-white"
                >
                  Dismiss Overlay
                </button>
                {warmupFinished && (
                  <button 
                    onClick={() => {
                      setWarmupActive(false);
                      handleSelectSongFromCatalog(SONGS[0]);
                    }}
                    className="flex-1 py-3 rounded-xl bg-gradient-to-r from-secondary to-primary text-on-primary text-xs font-bold hover:brightness-110 hover:shadow-lg glow-pink"
                  >
                    Choose Practice
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
