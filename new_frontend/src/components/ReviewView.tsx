import { useState, useEffect, useRef } from "react";
import { motion } from "motion/react";
import { 
  Play, 
  Pause, 
  SkipBack, 
  SkipForward, 
  ZoomIn, 
  ZoomOut, 
  CheckCircle2, 
  AlertTriangle, 
  Clock, 
  Sparkles,
  Award,
  Circle,
  TrendingUp,
  X
} from "lucide-react";
import { Song, PerformanceResult, TaskConfig, UiSegment } from "../types";
import { buildImprovementPath } from "../utils/improvementPath";
import MelSpectrogramView from "./MelSpectrogramView";
import PitchLane from "./PitchLane";
import PosteriorConfidenceMap from "./PosteriorConfidenceMap";
import RecordingPlaybackControls from "./RecordingPlaybackControls";
import SegmentMarkerTrack from "./SegmentMarkerTrack";
import SpectralToneProxyMap from "./SpectralToneProxyMap";
import WaveformTimeline from "./WaveformTimeline";

interface ReviewViewProps {
  song: Song;
  result: PerformanceResult;
  recordingUrl?: string | null;
  recordingLabel?: string | null;
  onTryAgainSameTask?: () => void;
  onBackToTaskSetup?: () => void;
  onPracticeTask?: (taskConfig: TaskConfig, presetId?: string) => void;
  onClose: () => void;
}

export default function ReviewView({ 
  song, 
  result, 
  recordingUrl,
  recordingLabel,
  onTryAgainSameTask,
  onBackToTaskSetup,
  onPracticeTask,
  onClose 
}: ReviewViewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 0.5 | 1.5>(1);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeTab, setActiveTab] = useState<"Pitch" | "Confidence">("Pitch");
  const [selectedRegion, setSelectedRegion] = useState<UiSegment | null>(null);

  // Track coordinates and state
  const timelineRef = useRef<HTMLDivElement>(null);
  const totalDuration = Math.max(result.uiReadyAnalysis?.audio?.duration_s || 0, 0.1);
  const recommendedFocus = result.uiReadyAnalysis ? buildImprovementPath(result.uiReadyAnalysis).primaryFocus : null;

  useEffect(() => {
    let playTimer: NodeJS.Timeout;
    if (isPlaying) {
      playTimer = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= totalDuration) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 1000 / playbackSpeed);
    }
    return () => clearInterval(playTimer);
  }, [isPlaying, playbackSpeed]);

  const formatTimestamp = (sec: number) => {
    const whole = Math.max(0, Math.floor(sec));
    const minutes = Math.floor(whole / 60);
    const secs = whole % 60;
    return `${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const lyricIndex = Math.min(
    Math.floor((currentTime / totalDuration) * song.lyrics.length),
    song.lyrics.length - 1
  );
  const activeLyric = song.lyrics[lyricIndex] || "In the shadows of the neon light...";

  const cardsToRender = result.coachingNotes && result.coachingNotes.length > 0 
    ? result.coachingNotes.map((note) => {
        let typeColor = "text-[#3cddc7]";
        let borderColor = "border-[#3cddc7]/20";
        let iconType = "check";
        
        if (note.type === "warning") {
          typeColor = "text-primary";
          borderColor = "border-primary/25";
          iconType = "warning";
        } else if (note.type === "info") {
          typeColor = "text-[#ddb7ff]";
          borderColor = "border-[#ddb7ff]/25";
          iconType = "sparkle";
        }
        
        return {
          category: note.category || "Vocal Score",
          text: note.text,
          icon: iconType,
          color: typeColor,
          borderColor,
          bgColor: "bg-[#131520]/80"
        };
      })
    : [
        {
          category: "REVIEW READY",
          text: "Replay your recorded take and inspect the frame-level analysis when available.",
          icon: "sparkle",
          color: "text-[#ddb7ff]",
          borderColor: "border-[#ddb7ff]/25 shadow-[0_4px_12px_rgba(221,183,255,0.05)]",
          bgColor: "bg-[#131520]/85"
        }
      ];

  return (
    <div className="flex flex-col h-full bg-[#080911] min-h-[90vh] rounded-[32px] overflow-hidden animate-fade-in relative border border-white/5 shadow-2xl">
      
      {/* Playback HUD Navigation */}
      <header className="flex justify-between items-center px-8 md:px-12 py-7 border-b border-white/5 bg-[#0f111a]/60 backdrop-blur-md">
        <div className="flex items-center gap-6">
          <div>
            <h1 className="font-display font-black text-2xl text-white tracking-tight uppercase">
              {song.title}
            </h1>
            <p className="text-[10px] font-bold uppercase tracking-widest text-[#ddb7ff] mt-0.5">
              Performance Review • Take 4
            </p>
          </div>
          <div className="flex items-center gap-1.5 bg-[#00302a]/50 text-[#3cddc7] border border-[#3cddc7]/30 px-3.5 py-1.5 rounded-full text-xs font-black shadow-[0_4px_12px_rgba(60,221,199,0.1)]">
            <Sparkles className="w-3.5 h-3.5 animate-pulse" />
            <span>{result.intonation}% Accuracy</span>
          </div>
        </div>

        <button 
          onClick={onClose}
          className="p-2.5 text-on-surface-variant hover:text-white hover:bg-white/5 rounded-full transition-all duration-300"
          aria-label="Close review"
        >
          <X className="w-5 h-5" />
        </button>
      </header>

      {/* Main Review Visualization Content canvas */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        
        {/* Left main: Retroglow mic HUD panel */}
        <section className="flex-1 p-6 md:p-8 flex flex-col justify-between items-center relative gap-6">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,177,192,0.08)_0%,transparent_70%)] pointer-events-none" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_30%,rgba(221,183,255,0.05)_0%,transparent_60%)] pointer-events-none" />

          {/* Video / Mic Showcase Card */}
          <div className="w-full flex-1 relative bg-[#131520]/45 border border-white/5 rounded-[28px] overflow-hidden flex items-center justify-center py-6 min-h-[380px] hover:border-white/10 transition-colors duration-300">
            
            {/* Top Left PITCH CONFIDENCE styling to match image.png */}
            <div className="absolute top-6 left-6 bg-[#131520]/80 backdrop-blur-md border border-white/10 p-3.5 rounded-2xl z-20 flex flex-col gap-1.5 shadow-xl select-none">
              <span className="text-[9px] font-extrabold text-[#ffb1c0] tracking-widest uppercase">
                PITCH CONFIDENCE
              </span>
              <div className="h-1.5 w-24 bg-white/10 rounded-full overflow-hidden relative">
                <div className="absolute top-0 left-0 bottom-0 w-[94%] bg-gradient-to-r from-[#3cddc7] via-[#ddb7ff] to-primary rounded-full" />
              </div>
            </div>

            {/* Top Right REC SYNC to match image.png */}
            <div className="absolute top-6 right-6 bg-[#4c1020]/25 border border-[#ffb1c0]/25 px-3.5 py-1.5 rounded-full text-[10px] text-primary font-black tracking-widest uppercase flex items-center gap-1.5 z-20 select-none shadow-lg">
              <span className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_8px_rgba(255,177,192,0.6)]" />
              <span>REC SYNC</span>
            </div>

            {/* Vintage style hardware condenser microphone image centered */}
            <div className="flex flex-col items-center justify-center space-y-6 relative max-w-sm z-10 text-center w-full px-6">
              <div className="relative flex items-center justify-center">
                {/* Micro Back-glow bubble ring aura */}
                <span className="w-56 h-56 rounded-full bg-primary/20 absolute blur-[70px] animate-pulse pointer-events-none" />
                <span className="w-44 h-44 rounded-full bg-[#ddb7ff]/20 absolute blur-[40px] pointer-events-none animate-bounce duration-[8000ms]" />
                
                <img 
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuDxpI7FgU9FNUy001yzMNQfZdraRCuA1_acXoAJ8NUrnXYmioVldCg8BK25azx2I8Abj5_esb9GMn-tNTR_5JfBMrrtVcZzxlf-RonFh4r-OZwiQIYD5Tf5L8trBPPZ6AXqcAjeMpJhgnLeudnSjwaU7auCcnQKpJT4SAvDFIqwJkBX5pMImSJwOIcfL8VhRsVMgPJlHYTr-FTh1dUL8SdNjoVFyxmVI0nOb8oAZgj-gwuggiajNWWJ0XYwpe1-zTE7OazAbfOFzIpo" 
                  alt="Vintage condenser mic showcase glow" 
                  className="w-auto h-[280px] object-contain relative z-10 select-none pointer-events-none drop-shadow-[0_20px_40px_rgba(0,0,0,0.8)] hover:scale-102 transition-transform duration-700"
                />
              </div>

              {/* Centered lyric bar */}
              <div className="w-full bg-[#131520]/85 backdrop-blur-md border border-white/10 py-4 px-6 rounded-[22px] shadow-2xl relative z-20 max-w-sm">
                <span className="text-[9px] font-extrabold text-primary tracking-widest uppercase block mb-1">
                  CURRENT LYRIC
                </span>
                <p className="font-display font-black text-base md:text-lg tracking-wide text-white leading-relaxed">
                  "{activeLyric}"
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Right main sidebar feedback panel */}
        <aside className="w-full lg:w-[410px] bg-[#0c0e17]/40 border-t lg:border-t-0 lg:border-l border-white/5 flex flex-col p-8 space-y-6">
          <div className="flex justify-between items-center pb-2">
            <h3 className="font-display text-lg font-black text-white uppercase tracking-tight">
              Coaching Feedback
            </h3>
            <Sparkles className="w-5 h-5 text-[#ddb7ff] animate-pulse" />
          </div>

          <div className="flex-1 overflow-y-auto space-y-4 pr-1 no-scrollbar">
            {cardsToRender.map((card, index) => (
              <div 
                key={index}
                className={`p-5 rounded-[22px] border ${card.bgColor} ${card.borderColor} transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg flex flex-col gap-2.5`}
              >
                <div className="flex items-center gap-2.5">
                  {card.icon === "check" && <CheckCircle2 className="w-4 h-4 text-[#3cddc7]" />}
                  {card.icon === "warning" && <AlertTriangle className="w-4 h-4 text-primary" />}
                  {card.icon === "sparkle" && <Sparkles className="w-4 h-4 text-[#ddb7ff]" />}
                  
                  <span className={`text-[10px] font-black tracking-widest uppercase ${card.color}`}>
                    {card.category}
                  </span>
                </div>
                <p className="text-[12px] text-white/70 leading-relaxed font-sans font-medium">
                  {card.text}
                </p>
              </div>
            ))}
          </div>

          <button className="w-full py-4 rounded-[22px] bg-transparent border border-white/15 hover:border-primary/40 hover:bg-white/5 transition-all duration-300 text-xs font-black text-white uppercase tracking-widest shadow-md">
            View Full Analysis
          </button>
        </aside>

      </div>

      {result.uiReadyAnalysis && (
        <section className="bg-[#080911] border-t border-white/10 p-6 md:p-8 space-y-6">
          <div>
            <p className="text-[10px] font-black text-[#ddb7ff] tracking-widest uppercase">Frame-by-frame review</p>
            <h2 className="font-display font-black text-2xl text-white mt-1">Analysis Timeline</h2>
          </div>
          <div className="grid xl:grid-cols-2 gap-6">
            <div className="xl:col-span-2">
              <MelSpectrogramView analysis={result.uiReadyAnalysis} compact />
            </div>
            <PitchLane analysis={result.uiReadyAnalysis} compact />
            <WaveformTimeline analysis={result.uiReadyAnalysis} compact />
            <PosteriorConfidenceMap analysis={result.uiReadyAnalysis} compact />
            <SpectralToneProxyMap analysis={result.uiReadyAnalysis} compact />
            <div className="xl:col-span-2">
              <SegmentMarkerTrack
                analysis={result.uiReadyAnalysis}
                compact
                selectedSegmentId={selectedRegion?.id || null}
                onSelectSegment={(segment) => {
                  setSelectedRegion(segment);
                  setCurrentTime(Number(segment.start_s || 0));
                }}
              />
            </div>
          </div>
        </section>
      )}

      {/* BOTTOM Detail Scrub Timeline Controls panel */}
      <footer className="bg-[#0c0e17]/90 border-t border-white/10 p-6 md:p-8 flex flex-col gap-5">
        <RecordingPlaybackControls
          audioUrl={recordingUrl}
          label={recordingLabel || "Actual Recorded Take"}
          selectedRegion={selectedRegion}
          onTryAgainSameTask={onTryAgainSameTask}
          onBackToTaskSetup={onBackToTaskSetup}
          recommendedFocus={recommendedFocus}
          onPracticeTask={onPracticeTask}
          compact
        />
        
        {/* Playback Controls & zoom row */}
        <div className="flex justify-between items-center flex-wrap gap-4">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3.5">
              <button 
                onClick={() => setCurrentTime(Math.max(0, currentTime - 5))}
                className="text-on-surface-variant hover:text-white transition-colors p-2 rounded-lg hover:bg-white/5"
                aria-label="Skip back 5 seconds"
              >
                <SkipBack className="w-5 h-5" />
              </button>
              
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-12 h-12 rounded-full bg-primary hover:scale-105 transition-all text-on-primary flex items-center justify-center shadow-lg hover:shadow-primary/20 glow-pink"
                aria-label={isPlaying ? "Pause track" : "Play track"}
              >
                {isPlaying ? <Pause className="w-5.5 h-5.5 fill-current" /> : <Play className="w-5.5 h-5.5 fill-current ml-0.5" />}
              </button>
              
              <button 
                onClick={() => setCurrentTime(Math.min(totalDuration, currentTime + 5))}
                className="text-on-surface-variant hover:text-white transition-colors p-2 rounded-lg hover:bg-white/5"
                aria-label="Skip forward 5 seconds"
              >
                <SkipForward className="w-5 h-5" />
              </button>
            </div>

            <div className="text-sm font-semibold text-on-surface-variant tabular-nums flex items-center gap-2 bg-white/5 py-1.5 px-3.5 rounded-xl border border-white/5">
              <Clock className="w-4 h-4 text-primary" />
              <span className="text-white font-extrabold">{formatTimestamp(currentTime)}</span>
              <span className="text-on-surface-variant/40">/</span>
              <span>{formatTimestamp(totalDuration)}</span>
            </div>

            {/* Micro Multi speed selector */}
            <div className="flex bg-white/5 rounded-full p-1 border border-white/5">
              {[0.5, 1, 1.5].map((speed) => (
                <button
                  key={speed}
                  onClick={() => setPlaybackSpeed(speed as any)}
                  className={`px-3 py-1 rounded-full text-xs font-black uppercase transition-all ${
                    playbackSpeed === speed ? "bg-primary text-on-primary shadow-md" : "text-on-surface-variant hover:text-white"
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3 bg-white/5 px-4.5 py-2.5 rounded-2xl border border-white/5">
            <ZoomOut className="w-4 h-4 text-on-surface-variant hover:text-white cursor-pointer" onClick={() => setZoomLevel(Math.max(1, zoomLevel - 0.5))} />
            <span className="text-xs font-black text-on-surface uppercase tracking-wider px-1">Timeline Zoom</span>
            <ZoomIn className="w-4 h-4 text-on-surface-variant hover:text-white cursor-pointer" onClick={() => setZoomLevel(Math.min(3, zoomLevel + 0.5))} />
          </div>
        </div>

        {/* Pitch Accuracy Overlap Header Banner */}
        <div className="flex justify-between items-center px-1">
          <span className="text-[10px] font-black text-[#ddb7ff] tracking-widest uppercase">
            PITCH ACCURACY OVERLAP
          </span>
          <span className="bg-[#3cddc7]/10 border border-[#3cddc7]/30 text-[#3cddc7] px-4 py-1 rounded-full text-xs font-black shadow-md shadow-[#3cddc7]/5">
            {result.intonation}% Perfect Match
          </span>
        </div>

        {/* Scrolling Detailed Timeline visualizer bar */}
        <div className="relative py-4 px-1 overflow-hidden bg-[#131520]/90 rounded-2xl border border-white/5 shadow-2xl">
          <div 
            className="h-16 relative select-none rounded-lg cursor-pointer transition-all duration-300"
            ref={timelineRef}
            onClick={(e) => {
              if (timelineRef.current) {
                const rect = timelineRef.current.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                setCurrentTime(Math.round(pos * totalDuration));
              }
            }}
          >
            {/* Pitch alignment background guide wave SVG */}
            <svg 
              className="absolute inset-x-0 inset-y-0 w-full h-full opacity-90" 
              preserveAspectRatio="none"
              style={{ transform: `scaleX(${zoomLevel})` }}
            >
              {/* Perfect Guide Reference Wave (light white outline) */}
              <path 
                d="M0,32 Q100,6 200,48 T400,32 T600,48 T800,18 T1000,48 T1200,32" 
                fill="none" 
                stroke="rgba(255, 255, 255, 0.15)" 
                strokeWidth="3.5" 
                strokeDasharray="4 4"
              />

              {/* User Actual Sing Wave (moving color gradient path) */}
              <path 
                d="M0,34 Q100,10 195,46 T402,30 T605,45 T795,20 T1003,44 T1200,34" 
                fill="none" 
                stroke="url(#reviewWaveframeGradient)" 
                strokeWidth="4.5" 
                strokeLinecap="round"
              />

              <defs>
                <linearGradient id="reviewWaveframeGradient" x1="0%" x2="100%" y1="0%" y2="0%">
                  <stop offset="0%" stopColor="#3cddc7" />
                  <stop offset="35%" stopColor="#ddb7ff" />
                  <stop offset="70%" stopColor="#ffb1c0" />
                  <stop offset="100%" stopColor="#3cddc7" />
                </linearGradient>
              </defs>
            </svg>

            {/* Bubble 1: Verse 1 Star rating pinpoint */}
            <div 
              className="absolute top-2 flex flex-col items-center transition-all duration-300 hover:scale-110" 
              style={{ left: `${24 * zoomLevel}%` }}
              title="Excellent pitch accuracy"
            >
              <div className="w-6 h-6 rounded-full bg-[#3cddc7] border-2 border-[#131520] flex items-center justify-center shadow-[0_0_12px_rgba(60,221,199,0.5)] cursor-pointer">
                <Sparkles className="w-3.5 h-3.5 text-[#00201c]" />
              </div>
              <div className="h-10 w-[1px] bg-[#3cddc7]/30 border-dashed" />
            </div>

            {/* Bubble 2: Verse 2 Exclamation warning pinpoint */}
            <div 
              className="absolute top-1.5 flex flex-col items-center transition-all duration-300 hover:scale-110" 
              style={{ left: `${72 * zoomLevel}%` }}
              title="Slight pitch drift detected"
            >
              <div className="w-6 h-6 rounded-full bg-[#ffb1c0] border-2 border-[#131520] flex items-center justify-center shadow-[0_0_12px_rgba(255,177,192,0.5)] cursor-pointer">
                <span className="text-xs font-black text-[#5a0023]">!</span>
              </div>
              <div className="h-11 w-[1px] bg-primary/30 border-dashed" />
            </div>

            {/* Full height thick glowing tracker playhead bar */}
            <div 
              className="absolute top-0 bottom-0 w-[2px] bg-gradient-to-b from-primary/30 via-primary to-primary/30 z-20 shadow-[0_0_12px_#ffb1c0] transition-all duration-150 ease-out pointer-events-none" 
              style={{ left: `${(currentTime / totalDuration) * 100}%` }}
            >
              {/* Sliding Tracker circle cursor with thick white border */}
              <div className="w-4 h-4 rounded-full bg-primary border-2 border-white absolute -left-[7px] top-6 shadow-[0_0_12px_#ffb1c0] z-30 scale-105" />
              <div className="w-3.5 h-3.5 bg-primary rotate-45 absolute -top-1.5 -left-[5px]" />
            </div>
          </div>

          {/* Bar timeline markers metadata */}
          <div className="flex justify-between px-4 mt-2 text-[10px] font-bold tracking-widest text-on-surface-variant/40 uppercase">
            <span>Intro</span>
            <span>Verse 1</span>
            <span>Chorus</span>
            <span>Verse 2</span>
            <span>Bridge</span>
            <span>Outro</span>
          </div>
        </div>

      </footer>

    </div>
  );
}
