import { useState, useEffect } from "react";
import { motion } from "motion/react";
import {
  RotateCcw,
  ArrowLeft,
  Award,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Info,
  Layers,
  Activity,
  ChevronRight,
  TrendingUp,
  Wind,
  Music,
  Loader
} from "lucide-react";
import { PerformanceResult, CoachingNote } from "../types";

interface ResultsViewProps {
  result: PerformanceResult;
  onRetake: () => void;
  onBackToDashboard: () => void;
}

export default function ResultsView({ 
  result, 
  onRetake, 
  onBackToDashboard 
}: ResultsViewProps) {
  const [geminiNotes, setGeminiNotes] = useState<CoachingNote[] | null>(null);
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [geminiAvailable, setGeminiAvailable] = useState<boolean | null>(null);
  const [geminiError, setGeminiError] = useState<string | null>(null);

  // Check once on mount whether the Gemini API key is configured
  useEffect(() => {
    fetch("/api/coaching-status")
      .then(r => r.json())
      .then(d => setGeminiAvailable(d.available))
      .catch(() => setGeminiAvailable(false));
  }, []);

  async function handleGetAICoaching() {
    setIsLoadingAI(true);
    setGeminiError(null);
    try {
      const response = await fetch("/api/coaching-feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          songTitle: result.songTitle,
          artist: result.artist,
          score: result.overallScore,
          intonation: result.intonation,
          rhythm: result.rhythm,
          timbre: result.timbre,
          dynamics: result.dynamics,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setGeminiError(data.error || "AI coaching failed");
        return;
      }
      if (data.coachingNotes?.length > 0) {
        setGeminiNotes(data.coachingNotes);
      }
    } catch (err: any) {
      setGeminiError(err.message || "Network error");
    } finally {
      setIsLoadingAI(false);
    }
  }

  // Circumference calculation for SVG score ring (Radius 110 => Circumference roughly 691.15)
  const radius = 110;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (result.overallScore / 100) * circumference;

  const getWeightDescription = (label: string) => {
    switch (label) {
      case "Intonation": return "STABILITY & TONALITY (40% WEIGHT)";
      case "Rhythm": return "TEMPORAL PRECISION & OFFSET NOTES (25% WEIGHT)";
      case "Timbre": return "RESONANCE & HARMONIC OVERTONES (20% WEIGHT)";
      default: return "EXPRESSIVE DYNAMICS & VOLUME CONTROL (15% WEIGHT)";
    }
  };

  const getMetricColor = (val: number) => {
    if (val > 85) return "text-primary";
    if (val > 70) return "text-tertiary";
    return "text-secondary";
  };

  return (
    <div className="space-y-10 pb-12 animate-fade-in relative z-10">
      
      {/* Floating abstract decorative background blurs */}
      <div className="absolute top-[-80px] left-[-80px] w-96 h-96 rounded-full bg-primary/10 blur-[90px] pointer-events-none -z-10" />
      <div className="absolute bottom-[-100px] right-[-100px] w-96 h-96 rounded-full bg-secondary/10 blur-[90px] pointer-events-none -z-10" />

      {/* Title Header with Breadcrumbs */}
      <section className="text-center md:text-left flex flex-col md:flex-row justify-between items-center gap-6 bg-surface-container/20 p-6 rounded-2xl border border-white/5">
        <div>
          <h1 className="font-display font-extrabold text-3xl md:text-4xl text-white mb-2">
            Session Complete
          </h1>
          <p className="text-on-surface-variant font-sans text-sm md:text-base">
            You just sang <span className="text-secondary font-bold">"{result.songTitle}"</span> by {result.artist}.
          </p>
          <div className="mt-2.5 flex items-center justify-center md:justify-start gap-1.5 text-on-surface-variant/50 text-[10px] font-bold tracking-wider uppercase">
            <CheckCircle2 className="w-4.5 h-4.5 text-tertiary" />
            <span>POWERED BY PESNQ WEIGHTED BIO-COACH ANALYSIS</span>
          </div>
        </div>
        <button 
          onClick={onBackToDashboard}
          className="flex items-center gap-2 px-5 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all font-semibold text-xs"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Dashboard
        </button>
      </section>

      {/* Main feedback dashboard structure */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Col: Centerpiece Circular Score dial */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center glass-card rounded-[24px] p-8 md:p-10 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-transparent to-secondary/5 pointer-events-none" />
          
          <div className="relative w-64 h-64 flex items-center justify-center mb-6">
            <svg className="w-full h-full transform -rotate-90">
              {/* Outer static circle */}
              <circle 
                className="text-surface-container-highest"
                cx="128" 
                cy="128" 
                fill="transparent" 
                r={radius} 
                stroke="currentColor" 
                strokeWidth="12" 
              />
              {/* Filled tracking score circle */}
              <motion.circle 
                className="text-primary"
                cx="128" 
                cy="128" 
                fill="transparent" 
                r={radius} 
                stroke="currentColor" 
                strokeWidth="12" 
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: strokeDashoffset }}
                transition={{ duration: 1.2, ease: "easeOut" }}
                strokeLinecap="round"
                style={{
                  filter: 'drop-shadow(0 0 12px rgba(255,177,192,0.4))'
                }}
              />
            </svg>

            {/* Overall center numbers */}
            <div className="absolute inset-0 flex flex-col items-center justify-center mt-[-4px]">
              <span className="font-display font-black text-6xl text-primary md:text-7xl">
                {result.overallScore}
              </span>
              <span className="text-[10px] font-bold tracking-widest text-on-surface-variant mt-1 uppercase">
                Vocal Rating
              </span>
            </div>
          </div>

          <div className="text-center space-y-3 px-2">
            <h2 className="font-display font-extrabold text-xl md:text-2xl text-secondary">
              {result.overallScore > 90 ? "Grammy-Ready Pitch!" : 
               result.overallScore > 80 ? "Stunning High Notes!" : "Magnificent Control!"}
            </h2>
            <p className="text-xs md:text-sm text-on-surface-variant/90 leading-relaxed max-w-sm">
              Your vocal expansion range is blooming beautifully. Deep control preserved tone stability accurately through challenging keys transitions.
            </p>
            <div className="pt-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-surface-container-highest border border-white/5 rounded-full text-[10px] font-bold text-on-surface uppercase tracking-wider">
                <Wind className="w-3.5 h-3.5 text-primary" />
                Integrated Resonance: <span className="text-primary">OPTIMAL</span>
              </span>
            </div>
          </div>
        </div>

        {/* Right Col: Breakdown parameters list & smart notes */}
        <div className="lg:col-span-7 flex flex-col gap-8">
          
          {/* Detailed Performance Breakdown */}
          <div className="glass-card rounded-[24px] p-6 md:p-8">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-display text-lg font-bold text-on-surface">
                Detailed Performance Breakdown
              </h3>
              <Info className="w-5 h-5 text-on-surface-variant/40" />
            </div>

            <div className="space-y-6">
              {[
                { label: "Intonation", val: result.intonation, colorClass: "from-secondary to-primary" },
                { label: "Rhythm", val: result.rhythm, colorClass: "to-tertiary from-tertiary/60" },
                { label: "Timbre", val: result.timbre, colorClass: "from-secondary to-secondary-fixed" },
                { label: "Dynamics", val: result.dynamics, colorClass: "from-primary to-primary-container" },
              ].map((item, idx) => (
                <div key={idx} className="space-y-2">
                  <div className="flex justify-between items-end">
                    <div className="flex flex-col">
                      <span className="text-sm font-bold text-on-surface">
                        {item.label}
                      </span>
                      <span className="text-[9px] text-on-surface-variant/40 font-bold uppercase tracking-wider">
                        {getWeightDescription(item.label)}
                      </span>
                    </div>
                    <span className={`font-mono text-sm font-bold ${getMetricColor(item.val)}`}>
                      {item.val}%
                    </span>
                  </div>
                  <div className="w-full h-3.5 bg-surface-container-highest rounded-full overflow-hidden border border-white/5">
                    <motion.div 
                      className={`h-full rounded-full bg-gradient-to-r ${item.colorClass}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${item.val}%` }}
                      transition={{ duration: 1, delay: idx * 0.15 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ML Coaching Feedback (primary) + Gemini enhancement */}
          <div className="glass-card rounded-[24px] p-6 md:p-8 relative overflow-hidden border border-tertiary/10">
            <div className="absolute top-0 right-0 w-32 h-32 rounded-full bg-tertiary/5 blur-[50px] pointer-events-none" />

            {/* Section header */}
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                <h3 className="font-display text-lg font-bold text-on-surface">
                  {geminiNotes ? "AI Coaching Feedback" : "Vocal Model Coaching"}
                </h3>
              </div>
              <span className="text-[10px] font-bold text-primary bg-primary/10 border border-primary/20 px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                {geminiNotes ? "Gemini" : "ML Model"}
              </span>
            </div>

            {/* ML model summary */}
            {!geminiNotes && result.mlAnalysis?.summary && (
              <p className="text-sm text-on-surface-variant/90 mb-5 leading-relaxed">
                {result.mlAnalysis.summary}
              </p>
            )}

            {/* Gemini notes (after user clicks the button) */}
            {geminiNotes ? (
              <ul className="space-y-6 mb-6">
                {geminiNotes.map((note, idx) => (
                  <li key={idx} className="flex gap-4">
                    <div className="mt-1 flex-shrink-0">
                      {note.type === "success" ? (
                        <div className="w-7 h-7 bg-tertiary/10 rounded-full flex items-center justify-center text-tertiary">
                          <CheckCircle2 className="w-4 h-4" />
                        </div>
                      ) : note.type === "warning" ? (
                        <div className="w-7 h-7 bg-primary/10 rounded-full flex items-center justify-center text-primary">
                          <AlertCircle className="w-4 h-4" />
                        </div>
                      ) : (
                        <div className="w-7 h-7 bg-secondary/10 rounded-full flex items-center justify-center text-secondary">
                          <Info className="w-4 h-4" />
                        </div>
                      )}
                    </div>
                    <div className="flex-1">
                      <span className="text-[9px] font-bold text-tertiary uppercase tracking-wider">
                        {note.category}
                      </span>
                      <h4 className="font-bold text-sm text-white mb-0.5">{note.title}</h4>
                      <p className="text-xs text-on-surface-variant/90 leading-relaxed">{note.text}</p>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              /* ML model issues + exercises */
              <div className="space-y-4 mb-6">
                {result.mlAnalysis?.issues && result.mlAnalysis.issues.length > 0 ? (
                  result.mlAnalysis.issues.map((issue: string, idx: number) => (
                    <div key={idx} className="flex gap-3">
                      <div className="mt-0.5 flex-shrink-0 w-7 h-7 bg-primary/10 rounded-full flex items-center justify-center">
                        <AlertCircle className="w-4 h-4 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="text-xs font-semibold text-white mb-0.5">{issue}</p>
                        {result.mlAnalysis?.exercises?.[idx] && (
                          <p className="text-[11px] text-tertiary leading-relaxed">
                            Exercise: {result.mlAnalysis.exercises[idx]}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  /* fallback coachingNotes if ML analysis unavailable */
                  <ul className="space-y-6">
                    {result.coachingNotes.map((note, idx) => (
                      <li key={idx} className="flex gap-4">
                        <div className="mt-1 flex-shrink-0">
                          {note.type === "success" ? (
                            <div className="w-7 h-7 bg-tertiary/10 rounded-full flex items-center justify-center text-tertiary">
                              <CheckCircle2 className="w-4 h-4" />
                            </div>
                          ) : note.type === "warning" ? (
                            <div className="w-7 h-7 bg-primary/10 rounded-full flex items-center justify-center text-primary">
                              <AlertCircle className="w-4 h-4" />
                            </div>
                          ) : (
                            <div className="w-7 h-7 bg-secondary/10 rounded-full flex items-center justify-center text-secondary">
                              <Info className="w-4 h-4" />
                            </div>
                          )}
                        </div>
                        <div className="flex-1">
                          <span className="text-[9px] font-bold text-tertiary uppercase tracking-wider">
                            {note.category}
                          </span>
                          <h4 className="font-bold text-sm text-white mb-0.5">{note.title}</h4>
                          <p className="text-xs text-on-surface-variant/90 leading-relaxed">{note.text}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {/* Get AI Coaching button */}
            {!geminiNotes && (
              <div className="border-t border-white/5 pt-5">
                <button
                  onClick={handleGetAICoaching}
                  disabled={geminiAvailable === false || isLoadingAI}
                  title={geminiAvailable === false ? "Gemini API key not configured — add GEMINI_API_KEY to .env" : undefined}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-tertiary/20 to-primary/20 border border-tertiary/30 text-sm font-bold text-white hover:brightness-110 active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100"
                >
                  {isLoadingAI ? (
                    <Loader className="w-4 h-4 animate-spin" />
                  ) : (
                    <Sparkles className="w-4 h-4 text-tertiary" />
                  )}
                  {isLoadingAI
                    ? "Consulting AI Coach…"
                    : geminiAvailable === false
                    ? "AI Coaching Unavailable"
                    : "Get AI Coaching"}
                </button>
                {geminiError && (
                  <p className="mt-2 text-[11px] text-primary text-center">{geminiError}</p>
                )}
                {geminiAvailable === false && (
                  <p className="mt-2 text-[10px] text-on-surface-variant/50 text-center">
                    Add <span className="font-bold text-primary/70">GEMINI_API_KEY</span> to{" "}
                    <code className="text-tertiary/70">new_frontend/.env</code> to enable
                  </p>
                )}
              </div>
            )}

            {/* Reset to ML coaching if user wants to go back */}
            {geminiNotes && (
              <button
                onClick={() => setGeminiNotes(null)}
                className="mt-4 text-[11px] text-on-surface-variant/50 hover:text-on-surface-variant transition-colors underline underline-offset-2"
              >
                Show ML model coaching instead
              </button>
            )}
          </div>

        </div>

      </div>

      {/* ML Analysis Detailed Metrics - New Section */}
      {result.mlAnalysis && (
        <section className="space-y-6">
          <h2 className="font-display font-extrabold text-xl md:text-2xl text-white">
            Advanced ML Analysis
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Pitch Metrics */}
            <div className="glass-card rounded-[24px] p-6 border border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-5 h-5 text-primary" />
                <h4 className="font-bold text-sm text-on-surface">Pitch Analysis</h4>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Accuracy</span>
                  <span className="text-sm font-bold text-primary">{result.mlAnalysis.pitchAccuracy.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Drift (cents)</span>
                  <span className="text-sm font-bold text-tertiary">{result.mlAnalysis.pitchDrift.toFixed(1)}</span>
                </div>
              </div>
            </div>

            {/* Breath & Onset Detection */}
            <div className="glass-card rounded-[24px] p-6 border border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <Wind className="w-5 h-5 text-secondary" />
                <h4 className="font-bold text-sm text-on-surface">Breath & Onset</h4>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Breath Points</span>
                  <span className="text-sm font-bold text-secondary">{result.mlAnalysis.breathCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Onsets Detected</span>
                  <span className="text-sm font-bold text-secondary">{result.mlAnalysis.onsetCount}</span>
                </div>
              </div>
            </div>

            {/* Technique Detection */}
            <div className="glass-card rounded-[24px] p-6 border border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-tertiary" />
                <h4 className="font-bold text-sm text-on-surface">Technique</h4>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Detected</span>
                  <span className="text-sm font-bold text-tertiary capitalize">{result.mlAnalysis.technique}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Confidence</span>
                  <span className="text-sm font-bold text-primary">{result.mlAnalysis.techniqueConfidence.toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Voice Quality Metrics */}
            {result.mlAnalysis.voiceQuality && (
              <div className="glass-card rounded-[24px] p-6 border border-white/5">
                <div className="flex items-center gap-2 mb-4">
                  <Sparkles className="w-5 h-5 text-primary" />
                  <h4 className="font-bold text-sm text-on-surface">Voice Quality</h4>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-on-surface-variant">HNR (dB)</span>
                    <span className="text-sm font-bold text-primary">{result.mlAnalysis.voiceQuality.hnrDb.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-on-surface-variant">Jitter</span>
                    <span className="text-sm font-bold text-secondary">{result.mlAnalysis.voiceQuality.jitterPercent.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-on-surface-variant">Breathiness</span>
                    <span className="text-sm font-bold text-tertiary capitalize">{result.mlAnalysis.voiceQuality.breathiness}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Phrase Information */}
            <div className="glass-card rounded-[24px] p-6 border border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-5 h-5 text-secondary" />
                <h4 className="font-bold text-sm text-on-surface">Phrases</h4>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-on-surface-variant">Count</span>
                  <span className="text-sm font-bold text-secondary">{result.mlAnalysis.phraseLengths.length}</span>
                </div>
                {result.mlAnalysis.phraseLengths.length > 0 && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-on-surface-variant">Avg Length</span>
                      <span className="text-sm font-bold text-primary">
                        {(result.mlAnalysis.phraseLengths.reduce((a: number, b: number) => a + b, 0) / result.mlAnalysis.phraseLengths.length).toFixed(1)}s
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Notes Detected */}
            {result.mlAnalysis.notes.length > 0 && (
              <div className="glass-card rounded-[24px] p-6 border border-white/5">
                <div className="flex items-center gap-2 mb-4">
                  <Music className="w-5 h-5 text-primary" />
                  <h4 className="font-bold text-sm text-on-surface">Notes</h4>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-on-surface-variant">Detected</span>
                    <span className="text-sm font-bold text-primary">{result.mlAnalysis.notes.length}</span>
                  </div>
                  {result.mlAnalysis.notes.length > 0 && (
                    <div className="text-[10px] text-on-surface-variant/70">
                      Range: {Math.min(...result.mlAnalysis.notes.map((n: any) => n.noteName))} - {Math.max(...result.mlAnalysis.notes.map((n: any) => n.noteName))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

        </section>
      )}

      {/* Primary actions row */}
      <section className="flex flex-col sm:flex-row items-center justify-center gap-6 pt-6">
        <button 
          onClick={onRetake}
          className="w-full sm:w-auto px-10 py-4 rounded-full border-2 border-primary text-primary text-sm font-bold hover:bg-primary/10 active:scale-95 transition-all duration-200 flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-4 h-4" />
          Review Session Pitch
        </button>
        <button 
          onClick={onBackToDashboard}
          className="w-full sm:w-auto px-10 py-4 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary text-sm font-bold hover:brightness-110 hover:shadow-[0_0_20px_rgba(255,177,192,0.5)] active:scale-95 transition-all duration-200 flex items-center justify-center gap-2 glow-pink"
        >
          <RotateCcw className="w-4 h-4 text-on-primary" />
          Try Another Song
        </button>
      </section>

    </div>
  );
}
