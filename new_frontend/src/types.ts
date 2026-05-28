export interface Song {
  id: string;
  title: string;
  artist: string;
  genre: string;
  difficulty: "EASY" | "MEDIUM" | "HARD";
  duration: string;
  bpm: number;
  imageUrl: string;
  featured?: boolean;
  lyrics: string[];
  referencePitchSeq: number[]; // Sequence of heights 0-100 indicating target notes
}

export interface Exercise {
  id: string;
  title: string;
  description: string;
  duration: string;
  type: "breath" | "pitch" | "agility" | "assessment";
  difficulty: "EASY" | "MEDIUM" | "HARD";
  progress: number; // 0 to 100
}

export interface CoachingNote {
  type: "success" | "warning" | "info";
  title: string;
  category: string;
  text: string;
}

// ML Analysis Types
export interface NoteSegment {
  startSeconds: number;
  durationSeconds: number;
  pitchHz: number;
  noteName: string;
  centsError: number;
  stabilityCents: number;
  vibrato?: VibratoInfo;
}

export interface VibratoInfo {
  rateHz: number;
  depthCents: number;
  regularity: number;
}

export interface VoiceQuality {
  hnrDb: number;
  jitterPercent: number;
  shimmerPercent: number;
  breathiness: "clear" | "mild" | "breathy";
  isUnstable: boolean;
}

export interface FrameData {
  pitch: number[];
  voiced: number[];
  breath: number[];
  onset: number[];
  hopLength: number;
}

export interface MLAnalysisResult {
  score: number;
  summary: string;
  issues: string[];
  exercises: string[];
  songTitle: string;
  artist: string;
  pitchAccuracy: number;
  pitchDrift: number;
  phraseLengths: number[];
  breathCount: number;
  onsetCount: number;
  onsetClarity: number;
  technique: string;
  techniqueConfidence: number;
  allTechniqueScores: Record<string, number>;
  notes: NoteSegment[];
  voiceQuality: VoiceQuality | null;
  vibrato: Record<string, any>;
  frameData: FrameData;
}

export interface PerformanceResult {
  songId: string;
  songTitle: string;
  artist: string;
  overallScore: number;
  intonation: number;
  rhythm: number;
  timbre: number;
  dynamics: number;
  coachingNotes: CoachingNote[];
  // ML analysis data (optional for backwards compatibility)
  mlAnalysis?: MLAnalysisResult;
  recordedAt?: string;
}
