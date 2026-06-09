/** One analysis frame from the live streaming backend (10ms hop). */
export interface LiveFrame {
  t_ms: number;
  pitch_hz: number;        // 0 = unvoiced
  voiced: boolean;
  loudness_db: number;     // dBFS, -60 = silence
  breath: boolean;
  onset: boolean;
  vibrato_rate_hz: number;
  vibrato_depth_cents: number;
  tempo_bpm: number;       // 0 = not enough data yet
  technique: string;
  technique_conf: number;
  proc_ms: number;         // server-side batch processing time (RTF = proc_ms / (BATCH_FRAMES * 10))
}

/** How the reference audio was produced. */
export type ReferenceAudioType =
  | "human_vocal"    // real human recording from a dataset (e.g. GTSinger)
  | "synth_melody"   // MIDI-rendered note sequence
  | "generated_tone" // single synthesised sine/saw tone (fallback)
  | "none";          // free singing – no reference

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
  /** URL to a real vocal reference recording served by the backend. */
  referenceAudioUrl?: string;
  /** Short description of the vocal style (e.g. "Vibrato – Female Alto") */
  referenceStyle?: string;
  /** How the reference audio was produced — used to display a badge in the UI. */
  referenceType?: ReferenceAudioType;
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

export interface PracticePreset {
  id: string;
  title: string;
  description: string;
  category: "pitch" | "slide" | "scale" | "song" | "free";
  difficulty: "EASY" | "MEDIUM" | "HARD";
  duration: string;
  source: "generated_reference" | "midi_reference" | "dataset_reference" | "song_reference";
  song: Song;
  taskConfig: TaskConfig;
}

export interface CoachingNote {
  type: "success" | "warning" | "info";
  title: string;
  category: string;
  text: string;
}

export type TaskType =
  | "free_singing"
  | "reference_song"
  | "sustained_note"
  | "pitch_slide"
  | "scale"
  | "interval"
  | "rhythm"
  | "breath_control"
  | "tone_consistency"
  | "note_match"
  | "phrase_practice";

export interface TaskConfig {
  task_type: TaskType | string;
  target?: Record<string, any> | null;
  reference?: Record<string, any> | null;
  skill_focus?: string | string[] | null;
  scoring_mode?: string | null;
  strictness?: string | null;
  expected_duration?: Record<string, number> | number | null;
  expected_direction?: string | null;
}

export type PracticeSessionState =
  | "ready"
  | "listening_to_reference"
  | "recording"
  | "analyzing"
  | "review"
  | "invalid_input"
  | "mic_unavailable"
  | "error";

export interface RecordedAttempt {
  audioBlob: Blob;
  audioUrl: string;
  mimeType: string;
  recordedAt: string;
  sourceLabel?: string;
}

export type InputType =
  | "no_voice_or_noise"
  | "speech_like_or_non_singing"
  | "low_confidence_or_unreliable"
  | "diagnostic_sustained_tone"
  | "diagnostic_pitch_slide"
  | "analyzable_singing"
  | string;

export interface AnalysisValidity {
  is_analyzable: boolean;
  input_type: InputType;
  confidence?: number | null;
  reason_codes?: string[];
  summary_metrics?: Record<string, any>;
}

export interface UiFrame {
  time_s: number;
  frame_index: number;
  f0_hz: number | null;
  target_f0_hz?: number | null;
  target_note?: string | null;
  cents_error?: number | null;
  voiced: boolean;
  voice_confidence?: number | null;
  pitch_confidence?: number | null;
  selected_f0_source?: string;
  selected_vad_source?: string;
  volume?: {
    rms?: number | null;
    rms_db?: number | null;
  };
  spectral_tone_proxy?: Record<string, any>;
  signal_quality?: Record<string, boolean>;
  caveats?: string[];
  debug_flags?: string[];
  source_values?: Record<string, any>;
}

export interface UiSegment {
  id?: string;
  type?: string;
  start_s?: number;
  end_s?: number;
  duration_s?: number;
  median_f0_hz?: number | null;
  min_f0_hz?: number | null;
  max_f0_hz?: number | null;
  stability_cents?: number | null;
  voiced_coverage?: number | null;
  confidence?: number | null;
  source?: string;
  ui_severity?: "info" | "warning" | "error" | string;
  summary?: string;
  actionable_hint?: string;
  target_f0_hz?: number | null;
  sung_median_f0_hz?: number | null;
  median_cents_error?: number | null;
  f0_coverage?: number | null;
  proxy_metrics?: Record<string, any>;
  caveats?: string[];
}

export interface UiSegments {
  notes?: UiSegment[];
  phrases?: UiSegment[];
  dropouts?: UiSegment[];
  unstable_pitch_regions?: UiSegment[];
  low_confidence_regions?: UiSegment[];
  reference_pitch_error_regions?: UiSegment[];
  breath_phrase_proxy_regions?: UiSegment[];
  tone_consistency_proxy_regions?: UiSegment[];
  [key: string]: UiSegment[] | undefined;
}

export interface TaskResult {
  task_type?: string;
  status?: string;
  score_status?: string;
  full_song_score?: number | null;
  diagnostic_score?: number | null;
  summary?: string;
  next_exercise_suggestion?: Record<string, string> | null;
}

export interface FeedbackPolicy {
  allowed_feedback?: string[];
  blocked_feedback?: Array<{
    type?: string;
    reason?: string;
  }>;
  caveats?: string[];
}

export type Subscores = Record<string, any>;

/** Per-sub-score breakdown returned by the backend for pitch_slide tasks. */
export interface PitchSlideScoreBreakdown {
  /** 0–1: how close the opening note was to the target start Hz */
  start_note_accuracy: number | null;
  /** 0–1: how close the closing note was to the target end Hz */
  end_note_accuracy: number | null;
  /** Whether the detected direction matches the expected direction */
  direction_correct: boolean;
  /** 0–1: penalises large frame-to-frame pitch jumps */
  smoothness_score: number;
  /** 0–1: penalises deviation from the ideal linear slide contour (null when no target provided) */
  contour_deviation_score: number | null;
  /** Raw RMS contour deviation in cents */
  contour_deviation_cents: number | null;
  /** Overall composite score 0–100 (capped at 70 when direction wrong) */
  overall: number;
  /** True when the cap was applied */
  score_capped: boolean;
  /** Human-readable feedback lines keyed by sub-score name */
  feedback: Record<string, string>;
}

export interface CoachingCategoryResult {
  status?: "complete" | "not_enough_evidence" | string;
  score?: number | null;
  confidence?: number | null;
  metrics?: Record<string, any>;
  evidence_segments?: Array<Record<string, any>>;
  caveats?: string[];
  recommended_exercise?: string | null;
  source?: string;
}

export interface CoachingCategories {
  schema_version?: string;
  policy?: Record<string, any>;
  vibrato?: CoachingCategoryResult;
  slide?: CoachingCategoryResult;
}

export interface SourceStrategyDebug {
  source_values_included?: boolean;
  source_strategy?: Record<string, any>;
  hybrid_metrics?: Record<string, any>;
  source_summaries?: Record<string, any>;
  existing_eval_json?: string;
  [key: string]: any;
}

export interface UiSpectrogramVisualization {
  kind?: string;
  is_display_downsampled?: boolean;
  sample_rate?: number;
  hop_s?: number;
  source_hop_s?: number;
  frame_stride?: number;
  n_mels?: number;
  frequency_min_hz?: number;
  frequency_max_hz?: number;
  mel_scale?: string;
  value_scale?: string;
  time_s?: number[];
  mel_frequencies_hz?: Array<number | null>;
  values?: number[][];
  caveat?: string;
  error?: string;
}

export interface UiPosteriorgramVisualization {
  kind?: string;
  is_display_downsampled?: boolean;
  hop_s?: number;
  source_hop_s?: number;
  frame_stride?: number;
  row_labels?: string[];
  time_s?: number[];
  values?: number[][];
  caveat?: string;
}

export interface UiVisualizations {
  schema_version?: string;
  spectrogram?: UiSpectrogramVisualization | null;
  posteriorgram?: UiPosteriorgramVisualization | null;
}

export interface ReferenceAlignment {
  method?: string;
  status?: string;
  reference_duration_s?: number;
  aligned_start_s?: number;
  aligned_end_s?: number;
  aligned_duration_s?: number;
  tempo_scale?: number;
  active_frame_count?: number;
  frame_count?: number;
  caveat?: string;
}

export interface UiReadyPerformance {
  schema_version?: string;
  device?: string;
  include_frames?: boolean;
  debug?: boolean;
  audio_duration_s?: number | null;
  h2_frame_export_s?: number | null;
  task_evaluator_s?: number | null;
  visualization_s?: number | null;
  ui_ready_total_s?: number | null;
  ui_ready_realtime_factor?: number | null;
  checkpoint_coaching_s?: number | null;
  build_ui_ready_response_s?: number | null;
  endpoint_total_s?: number | null;
  caveat?: string;
}

export interface UiReadyAnalysis {
  schema_version?: string;
  input_path?: string;
  audio?: {
    duration_s?: number;
    sample_rate?: number;
    hop_s?: number;
    channels?: number;
  };
  task_config?: TaskConfig;
  analysis_validity?: AnalysisValidity;
  frames?: UiFrame[];
  segments?: UiSegments;
  task_result?: TaskResult;
  subscores?: Subscores;
  feedback_policy?: FeedbackPolicy;
  reference_alignment?: ReferenceAlignment;
  performance?: UiReadyPerformance;
  proxy_metrics?: Record<string, any>;
  h3_task_evaluator?: Record<string, any>;
  coaching_categories?: CoachingCategories;
  visualizations?: UiVisualizations;
  debug?: SourceStrategyDebug;
}

export type AnalysisPayload = MLAnalysisResult | UiReadyAnalysis | Record<string, any>;

// ---------------------------------------------------------------------------
// Human reference catalog types (Phase 3 API contract)
// ---------------------------------------------------------------------------

export interface F0Summary {
  mean_hz: number;
  std_hz: number;
  voiced_fraction: number;
}

/**
 * One entry from GET /api/reference/catalog or /api/reference/exercise/{id}.
 * The target_pitch_vector is a pre-computed F0 array at 10 ms hop (0.0 = unvoiced).
 * reference_audio_url points to the WAV served by /api/audio/file — consumable
 * by a native <audio> element with no custom synthesis required.
 */
export interface ExerciseReferencePayload {
  schema_version?: string;
  asset_id: string;
  dataset: "vocalset" | "mir1k" | "gtsinger" | string;
  exercise_type: "sustained_note" | "vibrato" | "pitch_slide" | "long_note" | string;
  exercise_type_tags?: string[];
  singer_id: string;
  voice_type?: string;
  technique: string;
  note_name?: string | null;
  f0_target_hz?: number | null;
  duration_s: number;
  hop_s: number;
  /** Backend-served URL for the native <audio> player — no synthesis required. */
  audio_url: string;
  /** Per-asset NPZ download URL (rarely needed by frontend). */
  vector_url?: string;
  /** Pre-computed F0 array at hop_s resolution. Length = floor(duration_s / hop_s). */
  target_pitch_vector: number[];
  voiced_vector: boolean[];
  f0_summary: F0Summary;
  task_config: TaskConfig;
}

export interface ReferenceCatalogResponse {
  schema_version: string;
  total_matched: number;
  limit: number;
  entries: ExerciseReferencePayload[];
  warning?: string;
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
  uiReadyAnalysis?: UiReadyAnalysis;
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
  uiReadyAnalysis?: UiReadyAnalysis;
  analysisUnavailable?: boolean;
  analysisError?: string;
  taskConfig?: TaskConfig;
  sessionState?: PracticeSessionState;
  recordedAt?: string;
}
