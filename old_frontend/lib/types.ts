export interface VibratoInfo {
  rate_hz: number
  depth_cents: number
  regularity: number
}

export interface NoteSegment {
  start_s: number
  duration_s: number
  pitch_hz: number
  note_name: string
  cents_error: number
  stability_cents: number
  vibrato: VibratoInfo | null
}

export interface VoiceQuality {
  hnr_db: number
  jitter_pct: number
  shimmer_pct: number
  breathiness: 'clear' | 'mild' | 'breathy'
  is_unstable: boolean
}

export interface CoachingResult {
  pitch_hz: number[]
  voiced: boolean[]
  breath_frames: boolean[]
  onset_frames: boolean[]
  hop_s: number

  technique: string
  technique_confidence: number
  all_technique_scores: Record<string, number>

  pitch_accuracy: number
  pitch_drift_cents: number
  phrase_lengths_s: number[]
  breath_count: number
  onset_count: number
  onset_clarity: number

  notes: NoteSegment[]
  voice_quality: VoiceQuality | null
  vibrato_stats: Record<string, number>

  score: number
  summary: string
  issues: string[]
  exercises: string[]
}
