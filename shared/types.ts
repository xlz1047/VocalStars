export interface AnalysisInsight {
  pitch_stability: string
  rhythm_timing: string
  breath_consistency: string
  vocal_stability: string
  transition_quality: string
  strain_indicators: string
}

export interface CoachingRecommendation {
  category: string
  details: Record<string, string>
}

export interface ProgressMetric {
  metric_name: string
  values: Record<string, number>
}
