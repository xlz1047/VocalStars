import { PracticePreset, TaskConfig, UiReadyAnalysis } from "../types";
import { PRACTICE_PRESETS } from "./musicDb";

export type ImprovementConfidence = "high" | "medium" | "low";

export interface ImprovementFocus {
  id: string;
  label: string;
  summary: string;
  severity: number;
  confidence: ImprovementConfidence;
  evidence: string[];
  practiceLabel: string;
  taskConfig: TaskConfig;
  /** When set, the focus links to a specific PRACTICE_PRESETS entry.
   *  The UI uses this to show the preset's title, badge, and referenceStyle. */
  presetId?: string;
  caveat?: string;
}

/** Maps technique/skill tags to the best matching GTSinger (or synth) preset. */
const TECHNIQUE_PRESET_MAP: Record<string, { presetId: string; practiceLabel: string; summary: string }> = {
  vibrato:          { presetId: "gts-vibrato-alto1-memory",             practiceLabel: "Listen & match: vibrato (human vocal)",      summary: "Hear a real vibrato example, then try to match the steady oscillation on long notes." },
  pitch_stability:  { presetId: "sustained-c4",                         practiceLabel: "Practice steady sustain",                    summary: "Build a more stable pitch centre before adding phrases." },
  breath_control:   { presetId: "gts-breathy-alto2-easy-on-me",         practiceLabel: "Listen & match: breathy tone (human vocal)", summary: "Hear how a breathy tone sounds and practice the open, airy onset." },
  tone_consistency: { presetId: "gts-breathy-alto1-let-it-go",          practiceLabel: "Listen & match: breathy tone (human vocal)", summary: "Sustain a consistent tonal quality through each phrase." },
  slide_smoothness: { presetId: "rising-slide-220-440",                 practiceLabel: "Practice a slow vocal siren",                summary: "Slow, uninterrupted glides build the muscle memory for smooth transitions." },
  register_blend:   { presetId: "gts-mixed-tenor1-shallow",             practiceLabel: "Listen & match: mixed voice (human vocal)",  summary: "Hear a seamless chest-to-head blend and practice smoothing the passaggio." },
  glissando:        { presetId: "gts-glissando-alto2-someone-like-you", practiceLabel: "Listen & match: glissando (human vocal)",   summary: "Hear expressive inter-note slides and practise matching the glide shape." },
  phrase_continuity:{ presetId: "free-human-phrase",                    practiceLabel: "Sing a short connected phrase",              summary: "Focus on keeping one comfortable phrase connected without dropping out." },
};

/** Module-level registry for dynamically fetched catalog presets.
 *  Populated by registerPresets() called from useHumanReferencePresets. */
const _dynamicPresets = new Map<string, PracticePreset>();

/** Register API-fetched presets so resolvePreset can find them. */
export function registerPresets(presets: PracticePreset[]): void {
  for (const p of presets) _dynamicPresets.set(p.id, p);
}

export function resolvePreset(presetId: string): PracticePreset | undefined {
  return _dynamicPresets.get(presetId) ?? PRACTICE_PRESETS.find((p) => p.id === presetId);
}

export interface ImprovementPath {
  schemaVersion: "improvement_path.v1";
  confidence: ImprovementConfidence;
  primaryFocus: ImprovementFocus | null;
  rankedFocusAreas: ImprovementFocus[];
}

const INVALID_INPUT_TYPES = new Set(["no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"]);

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function confidenceFrom(score: number, proxyOnly = false): ImprovementConfidence {
  if (proxyOnly) return score >= 0.75 ? "medium" : "low";
  if (score >= 0.78) return "high";
  if (score >= 0.5) return "medium";
  return "low";
}

function sustainedNoteTask(): TaskConfig {
  return {
    task_type: "sustained_note",
    target: { note: "C4", f0_hz: 261.63 },
    reference: null,
    skill_focus: ["pitch_stability", "voiced_continuity"],
    scoring_mode: "diagnostic",
    strictness: "beginner",
    expected_duration: 5,
  };
}

function noteMatchTask(): TaskConfig {
  return {
    task_type: "note_match",
    target: { note: "E4", f0_hz: 329.63 },
    reference: null,
    skill_focus: ["pitch_matching"],
    scoring_mode: "diagnostic",
    strictness: "beginner",
  };
}

function pitchSlideTask(): TaskConfig {
  return {
    task_type: "pitch_slide",
    target: { start_f0_hz: 220, end_f0_hz: 440 },
    reference: { type: "generated_glide", start_f0_hz: 220, end_f0_hz: 440 },
    skill_focus: ["slide_smoothness", "continuity"],
    scoring_mode: "diagnostic",
    strictness: "beginner",
    expected_duration: 5,
    expected_direction: "up",
  };
}

function freePhraseTask(): TaskConfig {
  return {
    task_type: "free_singing",
    target: null,
    reference: null,
    skill_focus: ["general_pitch", "phrase_continuity"],
    scoring_mode: "no_reference",
    strictness: "beginner",
  };
}

function addFocus(foci: ImprovementFocus[], focus: ImprovementFocus | null) {
  if (!focus || focus.severity < 0.25) return;
  foci.push(focus);
}

export function buildImprovementPath(analysis: UiReadyAnalysis): ImprovementPath {
  const validity = analysis.analysis_validity;
  if (validity?.input_type && INVALID_INPUT_TYPES.has(validity.input_type)) {
    return {
      schemaVersion: "improvement_path.v1",
      confidence: "low",
      primaryFocus: null,
      rankedFocusAreas: [],
    };
  }

  const subscores = analysis.subscores || {};
  const raw = (subscores.raw_metrics || {}) as Record<string, unknown>;
  const proxy = (subscores.proxy_metrics || analysis.proxy_metrics || {}) as Record<string, any>;
  const foci: ImprovementFocus[] = [];

  const pitchStabilityScore = asNumber(subscores.pitch_stability)
    ?? asNumber(subscores.general_pitch_stability)
    ?? null;
  const f0StabilityCents = asNumber(raw.f0_stability_cents);
  const dropoutRate = asNumber(subscores.dropout_rate);
  const pitchSeverity = pitchStabilityScore !== null
    ? 1 - pitchStabilityScore
    : f0StabilityCents !== null
    ? clamp01(f0StabilityCents / 180)
    : 0;
  addFocus(foci, {
    id: "pitch_stability",
    label: "Pitch stability",
    summary: "Build steadier pitch before increasing phrase length.",
    severity: pitchSeverity,
    confidence: confidenceFrom(pitchSeverity),
    evidence: [
      f0StabilityCents !== null ? `${Math.round(f0StabilityCents)} cents stability spread` : null,
      dropoutRate !== null ? `${Math.round(dropoutRate * 100)}% selected-f0 dropout` : null,
    ].filter(Boolean) as string[],
    practiceLabel: "Practice steady sustain",
    taskConfig: sustainedNoteTask(),
  });

  const referenceAccuracy = asNumber(subscores.reference_pitch_accuracy);
  const medianAbs = asNumber(raw.median_abs_cents_error);
  const referenceCoverage = asNumber(subscores.reference_f0_coverage);
  const referenceSeverity = referenceAccuracy !== null
    ? 1 - referenceAccuracy
    : medianAbs !== null
    ? clamp01(medianAbs / 250)
    : 0;
  addFocus(foci, {
    id: "reference_pitch",
    label: "Melody pitch matching",
    summary: "Work note by note against the reference contour.",
    severity: Math.max(referenceSeverity, referenceCoverage !== null ? clamp01(0.65 - referenceCoverage) : 0),
    confidence: confidenceFrom(referenceSeverity),
    evidence: [
      medianAbs !== null ? `${Math.round(medianAbs)} cents median reference error` : null,
      referenceCoverage !== null ? `${Math.round(referenceCoverage * 100)}% reference f0 coverage` : null,
    ].filter(Boolean) as string[],
    practiceLabel: "Practice note matching",
    taskConfig: noteMatchTask(),
    caveat: "Reference feedback is provisional contour guidance, not full song scoring.",
  });

  const slideBreakdown = subscores.pitch_slide_breakdown as Record<string, any> | undefined;
  const smoothness = asNumber(slideBreakdown?.smoothness_score ?? subscores.slide_smoothness);
  const contour = asNumber(slideBreakdown?.contour_deviation_score);
  const directionWrong = slideBreakdown?.direction_correct === false;
  const slideSeverity = Math.max(
    smoothness !== null ? 1 - smoothness : 0,
    contour !== null ? 1 - contour : 0,
    directionWrong ? 0.9 : 0,
  );
  addFocus(foci, {
    id: "slide_smoothness",
    label: "Slide control",
    summary: "Slow glides will help connect notes without jumps.",
    severity: slideSeverity,
    confidence: confidenceFrom(slideSeverity),
    evidence: [
      smoothness !== null ? `${Math.round(smoothness * 100)}% smoothness` : null,
      directionWrong ? "slide direction mismatch" : null,
    ].filter(Boolean) as string[],
    practiceLabel: "Practice a slow siren",
    taskConfig: pitchSlideTask(),
  });

  const phraseContinuity = asNumber(subscores.phrase_continuity);
  const phraseSeverity = phraseContinuity !== null ? 1 - phraseContinuity : 0;
  addFocus(foci, {
    id: "phrase_continuity",
    label: "Phrase continuity",
    summary: "Short connected phrases are the next useful step.",
    severity: phraseSeverity,
    confidence: confidenceFrom(phraseSeverity, true),
    evidence: phraseContinuity !== null ? [`${Math.round(phraseContinuity * 100)}% phrase continuity proxy`] : [],
    practiceLabel: "Practice a short phrase",
    taskConfig: freePhraseTask(),
    caveat: "This is a continuity proxy, not a breath-support diagnosis.",
  });

  const breathProxyRegions = analysis.segments?.breath_phrase_proxy_regions?.length || 0;
  const toneProxyRegions = analysis.segments?.tone_consistency_proxy_regions?.length || 0;
  const proxySeverity = clamp01((breathProxyRegions + toneProxyRegions) / 5);
  addFocus(foci, {
    id: "recording_or_tone_consistency",
    label: "Consistency through the take",
    summary: "Repeat a short comfortable phrase and keep the signal clean.",
    severity: proxySeverity,
    confidence: confidenceFrom(proxySeverity, true),
    evidence: [
      breathProxyRegions ? `${breathProxyRegions} phrase-continuity proxy regions` : null,
      toneProxyRegions ? `${toneProxyRegions} tone-consistency proxy regions` : null,
    ].filter(Boolean) as string[],
    practiceLabel: "Practice a short phrase",
    taskConfig: freePhraseTask(),
    caveat: "Proxy regions can guide review, but they are not technique or vocal-health diagnoses.",
  });

  // ── Technique-specific recommendations ─────────────────────────────────────
  // If the task had a declared skill_focus (e.g. vibrato, glissando) and the
  // overall score was low, surface the matching GTSinger human-vocal preset.
  const score = asNumber(analysis.task_result?.diagnostic_score ?? analysis.task_result?.full_song_score);
  const skillFocusRaw = analysis.task_config?.skill_focus;
  const skillTags: string[] = Array.isArray(skillFocusRaw)
    ? skillFocusRaw.map(String)
    : typeof skillFocusRaw === "string"
    ? [skillFocusRaw]
    : [];

  for (const tag of skillTags) {
    const mapping = TECHNIQUE_PRESET_MAP[tag];
    if (!mapping) continue;
    const preset = resolvePreset(mapping.presetId);
    if (!preset) continue;
    // Only surface when score is absent (unknown) or below 70.
    const scoreLow = score === null || score < 70;
    if (!scoreLow) continue;
    const severity = score !== null ? clamp01(1 - score / 100) : 0.55;
    addFocus(foci, {
      id: `technique_${tag}`,
      label: preset.title,
      summary: mapping.summary,
      severity,
      confidence: "medium",
      evidence: [
        score !== null ? `Score ${score}/100` : "Score unavailable",
        `Skill focus: ${tag.replaceAll("_", " ")}`,
      ],
      practiceLabel: mapping.practiceLabel,
      taskConfig: preset.taskConfig,
      presetId: mapping.presetId,
    });
    break; // only the first matching tag per session
  }

  const ranked = foci
    .filter((focus) => focus.evidence.length > 0 || focus.severity >= 0.45)
    .sort((a, b) => b.severity - a.severity)
    .slice(0, 3);
  const primary = ranked[0] || null;
  return {
    schemaVersion: "improvement_path.v1",
    confidence: primary?.confidence || "low",
    primaryFocus: primary,
    rankedFocusAreas: ranked,
  };
}
