/**
 * Client for the /api/reference/* human reference catalog endpoints.
 *
 * All reference tracks are pre-vectorized offline. The target_pitch_vector
 * is a float array at 10 ms hop — never computed on the fly in production.
 */

import type {
  ExerciseReferencePayload,
  ReferenceCatalogResponse,
  PracticePreset,
  Song,
  TaskConfig,
} from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface CatalogQueryParams {
  exercise_type?: "sustained_note" | "vibrato" | "pitch_slide" | "long_note" | string;
  dataset?: "vocalset" | "mir1k" | "gtsinger" | string;
  technique?: string;
  note?: string;
  singer?: string;
  limit?: number;
  /** Include target_pitch_vector in list results (large — prefer fetching per-exercise). */
  include_vectors?: boolean;
}

/** Fetch a filtered list of reference catalog entries. */
export async function fetchReferenceCatalog(
  params: CatalogQueryParams = {},
): Promise<ReferenceCatalogResponse> {
  const url = new URL(`${API_BASE}/api/reference/catalog`);
  if (params.exercise_type) url.searchParams.set("exercise_type", params.exercise_type);
  if (params.dataset)        url.searchParams.set("dataset",        params.dataset);
  if (params.technique)      url.searchParams.set("technique",      params.technique);
  if (params.note)           url.searchParams.set("note",           params.note);
  if (params.singer)         url.searchParams.set("singer",         params.singer);
  if (params.limit != null)  url.searchParams.set("limit",          String(params.limit));
  if (params.include_vectors) url.searchParams.set("include_vectors", "true");

  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`Reference catalog fetch failed: ${res.status}`);
  return res.json() as Promise<ReferenceCatalogResponse>;
}

/**
 * Fetch a single fully-populated exercise reference including target_pitch_vector.
 * This is the call to make before the user starts a practice session.
 */
export async function fetchExerciseReference(
  assetId: string,
): Promise<ExerciseReferencePayload> {
  const encoded = encodeURIComponent(assetId);
  const url = `${API_BASE}/api/reference/exercise/${encoded}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Exercise reference fetch failed: ${res.status} for ${assetId}`);
  return res.json() as Promise<ExerciseReferencePayload>;
}

/** Fetch catalog stats (entry counts by dataset / exercise type). */
export async function fetchReferenceStats(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/reference/stats`);
  if (!res.ok) throw new Error(`Reference stats fetch failed: ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Conversion helpers: ExerciseReferencePayload → PracticePreset / Song
// ---------------------------------------------------------------------------

function durationLabel(duration_s: number): string {
  const m = Math.floor(duration_s / 60);
  const s = Math.floor(duration_s % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function exerciseCategory(
  exercise_type: string,
): PracticePreset["category"] {
  if (exercise_type === "pitch_slide") return "slide";
  if (exercise_type === "sustained_note" || exercise_type === "long_note") return "pitch";
  if (exercise_type === "vibrato") return "song";
  return "free";
}

function exerciseTitle(entry: ExerciseReferencePayload): string {
  const techLabel = entry.technique.replace(/_/g, " ");
  if (entry.note_name) return `${techLabel} — ${entry.note_name}`;
  return `${techLabel} (${entry.singer_id})`;
}

/**
 * Build a normalised 0–100 pitch height sequence from the target_pitch_vector.
 * Used by the PitchLane visualisation which does not yet consume raw Hz values.
 */
function pitchVectorToHeightSeq(vector: number[]): number[] {
  const voiced = vector.filter((v) => v > 0);
  if (voiced.length === 0) return vector.map(() => 50);
  const minHz = Math.min(...voiced);
  const maxHz = Math.max(...voiced);
  const range = maxHz - minHz || 1;
  return vector.map((hz) => (hz <= 0 ? 0 : Math.round(((hz - minHz) / range) * 80 + 10)));
}

/**
 * Convert one ExerciseReferencePayload into a PracticePreset.
 *
 * The task_config.reference block carries the pre-computed f0_hz vector so
 * that /api/audio/analyze-with-ml can populate UiFrame.target_f0_hz without
 * any server-side pitch extraction.
 */
export function payloadToPreset(entry: ExerciseReferencePayload): PracticePreset {
  const song: Song = {
    id: entry.asset_id,
    title: exerciseTitle(entry),
    artist: `${entry.singer_id} — ${entry.dataset}`,
    genre: entry.technique.replace(/_/g, " "),
    difficulty: "EASY",
    duration: durationLabel(entry.duration_s),
    bpm: 72,
    imageUrl:
      "https://images.unsplash.com/photo-1516280440614-37939bbacd81?w=400&auto=format&fit=crop&q=80",
    lyrics: ["Listen to the reference", "Then sing along", "Review your pitch"],
    referencePitchSeq: pitchVectorToHeightSeq(entry.target_pitch_vector),
    referenceAudioUrl: entry.audio_url,
    referenceStyle: `${entry.technique.replace(/_/g, " ")} — ${entry.voice_type ?? entry.singer_id}`,
    referenceType: "human_vocal",
  };

  // Embed the F0 vector into the task_config so the backend's apply_h3 evaluator
  // receives it as task_config.reference.f0_hz and populates UiFrame.target_f0_hz.
  const taskConfig: TaskConfig = {
    ...entry.task_config,
    reference: {
      ...(entry.task_config?.reference ?? {}),
      f0_hz: entry.target_pitch_vector,
      voiced: entry.voiced_vector,
      hop_s: entry.hop_s,
      asset_id: entry.asset_id,
    },
  };

  return {
    id: entry.asset_id,
    title: exerciseTitle(entry),
    description: `Practice ${entry.exercise_type.replace("_", " ")} with a real human vocal reference.`,
    category: exerciseCategory(entry.exercise_type),
    difficulty: "EASY",
    duration: durationLabel(entry.duration_s),
    source: "dataset_reference",
    song,
    taskConfig,
  };
}

/**
 * Fetch and convert a list of presets for a given exercise type.
 * Returns an empty array if the catalog is not yet built.
 */
export async function fetchPresetsForExerciseType(
  exercise_type: string,
  limit = 20,
): Promise<PracticePreset[]> {
  try {
    const catalog = await fetchReferenceCatalog({ exercise_type, limit, include_vectors: true });
    return catalog.entries.map(payloadToPreset);
  } catch {
    return [];
  }
}
