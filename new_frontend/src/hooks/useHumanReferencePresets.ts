/**
 * Fetches human vocal reference presets from /api/reference/catalog for the
 * three target exercise types. Results are cached in module state so
 * re-mounts do not re-fetch.
 *
 * Falls back gracefully to an empty list if the catalog is not yet built
 * (e.g., during development before preprocess_human_references.py has run).
 */

import { useState, useEffect } from "react";
import type { PracticePreset } from "../types";
import { fetchReferenceCatalog, payloadToPreset } from "../utils/referenceApi";
import { registerPresets } from "../utils/improvementPath";
import { registerLearningPresets } from "../utils/learningPath";

export type HumanExerciseType = "sustained_note" | "vibrato" | "pitch_slide";

interface FetchState {
  presets: PracticePreset[];
  loading: boolean;
  error: string | null;
  catalogSize: number;
}

// Module-level cache keyed by exercise_type to avoid redundant API calls
const _cache = new Map<string, PracticePreset[]>();

async function fetchForType(
  exercise_type: HumanExerciseType,
  limit: number,
): Promise<PracticePreset[]> {
  if (_cache.has(exercise_type)) return _cache.get(exercise_type)!;
  const resp = await fetchReferenceCatalog({
    exercise_type,
    limit,
    include_vectors: true,
  });
  if (!resp.entries?.length) return [];
  const presets = resp.entries.map(payloadToPreset);
  _cache.set(exercise_type, presets);
  return presets;
}

/**
 * Returns presets for one or more human exercise types merged into a
 * single sorted list. Loading is per-type and results appear incrementally.
 */
export function useHumanReferencePresets(
  types: HumanExerciseType[] = ["sustained_note", "vibrato", "pitch_slide"],
  limitPerType = 10,
): FetchState {
  const [presets, setPresets] = useState<PracticePreset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [catalogSize, setCatalogSize] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.allSettled(
      types.map((t) => fetchForType(t, limitPerType)),
    ).then((results) => {
      if (cancelled) return;
      const merged: PracticePreset[] = [];
      let anyError = false;
      for (const result of results) {
        if (result.status === "fulfilled") {
          merged.push(...result.value);
        } else {
          anyError = true;
        }
      }
      setPresets(merged);
      setCatalogSize(merged.length);
      if (merged.length) {
        registerPresets(merged);
        registerLearningPresets(merged);
      }
      if (anyError && !merged.length) {
        setError("Reference catalog unavailable — run preprocessing scripts.");
      }
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [types.join(","), limitPerType]);

  return { presets, loading, error, catalogSize };
}

/**
 * Fetch a single exercise reference preset by asset_id.
 * Useful for deep-linking to a specific exercise.
 */
export function useExerciseReference(assetId: string | null): {
  preset: PracticePreset | null;
  loading: boolean;
} {
  const [preset, setPreset] = useState<PracticePreset | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!assetId) return;
    let cancelled = false;
    setLoading(true);
    import("../utils/referenceApi").then(({ fetchExerciseReference, payloadToPreset }) =>
      fetchExerciseReference(assetId).then((payload) => {
        if (!cancelled) setPreset(payloadToPreset(payload));
      }),
    ).catch(() => {
      if (!cancelled) setPreset(null);
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [assetId]);

  return { preset, loading };
}
