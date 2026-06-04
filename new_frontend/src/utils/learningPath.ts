import { useCallback, useEffect, useState } from "react";
import { PRACTICE_PRESETS } from "./musicDb";
import { PracticePreset } from "../types";

const STORAGE_KEY = "vs_learning_path_completed";

export interface LearningStep {
  presetId: string;
  milestone: string;
  preset: PracticePreset;
}

const STEP_DEFINITIONS: { presetId: string; milestone: string }[] = [
  { presetId: "sustained-c4",              milestone: "Steady pitch" },
  { presetId: "note-match-e4",             milestone: "Note matching" },
  { presetId: "rising-slide-220-440",      milestone: "Vocal slides" },
  { presetId: "c-major-scale-fragment",    milestone: "Scale movement" },
  { presetId: "gts-breathy-alto2-easy-on-me",  milestone: "Breathy tone" },
  { presetId: "gts-vibrato-alto1-memory",  milestone: "Vibrato" },
];

/** Static learning steps resolved from PRACTICE_PRESETS at module load time. */
export const LEARNING_STEPS: LearningStep[] = STEP_DEFINITIONS.flatMap(({ presetId, milestone }) => {
  const preset = PRACTICE_PRESETS.find((p) => p.id === presetId);
  return preset ? [{ presetId, milestone, preset }] : [];
});

/** Runtime-extended steps populated by the human reference catalog hook.
 *  Components should prefer this over LEARNING_STEPS when available. */
let _dynamicSteps: LearningStep[] = [...LEARNING_STEPS];

/** Replace the dynamic steps list with static + catalog-sourced steps.
 *  Called by useHumanReferencePresets after a successful fetch. */
export function registerLearningPresets(presets: PracticePreset[]): void {
  const HUMAN_MILESTONES: Array<{ category: PracticePreset["category"]; milestone: string }> = [
    { category: "pitch",  milestone: "Sustained tone" },
    { category: "slide",  milestone: "Pitch slide" },
    { category: "pitch",  milestone: "Vibrato" },
  ];
  const extra: LearningStep[] = [];
  for (const { category, milestone } of HUMAN_MILESTONES) {
    const preset = presets.find((p) => p.category === category && p.source === "dataset_reference");
    if (preset && !_dynamicSteps.some((s) => s.presetId === preset.id)) {
      extra.push({ presetId: preset.id, milestone, preset });
    }
  }
  _dynamicSteps = [...LEARNING_STEPS, ...extra];
}

export function getDynamicLearningSteps(): LearningStep[] {
  return _dynamicSteps;
}

function readCompleted(): Set<string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
  } catch {
    return new Set();
  }
}

function writeCompleted(ids: Set<string>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...ids]));
  } catch {
    // storage unavailable — silently ignore
  }
}

export function useLearningPath() {
  const [completed, setCompleted] = useState<Set<string>>(readCompleted);

  useEffect(() => {
    writeCompleted(completed);
  }, [completed]);

  const markCompleted = useCallback((presetId: string) => {
    setCompleted((prev) => new Set([...prev, presetId]));
  }, []);

  const resetProgress = useCallback(() => {
    setCompleted(new Set());
  }, []);

  const nextStep = LEARNING_STEPS.find((s) => !completed.has(s.presetId)) ?? null;

  return { completed, markCompleted, resetProgress, nextStep };
}
