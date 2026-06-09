import { TaskConfig } from "../types";

export interface ReferenceMelodyNote {
  note: string;
  f0_hz: number;
  duration_s: number;
}

export interface ReferenceMelody {
  id: string;
  title: string;
  description: string;
  source: "public_domain" | "traditional" | "practice_pattern";
  task_type: "reference_song" | "scale" | "interval" | "phrase_practice";
  key?: string;
  notes: ReferenceMelodyNote[];
  lyrics?: string[];
  caveat: string;
}

export const NOTE_F0: Record<string, number> = {
  G3: 196.0,
  "G#3": 207.65,
  A3: 220.0,
  "A#3": 233.08,
  B3: 246.94,
  C4: 261.63,
  "C#4": 277.18,
  D4: 293.66,
  "D#4": 311.13,
  E4: 329.63,
  F4: 349.23,
  "F#4": 369.99,
  G4: 392.0,
  "G#4": 415.3,
  A4: 440.0,
  "A#4": 466.16,
  B4: 493.88,
  C5: 523.25,
  D5: 587.33,
  E5: 659.25,
};

function note(noteName: string, duration_s = 0.65): ReferenceMelodyNote {
  const f0 = NOTE_F0[noteName];
  if (!f0) {
    throw new Error(`Missing f0 for note ${noteName}`);
  }
  return { note: noteName, f0_hz: f0, duration_s };
}

function melody(
  id: string,
  title: string,
  description: string,
  source: ReferenceMelody["source"],
  task_type: ReferenceMelody["task_type"],
  notes: ReferenceMelodyNote[],
  options: Pick<ReferenceMelody, "key" | "lyrics"> = {}
): ReferenceMelody {
  return {
    id,
    title,
    description,
    source,
    task_type,
    key: options.key,
    notes,
    lyrics: options.lyrics,
    caveat: "Reference playback is a melody guide. Current scoring is not full reference-song accuracy until melody alignment is implemented.",
  };
}

export const REFERENCE_MELODIES: Record<string, ReferenceMelody> = {
  cMajorScale: melody(
    "c-major-scale",
    "C Major Scale",
    "Stepwise do-re-mi scale practice for beginner pitch movement.",
    "practice_pattern",
    "scale",
    ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"].map((name) => note(name, 0.5)),
    { key: "C major", lyrics: ["do", "re", "mi", "fa", "sol", "la", "ti", "do"] }
  ),
  majorThird: melody(
    "major-third-c4-e4",
    "Major Third C4-E4",
    "Two-note interval drill for hearing and singing a stable pitch jump.",
    "practice_pattern",
    "interval",
    [note("C4", 0.85), note("E4", 1.0)],
    { key: "C major", lyrics: ["do", "mi"] }
  ),
  twinkleOpening: melody(
    "twinkle-opening",
    "Twinkle Twinkle Opening",
    "Common beginner melody for simple repeated notes and stepwise motion.",
    "traditional",
    "reference_song",
    [
      note("C4", 0.5),
      note("C4", 0.5),
      note("G4", 0.5),
      note("G4", 0.5),
      note("A4", 0.5),
      note("A4", 0.5),
      note("G4", 1.0),
      note("F4", 0.5),
      note("F4", 0.5),
      note("E4", 0.5),
      note("E4", 0.5),
      note("D4", 0.5),
      note("D4", 0.5),
      note("C4", 1.0),
    ],
    {
      key: "C major",
      lyrics: ["Twin", "kle", "twin", "kle", "lit", "tle", "star", "how", "I", "won", "der", "what", "you", "are"],
    }
  ),
  amazingGraceOpening: melody(
    "amazing-grace-opening",
    "Amazing Grace Opening",
    "Public-domain style phrase for phrase continuity and melody contour.",
    "public_domain",
    "reference_song",
    [
      note("G4", 0.75),
      note("C5", 1.1),
      note("E5", 0.35),
      note("C5", 0.35),
      note("E5", 1.1),
      note("D5", 0.75),
      note("C5", 1.1),
      note("A4", 0.75),
      note("G4", 1.35),
    ],
    { key: "C major", lyrics: ["A", "maz", "ing", "grace", "how", "sweet", "the", "sound"] }
  ),
  auldLangSyneOpening: melody(
    "auld-lang-syne-opening",
    "Auld Lang Syne Opening",
    "Traditional phrase for gentle contour and phrase endings.",
    "public_domain",
    "reference_song",
    [
      note("G4", 0.55),
      note("C5", 0.55),
      note("C5", 0.85),
      note("C5", 0.55),
      note("E5", 0.55),
      note("D5", 0.85),
      note("C5", 0.55),
      note("D5", 0.55),
      note("E5", 0.85),
      note("D5", 0.55),
      note("C5", 1.1),
    ],
    { key: "C major", lyrics: ["Should", "auld", "ac", "quain", "tance", "be", "for", "got", "and", "nev", "er"] }
  ),
};

export function taskConfigFromReferenceMelody(reference: ReferenceMelody): TaskConfig {
  const noteNames = reference.notes.map((item) => item.note);
  const f0 = reference.notes.map((item) => item.f0_hz);
  const durations = reference.notes.map((item) => item.duration_s);
  const baseReference = {
    type: "midi_note_sequence",
    source: "built_in_reference_melody_pack",
    id: reference.id,
    title: reference.title,
    notes: noteNames,
    f0_hz: f0,
    durations_s: durations,
    lyrics: reference.lyrics,
    caveat: reference.caveat,
  };

  if (reference.task_type === "scale") {
    return {
      task_type: "scale",
      target: { key: reference.key, notes: noteNames, f0_hz: f0 },
      reference: baseReference,
      skill_focus: ["pitch_steps", "intonation", "voiced_continuity"],
      scoring_mode: "provisional",
      strictness: "beginner",
    };
  }

  if (reference.task_type === "interval") {
    return {
      task_type: "interval",
      target: { notes: noteNames, f0_hz: f0 },
      reference: baseReference,
      skill_focus: ["interval_accuracy", "pitch_transition"],
      scoring_mode: "provisional",
      strictness: "beginner",
    };
  }

  return {
    task_type: "reference_song",
    reference: baseReference,
    skill_focus: ["melody_contour", "phrase_continuity", "voiced_continuity"],
    scoring_mode: "insufficient_reference_info",
    strictness: "beginner",
  };
}

export function referencePitchSeqFromMelody(reference: ReferenceMelody): number[] {
  const f0Values = reference.notes.map((item) => item.f0_hz);
  const min = Math.min(...f0Values);
  const max = Math.max(...f0Values);
  return f0Values.map((f0) => {
    if (Math.abs(max - min) < 1e-9) return 50;
    const normalized = (f0 - min) / (max - min);
    return Math.round(68 - normalized * 36);
  });
}

/**
 * Convert a referencePitchSeq (0–100 height scale) back to approximate Hz values,
 * then return a reference block suitable for `taskConfig.reference` so the H3
 * evaluator can compare the sung pitch against the reference contour.
 *
 * The inverse of `frequencyToHeight` (minFreq=65 Hz, maxFreq=1046 Hz, log2 scale).
 * Returns null when the sequence is empty or trivially flat.
 */
export function pitchSeqToReferenceNotes(
  seq: number[],
  totalDurationS: number
): { f0_hz: number[]; durations_s: number[]; notes: null[] } | null {
  if (!seq.length) return null;
  const minFreq = 65;
  const maxFreq = 1046;
  const f0_hz = seq.map((height) => {
    // Invert: height = 100 - clamped * 80 - 10  →  clamped = (90 - height) / 80
    const clamped = Math.max(0, Math.min(1, (90 - height) / 80));
    return Math.round(minFreq * Math.pow(maxFreq / minFreq, clamped));
  });
  const unique = new Set(f0_hz);
  if (unique.size <= 1) return null; // flat sequence — not useful for contour comparison
  const durationEach = totalDurationS / seq.length;
  return {
    f0_hz,
    durations_s: seq.map(() => parseFloat(durationEach.toFixed(3))),
    notes: seq.map(() => null),
  };
}
