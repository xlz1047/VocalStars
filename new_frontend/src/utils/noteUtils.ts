const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/**
 * Convert a frequency in Hz to the nearest note name + octave, e.g. 261.63 → "C4".
 * Returns "--" for zero, negative, or non-finite values.
 * When `includeCents` is true, appends the deviation if ≥ 10 ¢, e.g. "C4 +32¢".
 */
export function hzToNoteName(hz: number, includeCents = false): string {
  if (!Number.isFinite(hz) || hz <= 0) return "--";
  // semitone is relative to A4=0. Shift by +9 to make it C-relative (C is 9
  // semitones below A in the same octave) so NOTE_NAMES[0]="C" aligns correctly.
  const semitone = 12 * Math.log2(hz / 440);
  const nearest = Math.round(semitone);
  const cRelative = nearest + 9;
  const noteIndex = ((cRelative % 12) + 12) % 12;
  const octave = Math.floor(cRelative / 12) + 4;
  const name = NOTE_NAMES[noteIndex] + String(octave);
  if (!includeCents) return name;
  const centsOff = Math.round((semitone - nearest) * 100);
  if (Math.abs(centsOff) < 10) return name;
  return `${name} ${centsOff >= 0 ? "+" : ""}${centsOff}¢`;
}

/** Returns the MIDI note number for a given Hz (69 = A4 = 440 Hz). */
export function hzToMidi(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440);
}

/** Returns Hz for a MIDI note number. */
export function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}
