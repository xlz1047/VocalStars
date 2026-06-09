import { TaskConfig } from "../types";

export type ReferenceToneKind = "single_tone" | "pitch_slide" | "note_sequence";

export interface ReferenceToneNote {
  f0Hz: number;
  durationSeconds: number;
  label?: string;
}

export interface ReferenceTonePlan {
  kind: ReferenceToneKind;
  label: string;
  startF0Hz: number;
  endF0Hz?: number;
  durationSeconds: number;
  countIn: boolean;
  notes?: ReferenceToneNote[];
}

export interface ReferenceTonePlayback {
  stop: () => void;
  finished: Promise<void>;
}

const DEFAULT_TARGET_F0 = 261.63;
const DEFAULT_SLIDE_LOW_F0 = 220;
const DEFAULT_SLIDE_HIGH_F0 = 440;
const MAX_GAIN = 0.18;

function numericDuration(value: TaskConfig["expected_duration"], fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0.5, Math.min(value, 10));
  }
  return fallback;
}

function targetF0(taskConfig?: TaskConfig | null): number {
  const f0 = taskConfig?.target?.f0_hz;
  return typeof f0 === "number" && Number.isFinite(f0) && f0 > 0 ? f0 : DEFAULT_TARGET_F0;
}

function targetLabel(taskConfig?: TaskConfig | null): string {
  const note = taskConfig?.target?.note;
  return typeof note === "string" && note ? note : `${targetF0(taskConfig).toFixed(1)} Hz`;
}

function numericArray(value: unknown): number[] {
  return Array.isArray(value)
    ? value.filter((item): item is number => typeof item === "number" && Number.isFinite(item) && item > 0)
    : [];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
}

function noteSequenceFromTask(taskConfig?: TaskConfig | null): ReferenceToneNote[] {
  const target = taskConfig?.target || {};
  const reference = taskConfig?.reference || {};
  const f0s = numericArray((reference as any).f0_hz).length
    ? numericArray((reference as any).f0_hz)
    : numericArray((target as any).f0_hz);
  const labels = stringArray((reference as any).notes).length
    ? stringArray((reference as any).notes)
    : stringArray((target as any).notes);
  const durations = numericArray((reference as any).durations_s);
  return f0s.map((f0, index) => ({
    f0Hz: f0,
    durationSeconds: durations[index] || 0.7,
    label: labels[index],
  }));
}

export function getReferenceTonePlan(taskConfig?: TaskConfig | null): ReferenceTonePlan | null {
  const taskType = taskConfig?.task_type;

  if (taskType === "sustained_note") {
    return {
      kind: "single_tone",
      label: `Target ${targetLabel(taskConfig)}`,
      startF0Hz: targetF0(taskConfig),
      durationSeconds: numericDuration(taskConfig.expected_duration, 3),
      countIn: true,
    };
  }

  if (taskType === "note_match") {
    return {
      kind: "single_tone",
      label: `Match ${targetLabel(taskConfig)}`,
      startF0Hz: targetF0(taskConfig),
      durationSeconds: 1.25,
      countIn: true,
    };
  }

  if (taskType === "pitch_slide") {
    const direction = taskConfig?.expected_direction === "down" ? "down" : "up";
    return {
      kind: "pitch_slide",
      label: direction === "down" ? "Falling glide" : "Rising glide",
      startF0Hz: direction === "down" ? DEFAULT_SLIDE_HIGH_F0 : DEFAULT_SLIDE_LOW_F0,
      endF0Hz: direction === "down" ? DEFAULT_SLIDE_LOW_F0 : DEFAULT_SLIDE_HIGH_F0,
      durationSeconds: numericDuration(taskConfig.expected_duration, 3),
      countIn: true,
    };
  }

  const sequence = noteSequenceFromTask(taskConfig);
  if ((taskType === "scale" || taskType === "interval" || taskType === "reference_song" || sequence.length > 1) && sequence.length > 0) {
    return {
      kind: "note_sequence",
      label: taskConfig?.reference?.title || taskConfig?.target?.key || "Reference note sequence",
      startF0Hz: sequence[0].f0Hz,
      durationSeconds: sequence.reduce((sum, note) => sum + note.durationSeconds, 0),
      countIn: true,
      notes: sequence,
    };
  }

  return null;
}

export function playReferenceTone(plan: ReferenceTonePlan, volume: number): ReferenceTonePlayback {
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
  const context = new AudioContextClass();
  const masterGain = context.createGain();
  const safeVolume = Math.max(0, Math.min(volume, 1));
  const outputGain = safeVolume * MAX_GAIN;
  const oscillators: OscillatorNode[] = [];
  let stopped = false;

  masterGain.gain.setValueAtTime(0, context.currentTime);
  masterGain.connect(context.destination);

  const makeOscillator = (frequency: number, startTime: number, stopTime: number, gain = outputGain) => {
    const oscillator = context.createOscillator();
    const toneGain = context.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(frequency, startTime);
    toneGain.gain.setValueAtTime(0, startTime);
    toneGain.gain.linearRampToValueAtTime(gain, startTime + 0.02);
    toneGain.gain.setValueAtTime(gain, Math.max(startTime + 0.02, stopTime - 0.04));
    toneGain.gain.linearRampToValueAtTime(0, stopTime);
    oscillator.connect(toneGain);
    toneGain.connect(masterGain);
    oscillator.start(startTime);
    oscillator.stop(stopTime + 0.03);
    oscillators.push(oscillator);
    return oscillator;
  };

  const startAt = context.currentTime + 0.03;
  let toneStart = startAt;

  masterGain.gain.setValueAtTime(1, context.currentTime);
  void context.resume();

  if (plan.countIn) {
    const beepGain = outputGain * 0.65;
    makeOscillator(880, startAt, startAt + 0.09, beepGain);
    makeOscillator(880, startAt + 0.36, startAt + 0.45, beepGain);
    toneStart = startAt + 0.72;
  }

  if (plan.kind === "note_sequence" && plan.notes?.length) {
    let cursor = toneStart;
    plan.notes.forEach((note) => {
      const noteDuration = Math.max(0.12, note.durationSeconds);
      const noteEnd = cursor + noteDuration;
      makeOscillator(note.f0Hz, cursor, noteEnd);
      cursor = noteEnd + 0.04;
    });
  } else {
    const toneEnd = toneStart + plan.durationSeconds;
    const oscillator = makeOscillator(plan.startF0Hz, toneStart, toneEnd);

    if (plan.kind === "pitch_slide" && plan.endF0Hz) {
      oscillator.frequency.setValueAtTime(plan.startF0Hz, toneStart);
      oscillator.frequency.exponentialRampToValueAtTime(plan.endF0Hz, toneEnd);
    }
  }

  const cleanup = async () => {
    try {
      masterGain.disconnect();
      oscillators.forEach((node) => node.disconnect());
    } catch {
      // Nodes may already be disconnected after a scheduled stop.
    }
    if (context.state !== "closed") {
      await context.close();
    }
  };

  const finished = new Promise<void>((resolve) => {
    window.setTimeout(() => {
      if (!stopped) {
        stopped = true;
        void cleanup().finally(resolve);
      } else {
        resolve();
      }
    }, (plan.durationSeconds + (plan.countIn ? 0.85 : 0.1)) * 1000);
  });

  return {
    stop: () => {
      if (stopped) return;
      stopped = true;
      const now = context.currentTime;
      oscillators.forEach((node) => {
        try {
          node.stop(now + 0.02);
        } catch {
          // Ignore if the oscillator already reached its scheduled stop.
        }
      });
      void cleanup();
    },
    finished,
  };
}
