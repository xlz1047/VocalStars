import { useMemo, useState } from "react";
import { ArrowLeft, ChevronRight, FileJson, ListMusic, Music2, Radio, SlidersHorizontal, Waves, Mic } from "lucide-react";
import { Song, TaskConfig, TaskType } from "../types";
import ReferenceTonePlayer from "./ReferenceTonePlayer";
import {
  NOTE_F0,
  REFERENCE_MELODIES,
  ReferenceMelody,
  taskConfigFromReferenceMelody,
} from "../utils/referenceMelodies";

interface TaskPracticeSetupProps {
  song: Song;
  initialTaskConfig?: TaskConfig | null;
  onStartPractice: (taskConfig: TaskConfig) => void;
  onBack: () => void;
}

type MvpTaskType = Extract<
  TaskType,
  "sustained_note" | "pitch_slide" | "note_match" | "free_singing" | "scale" | "interval" | "reference_song"
>;
type TargetNote = "A3" | "C4" | "E4" | "G4";
type DurationTarget = 3 | 5 | 7;
type SlideDirection = "up" | "down";
type ReferenceSource = keyof typeof REFERENCE_MELODIES | "custom";

const TARGET_NOTES: Array<{ note: TargetNote; f0_hz: number }> = [
  { note: "A3", f0_hz: 220.0 },
  { note: "C4", f0_hz: 261.63 },
  { note: "E4", f0_hz: 329.63 },
  { note: "G4", f0_hz: 392.0 },
];

const DURATION_TARGETS: DurationTarget[] = [3, 5, 7];
const DEFAULT_REFERENCE_BY_TASK: Record<"scale" | "interval" | "reference_song", ReferenceSource> = {
  scale: "cMajorScale",
  interval: "majorThird",
  reference_song: "twinkleOpening",
};

const TASKS: Array<{
  type: MvpTaskType;
  title: string;
  description: string;
  icon: typeof Radio;
}> = [
  {
    type: "sustained_note",
    title: "Sustained Note",
    description: "Hold one steady note for a short target duration.",
    icon: Radio,
  },
  {
    type: "pitch_slide",
    title: "Pitch Slide",
    description: "Glide smoothly in one direction without treating it as a song.",
    icon: Waves,
  },
  {
    type: "note_match",
    title: "Note Match",
    description: "Aim for a single target pitch and check how close you land.",
    icon: Music2,
  },
  {
    type: "scale",
    title: "Scale",
    description: "Practice stepwise note movement against a note-sequence reference.",
    icon: ListMusic,
  },
  {
    type: "interval",
    title: "Interval",
    description: "Sing a two-note jump and review the contour without full-song scoring.",
    icon: Music2,
  },
  {
    type: "reference_song",
    title: "Reference Melody",
    description: "Compare your sung f0 contour to a short public-domain or user-provided phrase.",
    icon: FileJson,
  },
  {
    type: "free_singing",
    title: "Free Singing",
    description: "Sing a short phrase with general feedback, not reference-song scoring.",
    icon: SlidersHorizontal,
  },
];

function noteInfo(note: TargetNote) {
  return TARGET_NOTES.find((item) => item.note === note) || TARGET_NOTES[1];
}

function normalizeNoteName(value: string) {
  return value.trim().replace("♯", "#").replace("♭", "b").toUpperCase().replace("B", "B");
}

function f0ForNoteName(noteName: string): number | null {
  const normalized = normalizeNoteName(noteName);
  return NOTE_F0[normalized] || null;
}

function referenceFromCustomJson(text: string): ReferenceMelody | null {
  if (!text.trim()) return null;
  const parsed = JSON.parse(text);
  const rawNotes = Array.isArray(parsed.notes) ? parsed.notes : [];
  const rawF0 = Array.isArray(parsed.f0_hz) ? parsed.f0_hz : [];
  const rawDurations = Array.isArray(parsed.durations_s) ? parsed.durations_s : [];
  const objectNotes = rawNotes.filter((item: unknown) => typeof item === "object" && item !== null);
  const noteNames = objectNotes.length
    ? objectNotes.map((item: any) => String(item.note || item.name || ""))
    : rawNotes.map((item: unknown) => String(item));
  const f0Values = objectNotes.length
    ? objectNotes.map((item: any) => Number(item.f0_hz || item.frequency_hz || f0ForNoteName(String(item.note || ""))))
    : noteNames.map((name: string, index: number) => Number(rawF0[index] || f0ForNoteName(name)));
  const durations = objectNotes.length
    ? objectNotes.map((item: any) => Number(item.duration_s || item.duration || 0.6))
    : noteNames.map((_name: string, index: number) => Number(rawDurations[index] || 0.6));

  const notes = noteNames
    .map((name: string, index: number) => ({
      note: name.trim(),
      f0_hz: f0Values[index],
      duration_s: durations[index],
    }))
    .filter((item: { note: string; f0_hz: number; duration_s: number }) => (
      item.note && Number.isFinite(item.f0_hz) && item.f0_hz > 0 && Number.isFinite(item.duration_s) && item.duration_s > 0
    ));

  if (notes.length < 2) {
    throw new Error("Custom reference needs at least two notes with f0_hz or known note names.");
  }

  return {
    id: "custom-reference-melody",
    title: typeof parsed.title === "string" ? parsed.title : "Custom Reference Melody",
    description: typeof parsed.description === "string"
      ? parsed.description
      : "User-provided note sequence for provisional singing contour practice.",
    source: "practice_pattern",
    task_type: "reference_song",
    key: typeof parsed.key === "string" ? parsed.key : undefined,
    notes,
    lyrics: Array.isArray(parsed.lyrics) ? parsed.lyrics.map((item: unknown) => String(item)) : undefined,
    caveat: "User-provided reference melody. Current scoring is provisional f0-contour feedback only.",
  };
}

function buildTaskConfig(
  taskType: MvpTaskType,
  targetNote: TargetNote,
  durationTarget: DurationTarget,
  direction: SlideDirection,
  referenceMelody: ReferenceMelody | null
): TaskConfig {
  const target = noteInfo(targetNote);

  if (taskType === "sustained_note") {
    return {
      task_type: "sustained_note",
      target: { note: target.note, f0_hz: target.f0_hz },
      reference: null,
      skill_focus: ["pitch_stability", "voiced_continuity"],
      scoring_mode: "diagnostic",
      strictness: "beginner",
      expected_duration: durationTarget,
    };
  }

  if (taskType === "pitch_slide") {
    return {
      task_type: "pitch_slide",
      target: null,
      reference: null,
      skill_focus: ["slide_smoothness", "continuity"],
      scoring_mode: "diagnostic",
      strictness: "beginner",
      expected_duration: durationTarget,
      expected_direction: direction,
    };
  }

  if (taskType === "note_match") {
    return {
      task_type: "note_match",
      target: { note: target.note, f0_hz: target.f0_hz },
      reference: null,
      skill_focus: ["pitch_matching"],
      scoring_mode: "diagnostic",
      strictness: "beginner",
    };
  }

  if ((taskType === "scale" || taskType === "interval" || taskType === "reference_song") && referenceMelody) {
    const config = taskConfigFromReferenceMelody({
      ...referenceMelody,
      task_type: taskType === "reference_song" ? "reference_song" : taskType,
    });
    return {
      ...config,
      scoring_mode: "provisional",
      strictness: "beginner",
    };
  }

  return {
    task_type: "free_singing",
    target: null,
    reference: null,
    skill_focus: ["general_pitch", "phrase_continuity"],
    scoring_mode: "no_reference",
    strictness: "beginner",
  };
}

function formatTaskType(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function TaskPracticeSetup({
  song,
  initialTaskConfig,
  onStartPractice,
  onBack,
}: TaskPracticeSetupProps) {
  const initialType = TASKS.some((task) => task.type === initialTaskConfig?.task_type)
    ? (initialTaskConfig?.task_type as MvpTaskType)
    : "sustained_note";
  const [taskType, setTaskType] = useState<MvpTaskType>(initialType);
  const [targetNote, setTargetNote] = useState<TargetNote>((initialTaskConfig?.target?.note as TargetNote) || "C4");
  const [durationTarget, setDurationTarget] = useState<DurationTarget>(
    (typeof initialTaskConfig?.expected_duration === "number" ? initialTaskConfig.expected_duration : 5) as DurationTarget
  );
  const [direction, setDirection] = useState<SlideDirection>(
    initialTaskConfig?.expected_direction === "down" ? "down" : "up"
  );
  const [referenceSource, setReferenceSource] = useState<ReferenceSource>(() => {
    if (initialTaskConfig?.task_type === "scale") return "cMajorScale";
    if (initialTaskConfig?.task_type === "interval") return "majorThird";
    if (initialTaskConfig?.task_type === "reference_song") return "twinkleOpening";
    return "twinkleOpening";
  });
  const [customReferenceText, setCustomReferenceText] = useState(
    '{\n  "title": "Custom Phrase",\n  "notes": ["C4", "C4", "G4", "G4", "A4", "A4", "G4"],\n  "durations_s": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]\n}'
  );

  const { selectedReferenceMelody, customReferenceError } = useMemo(() => {
    const source = referenceSource === "custom"
      ? "custom"
      : (REFERENCE_MELODIES[referenceSource] ? referenceSource : DEFAULT_REFERENCE_BY_TASK.reference_song);
    if (source !== "custom") {
      return { selectedReferenceMelody: REFERENCE_MELODIES[source], customReferenceError: null };
    }
    try {
      const parsed = referenceFromCustomJson(customReferenceText);
      return { selectedReferenceMelody: parsed, customReferenceError: null };
    } catch (error) {
      return {
        selectedReferenceMelody: null,
        customReferenceError: error instanceof Error ? error.message : "Custom reference JSON could not be parsed.",
      };
    }
  }, [customReferenceText, referenceSource]);

  const taskConfig = useMemo(
    () => buildTaskConfig(taskType, targetNote, durationTarget, direction, selectedReferenceMelody),
    [taskType, targetNote, durationTarget, direction, selectedReferenceMelody]
  );

  // ── Human reference catalog mode ─────────────────────────────────────────
  // When the task config was built by the catalog API (dense F0 contour, hop_s
  // present), skip the manual setup UI and show a read-only review + player.
  const isHumanReference =
    initialTaskConfig?.reference?.type === "human_vocal_reference" &&
    Array.isArray(initialTaskConfig?.reference?.f0_hz) &&
    typeof initialTaskConfig?.reference?.hop_s === "number";

  if (isHumanReference && initialTaskConfig) {
    const ref = initialTaskConfig.reference as Record<string, unknown>;
    const technique = typeof ref.technique === "string" ? ref.technique.replace(/_/g, " ") : "Human Reference";
    const singer = typeof ref.singer === "string" ? ref.singer : "";
    const durationFrames = Array.isArray(ref.f0_hz) ? (ref.f0_hz as unknown[]).length : 0;
    const durationS = durationFrames * (typeof ref.hop_s === "number" ? ref.hop_s : 0.01);
    const skillFocus = Array.isArray(initialTaskConfig.skill_focus) ? initialTaskConfig.skill_focus : [];

    return (
      <div className="space-y-8 animate-fade-in max-w-3xl">
        <section className="bg-surface-container/20 p-6 rounded-2xl border border-white/5">
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-xs font-bold text-on-surface-variant hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <span className="inline-block text-[10px] uppercase tracking-widest font-bold text-tertiary mb-2">
            Human Vocal Reference
          </span>
          <h1 className="font-display font-extrabold text-3xl text-white">
            {technique}
          </h1>
          {singer && (
            <p className="text-on-surface-variant text-sm mt-1">{singer} · {Math.round(durationS)}s</p>
          )}
          {skillFocus.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {skillFocus.map((s) => (
                <span key={s} className="text-[10px] uppercase tracking-wider px-2.5 py-1 rounded-full bg-primary/10 text-primary border border-primary/20">
                  {String(s).replace(/_/g, " ")}
                </span>
              ))}
            </div>
          )}
        </section>

        {song.referenceAudioUrl && (
          <section className="bg-surface-container/20 p-6 rounded-2xl border border-white/5 space-y-3">
            <p className="text-xs font-bold uppercase tracking-widest text-on-surface-variant">
              Listen to the reference
            </p>
            <ReferenceTonePlayer
              taskConfig={initialTaskConfig}
              referenceAudioUrl={song.referenceAudioUrl}
              referenceStyle={song.referenceStyle}
              referenceType="human_vocal"
            />
          </section>
        )}

        <section className="bg-surface-container/20 p-5 rounded-2xl border border-white/5 text-sm text-on-surface-variant space-y-1">
          <p>· Pre-computed pitch contour: <span className="text-white">{durationFrames} frames @ 10 ms hop</span></p>
          <p>· Alignment: voiced-span linear time warp</p>
          <p>· Scoring: reference_song diagnostic</p>
        </section>

        <button
          onClick={() => onStartPractice(initialTaskConfig)}
          className="w-full inline-flex items-center justify-center gap-3 px-8 py-4 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary font-bold text-sm hover:scale-[1.02] hover:brightness-110 active:scale-95 transition-all"
        >
          <Mic className="w-5 h-5" />
          Start Recording
        </button>
      </div>
    );
  }

  const selectedTask = TASKS.find((task) => task.type === taskType) || TASKS[0];
  const SelectedIcon = selectedTask.icon;
  const showTargetNote = taskType === "sustained_note" || taskType === "note_match";
  const showDuration = taskType === "sustained_note" || taskType === "pitch_slide";
  const showReferenceSelector = taskType === "scale" || taskType === "interval" || taskType === "reference_song";
  const canStart = !showReferenceSelector || Boolean(selectedReferenceMelody);

  const handleTaskTypeChange = (nextType: MvpTaskType) => {
    setTaskType(nextType);
    if (nextType === "scale" || nextType === "interval" || nextType === "reference_song") {
      setReferenceSource((current) => current === "custom" ? current : DEFAULT_REFERENCE_BY_TASK[nextType]);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in max-w-6xl">
      <section className="flex flex-col md:flex-row md:items-center justify-between gap-5 bg-surface-container/20 p-6 rounded-2xl border border-white/5">
        <div>
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-xs font-bold text-on-surface-variant hover:text-white transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            Back
          </button>
          <p className="text-[10px] uppercase tracking-widest font-bold text-secondary">Guided Practice Setup</p>
          <h1 className="font-display font-extrabold text-3xl md:text-4xl text-white mt-2">
            Choose Your Exercise
          </h1>
          <p className="text-sm text-on-surface-variant mt-2">
            The selected exercise is sent with your recording so analysis can stay task-aware.
          </p>
        </div>
        <div className="rounded-2xl bg-surface-container-high/70 border border-white/5 p-4 min-w-[220px]">
          <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Session Context</p>
          <p className="text-sm font-bold text-white mt-1">{song.title}</p>
          <p className="text-xs text-on-surface-variant">{song.artist}</p>
        </div>
      </section>

      <div className="grid lg:grid-cols-12 gap-8">
        <section className="lg:col-span-7 grid sm:grid-cols-2 gap-4">
          {TASKS.map((task) => {
            const Icon = task.icon;
            const selected = task.type === taskType;
            return (
              <button
                key={task.type}
                onClick={() => handleTaskTypeChange(task.type)}
                className={`text-left rounded-2xl border p-5 transition-all ${
                  selected
                    ? "bg-primary/10 border-primary/50 shadow-[0_0_20px_rgba(255,177,192,0.12)]"
                    : "bg-surface-container/30 border-white/5 hover:border-white/15 hover:bg-white/[0.04]"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${
                    selected ? "bg-primary/15 text-primary" : "bg-white/5 text-on-surface-variant"
                  }`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-lg text-white">{task.title}</h3>
                    <p className="text-xs text-on-surface-variant leading-relaxed mt-1">{task.description}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </section>

        <section className="lg:col-span-5 glass-card rounded-[24px] p-6 md:p-8 space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-secondary/10 text-secondary flex items-center justify-center">
              <SelectedIcon className="w-6 h-6" />
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">Selected Task</p>
              <h2 className="font-display font-extrabold text-2xl text-white">{selectedTask.title}</h2>
            </div>
          </div>

          {showTargetNote && (
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
                Target Note
              </label>
              <div className="grid grid-cols-4 gap-2">
                {TARGET_NOTES.map((item) => (
                  <button
                    key={item.note}
                    onClick={() => setTargetNote(item.note)}
                    className={`py-3 rounded-xl border text-sm font-bold transition-all ${
                      targetNote === item.note
                        ? "border-tertiary bg-tertiary/15 text-tertiary"
                        : "border-white/5 bg-white/5 text-on-surface-variant hover:text-white"
                    }`}
                  >
                    {item.note}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-on-surface-variant">
                {targetNote} is {noteInfo(targetNote).f0_hz.toFixed(2)} Hz.
              </p>
            </div>
          )}

          {showDuration && (
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
                Duration Target
              </label>
              <div className="grid grid-cols-3 gap-2">
                {DURATION_TARGETS.map((seconds) => (
                  <button
                    key={seconds}
                    onClick={() => setDurationTarget(seconds)}
                    className={`py-3 rounded-xl border text-sm font-bold transition-all ${
                      durationTarget === seconds
                        ? "border-primary bg-primary/15 text-primary"
                        : "border-white/5 bg-white/5 text-on-surface-variant hover:text-white"
                    }`}
                  >
                    {seconds}s
                  </button>
                ))}
              </div>
            </div>
          )}

          {taskType === "pitch_slide" && (
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
                Slide Direction
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(["up", "down"] as SlideDirection[]).map((item) => (
                  <button
                    key={item}
                    onClick={() => setDirection(item)}
                    className={`py-3 rounded-xl border text-sm font-bold capitalize transition-all ${
                      direction === item
                        ? "border-tertiary bg-tertiary/15 text-tertiary"
                        : "border-white/5 bg-white/5 text-on-surface-variant hover:text-white"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}

          {showReferenceSelector && (
            <div className="space-y-4 rounded-xl bg-surface-container-high/50 border border-white/5 p-4">
              <div>
                <label className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
                  Reference Melody
                </label>
                <select
                  value={referenceSource}
                  onChange={(event) => setReferenceSource(event.target.value as ReferenceSource)}
                  className="mt-2 w-full rounded-xl bg-background/70 border border-white/10 px-3 py-3 text-sm text-white outline-none focus:border-primary"
                >
                  {Object.entries(REFERENCE_MELODIES).map(([key, reference]) => (
                    <option key={key} value={key}>
                      {reference.title}
                    </option>
                  ))}
                  <option value="custom">Custom note-sequence JSON</option>
                </select>
              </div>

              {referenceSource === "custom" && (
                <div className="space-y-2">
                  <label className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant">
                    Custom JSON
                  </label>
                  <textarea
                    value={customReferenceText}
                    onChange={(event) => setCustomReferenceText(event.target.value)}
                    rows={7}
                    spellCheck={false}
                    className="w-full rounded-xl bg-background/70 border border-white/10 px-3 py-3 text-xs font-mono text-on-surface outline-none focus:border-primary"
                  />
                  {customReferenceError ? (
                    <p className="text-[11px] text-error leading-relaxed">{customReferenceError}</p>
                  ) : (
                    <p className="text-[11px] text-on-surface-variant leading-relaxed">
                      Use note names plus durations, or provide matching `f0_hz` and `durations_s` arrays.
                    </p>
                  )}
                </div>
              )}

              {selectedReferenceMelody && (
                <div className="rounded-xl bg-secondary/10 border border-secondary/20 p-3">
                  <p className="text-xs font-bold text-white">{selectedReferenceMelody.title}</p>
                  <p className="text-[11px] text-on-surface-variant leading-relaxed mt-1">
                    {selectedReferenceMelody.notes.length} notes. Provisional contour feedback only; rhythm and full-song accuracy remain blocked.
                  </p>
                </div>
              )}
            </div>
          )}

          {taskType === "free_singing" && (
            <div className="rounded-xl bg-secondary/10 border border-secondary/20 p-4">
              <p className="text-xs text-on-surface-variant leading-relaxed">
                Free singing gives general feedback only. It does not claim reference-melody accuracy.
              </p>
            </div>
          )}

          <ReferenceTonePlayer
            taskConfig={taskConfig}
            referenceAudioUrl={song.referenceAudioUrl}
            referenceStyle={song.referenceStyle}
          />

          <div className="rounded-xl bg-surface-container-high/60 border border-white/5 p-4">
            <p className="text-[10px] uppercase tracking-wider font-bold text-on-surface-variant mb-2">
              Task Config Preview
            </p>
            <p className="text-xs text-on-surface-variant">
              Type: <span className="font-bold text-white">{formatTaskType(taskConfig.task_type)}</span>
            </p>
            {taskConfig.target?.note && (
              <p className="text-xs text-on-surface-variant mt-1">
                Target: <span className="font-bold text-white">{taskConfig.target.note}</span>
              </p>
            )}
            {taskConfig.expected_duration && (
              <p className="text-xs text-on-surface-variant mt-1">
                Duration: <span className="font-bold text-white">{taskConfig.expected_duration}s</span>
              </p>
            )}
            {taskConfig.expected_direction && (
              <p className="text-xs text-on-surface-variant mt-1">
                Direction: <span className="font-bold text-white capitalize">{taskConfig.expected_direction}</span>
              </p>
            )}
            {taskConfig.reference?.title && (
              <p className="text-xs text-on-surface-variant mt-1">
                Reference: <span className="font-bold text-white">{taskConfig.reference.title}</span>
              </p>
            )}
          </div>

          <button
            onClick={() => onStartPractice(taskConfig)}
            disabled={!canStart}
            className="w-full px-6 py-4 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary text-sm font-bold hover:brightness-110 active:scale-95 transition-all flex items-center justify-center gap-2 glow-pink disabled:opacity-45 disabled:cursor-not-allowed"
          >
            Start Recording
            <ChevronRight className="w-4 h-4 text-on-primary" />
          </button>
        </section>
      </div>
    </div>
  );
}
