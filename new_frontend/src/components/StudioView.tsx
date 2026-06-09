import { useState, useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Play,
  Square,
  Volume2,
  Mic,
  MicOff,
  RefreshCw,
  SkipForward,
  Music,
  VolumeX,
  FileText,
  FileAudio,
  Loader
} from "lucide-react";
import { LiveFrame, PracticeSessionState, RecordedAttempt, Song, PerformanceResult, TaskConfig } from "../types";
import { detectPitch, frequencyToHeight } from "../utils/pitchDetector";
import { hzToNoteName } from "../utils/noteUtils";
import { useLiveAnalysis } from "../utils/useLiveAnalysis";
import { useGtsingerCatalog, findSongPhrases } from "../utils/useGtsingerCatalog";
import LivePitchCanvas, { TargetContourPoint } from "./LivePitchCanvas";
import ReferenceTonePlayer from "./ReferenceTonePlayer";
import {
  startAudioRecording,
  stopAudioRecording,
  pauseAudioRecording,
  resumeAudioRecording,
  getBrowserRecordingSupport,
  getRecordingDiagnostics,
  RecordingDiagnostics,
  analyzeAudioWithML,
  createAnalysisUnavailablePerformanceResult,
  mlAnalysisToPerformanceResult
} from "../utils/audioAnalysis";

interface StudioViewProps {
  song: Song;
  taskConfig?: TaskConfig | null;
  onSessionComplete: (results: PerformanceResult, attempt?: Omit<RecordedAttempt, "audioUrl">) => void;
  onExit: () => void;
}

export default function StudioView({
  song,
  taskConfig,
  onSessionComplete,
  onExit
}: StudioViewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [hasActiveRecording, setHasActiveRecording] = useState(false);
  const [hasMicPermission, setHasMicPermission] = useState<boolean | null>(null);
  const [micActive, setMicActive] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [frozenFrames, setFrozenFrames] = useState<LiveFrame[]>([]);
  const [sessionState, setSessionState] = useState<PracticeSessionState>("ready");
  const [devAudioFile, setDevAudioFile] = useState<File | null>(null);
  const [devSamples, setDevSamples] = useState<{ label: string; path: string }[]>([]);
  const [selectedDevPath, setSelectedDevPath] = useState<string>("");
  const [audioInputDevices, setAudioInputDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedAudioDeviceId, setSelectedAudioDeviceId] = useState<string>("");
  const [recordingDiagnostics, setRecordingDiagnostics] = useState<RecordingDiagnostics>({
    state: "inactive",
    mimeType: null,
    chunkCount: 0,
    bytesCollected: 0,
    durationSeconds: 0,
  });

  // Audio state
  const [volume, setVolume] = useState(80);
  const [isMuted, setIsMuted] = useState(false);
  const [playSpeed, setPlaySpeed] = useState<1 | 0.5 | 1.5>(1);
  const [showLyricsText, setShowLyricsText] = useState(true);
  const [progressSeconds, setProgressSeconds] = useState(0);

  // Pitch and analysis state
  const [activePitch, setActivePitch] = useState<number>(50);
  const [activeFrequencyHz, setActiveFrequencyHz] = useState<number | null>(null);
  const [inputLevel, setInputLevel] = useState(0);
  const [isVoicedNow, setIsVoicedNow] = useState(false);
  const [capturedPitches, setCapturedPitches] = useState<{x: number, y: number}[]>([]);
  const [currentLineIndex, setCurrentLineIndex] = useState(1);

  // Score parameters computed on-the-fly or simulated based on mic output
  const [scoreList, setScoreList] = useState<number[]>([]);
  const [averageAccuracy, setAverageAccuracy] = useState(98);

  // Audio Context Ref
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const [micStream, setMicStream] = useState<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioRecordingRef = useRef<boolean>(false);
  const devFileInputRef = useRef<HTMLInputElement | null>(null);
  const recordingSupport = getBrowserRecordingSupport();

  // Live analysis (WebSocket + neural model streaming)
  const liveAnalysis = useLiveAnalysis(micStream, isRecording);

  // GTSinger catalog for phrase picker
  const gtsCatalog = useGtsingerCatalog();
  const catalogClips = song.referenceAudioUrl
    ? findSongPhrases(
        gtsCatalog,
        song.referenceAudioUrl
          .replace(/^https?:\/\/[^/]+/, "")   // strip origin if present
          .replace("/api/audio/file?path=", "")
      )
    : [];

  // Responsive canvas width
  const pitchContainerRef = useRef<HTMLDivElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(760);
  useEffect(() => {
    const el = pitchContainerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(([entry]) => {
      const w = Math.floor(entry.contentRect.width);
      if (w > 0) setCanvasWidth(w);
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Setup local triggers
  useEffect(() => {
    // Start interval timer for song progress & simulated audio playback
    let progressInterval: NodeJS.Timeout;
    if (isRecording) {
      progressInterval = setInterval(() => {
        setProgressSeconds(prev => {
          const nextVal = prev + 1;
          if (nextVal >= 225) { // 3 min 45 secs target
            handleFinishSession();
          }
          return nextVal;
        });

        // Alternate or update lyrics line
        if (progressSeconds > 0 && progressSeconds % 4 === 0) {
          setCurrentLineIndex(prev => (prev + 1) % song.lyrics.length);
        }
      }, 1000);
    }

    return () => {
      clearInterval(progressInterval);
    };
  }, [isRecording, progressSeconds]);

  // Freeze live pitch frames when recording stops so the graph remains visible
  // during the "analyzing" state and doesn't vanish the moment recording ends.
  useEffect(() => {
    if (isRecording) {
      setFrozenFrames([]);
    } else if (liveAnalysis.frames.length > 0) {
      setFrozenFrames(liveAnalysis.frames);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording]);

  useEffect(() => {
    if (!hasActiveRecording) {
      setRecordingDiagnostics(getRecordingDiagnostics());
      return;
    }
    const interval = window.setInterval(() => {
      setRecordingDiagnostics(getRecordingDiagnostics());
    }, 250);
    return () => window.clearInterval(interval);
  }, [hasActiveRecording, isRecording]);

  const cleanupMicrophoneResources = (updateState = true) => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    if (updateState) {
      setMicStream(null);
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    setInputLevel(0);
    setIsVoicedNow(false);
    setRecordingDiagnostics(getRecordingDiagnostics());
    if (updateState) {
      setMicActive(false);
    }
  };

  const stopActiveRecordingSilently = (updateState = true) => {
    if (!audioRecordingRef.current) return;
    audioRecordingRef.current = false;
    if (updateState) {
      setHasActiveRecording(false);
    }
    void stopAudioRecording().catch((error) => {
      console.warn("Failed to stop active recording during cleanup:", error);
    });
  };

  const refreshAudioInputDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const inputs = devices.filter((device) => device.kind === "audioinput");
      setAudioInputDevices(inputs);
      setSelectedAudioDeviceId((current) => {
        if (current && inputs.some((device) => device.deviceId === current)) {
          return current;
        }
        return inputs[0]?.deviceId || "";
      });
    } catch (error) {
      console.warn("Could not enumerate microphone devices:", error);
    }
  };

  const handleMicrophoneUnavailable = (error: unknown, fallbackMessage: string) => {
    console.warn(fallbackMessage, error);
    cleanupMicrophoneResources();
    audioRecordingRef.current = false;
    setHasActiveRecording(false);
    setHasMicPermission(false);
    setIsRecording(false);
    setSessionState("mic_unavailable");
    setMicError(error instanceof Error ? error.message : fallbackMessage);
  };

  const baseAudioConstraints = (): MediaTrackConstraints => ({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,
  });

  const openMicrophoneStream = async (): Promise<MediaStream> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not expose microphone recording APIs.");
    }
    if (!recordingSupport.isSecureContext) {
      throw new Error("Microphone recording requires HTTPS or localhost. Open the app on localhost/127.0.0.1 or an HTTPS URL.");
    }

    const defaultConstraints = baseAudioConstraints();
    if (!selectedAudioDeviceId) {
      return navigator.mediaDevices.getUserMedia({ audio: defaultConstraints });
    }

    try {
      return await navigator.mediaDevices.getUserMedia({
        audio: {
          ...defaultConstraints,
          deviceId: { ideal: selectedAudioDeviceId },
        },
      });
    } catch (error) {
      console.warn("Selected microphone was unavailable; retrying with the default microphone.", error);
      setSelectedAudioDeviceId("");
      return navigator.mediaDevices.getUserMedia({ audio: defaultConstraints });
    }
  };

  const initMicrophone = async () => {
    cleanupMicrophoneResources();
    setMicError(null);
    setAnalysisError(null);
    setSessionState("ready");

    try {
      const stream = await openMicrophoneStream();
      setHasMicPermission(true);
      setMicActive(true);
      micStreamRef.current = stream;
      setMicStream(stream);
      void refreshAudioInputDevices();

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyserRef.current = analyser;
      source.connect(analyser);

      try {
        await startAudioRecording(stream);
        setRecordingDiagnostics(getRecordingDiagnostics());
        audioRecordingRef.current = true;
        setHasActiveRecording(true);
        setIsRecording(true);
        setSessionState("recording");
      } catch (err) {
        handleMicrophoneUnavailable(err, "Microphone opened, but recording could not start.");
        return;
      }

      trackLivePitch();
    } catch (err) {
      handleMicrophoneUnavailable(err, "Microphone is unavailable or access was blocked.");
    }
  };

  // Fetch pre-selected dev samples from backend (dev mode only)
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
    fetch(`${apiBase}/api/audio/dev-samples`)
      .then(r => r.ok ? r.json() : [])
      .then((data: { label: string; path: string }[]) => setDevSamples(data))
      .catch(() => {/* backend not running — silent */ });
  }, []);

  // Clean up microphone resources when leaving the current Studio take.
  useEffect(() => {
    void refreshAudioInputDevices();
    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener("devicechange", refreshAudioInputDevices);
    }
    return () => {
      if (navigator.mediaDevices?.removeEventListener) {
        navigator.mediaDevices.removeEventListener("devicechange", refreshAudioInputDevices);
      }
      stopActiveRecordingSilently(false);
      cleanupMicrophoneResources(false);
    };
  }, []);

  // Mic-driven live pitch loop
  const trackLivePitch = () => {
    if (!analyserRef.current) return;
    const bufferLength = analyserRef.current.fftSize;
    const dataArray = new Float32Array(bufferLength);
    analyserRef.current.getFloatTimeDomainData(dataArray);
    let rmsSum = 0;
    for (let i = 0; i < dataArray.length; i += 1) {
      rmsSum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(rmsSum / dataArray.length);
    const level = Math.max(0, Math.min(1, rms / 0.12));
    setInputLevel(level);

    const detected = detectPitch(dataArray, audioContextRef.current?.sampleRate || 44100);
    let targetY = 50;

    if (detected !== -1) {
      targetY = frequencyToHeight(detected);
      setActiveFrequencyHz(detected);
      setIsVoicedNow(true);
      setActivePitch(targetY);

      // Save pitch for final review/accuracy breakdown
      setCapturedPitches(prev => {
        const next = [...prev, { x: prev.length * 6, y: targetY }];
        if (next.length > 200) next.shift(); // Keep bounded
        return next;
      });

      // Maintain running alignment score
      const currentRefFreq = song.referencePitchSeq[progressSeconds % song.referencePitchSeq.length];
      const diff = Math.abs(targetY - currentRefFreq);
      const acc = Math.max(0, Math.min(100, 100 - diff * 1.5));
      setScoreList(prev => [...prev, acc]);
    } else {
      setActiveFrequencyHz(null);
      setIsVoicedNow(false);
    }

    animationFrameRef.current = requestAnimationFrame(trackLivePitch);
  };

  // Safe fallback waveform simulation
  const simulateVocalPitch = () => {
    const updateSimulatedPitch = () => {
      const time = Date.now() / 400;
      // Synthesize a very realistic singing wave that holds keys
      const sampleCenter = song.referencePitchSeq[progressSeconds % song.referencePitchSeq.length] || 50;
      const wave = sampleCenter + Math.sin(time) * 10 + Math.cos(time * 0.7) * 4;

      setActivePitch(wave);
      setCapturedPitches(prev => {
        const next = [...prev, { x: prev.length * 6, y: wave }];
        if (next.length > 200) next.shift();
        return next;
      });

      animationFrameRef.current = requestAnimationFrame(updateSimulatedPitch);
    };
    animationFrameRef.current = requestAnimationFrame(updateSimulatedPitch);
  };

  // Calculate matching percentage
  useEffect(() => {
    if (scoreList.length > 0) {
      const sum = scoreList.reduce((a, b) => a + b, 0);
      setAverageAccuracy(Math.round(sum / scoreList.length));
    }
  }, [scoreList]);

  const formatTime = (sec: number) => {
    const min = Math.floor(sec / 60);
    const remaining = sec % 60;
    return `${min.toString().padStart(2, "0")}:${remaining.toString().padStart(2, "0")}`;
  };

  const formatTaskToken = (value?: string | null) => {
    if (!value) return "Free Singing";
    return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  };

  const taskInstruction = (() => {
    const type = taskConfig?.task_type || "free_singing";
    const targetNote = taskConfig?.target?.note;
    const duration = typeof taskConfig?.expected_duration === "number" ? taskConfig.expected_duration : undefined;
    if (type === "sustained_note") {
      return `Hold ${targetNote || "your selected note"} as steadily as you can${duration ? ` for about ${duration}s` : ""}.`;
    }
    if (type === "pitch_slide") {
      return `Slide ${taskConfig?.expected_direction || "smoothly"} through your range with an even, connected sound.`;
    }
    if (type === "note_match") {
      return `Listen internally for ${targetNote || "the target note"}, then sing one clear matching note.`;
    }
    if (type === "scale") {
      const key = typeof taskConfig?.target?.key === "string" ? taskConfig.target.key : "the scale";
      return `Sing ${key} one note at a time, keeping each step centered and connected.`;
    }
    if (type === "interval") {
      return "Listen to the two-note reference, then sing both notes with a clear pitch change.";
    }
    if (type === "reference_song") {
      const title = typeof taskConfig?.reference?.title === "string" ? taskConfig.reference.title : "the reference phrase";
      return `Listen to ${title}, then sing the phrase. Scoring remains provisional until reference alignment is complete.`;
    }
    if (type === "rhythm") {
      return "Sing or tap the pattern while keeping onsets close to the reference timing.";
    }
    return "Sing a short phrase freely. Feedback will be general and not reference-melody scoring.";
  })();

  const effectiveTaskConfig: TaskConfig = taskConfig || {
    task_type: "free_singing",
    scoring_mode: "no_reference",
    skill_focus: ["general_pitch"],
  };

  const resetTakeState = () => {
    setProgressSeconds(0);
    setCapturedPitches([]);
    setScoreList([]);
    setAverageAccuracy(98);
    setCurrentLineIndex(1);
    setActivePitch(50);
    setActiveFrequencyHz(null);
    setInputLevel(0);
    setIsVoicedNow(false);
    setRecordingDiagnostics(getRecordingDiagnostics());
  };

  const submitAudioBlobForAnalysis = async (
    audioBlob: Blob,
    sourceLabel: string,
    fileName?: string,
    diagnosticsOverride?: RecordingDiagnostics
  ) => {
    if (!audioBlob.size) {
      throw new Error("Recording did not produce audio data. Check microphone permission and try again.");
    }
    const diagnostics = diagnosticsOverride || recordingDiagnostics;
    if (sourceLabel === "My Recording" && diagnostics.durationSeconds < 0.75) {
      throw new Error("Recording was too short to analyze. Record at least one second of singing.");
    }

    const recordedAttempt: Omit<RecordedAttempt, "audioUrl"> = {
      audioBlob,
      mimeType: audioBlob.type || "audio/wav",
      recordedAt: new Date().toISOString(),
      sourceLabel,
    };

    try {
      const mlAnalysis = await analyzeAudioWithML(
        audioBlob,
        song.title,
        song.artist,
        import.meta.env.VITE_API_URL || "http://localhost:8000",
        effectiveTaskConfig,
        fileName
      );

      const performanceResult = mlAnalysisToPerformanceResult(song.id, mlAnalysis, song.title, song.artist, effectiveTaskConfig);
      performanceResult.sessionState = "review";
      setSessionState("review");
      onSessionComplete(performanceResult, recordedAttempt);
    } catch (mlError) {
      console.error("ML analysis failed:", mlError);
      const message = mlError instanceof Error ? mlError.message : "Analysis failed";
      setAnalysisError(message);
      setSessionState("error");
      const unavailable = createAnalysisUnavailablePerformanceResult(song.id, song.title, song.artist, message, effectiveTaskConfig);
      unavailable.sessionState = "error";
      onSessionComplete(unavailable, recordedAttempt);
    }
  };

  const handleFinishSession = async () => {
    if (!audioRecordingRef.current) {
      setIsAnalyzing(false);
      setIsRecording(false);
      setSessionState(hasMicPermission === false ? "mic_unavailable" : "ready");
      setMicError("No recording is available yet. Try the microphone again or use dev test audio.");
      return;
    }

    try {
      setIsAnalyzing(true);
      setSessionState("analyzing");
      setAnalysisError(null);

      // Stop audio recording
      const finalRecordingDiagnostics = getRecordingDiagnostics();
      const audioBlob = await stopAudioRecording();
      audioRecordingRef.current = false;
      setHasActiveRecording(false);
      setIsRecording(false);
      cleanupMicrophoneResources();
      await submitAudioBlobForAnalysis(audioBlob, "My Recording", undefined, finalRecordingDiagnostics);
    } catch (error) {
      console.error("Error finishing session:", error);
      const message = error instanceof Error ? error.message : "Unknown error";
      setAnalysisError(message);
      setSessionState("error");
      const unavailable = createAnalysisUnavailablePerformanceResult(song.id, song.title, song.artist, message, effectiveTaskConfig);
      unavailable.sessionState = "error";
      onSessionComplete(unavailable);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeDevAudio = async () => {
    if (!devAudioFile) return;

    try {
      setIsAnalyzing(true);
      setSessionState("analyzing");
      setAnalysisError(null);

      if (audioRecordingRef.current) {
        await stopAudioRecording();
        audioRecordingRef.current = false;
        setHasActiveRecording(false);
        cleanupMicrophoneResources();
      }
      setIsRecording(false);

      await submitAudioBlobForAnalysis(devAudioFile, `Dev test audio: ${devAudioFile.name}`, devAudioFile.name);
    } catch (error) {
      console.error("Error analyzing dev test audio:", error);
      const message = error instanceof Error ? error.message : "Unknown error";
      setAnalysisError(message);
      setSessionState("error");
      const unavailable = createAnalysisUnavailablePerformanceResult(song.id, song.title, song.artist, message, effectiveTaskConfig);
      unavailable.sessionState = "error";
      onSessionComplete(unavailable);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeDevSample = async () => {
    if (!selectedDevPath) return;
    try {
      setIsAnalyzing(true);
      setSessionState("analyzing");
      setAnalysisError(null);
      if (audioRecordingRef.current) {
        await stopAudioRecording();
        audioRecordingRef.current = false;
        setHasActiveRecording(false);
        cleanupMicrophoneResources();
      }
      setIsRecording(false);
      const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const resp = await fetch(`${apiBase}/api/audio/file?path=${encodeURIComponent(selectedDevPath)}`);
      if (!resp.ok) throw new Error(`Failed to fetch sample: ${resp.statusText}`);
      const blob = await resp.blob();
      const name = selectedDevPath.split("/").pop() ?? "sample.wav";
      const file = new File([blob], name, { type: "audio/wav" });
      await submitAudioBlobForAnalysis(file, `Dev sample: ${name}`, name);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setAnalysisError(message);
      setSessionState("error");
      const unavailable = createAnalysisUnavailablePerformanceResult(song.id, song.title, song.artist, message, effectiveTaskConfig);
      unavailable.sessionState = "error";
      onSessionComplete(unavailable);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTryMicrophoneAgain = () => {
    audioRecordingRef.current = false;
    setHasActiveRecording(false);
    setIsRecording(false);
    setMicError(null);
    setSessionState("ready");
    resetTakeState();
    void initMicrophone();
  };

  const handleStartMicrophone = () => {
    resetTakeState();
    setIsRecording(false);
    setMicError(null);
    setAnalysisError(null);
    void initMicrophone();
  };

  const handleUseDevTestAudio = () => {
    devFileInputRef.current?.click();
  };

  const handleRecordingToggle = () => {
    if (!hasActiveRecording || isAnalyzing) return;
    if (isRecording) {
      pauseAudioRecording();
      setIsRecording(false);
      setSessionState("ready");
    } else {
      resumeAudioRecording();
      setIsRecording(true);
      setSessionState("recording");
    }
  };

  // Dynamic Lyrics scroll lines
  const prevLine = song.lyrics[(currentLineIndex - 1 + song.lyrics.length) % song.lyrics.length];
  const activeLine = song.lyrics[currentLineIndex];
  const nextLine1 = song.lyrics[(currentLineIndex + 1) % song.lyrics.length];
  const nextLine2 = song.lyrics[(currentLineIndex + 2) % song.lyrics.length];
  const taskType = taskConfig?.task_type || "free_singing";
  const isDiagnosticTask = ["sustained_note", "pitch_slide", "note_match", "scale", "interval", "rhythm", "breath_control", "tone_consistency"].includes(taskType);

  // Build a time-series target contour from task config for the live canvas overlay.
  // Wrapped in useMemo so it is not rebuilt on every 10ms live-frame update.
  const targetContour: TargetContourPoint[] | null = useMemo(() => {
    const type = taskConfig?.task_type;
    if (!type || !taskConfig) return null;

    if (type === "pitch_slide") {
      // Use explicit > 0 guard so that an accidental 0 value does not silently
      // replace the reference fallback (which ?? would not catch).
      const tgt = taskConfig.target as Record<string, unknown> | null | undefined;
      const ref = taskConfig.reference as Record<string, unknown> | null | undefined;
      const startRaw = typeof tgt?.start_f0_hz === "number" && (tgt.start_f0_hz as number) > 0
        ? tgt.start_f0_hz as number
        : typeof ref?.start_f0_hz === "number" && (ref.start_f0_hz as number) > 0
        ? ref.start_f0_hz as number
        : null;
      const endRaw = typeof tgt?.end_f0_hz === "number" && (tgt.end_f0_hz as number) > 0
        ? tgt.end_f0_hz as number
        : typeof ref?.end_f0_hz === "number" && (ref.end_f0_hz as number) > 0
        ? ref.end_f0_hz as number
        : null;
      if (startRaw === null || endRaw === null) return null;
      const dur = typeof taskConfig.expected_duration === "number" ? taskConfig.expected_duration : 5;
      const steps = Math.ceil(dur / 0.1);
      return Array.from({ length: steps + 1 }, (_, i) => {
        const t = (i / steps) * dur;
        const hz = startRaw + (endRaw - startRaw) * (i / steps);
        return { t_s: t, f0_hz: hz, note_name: hzToNoteName(hz) };
      });
    }

    if (type === "sustained_note") {
      // Human reference catalog: dense contour in reference.f0_hz with reference.hop_s
      const ref = taskConfig.reference as Record<string, unknown> | null | undefined;
      if (Array.isArray(ref?.f0_hz) && typeof ref?.hop_s === "number" && ref.hop_s > 0) {
        const hop = ref.hop_s as number;
        const f0Array = ref.f0_hz as number[];
        return f0Array
          .map((hz, i) => ({ t_s: i * hop, f0_hz: hz, note_name: hzToNoteName(hz) }))
          .filter((p) => p.f0_hz > 0);
      }
      // Legacy scalar target
      const hz = taskConfig.target?.f0_hz;
      const dur = typeof taskConfig.expected_duration === "number" ? taskConfig.expected_duration : 5;
      if (typeof hz !== "number" || hz <= 0) return null;
      return [
        { t_s: 0, f0_hz: hz, note_name: hzToNoteName(hz) },
        { t_s: dur, f0_hz: hz, note_name: hzToNoteName(hz) },
      ];
    }

    if (type === "scale" || type === "interval" || type === "reference_song") {
      const ref = taskConfig.reference ?? taskConfig.target;
      const f0Array: number[] = Array.isArray(ref?.f0_hz) ? ref.f0_hz : [];
      if (!f0Array.length) return null;

      // Dense contour format from the human reference catalog API:
      // reference.hop_s is present and f0_hz is a time-series array (0 = unvoiced).
      if (typeof ref?.hop_s === "number" && ref.hop_s > 0) {
        const hop = ref.hop_s as number;
        return f0Array
          .map((hz, i) => ({ t_s: i * hop, f0_hz: hz, note_name: hzToNoteName(hz) }))
          .filter((p) => p.f0_hz > 0);
      }

      // Legacy note-sequence format: f0_hz = [261.63, 293.66, ...] with durations_s.
      const durs: number[] = Array.isArray(ref?.durations_s) ? ref.durations_s : [];
      const points: TargetContourPoint[] = [];
      let cursor = 0;
      f0Array.forEach((hz, i) => {
        const d = durs[i] ?? 0.7;
        // Use 99% of duration for the held portion; avoids the -0.01 hardcode
        // that would invert for durations shorter than 10 ms.
        points.push({ t_s: cursor, f0_hz: hz, note_name: hzToNoteName(hz) });
        points.push({ t_s: cursor + d * 0.99, f0_hz: hz, note_name: hzToNoteName(hz) });
        cursor += d;
      });
      return points;
    }

    return null;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskConfig]);

  const exerciseDuration: number | undefined = (() => {
    if (typeof taskConfig?.expected_duration === "number") return taskConfig.expected_duration;
    if (targetContour?.length) return targetContour[targetContour.length - 1].t_s;
    return undefined;
  })();
  const targetF0Hz = typeof taskConfig?.target?.f0_hz === "number" ? taskConfig.target.f0_hz : null;
  const targetDisplay = targetF0Hz
    ? `${targetF0Hz.toFixed(2)} Hz`
    : typeof taskConfig?.target?.key === "string"
    ? taskConfig.target.key
    : typeof taskConfig?.reference?.title === "string"
    ? taskConfig.reference.title
    : taskConfig?.expected_direction || "free contour";
  const centsError = activeFrequencyHz && targetF0Hz
    ? Math.round(1200 * Math.log2(activeFrequencyHz / targetF0Hz))
    : null;
  const pitchStatusText = activeFrequencyHz
    ? `${activeFrequencyHz.toFixed(1)} Hz${centsError !== null ? ` • ${centsError > 0 ? "+" : ""}${centsError} cents` : ""}`
    : isRecording
    ? "Listening for singing…"
    : "Mic off";
  const inputLevelPercent = Math.round(inputLevel * 100);
  const recordingBytesLabel = recordingDiagnostics.bytesCollected > 1024
    ? `${(recordingDiagnostics.bytesCollected / 1024).toFixed(1)} KB`
    : `${recordingDiagnostics.bytesCollected} B`;

  return (
    <div className="flex flex-col gap-8 pb-10">

      {/* Title Header */}
      <div className="flex justify-between items-center bg-surface-container/30 border border-white/5 rounded-2xl p-6">
        <div>
          <h1 className="font-display font-extrabold text-2xl md:text-3xl text-white">
            {song.title}
          </h1>
          <p className="font-sans text-xs text-secondary font-bold uppercase tracking-wider">
            {song.artist} <span className="text-on-surface-variant/40 mx-2">•</span> {song.genre}
          </p>
          <div className="mt-3 rounded-xl bg-surface-container-high/60 border border-white/5 px-4 py-3 max-w-2xl">
            <p className="text-[10px] uppercase tracking-wider font-bold text-tertiary">
              {formatTaskToken(taskConfig?.task_type || "free_singing")}
            </p>
            <p className="text-xs text-on-surface-variant mt-1 leading-relaxed">
              {taskInstruction}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-gradient-to-r from-tertiary/10 to-primary/10 border border-tertiary/20 px-4 py-1.5 rounded-full">
            {hasMicPermission === false ? (
              <MicOff className="w-4 h-4 text-error" />
            ) : (
              <Mic className="w-4 h-4 text-tertiary animate-pulse" />
            )}
            <span className="text-xs font-bold text-tertiary uppercase tracking-wider">
              {hasMicPermission === false ? "MIC UNAVAILABLE" : micActive ? "LIVE INPUT ACTIVE" : "INPUT IDLE"}
            </span>
          </div>
          <span className="hidden md:inline-flex bg-white/5 border border-white/10 text-on-surface-variant px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider">
            {sessionState.replaceAll("_", " ")}
          </span>
          <button
            onClick={onExit}
            className="text-xs font-semibold px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-all"
          >
            End Practice
          </button>
        </div>
      </div>

      <ReferenceTonePlayer
        taskConfig={taskConfig || { task_type: "free_singing" }}
        disabled={hasActiveRecording || isAnalyzing}
        onPlaybackStateChange={setSessionState}
        referenceAudioUrl={song.referenceAudioUrl}
        referenceStyle={song.referenceStyle}
        referenceType={song.referenceType}
        catalogClips={catalogClips.length > 1 ? catalogClips : undefined}
      />

      {!hasActiveRecording && sessionState !== "mic_unavailable" && sessionState !== "analyzing" && (
        <section className="glass-card rounded-2xl p-5 border border-tertiary/20 bg-tertiary/5">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-11 h-11 rounded-xl bg-tertiary/10 text-tertiary flex items-center justify-center flex-shrink-0">
                <Mic className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-widest font-bold text-tertiary">
                  Browser microphone
                </p>
                <h3 className="font-display font-bold text-lg text-white mt-1">
                  Start a live recording when you are ready
                </h3>
                <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
                  Hear the reference first, then start the microphone. Recording happens directly in the browser and is sent to the same task-aware analysis path.
                </p>
                {import.meta.env.DEV && !recordingSupport.isSecureContext && (
                  <p className="text-[11px] text-yellow-400/70 mt-2 font-mono">
                    ⚠ Not a secure context — microphone requires HTTPS or localhost.
                  </p>
                )}
                <div className="mt-4 grid sm:grid-cols-[minmax(0,1fr)_auto] gap-2 max-w-xl">
                  <select
                    value={selectedAudioDeviceId}
                    onChange={(event) => setSelectedAudioDeviceId(event.target.value)}
                    disabled={isAnalyzing || hasActiveRecording || audioInputDevices.length === 0}
                    className="w-full rounded-xl bg-background/60 border border-white/10 px-3 py-2.5 text-xs text-on-surface disabled:opacity-50"
                    aria-label="Microphone input device"
                  >
                    {audioInputDevices.length === 0 ? (
                      <option value="">Default microphone</option>
                    ) : (
                      audioInputDevices.map((device, index) => (
                        <option key={device.deviceId || `audio-input-${index}`} value={device.deviceId}>
                          {device.label || `Microphone ${index + 1}`}
                        </option>
                      ))
                    )}
                  </select>
                  <button
                    onClick={refreshAudioInputDevices}
                    disabled={isAnalyzing || hasActiveRecording}
                    className="px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
                  </button>
                </div>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
              <button
                onClick={handleStartMicrophone}
                disabled={isAnalyzing || !recordingSupport.hasMediaDevices || (!recordingSupport.hasWebAudioRecorder && !recordingSupport.hasMediaRecorder)}
                className="px-5 py-3 rounded-xl bg-gradient-to-r from-secondary to-primary text-on-primary text-xs font-bold hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <Mic className="w-4 h-4" />
                Start microphone recording
              </button>
              {import.meta.env.DEV && (
                <button
                  onClick={handleUseDevTestAudio}
                  disabled={isAnalyzing}
                  className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Use dev WAV
                </button>
              )}
            </div>
          </div>
        </section>
      )}

      {sessionState === "mic_unavailable" && (
        <section className="glass-card rounded-2xl p-5 border border-error/25 bg-error/5">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-11 h-11 rounded-xl bg-error/10 text-error flex items-center justify-center flex-shrink-0">
                <MicOff className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-widest font-bold text-error">
                  Microphone unavailable
                </p>
                <h3 className="font-display font-bold text-lg text-white mt-1">
                  Recording did not start
                </h3>
                <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
                  {micError || "Allow microphone access, then try again. You can also use dev test audio while developing."}
                </p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row sm:items-center gap-3">
              <button
                onClick={handleTryMicrophoneAgain}
                disabled={isAnalyzing}
                className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Try microphone again
              </button>
              <button
                onClick={refreshAudioInputDevices}
                disabled={isAnalyzing}
                className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Refresh devices
              </button>
              {import.meta.env.DEV && (
                <button
                  onClick={handleUseDevTestAudio}
                  disabled={isAnalyzing}
                  className="px-4 py-2.5 rounded-xl bg-tertiary/15 border border-tertiary/25 text-tertiary text-xs font-bold hover:bg-tertiary/20 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Use dev test audio
                </button>
              )}
              <button
                onClick={onExit}
                className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface-variant hover:bg-white/10"
              >
                Back to task setup
              </button>
            </div>
          </div>
        </section>
      )}

      {import.meta.env.DEV && (
        <section className="glass-card rounded-2xl p-5 border border-tertiary/20 bg-tertiary/5">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-11 h-11 rounded-xl bg-tertiary/10 text-tertiary flex items-center justify-center flex-shrink-0">
                <FileAudio className="w-5 h-5" />
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-widest font-bold text-tertiary">
                  Dev test audio
                </p>
                <h3 className="font-display font-bold text-lg text-white mt-1">
                  Run a local WAV fixture
                </h3>
                <p className="text-xs text-on-surface-variant leading-relaxed mt-1">
                  Development-only path for MVP E2E checks. The selected WAV uses the current task config and the same ML analysis request as a recorded take.
                </p>
                {devAudioFile && (
                  <p className="text-[11px] text-on-surface-variant mt-2">
                    Selected: <span className="text-white font-bold">{devAudioFile.name}</span>
                  </p>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-3 w-full sm:w-auto">
              {/* Pre-selected sample dropdown */}
              {devSamples.length > 0 && (
                <div className="flex flex-col sm:flex-row gap-2">
                  <select
                    value={selectedDevPath}
                    onChange={e => setSelectedDevPath(e.target.value)}
                    className="flex-1 px-3 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs text-on-surface appearance-none cursor-pointer focus:outline-none focus:ring-1 focus:ring-tertiary/50"
                  >
                    <option value="">— select a test clip —</option>
                    {devSamples.map(s => (
                      <option key={s.path} value={s.path}>{s.label}</option>
                    ))}
                  </select>
                  <button
                    onClick={handleAnalyzeDevSample}
                    disabled={!selectedDevPath || isAnalyzing}
                    className="px-4 py-2.5 rounded-xl bg-tertiary/15 border border-tertiary/25 text-tertiary text-xs font-bold hover:bg-tertiary/20 disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                  >
                    {isAnalyzing ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              )}
              {/* Fallback: manual file picker */}
              <div className="flex flex-row items-center gap-2">
                <label className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 cursor-pointer text-center">
                  Upload WAV
                  <input
                    ref={devFileInputRef}
                    type="file"
                    accept=".wav,audio/wav,audio/x-wav"
                    className="hidden"
                    onChange={(event) => setDevAudioFile(event.target.files?.[0] || null)}
                  />
                </label>
                {devAudioFile && (
                  <button
                    onClick={handleAnalyzeDevAudio}
                    disabled={isAnalyzing}
                    className="px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold text-on-surface hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {isAnalyzing ? "Analyzing..." : devAudioFile.name.slice(0, 20)}
                  </button>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {isDiagnosticTask ? (
        <div className="glass-card rounded-3xl py-10 px-6 min-h-[220px] flex flex-col items-center justify-center text-center border border-white/5">
          <p className="text-[10px] uppercase tracking-widest font-bold text-tertiary">
            Human vocal target
          </p>
          <h2 className="font-display font-extrabold text-3xl md:text-5xl text-white mt-3">
            {taskConfig?.target?.note || formatTaskToken(taskType)}
          </h2>
          <p className="text-sm text-on-surface-variant mt-3 max-w-2xl leading-relaxed">
            {taskInstruction}
          </p>
          <div className="mt-6 grid sm:grid-cols-3 gap-3 w-full max-w-3xl">
            <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">Target</p>
              <p className="font-mono text-lg text-white mt-1">
                {targetDisplay}
              </p>
            </div>
            <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">Live voice</p>
              <p className="font-mono text-lg text-white mt-1">
                {activeFrequencyHz ? `${activeFrequencyHz.toFixed(1)} Hz` : "waiting"}
              </p>
            </div>
            <div className="rounded-2xl bg-white/5 border border-white/10 p-4">
              <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">Pitch error</p>
              <p className={`font-mono text-lg mt-1 ${
                centsError === null ? "text-on-surface-variant" : Math.abs(centsError) <= 25 ? "text-tertiary" : "text-primary"
              }`}>
                {centsError === null ? "n/a" : `${centsError > 0 ? "+" : ""}${centsError} cents`}
              </p>
            </div>
          </div>
          <div className="mt-4 w-full max-w-3xl rounded-2xl bg-white/5 border border-white/10 p-4 text-left">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">
                  Mic input health
                </p>
                <p className="text-xs text-on-surface-variant mt-1">
                  {hasActiveRecording
                    ? isVoicedNow
                      ? "Voiced singing detected"
                      : inputLevel > 0.08
                      ? "Signal present, no stable pitch yet"
                      : "Very quiet input"
                    : "Start recording to monitor input"}
                </p>
              </div>
              <span className={`text-xs font-bold px-3 py-1 rounded-full border ${
                isVoicedNow
                  ? "text-tertiary border-tertiary/30 bg-tertiary/10"
                  : inputLevel > 0.08
                  ? "text-primary border-primary/30 bg-primary/10"
                  : "text-on-surface-variant border-white/10 bg-white/5"
              }`}>
                {hasActiveRecording ? `${inputLevelPercent}% level` : "idle"}
              </span>
            </div>
            <div className="mt-3 h-2 rounded-full bg-background/70 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  isVoicedNow ? "bg-tertiary" : inputLevel > 0.08 ? "bg-primary" : "bg-white/20"
                }`}
                style={{ width: `${hasActiveRecording ? inputLevelPercent : 0}%` }}
              />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-on-surface-variant">
              <span>
                Recorder: <strong className="text-white">{recordingDiagnostics.state}</strong>
              </span>
              <span>
                Chunks: <strong className="text-white">{recordingDiagnostics.chunkCount}</strong>
              </span>
              <span>
                Data: <strong className="text-white">{recordingBytesLabel}</strong>
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative glass-card rounded-3xl py-10 px-6 min-h-[280px] flex flex-col items-center justify-center overflow-hidden">
          <div className="absolute inset-y-0 left-0 right-0 bg-gradient-to-b from-background via-transparent to-background pointer-events-none z-10" />

          <div className="space-y-6 text-center z-0 transition-all duration-1000 ease-in-out">
            <p className="text-sm md:text-lg font-medium opacity-20 filter blur-[0.5px]">
              {prevLine}
            </p>
            <motion.p
              key={activeLine}
              initial={{ scale: 0.95, opacity: 0.5, filter: "blur(4px)" }}
              animate={{ scale: 1.1, opacity: 1, filter: "blur(0px)" }}
              className="text-xl md:text-3xl font-display font-bold text-primary drop-shadow-[0_0_15px_rgba(255,177,192,0.4)]"
            >
              {activeLine}
            </motion.p>
            <p className="text-sm md:text-lg font-medium opacity-25">
              {nextLine1}
            </p>
            <p className="text-xs md:text-sm font-medium opacity-10">
              {nextLine2}
            </p>
          </div>
        </div>
      )}

      {/* Live Pitch Monitor */}
      <div className="glass-card rounded-2xl p-4 space-y-2">
        <div className="flex justify-between items-center px-1">
          <span className="text-[10px] font-bold text-on-surface-variant/60 uppercase tracking-widest">
            Live Vocal Analysis
          </span>
          <span className="bg-tertiary/15 border border-tertiary/30 text-tertiary px-3 py-1 rounded-full text-xs font-bold font-mono tracking-wider">
            {pitchStatusText}
          </span>
        </div>

        {/* Canvas container — ref used for responsive width measurement */}
        <div
          ref={pitchContainerRef}
          className="w-full"
          style={!isRecording && frozenFrames.length > 0 ? { overflowX: "auto", overflowY: "hidden" } : undefined}
        >
          {isRecording || frozenFrames.length > 0 ? (
            <LivePitchCanvas
              frames={isRecording ? liveAnalysis.frames : frozenFrames}
              isConnected={isRecording ? liveAnalysis.isConnected : false}
              latencyMs={isRecording ? liveAnalysis.latencyMs : undefined}
              rtf={isRecording ? liveAnalysis.rtf : undefined}
              height={260}
              connectionError={isRecording ? liveAnalysis.connectionError : undefined}
              targetContour={targetContour}
              exerciseDuration={exerciseDuration}
              frozen={!isRecording && frozenFrames.length > 0}
            />
          ) : (
            <div className="overflow-hidden h-24 rounded-xl bg-surface-container-lowest/70 border border-white/5 flex items-center justify-center">
              <span className="text-xs text-on-surface-variant/40">Start recording to see live pitch</span>
            </div>
          )}
        </div>

        {/* Signal summary row — shown during recording */}
        {isRecording && liveAnalysis.latestFrame && (
          <div className="flex gap-4 px-1 text-[10px] font-mono text-on-surface-variant/60">
            <span>
              {liveAnalysis.latestFrame.voiced
                ? `♪ ${liveAnalysis.latestFrame.pitch_hz.toFixed(0)} Hz`
                : "— silent"}
            </span>
            {liveAnalysis.latestFrame.tempo_bpm > 0 && (
              <span>♩= {liveAnalysis.latestFrame.tempo_bpm.toFixed(0)}</span>
            )}
            {liveAnalysis.latestFrame.vibrato_rate_hz > 0 && (
              <span className="text-[#ff9f43]">
                vib {liveAnalysis.latestFrame.vibrato_rate_hz.toFixed(1)} Hz · {liveAnalysis.latestFrame.vibrato_depth_cents.toFixed(0)}¢
              </span>
            )}
            <span>
              {liveAnalysis.latestFrame.loudness_db.toFixed(0)} dBFS
            </span>
          </div>
        )}
      </div>

      {/* Dashboard Audio Console Controls — sticky so recording button is always visible */}
      <div className="sticky bottom-4 z-30 glass-card rounded-3xl p-6 flex flex-col md:flex-row justify-between items-center gap-6 border border-white/8 backdrop-blur-xl shadow-2xl">

        {/* BPM & Live Volume meter */}
        <div className="flex items-center gap-8 w-full md:w-auto">
          {!isDiagnosticTask && (
            <div className="flex flex-col items-center">
              <div className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center text-on-surface-variant">
                <Music className="w-5 h-5 text-on-surface" />
              </div>
              <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider mt-1.5">
                BPM: {song.bpm}
              </span>
            </div>
          )}

          {isDiagnosticTask ? (
            <div className="rounded-2xl bg-white/5 border border-white/10 px-4 py-3">
              <p className="text-[10px] uppercase tracking-wider text-on-surface-variant font-bold">
                Recording source
              </p>
              <p className="text-sm text-white font-bold mt-1">
                {hasActiveRecording ? "Browser microphone" : "Waiting to start"}
              </p>
              {hasActiveRecording && (
                <p className="text-[11px] text-on-surface-variant mt-1">
                  {isVoicedNow ? "Voiced" : inputLevel > 0.08 ? "Signal" : "Quiet"} • {inputLevelPercent}% level
                </p>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-1 w-full md:w-36">
              <div className="flex justify-between items-center text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                <span>Backing Volume</span>
                <span>{isMuted ? "MUTED" : `${volume}%`}</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className="text-on-surface-variant hover:text-white transition-colors"
                >
                  {isMuted ? <VolumeX className="w-4 h-4 text-error" /> : <Volume2 className="w-4 h-4" />}
                </button>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={volume}
                  onChange={(e) => {
                    setVolume(Number(e.target.value));
                    setIsMuted(false);
                  }}
                  className="w-full accent-tertiary h-1 bg-white/10 rounded-full appearance-none cursor-pointer"
                />
              </div>
            </div>
          )}
        </div>

        {/* Audio Recording State Button */}
        <div className="flex items-center gap-6">
          <button
            onClick={handleRecordingToggle}
            disabled={!hasActiveRecording || isAnalyzing}
            className={`w-14 h-14 rounded-full flex items-center justify-center transition-all ${
              isRecording ? "bg-error/20 border border-error text-error scale-105" : "bg-white/10 hover:bg-white/15"
            } disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {isRecording ? <Square className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 text-white" />}
          </button>

          <button
            onClick={handleFinishSession}
            disabled={isAnalyzing || !hasActiveRecording}
            className="px-8 py-3.5 rounded-full bg-gradient-to-r from-secondary to-primary text-on-primary font-bold hover:brightness-110 hover:shadow-[0_0_20px_rgba(255,177,192,0.5)] active:scale-95 transition-all duration-300 shadow-xl glow-pink flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? (
              <>
                <Loader className="w-4 h-4 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : !hasActiveRecording ? (
              <>
                <span>No recording available</span>
                <MicOff className="w-4 h-4" />
              </>
            ) : (
              <>
                <span>Finish &amp; Review</span>
                <SkipForward className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

        {/* Status duration and controls toggle */}
        <div className="flex items-center justify-between w-full md:w-auto gap-8">
          <div className="text-right">
            <p className="font-display font-extrabold text-2xl text-primary font-mono tracking-wider tabular-nums">
              {formatTime(progressSeconds)}
            </p>
            <p className="text-[10px] font-bold text-on-surface-variant uppercase tracking-widest mt-0.5">
              Live Progress
            </p>
          </div>

          <div className="h-10 w-[1px] bg-white/10 hidden md:block" />

          {!isDiagnosticTask && (
            <button
              onClick={() => setShowLyricsText(!showLyricsText)}
              className={`flex flex-col items-center gap-1 group cursor-pointer ${
                showLyricsText ? "text-primary" : "text-on-surface-variant"
              }`}
            >
              <div className={`w-10 h-10 rounded-full border border-white/10 flex items-center justify-center transition-all ${
                showLyricsText ? "bg-primary/10 border-primary" : "group-hover:bg-white/5"
              }`}>
                <FileText className="w-5 h-5" />
              </div>
              <span className="text-[10px] font-bold uppercase tracking-wider">
                {showLyricsText ? "Lyrics On" : "Lyrics Off"}
              </span>
            </button>
          )}
        </div>

      </div>

    </div>
  );
}
