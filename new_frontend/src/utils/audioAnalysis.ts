import {
  AnalysisPayload,
  FeedbackPolicy,
  MLAnalysisResult,
  PerformanceResult,
  TaskConfig,
  TaskResult,
  UiReadyAnalysis,
} from "../types";

/**
 * Utility functions for audio recording, upload, and ML analysis
 */

// Recording state for browser microphone capture. We prefer browser-encoded WAV
// because the backend's librosa/soundfile path cannot reliably decode every
// browser MediaRecorder codec (notably WebM/Opus without ffmpeg).
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordingStream: MediaStream | null = null;
let recordingOwnsStream = false;
let recordingStartedAtMs: number | null = null;
let wavAudioContext: AudioContext | null = null;
let wavSource: MediaStreamAudioSourceNode | null = null;
let wavProcessor: ScriptProcessorNode | null = null;
let wavMonitorGain: GainNode | null = null;
let wavChunks: Float32Array[] = [];
let wavBytesCollected = 0;
let wavRecordingState: RecordingState | "inactive" = "inactive";
let wavSampleRate = 44100;

export interface RecordingDiagnostics {
  state: RecordingState | "inactive";
  mimeType: string | null;
  chunkCount: number;
  bytesCollected: number;
  durationSeconds: number;
}

export function getSupportedRecordingMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return undefined;
  }
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export function getBrowserRecordingSupport() {
  const hasWebAudioRecorder = typeof window !== "undefined" && Boolean(window.AudioContext || (window as any).webkitAudioContext);
  return {
    isSecureContext: typeof window === "undefined" ? true : window.isSecureContext,
    hasMediaDevices: typeof navigator !== "undefined" && Boolean(navigator.mediaDevices?.getUserMedia),
    hasWebAudioRecorder,
    hasMediaRecorder: typeof MediaRecorder !== "undefined",
    preferredMimeType: hasWebAudioRecorder ? "audio/wav" : getSupportedRecordingMimeType() || null,
  };
}

function recordingFileNameForType(mimeType: string): string {
  if (mimeType.includes("mp4")) return "recording.m4a";
  if (mimeType.includes("ogg")) return "recording.ogg";
  if (mimeType.includes("wav")) return "recording.wav";
  return "recording.webm";
}

/**
 * Start recording audio from the user's microphone
 */
export async function startAudioRecording(stream?: MediaStream): Promise<void> {
  try {
    if ((mediaRecorder && mediaRecorder.state !== "inactive") || wavRecordingState !== "inactive") {
      throw new Error("A recording is already active");
    }

    recordingOwnsStream = !stream;
    recordingStream = stream || await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      },
    });

    if (await startWavRecording(recordingStream)) {
      return;
    }

    if (typeof MediaRecorder === "undefined") {
      throw new Error("This browser does not support MediaRecorder or Web Audio WAV recording.");
    }

    const mimeType = getSupportedRecordingMimeType();
    mediaRecorder = mimeType
      ? new MediaRecorder(recordingStream, { mimeType })
      : new MediaRecorder(recordingStream);

    audioChunks = [];
    recordingStartedAtMs = Date.now();

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.start(250);
  } catch (error) {
    console.error("Error starting audio recording:", error);
    throw error;
  }
}

async function startWavRecording(stream: MediaStream): Promise<boolean> {
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
  if (!AudioContextClass) return false;

  try {
    wavAudioContext = new AudioContextClass();
    if (wavAudioContext.state === "suspended") {
      await wavAudioContext.resume();
    }
    wavSampleRate = wavAudioContext.sampleRate;
    wavSource = wavAudioContext.createMediaStreamSource(stream);
    wavProcessor = wavAudioContext.createScriptProcessor(4096, 1, 1);
    wavMonitorGain = wavAudioContext.createGain();
    wavMonitorGain.gain.value = 0;
    wavChunks = [];
    wavBytesCollected = 0;
    wavRecordingState = "recording";
    recordingStartedAtMs = Date.now();

    wavProcessor.onaudioprocess = (event) => {
      if (wavRecordingState !== "recording") return;
      const input = event.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);
      wavChunks.push(copy);
      wavBytesCollected += copy.length * 2;
    };

    wavSource.connect(wavProcessor);
    wavProcessor.connect(wavMonitorGain);
    wavMonitorGain.connect(wavAudioContext.destination);
    return true;
  } catch (error) {
    console.warn("Web Audio WAV recording unavailable; falling back to MediaRecorder.", error);
    cleanupWavRecorder(false);
    return false;
  }
}

/**
 * Stop recording and return the audio blob
 */
export async function stopAudioRecording(): Promise<Blob> {
  if (wavRecordingState !== "inactive") {
    const audioBlob = encodeWavBlob(wavChunks, wavSampleRate);
    cleanupRecordingStream();
    await cleanupWavRecorder(true);
    recordingStartedAtMs = null;
    return audioBlob;
  }

  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject(new Error("No active recording"));
      return;
    }

    mediaRecorder.onstop = () => {
      const mimeType = mediaRecorder?.mimeType || audioChunks[0]?.type || "audio/webm";
      const audioBlob = new Blob(audioChunks, { type: mimeType });

      // Clean up. If Studio provided the stream, Studio owns track cleanup.
      if (recordingOwnsStream && recordingStream) {
        recordingStream.getTracks().forEach(track => track.stop());
      }
      mediaRecorder = null;
      recordingStream = null;
      recordingOwnsStream = false;
      recordingStartedAtMs = null;
      audioChunks = [];

      resolve(audioBlob);
    };

    mediaRecorder.stop();
  });
}

export function pauseAudioRecording(): void {
  if (wavRecordingState === "recording") {
    wavRecordingState = "paused";
    return;
  }
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.pause();
}

export function resumeAudioRecording(): void {
  if (wavRecordingState === "paused") {
    wavRecordingState = "recording";
    return;
  }
  if (!mediaRecorder || mediaRecorder.state !== "paused") return;
  mediaRecorder.resume();
}

export function getRecordingDiagnostics(): RecordingDiagnostics {
  const isWavRecording = wavRecordingState !== "inactive";
  const bytesCollected = isWavRecording ? wavBytesCollected : audioChunks.reduce((sum, chunk) => sum + chunk.size, 0);
  const durationSeconds = recordingStartedAtMs ? (Date.now() - recordingStartedAtMs) / 1000 : 0;
  return {
    state: isWavRecording ? wavRecordingState : mediaRecorder?.state || "inactive",
    mimeType: isWavRecording ? "audio/wav" : mediaRecorder?.mimeType || null,
    chunkCount: isWavRecording ? wavChunks.length : audioChunks.length,
    bytesCollected,
    durationSeconds,
  };
}

function cleanupRecordingStream(): void {
  if (recordingOwnsStream && recordingStream) {
    recordingStream.getTracks().forEach(track => track.stop());
  }
  recordingStream = null;
  recordingOwnsStream = false;
}

async function cleanupWavRecorder(closeContext: boolean): Promise<void> {
  if (wavProcessor) {
    wavProcessor.onaudioprocess = null;
    wavProcessor.disconnect();
  }
  wavSource?.disconnect();
  wavMonitorGain?.disconnect();
  if (closeContext && wavAudioContext && wavAudioContext.state !== "closed") {
    await wavAudioContext.close();
  }
  wavAudioContext = null;
  wavSource = null;
  wavProcessor = null;
  wavMonitorGain = null;
  wavChunks = [];
  wavBytesCollected = 0;
  wavRecordingState = "inactive";
  wavSampleRate = 44100;
}

function encodeWavBlob(chunks: Float32Array[], sampleRate: number): Blob {
  const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + sampleCount * 2);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + sampleCount * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, sampleCount * 2, true);

  let offset = 44;
  chunks.forEach((chunk) => {
    for (let i = 0; i < chunk.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  });

  return new Blob([view], { type: "audio/wav" });
}

function writeAscii(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

/**
 * Upload audio to backend for ML analysis
 */
export async function analyzeAudioWithML(
  audioBlob: Blob,
  songTitle: string,
  artist: string,
  backendUrl: string = "http://localhost:8000",
  taskConfig?: TaskConfig,
  fileName?: string
): Promise<AnalysisPayload> {
  try {
    const formData = new FormData();
    const uploadName = fileName || (audioBlob instanceof File ? audioBlob.name : recordingFileNameForType(audioBlob.type));
    formData.append("file", audioBlob, uploadName);
    formData.append("song_title", songTitle);
    formData.append("artist", artist);
    formData.append("response_mode", "ui_ready");
    formData.append("include_ui_ready_analysis", "true");
    formData.append("include_frames", "true");
    formData.append("debug", "false");
    if (taskConfig) {
      formData.append("task_config", JSON.stringify(taskConfig));
    }

    const response = await fetch(
      `${backendUrl}/api/audio/analyze-with-ml`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`);
    }

    const result = await response.json();

    if (result.status === "error") {
      throw new Error(result.error || "Analysis failed");
    }

    return extractAnalysisPayload(result);
  } catch (error) {
    console.error("Error analyzing audio:", error);
    throw error;
  }
}

function extractAnalysisPayload(response: any): AnalysisPayload {
  const data = response?.data ?? response;
  return data?.ui_ready_analysis ?? data?.uiReadyAnalysis ?? data?.analysis ?? data;
}

/**
 * Get coaching feedback from Gemini (legacy, still available)
 */
export async function getGeminiCoachingFeedback(
  result: Partial<PerformanceResult>,
  serverUrl: string = "http://localhost:3000"
): Promise<any> {
  try {
    const response = await fetch(`${serverUrl}/api/coaching-feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        songTitle: result.songTitle,
        artist: result.artist,
        score: result.overallScore,
        intonation: result.intonation,
        rhythm: result.rhythm,
        timbre: result.timbre,
        dynamics: result.dynamics,
      }),
    });

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error getting Gemini coaching:", error);
    throw error;
  }
}

/**
 * Convert ML analysis result to PerformanceResult format
 */
export function mlAnalysisToPerformanceResult(
  songId: string,
  analysisPayload: AnalysisPayload,
  fallbackSongTitle: string = "Practice Exercise",
  fallbackArtist: string = "AI Coach",
  fallbackTaskConfig?: TaskConfig
): PerformanceResult {
  const uiReadyAnalysis = extractUiReadyAnalysis(analysisPayload);
  const legacyAnalysis = extractLegacyAnalysis(analysisPayload);

  if (uiReadyAnalysis) {
    return uiReadyAnalysisToPerformanceResult(
      songId,
      uiReadyAnalysis,
      legacyAnalysis,
      fallbackSongTitle,
      fallbackArtist,
      fallbackTaskConfig
    );
  }

  const mlAnalysis = legacyAnalysis as MLAnalysisResult;
  if (!mlAnalysis) {
    return createAnalysisUnavailablePerformanceResult(
      songId,
      fallbackSongTitle,
      fallbackArtist,
      "The backend response did not match a supported analysis format.",
      fallbackTaskConfig
    );
  }
  return {
    songId,
    songTitle: mlAnalysis.songTitle,
    artist: mlAnalysis.artist,
    overallScore: mlAnalysis.score,
    intonation: Math.round(mlAnalysis.pitchAccuracy),
    rhythm: Math.round(mlAnalysis.onsetClarity * 100),
    timbre: Math.round((mlAnalysis.techniqueConfidence || 75)),
    dynamics: Math.round((mlAnalysis.voiceQuality?.hnrDb || 20) * 2.5), // Normalize HNR to 0-100
    coachingNotes: [
      {
        type: "info",
        category: "Model Summary",
        title: "Analysis Complete",
        text: mlAnalysis.summary || "The model returned a legacy analysis result.",
      },
      {
        type: "info",
        category: "Pitch",
        title: `${Math.round(mlAnalysis.pitchAccuracy)}% Pitch Estimate`,
        text: `Estimated pitch drift was ${Math.abs(mlAnalysis.pitchDrift).toFixed(1)} cents. This is legacy model output, not reference-melody scoring.`,
      },
      {
        type: "info",
        category: "Signal Cues",
        title: `${mlAnalysis.onsetCount} Onsets Detected`,
        text: "Onset and signal cues are provisional and should not be treated as technique diagnosis.",
      },
    ],
    mlAnalysis,
    uiReadyAnalysis: mlAnalysis.uiReadyAnalysis,
    taskConfig: mlAnalysis.uiReadyAnalysis?.task_config || fallbackTaskConfig,
    recordedAt: new Date().toISOString(),
  };
}

function extractUiReadyAnalysis(payload: AnalysisPayload): UiReadyAnalysis | undefined {
  const data = payload as any;
  if (isUiReadyAnalysis(data)) return data;
  if (isUiReadyAnalysis(data?.ui_ready_analysis)) return data.ui_ready_analysis;
  if (isUiReadyAnalysis(data?.uiReadyAnalysis)) return data.uiReadyAnalysis;
  if (isUiReadyAnalysis(data?.analysis)) return data.analysis;
  return undefined;
}

function extractLegacyAnalysis(payload: AnalysisPayload): MLAnalysisResult | undefined {
  const data = payload as any;
  if (data?.score !== undefined && data?.pitchAccuracy !== undefined) return data as MLAnalysisResult;
  if (data?.legacy_analysis?.score !== undefined) return data.legacy_analysis as MLAnalysisResult;
  if (data?.ml_analysis?.score !== undefined) return data.ml_analysis as MLAnalysisResult;
  return undefined;
}

function isUiReadyAnalysis(value: any): value is UiReadyAnalysis {
  return Boolean(value && (value.analysis_validity || value.task_result || value.feedback_policy));
}

function uiReadyAnalysisToPerformanceResult(
  songId: string,
  analysis: UiReadyAnalysis,
  legacyAnalysis?: MLAnalysisResult,
  fallbackSongTitle: string = "Practice Exercise",
  fallbackArtist: string = "AI Coach",
  fallbackTaskConfig?: TaskConfig
): PerformanceResult {
  const effectiveTaskConfig = analysis.task_config || fallbackTaskConfig;
  const taskResult = analysis.task_result || {};
  const feedbackPolicy = analysis.feedback_policy || {};
  const score = scoreFromTaskResult(taskResult);
  const blocked = blockedTypes(feedbackPolicy);
  const fullSongBlocked = blocked.has("full_song_score") || taskResult.full_song_score === null;
  const displayScore = fullSongBlocked && taskResult.diagnostic_score === null ? 0 : score;
  const notes = safeCoachingNotes(analysis);

  return {
    songId,
    songTitle: legacyAnalysis?.songTitle || fallbackSongTitle,
    artist: legacyAnalysis?.artist || fallbackArtist || formatTaskType(effectiveTaskConfig?.task_type || taskResult.task_type || "AI Coach"),
    overallScore: displayScore,
    intonation: displayScore,
    rhythm: 0,
    timbre: 0,
    dynamics: 0,
    coachingNotes: notes,
    mlAnalysis: legacyAnalysis ? { ...legacyAnalysis, uiReadyAnalysis: analysis } : undefined,
    uiReadyAnalysis: analysis,
    taskConfig: effectiveTaskConfig,
    recordedAt: new Date().toISOString(),
  };
}

function scoreFromTaskResult(taskResult: TaskResult): number {
  const score = taskResult.full_song_score ?? taskResult.diagnostic_score;
  return typeof score === "number" && Number.isFinite(score) ? Math.max(0, Math.min(100, Math.round(score))) : 0;
}

function blockedTypes(policy: FeedbackPolicy): Set<string> {
  return new Set((policy.blocked_feedback || []).map((item) => String(item.type || "").toLowerCase()));
}

function safeCoachingNotes(analysis: UiReadyAnalysis): PerformanceResult["coachingNotes"] {
  const validity = analysis.analysis_validity;
  const taskResult = analysis.task_result;
  const caveats = analysis.feedback_policy?.caveats || [];
  const invalidTypes = new Set(["no_voice_or_noise", "speech_like_or_non_singing", "low_confidence_or_unreliable"]);

  if (validity?.input_type && invalidTypes.has(validity.input_type)) {
    return [
      {
        type: "info",
        category: "Analysis",
        title: invalidTitle(validity.input_type),
        text: taskResult?.summary || "Singing coaching was not generated for this recording.",
      },
    ];
  }

  const notes: PerformanceResult["coachingNotes"] = [
    {
      type: "info",
      category: "Task Result",
      title: formatTaskType(taskResult?.task_type || analysis.task_config?.task_type || "Practice"),
      text: taskResult?.summary || "Task-specific analysis completed.",
    },
  ];
  const categories = analysis.coaching_categories;
  const categoryNotes = [
    { key: "vibrato", title: "Vibrato control", category: categories?.vibrato },
    { key: "slide", title: "Slide control", category: categories?.slide },
  ];
  categoryNotes.forEach((item) => {
    if (!item.category) return;
    const status = item.category.status || "not_enough_evidence";
    const confidence = typeof item.category.confidence === "number" ? item.category.confidence : 0;
    const scoreText = typeof item.category.score === "number" ? ` Score: ${Math.round(item.category.score)}.` : "";
    const caveat = item.category.caveats?.[0];
    if (status === "complete") {
      notes.push({
        type: confidence >= 0.7 ? "success" : "info",
        category: item.title,
        title: confidence >= 0.7 ? "Evidence found" : "Possible evidence found",
        text: `${item.category.recommended_exercise || "Use the detailed coaching card for the next exercise."}${scoreText}`,
      });
    } else if (caveat) {
      notes.push({
        type: "info",
        category: item.title,
        title: "Not enough evidence",
        text: caveat,
      });
    }
  });
  if (caveats[0]) {
    notes.push({
      type: "info",
      category: "Caveat",
      title: "Important Context",
      text: caveats[0],
    });
  }
  return notes;
}

function invalidTitle(inputType: string): string {
  if (inputType === "no_voice_or_noise") return "No analyzable singing detected";
  if (inputType === "speech_like_or_non_singing") return "Speech or non-singing voice detected";
  return "Analysis confidence was too low";
}

function formatTaskType(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function createAnalysisUnavailablePerformanceResult(
  songId: string,
  songTitle: string,
  artist: string,
  error: string,
  taskConfig?: TaskConfig
): PerformanceResult {
  return {
    songId,
    songTitle,
    artist,
    overallScore: 0,
    intonation: 0,
    rhythm: 0,
    timbre: 0,
    dynamics: 0,
    coachingNotes: [
      {
        type: "info",
        category: "Analysis",
        title: "Analysis unavailable",
        text: "We could not analyze this take. Your recording was saved for this session, but no singing score or coaching was generated.",
      },
    ],
    analysisUnavailable: true,
    analysisError: error,
    taskConfig,
    recordedAt: new Date().toISOString(),
  };
}
