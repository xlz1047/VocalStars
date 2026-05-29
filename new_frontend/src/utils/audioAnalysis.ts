import { PerformanceResult, MLAnalysisResult } from "../types";

/**
 * Utility functions for audio recording, upload, and ML analysis
 */

// MediaRecorder instance for audio capture
let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let recordingStream: MediaStream | null = null;

/**
 * Start recording audio from the user's microphone
 */
export async function startAudioRecording(): Promise<void> {
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      }
    });

    mediaRecorder = new MediaRecorder(recordingStream, {
      mimeType: "audio/webm;codecs=opus",
    });

    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.start();
  } catch (error) {
    console.error("Error starting audio recording:", error);
    throw error;
  }
}

/**
 * Stop recording and return the audio blob
 */
export async function stopAudioRecording(): Promise<Blob> {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder) {
      reject(new Error("No active recording"));
      return;
    }

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      
      // Clean up
      if (recordingStream) {
        recordingStream.getTracks().forEach(track => track.stop());
      }
      mediaRecorder = null;
      recordingStream = null;
      audioChunks = [];

      resolve(audioBlob);
    };

    mediaRecorder.stop();
  });
}

/**
 * Upload audio to backend for ML analysis
 */
export async function analyzeAudioWithML(
  audioBlob: Blob,
  songTitle: string,
  artist: string,
  backendUrl: string = "http://localhost:8000"
): Promise<MLAnalysisResult> {
  try {
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");
    formData.append("song_title", songTitle);
    formData.append("artist", artist);

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

    return result.data;
  } catch (error) {
    console.error("Error analyzing audio:", error);
    throw error;
  }
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
  mlAnalysis: MLAnalysisResult
): PerformanceResult {
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
        category: "Technique",
        title: `${mlAnalysis.technique} Detected`,
        text: `Confidence: ${mlAnalysis.techniqueConfidence}%. Your singing shows characteristics of ${mlAnalysis.technique} technique.`,
      },
      {
        type: "success",
        category: "Pitch",
        title: `${mlAnalysis.pitchAccuracy}% Pitch Accuracy`,
        text: `You maintained excellent pitch control with an average drift of ${Math.abs(mlAnalysis.pitchDrift)} cents. Keep up the precision!`,
      },
      {
        type: "info",
        category: "Breathing",
        title: `${mlAnalysis.breathCount} Breath Points Detected`,
        text: `Good breath pacing with ${mlAnalysis.breathCount} breathing points throughout the recording.`,
      },
    ],
    mlAnalysis,
    recordedAt: new Date().toISOString(),
  };
}
