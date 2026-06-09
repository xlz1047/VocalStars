/**
 * A simple autocorrelation-based pitch detection algorithm for live vocal feedback.
 * Highly responsive and runs efficiently inside RequestAnimationFrame.
 */
export function detectPitch(buffer: Float32Array, sampleRate: number): number {
  // Check the signal level to ensure it is not silent background noise
  let rawSum = 0;
  for (let i = 0; i < buffer.length; i++) {
    rawSum += buffer[i] * buffer[i];
  }
  const rms = Math.sqrt(rawSum / buffer.length);
  if (rms < 0.01) {
    return -1; // Root-mean-square is too low (silence)
  }

  // Trim silence from the start and end of the buffer
  let r1 = 0;
  let r2 = buffer.length - 1;
  const thres = 0.002;
  for (let i = 0; i < buffer.length / 2; i++) {
    if (Math.abs(buffer[i]) < thres) {
      r1 = i;
    } else {
      break;
    }
  }
  for (let i = buffer.length - 1; i >= buffer.length / 2; i--) {
    if (Math.abs(buffer[i]) < thres) {
      r2 = i;
    } else {
      break;
    }
  }

  const trimmed = buffer.subarray(r1, r2);
  if (trimmed.length < 64) {
    return -1; // Not enough signal
  }

  // Auto-correlation
  const size = trimmed.length;
  const c = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size - i; j++) {
      c[i] += trimmed[j] * trimmed[j + i];
    }
  }

  // Find the first peak
  let d = 0;
  while (d < size - 1 && c[d] > c[d + 1]) {
    d++;
  }

  let maxval = -1;
  let maxpos = -1;
  for (let i = d; i < size - 1; i++) {
    if (c[i] > c[i - 1] && c[i] > c[i + 1]) {
      if (c[i] > maxval) {
        maxval = c[i];
        maxpos = i;
      }
    }
  }

  if (maxpos !== -1) {
    const frequency = sampleRate / maxpos;
    // Human vocal range limits: 60Hz to 2000Hz (C1 to C7)
    if (frequency > 60 && frequency < 2000) {
      return frequency;
    }
  }

  return -1;
}

/**
 * Helper to convert frequency in Hz to a standardized height offset (0 to 100) on our SVG canvas
 */
export function frequencyToHeight(frequency: number): number {
  if (frequency <= 0) return 50; // default centered
  // Let's use a logarithmic scale based on C2 (65Hz) to C6 (1046Hz)
  const minFreq = 65;
  const maxFreq = 1046;
  const val = Math.log2(frequency / minFreq) / Math.log2(maxFreq / minFreq);
  // Cap between 0 and 1, multiply by 100 and invert so higher frequency is higher visually (lower SVG y)
  const clamped = Math.min(Math.max(val, 0), 1);
  return Math.round(100 - clamped * 80 - 10); // leave 10px padding top/bottom
}
