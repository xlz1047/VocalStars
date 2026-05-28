# VocalStars Product Requirements Document (PRD)

## 1. Project Vision
VocalStars is a high-fidelity singing evaluation and coaching platform designed to provide fair, musical, and meaningful feedback. Unlike traditional karaoke applications that rely on rigid pitch-matching, VocalStars utilizes a research-grounded approach to assess vocal performance across multiple dimensions, allowing for expressive variation and stylistic nuance.

---

## 2. Target Audience
- **Aspiring Singers:** Individuals looking to improve their vocal technique with objective data.
- **Vocal Coaches:** Professionals seeking a tool to supplement student training with detailed analytical reports.
- **Hobbyists:** Users who want a more sophisticated and rewarding singing experience than standard karaoke apps.

---

## 3. Core Experience & User Journey
1. **Discovery:** User explores curated songs and daily vocal fitness warmups on a personalized dashboard.
2. **Recording:** User enters a focused studio interface to sing along with a reference track, receiving real-time "perfect pitch" visual reinforcement.
3. **Evaluation:** The system analyzes the performance using the PESnQ (Perceptual Evaluation of Singing Quality) model.
4. **Analysis:** User reviews a multi-dimensional score breakdown (Intonation, Rhythm, Timbre, Dynamics) with scientific coaching notes.
5. **Review & Practice:** User replays their recording with synchronized visual overlays to identify specific growth areas and accesses recommended exercises.

---

## 4. Functional Requirements

### 4.1 Dashboard (Home)
- **Personalized Content:** Hero section featuring "Song of the Day."
- **Daily Routine:** Curated vocal fitness and warmups (Breath Support, Pitch & Resonance, Agility & Range).
- **Library Access:** Categorized song lists (Trending, New for You) with difficulty badges (Easy, Medium, Hard).
- **Navigation:** Global full-width header with search, notifications, and profile. Collapsible sidebar for secondary navigation (Dashboard, Practice, Exercise, History, Library).

### 4.2 Recording Interface (The Studio)
- **Visual Feedback:** Real-time dual-waveform display showing singer's pitch vs. reference.
- **"Perfect Pitch" State:** Neon pulse animations and glow effects when the singer hits target notes within acceptable tolerance.
- **Controls:** Session management (Play/Pause, Stop, Restart), volume/monitor controls, and BPM display.
- **Lyrics Display:** High-contrast, scrolling lyrics synced to the musical timeline.

### 4.3 Advanced Performance Analysis (Scoring)
- **PESnQ Rubric:** A weighted multi-dimensional score:
    - **Intonation (40%):** Stability, tonality modulations, and key retention.
    - **Temporal Precision (25%):** Rhythmic alignment and millisecond-accurate note onset tracking.
    - **Vocal Timbre (20%):** Spectral resonance, vowel clarity, and breathiness detection.
    - **Musical Dynamics (15%):** Expressivity, RMS energy stability, and volume control.
- **Scientific Coaching Notes:** Automated feedback using professional terminology (e.g., subglottic pressure, spectral slope, air leakage).
- **Breath Support Integration:** Evaluation of aerodynamic control via Harmonics-to-Noise Ratio and spectral slope analysis.

### 4.4 Playback Review
- **Synchronized Waveform:** Replay the performance with the target pitch line and recorded vocal path visible.
- **Interactive Feedback:** Coaching notes mapped to specific timestamps in the recording.
- **Analysis Toggle:** Ability to view "Full Analysis" or a simplified summary.

---

## 5. Technical Requirements & Design System

### 5.1 Design System (Sonic Stage)
- **Theme:** Dark Mode (#11131c) with vibrant neon accents (Primary: #ff2d78).
- **Typography:** Syne (Bold, modern, high-tech aesthetic).
- **Visual Language:** Fluid, expressive visualizations (glows, pulses, gradients) rather than rigid blocks.

### 5.2 Audio Processing
- **Signal Extraction:** Mel-Frequency Pitch Trajectory for intonation analysis.
- **Spectral Analysis:** Spectral Center of Gravity and Formant tracking for timbre assessment.
- **Energy Tracking:** RMS energy calculation for dynamics and breath stability.

---

## 6. Success Metrics
- **Accuracy Perception:** User feedback on how "fair" and "musical" the scoring feels compared to their own perception.
- **Improvement Rate:** Tracking user score progression over time across the four pillars.
- **Engagement:** Frequency of daily warmup completions and recording takes per session.