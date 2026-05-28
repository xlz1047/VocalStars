# VocalStars New Frontend

Modern, responsive frontend for VocalStars vocal coaching application built with **React**, **Vite**, **TypeScript**, and **Tailwind CSS**.

## Features

- 🎤 **Real-time Audio Recording** - Capture vocal performances with browser microphone
- 📊 **Live Pitch Tracking** - Visual feedback during recording sessions
- 🧠 **ML-Powered Analysis** - Automatic vocal analysis using ml_new models:
  - Pitch accuracy and stability
  - Breath detection and phrasing
  - Vocal technique classification
  - Voice quality metrics (HNR, jitter, shimmer)
  - Note segmentation and vibrato detection
- 💬 **Coaching Feedback** - AI-generated personalized coaching recommendations
- 📈 **Performance Tracking** - History of vocal takes and progress
- 🎵 **Song Library** - Browse and select songs for practice
- 🔥 **Modern UI** - Responsive design with smooth animations

## Tech Stack

- **Frontend Framework**: React 19 with Vite 6
- **Language**: TypeScript 5.8
- **Styling**: Tailwind CSS 4 + custom design tokens
- **Build Tool**: Vite with esbuild
- **Animation**: Motion (framer-motion alternative)
- **Icons**: Lucide React
- **Backend Communication**: Fetch API

## Prerequisites

- **Node.js** 18+ with npm
- **Backend API** running on `http://localhost:8000` (FastAPI)
- **Optional**: GPU for faster ML inference

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Copy the example environment file and update:

```bash
cp .env.local.example .env.local
```

Edit `.env.local`:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_FRONTEND_URL=http://localhost:3000
VITE_API_URL=http://localhost:8000
```

### 3. Start Development Server

```bash
npm run dev
```

App will be available at: **`http://localhost:3000`**

### 4. Build for Production

```bash
npm run build
npm run start
```

## Project Structure

```
new_frontend/
├── src/
│   ├── components/          # React components
│   │   ├── App.tsx         # Main app component
│   │   ├── DashboardView.tsx   # Song selection & dashboard
│   │   ├── StudioView.tsx      # Recording interface
│   │   ├── ResultsView.tsx     # Analysis results display
│   │   ├── ReviewView.tsx      # Playback & scrubbing
│   │   ├── Sidebar.tsx         # Navigation sidebar
│   │   └── TopAppBar.tsx       # Header
│   ├── utils/
│   │   ├── audioAnalysis.ts    # Audio recording & ML API
│   │   ├── pitchDetector.ts    # Live pitch detection
│   │   └── musicDb.ts          # Song catalog
│   ├── types.ts            # TypeScript interfaces
│   ├── index.css           # Global styles
│   └── main.tsx            # App entry point
├── server.ts               # Express server for production
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
└── README.md
```

## Key Components

### StudioView (`src/components/StudioView.tsx`)

Recording and live feedback interface:

- Real-time pitch visualization from microphone
- Lyrics teleprompter with automatic scrolling
- Volume and playback speed controls
- Audio recording in background
- Session completion with ML analysis

### ResultsView (`src/components/ResultsView.tsx`)

Comprehensive analysis display:

- Overall performance score (0-100)
- Traditional metrics (intonation, rhythm, timbre, dynamics)
- ML analysis results:
  - Pitch accuracy and drift
  - Breath/onset detection counts
  - Detected technique with confidence
  - Voice quality analysis
  - Individual note analysis
  - Recommended exercises

### DashboardView (`src/components/DashboardView.tsx`)

Main interface for song selection and warmup exercises

### ReviewView (`src/components/ReviewView.tsx`)

Pitch contour playback and scrubbing

## API Integration

### ML Analysis Endpoint

The frontend communicates with the backend's ML analysis endpoint:

```
POST /api/audio/analyze-with-ml

Request:
- file: WebM audio blob
- song_title: string
- artist: string

Response:
{
  status: "success",
  data: {
    score: 85,
    summary: "...",
    issues: [...],
    exercises: [...],
    pitchAccuracy: 92.5,
    // ... (full ML analysis data)
  }
}
```

See [../docs/new-frontend-ml-integration.md](../docs/new-frontend-ml-integration.md) for full API documentation.

## Audio Recording

The app uses the browser's **MediaRecorder API** for audio capture:

- Audio format: WebM (Opus codec)
- Sample rate: Auto-detected (typically 48kHz)
- Mono or stereo: Auto
- Transmitted to backend for ML analysis

## Live Pitch Detection

Pitch is detected from microphone audio using FFT analysis:

- FFT size: 2048 bins
- Updates: ~60Hz
- Algorithm: YIN algorithm approximation
- Display: Real-time waveform overlay on reference melody

## Performance Considerations

- Audio encoding: ~2-5 seconds for 3-minute recording
- ML inference: ~3-10 seconds backend processing (CPU)
- UI animations: 60 FPS smooth
- Memory: ~200MB typical usage

## Development

### Available Scripts

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run lint

# Build for production
npm run build

# Start production server
npm run start

# Clean build artifacts
npm run clean
```

### Environment Variables

Create `.env.local` to override defaults:

```env
# Backend API endpoint
REACT_APP_API_URL=http://localhost:8000
REACT_APP_FRONTEND_URL=http://localhost:3000

# Feature flags
REACT_APP_ENABLE_ML_ANALYSIS=true
REACT_APP_ENABLE_GEMINI_COACHING=true
```

### Adding Components

1. Create component in `src/components/`
2. Add TypeScript interfaces to `src/types.ts`
3. Import and use in appropriate parent component

### Styling

Uses **Tailwind CSS** with custom design tokens:

```typescript
// Global design tokens in tailwind.config.ts
const colors = {
  primary: '#ffb1c0',
  secondary: '#ff69b4',
  tertiary: '#3cddc7',
  // ... (see tailwind config)
};
```

## Testing

Currently uses browser dev tools for testing. Future enhancements:

- Unit tests with Vitest
- E2E tests with Playwright
- Component testing with React Testing Library

## Troubleshooting

### Microphone Permission Denied

- Check browser microphone permissions
- Ensure site is HTTPS in production
- Reload page and retry

### Audio Recording Failed

- Check console for specific errors
- Verify MediaRecorder API support
- Ensure sufficient disk space

### Backend Connection Error

- Verify backend is running on configured URL
- Check CORS headers from backend
- Ensure API endpoint exists

### Performance Issues

- Reduce reference pitch array complexity
- Close other browser tabs
- Check browser dev tools Performance tab
- Consider GPU acceleration with CUDA

## Deployment

### Production Build

```bash
npm run build
npm run start
```

### Docker Deployment

See [docker-compose.yml](../docker-compose.yml) for containerized deployment.

### Environment for Production

```env
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_FRONTEND_URL=https://yourdomain.com
NODE_ENV=production
```

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and test locally
3. Submit pull request with description

## Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [TypeScript](https://www.typescriptlang.org/)
- [ML Integration Guide](../docs/new-frontend-ml-integration.md)
- [Backend API](../backend/README.md)

## License

MIT

## Support

For issues or questions, refer to:
1. [ML Integration Documentation](../docs/new-frontend-ml-integration.md)
2. Project issues tracker
3. Backend team for API issues

---

**Built with ❤️ for beginner vocal coaches**

