import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT || 3000);

app.use(express.json());

const goldenExamplesPath = path.resolve(process.cwd(), "..", "reports", "golden_ui_examples");
app.use("/golden-ui-examples", express.static(goldenExamplesPath));

// Initialize Google Gen AI
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      "User-Agent": "build",
    }
  }
});

// Gemini availability check — called once on mount by ResultsView
app.get("/api/coaching-status", (_req, res) => {
  res.json({ available: !!process.env.GEMINI_API_KEY });
});

// AI Professional Coaching endpoint proxy
app.post("/api/coaching-feedback", async (req, res) => {
  const { songTitle, artist, score, intonation, rhythm, timbre, dynamics } = req.body;

  if (!process.env.GEMINI_API_KEY) {
    return res.status(400).json({ 
      error: "GEMINI_API_KEY is missing in backend environment variables.",
      coachingNotes: [] 
    });
  }

  try {
    const prompt = `You are a world-class professional vocal bio-mechanic and vocal coach in a high-end recording studio.
Analyze this vocal performance take statistics and output 3 high-fidelity scientific coaching recommendations:
- Song: "${songTitle}" by ${artist}
- Weighted Performance PESnQ Score: ${score} out of 100
- Intonation precision: ${intonation}%
- Rhythm precision: ${rhythm}%
- Timbre overtones: ${timbre}%
- Volume Dynamics control: ${dynamics}%

Structure your response as a JSON array of 3 coaching notes, each having:
1. "category": A 2-3 word string indicating the technical domain (e.g., "Spectral Resonance", "Vocal Tract Geometry", "Subglottic Breath", "Vibrato Modulation").
2. "type": One of "success", "warning", or "info". Always include precisely 1 "success" (highlighting what they did best), 1 "warning" (the most critical vocal adjustment needed), and 1 "info" (a technical tip to try).
3. "title": A concise, encouraging, human title for the recommendation.
4. "text": A 2-sentence highly technical, coaching feedback tip containing professional vocabulary (like subglottic pressure, vocal sirens, formant region, larynx height, pharyngeal resonance).

Analyze objectively and match the vocabulary to their ratings.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["coachingNotes"],
          properties: {
            coachingNotes: {
              type: Type.ARRAY,
              description: "Array of exactly 3 coaching advice items based on the vocal metrics.",
              items: {
                type: Type.OBJECT,
                required: ["type", "category", "title", "text"],
                properties: {
                  type: {
                    type: Type.STRING,
                    description: "One of success, warning, or info."
                  },
                  category: {
                    type: Type.STRING,
                    description: "The technological or performance category."
                  },
                  title: {
                    type: Type.STRING,
                    description: "A short professional title."
                  },
                  text: {
                    type: Type.STRING,
                    description: "A 2-sentence feedback or correction text using vocal coaching terms."
                  }
                }
              }
            }
          }
        }
      }
    });

    const info = JSON.parse(response.text || "{}");
    res.json(info);
  } catch (err: any) {
    console.error("Gemini API call failed on server:", err);
    res.status(500).json({ 
      error: err.message,
      coachingNotes: [] 
    });
  }
});

// Configure Vite middleware in development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`VocalStars Node server listening at http://0.0.0.0:${PORT}`);
  });
}

startServer();
