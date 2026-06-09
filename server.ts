import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Set up body parsers with limits for processing camera base64 uploads
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ extended: true, limit: "15mb" }));

// Lazy initializer for Google GenAI client
let aiInstance: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI | null {
  if (!aiInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (apiKey && apiKey !== "MY_GEMINI_API_KEY") {
      aiInstance = new GoogleGenAI({
        apiKey: apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });
    }
  }
  return aiInstance;
}

// 1. Health check endpoint
app.get("/api/health", (req, res) => {
  const hasKey = !!process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY";
  res.json({
    status: "ok",
    environment: process.env.NODE_ENV || "development",
    hasApiKey: hasKey
  });
});

// 2. Real-time AI Scan endpoint using Gemini 3.5 Flash (multimodal)
app.post("/api/scan", async (req, res): Promise<any> => {
  try {
    const { image, mimeType } = req.body;
    if (!image) {
      return res.status(400).json({ error: "No image provided for scanning." });
    }

    // Clean base64 string if it has headers
    let base64Data = image;
    let actualMimeType = mimeType || "image/jpeg";
    
    if (image.startsWith("data:")) {
      const parts = image.split(",");
      base64Data = parts[1];
      const mimeMatch = parts[0].match(/data:(.*?);/);
      if (mimeMatch) {
         actualMimeType = mimeMatch[1];
      }
    }

    const ai = getGeminiClient();

    // If API key is not configured, we fall back to a high-quality simulated response 
    // to keep developers unblocked and preserve offline preview capability.
    if (!ai) {
      console.warn("GEMINI_API_KEY is not configured or placeholder detected. Falling back to Simulated Analysis.");
      
      // We simulate a smart result based on typical characteristics
      // (or random distribution of day 7, 10, 14, infertile, dead)
      const mockStages = [
        {
          ageDays: 7,
          status: "Healthy Embryo" as const,
          confidence: 94.2,
          description: "(Simulation: No API Key) Strong heart development. A clear dark spot representing the embryo's eye is visible with beautifully extending spider-like blood vessels branch throughout the upper yolk. The cell size is developing nicely."
        },
        {
          ageDays: 14,
          status: "Healthy Embryo" as const,
          confidence: 98.4,
          description: "(Simulation: No API Key) The embryo occupies most of the space. The air cell at the large end of the egg has increased in volume tremendously and presents a very sharp and straight border. Moving mass is active."
        },
        {
          ageDays: 0,
          status: "Infertile Egg" as const,
          confidence: 99.1,
          description: "(Simulation: No API Key) Perfectly clear orange/yellow egg glow under intense light source. Absolutely no veins, blood spots, or dark embryonic shadow. Light transfers cleanly."
        },
        {
          ageDays: 5,
          status: "Healthy Embryo" as const,
          confidence: 89.7,
          description: "(Simulation: No API Key) Early development visible. A faint but definite dark spot surrounded by subtle red vitelline networks stretching over the yolk surface is noticeable."
        },
        {
          ageDays: 0,
          status: "Dead Embryo" as const,
          confidence: 91.3,
          description: "(Simulation: No API Key) Presence of a thick, fixed blood ring circling around the shell wall. Main embryo mass looks stagnant and collapsed into a light sediment without active branching veins."
        }
      ];

      const chosen = mockStages[Math.floor(Math.random() * mockStages.length)];
      return res.json({
        ...chosen,
        ageWeeks: chosen.ageDays > 0 ? Math.floor(chosen.ageDays / 7) : 0,
        simulated: true,
        message: "API key is unconfigured. Showing high-fidelity offline simulation. To use real-time Gemini Vision, add a valid 'GEMINI_API_KEY' in the Secrets panel."
      });
    }

    // Call the real server-side Gemini 3.5 Flash model
    const imagePart = {
      inlineData: {
        mimeType: actualMimeType,
        data: base64Data,
      },
    };

    const promptText = `
      You are an expert avian veterinarian and commercial poultry incubator master. 
      Analyze this egg candling image (where a bright light illuminates the inside of a fertilized bird egg).
      Determine whether the egg contains a "Healthy Embryo", is an "Infertile Egg" (completely clear inside with no development), or is a "Dead Embryo" (presence of blood rings, dark spots settled with no active veins).
      
      If it is a "Healthy Embryo", estimate the developmental age strictly in DAYS (an integer between 1 and 21). Be as precise as possible based on standard chicken egg candling guides (spider veins radiating, eye size, opaque mass coverage, air cell size size).
      If it is "Infertile Egg" or "Dead Embryo", return the age as 0 days.

      Provide a specialized expert description explaining what can be seen in the image, or what features justify your developmental assessment (such as vascular branching, opaque body contour, air pocket size, or blood rings). Keep it professional, informative, and encouraging.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: {
        parts: [imagePart, { text: promptText }]
      },
      config: {
        systemInstruction: "You are Smart Egg AI, a highly specialized poultry-expert neural network. You output structured candling analysis results.",
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["ageDays", "status", "confidence", "description"],
          properties: {
            ageDays: {
              type: Type.INTEGER,
              description: "Estimated age of the healthy embryo in days (1 to 21). Must be 0 if status is Infertile Egg or Dead Embryo."
            },
            status: {
              type: Type.STRING,
              description: "Status category. Must be one of: 'Healthy Embryo', 'Infertile Egg', 'Dead Embryo'"
            },
            confidence: {
              type: Type.NUMBER,
              description: "Model confidence score as a percentage between 0 and 100. Be realistic and precise (e.g. 96.5)"
            },
            description: {
              type: Type.STRING,
              description: "A highly informative, poultry-expert insight (2 to 4 sentences) detailing the vision features seen in the egg (veins, mass, air spot, ring)."
            }
          }
        }
      }
    });

    const textResponse = response.text;
    if (!textResponse) {
       throw new Error("No response content generated by Gemini.");
    }

    const result = JSON.parse(textResponse.trim());
    
    // Ensure ageWeeks is calculated
    const finalAgeDays = typeof result.ageDays === "number" ? result.ageDays : 0;
    const finalAgeWeeks = finalAgeDays > 0 ? Math.floor(finalAgeDays / 7) : 0;

    res.json({
      ageDays: finalAgeDays,
      ageWeeks: finalAgeWeeks,
      status: result.status || "Healthy Embryo",
      confidence: typeof result.confidence === "number" ? result.confidence : 90.0,
      description: result.description || "Analysis completed successfully.",
      simulated: false
    });

  } catch (error: any) {
    console.error("Gemini scanning API error:", error);
    res.status(500).json({
      error: "AI scanning failed.",
      errorMessage: error.message || "Unknown error during image perception.",
      simulated: true, // Fail-safe to allow user to keep exploring
      ageDays: 7,
      ageWeeks: 1,
      status: "Healthy Embryo",
      confidence: 85.0,
      description: "Fallback analysis due to API timeout or perception limit. Typical Day 7 vascular structures apparent."
    });
  }
});

// Configure Vite or statically serve standard production build
async function setupViteAndListen() {
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
    console.log(`Smart Egg AI Scanner Server listening on http://0.0.0.0:${PORT}`);
  });
}

setupViteAndListen();
