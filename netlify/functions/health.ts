import type { Handler, HandlerEvent, HandlerContext } from '@netlify/functions';
import { GoogleGenAI } from '@google/genai';

let ai: GoogleGenAI | null = null;
const geminiKey = process.env.GEMINI_API_KEY;
if (geminiKey && geminiKey !== 'MY_GEMINI_API_KEY') {
  try {
    ai = new GoogleGenAI({ apiKey: geminiKey });
  } catch {
    // fallback
  }
}

export const handler: Handler = async (event: HandlerEvent, context: HandlerContext) => {
  return {
    statusCode: 200,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      status: 'ok',
      hasGemini: Boolean(ai),
      modelBackend: 'Python Deep Learning Heuristics & Deep Neural Frameworks (EfficientNet / Xception / ResNet)',
      timestamp: new Date().toISOString(),
      environment: 'Netlify Serverless'
    })
  };
};
