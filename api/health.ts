import type { VercelRequest, VercelResponse } from '@vercel/node';
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

export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  res.json({
    status: 'ok',
    hasGemini: Boolean(ai),
    modelBackend: 'Python Deep Learning Heuristics & Deep Neural Frameworks (EfficientNet / Xception / ResNet)',
    timestamp: new Date().toISOString(),
    environment: 'Vercel Serverless'
  });
}
