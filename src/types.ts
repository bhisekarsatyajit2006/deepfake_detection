/**
 * Types and interfaces for the Deepfake Detection System
 */

export type MediaType = 'video' | 'image';

export type PredictionLabel = 'REAL' | 'DEEPFAKE';

export type ModelArchId = 'efficientnet-b4' | 'xception' | 'resnet-50';

export interface BoundingBox {
  x: number; // 0 to 1 normalized or pixel
  y: number;
  width: number;
  height: number;
}

export interface FrameAnalysis {
  frameIndex: number;
  timestamp: number; // seconds
  timestampFormatted: string; // e.g. "00:02.40"
  prediction: PredictionLabel;
  confidence: number; // 0 to 100 percentage
  fakeScore: number; // 0 to 1
  realScore: number; // 0 to 1
  faceDetected: boolean;
  boundingBox?: BoundingBox;
  faceCropUrl?: string;
  frameDataUrl?: string;
  heatmapDataUrl?: string;
  fftSpectrumDataUrl?: string;
  landmarksDataUrl?: string;
  artifacts: {
    boundaryBlendingScore: number; // 0-100
    frequencyDissonance: number; // 0-100 (DCT high-freq noise)
    eyeSymmetryScore: number; // 0-100
    textureConsistency: number; // 0-100
  };
}

export interface SingleCollageSideAnalysis {
  side: 'left' | 'right';
  title: string; // e.g. "Photo A (Left)"
  prediction: PredictionLabel;
  confidence: number;
  fakeScore: number;
  realScore: number;
  imageUrl: string;
  faceCropUrl?: string;
  heatmapUrl?: string;
  fftSpectrumUrl?: string;
  landmarksUrl?: string;
  boundingBox?: BoundingBox;
  artifacts: {
    boundaryBlendingScore: number;
    frequencyDissonance: number;
    eyeSymmetryScore: number;
    textureConsistency: number;
  };
  forensicPoints: string[];
}

export interface CollageComparisonResult {
  isCollage: boolean;
  whichIsFake: 'LEFT' | 'RIGHT' | 'BOTH' | 'NEITHER';
  verdictSentence: string;
  leftAnalysis: SingleCollageSideAnalysis;
  rightAnalysis: SingleCollageSideAnalysis;
  differentialFindings: string[];
  fullCollageUrl: string;
}

export interface DetectionResult {
  mediaType: MediaType;
  fileName: string;
  fileSizeFormatted: string;
  duration?: number; // seconds for video
  dimensions: {
    width: number;
    height: number;
  };
  overallPrediction: PredictionLabel;
  modelConfidence: number; // e.g. 93.4
  framesAnalyzed: number;
  realFramesCount: number;
  realFramesPercentage: number;
  fakeFramesCount: number;
  fakeFramesPercentage: number;
  selectedModel: ModelArchId;
  analyzedFrames: FrameAnalysis[];
  executionTimeMs: number;
  samplingInterval: number; // e.g. every 10 frames
  forensicInsights?: string[];
  sha256Checksum?: string;
  codecInfo?: string;
  collageComparison?: CollageComparisonResult;
  biometrics?: {
    ppgPulseDetected: boolean;
    blinkCadencePerMin: number;
    landmarkJitterScore: number;
    bloodVolumePulseRate: number;
  };
}

export interface ModelBenchmark {
  id: ModelArchId;
  name: string;
  architecture: string;
  parameters: string;
  inputResolution: string;
  trainingDataset: string;
  ffppAucRoc: number; // FaceForensics++ c23
  celebDfAucRoc: number; // Celeb-DF v2
  accuracy: number;
  f1Score: number;
  inferenceLatencyMs: number;
  description: string;
  strengths: string[];
}

export interface DatasetSplitMetrics {
  name: string;
  totalVideos: number;
  totalFrames: number;
  trainSplit: {
    videos: number;
    percentage: number;
    subjects: string;
  };
  valSplit: {
    videos: number;
    percentage: number;
    subjects: string;
  };
  testSplit: {
    videos: number;
    percentage: number;
    subjects: string;
  };
  zeroLeakageProtocol: string;
}
