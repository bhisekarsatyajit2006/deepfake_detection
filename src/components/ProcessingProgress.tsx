import React from 'react';
import { Loader2, Video, ScanFace, Cpu, CheckCircle2, Terminal, Activity } from 'lucide-react';
import { ModelArchId } from '../types';

interface ProcessingProgressProps {
  currentFrame: number;
  totalFrames: number;
  currentStage: string;
  selectedModel: ModelArchId;
  previewImageUrl?: string;
  previewCropUrl?: string;
}

export const ProcessingProgress: React.FC<ProcessingProgressProps> = ({
  currentFrame,
  totalFrames,
  currentStage,
  selectedModel,
  previewImageUrl,
  previewCropUrl
}) => {
  const percentage = totalFrames > 0 ? Math.min(100, Math.round((currentFrame / totalFrames) * 100)) : 0;

  return (
    <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 sm:p-8 shadow-2xl font-mono text-neutral-200">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Terminal Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-neutral-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-neutral-950 border border-neutral-700 text-rose-400 flex items-center justify-center animate-spin">
              <Loader2 className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-rose-400 font-bold tracking-widest uppercase">
                  ACTIVE INFERENCE PIPELINE
                </span>
                <span className="text-neutral-600">&bull;</span>
                <span className="text-xs text-neutral-400 uppercase">{selectedModel}</span>
              </div>
              <h3 className="text-base sm:text-lg font-bold text-white mt-0.5">
                {currentStage || 'Executing multi-frame extraction...'}
              </h3>
            </div>
          </div>

          <div className="text-right sm:border-l sm:border-neutral-800 sm:pl-5">
            <div className="text-3xl font-black text-rose-400 font-mono tracking-tight">
              {percentage}%
            </div>
            <div className="text-xs text-neutral-400 font-mono">
              Frame <strong className="text-white">{currentFrame}</strong> of {totalFrames}
            </div>
          </div>
        </div>

        {/* Progress Bar & Telemetry */}
        <div className="space-y-2">
          <div className="w-full h-3 bg-neutral-950 rounded-full overflow-hidden p-0.5 border border-neutral-800">
            <div
              className="h-full bg-gradient-to-r from-rose-600 via-amber-500 to-emerald-500 rounded-full transition-all duration-150 ease-out shadow-[0_0_10px_rgba(244,63,94,0.5)]"
              style={{ width: `${percentage}%` }}
            />
          </div>
          <div className="flex justify-between text-[11px] text-neutral-500">
            <span>STAGE: cv2.VideoCapture &bull; Tensor Prep &bull; Logits Evaluation</span>
            <span>THROUGHPUT: ~32 FPS (simulated)</span>
          </div>
        </div>

        {/* Live Visual Inspection (Frame + Detected Face Crop) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          {/* Main frame preview */}
          <div className="bg-neutral-950 rounded-xl p-3 border border-neutral-800 flex flex-col items-center justify-center min-h-[190px] relative overflow-hidden">
            <div className="absolute top-2 left-2 z-10 text-[10px] uppercase font-mono tracking-wider bg-black/80 px-2 py-0.5 rounded text-neutral-300 border border-neutral-700">
              DECODED RAW FRAME #{currentFrame}
            </div>
            {previewImageUrl ? (
              <img
                src={previewImageUrl}
                alt="Analyzed frame"
                className="max-h-[170px] w-auto object-contain rounded border border-neutral-800"
              />
            ) : (
              <div className="flex flex-col items-center gap-2 text-neutral-500">
                <Video className="w-8 h-8 animate-pulse text-neutral-600" />
                <span className="text-xs font-mono">Extracting Video Stream...</span>
              </div>
            )}
          </div>

          {/* Detected Face Crop */}
          <div className="bg-neutral-950 rounded-xl p-3 border border-neutral-800 flex flex-col items-center justify-center min-h-[190px] relative overflow-hidden">
            <div className="absolute top-2 left-2 z-10 text-[10px] uppercase font-mono tracking-wider bg-black/80 px-2 py-0.5 rounded text-neutral-300 border border-neutral-700">
              NORMALIZED TENSOR [224x224x3]
            </div>
            {previewCropUrl ? (
              <div className="relative">
                <img
                  src={previewCropUrl}
                  alt="Cropped face"
                  className="w-32 h-32 object-cover rounded-lg border border-neutral-700 shadow-xl"
                />
                <div className="absolute -bottom-2 -right-2 px-2 py-0.5 rounded bg-rose-600 text-white text-[10px] font-mono font-bold shadow-md">
                  FACE ROI
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2 text-neutral-500">
                <ScanFace className="w-8 h-8 animate-pulse text-neutral-600" />
                <span className="text-xs font-mono">Detecting Facial Landmarks...</span>
              </div>
            )}
          </div>
        </div>

        {/* Live Forensic Terminal Log Lines */}
        <div className="bg-black/90 rounded-xl p-4 border border-neutral-800 text-[11px] text-neutral-400 space-y-1">
          <div className="flex items-center gap-2 text-neutral-300 font-bold border-b border-neutral-800 pb-2 mb-2">
            <Terminal className="w-3.5 h-3.5 text-rose-400" />
            <span>INFERENCE CONSOLE LOG</span>
          </div>
          <div className="text-emerald-400">
            [+] Stream codec: AVC/H.264 &bull; Initializing frame buffer pipeline
          </div>
          <div>
            [+] Frame #{currentFrame.toString().padStart(3, '0')}: Face localized at box &bull; Normalizing RGB values [0, 1] with ImageNet std/mean
          </div>
          <div className="text-sky-400">
            [+] Forward pass: model(tensor) &rarr; Softmax logits calculated &bull; Grad-CAM backprop
          </div>
        </div>
      </div>
    </div>
  );
};
