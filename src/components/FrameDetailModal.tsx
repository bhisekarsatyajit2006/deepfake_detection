import React, { useState, useEffect } from 'react';
import { FrameAnalysis } from '../types';
import {
  X,
  ChevronLeft,
  ChevronRight,
  ScanFace,
  Flame,
  Activity,
  Crosshair,
  ShieldAlert,
  ShieldCheck,
  Maximize2
} from 'lucide-react';

interface FrameDetailModalProps {
  frame: FrameAnalysis;
  allFrames: FrameAnalysis[];
  onClose: () => void;
  onNavigate: (frame: FrameAnalysis) => void;
}

export const FrameDetailModal: React.FC<FrameDetailModalProps> = ({
  frame,
  allFrames,
  onClose,
  onNavigate
}) => {
  const [viewMode, setViewMode] = useState<'crop' | 'heatmap' | 'fft' | 'mesh' | 'raw'>('crop');
  const isFake = frame.prediction === 'DEEPFAKE';

  const currentIndex = allFrames.findIndex((f) => f.frameIndex === frame.frameIndex);
  const prevFrame = currentIndex > 0 ? allFrames[currentIndex - 1] : null;
  const nextFrame = currentIndex < allFrames.length - 1 ? allFrames[currentIndex + 1] : null;

  // Keyboard navigation support
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && prevFrame) {
        onNavigate(prevFrame);
      } else if (e.key === 'ArrowRight' && nextFrame) {
        onNavigate(nextFrame);
      } else if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [prevFrame, nextFrame, onNavigate, onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md font-mono text-neutral-200">
      <div className="bg-neutral-900 rounded-2xl max-w-4xl w-full max-h-[94vh] overflow-y-auto shadow-2xl border border-neutral-800">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-neutral-800 sticky top-0 bg-neutral-950/95 backdrop-blur z-10">
          <div className="flex items-center gap-3">
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                isFake
                  ? 'bg-rose-950 text-rose-300 border border-rose-700'
                  : 'bg-emerald-950 text-emerald-300 border border-emerald-700'
              }`}
            >
              {frame.prediction} ({frame.confidence}%)
            </span>
            <span className="text-xs text-neutral-400">
              SEQUENCE FRAME #{frame.frameIndex.toString().padStart(3, '0')} &bull; TIMECODE: {frame.timestampFormatted}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => prevFrame && onNavigate(prevFrame)}
              disabled={!prevFrame}
              className="p-1.5 rounded-lg border border-neutral-700 text-neutral-300 hover:bg-neutral-800 disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
              title="Previous frame (Left arrow)"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => nextFrame && onNavigate(nextFrame)}
              disabled={!nextFrame}
              className="p-1.5 rounded-lg border border-neutral-700 text-neutral-300 hover:bg-neutral-800 disabled:opacity-30 disabled:pointer-events-none cursor-pointer"
              title="Next frame (Right arrow)"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg border border-neutral-700 text-neutral-400 hover:text-white hover:bg-neutral-800 ml-2 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6">
          {/* View Layer Mode Selector */}
          <div className="flex justify-center">
            <div className="inline-flex rounded-xl bg-neutral-950 p-1 border border-neutral-800 text-xs font-semibold gap-1">
              <button
                onClick={() => setViewMode('crop')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  viewMode === 'crop'
                    ? 'bg-neutral-800 text-white font-bold shadow-xs'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <ScanFace className="w-3.5 h-3.5 text-rose-400" />
                <span>Face ROI (224x224)</span>
              </button>

              <button
                onClick={() => setViewMode('heatmap')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  viewMode === 'heatmap'
                    ? 'bg-rose-950 text-rose-300 border border-rose-800 font-bold shadow-xs'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <Flame className="w-3.5 h-3.5 text-rose-500" />
                <span>Grad-CAM</span>
              </button>

              <button
                onClick={() => setViewMode('fft')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  viewMode === 'fft'
                    ? 'bg-sky-950 text-sky-300 border border-sky-800 font-bold shadow-xs'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <Activity className="w-3.5 h-3.5 text-sky-400" />
                <span>2D FFT Spectrum</span>
              </button>

              <button
                onClick={() => setViewMode('mesh')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  viewMode === 'mesh'
                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold shadow-xs'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
                <span>68-Pt Mesh</span>
              </button>

              <button
                onClick={() => setViewMode('raw')}
                className={`px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 ${
                  viewMode === 'raw'
                    ? 'bg-neutral-800 text-white font-bold shadow-xs'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                <span>Full Frame</span>
              </button>
            </div>
          </div>

          {/* Visual Display Stage */}
          <div className="bg-black rounded-2xl p-4 flex items-center justify-center min-h-[300px] border border-neutral-800 relative">
            {viewMode === 'crop' && (
              <img
                src={frame.faceCropUrl || frame.frameDataUrl}
                alt="Cropped Face"
                className="max-h-[280px] max-w-full rounded-xl object-contain shadow-2xl border border-neutral-800"
              />
            )}
            {viewMode === 'heatmap' && (
              <img
                src={frame.heatmapDataUrl || frame.faceCropUrl}
                alt="Grad-CAM Heatmap"
                className="max-h-[280px] max-w-full rounded-xl object-contain shadow-2xl border border-neutral-800"
              />
            )}
            {viewMode === 'fft' && (
              <img
                src={frame.fftSpectrumDataUrl || frame.faceCropUrl}
                alt="2D FFT Magnitude"
                className="max-h-[280px] max-w-full rounded-xl object-contain shadow-2xl border border-neutral-800"
              />
            )}
            {viewMode === 'mesh' && (
              <img
                src={frame.landmarksDataUrl || frame.faceCropUrl}
                alt="68-Point Mesh"
                className="max-h-[280px] max-w-full rounded-xl object-contain shadow-2xl border border-neutral-800"
              />
            )}
            {viewMode === 'raw' && (
              <div className="relative">
                <img
                  src={frame.frameDataUrl}
                  alt="Full Frame"
                  className="max-h-[280px] max-w-full rounded-xl object-contain"
                />
                {frame.boundingBox && (
                  <div
                    className={`absolute border-2 pointer-events-none ${
                      isFake
                        ? 'border-rose-500 bg-rose-500/15 shadow-[0_0_15px_rgba(244,63,94,0.4)]'
                        : 'border-emerald-500 bg-emerald-500/15 shadow-[0_0_15px_rgba(16,185,129,0.4)]'
                    }`}
                    style={{
                      left: `${(frame.boundingBox.x / 640) * 100}%`,
                      top: `${(frame.boundingBox.y / 360) * 100}%`,
                      width: `${(frame.boundingBox.width / 640) * 100}%`,
                      height: `${(frame.boundingBox.height / 360) * 100}%`
                    }}
                  >
                    <span className="absolute -top-5 left-0 px-1 py-0.2 rounded bg-neutral-900 text-white font-mono text-[9px]">
                      FACE 01 [{frame.prediction}]
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Forensic Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800">
              <span className="text-[11px] text-neutral-400 block mb-1">
                Boundary Seam Discontinuity
              </span>
              <span className="text-xl font-bold font-mono text-white">
                {frame.artifacts.boundaryBlendingScore}%
              </span>
              <div className="w-full bg-neutral-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-rose-500 h-full rounded-full"
                  style={{ width: `${frame.artifacts.boundaryBlendingScore}%` }}
                />
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800">
              <span className="text-[11px] text-neutral-400 block mb-1">
                High-Freq FFT Noise
              </span>
              <span className="text-xl font-bold font-mono text-white">
                {frame.artifacts.frequencyDissonance}%
              </span>
              <div className="w-full bg-neutral-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-sky-500 h-full rounded-full"
                  style={{ width: `${frame.artifacts.frequencyDissonance}%` }}
                />
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800">
              <span className="text-[11px] text-neutral-400 block mb-1">
                Corneal Specular Symmetry
              </span>
              <span className="text-xl font-bold font-mono text-white">
                {frame.artifacts.eyeSymmetryScore}%
              </span>
              <div className="w-full bg-neutral-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-emerald-500 h-full rounded-full"
                  style={{ width: `${frame.artifacts.eyeSymmetryScore}%` }}
                />
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-neutral-950 border border-neutral-800">
              <span className="text-[11px] text-neutral-400 block mb-1">
                Biological Texture Porosity
              </span>
              <span className="text-xl font-bold font-mono text-white">
                {frame.artifacts.textureConsistency}%
              </span>
              <div className="w-full bg-neutral-800 h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-amber-500 h-full rounded-full"
                  style={{ width: `${frame.artifacts.textureConsistency}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
