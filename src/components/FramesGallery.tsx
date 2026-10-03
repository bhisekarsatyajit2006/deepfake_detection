import React, { useState } from 'react';
import { FrameAnalysis } from '../types';
import { ScanFace, Eye, Flame, Activity, Crosshair, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface FramesGalleryProps {
  frames: FrameAnalysis[];
  onSelectFrame: (frame: FrameAnalysis) => void;
  activeFrameIndex?: number;
}

export const FramesGallery: React.FC<FramesGalleryProps> = ({
  frames,
  onSelectFrame,
  activeFrameIndex
}) => {
  const [filter, setFilter] = useState<'all' | 'fake' | 'real' | 'top'>('all');
  const [viewLayer, setViewLayer] = useState<'crop' | 'heatmap' | 'fft' | 'mesh'>('crop');

  const filteredFrames = frames.filter((f) => {
    if (filter === 'fake') return f.prediction === 'DEEPFAKE';
    if (filter === 'real') return f.prediction === 'REAL';
    if (filter === 'top') return f.fakeScore > 0.85 || f.realScore > 0.95;
    return true;
  });

  return (
    <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 shadow-xl space-y-6 font-mono">
      {/* Header & Filter Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800">
        <div>
          <div className="flex items-center gap-2">
            <ScanFace className="w-5 h-5 text-rose-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              EVIDENCE FRAME MATRIX & FACE ROIs
            </h3>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">
            Displaying {filteredFrames.length} of {frames.length} sequential evaluated frames &bull; Click to isolate
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Layer View Mode Switcher */}
          <div className="flex items-center bg-neutral-950 rounded-lg p-1 border border-neutral-800 text-[11px]">
            <button
              onClick={() => setViewLayer('crop')}
              className={`px-2 py-1 rounded transition-colors ${
                viewLayer === 'crop' ? 'bg-neutral-800 text-white font-bold' : 'text-neutral-400 hover:text-white'
              }`}
            >
              Face ROI
            </button>
            <button
              onClick={() => setViewLayer('heatmap')}
              className={`px-2 py-1 rounded transition-colors ${
                viewLayer === 'heatmap' ? 'bg-rose-950 text-rose-300 font-bold' : 'text-neutral-400 hover:text-white'
              }`}
            >
              Grad-CAM
            </button>
            <button
              onClick={() => setViewLayer('fft')}
              className={`px-2 py-1 rounded transition-colors ${
                viewLayer === 'fft' ? 'bg-sky-950 text-sky-300 font-bold' : 'text-neutral-400 hover:text-white'
              }`}
            >
              2D FFT
            </button>
          </div>

          {/* Filter Pills */}
          <div className="flex items-center bg-neutral-950 rounded-lg p-1 border border-neutral-800 text-[11px]">
            <button
              onClick={() => setFilter('all')}
              className={`px-2.5 py-1 rounded transition-colors ${
                filter === 'all'
                  ? 'bg-neutral-800 text-white font-bold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              All ({frames.length})
            </button>
            <button
              onClick={() => setFilter('fake')}
              className={`px-2.5 py-1 rounded transition-colors ${
                filter === 'fake'
                  ? 'bg-rose-950 text-rose-300 font-bold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Fake ({frames.filter((f) => f.prediction === 'DEEPFAKE').length})
            </button>
            <button
              onClick={() => setFilter('real')}
              className={`px-2.5 py-1 rounded transition-colors ${
                filter === 'real'
                  ? 'bg-emerald-950 text-emerald-300 font-bold'
                  : 'text-neutral-400 hover:text-white'
              }`}
            >
              Real ({frames.filter((f) => f.prediction === 'REAL').length})
            </button>
          </div>
        </div>
      </div>

      {/* Frames Grid */}
      {filteredFrames.length === 0 ? (
        <div className="text-center py-12 text-neutral-500 text-xs">
          No frames match active filter criteria.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {filteredFrames.map((frame) => {
            const isFake = frame.prediction === 'DEEPFAKE';
            let displayImg = frame.faceCropUrl || frame.frameDataUrl;
            if (viewLayer === 'heatmap') displayImg = frame.heatmapDataUrl || frame.faceCropUrl;
            if (viewLayer === 'fft') displayImg = frame.fftSpectrumDataUrl || frame.faceCropUrl;
            if (viewLayer === 'mesh') displayImg = frame.landmarksDataUrl || frame.faceCropUrl;

            const isSelected = activeFrameIndex !== undefined && frame.frameIndex === activeFrameIndex + 1;

            return (
              <div
                key={frame.frameIndex}
                onClick={() => onSelectFrame(frame)}
                className={`group rounded-xl border p-2 transition-all cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'border-white bg-neutral-800 ring-2 ring-rose-500 shadow-xl'
                    : isFake
                    ? 'border-rose-950 bg-neutral-950 hover:border-rose-700 hover:bg-rose-950/20'
                    : 'border-emerald-950 bg-neutral-950 hover:border-emerald-700 hover:bg-emerald-950/20'
                }`}
              >
                {/* Visual Face Crop or Heatmap */}
                <div className="relative aspect-square w-full rounded-lg bg-black overflow-hidden mb-2 border border-neutral-800">
                  {displayImg ? (
                    <img
                      src={displayImg}
                      alt={`Frame ${frame.frameIndex}`}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-neutral-600 text-xs">
                      No Face
                    </div>
                  )}

                  {/* Frame Timecode badge */}
                  <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/85 text-neutral-300 font-mono text-[9px] font-bold border border-neutral-700">
                    {frame.timestampFormatted}
                  </span>

                  {/* Layer indicator */}
                  {viewLayer !== 'crop' && (
                    <span className="absolute bottom-1 right-1 px-1 py-0.5 rounded bg-black/85 text-neutral-300 font-mono text-[8px] font-bold uppercase border border-neutral-700">
                      {viewLayer}
                    </span>
                  )}
                </div>

                {/* Frame Details */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="font-semibold text-neutral-400">
                      #{frame.frameIndex}
                    </span>
                    <span
                      className={`font-bold ${
                        isFake ? 'text-rose-400' : 'text-emerald-400'
                      }`}
                    >
                      {frame.confidence}%
                    </span>
                  </div>

                  {/* Status Pill */}
                  <div
                    className={`w-full py-0.5 text-center rounded text-[9px] font-bold uppercase tracking-wider ${
                      isFake
                        ? 'bg-rose-950 text-rose-300 border border-rose-800'
                        : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                    }`}
                  >
                    {frame.prediction}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
