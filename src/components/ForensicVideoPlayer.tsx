import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Scan,
  Flame,
  Activity,
  Layers,
  Split,
  Maximize2,
  Sliders,
  ShieldAlert,
  ShieldCheck,
  Eye,
  Crosshair
} from 'lucide-react';
import { FrameAnalysis, PredictionLabel } from '../types';

interface ForensicVideoPlayerProps {
  frames: FrameAnalysis[];
  currentFrameIndex?: number;
  onSelectFrameIndex?: (index: number) => void;
  onFrameChange?: (index: number) => void;
  mediaType?: 'video' | 'image';
  videoDuration?: number;
  videoName?: string;
  fileName?: string;
  prediction?: PredictionLabel;
  confidence?: number;
  onSelectFrameForModal?: (frame: FrameAnalysis) => void;
}

export type ForensicViewMode = 'rgb' | 'heatmap' | 'fft' | 'mesh' | 'split';

export const ForensicVideoPlayer: React.FC<ForensicVideoPlayerProps> = ({
  frames,
  currentFrameIndex,
  onSelectFrameIndex,
  onFrameChange,
  mediaType = 'video',
  videoDuration = 10,
  videoName,
  fileName = videoName || 'suspect_evidence.mp4',
  prediction,
  confidence,
  onSelectFrameForModal
}) => {
  const [internalIndex, setInternalIndex] = useState(0);

  // Clamp indices to valid frame boundaries
  const frameCount = frames.length;
  const safeCurrentIndex =
    currentFrameIndex !== undefined && frameCount > 0
      ? Math.max(0, Math.min(frameCount - 1, currentFrameIndex))
      : undefined;

  const activeIndex =
    safeCurrentIndex !== undefined
      ? safeCurrentIndex
      : frameCount > 0
      ? Math.max(0, Math.min(frameCount - 1, internalIndex))
      : 0;

  // Refs to allow timer to access latest state without calling state updates inside updater callbacks
  const activeIndexRef = useRef(activeIndex);
  activeIndexRef.current = activeIndex;

  const onFrameChangeRef = useRef(onFrameChange);
  onFrameChangeRef.current = onFrameChange;

  const onSelectFrameIndexRef = useRef(onSelectFrameIndex);
  onSelectFrameIndexRef.current = onSelectFrameIndex;

  const updateIndex = (newIdx: number) => {
    const clampedIdx = Math.max(0, Math.min(frameCount - 1, newIdx));
    setInternalIndex(clampedIdx);
    onSelectFrameIndexRef.current?.(clampedIdx);
    onFrameChangeRef.current?.(clampedIdx);
  };

  const [isPlaying, setIsPlaying] = useState(false);
  const [viewMode, setViewMode] = useState<ForensicViewMode>('rgb');
  const [splitPosition, setSplitPosition] = useState(50); // percentage for split slider
  const [showTargetReticle, setShowTargetReticle] = useState(true);

  const currentFrame = frames[activeIndex] || frames[0];
  const isFake = currentFrame?.prediction === 'DEEPFAKE';

  // Stop playback if frame count becomes 1 or 0
  useEffect(() => {
    if (frameCount <= 1 && isPlaying) {
      setIsPlaying(false);
    }
  }, [frameCount, isPlaying]);

  // Playback timer simulation across frames (never call parent setState inside a child state updater)
  useEffect(() => {
    if (!isPlaying || frameCount <= 1) return;

    const interval = setInterval(() => {
      const next = (activeIndexRef.current + 1) % frameCount;
      setInternalIndex(next);
      onSelectFrameIndexRef.current?.(next);
      onFrameChangeRef.current?.(next);
    }, 120); // ~8-10 FPS playback simulation through keyframes

    return () => {
      clearInterval(interval);
    };
  }, [isPlaying, frameCount]);

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetIdx = Math.floor(ratio * frames.length);
    updateIndex(Math.min(frames.length - 1, targetIdx));
  };

  const handlePrevFrame = () => {
    setIsPlaying(false);
    updateIndex(Math.max(0, activeIndex - 1));
  };

  const handleNextFrame = () => {
    setIsPlaying(false);
    updateIndex(Math.min(frames.length - 1, activeIndex + 1));
  };

  if (!currentFrame) return null;

  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col font-mono">
      {/* Console Top Toolbar */}
      <div className="bg-neutral-950 px-4 py-2.5 border-b border-neutral-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 font-bold text-white">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>OPTICAL STAGE 01</span>
          </div>
          <span className="text-neutral-600 font-mono">|</span>
          <span className="text-neutral-400 font-mono truncate max-w-[200px] sm:max-w-xs">
            {fileName}
          </span>
          <span className="hidden md:inline px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-300 text-[10px]">
            {mediaType === 'video' ? 'H.264 / 29.97 FPS' : 'STATIC IMAGE'}
          </span>
        </div>

        {/* View Layer Selector Tabs */}
        <div className="flex items-center bg-neutral-900 rounded-lg p-1 border border-neutral-800 gap-1 text-[11px]">
          <button
            onClick={() => setViewMode('rgb')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1.5 cursor-pointer ${
              viewMode === 'rgb'
                ? 'bg-neutral-800 text-white font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white'
            }`}
            title="Raw RGB with Face Detection Bounding Box"
          >
            <Scan className="w-3.5 h-3.5" />
            <span>RGB + BBOX</span>
          </button>

          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1.5 cursor-pointer ${
              viewMode === 'heatmap'
                ? 'bg-rose-950 text-rose-300 border border-rose-800/80 font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white'
            }`}
            title="Grad-CAM Deep Activation Heatmap"
          >
            <Flame className="w-3.5 h-3.5 text-rose-400" />
            <span>GRAD-CAM</span>
          </button>

          <button
            onClick={() => setViewMode('fft')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1.5 cursor-pointer ${
              viewMode === 'fft'
                ? 'bg-sky-950 text-sky-300 border border-sky-800/80 font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white'
            }`}
            title="2D Fourier Frequency Spectrum (Lattice Spike Analysis)"
          >
            <Activity className="w-3.5 h-3.5 text-sky-400" />
            <span>2D FFT</span>
          </button>

          <button
            onClick={() => setViewMode('mesh')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1.5 cursor-pointer ${
              viewMode === 'mesh'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/80 font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white'
            }`}
            title="68-Point Facial Landmark Topology Mesh"
          >
            <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
            <span>68-PT MESH</span>
          </button>

          <button
            onClick={() => setViewMode('split')}
            className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1.5 cursor-pointer ${
              viewMode === 'split'
                ? 'bg-amber-950 text-amber-300 border border-amber-800/80 font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white'
            }`}
            title="A/B Split Screen (RGB vs Grad-CAM)"
          >
            <Split className="w-3.5 h-3.5 text-amber-400" />
            <span>A/B SPLIT</span>
          </button>
        </div>
      </div>

      {/* Main Optical Canvas / Inspection Stage */}
      <div className="relative bg-black aspect-video sm:max-h-[500px] flex items-center justify-center overflow-hidden select-none">
        {/* Layer 1: RGB Image Frame */}
        <img
          src={currentFrame.frameDataUrl}
          alt={`Frame ${currentFrame.frameIndex}`}
          className="w-full h-full object-contain"
        />

        {/* Layer 2: Grad-CAM Overlay */}
        {viewMode === 'heatmap' && currentFrame.heatmapDataUrl && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <img
              src={currentFrame.heatmapDataUrl}
              alt="Grad-CAM"
              className="max-h-full max-w-full object-contain mix-blend-screen opacity-90"
            />
          </div>
        )}

        {/* Layer 3: 2D FFT Magnitude Spectrum Mode */}
        {viewMode === 'fft' && (
          <div className="absolute inset-0 bg-neutral-950/95 flex flex-col items-center justify-center p-6">
            <div className="relative max-w-xs w-full aspect-square bg-[#050811] rounded-xl border border-sky-900/60 p-2 shadow-2xl flex flex-col items-center justify-center">
              {currentFrame.fftSpectrumDataUrl ? (
                <img
                  src={currentFrame.fftSpectrumDataUrl}
                  alt="2D FFT Spectrum"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <div className="text-center text-sky-400 text-xs font-mono">
                  Synthesizing 2D Fourier Tensor...
                </div>
              )}
              <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/80 text-[10px] text-sky-300 font-mono border border-sky-800">
                2D RADIAL POWER SPECTRUM
              </div>
            </div>
            <div className="mt-3 text-center max-w-md">
              <span className={`text-xs font-mono font-bold ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
                {isFake
                  ? 'HIGH-FREQUENCY PERIODIC LATTICE ANOMALY DETECTED'
                  : 'NATURAL RADIAL EXPONENTIAL FREQUENCY DECAY'}
              </span>
              <p className="text-[11px] text-neutral-400 font-mono mt-1">
                {isFake
                  ? 'Prominent spectral spikes created by deep generative upscaling & transposed convolutions.'
                  : 'Uniform spatial power decay characteristic of natural camera sensor optics.'}
              </p>
            </div>
          </div>
        )}

        {/* Layer 4: 68-Point Landmark Mesh Overlay */}
        {viewMode === 'mesh' && (
          <div className="absolute inset-0 bg-neutral-950/90 flex flex-col items-center justify-center p-4">
            <div className="relative max-w-xs w-full aspect-square bg-black rounded-xl border border-neutral-700 p-2 shadow-2xl flex items-center justify-center">
              {currentFrame.landmarksDataUrl ? (
                <img
                  src={currentFrame.landmarksDataUrl}
                  alt="68-Point Mesh"
                  className="w-full h-full object-contain rounded-lg"
                />
              ) : (
                <img
                  src={currentFrame.faceCropUrl || currentFrame.frameDataUrl}
                  alt="Face"
                  className="w-full h-full object-contain rounded-lg opacity-40"
                />
              )}
              <div className="absolute top-3 left-3 px-2 py-0.5 rounded bg-black/80 text-[10px] text-emerald-300 font-mono border border-emerald-800">
                68-PT DLIB/FACENET MESH
              </div>
            </div>
            <div className="mt-2 text-center text-xs font-mono text-neutral-400">
              Facial geometry stability: <strong className="text-white">{100 - currentFrame.artifacts.boundaryBlendingScore}% coherent</strong>
            </div>
          </div>
        )}

        {/* Layer 5: A/B Split Screen Mode */}
        {viewMode === 'split' && currentFrame.heatmapDataUrl && (
          <div
            className="absolute inset-0 pointer-events-none overflow-hidden"
            style={{ clipPath: `inset(0 0 0 ${splitPosition}%)` }}
          >
            <div className="absolute inset-0 flex items-center justify-center bg-black">
              <img
                src={currentFrame.heatmapDataUrl}
                alt="Heatmap layer"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        )}

        {/* Split separator bar */}
        {viewMode === 'split' && (
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-rose-500 shadow-[0_0_10px_#f43f5e] pointer-events-none z-30"
            style={{ left: `${splitPosition}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-rose-600 border border-white flex items-center justify-center text-[10px] text-white font-bold shadow-lg">
              AB
            </div>
          </div>
        )}

        {/* HUD Target Reticle / Face Bounding Box */}
        {showTargetReticle && currentFrame.boundingBox && viewMode !== 'fft' && viewMode !== 'mesh' && (
          <div
            className={`absolute border-2 pointer-events-none transition-all duration-75 z-20 ${
              isFake
                ? 'border-rose-500 bg-rose-500/10 shadow-[0_0_20px_rgba(244,63,94,0.3)]'
                : 'border-emerald-500 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.3)]'
            }`}
            style={{
              left: `${(currentFrame.boundingBox.x / 640) * 100}%`,
              top: `${(currentFrame.boundingBox.y / 360) * 100}%`,
              width: `${(currentFrame.boundingBox.width / 640) * 100}%`,
              height: `${(currentFrame.boundingBox.height / 360) * 100}%`
            }}
          >
            {/* Corner brackets */}
            <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-white" />
            <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-white" />
            <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-white" />
            <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-white" />

            {/* Bounding Box Info Tag */}
            <div className="absolute -top-6 left-0 flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-black/90 border border-neutral-700 text-[10px] font-mono text-white">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isFake ? 'bg-rose-500 animate-ping' : 'bg-emerald-500'
                }`}
              />
              <span>ROI 01 [{currentFrame.prediction}]</span>
              <span className="text-neutral-400">({currentFrame.confidence}%)</span>
            </div>
          </div>
        )}

        {/* Top-Right Telemetry HUD Overlay */}
        <div className="absolute top-3 right-3 bg-black/85 backdrop-blur-md rounded-xl p-3 border border-neutral-800 text-xs font-mono space-y-1 z-20 pointer-events-none min-w-[190px]">
          <div className="flex items-center justify-between text-neutral-400 text-[10px] border-b border-neutral-800 pb-1 mb-1">
            <span>SEQUENCE TELEMETRY</span>
            <span className="text-emerald-400">SYNC OK</span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">FRAME:</span>
            <span className="text-white font-bold">
              #{currentFrame.frameIndex.toString().padStart(3, '0')} / {frames.length}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">TIMECODE:</span>
            <span className="text-white">{currentFrame.timestampFormatted}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">VERDICT:</span>
            <span className={`font-bold ${isFake ? 'text-rose-400' : 'text-emerald-400'}`}>
              {currentFrame.prediction}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">CONFIDENCE:</span>
            <span className="text-white font-bold">{currentFrame.confidence}%</span>
          </div>
          <div className="flex justify-between border-t border-neutral-800/80 pt-1 mt-1 text-[10px]">
            <span className="text-neutral-500">SEAM VARIANCE:</span>
            <span className={isFake ? 'text-rose-400' : 'text-neutral-300'}>
              {currentFrame.artifacts.boundaryBlendingScore}%
            </span>
          </div>
        </div>

        {/* Modal deep-dive trigger */}
        {onSelectFrameForModal && (
          <button
            onClick={() => onSelectFrameForModal(currentFrame)}
            className="absolute bottom-3 right-3 px-2.5 py-1.5 rounded-lg bg-neutral-900/90 hover:bg-neutral-800 border border-neutral-700 text-neutral-200 text-xs font-mono flex items-center gap-1.5 backdrop-blur z-20 cursor-pointer"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span>Deep Dive Inspect</span>
          </button>
        )}
      </div>

      {/* Console Bottom Transport & Scrubber Bar */}
      <div className="bg-neutral-950 p-4 border-t border-neutral-800 space-y-3">
        {/* Visual Frame-by-Frame Heatmap Timeline Scrubber */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs font-mono text-neutral-400">
            <span>SEQUENCE TIMELINE ({frames.length} FRAMES ANALYZED)</span>
            <div className="flex items-center gap-3 text-[10px]">
              <span className="flex items-center gap-1 text-emerald-400">
                <span className="w-2 h-2 rounded-full bg-emerald-500" /> Real
              </span>
              <span className="flex items-center gap-1 text-rose-400">
                <span className="w-2 h-2 rounded-full bg-rose-500" /> Deepfake
              </span>
            </div>
          </div>

          {/* Scrubber Bar */}
          <div
            onClick={handleTimelineClick}
            className="relative h-6 w-full bg-neutral-900 rounded-md overflow-hidden cursor-pointer flex border border-neutral-800 group hover:border-neutral-600 transition-colors"
          >
            {frames.map((frame, idx) => {
              const isFk = frame.prediction === 'DEEPFAKE';
              return (
                <div
                  key={idx}
                  className={`h-full flex-1 transition-opacity ${
                    isFk
                      ? 'bg-rose-600 hover:bg-rose-400'
                      : 'bg-emerald-600 hover:bg-emerald-400'
                  } ${idx === activeIndex ? 'brightness-150 ring-2 ring-white z-10' : 'opacity-85'}`}
                  title={`Frame #${frame.frameIndex} | ${frame.timestampFormatted} | ${frame.prediction} (${frame.confidence}%)`}
                />
              );
            })}

            {/* Playhead indicator */}
            <div
              className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white] z-20 pointer-events-none"
              style={{
                left: `${((activeIndex + 0.5) / frames.length) * 100}%`
              }}
            />
          </div>
        </div>

        {/* Video Controls & Transport */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          {/* Transport buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-3.5 py-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-white text-xs font-mono font-bold flex items-center gap-2 transition-colors border border-neutral-700 cursor-pointer"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isPlaying ? 'PAUSE' : 'PLAY SEQUENCE'}</span>
            </button>

            <button
              onClick={handlePrevFrame}
              className="p-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors border border-neutral-700 cursor-pointer"
              title="Previous Frame"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={handleNextFrame}
              className="p-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white transition-colors border border-neutral-700 cursor-pointer"
              title="Next Frame"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>

            <span className="font-mono text-xs text-neutral-400 ml-2">
              <strong className="text-white font-bold">{activeIndex + 1}</strong> / {frames.length} frames
            </span>
          </div>

          {/* Quick toggle settings */}
          <div className="flex items-center gap-3 text-xs font-mono text-neutral-400">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={showTargetReticle}
                onChange={(e) => setShowTargetReticle(e.target.checked)}
                className="rounded bg-neutral-800 border-neutral-700 text-rose-500 focus:ring-0"
              />
              <span>Target Reticle</span>
            </label>

            {viewMode === 'split' && (
              <div className="flex items-center gap-2">
                <span className="text-[10px]">Split:</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={splitPosition}
                  onChange={(e) => setSplitPosition(Number(e.target.value))}
                  className="w-20 accent-rose-500 h-1 bg-neutral-700 rounded-lg cursor-pointer"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
