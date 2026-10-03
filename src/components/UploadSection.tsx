import React, { useState, useRef } from 'react';
import {
  Upload,
  Video,
  Image as ImageIcon,
  Sparkles,
  Sliders,
  AlertCircle,
  FileCheck2,
  Cpu,
  Fingerprint,
  FolderOpen,
  ArrowRight,
  ShieldAlert
} from 'lucide-react';
import { ModelArchId, MediaType } from '../types';
import { SAMPLE_MEDIA_LIST, SampleMedia } from '../data/samplesData';
import { MODEL_BENCHMARKS } from '../data/modelsData';

interface UploadSectionProps {
  onFileSelected: (file: File, mediaType: MediaType, interval: number, targetFrames: number) => void;
  onSampleSelected: (sample: SampleMedia, interval: number, targetFrames: number) => void;
  selectedModel: ModelArchId;
  setSelectedModel: (model: ModelArchId) => void;
  detectionSensitivity: 'strict' | 'balanced' | 'conservative';
  setDetectionSensitivity: (sensitivity: 'strict' | 'balanced' | 'conservative') => void;
  collageMode: 'auto' | 'force_collage' | 'standard';
  setCollageMode: (mode: 'auto' | 'force_collage' | 'standard') => void;
  isProcessing: boolean;
}

export const UploadSection: React.FC<UploadSectionProps> = ({
  onFileSelected,
  onSampleSelected,
  selectedModel,
  setSelectedModel,
  detectionSensitivity,
  setDetectionSensitivity,
  collageMode,
  setCollageMode,
  isProcessing
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [frameInterval, setFrameInterval] = useState<number>(10);
  const [targetFrames, setTargetFrames] = useState<number>(120);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const processFile = (file: File) => {
    setErrorMessage(null);
    const fileName = file.name.toLowerCase();
    const isVideo =
      fileName.endsWith('.mp4') ||
      fileName.endsWith('.avi') ||
      fileName.endsWith('.mov') ||
      file.type.startsWith('video/');
    const isImage =
      fileName.endsWith('.jpg') ||
      fileName.endsWith('.jpeg') ||
      fileName.endsWith('.png') ||
      file.type.startsWith('image/');

    if (!isVideo && !isImage) {
      setErrorMessage(
        'Unsupported file container. Please supply standard Forensic Video (.MP4, .AVI, .MOV) or Still Frame (.JPG, .PNG).'
      );
      return;
    }

    const mediaType: MediaType = isVideo ? 'video' : 'image';
    onFileSelected(file, mediaType, frameInterval, isVideo ? targetFrames : 1);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Evidence Intake Banner & Parameter Console */}
      <div className="bg-neutral-900 text-neutral-100 rounded-2xl p-6 sm:p-7 shadow-xl border border-neutral-800">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-800 border border-neutral-700 text-xs font-mono text-neutral-300 mb-3">
              <Fingerprint className="w-3.5 h-3.5 text-rose-400" />
              <span>DIGITAL EVIDENCE INGESTION & FORENSIC SCREENING</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white mb-2 font-mono">
              EVIDENCE INTAKE WORKBENCH
            </h1>
            <p className="text-neutral-400 text-xs sm:text-sm leading-relaxed">
              Accepts suspicious digital recordings or images. Samples frames at uniform intervals, executes
              facial localization and bounding box isolation, normalizes tensors, and feeds sequence frames through fine-tuned
              PyTorch transfer networks to detect manipulation artifacts and synthesis seams.
            </p>
          </div>

          {/* System Spec Pill */}
          <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 text-xs font-mono space-y-1.5 min-w-[240px]">
            <div className="text-neutral-500 font-bold uppercase text-[10px] tracking-wider">
              INSPECTION PROTOCOL
            </div>
            <div className="flex justify-between text-neutral-300">
              <span>Pipeline:</span>
              <span className="text-emerald-400 font-semibold">OpenCV + PyTorch</span>
            </div>
            <div className="flex justify-between text-neutral-300">
              <span>Resolution:</span>
              <span className="text-white">224x224 / 299x299</span>
            </div>
            <div className="flex justify-between text-neutral-300">
              <span>Datasets:</span>
              <span className="text-sky-400">FF++ (c23) / Celeb-DF</span>
            </div>
          </div>
        </div>

        {/* Video & Pipeline Configuration Parameters */}
        <div className="mt-6 pt-5 border-t border-neutral-800/80 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs font-mono">
          <div>
            <label className="block text-neutral-400 font-semibold mb-1.5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-sky-400" />
              Transfer Learning Architecture
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelArchId)}
              disabled={isProcessing}
              className="w-full bg-neutral-950 border border-neutral-700 text-white rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              {MODEL_BENCHMARKS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.accuracy}% Acc &bull; {m.inputResolution})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-neutral-400 font-semibold mb-1.5 flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              Detection Sensitivity
            </label>
            <select
              value={detectionSensitivity}
              onChange={(e) => setDetectionSensitivity(e.target.value as 'strict' | 'balanced' | 'conservative')}
              disabled={isProcessing}
              className="w-full bg-neutral-950 border border-neutral-700 text-white rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              <option value="balanced">Balanced (NIST Benchmark - Standard Recommended)</option>
              <option value="strict">Strict (Subtle AI Video / Sora / FaceSwap)</option>
              <option value="conservative">Conservative (Low False-Positive Filter)</option>
            </select>
          </div>

          <div>
            <label className="block text-neutral-400 font-semibold mb-1.5 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Collage / Dual Comparison
            </label>
            <select
              value={collageMode}
              onChange={(e) => setCollageMode(e.target.value as 'auto' | 'force_collage' | 'standard')}
              disabled={isProcessing}
              className="w-full bg-neutral-950 border border-neutral-700 text-white rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              <option value="auto">Auto-Detect Real vs Fake (Default)</option>
              <option value="force_collage">Force Side-by-Side Dual Analysis</option>
              <option value="standard">Single File Standard Mode</option>
            </select>
          </div>

          <div>
            <label className="block text-neutral-400 font-semibold mb-1.5 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-amber-400" />
              Frame Sampling Interval
            </label>
            <select
              value={frameInterval}
              onChange={(e) => setFrameInterval(Number(e.target.value))}
              disabled={isProcessing}
              className="w-full bg-neutral-950 border border-neutral-700 text-white rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              <option value={5}>Every 5th frame (Dense forensic audit)</option>
              <option value={10}>Every 10th frame (Standard default)</option>
              <option value={15}>Every 15th frame (Rapid triage)</option>
              <option value={30}>Every 30th frame (~1 fps macro sampling)</option>
            </select>
          </div>

          <div>
            <label className="block text-neutral-400 font-semibold mb-1.5 flex items-center gap-1.5">
              <Video className="w-3.5 h-3.5 text-emerald-400" />
              Target Analyzed Frames Cap
            </label>
            <select
              value={targetFrames}
              onChange={(e) => setTargetFrames(Number(e.target.value))}
              disabled={isProcessing}
              className="w-full bg-neutral-950 border border-neutral-700 text-white rounded-lg px-3 py-2 text-xs font-mono focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              <option value={30}>30 frames (Quick verification)</option>
              <option value={60}>60 frames (Standard sample)</option>
              <option value={120}>120 frames (Full sequence consensus - Recommended)</option>
              <option value={180}>180 frames (Maximum depth forensic)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Upload Drop Zone / Evidence Intake Bay */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center transition-all cursor-pointer select-none ${
          isDragOver
            ? 'border-rose-500 bg-neutral-900/90 shadow-[0_0_25px_rgba(244,63,94,0.15)] scale-[1.005]'
            : 'border-neutral-700/80 hover:border-neutral-500 bg-neutral-900/60 hover:bg-neutral-900'
        } ${isProcessing ? 'pointer-events-none opacity-60' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".mp4,.avi,.mov,video/mp4,video/avi,video/quicktime,.jpg,.jpeg,.png,image/jpeg,image/png"
          onChange={handleFileInputChange}
          className="hidden"
        />

        <div className="max-w-md mx-auto flex flex-col items-center">
          <div className="w-16 h-16 rounded-2xl bg-neutral-950 border border-neutral-800 flex items-center justify-center mb-4 text-white shadow-inner group-hover:border-neutral-700">
            <Upload className="w-8 h-8 text-neutral-300" />
          </div>

          <h3 className="text-lg font-bold text-white font-mono mb-1">
            INGEST EVIDENCE FILE FOR ANALYSIS
          </h3>
          <p className="text-xs text-neutral-400 font-mono mb-4">
            Drag and drop suspect media or click to browse local storage
          </p>

          <div className="flex flex-wrap items-center justify-center gap-2 mb-5 font-mono text-xs">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-neutral-800 border border-neutral-700 text-neutral-200">
              <Video className="w-3.5 h-3.5 text-rose-400" />
              VIDEO: MP4, AVI, MOV (Primary)
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-neutral-800 border border-neutral-700 text-neutral-200">
              <ImageIcon className="w-3.5 h-3.5 text-emerald-400" />
              IMAGE: JPG, PNG
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-amber-950/60 border border-amber-800 text-amber-300">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              COLLAGE: Real vs Fake Split Photos
            </span>
          </div>

          <button
            type="button"
            className="px-6 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs font-bold tracking-wider uppercase shadow-lg shadow-rose-950/40 transition-all flex items-center gap-2"
          >
            <FolderOpen className="w-4 h-4" />
            <span>Select Media File</span>
          </button>
        </div>
      </div>

      {/* Error alert */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-300 text-xs font-mono flex items-center gap-3">
          <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400" />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* Quick Test with Research Samples */}
      <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <h3 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              EVIDENCE VAULT: BENCHMARK TEST SAMPLES
            </h3>
          </div>
          <span className="text-[11px] font-mono text-neutral-400">
            Verified ground-truth media from FaceForensics++ & Celeb-DF v2
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {SAMPLE_MEDIA_LIST.map((sample) => {
            const isFake = sample.groundTruth === 'DEEPFAKE';
            return (
              <div
                key={sample.id}
                onClick={() => !isProcessing && onSampleSelected(sample, frameInterval, targetFrames)}
                className={`p-4 rounded-xl border transition-all text-left flex flex-col justify-between cursor-pointer group ${
                  isFake
                    ? 'border-rose-900/60 bg-neutral-950/80 hover:bg-rose-950/20 hover:border-rose-700'
                    : 'border-emerald-900/60 bg-neutral-950/80 hover:bg-emerald-950/20 hover:border-emerald-700'
                } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span
                      className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded tracking-wider uppercase ${
                        isFake
                          ? 'bg-rose-950 text-rose-400 border border-rose-800'
                          : 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                      }`}
                    >
                      {sample.groundTruth}
                    </span>
                    <span className="text-[11px] text-neutral-500 font-mono">
                      {sample.format} &bull; {sample.sizeFormatted} &bull; {sample.simulatedPreset.recommendedFrames} Frames
                    </span>
                  </div>

                  <h4 className="font-bold text-white text-sm font-mono mb-1 group-hover:text-rose-400 transition-colors">
                    {sample.title}
                  </h4>
                  <p className="text-xs text-neutral-400 mb-3 leading-relaxed">
                    {sample.description}
                  </p>
                </div>

                <div className="pt-2 border-t border-neutral-800/80 flex items-center justify-between text-xs font-mono text-neutral-400">
                  <span className="text-neutral-500">{sample.sourceDataset}</span>
                  <span className="text-white font-bold group-hover:text-rose-400 flex items-center gap-1">
                    <span>Execute Pipeline</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
