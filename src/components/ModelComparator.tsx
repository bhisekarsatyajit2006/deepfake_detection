import React, { useState } from 'react';
import { Layers, CheckCircle2, AlertTriangle, ShieldCheck, Database, GitBranch, Cpu, Award } from 'lucide-react';
import { MODEL_BENCHMARKS, DATASET_SPLIT_INFO } from '../data/modelsData';
import { ModelArchId } from '../types';

interface ModelComparatorProps {
  selectedModel: ModelArchId;
  onSelectModel: (id: ModelArchId) => void;
}

export const ModelComparator: React.FC<ModelComparatorProps> = ({
  selectedModel,
  onSelectModel
}) => {
  const [activeDatasetTab, setActiveDatasetTab] = useState<number>(0);

  return (
    <div className="space-y-8 font-mono text-neutral-200">
      {/* Header */}
      <div className="bg-neutral-900 rounded-2xl p-6 sm:p-8 shadow-xl border border-neutral-800">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-950 border border-neutral-800 text-xs font-semibold text-rose-400 mb-3">
            <Cpu className="w-3.5 h-3.5" />
            <span>NEURAL BACKBONE BENCHMARK & EVALUATION MATRIX</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black tracking-tight text-white mb-2 uppercase">
            Model Evaluation & Comparative Analysis
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400 leading-relaxed">
            Rigorous cross-evaluation of transfer-learning neural networks fine-tuned for deepfake classification:
            <strong className="text-white"> EfficientNet-B4</strong> vs <strong className="text-white">Xception Net</strong> vs <strong className="text-white">ResNet-50</strong>.
            Trained on research benchmarks partitioned with strict actor-isolated zero-leakage boundaries.
          </p>
        </div>
      </div>

      {/* Model Cards Comparison Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-neutral-300 flex items-center gap-2">
            <Layers className="w-4 h-4 text-rose-400" />
            <span>Transfer Learning Backbones</span>
          </h3>
          <span className="text-[11px] text-neutral-500">
            Click a card to activate model head
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {MODEL_BENCHMARKS.map((model) => {
            const isSelected = selectedModel === model.id;
            return (
              <div
                key={model.id}
                onClick={() => onSelectModel(model.id)}
                className={`rounded-2xl border p-5 sm:p-6 transition-all cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'border-rose-500 bg-neutral-900 shadow-[0_0_20px_rgba(244,63,94,0.15)] ring-1 ring-rose-500'
                    : 'border-neutral-800 bg-neutral-900/60 hover:border-neutral-700 hover:bg-neutral-900'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-xs font-mono font-bold text-neutral-400">
                      {model.parameters} &bull; {model.inputResolution}
                    </span>
                    {isSelected && (
                      <span className="px-2 py-0.5 rounded bg-rose-600 text-white font-mono text-[10px] font-bold uppercase">
                        ACTIVE HEAD
                      </span>
                    )}
                  </div>

                  <h4 className="text-lg font-bold text-white mb-1">{model.name}</h4>
                  <div className="text-xs text-neutral-400 font-mono mb-3">
                    Arch: <span className="text-neutral-200">{model.architecture}</span>
                  </div>

                  <p className="text-xs text-neutral-400 mb-4 leading-relaxed">
                    {model.description}
                  </p>

                  {/* Benchmark metric highlights */}
                  <div className="grid grid-cols-2 gap-2 bg-neutral-950 p-3 rounded-xl border border-neutral-800 text-xs mb-4">
                    <div>
                      <span className="text-[10px] text-neutral-500 uppercase block">AUC-ROC (FF++)</span>
                      <span className="text-base font-bold text-white">{model.ffppAucRoc.toFixed(3)}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-neutral-500 uppercase block">AUC-ROC (Celeb-DF)</span>
                      <span className="text-base font-bold text-white">{model.celebDfAucRoc.toFixed(3)}</span>
                    </div>
                    <div className="mt-1">
                      <span className="text-[10px] text-neutral-500 uppercase block">Accuracy</span>
                      <span className="text-base font-bold text-emerald-400">{model.accuracy}%</span>
                    </div>
                    <div className="mt-1">
                      <span className="text-[10px] text-neutral-500 uppercase block">Inference Latency</span>
                      <span className="text-base font-bold text-sky-400">{model.inferenceLatencyMs}ms</span>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[10px] text-neutral-400 font-bold uppercase mb-2">Key Strengths</div>
                  <ul className="space-y-1 text-xs text-neutral-400 mb-4">
                    {model.strengths.map((st, i) => (
                      <li key={i} className="flex items-center gap-1.5">
                        <CheckCircle2 className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                        <span>{st}</span>
                      </li>
                    ))}
                  </ul>

                  <button
                    type="button"
                    className={`w-full py-2 rounded-xl text-xs font-bold transition-colors ${
                      isSelected
                        ? 'bg-rose-600 text-white'
                        : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                    }`}
                  >
                    {isSelected ? 'Currently Selected' : 'Switch to this Model'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Side-by-Side Detailed Metrics Table */}
      <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 shadow-xl overflow-hidden">
        <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
          <Award className="w-4 h-4 text-amber-400" />
          <span>Evaluation Metrics Across Standard Benchmarks</span>
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-500 uppercase tracking-wider">
                <th className="py-3 px-4">Model Architecture</th>
                <th className="py-3 px-4">Input Resolution</th>
                <th className="py-3 px-4">AUC-ROC (FF++ c23)</th>
                <th className="py-3 px-4">AUC-ROC (Celeb-DF)</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">F1 Score</th>
                <th className="py-3 px-4">Latency (ms)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/60 text-neutral-300">
              {MODEL_BENCHMARKS.map((m) => (
                <tr key={m.id} className={selectedModel === m.id ? 'bg-neutral-950 font-bold text-white' : ''}>
                  <td className="py-3 px-4 flex items-center gap-2">
                    {selectedModel === m.id && <span className="w-2 h-2 rounded-full bg-rose-500" />}
                    <span>{m.name.split(' (')[0]}</span>
                  </td>
                  <td className="py-3 px-4 text-neutral-500">{m.inputResolution}</td>
                  <td className="py-3 px-4 font-bold text-white">{m.ffppAucRoc.toFixed(3)}</td>
                  <td className="py-3 px-4 font-bold text-white">{m.celebDfAucRoc.toFixed(3)}</td>
                  <td className="py-3 px-4 text-emerald-400 font-bold">{m.accuracy}%</td>
                  <td className="py-3 px-4">{m.f1Score.toFixed(3)}</td>
                  <td className="py-3 px-4 text-sky-400">{m.inferenceLatencyMs} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* RESEARCH DATASET PARTITIONING & ZERO-LEAKAGE EXPLANATION */}
      <div className="bg-neutral-900 rounded-2xl border border-neutral-800 p-6 sm:p-8 shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-neutral-800">
          <div>
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-sky-400" />
              <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                DATASET PARTITIONING & ZERO-LEAKAGE PROTOCOL
              </h3>
            </div>
            <p className="text-xs text-neutral-400 mt-1">
              Rigorous identity-isolated partitioning preventing actor and subject overlap across Train / Val / Test
            </p>
          </div>

          <div className="flex items-center bg-neutral-950 border border-neutral-800 rounded-lg p-1 text-xs font-semibold gap-1">
            {DATASET_SPLIT_INFO.map((dataset, idx) => (
              <button
                key={idx}
                onClick={() => setActiveDatasetTab(idx)}
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  activeDatasetTab === idx
                    ? 'bg-neutral-800 text-white font-bold'
                    : 'text-neutral-400 hover:text-white'
                }`}
              >
                {dataset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Active Dataset Split Visualization */}
        {(() => {
          const ds = DATASET_SPLIT_INFO[activeDatasetTab];
          return (
            <div className="space-y-6">
              {/* Dataset summary header */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800">
                  <span className="text-[10px] text-neutral-500 uppercase block mb-1">Total Videos</span>
                  <span className="text-xl font-bold text-white">{ds.totalVideos.toLocaleString()}</span>
                </div>
                <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800">
                  <span className="text-[10px] text-neutral-500 uppercase block mb-1">Total Frames</span>
                  <span className="text-xl font-bold text-white">{ds.totalFrames.toLocaleString()}</span>
                </div>
                <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800">
                  <span className="text-[10px] text-neutral-500 uppercase block mb-1">Split Ratio</span>
                  <span className="text-xl font-bold text-rose-400">72% / 14% / 14%</span>
                </div>
                <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800">
                  <span className="text-[10px] text-neutral-500 uppercase block mb-1">Leakage Status</span>
                  <span className="text-xl font-bold text-emerald-400">0.0% Overlap</span>
                </div>
              </div>

              {/* Three Split Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-neutral-950 rounded-xl p-5 border border-neutral-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white text-sm">TRAINING SET</span>
                    <span className="px-2 py-0.5 rounded bg-sky-950 text-sky-400 text-xs font-bold">
                      {ds.trainSplit.percentage}%
                    </span>
                  </div>
                  <div className="text-2xl font-black text-white mb-2">
                    {ds.trainSplit.videos.toLocaleString()} <span className="text-xs font-normal text-neutral-500">videos</span>
                  </div>
                  <p className="text-xs text-neutral-400 leading-relaxed">
                    {ds.trainSplit.subjects}
                  </p>
                </div>

                <div className="bg-neutral-950 rounded-xl p-5 border border-neutral-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white text-sm">VALIDATION SET</span>
                    <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-400 text-xs font-bold">
                      {ds.valSplit.percentage}%
                    </span>
                  </div>
                  <div className="text-2xl font-black text-white mb-2">
                    {ds.valSplit.videos.toLocaleString()} <span className="text-xs font-normal text-neutral-500">videos</span>
                  </div>
                  <p className="text-xs text-neutral-400 leading-relaxed">
                    {ds.valSplit.subjects}
                  </p>
                </div>

                <div className="bg-neutral-950 rounded-xl p-5 border border-neutral-800">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-white text-sm">TESTING SET</span>
                    <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 text-xs font-bold">
                      {ds.testSplit.percentage}%
                    </span>
                  </div>
                  <div className="text-2xl font-black text-white mb-2">
                    {ds.testSplit.videos.toLocaleString()} <span className="text-xs font-normal text-neutral-500">videos</span>
                  </div>
                  <p className="text-xs text-neutral-400 leading-relaxed">
                    {ds.testSplit.subjects}
                  </p>
                </div>
              </div>

              {/* Zero-Leakage Protocol Policy Banner */}
              <div className="bg-neutral-950 rounded-xl p-4 border border-neutral-800 flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                <div className="text-xs leading-relaxed">
                  <strong className="text-white block mb-1">
                    Actor-Isolated Zero-Leakage Enforcement
                  </strong>
                  <span className="text-neutral-400">{ds.zeroLeakageProtocol}</span>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
};
