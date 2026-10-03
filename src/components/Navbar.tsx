import React, { useState, useEffect } from 'react';
import { ShieldAlert, Cpu, Code2, Layers, Video, Terminal, Radio, Server } from 'lucide-react';
import { ModelArchId } from '../types';
import { MODEL_BENCHMARKS } from '../data/modelsData';

interface NavbarProps {
  activeTab: 'detect' | 'models' | 'code';
  setActiveTab: (tab: 'detect' | 'models' | 'code') => void;
  selectedModel: ModelArchId;
  setSelectedModel: (model: ModelArchId) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  selectedModel,
  setSelectedModel
}) => {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().slice(17, 25) + ' UTC');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-40 w-full border-b border-neutral-800 bg-neutral-950 text-neutral-200">
      {/* Top Telemetry & Status HUD Bar */}
      <div className="bg-neutral-900/90 border-b border-neutral-800/80 px-4 sm:px-6 lg:px-8 py-1 text-[11px] font-mono flex items-center justify-between text-neutral-400">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-semibold">CORE STATUS: ONLINE</span>
          </div>
          <span className="text-neutral-700 hidden sm:inline">|</span>
          <div className="hidden sm:flex items-center gap-1.5 text-neutral-300">
            <Server className="w-3 h-3 text-sky-400" />
            <span>GPU: NVIDIA RTX 4090 [CUDA 12.4]</span>
          </div>
          <span className="text-neutral-700 hidden md:inline">|</span>
          <div className="hidden md:flex items-center gap-1.5 text-neutral-400">
            <span>ENGINE: PyTorch 2.4.1+cu124</span>
          </div>
        </div>

        <div className="flex items-center gap-4 text-neutral-400">
          <div className="hidden lg:flex items-center gap-1 text-[10px] text-neutral-500">
            <span>COMPLIANCE:</span>
            <span className="text-neutral-300 font-bold">DARPA MEDIFOR / NIST CSF</span>
          </div>
          <span className="text-neutral-700 hidden lg:inline">|</span>
          <div className="flex items-center gap-1.5 text-neutral-300 font-bold">
            <Radio className="w-3 h-3 text-rose-400 animate-pulse" />
            <span>{utcTime || '00:00:00 UTC'}</span>
          </div>
        </div>
      </div>

      {/* Primary Navigation Console */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand identity */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-600 to-rose-950 border border-rose-500/40 flex items-center justify-center text-white shadow-[0_0_15px_rgba(244,63,94,0.25)]">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono font-black text-white tracking-wider text-base sm:text-lg uppercase">
                SENTINEL-DF
              </span>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold rounded bg-rose-950 text-rose-300 border border-rose-800/80 tracking-widest uppercase">
                v4.8 PRO
              </span>
            </div>
            <p className="text-[11px] font-mono text-neutral-400 hidden md:block">
              Deepfake Forensic Video & Frame Neural Analysis Workstation
            </p>
          </div>
        </div>

        {/* Tactical Navigation Tabs */}
        <div className="flex items-center bg-neutral-900 rounded-xl p-1 border border-neutral-800 gap-1 text-xs font-mono">
          <button
            onClick={() => setActiveTab('detect')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-2 ${
              activeTab === 'detect'
                ? 'bg-rose-600 text-white font-bold shadow-md shadow-rose-900/30'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            <span>INVESTIGATION</span>
          </button>

          <button
            onClick={() => setActiveTab('models')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-2 ${
              activeTab === 'models'
                ? 'bg-neutral-800 text-white font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">BENCHMARK MATRIX</span>
            <span className="sm:hidden">MODELS</span>
          </button>

          <button
            onClick={() => setActiveTab('code')}
            className={`px-3 py-1.5 rounded-lg transition-all flex items-center gap-2 ${
              activeTab === 'code'
                ? 'bg-neutral-800 text-white font-bold shadow-xs'
                : 'text-neutral-400 hover:text-white hover:bg-neutral-800/60'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">PYTORCH CODE</span>
            <span className="sm:hidden">PYTHON</span>
          </button>
        </div>

        {/* Model Architecture Selector */}
        <div className="hidden lg:flex items-center gap-2 pl-4 border-l border-neutral-800">
          <Cpu className="w-4 h-4 text-sky-400" />
          <div className="text-right font-mono">
            <div className="text-[10px] text-neutral-500 uppercase">ACTIVE MODEL HEAD</div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as ModelArchId)}
              className="text-xs font-mono font-bold bg-neutral-900 hover:bg-neutral-800 text-neutral-200 border border-neutral-700 rounded-lg px-2.5 py-1 focus:outline-none focus:border-rose-500 cursor-pointer"
            >
              {MODEL_BENCHMARKS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name.split(' (')[0]} [{m.inputResolution}]
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </header>
  );
};
