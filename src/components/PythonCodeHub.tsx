import React, { useState } from 'react';
import { Code2, Copy, Check, Download, Terminal, FileCode, ExternalLink, Play } from 'lucide-react';
import { PYTHON_CODE_SNIPPETS } from '../data/modelsData';

export const PythonCodeHub: React.FC = () => {
  const [activeFile, setActiveFile] = useState<keyof typeof PYTHON_CODE_SNIPPETS>('inferencePy');
  const [copied, setCopied] = useState(false);

  const fileMetadata: Record<keyof typeof PYTHON_CODE_SNIPPETS, { title: string; filename: string; desc: string }> = {
    inferencePy: {
      title: 'Video & Image Inference Pipeline',
      filename: 'inference.py',
      desc: 'OpenCV frame interval extraction, face detection (MTCNN), normalization, and video-level majority/confidence aggregation'
    },
    datasetSplitPy: {
      title: 'Zero-Leakage Dataset Partitioner',
      filename: 'dataset_split.py',
      desc: 'Partitions FaceForensics++ / Celeb-DF into Train/Val/Test with strict actor/subject clustering to avoid data leakage'
    },
    modelsPy: {
      title: 'Deep Transfer Learning Architectures',
      filename: 'models.py',
      desc: 'PyTorch definitions of EfficientNet-B4, Xception Net, and ResNet-50 with custom classification heads'
    },
    trainPy: {
      title: 'Model Training & Evaluation Loop',
      filename: 'train.py',
      desc: 'End-to-end PyTorch training with BCEWithLogitsLoss, AdamW, Cosine Annealing, and ROC-AUC validation'
    },
    requirements: {
      title: 'Python Dependencies & Libraries',
      filename: 'requirements.txt',
      desc: 'Production pip requirements: PyTorch, torchvision, timm, facenet-pytorch, opencv-python, scikit-learn'
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(PYTHON_CODE_SNIPPETS[activeFile]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const filename = fileMetadata[activeFile].filename;
    const blob = new Blob([PYTHON_CODE_SNIPPETS[activeFile]], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 font-mono text-neutral-200">
      {/* Header */}
      <div className="bg-neutral-900 rounded-2xl p-6 sm:p-8 shadow-xl border border-neutral-800">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-950 border border-neutral-800 text-xs font-semibold text-rose-400 mb-3">
            <Terminal className="w-3.5 h-3.5" />
            <span>PRODUCTION PYTORCH & OPENCV BACKEND SCRIPTS</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-black tracking-tight text-white mb-2 uppercase">
            Python Deepfake Detection Codebase
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400 leading-relaxed">
            Full, modular Python implementation designed for execution on GPU workstations or Google Colab.
            Includes zero-leakage dataset partitioners, transfer-learning model definitions, and OpenCV temporal frame extraction.
          </p>
        </div>
      </div>

      {/* Code Viewer Container */}
      <div className="bg-neutral-900 rounded-2xl border border-neutral-800 overflow-hidden shadow-xl">
        {/* File Tabs */}
        <div className="flex flex-wrap items-center justify-between border-b border-neutral-800 bg-neutral-950 px-4 py-2 gap-2">
          <div className="flex flex-wrap items-center gap-1">
            {(Object.keys(fileMetadata) as Array<keyof typeof PYTHON_CODE_SNIPPETS>).map((key) => {
              const file = fileMetadata[key];
              const isActive = activeFile === key;
              return (
                <button
                  key={key}
                  onClick={() => setActiveFile(key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 cursor-pointer ${
                    isActive
                      ? 'bg-neutral-800 text-white border border-neutral-700 font-bold'
                      : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5 text-rose-400" />
                  <span>{file.filename}</span>
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 rounded-lg border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-medium transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Code'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold transition-colors flex items-center gap-1.5 shadow-md shadow-rose-950/50 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download {fileMetadata[activeFile].filename}</span>
            </button>
          </div>
        </div>

        {/* File Description Header */}
        <div className="px-6 py-3.5 border-b border-neutral-800/80 bg-neutral-950/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="text-xs font-bold text-white block">
              {fileMetadata[activeFile].title}
            </span>
            <span className="text-xs text-neutral-400">
              {fileMetadata[activeFile].desc}
            </span>
          </div>

          <div className="text-xs text-neutral-500 font-mono">
            CLI: <span className="text-rose-400 font-bold">python {fileMetadata[activeFile].filename}</span>
          </div>
        </div>

        {/* Code Content Box */}
        <div className="p-6 bg-black text-neutral-100 overflow-x-auto selection:bg-rose-900 selection:text-white">
          <pre className="font-mono text-xs leading-relaxed">
            <code>{PYTHON_CODE_SNIPPETS[activeFile]}</code>
          </pre>
        </div>
      </div>

      {/* Quick Terminal Guide */}
      <div className="p-6 rounded-2xl bg-neutral-900 border border-neutral-800">
        <h3 className="text-xs font-bold text-white uppercase tracking-wider mb-2 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span>Execution Protocol on Local Host or Cloud Compute</span>
        </h3>
        <div className="space-y-2 text-xs font-mono text-neutral-400 bg-neutral-950 p-4 rounded-xl border border-neutral-800">
          <div className="text-neutral-500"># 1. Environment initialization & dependency provisioning</div>
          <div className="text-emerald-400 font-bold">pip install -r requirements.txt</div>

          <div className="mt-3 text-neutral-500"># 2. Partition dataset with zero subject/actor identity leakage</div>
          <div className="text-emerald-400 font-bold">python dataset_split.py --dataset_meta metadata.json --train 0.70 --val 0.15 --test 0.15</div>

          <div className="mt-3 text-neutral-500"># 3. Fine-tune model with cosine-annealing transfer learning</div>
          <div className="text-emerald-400 font-bold">python train.py --model efficientnet_b4 --epochs 25 --batch_size 32</div>

          <div className="mt-3 text-neutral-500"># 4. Run video inference with OpenCV frame interval extraction</div>
          <div className="text-emerald-400 font-bold">python inference.py --video suspect_evidence.mp4 --interval 10 --model efficientnet_b4</div>
        </div>
      </div>
    </div>
  );
};
