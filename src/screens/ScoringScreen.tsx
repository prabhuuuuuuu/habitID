import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { computeTrustScore } from '../services/scoringEngine';
import type { SignalScore } from '../models/TrustScoreResult';

// Each signal takes ~400ms to "compute" for dramatic effect
const SIGNAL_REVEAL_INTERVAL = 480;

const PROCESSING_LABELS = [
  'Analysing keystroke rhythm...',
  'Measuring tilt variance...',
  'Running Verhoeff algorithm...',
  'Calculating OTP timing...',
  'Querying MSISDN age tables...',
  'Validating phone format...',
  'Cross-checking behavioral consistency...',
  'Computing trust vector...',
];

export function ScoringScreen() {
  const navigate = useNavigate();
  const { signals, setScoreResult } = useApp();

  const [_revealedCount, setRevealedCount] = useState(0); void _revealedCount;
  const [processingLabel, setProcessingLabel] = useState(PROCESSING_LABELS[0]);
  const [scoreReady, setScoreReady] = useState(false);
  const [runningTotal, setRunningTotal] = useState(0);
  const [signalScores, setSignalScores] = useState<SignalScore[]>([]);
  const [progress, setProgress] = useState(0);

  const resultRef = useRef<ReturnType<typeof computeTrustScore> | null>(null);
  const _labelIndexRef = useRef(0); void _labelIndexRef;

  useEffect(() => {
    // Compute score synchronously (it's fast, < 5ms)
    const result = computeTrustScore(signals);
    resultRef.current = result;
    setScoreResult(result);

    const totalSignals = result.signals.length;

    // Progressively reveal each signal with delay
    let revealIdx = 0;
    let labelIdx = 0;

    const interval = setInterval(() => {
      if (revealIdx < totalSignals) {
        const sig = result.signals[revealIdx];
        revealIdx++;
        labelIdx = Math.min(labelIdx + 1, PROCESSING_LABELS.length - 1);

        setRevealedCount(revealIdx);
        setProcessingLabel(PROCESSING_LABELS[labelIdx]);
        setProgress(Math.round((revealIdx / totalSignals) * 100));
        setRunningTotal((prev) => prev + sig.score);
        setSignalScores((prev) => [...prev, sig]);
      } else {
        clearInterval(interval);

        // Brief pause before navigating to results
        setTimeout(() => {
          setScoreReady(true);
          setTimeout(() => navigate('/result'), 800);
        }, 600);
      }
    }, SIGNAL_REVEAL_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  const RISK_COLORS: Record<string, string> = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
  };

  return (
    <div className="min-h-screen bg-[#08172c] flex flex-col max-w-md mx-auto text-white">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-white/10 rounded-xl flex items-center justify-center">
            <span className="text-white text-xs font-black">TS</span>
          </div>
          <span className="text-xs font-bold text-white/40 tracking-wider">TRUSTSCORE AI</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse-dot" />
          <span className="text-[10px] text-white/40 font-bold tracking-wider">COMPUTING</span>
        </div>
      </div>

      <div className="flex-1 px-6 py-4">
        {/* Running score */}
        <div className="text-center mb-8">
          <div
            className="text-7xl font-black leading-none mb-1 transition-all duration-300"
            style={{
              color: scoreReady ? '#10b981' : 'white',
              letterSpacing: '-3px',
            }}
          >
            {runningTotal}
          </div>
          <div className="text-white/40 text-xs font-bold tracking-widest">/ 100 TRUST SCORE</div>
        </div>

        {/* Progress bar */}
        <div className="mb-2">
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#10b981] rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        <div className="flex items-center justify-between mb-8">
          <span className="text-[10px] text-white/40 font-bold tracking-wider">{processingLabel}</span>
          <span className="text-[10px] text-white/40 font-bold">{progress}%</span>
        </div>

        {/* Signal reveal list */}
        <div className="space-y-3">
          {signalScores.map((sig, i) => (
            <div
              key={sig.label}
              className="flex items-center justify-between bg-white/5 rounded-2xl px-4 py-3 fade-slide"
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <div className="flex items-center gap-3">
                <span className="text-lg">{sig.icon}</span>
                <div>
                  <div className="text-xs font-bold text-white">{sig.label}</div>
                  <div className="text-[10px] text-white/40">{sig.detail.slice(0, 48)}{sig.detail.length > 48 ? '…' : ''}</div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ background: RISK_COLORS[sig.risk] }}
                />
                <span className="text-sm font-black text-white">
                  {sig.score}
                  <span className="text-white/30 text-xs font-normal">/{sig.maxScore}</span>
                </span>
              </div>
            </div>
          ))}

          {/* Pending signals (greyed out placeholders) */}
          {Array.from({ length: Math.max(0, 7 - signalScores.length) }).map((_, i) => (
            <div
              key={`pending-${i}`}
              className="flex items-center gap-3 bg-white/5 rounded-2xl px-4 py-3 opacity-30"
            >
              <div className="w-7 h-7 rounded-full bg-white/10" />
              <div className="flex-1">
                <div className="h-2 bg-white/20 rounded-full w-1/2 mb-1" />
                <div className="h-1.5 bg-white/10 rounded-full w-3/4" />
              </div>
              <div className="w-8 h-4 bg-white/10 rounded-full" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
