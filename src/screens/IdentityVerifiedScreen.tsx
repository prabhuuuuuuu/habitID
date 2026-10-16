import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { SignalCard } from '../components/SignalCard';
import { maskAadhaar } from '../services/aadhaarValidator';

const VERDICT_CONFIG = {
  VERIFIED: {
    color: '#10b981',
    bg: '#ecfdf5',
    icon: '✓',
    title: 'Identity Verified',
    subtitle: 'All trust signals passed. This session has been authenticated.',
  },
  REVIEW: {
    color: '#f59e0b',
    bg: '#fffbeb',
    icon: '!',
    title: 'Requires Review',
    subtitle: 'Some signals need manual review. Verification is pending.',
  },
  SUSPICIOUS: {
    color: '#ef4444',
    bg: '#fef2f2',
    icon: '✕',
    title: 'Suspicious Activity',
    subtitle: 'Behavioral signals indicate potential fraud risk.',
  },
};

export function IdentityVerifiedScreen() {
  const navigate = useNavigate();
  const { scoreResult, signals } = useApp();

  const [displayScore, setDisplayScore] = useState(0);
  const [ringOffset, setRingOffset] = useState(377); // full circumference
  const animRef = useRef<number>(0);

  useEffect(() => {
    if (!scoreResult) {
      navigate('/');
      return;
    }

    const target = scoreResult.totalScore;
    const circumference = 377; // 2π × 60
    const targetOffset = circumference - (target / 100) * circumference;
    const duration = 1600;
    const startTime = performance.now();

    function tick(now: number) {
      const t = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplayScore(Math.round(target * eased));
      setRingOffset(circumference - (circumference - targetOffset) * eased);
      if (t < 1) animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [scoreResult]);

  if (!scoreResult) return null;

  const config = VERDICT_CONFIG[scoreResult.verdict];
  const maskedAadhaar = maskAadhaar(signals.aadhaarNumber);

  return (
    <div className="min-h-screen bg-[#fbfbfd] flex flex-col max-w-md mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-5">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-[#08172c] rounded-xl flex items-center justify-center">
            <span className="text-white text-xs font-black">HI</span>
          </div>
          <span className="text-xs font-bold text-gray-400 tracking-wider">habitID</span>
        </div>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2">
          <rect x="3" y="11" width="18" height="11" rx="2" />
          <path d="M7 11V7a5 5 0 0110 0v4" />
        </svg>
      </div>

      <div className="flex-1 overflow-y-auto px-6 pb-10">
        {/* Score ring + verdict */}
        <div className="flex flex-col items-center py-6 mb-6 fade-slide">
          <div className="relative" style={{ width: 160, height: 160 }}>
            <svg width="160" height="160" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="80" cy="80" r="60" fill="none" stroke="#e5e7eb" strokeWidth="14" />
              <circle
                cx="80"
                cy="80"
                r="60"
                fill="none"
                stroke={config.color}
                strokeWidth="14"
                strokeLinecap="round"
                strokeDasharray="377"
                strokeDashoffset={ringOffset}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span
                className="text-4xl font-black leading-none"
                style={{ color: '#08172c', letterSpacing: '-2px' }}
              >
                {displayScore}
              </span>
              <span className="text-[10px] text-gray-400 font-bold">/100</span>
            </div>
          </div>

          {/* Verdict badge */}
          <div
            className="mt-5 flex items-center gap-2 px-5 py-2.5 rounded-full"
            style={{ background: config.bg }}
          >
            <span
              className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-black text-white"
              style={{ background: config.color }}
            >
              {config.icon}
            </span>
            <span className="text-sm font-black" style={{ color: config.color }}>
              {config.title}
            </span>
          </div>

          <p className="text-xs text-gray-400 text-center mt-3 leading-relaxed px-4">
            {config.subtitle}
          </p>
        </div>

        {/* Identity summary card */}
        <div
          className="rounded-2xl p-4 mb-6 fade-slide"
          style={{ animationDelay: '100ms', background: '#f3f5f7' }}
        >
          <div className="text-[10px] font-bold text-gray-400 tracking-widest mb-3">
            VERIFIED IDENTITY
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Phone</span>
              <span className="font-bold text-[#08172c]">{signals.phoneNumber || '—'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Aadhaar</span>
              <span className="font-bold text-[#08172c] tracking-widest">{maskedAadhaar}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-500">Computed in</span>
              <span className="font-bold text-[#08172c]">{scoreResult.processingMs}ms</span>
            </div>
          </div>
        </div>

        {/* Signal breakdown */}
        <div className="fade-slide mb-6" style={{ animationDelay: '200ms' }}>
          <div className="text-[10px] font-bold text-gray-400 tracking-widest mb-3">
            SIGNAL BREAKDOWN
          </div>
          <div className="space-y-3">
            {scoreResult.signals.map((sig, i) => (
              <SignalCard
                key={sig.label}
                signal={sig}
                animationDelay={i * 80}
              />
            ))}
          </div>
        </div>

        {/* Action buttons */}
        <div className="space-y-3 fade-slide" style={{ animationDelay: '500ms' }}>
          <button
            className="w-full py-4 rounded-2xl font-bold text-white text-sm bg-[#08172c] active:scale-[0.98] transition-transform"
            onClick={() => navigate('/')}
          >
            Start New Verification
          </button>
          <div className="flex items-center justify-center gap-1.5">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span className="text-[10px] font-bold text-gray-400 tracking-widest">
              POWERED BY habitID
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
