import { useEffect, useRef, useState } from 'react';

interface ScoreRingProps {
  score: number;          // 0–100
  size?: number;          // SVG size in px (default 200)
  strokeWidth?: number;   // ring thickness (default 16)
  animate?: boolean;      // animate fill on mount (default true)
  verdict?: 'VERIFIED' | 'REVIEW' | 'SUSPICIOUS';
}

const VERDICT_COLORS = {
  VERIFIED: '#10b981',
  REVIEW: '#f59e0b',
  SUSPICIOUS: '#ef4444',
};

export function ScoreRing({
  score,
  size = 200,
  strokeWidth = 16,
  animate = true,
  verdict = 'VERIFIED',
}: ScoreRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const targetOffset = circumference - (score / 100) * circumference;

  const [displayOffset, setDisplayOffset] = useState(circumference); // starts empty
  const [displayScore, setDisplayScore] = useState(0);
  const animFrameRef = useRef<number>(0);

  const color = VERDICT_COLORS[verdict];

  useEffect(() => {
    if (!animate) {
      setDisplayOffset(targetOffset);
      setDisplayScore(score);
      return;
    }

    const duration = 1400; // ms
    const startTime = performance.now();
    const startOffset = circumference;
    const startScore = 0;

    function tick(now: number) {
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3);

      setDisplayOffset(startOffset + (targetOffset - startOffset) * eased);
      setDisplayScore(Math.round(startScore + (score - startScore) * eased));

      if (t < 1) {
        animFrameRef.current = requestAnimationFrame(tick);
      }
    }

    animFrameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [score, animate, circumference, targetOffset]);

  const cx = size / 2;
  const cy = size / 2;

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} className="rotate-[-90deg]">
        {/* Track */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
        />
        {/* Progress arc */}
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={displayOffset}
          style={{ transition: animate ? undefined : 'none' }}
        />
      </svg>
      {/* Score text overlay */}
      <div
        className="absolute flex flex-col items-center justify-center"
        style={{ width: size, height: size }}
      >
        <span
          className="text-5xl font-black"
          style={{ color: '#08172c', letterSpacing: '-2px' }}
        >
          {displayScore}
        </span>
        <span className="text-xs font-bold text-gray-400 tracking-widest mt-1">
          / 100
        </span>
      </div>
    </div>
  );
}

// Wrapper that positions the SVG and overlay correctly
export function ScoreRingContainer({
  score,
  verdict,
  size = 200,
}: {
  score: number;
  verdict: 'VERIFIED' | 'REVIEW' | 'SUSPICIOUS';
  size?: number;
}) {
  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <ScoreRing score={score} verdict={verdict} size={size} />
    </div>
  );
}
