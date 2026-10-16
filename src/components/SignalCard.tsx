import type { SignalScore } from '../models/TrustScoreResult';

interface SignalCardProps {
  signal: SignalScore;
  animationDelay?: number; // ms
  revealed?: boolean;
}

const RISK_COLORS = {
  low: { bar: '#10b981', bg: '#ecfdf5', text: '#065f46' },
  medium: { bar: '#f59e0b', bg: '#fffbeb', text: '#92400e' },
  high: { bar: '#ef4444', bg: '#fef2f2', text: '#991b1b' },
};

export function SignalCard({ signal, animationDelay = 0, revealed = true }: SignalCardProps) {
  const colors = RISK_COLORS[signal.risk];
  const pct = Math.round((signal.score / signal.maxScore) * 100);

  return (
    <div
      className="rounded-2xl p-4 border border-gray-100 bg-white shadow-sm fade-slide"
      style={{
        animationDelay: `${animationDelay}ms`,
        opacity: revealed ? undefined : 0,
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xl">{signal.icon}</span>
          <div>
            <div className="text-sm font-bold text-[#08172c]">{signal.label}</div>
            <div className="text-xs text-gray-400">{signal.description}</div>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className="text-sm font-black text-[#08172c]">
            {signal.score}
            <span className="text-xs font-normal text-gray-400">/{signal.maxScore}</span>
          </span>
          <span
            className="text-[10px] font-bold px-2 py-0.5 rounded-full"
            style={{ background: colors.bg, color: colors.text }}
          >
            {signal.risk.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Score bar */}
      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden mb-2">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: colors.bar,
            transitionDelay: `${animationDelay + 200}ms`,
          }}
        />
      </div>

      <div className="text-[11px] text-gray-500">{signal.detail}</div>
    </div>
  );
}
