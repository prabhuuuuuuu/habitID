interface HeaderProps {
  rightElement?: React.ReactNode;
  showLock?: boolean;
}

export function Header({ rightElement, showLock = true }: HeaderProps) {
  return (
    <div className="flex items-center justify-between px-6 py-4">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-[#08172c] rounded-xl flex items-center justify-center">
          <span className="text-white text-xs font-black">TS</span>
        </div>
        <span className="text-xs font-bold text-gray-400 tracking-wider">TRUSTSCORE AI</span>
      </div>
      <div className="flex items-center gap-2">
        {rightElement}
        {showLock && (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2">
            <rect x="3" y="11" width="18" height="11" rx="2" />
            <path d="M7 11V7a5 5 0 0110 0v4" />
          </svg>
        )}
      </div>
    </div>
  );
}

export function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-1.5 px-6 pb-2">
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          className="h-1 flex-1 rounded-full transition-all duration-500"
          style={{
            background: i < current ? '#08172c' : i === current ? '#6b7280' : '#e5e7eb',
          }}
        />
      ))}
    </div>
  );
}

export function ComplianceBadge() {
  return (
    <div className="flex flex-col gap-2 mt-4">
      <div className="flex items-center gap-3 p-3 rounded-xl bg-[#eff4ff]">
        <div className="w-7 h-7 rounded-full bg-[#3b82f6]/20 flex items-center justify-center">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2.5">
            <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
        </div>
        <div>
          <div className="text-[10px] text-blue-600 font-bold tracking-wider">REGULATORY STANDARD</div>
          <div className="text-xs font-bold text-[#08172c]">RBI 2024 Compliant</div>
        </div>
      </div>
      <div className="flex items-center gap-3 p-3 rounded-xl bg-[#ecfdf5]">
        <div className="w-7 h-7 rounded-full bg-[#10b981]/20 flex items-center justify-center">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5">
            <rect x="3" y="11" width="18" height="11" rx="2" />
            <path d="M7 11V7a5 5 0 0110 0v4" />
          </svg>
        </div>
        <div>
          <div className="text-[10px] text-emerald-600 font-bold tracking-wider">DATA PROTECTION</div>
          <div className="text-xs font-bold text-[#08172c]">Privacy-First: DPDP 2023</div>
        </div>
      </div>
    </div>
  );
}

export function PrimaryButton({
  text,
  onClick,
  disabled = false,
  loading = false,
}: {
  text: string;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className="w-full py-4 px-6 rounded-2xl font-bold text-white text-sm tracking-wide transition-all duration-200 active:scale-[0.98]"
      style={{
        background: disabled || loading ? '#9ca3af' : '#08172c',
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
      }}
    >
      {loading ? (
        <span className="flex items-center justify-center gap-2">
          <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          </svg>
          Processing...
        </span>
      ) : text}
    </button>
  );
}
