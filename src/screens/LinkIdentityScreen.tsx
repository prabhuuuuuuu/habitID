import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { Header, StepIndicator, ComplianceBadge, PrimaryButton } from '../components/Header';

export function LinkIdentityScreen() {
  const navigate = useNavigate();
  const { setSignals, collector } = useApp();

  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [error, setError] = useState('');

  // Live keystroke counter for the "behavioral capture active" badge
  const [keystrokeCount, setKeystrokeCount] = useState(0);
  const counterRef = useRef(0);

  useEffect(() => {
    // Start capturing behavioral signals the moment this screen mounts
    collector.start();

    // Poll the collector snapshot for live keystroke count
    const interval = setInterval(() => {
      const snap = collector.snapshot();
      if (snap.length !== counterRef.current) {
        counterRef.current = snap.length;
        setKeystrokeCount(snap.length);
      }
    }, 200);

    return () => {
      clearInterval(interval);
      // Do NOT stop the collector here — we stop it when the user submits
    };
  }, [collector]);

  function handleContinue() {
    const trimmedPhone = phone.trim();
    const trimmedName = name.trim();

    if (!trimmedName || trimmedName.split(' ').length < 1) {
      setError('Please enter your full name.');
      return;
    }

    const digits = trimmedPhone.replace(/\D/g, '');
    if (digits.length < 10) {
      setError('Enter a valid 10-digit mobile number.');
      return;
    }

    setError('');

    // Stop collector and capture signals
    const result = collector.stop();

    setSignals({
      keystrokeEvents: result.keystrokeEvents,
      motionSamples: result.motionSamples,
      deviceIsMotionCapable: result.deviceIsMotionCapable,
      typingStartedAt: result.typingStartedAt,
      typingEndedAt: result.typingEndedAt,
      phoneNumber: `+91${digits.slice(-10)}`,
    });

    navigate('/aadhaar');
  }

  return (
    <div className="min-h-screen bg-[#fbfbfd] flex flex-col max-w-md mx-auto">
      <Header />
      <StepIndicator current={0} total={4} />

      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="fade-slide" style={{ animationDelay: '0ms' }}>
          {/* Capture indicator */}
          <div className="inline-flex items-center gap-2 bg-[#ecfdf5] px-3 py-1.5 rounded-full mb-6">
            <span
              className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse-dot"
            />
            <span className="text-[10px] font-bold text-[#10b981] tracking-wider">
              BEHAVIORAL CAPTURE ACTIVE — {keystrokeCount} keys
            </span>
          </div>

          <h1
            className="text-3xl font-black text-[#08172c] leading-tight mb-3"
            style={{ letterSpacing: '-1px' }}
          >
            Link Your Identity
          </h1>
          <p className="text-sm text-gray-400 leading-relaxed mb-8">
            Enter your legal name and mobile number. Your typing pattern is
            being measured in real-time as a behavioral trust signal.
          </p>

          {/* Name field */}
          <label className="text-[10px] font-bold text-gray-400 tracking-widest block mb-2">
            FULL NAME
          </label>
          <div className="relative mb-6">
            <input
              type="text"
              className="w-full bg-[#f3f5f7] rounded-2xl px-4 py-4 text-[#08172c] font-medium text-sm outline-none focus:ring-2 focus:ring-[#08172c]/20 transition"
              placeholder="Arjun Sharma"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoComplete="name"
            />
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-300">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </span>
          </div>

          {/* Phone field */}
          <label className="text-[10px] font-bold text-gray-400 tracking-widest block mb-2">
            MOBILE NUMBER
          </label>
          <div className="flex gap-2 mb-2">
            <div className="bg-[#f3f5f7] rounded-2xl px-4 py-4 flex items-center gap-2 shrink-0">
              <span className="text-xs font-bold text-gray-500">IN</span>
              <span className="text-sm font-bold text-[#08172c]">+91</span>
            </div>
            <div className="relative flex-1">
              <input
                type="tel"
                className="w-full bg-[#f3f5f7] rounded-2xl px-4 py-4 text-[#08172c] font-medium text-sm outline-none focus:ring-2 focus:ring-[#08172c]/20 transition"
                placeholder="98765 43210"
                value={phone}
                onChange={(e) => setPhone(e.target.value.replace(/[^\d\s-]/g, ''))}
                maxLength={14}
                inputMode="tel"
              />
              <span className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-300">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="5" y="2" width="14" height="20" rx="2" />
                  <line x1="12" y1="18" x2="12" y2="18.01" strokeLinecap="round" strokeWidth="2" />
                </svg>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 mb-8">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" strokeLinecap="round" strokeWidth="2.5" />
            </svg>
            <span className="text-[11px] text-gray-400">Verification code will be sent via SMS</span>
          </div>

          {error && (
            <div className="text-xs text-red-500 bg-red-50 rounded-xl px-4 py-3 mb-4 font-medium">
              {error}
            </div>
          )}

          <PrimaryButton text="Begin Secure Handshake" onClick={handleContinue} />

          <ComplianceBadge />

          <div className="flex items-center justify-center gap-1.5 mt-6 mb-8">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
            <span className="text-[10px] font-bold text-[#10b981] tracking-widest">
              END-TO-END ENCRYPTED · VAULT SECURE
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
