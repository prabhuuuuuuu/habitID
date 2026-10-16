import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { Header, StepIndicator, PrimaryButton } from '../components/Header';

const MOCK_OTP = '123456';
const OTP_LENGTH = 6;

export function OtpScreen() {
  const navigate = useNavigate();
  const { signals, setSignals } = useApp();

  const [digits, setDigits] = useState<string[]>(Array(OTP_LENGTH).fill(''));
  const [error, setError] = useState('');
  const [shake, setShake] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Record when OTP screen appeared (used implicitly via otpStartedAt signal)
  const _shownAt = useRef(performance.now()); void _shownAt;

  useEffect(() => {
    setSignals({ otpStartedAt: performance.now() });
    inputRefs.current[0]?.focus();
  }, []);

  const enteredOtp = digits.join('');

  function handleDigitChange(index: number, value: string) {
    const char = value.replace(/\D/g, '').slice(-1);
    const next = [...digits];
    next[index] = char;
    setDigits(next);

    if (char && index < OTP_LENGTH - 1) {
      inputRefs.current[index + 1]?.focus();
    }
    setError('');
  }

  function handleKeyDown(index: number, e: React.KeyboardEvent) {
    if (e.key === 'Backspace') {
      if (!digits[index] && index > 0) {
        const next = [...digits];
        next[index - 1] = '';
        setDigits(next);
        inputRefs.current[index - 1]?.focus();
      }
    }
  }

  function handlePaste(e: React.ClipboardEvent) {
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, OTP_LENGTH);
    if (pasted.length === OTP_LENGTH) {
      setDigits(pasted.split(''));
      inputRefs.current[OTP_LENGTH - 1]?.focus();
    }
    e.preventDefault();
  }

  function handleVerify() {
    if (enteredOtp.length < OTP_LENGTH) {
      setError('Enter all 6 digits.');
      return;
    }

    if (enteredOtp !== MOCK_OTP) {
      setError(`Incorrect OTP. (Hint: use ${MOCK_OTP} for demo)`);
      setShake(true);
      setTimeout(() => setShake(false), 600);
      return;
    }

    // Record OTP completion time
    setSignals({ otpCompletedAt: performance.now() });
    navigate('/scoring');
  }

  const maskedPhone = signals.phoneNumber
    ? `+91 ****  **${signals.phoneNumber.slice(-4)}`
    : '+91 ••••• •••••';

  return (
    <div className="min-h-screen bg-[#fbfbfd] flex flex-col max-w-md mx-auto">
      <Header />
      <StepIndicator current={2} total={4} />

      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="fade-slide">
          {/* Check icon */}
          <div className="w-12 h-12 rounded-full bg-[#ecfdf5] flex items-center justify-center mb-6">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5">
              <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          <h1
            className="text-3xl font-black text-[#08172c] leading-tight mb-3"
            style={{ letterSpacing: '-1px' }}
          >
            Verify your identity
          </h1>
          <p className="text-sm text-gray-400 leading-relaxed mb-2">
            A 6-digit code has been sent to
          </p>
          <p className="text-sm font-bold text-[#08172c] mb-8">{maskedPhone}</p>

          {/* OTP boxes */}
          <div
            className={`flex gap-3 justify-center mb-4 transition-transform ${shake ? 'animate-[shake_0.4s_ease-in-out]' : ''}`}
            style={shake ? { animation: 'shake 0.4s ease-in-out' } : {}}
            onPaste={handlePaste}
          >
            {digits.map((d, i) => (
              <input
                key={i}
                ref={(el) => { inputRefs.current[i] = el; }}
                type="tel"
                inputMode="numeric"
                maxLength={1}
                value={d}
                onChange={(e) => handleDigitChange(i, e.target.value)}
                onKeyDown={(e) => handleKeyDown(i, e)}
                className="w-12 h-14 text-center text-xl font-black rounded-2xl bg-[#f3f5f7] outline-none transition-all duration-150"
                style={{
                  border: d
                    ? '2px solid #08172c'
                    : error
                    ? '2px solid #ef4444'
                    : '2px solid transparent',
                  color: '#08172c',
                }}
              />
            ))}
          </div>

          {error && (
            <div className="text-xs text-red-500 bg-red-50 rounded-xl px-4 py-3 mb-6 text-center font-medium">
              {error}
            </div>
          )}

          {!error && (
            <p className="text-xs text-gray-400 text-center mb-8">
              Timing your response as a behavioral trust signal
            </p>
          )}

          {/* Demo notice */}
          <div className="bg-[#fffbeb] border border-amber-200 rounded-2xl p-4 mb-8">
            <div className="text-[10px] font-bold text-amber-600 tracking-widest mb-1">DEMO MODE</div>
            <div className="text-xs text-amber-700">
              This is a demonstration. Use OTP <strong className="font-black">123456</strong> to proceed.
              Your response time is being measured as a fraud-detection signal.
            </div>
          </div>

          <PrimaryButton
            text="Verify & Analyse"
            onClick={handleVerify}
            disabled={enteredOtp.length < OTP_LENGTH}
          />

          <div className="text-center mt-6">
            <button className="text-xs text-gray-400 underline underline-offset-2">
              Resend OTP
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
