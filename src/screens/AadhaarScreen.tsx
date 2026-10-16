import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { Header, StepIndicator, PrimaryButton } from '../components/Header';
import { validateAadhaar, formatAadhaar } from '../services/aadhaarValidator';

export function AadhaarScreen() {
  const navigate = useNavigate();
  const { setSignals } = useApp();

  const [raw, setRaw] = useState('');
  const [touched, setTouched] = useState(false);

  const formatted = formatAadhaar(raw);
  const validation = validateAadhaar(raw);
  const digitCount = raw.replace(/\D/g, '').length;

  function handleInput(value: string) {
    // Allow only digits, strip everything else, max 12 digits
    const digits = value.replace(/\D/g, '').slice(0, 12);
    setRaw(digits);
    setTouched(true);
  }

  function handleContinue() {
    if (!validation.valid) return;

    setSignals({
      aadhaarNumber: raw.replace(/\D/g, ''),
      aadhaarValid: true,
    });

    navigate('/otp');
  }

  const showError = touched && raw.length > 0 && !validation.valid && digitCount >= 4;
  const showSuccess = validation.valid;

  return (
    <div className="min-h-screen bg-[#fbfbfd] flex flex-col max-w-md mx-auto">
      <Header />
      <StepIndicator current={1} total={4} />

      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="fade-slide">
          {/* Icon */}
          <div className="w-12 h-12 rounded-full bg-[#eff4ff] flex items-center justify-center mb-6">
            <span className="text-2xl">🪪</span>
          </div>

          <h1
            className="text-3xl font-black text-[#08172c] leading-tight mb-3"
            style={{ letterSpacing: '-1px' }}
          >
            Aadhaar Verification
          </h1>
          <p className="text-sm text-gray-400 leading-relaxed mb-8">
            Enter your 12-digit Aadhaar number. We validate it locally using the
            official UIDAI Verhoeff check-digit algorithm — no data leaves your device.
          </p>

          <label className="text-[10px] font-bold text-gray-400 tracking-widest block mb-2">
            AADHAAR NUMBER
          </label>

          <div className="relative mb-2">
            <input
              type="tel"
              inputMode="numeric"
              className="w-full bg-[#f3f5f7] rounded-2xl px-4 py-4 text-[#08172c] font-medium text-sm outline-none transition tracking-widest"
              style={{
                border: showSuccess
                  ? '2px solid #10b981'
                  : showError
                  ? '2px solid #ef4444'
                  : '2px solid transparent',
              }}
              placeholder="XXXX XXXX XXXX"
              value={formatted}
              onChange={(e) => handleInput(e.target.value)}
              maxLength={14} // 12 digits + 2 spaces
            />
            {showSuccess && (
              <span className="absolute right-4 top-1/2 -translate-y-1/2">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M8 12l3 3 5-6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            )}
          </div>

          {/* Live validation feedback */}
          <div className="mb-8 min-h-[40px]">
            {showError && (
              <div className="flex items-center gap-2 text-xs text-red-500 bg-red-50 rounded-xl px-3 py-2">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" strokeLinecap="round" />
                  <line x1="9" y1="9" x2="15" y2="15" strokeLinecap="round" />
                </svg>
                {validation.reason}
              </div>
            )}
            {showSuccess && (
              <div className="flex items-center gap-2 text-xs text-emerald-600 bg-[#ecfdf5] rounded-xl px-3 py-2">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
                Verhoeff check-digit validated — Aadhaar format is correct
              </div>
            )}
            {!showError && !showSuccess && digitCount > 0 && (
              <div className="text-xs text-gray-400 px-1">
                {12 - digitCount} more digit{12 - digitCount !== 1 ? 's' : ''} needed
              </div>
            )}
          </div>

          {/* Demo hint */}
          <div className="bg-[#fffbeb] border border-amber-200 rounded-2xl p-4 mb-4">
            <div className="text-[10px] font-bold text-amber-600 tracking-widest mb-1">DEMO MODE</div>
            <div className="text-xs text-amber-700 mb-2">
              Use one of these valid demo numbers (real Verhoeff check digits):
            </div>
            <div className="space-y-1">
              {['4111 1111 1114', '2345 6789 0124', '4991 1866 5088'].map((n) => (
                <button
                  key={n}
                  onClick={() => { setRaw(n.replace(/\s/g, '')); setTouched(true); }}
                  className="block w-full text-left text-xs font-black text-amber-800 bg-amber-100 hover:bg-amber-200 rounded-xl px-3 py-1.5 tracking-widest transition"
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Algorithm info box */}
          <div className="bg-[#f3f5f7] rounded-2xl p-4 mb-8">
            <div className="text-[10px] font-bold text-gray-400 tracking-widest mb-2">
              HOW WE VALIDATE
            </div>
            <div className="space-y-2">
              {[
                ['Format check', '12 digits, no leading 0 or 1'],
                ['Verhoeff algorithm', 'Check-digit computation (dihedral group D5)'],
                ['Data residency', 'Fully local — never transmitted'],
              ].map(([label, desc]) => (
                <div key={label} className="flex items-start gap-2">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5" className="mt-0.5 shrink-0">
                    <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="text-xs text-gray-500">
                    <strong className="text-[#08172c]">{label}:</strong> {desc}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <PrimaryButton
            text="Confirm & Continue"
            onClick={handleContinue}
            disabled={!validation.valid}
          />
        </div>
      </div>
    </div>
  );
}
