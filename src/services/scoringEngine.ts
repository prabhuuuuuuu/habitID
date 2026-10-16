/**
 * ScoringEngine
 *
 * Computes a 0–100 trust score from collected behavioral signals.
 *
 * Signal weights:
 *   Typing rhythm entropy     25 pts  — is the inter-key timing human-like?
 *   Pointer/tilt stability    15 pts  — calm hand movement
 *   Aadhaar validity          15 pts  — Verhoeff check passes
 *   OTP completion timing     20 pts  — human response window (8–35s)
 *   SIM age heuristic          5 pts  — older prefix = established subscriber
 *   Phone format validity      5 pts  — valid India MSISDN
 *   Behavioral consistency    15 pts  — does typing pace agree with OTP pace?
 *
 * NOTE: signals are intentionally weighted so that static/deterministic
 * checks (Aadhaar + SIM + phone = 25 pts) can never alone push a session
 * to VERIFIED (70+). Genuine, internally-consistent behavior is required.
 */

import type { BehavioralSignals } from '../models/BehavioralSignal';
import type { TrustScoreResult, SignalScore } from '../models/TrustScoreResult';
import {
  typingRhythmEntropy,
  typingWPM,
  motionStability,
} from './behavioralCollector';
import { validateAadhaar } from './aadhaarValidator';
import { estimateSimAge } from './simAgeEstimator';

// Valid India mobile: starts with 6-9, exactly 10 digits
const INDIA_PHONE_RE = /^[6-9]\d{9}$/;

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

type TypingPace = 'none' | 'slow' | 'normal' | 'fast';
type OtpPace = 'instant' | 'quick' | 'human' | 'slow';

function classifyTypingPace(wpm: number, keystrokeCount: number): TypingPace {
  if (keystrokeCount < 4 || wpm === 0) return 'none';
  if (wpm < 10) return 'slow';
  if (wpm > 150) return 'fast';
  return 'normal';
}

function classifyOtpPace(otpSeconds: number): OtpPace {
  if (otpSeconds < 5) return 'instant';
  if (otpSeconds < 8) return 'quick';
  if (otpSeconds <= 35) return 'human';
  return 'slow';
}

/**
 * Cross-checks the pace of typing (name/phone entry) against the pace of
 * OTP entry. A person who deliberately types very slowly (as if trying to
 * look "careful") but then answers the OTP almost instantly is behaviorally
 * inconsistent — the two signals should not be scored independently.
 */
function behavioralConsistency(
  wpm: number,
  keystrokeCount: number,
  otpSeconds: number | null,
): { score: number; detail: string } {
  if (otpSeconds === null) {
    return { score: 5, detail: 'OTP not completed — consistency check skipped' };
  }

  const typingPace = classifyTypingPace(wpm, keystrokeCount);
  const otpPace = classifyOtpPace(otpSeconds);

  if (typingPace === 'slow' && (otpPace === 'instant' || otpPace === 'quick')) {
    return {
      score: 1,
      detail: `Typed deliberately slowly (${wpm} WPM) but answered the OTP in ${otpSeconds.toFixed(1)}s — mismatched pace typical of a rehearsed or scripted submission`,
    };
  }

  if (typingPace === 'none' && otpPace === 'instant') {
    return {
      score: 3,
      detail: `No typing rhythm captured and OTP entered in ${otpSeconds.toFixed(1)}s — cannot confirm genuine human pacing`,
    };
  }

  if (typingPace === 'fast') {
    return {
      score: 5,
      detail: `Implausibly fast typing (${wpm} WPM) — bot-like input pattern regardless of OTP timing`,
    };
  }

  if (typingPace === 'slow' && (otpPace === 'human' || otpPace === 'slow')) {
    return {
      score: 15,
      detail: 'Careful, unhurried typing matched by an equally deliberate OTP response — consistent',
    };
  }

  if (typingPace === 'normal' && otpPace === 'instant') {
    return {
      score: 8,
      detail: 'Fluent typing paired with a very quick OTP entry — plausible (e.g. SMS autofill), mild caution',
    };
  }

  if (typingPace === 'normal' && (otpPace === 'human' || otpPace === 'quick')) {
    return {
      score: 15,
      detail: 'Typing pace and OTP response time are consistent with genuine human behavior',
    };
  }

  return { score: 9, detail: 'No strong pace correlation available' };
}

export function computeTrustScore(signals: BehavioralSignals): TrustScoreResult {
  const start = performance.now();

  const signalScores: SignalScore[] = [];

  // ── 1. Typing Rhythm Entropy (25 pts) ──────────────────────────────────
  const { ratio: rhythmRatio, cv: rhythmCv } = typingRhythmEntropy(signals.keystrokeEvents);
  const wpm = typingWPM(signals.keystrokeEvents);
  const keystrokeCount = signals.keystrokeEvents.length;

  // WPM gate: very slow (deliberately hesitant) or very fast (automated) penalizes
  let wpmMod = 1.0;
  if (wpm === 0) wpmMod = 0.5; // no data at all
  else if (wpm < 5) wpmMod = 0.25; // near-motionless, deliberately slow typing
  else if (wpm < 10) wpmMod = 0.45; // unusually slow
  else if (wpm > 200) wpmMod = 0.4; // implausibly fast
  else if (wpm >= 15 && wpm <= 100) wpmMod = 1.0;
  else wpmMod = 0.85;

  const typingRaw = rhythmRatio * wpmMod;
  const typingScore = Math.round(clamp(typingRaw * 25, 0, 25));

  const cvLabel = rhythmCv === null ? 'n/a' : rhythmCv.toFixed(2);
  let typingDetail: string;
  if (keystrokeCount < 4) {
    typingDetail = 'Insufficient keystrokes captured';
  } else if (wpm < 10) {
    typingDetail = `${keystrokeCount} keys at ${wpm} WPM — unusually slow (CV ${cvLabel})`;
  } else if (wpm > 150) {
    typingDetail = `${wpm} WPM detected — implausibly fast (CV ${cvLabel})`;
  } else if (rhythmRatio >= 0.72) {
    typingDetail = `${keystrokeCount} keystrokes, ${wpm} WPM, timing CV ${cvLabel} — natural rhythm`;
  } else if (rhythmRatio >= 0.4) {
    typingDetail = `${keystrokeCount} keystrokes, ${wpm} WPM, timing CV ${cvLabel} — somewhat atypical rhythm`;
  } else {
    typingDetail = `${keystrokeCount} keystrokes, ${wpm} WPM, timing CV ${cvLabel} — too uniform/erratic for a typical human`;
  }

  signalScores.push({
    label: 'Typing Rhythm',
    description: 'Inter-keystroke timing entropy analysis',
    score: typingScore,
    maxScore: 25,
    icon: '⌨',
    detail: typingDetail,
    risk: typingScore >= 18 ? 'low' : typingScore >= 10 ? 'medium' : 'high',
  });

  // ── 2. Pointer / Tilt Stability (15 pts) ───────────────────────────────
  const stabilityRatio = motionStability(signals.motionSamples);
  const sampleCount = signals.motionSamples.length;
  const stabilityScore = Math.round(clamp(stabilityRatio * 15, 0, 15));

  const motionSource = signals.deviceIsMotionCapable
    ? 'accelerometer'
    : 'mouse velocity proxy';

  const stabilityDetail =
    sampleCount < 5
      ? `No movement data captured (${motionSource})`
      : `${sampleCount} samples via ${motionSource}, stability ratio ${(stabilityRatio * 100).toFixed(0)}%`;

  signalScores.push({
    label: 'Tilt / Motion',
    description: signals.deviceIsMotionCapable
      ? 'Device accelerometer — hand tilt during typing'
      : 'Mouse velocity proxy — pointer movement pattern',
    score: stabilityScore,
    maxScore: 15,
    icon: '📱',
    detail: stabilityDetail,
    risk: stabilityScore >= 11 ? 'low' : stabilityScore >= 6 ? 'medium' : 'high',
  });

  // ── 3. Aadhaar Validity (15 pts) ────────────────────────────────────────
  const aadhaarResult = validateAadhaar(signals.aadhaarNumber);
  const aadhaarScore = aadhaarResult.valid ? 15 : 0;

  signalScores.push({
    label: 'Aadhaar Check',
    description: 'Verhoeff check-digit algorithm (UIDAI standard)',
    score: aadhaarScore,
    maxScore: 15,
    icon: '🪪',
    detail: aadhaarResult.reason,
    risk: aadhaarResult.valid ? 'low' : 'high',
  });

  // ── 4. OTP Completion Timing (20 pts) ───────────────────────────────────
  let otpScore = 0;
  let otpDetail = 'OTP not completed';
  let otpSeconds: number | null = null;

  if (signals.otpStartedAt > 0 && signals.otpCompletedAt > signals.otpStartedAt) {
    otpSeconds = (signals.otpCompletedAt - signals.otpStartedAt) / 1000;

    if (otpSeconds < 2) {
      // Bot-speed: sub-2 second = pre-filled / automated
      otpScore = 1;
      otpDetail = `${otpSeconds.toFixed(1)}s — suspiciously fast (possible automation)`;
    } else if (otpSeconds < 5) {
      otpScore = 5;
      otpDetail = `${otpSeconds.toFixed(1)}s — very quick, borderline automated`;
    } else if (otpSeconds < 8) {
      // Ramp up toward the human sweet spot — no longer peaks at the fastest edge
      const scaled = (otpSeconds - 5) / 3;
      otpScore = Math.round(5 + scaled * 15);
      otpDetail = `${otpSeconds.toFixed(1)}s — quick, approaching typical range`;
    } else if (otpSeconds <= 35) {
      // Genuine human sweet spot: read SMS, switch apps, type 6 digits
      otpScore = 20;
      otpDetail = `${otpSeconds.toFixed(1)}s — typical human response window`;
    } else if (otpSeconds <= 60) {
      const scaled = 1 - (otpSeconds - 35) / 25;
      otpScore = Math.round(10 + scaled * 10);
      otpDetail = `${otpSeconds.toFixed(1)}s — slightly slow but acceptable`;
    } else {
      otpScore = 6;
      otpDetail = `${otpSeconds.toFixed(1)}s — very slow (distracted or assisted?)`;
    }
  }

  otpScore = clamp(otpScore, 0, 20);

  signalScores.push({
    label: 'OTP Response Time',
    description: 'Time from OTP display to correct entry',
    score: otpScore,
    maxScore: 20,
    icon: '⏱',
    detail: otpDetail,
    risk: otpScore >= 14 ? 'low' : otpScore >= 7 ? 'medium' : 'high',
  });

  // ── 5. SIM Age Heuristic (5 pts) ────────────────────────────────────────
  const simResult = estimateSimAge(signals.phoneNumber);
  const simScore = Math.round(simResult.score / 2); // rescale 0–10 → 0–5

  const simDetail =
    simResult.operatorHint !== 'Unknown'
      ? `Prefix maps to ${simResult.operatorHint} (${simResult.circleHint}) — ${simResult.category} number block`
      : `Prefix heuristic: ${simResult.category} block`;

  signalScores.push({
    label: 'SIM Age Signal',
    description: 'MSISDN prefix age heuristic (India)',
    score: simScore,
    maxScore: 5,
    icon: '📡',
    detail: simDetail,
    risk: simScore >= 4 ? 'low' : simScore >= 2 ? 'medium' : 'high',
  });

  // ── 6. Phone Format (5 pts) ─────────────────────────────────────────────
  const localPhone = signals.phoneNumber.replace(/\D/g, '');
  const localStripped = localPhone.startsWith('91')
    ? localPhone.slice(2)
    : localPhone;

  const phoneValid = INDIA_PHONE_RE.test(localStripped);
  const phoneScore = phoneValid ? 5 : 0;

  signalScores.push({
    label: 'Phone Format',
    description: 'India MSISDN format validation',
    score: phoneScore,
    maxScore: 5,
    icon: '📞',
    detail: phoneValid
      ? `+91 ${localStripped} — valid India mobile number`
      : `"${signals.phoneNumber}" — does not match India MSISDN pattern`,
    risk: phoneValid ? 'low' : 'high',
  });

  // ── 7. Behavioral Consistency (15 pts) ──────────────────────────────────
  const consistency = behavioralConsistency(wpm, keystrokeCount, otpSeconds);

  signalScores.push({
    label: 'Behavioral Consistency',
    description: 'Cross-checks typing pace against OTP response speed',
    score: consistency.score,
    maxScore: 15,
    icon: '🔗',
    detail: consistency.detail,
    risk: consistency.score >= 11 ? 'low' : consistency.score >= 6 ? 'medium' : 'high',
  });

  // ── Total ────────────────────────────────────────────────────────────────
  const totalScore = clamp(
    signalScores.reduce((sum, s) => sum + s.score, 0),
    0,
    100,
  );

  const verdict: TrustScoreResult['verdict'] =
    totalScore >= 70 ? 'VERIFIED' : totalScore >= 45 ? 'REVIEW' : 'SUSPICIOUS';

  return {
    totalScore,
    verdict,
    signals: signalScores,
    computedAt: Date.now(),
    processingMs: Math.round(performance.now() - start),
  };
}
