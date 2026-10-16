/**
 * Synthetic evaluation harness for the habitID engine.
 *
 * Generates labelled synthetic KYC sessions (genuine humans vs. several fraud
 * archetypes), feeds each one through the REAL `computeHabitID()` from
 * src/services/scoringEngine.ts, and writes:
 *   report/sim_results.csv    — one row per session (label, archetype, scores)
 *   report/sim_summary.json   — aggregate metrics
 *
 * Run:  npx tsx scripts/simulate.ts
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { computeHabitID } from '../src/services/scoringEngine';
import type {
  BehavioralSignals,
  KeystrokeEvent,
  MotionSample,
} from '../src/models/BehavioralSignal';

// ── Deterministic PRNG (mulberry32) ────────────────────────────────────────
let seed = 20260919;
function rand(): number {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const uniform = (a: number, b: number) => a + (b - a) * rand();
const gauss = (mu: number, sigma: number) => {
  const u = 1 - rand();
  const v = rand();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const pick = <T>(arr: T[]): T => arr[Math.floor(rand() * arr.length)];
const randDigits = (n: number) =>
  Array.from({ length: n }, () => Math.floor(rand() * 10)).join('');

// ── Verhoeff generator (to mint VALID synthetic Aadhaar numbers) ───────────
const D = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
  [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
  [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
  [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
  [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
  [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
  [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
  [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];
const P = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
  [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
  [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];
const INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];
function verhoeffDigit(num: string): number {
  let c = 0;
  const digits = num.split('').reverse().map(Number);
  for (let i = 0; i < digits.length; i++) c = D[c][P[(i + 1) % 8][digits[i]]];
  return INV[c];
}
function validAadhaar(): string {
  const body = String(2 + Math.floor(rand() * 8)) + randDigits(10);
  return body + verhoeffDigit(body);
}
function randomAadhaar(): string {
  return String(2 + Math.floor(rand() * 8)) + randDigits(11); // ~10% pass by chance
}

// ── Phone prefixes by SIM-age category (mirrors simAgeEstimator table) ─────
const VINTAGE = ['9810', '9820', '9830', '9840', '9845', '9890', '9900', '9415', '9431'];
const MATURE = ['9711', '9718', '7011', '7022', '8050', '8105'];
const MODERATE = ['6000', '7000', '7050', '8955', '8588', '8291'];
const RECENT = ['6200', '6201', '6202', '6203', '6204', '6205'];
const phone = (prefix: string) => '+91' + prefix + randDigits(6);

// ── Signal builders ────────────────────────────────────────────────────────
function keystrokes(count: number, meanFlight: number, cv: number, meanHold = 90): KeystrokeEvent[] {
  const evts: KeystrokeEvent[] = [];
  let t = 1000;
  for (let i = 0; i < count; i++) {
    const flight = i === 0 ? 0 : Math.max(1, gauss(meanFlight, meanFlight * cv));
    const hold = Math.max(5, gauss(meanHold, meanHold * 0.3));
    const downAt = t + flight;
    const upAt = downAt + hold;
    evts.push({ key: 'x', downAt, upAt, holdDuration: hold, flightTime: flight });
    t = upAt;
  }
  return evts;
}

function motion(count: number, meanV: number, cv: number): MotionSample[] {
  return Array.from({ length: count }, (_, i) => {
    const v = Math.max(0, gauss(meanV, meanV * cv));
    return { timestamp: i * 16, x: v, y: 0, velocity: v };
  });
}

interface Session {
  label: 'genuine' | 'fraud';
  archetype: string;
  signals: BehavioralSignals;
}

function base(): BehavioralSignals {
  return {
    keystrokeEvents: [],
    typingStartedAt: 0,
    typingEndedAt: 0,
    motionSamples: [],
    deviceIsMotionCapable: rand() < 0.5,
    otpStartedAt: 100_000,
    otpCompletedAt: 0,
    phoneNumber: '',
    aadhaarNumber: '',
    aadhaarValid: false,
  };
}

// Archetype 1 — genuine human on a legitimate device
function genuineHuman(): Session {
  const s = base();
  // 20–70 WPM  →  ~170–600 ms per keystroke incl. hold
  const meanFlight = uniform(120, 420);
  s.keystrokeEvents = keystrokes(Math.round(uniform(14, 30)), meanFlight, uniform(0.3, 0.8));
  s.motionSamples = motion(Math.round(uniform(40, 200)), uniform(50, 400), uniform(0.4, 1.4));
  s.otpCompletedAt = s.otpStartedAt + uniform(5, 45) * 1000;
  const r = rand();
  s.phoneNumber = phone(r < 0.5 ? pick(VINTAGE) : r < 0.85 ? pick(MATURE) : pick(MODERATE));
  s.aadhaarNumber = validAadhaar();
  s.aadhaarValid = true;
  return { label: 'genuine', archetype: 'Genuine user', signals: s };
}

// Archetype 2 — scripted bot / headless automation
function scriptedBot(): Session {
  const s = base();
  s.keystrokeEvents = keystrokes(Math.round(uniform(14, 30)), uniform(8, 40), uniform(0.0, 0.04), 12);
  s.motionSamples = rand() < 0.6 ? [] : motion(20, 0.5, 0.0); // none or perfectly constant
  s.deviceIsMotionCapable = false;
  s.otpCompletedAt = s.otpStartedAt + uniform(0.2, 1.8) * 1000;
  s.phoneNumber = phone(pick([...RECENT, ...MODERATE]));
  s.aadhaarNumber = randomAadhaar();
  return { label: 'fraud', archetype: 'Scripted bot', signals: s };
}

// Archetype 3 — SIM-swap / mule: real human hands, freshly issued SIM,
// OTP relayed from victim (slow), Aadhaar often guessed
function simSwapMule(): Session {
  const s = base();
  s.keystrokeEvents = keystrokes(Math.round(uniform(14, 30)), uniform(150, 400), uniform(0.3, 0.9));
  s.motionSamples = motion(Math.round(uniform(40, 200)), uniform(80, 500), uniform(1.2, 2.6)); // jittery / rushed
  s.otpCompletedAt = s.otpStartedAt + uniform(50, 180) * 1000;
  s.phoneNumber = phone(pick(RECENT));
  s.aadhaarNumber = rand() < 0.4 ? validAadhaar() : randomAadhaar();
  return { label: 'fraud', archetype: 'SIM-swap mule', signals: s };
}

// Archetype 4 — OTP farm / paste-in: human typing name, but OTP pre-filled
// via clipboard/automation, and burst-paste phone number
function otpFarm(): Session {
  const s = base();
  const human = keystrokes(Math.round(uniform(8, 14)), uniform(150, 350), uniform(0.3, 0.7));
  const paste = keystrokes(10, 3, 0.02, 8); // phone pasted as a burst
  s.keystrokeEvents = [...human, ...paste];
  s.motionSamples = motion(Math.round(uniform(10, 60)), uniform(200, 900), uniform(1.5, 3.0));
  s.otpCompletedAt = s.otpStartedAt + uniform(0.3, 3) * 1000;
  s.phoneNumber = phone(pick([...RECENT, ...MODERATE]));
  s.aadhaarNumber = rand() < 0.25 ? validAadhaar() : randomAadhaar();
  return { label: 'fraud', archetype: 'OTP farm', signals: s };
}

// Archetype 5 — invalid identity: random digits typed by a human
function fakeIdentity(): Session {
  const s = base();
  s.keystrokeEvents = keystrokes(Math.round(uniform(14, 30)), uniform(120, 400), uniform(0.3, 0.8));
  s.motionSamples = motion(Math.round(uniform(40, 200)), uniform(50, 400), uniform(0.4, 1.4));
  s.otpCompletedAt = s.otpStartedAt + uniform(5, 45) * 1000;
  s.phoneNumber = rand() < 0.5 ? '+91' + String(1 + Math.floor(rand() * 5)) + randDigits(9) : phone(pick(RECENT));
  s.aadhaarNumber = randomAadhaar();
  return { label: 'fraud', archetype: 'Fake identity', signals: s };
}

// ── Build dataset ──────────────────────────────────────────────────────────
const N_GENUINE = 1000;
const N_FRAUD_EACH = 250;

const sessions: Session[] = [];
for (let i = 0; i < N_GENUINE; i++) sessions.push(genuineHuman());
for (let i = 0; i < N_FRAUD_EACH; i++) {
  sessions.push(scriptedBot());
  sessions.push(simSwapMule());
  sessions.push(otpFarm());
  sessions.push(fakeIdentity());
}

// ── Score with the real engine ─────────────────────────────────────────────
interface Row {
  label: string;
  archetype: string;
  total: number;
  verdict: string;
  typing: number;
  motion: number;
  aadhaar: number;
  otp: number;
  sim: number;
  phone: number;
  ms: number;
}

const rows: Row[] = sessions.map((sess) => {
  const r = computeHabitID(sess.signals);
  const g = (label: string) => r.signals.find((x) => x.label === label)!.score;
  return {
    label: sess.label,
    archetype: sess.archetype,
    total: r.totalScore,
    verdict: r.verdict,
    typing: g('Typing Rhythm'),
    motion: g('Tilt / Motion'),
    aadhaar: g('Aadhaar Check'),
    otp: g('OTP Response Time'),
    sim: g('SIM Age Signal'),
    phone: g('Phone Format'),
    ms: r.processingMs,
  };
});

// ── Metrics ────────────────────────────────────────────────────────────────
// Binary decision used for metrics: VERIFIED → genuine, else → fraud
const tp = rows.filter((r) => r.label === 'fraud' && r.verdict !== 'VERIFIED').length;
const fn = rows.filter((r) => r.label === 'fraud' && r.verdict === 'VERIFIED').length;
const tn = rows.filter((r) => r.label === 'genuine' && r.verdict === 'VERIFIED').length;
const fp = rows.filter((r) => r.label === 'genuine' && r.verdict !== 'VERIFIED').length;

const accuracy = (tp + tn) / rows.length;
const precision = tp / (tp + fp);
const recall = tp / (tp + fn);
const f1 = (2 * precision * recall) / (precision + recall);

// ROC-AUC via rank statistic (fraud = positive, lower score = more positive)
const fraudScores = rows.filter((r) => r.label === 'fraud').map((r) => r.total);
const genScores = rows.filter((r) => r.label === 'genuine').map((r) => r.total);
let wins = 0;
for (const f of fraudScores) for (const g of genScores) wins += f < g ? 1 : f === g ? 0.5 : 0;
const auc = wins / (fraudScores.length * genScores.length);

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const byArchetype: Record<string, { n: number; meanScore: number; verified: number; review: number; suspicious: number }> = {};
for (const r of rows) {
  const a = (byArchetype[r.archetype] ??= { n: 0, meanScore: 0, verified: 0, review: 0, suspicious: 0 });
  a.n++;
  a.meanScore += r.total;
  if (r.verdict === 'VERIFIED') a.verified++;
  else if (r.verdict === 'REVIEW') a.review++;
  else a.suspicious++;
}
for (const a of Object.values(byArchetype)) a.meanScore = +(a.meanScore / a.n).toFixed(1);

const signalMeans = (label: string) => ({
  typing: +mean(rows.filter((r) => r.label === label).map((r) => r.typing)).toFixed(1),
  motion: +mean(rows.filter((r) => r.label === label).map((r) => r.motion)).toFixed(1),
  aadhaar: +mean(rows.filter((r) => r.label === label).map((r) => r.aadhaar)).toFixed(1),
  otp: +mean(rows.filter((r) => r.label === label).map((r) => r.otp)).toFixed(1),
  sim: +mean(rows.filter((r) => r.label === label).map((r) => r.sim)).toFixed(1),
  phone: +mean(rows.filter((r) => r.label === label).map((r) => r.phone)).toFixed(1),
});

const summary = {
  n_sessions: rows.length,
  n_genuine: genScores.length,
  n_fraud: fraudScores.length,
  confusion: { tp, fp, tn, fn },
  accuracy: +accuracy.toFixed(4),
  precision: +precision.toFixed(4),
  recall: +recall.toFixed(4),
  f1: +f1.toFixed(4),
  roc_auc: +auc.toFixed(4),
  mean_score_genuine: +mean(genScores).toFixed(1),
  mean_score_fraud: +mean(fraudScores).toFixed(1),
  mean_processing_ms: +mean(rows.map((r) => r.ms)).toFixed(3),
  by_archetype: byArchetype,
  signal_means: { genuine: signalMeans('genuine'), fraud: signalMeans('fraud') },
};

mkdirSync('report', { recursive: true });
writeFileSync(
  'report/sim_results.csv',
  'label,archetype,total,verdict,typing,motion,aadhaar,otp,sim,phone,ms\n' +
    rows.map((r) => Object.values(r).join(',')).join('\n'),
);
writeFileSync('report/sim_summary.json', JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
