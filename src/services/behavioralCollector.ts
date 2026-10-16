/**
 * BehavioralCollector
 *
 * Attaches to DOM events and collects:
 *  - Keystroke timing (hold duration + inter-key flight time)
 *  - Mouse velocity (desktop tilt proxy) OR DeviceMotion (mobile)
 *
 * Usage:
 *   const collector = new BehavioralCollector();
 *   collector.start();
 *   // ... user interacts ...
 *   const signals = collector.stop();
 */

import type { KeystrokeEvent, MotionSample } from '../models/BehavioralSignal';

interface KeydownRecord {
  key: string;
  downAt: number;
}

export class BehavioralCollector {
  private keystrokeEvents: KeystrokeEvent[] = [];
  private motionSamples: MotionSample[] = [];
  private deviceIsMotionCapable = false;

  private startedAt = 0;
  private lastKeyUpAt = 0;
  private lastMouseAt = 0;
  private lastMouseX = 0;
  private lastMouseY = 0;

  private pendingKeys: Map<string, KeydownRecord> = new Map();

  private _onKeyDown = (e: KeyboardEvent) => {
    const now = performance.now();
    // Only track printable characters and common keys
    if (e.key.length <= 1 || ['Backspace', 'Enter', 'Tab', 'Space'].includes(e.key)) {
      this.pendingKeys.set(e.code, { key: e.key, downAt: now });
      if (this.startedAt === 0) this.startedAt = now;
    }
  };

  private _onKeyUp = (e: KeyboardEvent) => {
    const now = performance.now();
    const record = this.pendingKeys.get(e.code);
    if (!record) return;
    this.pendingKeys.delete(e.code);

    const holdDuration = now - record.downAt;
    const flightTime = this.lastKeyUpAt > 0 ? record.downAt - this.lastKeyUpAt : 0;

    this.keystrokeEvents.push({
      key: record.key,
      downAt: record.downAt,
      upAt: now,
      holdDuration,
      flightTime,
    });

    this.lastKeyUpAt = now;
  };

  private _onMouseMove = (e: MouseEvent) => {
    const now = performance.now();
    if (this.lastMouseAt === 0) {
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      this.lastMouseAt = now;
      return;
    }
    const dt = now - this.lastMouseAt;
    if (dt < 16) return; // throttle to ~60fps

    const dx = e.clientX - this.lastMouseX;
    const dy = e.clientY - this.lastMouseY;
    const velocity = Math.sqrt(dx * dx + dy * dy) / (dt / 1000); // px/s

    this.motionSamples.push({ timestamp: now, x: dx, y: dy, velocity });

    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;
    this.lastMouseAt = now;
  };

  private _onDeviceMotion = (e: DeviceMotionEvent) => {
    const now = performance.now();
    const acc = e.acceleration;
    if (!acc) return;
    const x = acc.x ?? 0;
    const y = acc.y ?? 0;
    const velocity = Math.sqrt(x * x + y * y);
    this.motionSamples.push({ timestamp: now, x, y, velocity });
  };

  start(): void {
    this.keystrokeEvents = [];
    this.motionSamples = [];
    this.startedAt = 0;
    this.lastKeyUpAt = 0;
    this.lastMouseAt = 0;
    this.pendingKeys.clear();

    window.addEventListener('keydown', this._onKeyDown, true);
    window.addEventListener('keyup', this._onKeyUp, true);

    // Prefer real DeviceMotion (mobile)
    if (typeof DeviceMotionEvent !== 'undefined' && 'ontouchstart' in window) {
      window.addEventListener('devicemotion', this._onDeviceMotion, true);
      this.deviceIsMotionCapable = true;
    } else {
      window.addEventListener('mousemove', this._onMouseMove, true);
      this.deviceIsMotionCapable = false;
    }
  }

  stop(): {
    keystrokeEvents: KeystrokeEvent[];
    motionSamples: MotionSample[];
    deviceIsMotionCapable: boolean;
    typingStartedAt: number;
    typingEndedAt: number;
  } {
    window.removeEventListener('keydown', this._onKeyDown, true);
    window.removeEventListener('keyup', this._onKeyUp, true);
    window.removeEventListener('devicemotion', this._onDeviceMotion, true);
    window.removeEventListener('mousemove', this._onMouseMove, true);

    const typingEndedAt =
      this.keystrokeEvents.length > 0
        ? this.keystrokeEvents[this.keystrokeEvents.length - 1].upAt
        : 0;

    return {
      keystrokeEvents: [...this.keystrokeEvents],
      motionSamples: [...this.motionSamples],
      deviceIsMotionCapable: this.deviceIsMotionCapable,
      typingStartedAt: this.startedAt,
      typingEndedAt,
    };
  }

  // Partial snapshot without stopping
  snapshot(): KeystrokeEvent[] {
    return [...this.keystrokeEvents];
  }
}

// --- Analysis helpers used by ScoringEngine ---

/**
 * Shannon entropy of the inter-key flight times (normalized 0–1).
 * A human shows moderate entropy. A bot shows very low (uniform) or
 * very high (random) entropy.
 */
export interface TypingRhythmResult {
  ratio: number;      // 0–1 score multiplier
  cv: number | null;  // raw coefficient of variation, for diagnostics/display
}

export function typingRhythmEntropy(events: KeystrokeEvent[]): TypingRhythmResult {
  if (events.length < 4) return { ratio: 0.5, cv: null }; // insufficient data → neutral

  const flights = events
    .map((e) => e.flightTime)
    .filter((f) => f > 0 && f < 2000); // ignore outliers

  if (flights.length < 3) return { ratio: 0.5, cv: null };

  const mean = flights.reduce((a, b) => a + b, 0) / flights.length;
  const variance =
    flights.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / flights.length;
  const stdDev = Math.sqrt(variance);
  const cv = stdDev / mean; // coefficient of variation

  // Human typists have CV typically 0.3–0.8
  // Bots < 0.1 (too uniform) or > 1.5 (random burst)
  let ratio: number;
  if (cv < 0.05) ratio = 0.1;         // near-zero variance = bot
  else if (cv > 2.0) ratio = 0.2;     // chaotic = suspicious
  else if (cv >= 0.2 && cv <= 1.0) ratio = 1.0; // natural human range
  else if (cv < 0.2) ratio = cv / 0.2; // linear taper outside the sweet spot
  else ratio = Math.max(0, 1 - (cv - 1.0) / 1.0);

  return { ratio, cv };
}

/**
 * Words-per-minute computed from keystroke timeline.
 * A human filling a form: 20–80 WPM is plausible.
 */
export function typingWPM(events: KeystrokeEvent[]): number {
  if (events.length < 5) return 0;
  const duration =
    (events[events.length - 1].upAt - events[0].downAt) / 1000 / 60; // minutes
  if (duration <= 0) return 0;
  // Approximate words = keystrokes / 5
  return Math.round(events.length / 5 / duration);
}

/**
 * Motion stability score (0–1). For desktop mouse proxy:
 * Low velocity variance = calm hand = higher score.
 * Extreme high velocity = erratic movement.
 */
export function motionStability(samples: MotionSample[]): number {
  if (samples.length < 5) return 0.4; // not enough data → cannot confirm calm motion, mildly penalized

  const velocities = samples.map((s) => s.velocity);
  const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
  const variance =
    velocities.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / velocities.length;
  const stdDev = Math.sqrt(variance);

  // Normalize: low stdDev relative to mean = stable
  const cv = stdDev / (mean + 1); // +1 avoids divide by zero
  return Math.max(0, Math.min(1, 1 - cv / 3));
}
