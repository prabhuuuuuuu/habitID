// Captured during the name/phone typing phase
export interface KeystrokeEvent {
  key: string;
  downAt: number;    // performance.now() on keydown
  upAt: number;      // performance.now() on keyup
  holdDuration: number; // upAt - downAt (ms)
  flightTime: number;   // time since previous keyup (ms) — inter-key gap
}

// Captured from mouse/pointer movements (desktop tilt proxy)
// or DeviceMotionEvent (mobile)
export interface MotionSample {
  timestamp: number;
  x: number;  // accelerationX or mouse deltaX
  y: number;  // accelerationY or mouse deltaY
  velocity: number; // magnitude
}

export interface BehavioralSignals {
  // Typing
  keystrokeEvents: KeystrokeEvent[];
  typingStartedAt: number;
  typingEndedAt: number;

  // Motion / tilt
  motionSamples: MotionSample[];
  deviceIsMotionCapable: boolean;  // true = real accelerometer, false = mouse proxy

  // OTP
  otpStartedAt: number;    // when OTP screen was shown
  otpCompletedAt: number;  // when user submitted correct OTP

  // Phone
  phoneNumber: string;

  // Aadhaar
  aadhaarNumber: string;
  aadhaarValid: boolean;
}

export function emptySignals(): BehavioralSignals {
  return {
    keystrokeEvents: [],
    typingStartedAt: 0,
    typingEndedAt: 0,
    motionSamples: [],
    deviceIsMotionCapable: false,
    otpStartedAt: 0,
    otpCompletedAt: 0,
    phoneNumber: '',
    aadhaarNumber: '',
    aadhaarValid: false,
  };
}
