export interface SignalScore {
  label: string;
  description: string;
  score: number;     // 0–maxScore
  maxScore: number;
  icon: string;      // emoji shorthand for display
  detail: string;    // human-readable explanation
  risk: 'low' | 'medium' | 'high';
}

export interface TrustScoreResult {
  totalScore: number;        // 0–100
  verdict: 'VERIFIED' | 'REVIEW' | 'SUSPICIOUS';
  signals: SignalScore[];
  computedAt: number;        // Date.now()
  processingMs: number;      // how long computation took
}
