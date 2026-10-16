/**
 * SimAgeEstimator
 *
 * Estimates the approximate SIM/number age category from India's
 * MSISDN (mobile number) prefix ranges.
 *
 * India's numbering plan assigns blocks to operators and circles.
 * Older blocks (allocated ~2005–2012) tend to correlate with
 * long-standing subscribers. This is a heuristic signal only —
 * used as one factor in the trust score.
 *
 * Scoring:
 *   'vintage'  (pre-2010 blocks) → age score 10/10
 *   'mature'   (2010–2015 blocks)→ age score 7/10
 *   'moderate' (2015–2020 blocks)→ age score 5/10
 *   'recent'   (post-2020 blocks)→ age score 2/10
 *   'unknown'                    → age score 4/10 (neutral)
 */

type AgeCategory = 'vintage' | 'mature' | 'moderate' | 'recent' | 'unknown';

interface SimAgeResult {
  category: AgeCategory;
  score: number;       // 0–10
  operatorHint: string;
  circleHint: string;
}

// Prefix → [operator, circle, ageCategory]
// This is a representative (not exhaustive) table of India's MSISDN allocations.
const PREFIX_TABLE: Record<string, [string, string, AgeCategory]> = {
  // Airtel - vintage blocks (allocated ~2004-2008)
  '9810': ['Airtel', 'Delhi', 'vintage'],
  '9811': ['Airtel', 'Delhi', 'vintage'],
  '9820': ['Airtel', 'Mumbai', 'vintage'],
  '9821': ['Airtel', 'Mumbai', 'vintage'],
  '9830': ['Airtel', 'Kolkata', 'vintage'],
  '9831': ['Airtel', 'Kolkata', 'vintage'],
  '9840': ['Airtel', 'Tamil Nadu', 'vintage'],
  '9841': ['Airtel', 'Tamil Nadu', 'vintage'],
  '9850': ['Airtel', 'Maharashtra', 'vintage'],
  '9845': ['Airtel', 'Karnataka', 'vintage'],
  '9891': ['Airtel', 'Delhi', 'vintage'],

  // Vodafone/Vi - vintage
  '9890': ['Vi', 'Maharashtra', 'vintage'],
  '9892': ['Vi', 'Mumbai', 'vintage'],
  '9900': ['Vi', 'Karnataka', 'vintage'],
  '9901': ['Vi', 'Karnataka', 'vintage'],
  '9823': ['Vi', 'Mumbai', 'vintage'],
  '9867': ['Vi', 'Mumbai', 'vintage'],

  // BSNL - vintage
  '9415': ['BSNL', 'UP East', 'vintage'],
  '9416': ['BSNL', 'Haryana', 'vintage'],
  '9431': ['BSNL', 'Bihar', 'vintage'],
  '9433': ['BSNL', 'West Bengal', 'vintage'],
  '9437': ['BSNL', 'Orissa', 'vintage'],

  // Airtel - mature (2010-2015)
  '9711': ['Airtel', 'Delhi', 'mature'],
  '9718': ['Airtel', 'Delhi', 'mature'],
  '7011': ['Airtel', 'Delhi', 'mature'],
  '7012': ['Airtel', 'Kerala', 'mature'],
  '7022': ['Airtel', 'Karnataka', 'mature'],
  '8050': ['Airtel', 'Karnataka', 'mature'],
  '8105': ['Airtel', 'Karnataka', 'mature'],

  // Jio - moderate/recent (launched 2016)
  '6000': ['Jio', 'Various', 'moderate'],
  '6001': ['Jio', 'Various', 'moderate'],
  '6002': ['Jio', 'Various', 'moderate'],
  '6003': ['Jio', 'Various', 'moderate'],
  '7000': ['Jio', 'Various', 'moderate'],
  '7001': ['Jio', 'Various', 'moderate'],
  '7050': ['Jio', 'Various', 'moderate'],
  '8955': ['Jio', 'Various', 'moderate'],
  '9773': ['Jio', 'Various', 'moderate'],

  // Jio newer blocks (post-2020)
  '6200': ['Jio', 'Various', 'recent'],
  '6201': ['Jio', 'Various', 'recent'],
  '6202': ['Jio', 'Various', 'recent'],
  '6203': ['Jio', 'Various', 'recent'],
  '6204': ['Jio', 'Various', 'recent'],
  '6205': ['Jio', 'Various', 'recent'],

  // Airtel newer blocks
  '8588': ['Airtel', 'Delhi', 'moderate'],
  '8826': ['Airtel', 'Delhi', 'moderate'],

  // Vi newer
  '8291': ['Vi', 'Mumbai', 'moderate'],
  '8655': ['Vi', 'Mumbai', 'moderate'],
};

const AGE_SCORES: Record<AgeCategory, number> = {
  vintage: 10,
  mature: 7,
  moderate: 5,
  recent: 2,
  unknown: 4,
};

export function estimateSimAge(phone: string): SimAgeResult {
  // Strip country code and spaces
  const digits = phone.replace(/\D/g, '');
  const local = digits.startsWith('91') ? digits.slice(2) : digits;

  if (local.length < 10) {
    return { category: 'unknown', score: 4, operatorHint: 'Unknown', circleHint: 'Unknown' };
  }

  const prefix4 = local.slice(0, 4);
  const match = PREFIX_TABLE[prefix4];

  if (match) {
    const [operator, circle, category] = match;
    return {
      category,
      score: AGE_SCORES[category],
      operatorHint: operator,
      circleHint: circle,
    };
  }

  // Fallback heuristic by first digit of local number
  const firstDigit = local[0];
  if (['9', '8'].includes(firstDigit)) {
    return { category: 'mature', score: 6, operatorHint: 'Unknown', circleHint: 'Unknown' };
  }
  if (firstDigit === '7') {
    return { category: 'moderate', score: 5, operatorHint: 'Unknown', circleHint: 'Unknown' };
  }
  if (firstDigit === '6') {
    return { category: 'recent', score: 3, operatorHint: 'Jio/Unknown', circleHint: 'Unknown' };
  }

  return { category: 'unknown', score: 4, operatorHint: 'Unknown', circleHint: 'Unknown' };
}
