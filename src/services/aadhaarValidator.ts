/**
 * AadhaarValidator
 *
 * Implements the Verhoeff algorithm — the actual check-digit scheme
 * used by UIDAI for Aadhaar numbers.
 *
 * A valid Aadhaar number is:
 *  - Exactly 12 digits
 *  - Does not start with 0 or 1
 *  - Passes Verhoeff check-digit validation
 */

// Verhoeff multiplication table
const D: number[][] = [
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

// Verhoeff permutation table
const P: number[][] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
  [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
  [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
  [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
  [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
  [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
  [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

// Verhoeff inverse table (reserved for future check-digit generation)
// const INV: number[] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

function verhoeffCheck(num: string): boolean {
  let c = 0;
  const digits = num
    .split('')
    .reverse()
    .map(Number);

  for (let i = 0; i < digits.length; i++) {
    c = D[c][P[i % 8][digits[i]]];
  }

  return c === 0;
}

export interface AadhaarValidationResult {
  valid: boolean;
  reason: string;
}

export function validateAadhaar(raw: string): AadhaarValidationResult {
  const digits = raw.replace(/\s/g, '').replace(/-/g, '');

  if (!/^\d+$/.test(digits)) {
    return { valid: false, reason: 'Must contain digits only' };
  }

  if (digits.length !== 12) {
    return { valid: false, reason: `Must be 12 digits (got ${digits.length})` };
  }

  if (digits[0] === '0' || digits[0] === '1') {
    return { valid: false, reason: 'Cannot start with 0 or 1' };
  }

  if (!verhoeffCheck(digits)) {
    return { valid: false, reason: 'Check digit mismatch (invalid number)' };
  }

  return { valid: true, reason: 'Valid Aadhaar number' };
}

/**
 * Format an Aadhaar number for display: XXXX XXXX XXXX
 */
export function formatAadhaar(raw: string): string {
  const digits = raw.replace(/\D/g, '').slice(0, 12);
  return digits.replace(/(\d{4})(?=\d)/g, '$1 ').trim();
}

/**
 * Mask Aadhaar for display: XXXX XXXX 3456
 */
export function maskAadhaar(raw: string): string {
  const digits = raw.replace(/\D/g, '');
  if (digits.length !== 12) return formatAadhaar(raw);
  return `XXXX XXXX ${digits.slice(8)}`;
}
