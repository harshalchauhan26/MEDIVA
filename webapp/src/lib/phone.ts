/**
 * Normalize an Indian mobile number to E.164 (+91XXXXXXXXXX).
 * Accepts "9876543210", "+91 98765 43210", "098765-43210", etc.
 * Returns null if it is not a valid 10-digit Indian mobile (starts 6-9).
 */
export function normalizeIndianPhone(raw: string): string | null {
  const digits = raw.replace(/\D/g, "");
  let local = digits;
  if (local.length === 12 && local.startsWith("91")) local = local.slice(2);
  if (local.length === 11 && local.startsWith("0")) local = local.slice(1);
  if (!/^[6-9]\d{9}$/.test(local)) return null;
  return `+91${local}`;
}
