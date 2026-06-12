const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
});

/** Format a number as Indian Rupees, e.g. 1234.5 -> "₹1,234.50". */
export function formatINR(value: number): string {
  return inr.format(value);
}
