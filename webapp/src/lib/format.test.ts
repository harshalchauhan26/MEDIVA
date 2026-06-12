import { describe, it, expect } from "vitest";
import { formatINR } from "@/lib/format";

describe("formatINR", () => {
  it("uses an INR marker", () => {
    expect(formatINR(30)).toMatch(/₹|INR/);
  });

  it("groups thousands the Indian way", () => {
    expect(formatINR(1234.5)).toContain("1,234");
  });

  it("renders two decimals", () => {
    expect(formatINR(30)).toContain("30.00");
  });
});
