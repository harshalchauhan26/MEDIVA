import { describe, it, expect } from "vitest";
import { normalizeIndianPhone } from "@/lib/phone";

describe("normalizeIndianPhone", () => {
  it("normalizes a plain 10-digit number", () => {
    expect(normalizeIndianPhone("9876543210")).toBe("+919876543210");
  });

  it("strips +91 and spaces", () => {
    expect(normalizeIndianPhone("+91 98765 43210")).toBe("+919876543210");
  });

  it("strips a leading 0 and separators", () => {
    expect(normalizeIndianPhone("098765-43210")).toBe("+919876543210");
  });

  it("rejects too-short numbers", () => {
    expect(normalizeIndianPhone("12345")).toBeNull();
  });

  it("rejects numbers not starting 6-9", () => {
    expect(normalizeIndianPhone("1234567890")).toBeNull();
  });
});
