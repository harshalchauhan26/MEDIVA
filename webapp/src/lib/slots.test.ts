import { describe, it, expect } from "vitest";
import { computeOpenSlots, parseDateParam } from "@/lib/slots";

describe("computeOpenSlots", () => {
  // A far-future date so no slot is filtered out for being in the past.
  const date = new Date(2099, 0, 5);
  const dow = date.getDay();
  const blocks = [{ dayOfWeek: dow, startMinutes: 540, endMinutes: 600, slotMinutes: 30 }]; // 09:00-10:00

  it("returns one slot per slot-length within the block", () => {
    const slots = computeOpenSlots(blocks, [], date);
    expect(slots).toHaveLength(2); // 09:00 and 09:30
  });

  it("excludes booked slots", () => {
    const [first] = computeOpenSlots(blocks, [], date);
    const slots = computeOpenSlots(blocks, [{ startsAt: first, status: "CONFIRMED" }], date);
    expect(slots).toHaveLength(1);
  });

  it("ignores cancelled appointments", () => {
    const [first] = computeOpenSlots(blocks, [], date);
    const slots = computeOpenSlots(blocks, [{ startsAt: first, status: "CANCELLED" }], date);
    expect(slots).toHaveLength(2);
  });

  it("returns nothing when the block is on a different weekday", () => {
    const otherDay = [{ dayOfWeek: (dow + 1) % 7, startMinutes: 540, endMinutes: 600, slotMinutes: 30 }];
    expect(computeOpenSlots(otherDay, [], date)).toHaveLength(0);
  });
});

describe("parseDateParam", () => {
  it("parses a valid YYYY-MM-DD", () => {
    expect(parseDateParam("2099-01-05")).not.toBeNull();
  });

  it("rejects garbage", () => {
    expect(parseDateParam("nope")).toBeNull();
    expect(parseDateParam("")).toBeNull();
    expect(parseDateParam(null)).toBeNull();
  });
});
