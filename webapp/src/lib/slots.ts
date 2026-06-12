type Block = {
  dayOfWeek: number;
  startMinutes: number;
  endMinutes: number;
  slotMinutes: number;
};

type BookedAppointment = {
  startsAt: Date;
  status: string;
};

/**
 * Compute open appointment slots for one doctor on one calendar day,
 * from their recurring weekly availability minus booked appointments.
 * Past slots (relative to now) are excluded.
 */
export function computeOpenSlots(
  blocks: Block[],
  appointments: BookedAppointment[],
  date: Date
): Date[] {
  const dayOfWeek = date.getDay();
  const taken = new Set(
    appointments
      .filter((a) => a.status !== "CANCELLED")
      .map((a) => a.startsAt.getTime())
  );

  const slots: Date[] = [];
  for (const block of blocks.filter((b) => b.dayOfWeek === dayOfWeek)) {
    for (
      let minutes = block.startMinutes;
      minutes + block.slotMinutes <= block.endMinutes;
      minutes += block.slotMinutes
    ) {
      const slot = new Date(date);
      slot.setHours(0, 0, 0, 0);
      slot.setMinutes(minutes);
      if (slot.getTime() > Date.now() && !taken.has(slot.getTime())) {
        slots.push(slot);
      }
    }
  }
  return slots.sort((a, b) => a.getTime() - b.getTime());
}

/** Parse a YYYY-MM-DD string into a local-midnight Date. */
export function parseDateParam(value: string | null): Date | null {
  if (!value || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [y, m, d] = value.split("-").map(Number);
  const date = new Date(y, m - 1, d);
  return Number.isNaN(date.getTime()) ? null : date;
}
