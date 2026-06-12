import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { computeOpenSlots, parseDateParam } from "@/lib/slots";

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: { id: string } }) {
  const { searchParams } = new URL(request.url);
  const date = parseDateParam(searchParams.get("date"));
  if (!date) {
    return NextResponse.json({ error: "date must be YYYY-MM-DD." }, { status: 400 });
  }

  const dayStart = new Date(date);
  const dayEnd = new Date(date);
  dayEnd.setDate(dayEnd.getDate() + 1);

  const doctor = await prisma.doctor.findUnique({
    where: { id: params.id },
    include: {
      availability: true,
      appointments: { where: { startsAt: { gte: dayStart, lt: dayEnd } } },
    },
  });
  if (!doctor) {
    return NextResponse.json({ error: "Doctor not found." }, { status: 404 });
  }
  if (doctor.status === "ON_LEAVE") {
    return NextResponse.json({ slots: [], note: "Doctor is on leave." });
  }

  const slots = computeOpenSlots(doctor.availability, doctor.appointments, date);
  return NextResponse.json({ slots: slots.map((s) => s.toISOString()) });
}
