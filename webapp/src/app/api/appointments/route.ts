import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { getSession, AuthError } from "@/lib/auth";
import { errorResponse } from "@/lib/api";
import { bookAppointment } from "@/lib/booking";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const session = getSession();
    if (!session) throw new AuthError(401, "Login required.");

    if (session.role === "PATIENT") {
      const appointments = await prisma.appointment.findMany({
        where: { patientId: session.userId },
        include: { doctor: { include: { user: { select: { name: true } } } } },
        orderBy: { startsAt: "desc" },
        take: 50,
      });
      return NextResponse.json({
        appointments: appointments.map((a) => ({
          id: a.id,
          startsAt: a.startsAt.toISOString(),
          status: a.status,
          symptoms: a.symptoms,
          doctorName: a.doctor.user.name,
          specialty: a.doctor.specialty,
        })),
      });
    }

    throw new AuthError(403, "Use the doctor dashboard endpoint for schedules.");
  } catch (error) {
    return errorResponse(error);
  }
}

const BookSchema = z.object({
  doctorId: z.string().min(1),
  startsAt: z.string().datetime(),
  symptoms: z.string().min(3).max(1000),
});

export async function POST(request: Request) {
  try {
    const session = getSession();
    if (!session) throw new AuthError(401, "Please log in to book an appointment.");
    if (session.role !== "PATIENT") {
      throw new AuthError(403, "Only patients can book appointments.");
    }

    const parsed = BookSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.issues[0].message }, { status: 400 });
    }

    const result = await bookAppointment({
      patientId: session.userId,
      patientName: session.name,
      patientEmail: session.email,
      patientPhone: session.phone,
      doctorId: parsed.data.doctorId,
      startsAt: new Date(parsed.data.startsAt),
      symptoms: parsed.data.symptoms,
    });

    if (!result.ok) {
      return NextResponse.json({ error: result.error }, { status: result.status });
    }
    return NextResponse.json({ appointment: result.appointment }, { status: 201 });
  } catch (error) {
    return errorResponse(error);
  }
}
