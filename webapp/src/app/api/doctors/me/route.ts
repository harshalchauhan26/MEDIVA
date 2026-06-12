import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireRole } from "@/lib/auth";
import { errorResponse } from "@/lib/api";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const session = requireRole("DOCTOR");
    const doctor = await prisma.doctor.findUnique({
      where: { userId: session.userId },
      include: {
        user: { select: { name: true } },
        appointments: {
          where: { startsAt: { gte: new Date(new Date().setHours(0, 0, 0, 0)) } },
          include: { patient: { select: { name: true, email: true } } },
          orderBy: { startsAt: "asc" },
          take: 100,
        },
      },
    });
    if (!doctor) {
      return NextResponse.json({ error: "Doctor profile not found." }, { status: 404 });
    }
    return NextResponse.json({
      doctor: {
        id: doctor.id,
        name: doctor.user.name,
        specialty: doctor.specialty,
        status: doctor.status,
        rating: doctor.rating,
      },
      appointments: doctor.appointments.map((a) => ({
        id: a.id,
        startsAt: a.startsAt.toISOString(),
        endsAt: a.endsAt.toISOString(),
        status: a.status,
        symptoms: a.symptoms,
        patientName: a.patient.name,
        patientEmail: a.patient.email,
      })),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

const StatusSchema = z.object({
  status: z.enum(["AVAILABLE", "IN_SESSION", "ON_LEAVE"]),
});

export async function PATCH(request: Request) {
  try {
    const session = requireRole("DOCTOR");
    const parsed = StatusSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid status." }, { status: 400 });
    }
    const doctor = await prisma.doctor.update({
      where: { userId: session.userId },
      data: { status: parsed.data.status },
    });
    return NextResponse.json({ status: doctor.status });
  } catch (error) {
    return errorResponse(error);
  }
}
