import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { getSession, clearSessionCookie, AuthError } from "@/lib/auth";
import { errorResponse } from "@/lib/api";

// Self-service account deletion. Removes the user and all data that would
// otherwise block the delete (reservations, appointments). Doctor profile and
// availability cascade via the schema's onDelete rules.
export async function DELETE() {
  try {
    const session = getSession();
    if (!session) throw new AuthError(401, "Login required.");

    const user = await prisma.user.findUnique({
      where: { id: session.userId },
      include: { doctor: { select: { id: true } } },
    });
    if (!user) {
      clearSessionCookie();
      return NextResponse.json({ ok: true });
    }

    await prisma.$transaction(async (tx) => {
      await tx.reservation.deleteMany({ where: { userId: user.id } });
      await tx.appointment.deleteMany({ where: { patientId: user.id } });
      if (user.doctor) {
        // Appointments reference Doctor with RESTRICT; clear them first.
        await tx.appointment.deleteMany({ where: { doctorId: user.doctor.id } });
      }
      await tx.user.delete({ where: { id: user.id } });
    });

    clearSessionCookie();
    return NextResponse.json({ ok: true });
  } catch (error) {
    return errorResponse(error);
  }
}
