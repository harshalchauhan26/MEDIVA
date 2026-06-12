import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { getSession, AuthError } from "@/lib/auth";
import { errorResponse } from "@/lib/api";
import { sendMockNotification } from "@/lib/notifications";
import { sendWhatsApp } from "@/lib/whatsapp";

const UpdateSchema = z.object({
  status: z.enum(["CONFIRMED", "CANCELLED", "COMPLETED"]),
});

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  try {
    const session = getSession();
    if (!session) throw new AuthError(401, "Login required.");

    const parsed = UpdateSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid status." }, { status: 400 });
    }

    const appointment = await prisma.appointment.findUnique({
      where: { id: params.id },
      include: {
        patient: { select: { id: true, name: true, email: true, phone: true } },
        doctor: { include: { user: { select: { id: true, name: true } } } },
      },
    });
    if (!appointment) {
      return NextResponse.json({ error: "Appointment not found." }, { status: 404 });
    }

    const isOwningPatient =
      session.role === "PATIENT" && appointment.patient.id === session.userId;
    const isOwningDoctor =
      session.role === "DOCTOR" && appointment.doctor.user.id === session.userId;
    // Patients may only cancel; doctors may confirm/cancel/complete their own.
    const allowed =
      (isOwningPatient && parsed.data.status === "CANCELLED") || isOwningDoctor;
    if (!allowed) throw new AuthError(403, "You cannot update this appointment.");

    const updated = await prisma.appointment.update({
      where: { id: params.id },
      data: { status: parsed.data.status },
    });

    const when = appointment.startsAt.toLocaleString("en-IN", {
      dateStyle: "medium",
      timeStyle: "short",
    });
    sendMockNotification({
      channel: "email",
      to: appointment.patient.email,
      subject: `Appointment ${parsed.data.status.toLowerCase()}`,
      body: `Your appointment with ${appointment.doctor.user.name} on ${when} is now ${parsed.data.status}.`,
    });
    await sendWhatsApp(
      appointment.patient.phone,
      `MEDIVA 🏥\nYour appointment with ${appointment.doctor.user.name} on ${when} is now *${parsed.data.status}*.`
    );

    return NextResponse.json({ id: updated.id, status: updated.status });
  } catch (error) {
    return errorResponse(error);
  }
}
