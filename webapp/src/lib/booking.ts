import { Prisma } from "@prisma/client";
import { prisma } from "@/lib/db";
import { computeOpenSlots } from "@/lib/slots";
import { sendMockNotification } from "@/lib/notifications";
import { sendWhatsApp } from "@/lib/whatsapp";

export type BookingInput = {
  patientId: string;
  patientName: string;
  patientEmail: string;
  patientPhone?: string | null;
  doctorId: string;
  startsAt: Date;
  symptoms: string;
};

export type BookingResult =
  | { ok: true; appointment: { id: string; startsAt: string; doctorName: string; status: string } }
  | { ok: false; status: number; error: string };

/**
 * Shared booking logic used by both the REST endpoint and the MediVa
 * agent's book_appointment tool, so validation never diverges.
 */
export async function bookAppointment(input: BookingInput): Promise<BookingResult> {
  if (Number.isNaN(input.startsAt.getTime())) {
    return { ok: false, status: 400, error: "Invalid appointment time." };
  }

  const dayStart = new Date(input.startsAt);
  dayStart.setHours(0, 0, 0, 0);
  const dayEnd = new Date(dayStart);
  dayEnd.setDate(dayEnd.getDate() + 1);

  const doctor = await prisma.doctor.findUnique({
    where: { id: input.doctorId },
    include: {
      user: { select: { name: true } },
      availability: true,
      appointments: { where: { startsAt: { gte: dayStart, lt: dayEnd } } },
    },
  });
  if (!doctor) return { ok: false, status: 404, error: "Doctor not found." };
  if (doctor.status === "ON_LEAVE") {
    return { ok: false, status: 409, error: `${doctor.user.name} is currently on leave.` };
  }

  const openSlots = computeOpenSlots(doctor.availability, doctor.appointments, dayStart);
  const slot = openSlots.find((s) => s.getTime() === input.startsAt.getTime());
  if (!slot) {
    return { ok: false, status: 409, error: "That time slot is not available. Please pick another open slot." };
  }

  const block = doctor.availability.find((b) => b.dayOfWeek === input.startsAt.getDay());
  const slotMinutes = block?.slotMinutes ?? 30;

  try {
    const appointment = await prisma.appointment.create({
      data: {
        patientId: input.patientId,
        doctorId: input.doctorId,
        startsAt: input.startsAt,
        endsAt: new Date(input.startsAt.getTime() + slotMinutes * 60 * 1000),
        symptoms: input.symptoms,
        status: "PENDING",
      },
    });

    const when = input.startsAt.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
    sendMockNotification({
      channel: "email",
      to: input.patientEmail,
      subject: "Appointment request received",
      body: `Hi ${input.patientName}, your appointment with ${doctor.user.name} (${doctor.specialty}) on ${when} is pending confirmation. Reason: ${input.symptoms}`,
    });
    await sendWhatsApp(
      input.patientPhone ?? null,
      `MEDIVA 🏥\nHi ${input.patientName}, your appointment with ${doctor.user.name} (${doctor.specialty}) on ${when} is *PENDING confirmation*.\nReason: ${input.symptoms}\n\nReply CANCEL to cancel. This is an automated message — not medical advice.`
    );

    return {
      ok: true,
      appointment: {
        id: appointment.id,
        startsAt: appointment.startsAt.toISOString(),
        doctorName: doctor.user.name,
        status: appointment.status,
      },
    };
  } catch (error) {
    if (error instanceof Prisma.PrismaClientKnownRequestError && error.code === "P2002") {
      return { ok: false, status: 409, error: "That slot was just taken. Please pick another." };
    }
    throw error;
  }
}
