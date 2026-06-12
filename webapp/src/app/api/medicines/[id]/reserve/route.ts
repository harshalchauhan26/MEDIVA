import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireRole } from "@/lib/auth";
import { errorResponse } from "@/lib/api";
import { sendMockNotification } from "@/lib/notifications";
import { sendWhatsApp } from "@/lib/whatsapp";

const ReserveSchema = z.object({
  quantity: z.number().int().min(1).max(10).default(1),
});

export async function POST(request: Request, { params }: { params: { id: string } }) {
  try {
    const session = requireRole("PATIENT");
    const body = await request.json().catch(() => ({}));
    const parsed = ReserveSchema.safeParse(body ?? {});
    if (!parsed.success) {
      return NextResponse.json({ error: "Invalid quantity (1-10)." }, { status: 400 });
    }
    const quantity = parsed.data.quantity;

    const reservation = await prisma.$transaction(async (tx) => {
      const updated = await tx.medicine.updateMany({
        where: { id: params.id, quantity: { gte: quantity } },
        data: { quantity: { decrement: quantity } },
      });
      if (updated.count === 0) return null;
      return tx.reservation.create({
        data: { medicineId: params.id, userId: session.userId, quantity },
        include: { medicine: true },
      });
    });

    if (!reservation) {
      return NextResponse.json({ error: "Not enough stock to reserve." }, { status: 409 });
    }

    sendMockNotification({
      channel: "email",
      to: session.email,
      subject: `Pickup reservation: ${reservation.medicine.name}`,
      body: `Hi ${session.name}, ${quantity} x ${reservation.medicine.name} is reserved at shelf ${reservation.medicine.locationShelf}. Please collect within 48 hours.`,
    });
    await sendWhatsApp(
      session.phone ?? null,
      `MEDIVA 🏥\nHi ${session.name}, *${quantity} x ${reservation.medicine.name}* is reserved for pickup at shelf ${reservation.medicine.locationShelf}. Please collect within 48 hours.`
    );

    return NextResponse.json(
      {
        reservation: {
          id: reservation.id,
          medicine: reservation.medicine.name,
          quantity: reservation.quantity,
          status: reservation.status,
        },
      },
      { status: 201 }
    );
  } catch (error) {
    return errorResponse(error);
  }
}
