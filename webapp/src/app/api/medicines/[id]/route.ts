import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireRole } from "@/lib/auth";
import { errorResponse, serializeMedicine } from "@/lib/api";

const UpdateSchema = z.object({
  name: z.string().min(1).optional(),
  genericName: z.string().min(1).optional(),
  batchNumber: z.string().min(1).optional(),
  dosage: z.string().min(1).optional(),
  quantity: z.number().int().min(0).optional(),
  price: z.number().min(0).optional(),
  locationShelf: z.string().min(1).optional(),
  expiryDate: z.string().datetime().optional(),
});

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  try {
    requireRole("PHARMACIST", "ADMIN");
    const parsed = UpdateSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.issues[0].message }, { status: 400 });
    }
    const { expiryDate, ...rest } = parsed.data;
    const medicine = await prisma.medicine.update({
      where: { id: params.id },
      data: { ...rest, ...(expiryDate ? { expiryDate: new Date(expiryDate) } : {}) },
    });
    return NextResponse.json({ medicine: serializeMedicine(medicine) });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function DELETE(_request: Request, { params }: { params: { id: string } }) {
  try {
    requireRole("PHARMACIST", "ADMIN");
    await prisma.reservation.deleteMany({ where: { medicineId: params.id } });
    await prisma.medicine.delete({ where: { id: params.id } });
    return NextResponse.json({ ok: true });
  } catch (error) {
    return errorResponse(error);
  }
}
