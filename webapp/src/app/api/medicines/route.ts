import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { requireRole } from "@/lib/auth";
import { errorResponse, serializeMedicine } from "@/lib/api";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q")?.trim();

  const medicines = await prisma.medicine.findMany({
    where: q
      ? {
          OR: [
            { name: { contains: q, mode: "insensitive" } },
            { genericName: { contains: q, mode: "insensitive" } },
            { sku: { contains: q, mode: "insensitive" } },
          ],
        }
      : undefined,
    orderBy: { name: "asc" },
  });

  return NextResponse.json({ medicines: medicines.map(serializeMedicine) });
}

const CreateSchema = z.object({
  name: z.string().min(1),
  genericName: z.string().min(1),
  sku: z.string().min(1),
  batchNumber: z.string().min(1),
  dosage: z.string().min(1),
  quantity: z.number().int().min(0),
  price: z.number().min(0),
  locationShelf: z.string().min(1),
  expiryDate: z.string().datetime(),
});

export async function POST(request: Request) {
  try {
    requireRole("PHARMACIST", "ADMIN");
    const parsed = CreateSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.issues[0].message }, { status: 400 });
    }
    const medicine = await prisma.medicine.create({
      data: { ...parsed.data, expiryDate: new Date(parsed.data.expiryDate) },
    });
    return NextResponse.json({ medicine: serializeMedicine(medicine) }, { status: 201 });
  } catch (error) {
    return errorResponse(error);
  }
}
