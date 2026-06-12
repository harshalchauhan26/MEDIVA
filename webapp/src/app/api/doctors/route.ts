import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const specialty = searchParams.get("specialty")?.trim();
  const minRating = Number(searchParams.get("minRating") ?? "0");

  const doctors = await prisma.doctor.findMany({
    where: {
      ...(specialty && specialty !== "All"
        ? { specialty: { equals: specialty, mode: "insensitive" } }
        : {}),
      rating: { gte: Number.isFinite(minRating) ? minRating : 0 },
    },
    include: { user: { select: { name: true } }, availability: true },
    orderBy: { rating: "desc" },
  });

  return NextResponse.json({
    doctors: doctors.map((d) => ({
      id: d.id,
      name: d.user.name,
      specialty: d.specialty,
      bio: d.bio,
      rating: d.rating,
      status: d.status,
      availability: d.availability.map((b) => ({
        dayOfWeek: b.dayOfWeek,
        startMinutes: b.startMinutes,
        endMinutes: b.endMinutes,
        slotMinutes: b.slotMinutes,
      })),
    })),
  });
}
