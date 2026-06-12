import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { getSession, requireRole } from "@/lib/auth";
import { errorResponse } from "@/lib/api";

const FeedbackSchema = z.object({
  rating: z.number().int().min(1).max(5),
  message: z.string().min(1, "Tell us a little more.").max(2000),
  page: z.string().max(200).optional(),
  email: z.string().email().max(120).optional().or(z.literal("")),
});

// Anyone (logged in or not) can submit evaluation feedback.
export async function POST(request: Request) {
  const parsed = FeedbackSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    const message = parsed.error?.issues?.[0]?.message ?? "Invalid feedback.";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  const session = getSession();
  await prisma.feedback.create({
    data: {
      rating: parsed.data.rating,
      message: parsed.data.message.trim(),
      page: parsed.data.page || null,
      email: parsed.data.email || session?.email || null,
      userId: session?.userId ?? null,
    },
  });

  return NextResponse.json({ ok: true }, { status: 201 });
}

// Admins can read collected feedback (newest first).
export async function GET() {
  try {
    requireRole("ADMIN");
    const items = await prisma.feedback.findMany({
      orderBy: { createdAt: "desc" },
      take: 200,
    });
    return NextResponse.json({ items });
  } catch (error) {
    return errorResponse(error);
  }
}
