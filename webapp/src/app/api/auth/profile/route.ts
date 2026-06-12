import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { getSession, setSessionCookie, AuthError } from "@/lib/auth";
import { errorResponse } from "@/lib/api";
import { normalizeIndianPhone } from "@/lib/phone";

const ProfileSchema = z.object({
  name: z.string().min(2).max(80),
  phone: z.string().min(8).max(20),
});

export async function PATCH(request: Request) {
  try {
    const session = getSession();
    if (!session) throw new AuthError(401, "Login required.");

    const parsed = ProfileSchema.safeParse(await request.json().catch(() => null));
    if (!parsed.success) {
      return NextResponse.json({ error: "Name and mobile number are required." }, { status: 400 });
    }

    const phone = normalizeIndianPhone(parsed.data.phone);
    if (!phone) {
      return NextResponse.json(
        { error: "Enter a valid 10-digit Indian mobile number." },
        { status: 400 }
      );
    }

    const user = await prisma.user.update({
      where: { id: session.userId },
      data: { name: parsed.data.name.trim(), phone },
    });

    setSessionCookie({
      userId: user.id,
      role: user.role,
      name: user.name,
      email: user.email,
      phone: user.phone,
      image: user.image,
    });

    return NextResponse.json({ name: user.name, phone: user.phone });
  } catch (error) {
    return errorResponse(error);
  }
}
