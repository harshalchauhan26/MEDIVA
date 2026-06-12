import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { verifyPassword } from "@/lib/passwords";
import { setSessionCookie } from "@/lib/auth";

const LoginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

export async function POST(request: Request) {
  const parsed = LoginSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: "Email and password are required." }, { status: 400 });
  }

  const user = await prisma.user.findUnique({ where: { email: parsed.data.email.toLowerCase() } });
  if (!user || !user.passwordHash || !verifyPassword(parsed.data.password, user.passwordHash)) {
    return NextResponse.json(
      { error: "Invalid email or password. Google accounts must use 'Continue with Google'." },
      { status: 401 }
    );
  }

  setSessionCookie({
    userId: user.id,
    role: user.role,
    name: user.name,
    email: user.email,
    phone: user.phone,
    image: user.image,
  });
  return NextResponse.json({ id: user.id, name: user.name, role: user.role });
}
