import { NextResponse } from "next/server";
import { z } from "zod";
import { prisma } from "@/lib/db";
import { hashPassword } from "@/lib/passwords";
import { setSessionCookie } from "@/lib/auth";
import { normalizeIndianPhone } from "@/lib/phone";

// Public self-service registration. Real evaluators land here.
// Everyone who signs up via this route is a PATIENT; staff roles are seeded
// or promoted by an admin, never self-assigned.
const SignupSchema = z.object({
  name: z.string().min(2, "Name is too short.").max(80),
  email: z.string().email("Enter a valid email."),
  password: z.string().min(8, "Use at least 8 characters."),
  phone: z.string().min(8).max(20).optional().or(z.literal("")),
});

export async function POST(request: Request) {
  const parsed = SignupSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    const message = parsed.error?.issues?.[0]?.message ?? "Invalid sign-up details.";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  const email = parsed.data.email.toLowerCase().trim();

  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) {
    return NextResponse.json(
      { error: "An account with this email already exists. Try logging in." },
      { status: 409 }
    );
  }

  // Phone is optional at sign-up; it can be collected later in onboarding.
  let phone: string | null = null;
  if (parsed.data.phone) {
    phone = normalizeIndianPhone(parsed.data.phone);
    if (!phone) {
      return NextResponse.json(
        { error: "Enter a valid 10-digit Indian mobile number, or leave it blank." },
        { status: 400 }
      );
    }
  }

  const user = await prisma.user.create({
    data: {
      email,
      name: parsed.data.name.trim(),
      passwordHash: hashPassword(parsed.data.password),
      phone,
      role: "PATIENT",
    },
  });

  setSessionCookie({
    userId: user.id,
    role: user.role,
    name: user.name,
    email: user.email,
    phone: user.phone,
    image: user.image,
  });

  // Patients need a phone for WhatsApp confirmations — send them to onboarding
  // if they skipped it at sign-up.
  return NextResponse.json({
    id: user.id,
    name: user.name,
    role: user.role,
    next: user.phone ? "/doctors" : "/onboarding",
  });
}
