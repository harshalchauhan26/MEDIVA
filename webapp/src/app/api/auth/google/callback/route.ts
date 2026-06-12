import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { prisma } from "@/lib/db";
import { setSessionCookie } from "@/lib/auth";
import { exchangeCodeForUser, googleConfigured } from "@/lib/google";

export const dynamic = "force-dynamic";

const ROLE_HOME: Record<string, string> = {
  PATIENT: "/doctors",
  DOCTOR: "/doctor/dashboard",
  PHARMACIST: "/admin/inventory",
  ADMIN: "/admin/inventory",
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const savedState = cookies().get("g_oauth_state")?.value;
  cookies().delete("g_oauth_state");

  if (!googleConfigured() || !code || !state || state !== savedState) {
    return NextResponse.redirect(new URL("/login?error=google_failed", request.url));
  }

  try {
    const redirectUri = `${url.origin}/api/auth/google/callback`;
    const profile = await exchangeCodeForUser(code, redirectUri);

    // Find or create the user. New Google sign-ins are patients.
    const user = await prisma.user.upsert({
      where: { email: profile.email },
      update: { image: profile.picture ?? undefined },
      create: {
        email: profile.email,
        name: profile.name,
        image: profile.picture,
        role: "PATIENT",
        passwordHash: null,
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

    // Patients without a mobile number must onboard before using WhatsApp flows.
    if (user.role === "PATIENT" && !user.phone) {
      return NextResponse.redirect(new URL("/onboarding", request.url));
    }
    return NextResponse.redirect(new URL(ROLE_HOME[user.role] ?? "/", request.url));
  } catch {
    return NextResponse.redirect(new URL("/login?error=google_failed", request.url));
  }
}
