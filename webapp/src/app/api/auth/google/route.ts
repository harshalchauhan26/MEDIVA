import { NextResponse } from "next/server";
import crypto from "crypto";
import { cookies } from "next/headers";
import { buildGoogleAuthUrl, googleConfigured } from "@/lib/google";

export const dynamic = "force-dynamic";

export function GET(request: Request) {
  if (!googleConfigured()) {
    return NextResponse.redirect(new URL("/login?error=google_not_configured", request.url));
  }

  const origin = new URL(request.url).origin;
  const redirectUri = `${origin}/api/auth/google/callback`;
  const state = crypto.randomBytes(16).toString("hex");

  cookies().set("g_oauth_state", state, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 600,
  });

  return NextResponse.redirect(buildGoogleAuthUrl(redirectUri, state));
}
