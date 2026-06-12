import crypto from "crypto";
import { cookies } from "next/headers";
import type { Role } from "@prisma/client";

const COOKIE_NAME = "mediva_session";
const MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

export type Session = {
  userId: string;
  role: Role;
  name: string;
  email: string;
  phone?: string | null;
  image?: string | null;
};

function secret(): string {
  return process.env.AUTH_SECRET || "dev-secret-change-me";
}

function sign(payload: string): string {
  return crypto.createHmac("sha256", secret()).update(payload).digest("base64url");
}

export function encodeSession(session: Session): string {
  const payload = Buffer.from(JSON.stringify(session)).toString("base64url");
  return `${payload}.${sign(payload)}`;
}

export function decodeSession(token: string): Session | null {
  const [payload, signature] = token.split(".");
  if (!payload || !signature) return null;
  const expected = sign(payload);
  const a = Buffer.from(signature);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    return JSON.parse(Buffer.from(payload, "base64url").toString()) as Session;
  } catch {
    return null;
  }
}

export function getSession(): Session | null {
  const token = cookies().get(COOKIE_NAME)?.value;
  if (!token) return null;
  return decodeSession(token);
}

export function setSessionCookie(session: Session): void {
  cookies().set(COOKIE_NAME, encodeSession(session), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: MAX_AGE_SECONDS,
  });
}

export function clearSessionCookie(): void {
  cookies().delete(COOKIE_NAME);
}

export function requireRole(...roles: Role[]): Session {
  const session = getSession();
  if (!session) throw new AuthError(401, "Login required.");
  if (!roles.includes(session.role)) {
    throw new AuthError(403, "You do not have permission to do this.");
  }
  return session;
}

export class AuthError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}
