import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";

export async function GET() {
  const session = getSession();
  if (!session) return NextResponse.json({ user: null });
  return NextResponse.json({
    user: { id: session.userId, name: session.name, role: session.role, email: session.email },
  });
}
