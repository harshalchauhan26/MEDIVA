import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
import Providers from "@/components/Providers";
import AppShell from "@/components/AppShell";
import ChatWidget from "@/components/ChatWidget";
import FeedbackWidget from "@/components/FeedbackWidget";
import { getSession } from "@/lib/auth";

export const metadata: Metadata = {
  title: "MediVa Health — Provider Portal",
  description:
    "Medicine inventory, doctor appointments, and the MediVa 24/7 AI health assistant.",
};

export const dynamic = "force-dynamic";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const session = getSession();
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <Providers>
          <AppShell session={session}>{children}</AppShell>
          <ChatWidget />
          <FeedbackWidget />
        </Providers>
      </body>
    </html>
  );
}
