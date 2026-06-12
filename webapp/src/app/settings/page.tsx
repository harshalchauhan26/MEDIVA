import { redirect } from "next/navigation";
import { getSession } from "@/lib/auth";
import SettingsForm from "@/components/SettingsForm";

export const dynamic = "force-dynamic";

export default function SettingsPage() {
  const session = getSession();
  if (!session) redirect("/login");

  return (
    <div className="mx-auto max-w-2xl">
      <SettingsForm
        initial={{
          name: session.name,
          email: session.email,
          phone: session.phone ?? "",
          role: session.role,
        }}
      />
    </div>
  );
}
