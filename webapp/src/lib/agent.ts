import { prisma } from "@/lib/db";
import type { Session } from "@/lib/auth";
import { computeOpenSlots, parseDateParam } from "@/lib/slots";
import { bookAppointment } from "@/lib/booking";
import { serializeMedicine } from "@/lib/api";

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8000";

export const SYSTEM_PROMPT = `You are MediVa, the 24/7 AI health assistant for the MEDIVA healthcare platform.

STRICT RULES:
- You are an AI assistant, NOT a doctor. Never diagnose, prescribe, or replace professional medical care. For emergencies, tell the user to contact local emergency services immediately.
- For general medical knowledge questions (conditions, symptoms, treatments), ALWAYS call search_medical_knowledge, which retrieves passages from the indexed medical encyclopedia. Answer only from what it returns; if it has no answer, say so.
- For pharmacy stock, price, dosage, or shelf questions, call check_medicine_stock. All prices are in Indian Rupees (₹). When relevant, share the online buying links it returns (Tata 1mg, PharmEasy, Netmeds, Apollo).
- For finding doctors or open appointment times, call find_available_slots.
- Only call book_appointment after the user has clearly confirmed the doctor, the exact time, and a reason/symptoms. Booking requires the user to be logged in as a patient; if the tool reports they are not, politely ask them to log in first.
- Privacy (HIPAA framing): never reveal information about other patients, other people's appointments, or internal records beyond what the tools return for this user.
- Keep answers concise, warm, and structured. Dates/times you mention must match tool results exactly.`;

export const AGENT_TOOLS = [
  {
    type: "function" as const,
    function: {
      name: "search_medical_knowledge",
      description:
        "Search the indexed medical reference documents (RAG over the Gale Encyclopedia of Medicine) for general medical knowledge: conditions, symptoms, causes, treatments.",
      parameters: {
        type: "object",
        properties: {
          question: { type: "string", description: "The medical question to look up." },
        },
        required: ["question"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "check_medicine_stock",
      description:
        "Check live pharmacy inventory for a medicine by brand or generic name. Returns stock quantity, price, dosage, shelf location, and expiry.",
      parameters: {
        type: "object",
        properties: {
          name: { type: "string", description: "Medicine brand or generic name, e.g. 'paracetamol'." },
        },
        required: ["name"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "find_available_slots",
      description:
        "Find doctors (optionally by specialty) and their open appointment slots on a given date.",
      parameters: {
        type: "object",
        properties: {
          specialty: {
            type: "string",
            description: "Specialty filter, e.g. 'Cardiology'. Omit to search all doctors.",
          },
          date: { type: "string", description: "Date in YYYY-MM-DD format." },
        },
        required: ["date"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "book_appointment",
      description:
        "Book an appointment for the logged-in patient. Only call after the user confirmed doctor, exact slot time, and reason.",
      parameters: {
        type: "object",
        properties: {
          doctorId: { type: "string", description: "Doctor id from find_available_slots." },
          startsAt: { type: "string", description: "Exact slot start time in ISO 8601 format." },
          symptoms: { type: "string", description: "The patient's reason for the visit." },
        },
        required: ["doctorId", "startsAt", "symptoms"],
      },
    },
  },
];

async function searchMedicalKnowledge(question: string): Promise<string> {
  try {
    const response = await fetch(`${RAG_API_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!response.ok) {
      return JSON.stringify({ error: `Knowledge service returned ${response.status}.` });
    }
    const data = await response.json();
    return JSON.stringify({
      answer: data.answer,
      sources: (data.sources ?? []).map((s: { source?: string; page?: number }) => ({
        source: s.source,
        page: s.page,
      })),
    });
  } catch {
    return JSON.stringify({
      error: "The medical knowledge service is unreachable. Tell the user general knowledge lookups are temporarily unavailable.",
    });
  }
}

async function checkMedicineStock(name: string): Promise<string> {
  const medicines = await prisma.medicine.findMany({
    where: {
      OR: [
        { name: { contains: name, mode: "insensitive" } },
        { genericName: { contains: name, mode: "insensitive" } },
      ],
    },
    take: 5,
  });
  if (medicines.length === 0) {
    return JSON.stringify({ found: false, message: `No medicine matching "${name}" in inventory.` });
  }
  return JSON.stringify({
    found: true,
    currency: "INR",
    medicines: medicines.map((m) => {
      const s = serializeMedicine(m);
      return {
        name: s.name,
        genericName: s.genericName,
        inStock: s.quantity > 0,
        quantity: s.quantity,
        priceInr: s.price,
        priceLabel: `₹${s.price.toFixed(2)}`,
        dosage: s.dosage,
        shelf: s.locationShelf,
        expiryDate: s.expiryDate.slice(0, 10),
        lowStock: s.lowStock,
        buyOnline: s.buyLinks,
      };
    }),
  });
}

async function findAvailableSlots(specialty: string | undefined, dateStr: string): Promise<string> {
  const date = parseDateParam(dateStr);
  if (!date) return JSON.stringify({ error: "date must be in YYYY-MM-DD format." });

  const dayEnd = new Date(date);
  dayEnd.setDate(dayEnd.getDate() + 1);

  const doctors = await prisma.doctor.findMany({
    where: specialty
      ? { specialty: { contains: specialty, mode: "insensitive" } }
      : undefined,
    include: {
      user: { select: { name: true } },
      availability: true,
      appointments: { where: { startsAt: { gte: date, lt: dayEnd } } },
    },
    take: 10,
  });
  if (doctors.length === 0) {
    return JSON.stringify({ doctors: [], message: `No doctors found${specialty ? ` for specialty "${specialty}"` : ""}.` });
  }

  return JSON.stringify({
    date: dateStr,
    doctors: doctors.map((d) => ({
      doctorId: d.id,
      name: d.user.name,
      specialty: d.specialty,
      rating: d.rating,
      status: d.status,
      openSlots:
        d.status === "ON_LEAVE"
          ? []
          : computeOpenSlots(d.availability, d.appointments, date)
              .slice(0, 12)
              .map((s) => s.toISOString()),
    })),
  });
}

async function bookAppointmentTool(
  session: Session | null,
  args: { doctorId: string; startsAt: string; symptoms: string }
): Promise<string> {
  if (!session) {
    return JSON.stringify({ error: "User is not logged in. Ask them to log in as a patient first." });
  }
  if (session.role !== "PATIENT") {
    return JSON.stringify({ error: `User is logged in as ${session.role}, not a patient. Only patients can book.` });
  }
  const result = await bookAppointment({
    patientId: session.userId,
    patientName: session.name,
    patientEmail: session.email,
    patientPhone: session.phone,
    doctorId: args.doctorId,
    startsAt: new Date(args.startsAt),
    symptoms: args.symptoms,
  });
  if (!result.ok) return JSON.stringify({ error: result.error });
  return JSON.stringify({
    booked: true,
    appointmentId: result.appointment.id,
    doctorName: result.appointment.doctorName,
    startsAt: result.appointment.startsAt,
    status: result.appointment.status,
    note: "A mock confirmation email/SMS was triggered.",
  });
}

export async function executeTool(
  name: string,
  rawArgs: string,
  session: Session | null
): Promise<string> {
  let args: Record<string, string>;
  try {
    args = JSON.parse(rawArgs || "{}");
  } catch {
    return JSON.stringify({ error: "Invalid tool arguments." });
  }

  switch (name) {
    case "search_medical_knowledge":
      return searchMedicalKnowledge(args.question ?? "");
    case "check_medicine_stock":
      return checkMedicineStock(args.name ?? "");
    case "find_available_slots":
      return findAvailableSlots(args.specialty, args.date ?? "");
    case "book_appointment":
      return bookAppointmentTool(session, {
        doctorId: args.doctorId ?? "",
        startsAt: args.startsAt ?? "",
        symptoms: args.symptoms ?? "",
      });
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}
