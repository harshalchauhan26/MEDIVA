import { PrismaClient } from "@prisma/client";
import { hashPassword } from "../src/lib/passwords";

const prisma = new PrismaClient();

const PASSWORD = "mediva123";

function daysFromNow(days: number): Date {
  return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
}

async function main() {
  const passwordHash = hashPassword(PASSWORD);

  const admin = await prisma.user.upsert({
    where: { email: "admin@mediva.dev" },
    update: {},
    create: { email: "admin@mediva.dev", name: "Asha Admin", role: "ADMIN", passwordHash },
  });

  await prisma.user.upsert({
    where: { email: "pharma@mediva.dev" },
    update: {},
    create: { email: "pharma@mediva.dev", name: "Pranav Pharmacist", role: "PHARMACIST", passwordHash },
  });

  const patient = await prisma.user.upsert({
    where: { email: "patient@mediva.dev" },
    update: { phone: "+919999999999" },
    create: {
      email: "patient@mediva.dev",
      name: "Priya Patient",
      role: "PATIENT",
      phone: "+919999999999",
      passwordHash,
    },
  });

  const doctorSeeds = [
    {
      email: "dr.rao@mediva.dev",
      name: "Dr. Kavita Rao",
      specialty: "Cardiology",
      bio: "Cardiologist with 12 years of experience in preventive cardiology and hypertension management.",
      rating: 4.8,
    },
    {
      email: "dr.mehta@mediva.dev",
      name: "Dr. Arjun Mehta",
      specialty: "Dermatology",
      bio: "Dermatologist focused on chronic skin conditions, allergies, and pediatric dermatology.",
      rating: 4.6,
    },
    {
      email: "dr.iyer@mediva.dev",
      name: "Dr. Lakshmi Iyer",
      specialty: "General Medicine",
      bio: "General physician handling primary care, diabetes management, and routine checkups.",
      rating: 4.7,
    },
  ];

  const doctors = [] as { id: string; name: string }[];
  for (const seed of doctorSeeds) {
    const user = await prisma.user.upsert({
      where: { email: seed.email },
      update: {},
      create: { email: seed.email, name: seed.name, role: "DOCTOR", passwordHash },
    });
    const doctor = await prisma.doctor.upsert({
      where: { userId: user.id },
      update: {},
      create: {
        userId: user.id,
        specialty: seed.specialty,
        bio: seed.bio,
        rating: seed.rating,
      },
    });
    doctors.push({ id: doctor.id, name: seed.name });

    // Mon-Fri: 09:00-13:00 and 14:00-17:00, 30-minute slots.
    await prisma.availabilityBlock.deleteMany({ where: { doctorId: doctor.id } });
    for (let day = 1; day <= 5; day++) {
      await prisma.availabilityBlock.createMany({
        data: [
          { doctorId: doctor.id, dayOfWeek: day, startMinutes: 9 * 60, endMinutes: 13 * 60, slotMinutes: 30 },
          { doctorId: doctor.id, dayOfWeek: day, startMinutes: 14 * 60, endMinutes: 17 * 60, slotMinutes: 30 },
        ],
      });
    }
  }

  await prisma.reservation.deleteMany();
  await prisma.medicine.deleteMany();
  await prisma.medicine.createMany({
    data: [
      { name: "Paracetamol 500mg", genericName: "Acetaminophen", sku: "MED-PCM-500", batchNumber: "B2401", dosage: "500mg tablet, max 4/day", quantity: 240, price: 30, locationShelf: "A1", expiryDate: daysFromNow(540) },
      { name: "Amoxicillin 250mg", genericName: "Amoxicillin", sku: "MED-AMX-250", batchNumber: "B2402", dosage: "250mg capsule, 3x daily", quantity: 8, price: 85, locationShelf: "A2", expiryDate: daysFromNow(300) },
      { name: "Atorvastatin 10mg", genericName: "Atorvastatin", sku: "MED-ATV-010", batchNumber: "B2403", dosage: "10mg tablet, once nightly", quantity: 120, price: 120, locationShelf: "B1", expiryDate: daysFromNow(45) },
      { name: "Metformin 500mg", genericName: "Metformin HCl", sku: "MED-MET-500", batchNumber: "B2404", dosage: "500mg tablet, with meals", quantity: 180, price: 45, locationShelf: "B2", expiryDate: daysFromNow(420) },
      { name: "Amlodipine 5mg", genericName: "Amlodipine Besylate", sku: "MED-AML-005", batchNumber: "B2405", dosage: "5mg tablet, once daily", quantity: 5, price: 60, locationShelf: "B3", expiryDate: daysFromNow(380) },
      { name: "Cetirizine 10mg", genericName: "Cetirizine HCl", sku: "MED-CTZ-010", batchNumber: "B2406", dosage: "10mg tablet, once daily", quantity: 96, price: 35, locationShelf: "C1", expiryDate: daysFromNow(25) },
      { name: "Omeprazole 20mg", genericName: "Omeprazole", sku: "MED-OMP-020", batchNumber: "B2407", dosage: "20mg capsule, before breakfast", quantity: 64, price: 70, locationShelf: "C2", expiryDate: daysFromNow(500) },
      { name: "Ibuprofen 400mg", genericName: "Ibuprofen", sku: "MED-IBU-400", batchNumber: "B2408", dosage: "400mg tablet, after food", quantity: 150, price: 40, locationShelf: "C3", expiryDate: daysFromNow(610) },
      { name: "Salbutamol Inhaler", genericName: "Albuterol", sku: "MED-SAL-INH", batchNumber: "B2409", dosage: "100mcg/puff, as needed", quantity: 3, price: 280, locationShelf: "D1", expiryDate: daysFromNow(270) },
      { name: "Losartan 50mg", genericName: "Losartan Potassium", sku: "MED-LOS-050", batchNumber: "B2410", dosage: "50mg tablet, once daily", quantity: 88, price: 95, locationShelf: "D2", expiryDate: daysFromNow(330) },
      { name: "Azithromycin 500mg", genericName: "Azithromycin", sku: "MED-AZT-500", batchNumber: "B2411", dosage: "500mg tablet, once daily x3", quantity: 42, price: 130, locationShelf: "D3", expiryDate: daysFromNow(200) },
      { name: "Insulin Glargine", genericName: "Insulin Glargine", sku: "MED-INS-GLA", batchNumber: "B2412", dosage: "100 units/mL pen", quantity: 17, price: 450, locationShelf: "E1 (cold)", expiryDate: daysFromNow(150) },
    ],
  });

  // One sample appointment tomorrow at 09:30 with the cardiologist.
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  tomorrow.setHours(9, 30, 0, 0);
  const cardiologist = doctors[0];
  await prisma.appointment.deleteMany({ where: { patientId: patient.id } });
  // Skip the sample if tomorrow lands on a weekend (outside availability).
  if (tomorrow.getDay() >= 1 && tomorrow.getDay() <= 5) {
    await prisma.appointment.create({
      data: {
        patientId: patient.id,
        doctorId: cardiologist.id,
        startsAt: tomorrow,
        endsAt: new Date(tomorrow.getTime() + 30 * 60 * 1000),
        status: "CONFIRMED",
        symptoms: "Occasional chest tightness after exercise.",
      },
    });
  }

  console.log("Seed complete.");
  console.log(`Demo password for every account: ${PASSWORD}`);
  console.log("Accounts: admin@mediva.dev, pharma@mediva.dev, patient@mediva.dev, dr.rao@mediva.dev, dr.mehta@mediva.dev, dr.iyer@mediva.dev");
  console.log(`Admin user id: ${admin.id}`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
