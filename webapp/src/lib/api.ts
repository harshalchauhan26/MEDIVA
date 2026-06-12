import { NextResponse } from "next/server";
import { AuthError } from "@/lib/auth";
import { buyLinksFor } from "@/lib/buylinks";

export function errorResponse(error: unknown): NextResponse {
  if (error instanceof AuthError) {
    return NextResponse.json({ error: error.message }, { status: error.status });
  }
  console.error(error);
  return NextResponse.json({ error: "Something went wrong." }, { status: 500 });
}

export const LOW_STOCK_THRESHOLD = 10;
export const EXPIRY_WARNING_DAYS = 60;

export function isExpiringSoon(expiryDate: Date): boolean {
  const cutoff = Date.now() + EXPIRY_WARNING_DAYS * 24 * 60 * 60 * 1000;
  return expiryDate.getTime() <= cutoff;
}

type MedicineRow = {
  id: string;
  name: string;
  genericName: string;
  sku: string;
  batchNumber: string;
  dosage: string;
  quantity: number;
  price: unknown;
  locationShelf: string;
  expiryDate: Date;
};

export function serializeMedicine(medicine: MedicineRow) {
  return {
    id: medicine.id,
    name: medicine.name,
    genericName: medicine.genericName,
    sku: medicine.sku,
    batchNumber: medicine.batchNumber,
    dosage: medicine.dosage,
    quantity: medicine.quantity,
    price: Number(medicine.price),
    locationShelf: medicine.locationShelf,
    expiryDate: medicine.expiryDate.toISOString(),
    lowStock: medicine.quantity < LOW_STOCK_THRESHOLD,
    expiringSoon: isExpiringSoon(medicine.expiryDate),
    buyLinks: buyLinksFor(medicine.name),
  };
}
