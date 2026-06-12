import { describe, it, expect } from "vitest";
import { buyLinksFor } from "@/lib/buylinks";

describe("buyLinksFor", () => {
  const links = buyLinksFor("Paracetamol 500mg");

  it("returns the four partner pharmacies", () => {
    expect(links).toHaveLength(4);
    expect(links.map((l) => l.app)).toEqual(["Tata 1mg", "PharmEasy", "Netmeds", "Apollo"]);
  });

  it("produces valid https URLs", () => {
    for (const link of links) expect(link.url).toMatch(/^https:\/\//);
  });

  it("url-encodes the medicine name", () => {
    expect(links[0].url).toContain("Paracetamol%20500mg");
  });
});
