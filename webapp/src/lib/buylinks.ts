export type BuyLink = { app: string; url: string };

/**
 * Map a medicine to "buy online" search links on popular Indian pharmacy apps.
 * These are deep search URLs (not affiliate links) generated from the name, so
 * they stay valid even as the partner catalogues change.
 */
export function buyLinksFor(name: string): BuyLink[] {
  const q = encodeURIComponent(name);
  const slug = name.trim().toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
  return [
    { app: "Tata 1mg", url: `https://www.1mg.com/search/all?name=${q}` },
    { app: "PharmEasy", url: `https://pharmeasy.in/search/all?name=${q}` },
    { app: "Netmeds", url: `https://www.netmeds.com/catalogsearch/result?q=${q}` },
    { app: "Apollo", url: `https://www.apollopharmacy.in/search-medicines/${slug}` },
  ];
}
