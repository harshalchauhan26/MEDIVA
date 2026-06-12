type WhatsAppResult = {
  channel: "whatsapp";
  to: string | null;
  body: string;
  delivered: boolean;
  waLink?: string;
  sentAt: string;
};

const recent: WhatsAppResult[] = [];

function record(result: WhatsAppResult): WhatsAppResult {
  recent.push(result);
  if (recent.length > 100) recent.shift();
  return result;
}

/**
 * Send a WhatsApp message to a patient.
 *
 * - If WHATSAPP_TOKEN + WHATSAPP_PHONE_NUMBER_ID are configured, it uses the
 *   Meta WhatsApp Cloud API to actually deliver the message.
 * - Otherwise it logs the message and returns a wa.me click-to-chat link so the
 *   flow is fully functional in development without credentials.
 */
export async function sendWhatsApp(to: string | null, body: string): Promise<WhatsAppResult> {
  const sentAt = new Date().toISOString();

  if (!to) {
    console.log(`[WHATSAPP skipped] no phone on file\n${body}`);
    return record({ channel: "whatsapp", to: null, body, delivered: false, sentAt });
  }

  const e164 = to.replace(/\D/g, "");
  const waLink = `https://wa.me/${e164}?text=${encodeURIComponent(body)}`;
  const token = process.env.WHATSAPP_TOKEN;
  const phoneNumberId = process.env.WHATSAPP_PHONE_NUMBER_ID;

  if (!token || !phoneNumberId) {
    console.log(`[WHATSAPP mock] to=${to}\n${body}\nlink: ${waLink}`);
    return record({ channel: "whatsapp", to, body, delivered: false, waLink, sentAt });
  }

  try {
    const response = await fetch(
      `https://graph.facebook.com/v21.0/${phoneNumberId}/messages`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messaging_product: "whatsapp",
          to: e164,
          type: "text",
          text: { body },
        }),
      }
    );
    const delivered = response.ok;
    if (!delivered) {
      console.error(`[WHATSAPP error] ${response.status} ${await response.text()}`);
    }
    return record({ channel: "whatsapp", to, body, delivered, waLink, sentAt });
  } catch (error) {
    console.error("[WHATSAPP error]", error);
    return record({ channel: "whatsapp", to, body, delivered: false, waLink, sentAt });
  }
}

export function recentWhatsApp(): WhatsAppResult[] {
  return [...recent].reverse();
}
