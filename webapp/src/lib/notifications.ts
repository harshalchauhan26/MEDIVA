type Notification = {
  channel: "email" | "sms";
  to: string;
  subject: string;
  body: string;
  sentAt: string;
};

const recent: Notification[] = [];

/**
 * Mock notification trigger. In production this would call an email/SMS
 * provider (SendGrid, Twilio, etc.). Here it logs to the server console
 * and keeps an in-memory trail for debugging.
 */
export function sendMockNotification(input: Omit<Notification, "sentAt">): Notification {
  const notification: Notification = { ...input, sentAt: new Date().toISOString() };
  recent.push(notification);
  if (recent.length > 100) recent.shift();
  console.log(
    `[MOCK ${input.channel.toUpperCase()}] to=${input.to} subject="${input.subject}"\n${input.body}`
  );
  return notification;
}

export function recentNotifications(): Notification[] {
  return [...recent].reverse();
}
