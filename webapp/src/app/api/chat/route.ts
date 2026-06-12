import Groq from "groq-sdk";
import { z } from "zod";
import { getSession } from "@/lib/auth";
import { AGENT_TOOLS, SYSTEM_PROMPT, executeTool } from "@/lib/agent";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const ChatSchema = z.object({
  messages: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1).max(4000),
      })
    )
    .min(1)
    .max(30),
});

const MAX_TOOL_ROUNDS = 5;

const encoder = new TextEncoder();

function event(payload: Record<string, unknown>): Uint8Array {
  return encoder.encode(JSON.stringify(payload) + "\n");
}

export async function POST(request: Request) {
  if (!process.env.GROQ_API_KEY) {
    return Response.json({ error: "GROQ_API_KEY is not configured." }, { status: 500 });
  }

  const parsed = ChatSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return Response.json({ error: "Invalid chat payload." }, { status: 400 });
  }

  const session = getSession();
  const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
  const model = process.env.AGENT_MODEL || "llama-3.3-70b-versatile";

  const sessionNote = session
    ? `The user is logged in as ${session.name} (role: ${session.role}).`
    : "The user is NOT logged in.";

  // Keep only the recent turns to bound prompt size.
  const history = parsed.data.messages.slice(-12);
  const messages: Groq.Chat.Completions.ChatCompletionMessageParam[] = [
    { role: "system", content: `${SYSTEM_PROMPT}\n\n${sessionNote}` },
    ...history.map((m) => ({ role: m.role, content: m.content })),
  ];

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
          const completion = await groq.chat.completions.create({
            model,
            messages,
            tools: AGENT_TOOLS,
            tool_choice: "auto",
            temperature: 0.3,
            max_tokens: 900,
          });

          const choice = completion.choices[0]?.message;
          if (!choice) break;

          const toolCalls = choice.tool_calls ?? [];
          if (toolCalls.length === 0) {
            // Final answer: stream it out in word chunks.
            const text = choice.content ?? "";
            const words = text.split(/(\s+)/);
            let buffer = "";
            for (const word of words) {
              buffer += word;
              if (buffer.length >= 24) {
                controller.enqueue(event({ type: "token", text: buffer }));
                buffer = "";
              }
            }
            if (buffer) controller.enqueue(event({ type: "token", text: buffer }));
            break;
          }

          messages.push(choice);
          for (const call of toolCalls) {
            controller.enqueue(event({ type: "tool", name: call.function.name }));
            const result = await executeTool(call.function.name, call.function.arguments, session);
            messages.push({ role: "tool", tool_call_id: call.id, content: result });
          }
        }
        controller.enqueue(event({ type: "done" }));
      } catch (error) {
        console.error("MediVa agent error:", error);
        controller.enqueue(
          event({ type: "error", message: "MediVa hit a problem answering. Please try again." })
        );
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache",
    },
  });
}
