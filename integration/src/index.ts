/**
 * CRISP-T Chat Bot — Microsoft Teams + Slack
 *
 * This module implements a chat bot for CRISP-T using the Chat SDK.  It
 * bridges both the Microsoft Teams and Slack interfaces with the CRISP-T web
 * UI server (crisp-ui), letting researchers run analyses from either platform.
 *
 * Supported commands (mention the bot or use in DMs):
 *   @list / /list         — List available AI models
 *   @switch <model>       — Switch to a different AI model
 *   @crisp <message>      — Send a message to the active CRISP-T session
 *   @clear / /clear       — Clear the current CRISP-T session
 *   @help / /help         — Show available commands
 *
 * Webhook endpoints:
 *   POST /api/messages    — Microsoft Teams webhook
 *   POST /slack/events    — Slack Events API webhook
 *
 * Setup:
 *   1. Set environment variables (see .env.example)
 *   2. npm install && npm run build
 *   3. Start crisp-ui: crisp-ui (or let the bot attempt to start it)
 *   4. npm start
 *
 * @module crisp-t-bot
 */

import { spawn } from "child_process";
import * as http from "http";
import { fileURLToPath } from "url";

import axios, { AxiosError } from "axios";
import { Chat } from "chat";
import { createTeamsAdapter } from "@chat-adapter/teams";
import { createSlackAdapter } from "@chat-adapter/slack";
import { createMemoryState } from "@chat-adapter/state-memory";
import express, { Request as ExpressRequest, Response as ExpressResponse } from "express";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration from environment variables
// ─────────────────────────────────────────────────────────────────────────────

/** Base URL for the crisp-ui server */
const CRISP_UI_BASE_URL = process.env.CRISP_UI_URL ?? "http://127.0.0.1:5000";

/** Default AI model for new sessions */
const DEFAULT_MODEL = process.env.CRISP_DEFAULT_MODEL ?? "gpt-4.1";

/** Unique session ID used for the bot's CRISP-T session */
const SESSION_ID = "crisp-bot-session";

/** Port the bot HTTP server listens on */
const PORT = parseInt(process.env.PORT ?? "3978", 10);

/** Assistant-response polling settings (30 x 1s = ~30s max). */
const RESPONSE_POLL_MAX_ATTEMPTS = parseInt(
  process.env.CRISP_RESPONSE_POLL_MAX_ATTEMPTS ?? "30",
  10,
);
const RESPONSE_POLL_DELAY_MS = parseInt(
  process.env.CRISP_RESPONSE_POLL_DELAY_MS ?? "1000",
  10,
);

// ─────────────────────────────────────────────────────────────────────────────
// Session state
// ─────────────────────────────────────────────────────────────────────────────

/** The currently selected AI model */
let currentModel = DEFAULT_MODEL;

/** Whether a CRISP-T session has been created */
let sessionActive = false;

// ─────────────────────────────────────────────────────────────────────────────
// Health checking and server lifecycle
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check whether the crisp-ui server is reachable.
 * @returns true if the health endpoint responds successfully
 */
export async function isCrispUIRunning(): Promise<boolean> {
  try {
    const response = await axios.get(`${CRISP_UI_BASE_URL}/api/health`, {
      timeout: 3000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
}

/**
 * Attempt to start the crisp-ui server as a detached background process.
 * Waits up to 8 seconds for the server to become healthy.
 *
 * @returns true if the server started successfully, false otherwise
 */
export async function startCrispUI(): Promise<boolean> {
  return new Promise((resolve) => {
    try {
      const proc = spawn("crisp-ui", [], {
        detached: true,
        stdio: "ignore",
        shell: false,
      });
      proc.unref();
    } catch (err) {
      console.error("[crisp-t-bot] Failed to spawn crisp-ui:", err);
      resolve(false);
      return;
    }

    // Poll every second for up to 8 attempts
    let attempts = 0;
    const maxAttempts = 8;
    const interval = setInterval(async () => {
      attempts++;
      if (await isCrispUIRunning()) {
        clearInterval(interval);
        resolve(true);
        return;
      }
      if (attempts >= maxAttempts) {
        clearInterval(interval);
        resolve(false);
      }
    }, 1000);
  });
}

/**
 * Ensure the crisp-ui server is running.  Tries to start it if not already up.
 *
 * @returns true if the server is now running
 */
export async function ensureCrispUIRunning(): Promise<boolean> {
  if (await isCrispUIRunning()) {
    return true;
  }
  console.log("[crisp-t-bot] crisp-ui server not detected — attempting to start it...");
  const started = await startCrispUI();
  if (started) {
    console.log("[crisp-t-bot] crisp-ui server started successfully.");
  } else {
    console.error(
      "[crisp-t-bot] Could not start crisp-ui server. " +
        "Please run `crisp-ui` manually before starting the bot.",
    );
  }
  return started;
}

// ─────────────────────────────────────────────────────────────────────────────
// CRISP-T session management
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create the CRISP-T session if it does not already exist.
 * @throws if the session could not be created
 */
export async function ensureSession(): Promise<void> {
  if (sessionActive) return;

  await axios.post(`${CRISP_UI_BASE_URL}/api/session/create`, {
    session_id: SESSION_ID,
    model: currentModel,
    config: {},
  });
  sessionActive = true;
}

/**
 * Destroy the current CRISP-T session so it will be recreated on next use.
 */
export async function destroySession(): Promise<void> {
  if (!sessionActive) return;
  try {
    await axios.post(`${CRISP_UI_BASE_URL}/api/session/${SESSION_ID}/destroy`);
  } catch {
    // Session may already be gone — that is fine
  }
  sessionActive = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// CRISP-T command handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve a formatted list of available AI models from the crisp-ui server.
 */
export async function listModels(): Promise<string> {
  const response = await axios.get(`${CRISP_UI_BASE_URL}/api/models`);
  const models = response.data.models as string[];
  if (!models || models.length === 0) {
    return "No models found.";
  }
  const lines = models.map((m, i) => `${i + 1}. ${m}`).join("\n");
  return `**Available models:**\n${lines}\n\n_Current model: ${currentModel}_`;
}

/**
 * Switch the active AI model.  Destroys the current session so the next
 * message will create a fresh one with the new model.
 *
 * @param modelName - Name of the model to switch to
 */
export async function switchModel(modelName: string): Promise<string> {
  await destroySession();
  currentModel = modelName;
  // Create a new session immediately to validate the model name
  await ensureSession();
  return `Switched to model: **${modelName}**`;
}

/**
 * Send a message to the active CRISP-T session and return the assistant reply.
 *
 * Polls for the response with a short back-off.  Returns the last assistant
 * message in the history.
 *
 * @param message - The user's message text to forward to CRISP-T
 */
export async function sendCrispMessage(message: string): Promise<string> {
  await ensureSession();

  // Send the message
  await axios.post(`${CRISP_UI_BASE_URL}/api/session/${SESSION_ID}/send`, {
    prompt: message,
  });

  // Poll for the assistant response (up to ~30 s)
  for (let i = 0; i < RESPONSE_POLL_MAX_ATTEMPTS; i++) {
    await sleep(RESPONSE_POLL_DELAY_MS);

    const resp = await axios.get(
      `${CRISP_UI_BASE_URL}/api/session/${SESSION_ID}/messages`,
    );
    const messages = resp.data.messages as Array<{
      role: string;
      content: string;
      complete?: boolean;
    }>;

    // Find the last assistant message that is complete
    const lastAssistant = [...messages]
      .reverse()
      .find((m) => m.role === "assistant" && m.complete === true);

    if (lastAssistant) {
      return lastAssistant.content;
    }
  }

  return "⚠️ No response received within the timeout period. The model may still be processing.";
}

/**
 * Clear the active CRISP-T session.
 */
export async function clearSession(): Promise<string> {
  await destroySession();
  return "Session cleared. A new session will be created automatically on your next `@crisp` command.";
}

/**
 * Return the help message listing all supported commands.
 * @param platform - Optional platform name included in the header ('Teams' | 'Slack')
 */
export function getHelpText(platform?: string): string {
  const header = platform
    ? `**CRISP-T ${platform} Bot — available commands:**`
    : "**CRISP-T Bot — available commands:**";
  return [
    header,
    "",
    "• `@list` or `/list` — List available AI models",
    "• `@switch <model>` or `/switch <model>` — Switch to a different AI model",
    "• `@crisp <message>` or `/crisp <message>` — Send a message to CRISP-T",
    "• `@clear` or `/clear` — Clear the current CRISP-T session",
    "• `@help` or `/help` — Show this help message",
    "",
    "**Examples:**",
    "```",
    "@crisp Import the CSV from ./data and analyze topics",
    "@switch gpt-4.1",
    "@list",
    "```",
    "",
    `_Current model: ${currentModel}_`,
  ].join("\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Message routing helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Simple promise-based sleep. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Remove HTML tags from incoming message text using a linear scan.
 *
 * This avoids regex backtracking issues on adversarial inputs and keeps
 * normalization predictable for platform-provided message payloads.
 */
function stripHtmlTags(text: string): string {
  let out = "";
  let inTag = false;
  for (const ch of text) {
    if (ch === "<") {
      inTag = true;
      out += " ";
      continue;
    }
    if (ch === ">") {
      inTag = false;
      continue;
    }
    if (!inTag) {
      out += ch;
    }
  }
  return out;
}

/**
 * Extract the text after a command prefix from a raw message string.
 * Returns null if the pattern is not found.
 *
 * @param text  - Raw message text (may include HTML / @-mentions from Teams)
 * @param regex - Pattern with a capture group for the payload
 */
function extractPayload(text: string, regex: RegExp): string | null {
  const match = text.match(regex);
  return match ? match[1].trim() : null;
}

/**
 * Route an incoming message text to the appropriate CRISP-T handler.
 *
 * Returns the reply string, or null if the message is not intended for the bot.
 * All handler errors are caught and returned as user-visible error strings
 * (never thrown), so the caller can always post the returned string safely.
 *
 * @param rawText  - The full text of the incoming message
 * @param platform - Optional platform name for contextual help text ('Teams' | 'Slack')
 */
export async function routeMessage(rawText: string, platform?: string): Promise<string | null> {
  // Normalise: strip Teams HTML tags and collapse whitespace
  const text = stripHtmlTags(rawText).replace(/\s+/g, " ").trim();

  const lower = text.toLowerCase();

  try {
    // ── @list / /list ─────────────────────────────────────────────────────
    if (/(?:^|[\s,])(?:@list|\/list)\b/i.test(lower)) {
      return await listModels();
    }

    // ── @switch / /switch <model> ─────────────────────────────────────────
    if (/(?:^|[\s,])(?:@switch|\/switch)\b/i.test(lower)) {
      const modelName = extractPayload(text, /(?:@switch|\/switch)\s+(\S+)/i);
      if (!modelName) {
        return "❌ Please specify a model name.  Usage: `@switch <model-name>`";
      }
      return await switchModel(modelName);
    }

    // ── @crisp / /crisp <message> ─────────────────────────────────────────
    if (/(?:^|[\s,])(?:@crisp|\/crisp)\b/i.test(lower)) {
      const payload = extractPayload(text, /(?:@crisp|\/crisp)\s+([\s\S]+)/i);
      if (!payload) {
        return "❌ Please provide a message.  Usage: `@crisp <your message>`";
      }
      return await sendCrispMessage(payload);
    }

    // ── @clear / /clear ───────────────────────────────────────────────────
    if (/(?:^|[\s,])(?:@clear|\/clear)\b/i.test(lower)) {
      return await clearSession();
    }

    // ── @help / /help ─────────────────────────────────────────────────────
    if (/(?:^|[\s,])(?:@help|\/help)\b/i.test(lower)) {
      return getHelpText(platform);
    }
  } catch (err) {
    const detail =
      err instanceof AxiosError
        ? `HTTP error (${err.code ?? err.response?.status ?? "?"}): ${err.message}`
        : String(err);
    return `❌ Command failed: ${detail}`;
  }

  // Not a recognised command — ignore
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat SDK bot setup
// ─────────────────────────────────────────────────────────────────────────────

/** The Chat SDK instance — both Teams and Slack adapters registered. */
export const bot = new Chat({
  userName: "crisp-t-bot",
  adapters: {
    teams: createTeamsAdapter(),
    slack: createSlackAdapter(),
  },
  state: createMemoryState(),
});

/**
 * Detect which platform a message originated from by inspecting the raw payload.
 *
 * - Teams (Azure Bot Framework) messages contain `channelId` or `serviceUrl`.
 * - Slack messages contain an `event` wrapper or `team_id`.
 *
 * Returns a human-readable platform name for use in contextual help, or
 * `undefined` if the platform cannot be determined.
 */
function detectPlatform(
  message: Parameters<Parameters<typeof bot.onNewMention>[0]>[1],
): string | undefined {
  const raw = message.raw as Record<string, unknown> | undefined;
  if (!raw) return undefined;
  if (typeof raw["channelId"] === "string" || typeof raw["serviceUrl"] === "string") {
    return "Teams";
  }
  if (typeof raw["event"] === "object" || typeof raw["team_id"] === "string") {
    return "Slack";
  }
  return undefined;
}

/**
 * Shared handler for all incoming messages (mentions and DMs).
 * Checks crisp-ui health, routes the command, and posts the reply.
 * The platform is detected dynamically from the raw message payload.
 */
async function handleMessage(
  thread: Parameters<Parameters<typeof bot.onNewMention>[0]>[0],
  message: Parameters<Parameters<typeof bot.onNewMention>[0]>[1],
): Promise<void> {
  // Subscribe to follow-up messages in the same thread
  await thread.subscribe();

  // Ensure the crisp-ui server is reachable
  const serverRunning = await ensureCrispUIRunning();
  if (!serverRunning) {
    await thread.post(
      "❌ **CRISP-T server is not running.**\n" +
        "Please start it manually with the `crisp-ui` command, then try again.",
    );
    return;
  }

  const rawText = message.text ?? "";
  const platform = detectPlatform(message);

  try {
    const reply = await routeMessage(rawText, platform);

    if (reply !== null) {
      await thread.post(reply);
    }
    // If reply is null the message was not addressed to the bot — stay silent
  } catch (err) {
    const detail =
      err instanceof AxiosError
        ? `HTTP ${err.response?.status ?? "?"}: ${JSON.stringify(err.response?.data)}`
        : String(err);
    const platformLabel = platform ?? "bot";
    console.error(`[crisp-t-bot][${platformLabel}] Error handling message:`, detail);
    await thread.post(`❌ An error occurred: ${detail}`);
  }
}

// Register event handlers for both Teams and Slack
bot.onNewMention(handleMessage);
bot.onDirectMessage(handleMessage);

// ─────────────────────────────────────────────────────────────────────────────
// HTTP server (Express)
// ─────────────────────────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

/**
 * Bridge a Web Fetch API Response back to an Express response.
 */
async function sendWebResponse(webResponse: Response, res: ExpressResponse): Promise<void> {
  res.status(webResponse.status);
  webResponse.headers.forEach((value: string, key: string) => {
    res.setHeader(key, value);
  });
  const body = await webResponse.text();
  res.send(body);
}

/**
 * Build a Web Fetch API Request from an Express request.
 * The Chat SDK webhook handlers require the standard Web Fetch Request type.
 */
function buildWebRequest(req: ExpressRequest): Request {
  const url = `http://${req.headers.host ?? `localhost:${PORT}`}${req.url}`;
  return new Request(url, {
    method: req.method,
    headers: Object.entries(req.headers).reduce<Record<string, string>>(
      (acc, [key, value]) => {
        if (value !== undefined) {
          acc[key] = Array.isArray(value) ? value.join(", ") : value;
        }
        return acc;
      },
      {},
    ),
    body: JSON.stringify(req.body),
  });
}

/**
 * Microsoft Teams webhook endpoint.
 * Receives activity payloads from the Azure Bot Service.
 */
app.post(
  "/api/messages",
  async (req: ExpressRequest, res: ExpressResponse): Promise<void> => {
    try {
      const webResponse = await bot.webhooks.teams(buildWebRequest(req));
      await sendWebResponse(webResponse, res);
    } catch (err) {
      console.error("[crisp-t-bot] Teams webhook error:", err);
      res.status(500).json({ error: "Internal server error" });
    }
  },
);

/**
 * Slack Events API webhook endpoint.
 * Receives event payloads from the Slack Events API.
 */
app.post(
  "/slack/events",
  async (req: ExpressRequest, res: ExpressResponse): Promise<void> => {
    try {
      const webResponse = await bot.webhooks.slack(buildWebRequest(req));
      await sendWebResponse(webResponse, res);
    } catch (err) {
      console.error("[crisp-t-bot] Slack webhook error:", err);
      res.status(500).json({ error: "Internal server error" });
    }
  },
);

/** Simple health check endpoint for monitoring. */
app.get("/health", (_req: ExpressRequest, res: ExpressResponse): void => {
  res.json({ status: "ok", model: currentModel, sessionActive });
});

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Start the bot server and attempt to initialise the CRISP-T session.
 * This function is the main entry point when the module is run directly.
 */
export async function main(): Promise<http.Server> {
  return new Promise((resolve) => {
    const server = app.listen(PORT, async () => {
      console.log(`[crisp-t-bot] Bot listening on port ${PORT}`);
      console.log(`[crisp-t-bot] Teams webhook:  http://localhost:${PORT}/api/messages`);
      console.log(`[crisp-t-bot] Slack webhook:  http://localhost:${PORT}/slack/events`);
      console.log(
        "[crisp-t-bot] Register the Teams URL in your Azure Bot registration as the messaging endpoint.",
      );
      console.log(
        "[crisp-t-bot] Register the Slack URL in your Slack App as the Events API Request URL.",
      );

      // Attempt to ensure crisp-ui is running and create an initial session
      const running = await ensureCrispUIRunning();
      if (running) {
        try {
          await ensureSession();
          console.log(`[crisp-t-bot] CRISP-T session ready (model: ${currentModel})`);
        } catch (err) {
          console.warn(
            "[crisp-t-bot] Could not create initial CRISP-T session:",
            err,
          );
        }
      }

      resolve(server);
    });
  });
}

// Run when this module is the entry point (not when imported in tests)
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] === __filename) {
  main().catch((err) => {
    console.error("[crisp-t-bot] Fatal error:", err);
    process.exit(1);
  });
}
