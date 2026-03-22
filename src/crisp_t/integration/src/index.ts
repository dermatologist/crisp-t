/**
 * index.ts  –  CRISP-T Microsoft Teams Bot
 *
 * Entry point for the Chat SDK Teams integration.  The bot:
 *   1. Starts an HTTP webhook server that Microsoft Teams delivers events to.
 *   2. Listens for @-mentions and direct messages from Teams users.
 *   3. Parses messages to detect CRISP-T commands (crisp / crispt / crispviz).
 *   4. Executes recognised commands via {@link runCrispCommand} and posts the
 *      output back to the originating Teams thread.
 *   5. Replies with a help message for any non-command input.
 *
 * Environment variables (see .env.example):
 *   TEAMS_APP_ID       – Azure Bot Registration Application ID  (required)
 *   TEAMS_APP_PASSWORD – Azure Bot Registration client secret   (required)
 *   PORT               – Webhook listener port (default: 3978)
 *   CRISP_DATA_PATH    – Working directory for CRISP commands (default: ./workspace)
 *   CRISP_COMMAND_TIMEOUT – Max execution time in seconds (default: 300)
 *   LOG_LEVEL          – Logging verbosity (default: info)
 */

import "dotenv/config";
import * as fs from "fs";
import * as path from "path";

import { Chat } from "chat";
import { createTeamsAdapter } from "@chat-adapter/teams";
import { createMemoryState } from "@chat-adapter/state-memory";

import { parseCrispCommand, runCrispCommand } from "./crisp_runner";

// ─── Configuration ────────────────────────────────────────────────────────────

const APP_ID = process.env.TEAMS_APP_ID;
const APP_PASSWORD = process.env.TEAMS_APP_PASSWORD;
const PORT = parseInt(process.env.PORT ?? "3978", 10);
const DATA_PATH = process.env.CRISP_DATA_PATH ?? "./workspace";
const TIMEOUT_MS =
  parseInt(process.env.CRISP_COMMAND_TIMEOUT ?? "300", 10) * 1000;

if (!APP_ID || !APP_PASSWORD) {
  console.error(
    "❌  TEAMS_APP_ID and TEAMS_APP_PASSWORD must be set in environment variables (see .env.example)."
  );
  process.exit(1);
}

// Ensure the workspace directory exists so CRISP-T commands have a place to
// write output files even on a fresh deployment.
const resolvedDataPath = path.resolve(DATA_PATH);
if (!fs.existsSync(resolvedDataPath)) {
  try {
    fs.mkdirSync(resolvedDataPath, { recursive: true });
    console.log(`📁 Created workspace directory: ${resolvedDataPath}`);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(
      `❌ Failed to create workspace directory '${resolvedDataPath}': ${msg}\n` +
        "   Check that the process has write permission to the parent directory."
    );
    process.exit(1);
  }
}

// ─── Help text ────────────────────────────────────────────────────────────────

const HELP_TEXT = `
👋 **CRISP-T Teams Bot**

I can run CRISP-T qualitative research commands on your behalf.
Just @mention me (or send a DM) with one of the commands below.

**Available commands**

| Command | Description |
|---------|-------------|
| \`crisp <args>\` | Run the main CRISP-T analysis tool |
| \`crispt <args>\` | Corpus management and tagging |
| \`crispviz <args>\` | Visualisation and reporting |

**Examples**

\`\`\`
@crisp-bot crisp --help
@crisp-bot crispt --help
@crisp-bot crispviz --help
@crisp-bot crisp load-corpus --path data/study.csv
\`\`\`

> **Tip:** All data files are read from and saved to the configured workspace path.
> Contact your administrator to change the workspace location.
`.trim();

// ─── Chat SDK bot setup ───────────────────────────────────────────────────────

const bot = new Chat({
  userName: "crisp-bot",
  adapters: {
    teams: createTeamsAdapter({
      appId: APP_ID,
      appPassword: APP_PASSWORD,
    }),
  },
  // In-memory state is fine for a single-node deployment.
  // For multi-node or persistent deployments swap this for a Redis or
  // PostgreSQL state adapter (see Chat SDK docs).
  state: createMemoryState(),
  // Deduplicate events that may be delivered more than once by Teams.
  dedupeTtlMs: 60_000,
});

// ─── Event handlers ───────────────────────────────────────────────────────────

/**
 * Handle a new @mention in a channel thread that the bot is not yet subscribed
 * to.  Subscribe so subsequent replies are also processed, then respond.
 */
bot.onNewMention(async (thread, message) => {
  // Subscribe to keep receiving follow-up messages in this thread.
  await thread.subscribe();
  await handleUserMessage(thread, message.text ?? "");
});

/**
 * Handle a direct message to the bot in a DM thread.
 */
bot.onDirectMessage(async (thread, message) => {
  await thread.subscribe();
  await handleUserMessage(thread, message.text ?? "");
});

/**
 * Handle any subsequent message in a thread the bot is already subscribed to.
 */
bot.onSubscribedMessage(async (thread, message) => {
  await handleUserMessage(thread, message.text ?? "");
});

// ─── Core message handler ─────────────────────────────────────────────────────

/**
 * Inspect the user's message text, decide whether it is a CRISP-T command,
 * and post an appropriate reply back to the Teams thread.
 *
 * @param thread - The Chat SDK thread context (used to post replies).
 * @param text   - Raw message text from the Teams user.
 */
async function handleUserMessage(
  thread: Parameters<Parameters<typeof bot.onNewMention>[0]>[0],
  text: string
): Promise<void> {
  // Strip the bot @mention from the beginning of the message if present
  // (Teams often prepends "<at>bot-name</at>" to @mention messages).
  const clean = stripMention(text).trim();

  // Empty or whitespace-only message → show help.
  if (!clean) {
    await thread.post(HELP_TEXT);
    return;
  }

  // "help" → show help.
  if (/^help$/i.test(clean)) {
    await thread.post(HELP_TEXT);
    return;
  }

  // Attempt to parse as a CRISP-T command.
  const parsed = parseCrispCommand(clean);

  if (!parsed) {
    // Unknown input – provide usage guidance.
    await thread.post(
      `❓ I didn't recognise that as a CRISP-T command.\n\nType \`help\` to see available commands, or try:\n\`crisp --help\``
    );
    return;
  }

  // Let the user know we're working on it while the command runs.
  // Truncate the argument display to avoid surfacing sensitive values that
  // users may have inadvertently passed as command arguments.
  const argPreview = parsed.args.length > 0
    ? " " + parsed.args.join(" ").slice(0, 80) + (parsed.args.join(" ").length > 80 ? "…" : "")
    : "";
  await thread.post(`⏳ Running \`${parsed.command}${argPreview}\`…`);

  try {
    const result = await runCrispCommand(
      parsed.command,
      parsed.args,
      resolvedDataPath,
      TIMEOUT_MS
    );

    if (result.success) {
      await thread.post(
        `✅ **\`${parsed.command}\` completed**\n\n\`\`\`\n${result.output}\n\`\`\``
      );
    } else {
      await thread.post(
        `❌ **\`${parsed.command}\` failed**\n\n\`\`\`\n${result.output}\n\`\`\``
      );
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[crisp-bot] Unexpected error running command:", msg);
    await thread.post(
      `⚠️ An unexpected error occurred while running the command:\n\`${msg}\`\n\nPlease check the bot logs for details.`
    );
  }
}

// ─── Webhook server ───────────────────────────────────────────────────────────

// Wire the Teams webhook to an HTTP server.  The adapter exposes a standard
// Node `http.RequestListener` via `bot.webhooks.teams`.
import * as http from "http";

const server = http.createServer(async (req, res) => {
  // Delegate all requests to the Teams adapter webhook handler.
  await bot.webhooks.teams(req, res);
});

server.listen(PORT, () => {
  console.log(`🤖 CRISP-T Teams Bot listening on port ${PORT}`);
  console.log(`📁 Workspace directory: ${resolvedDataPath}`);
  console.log("Press Ctrl+C to stop.\n");
});

// ─── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Remove Teams @mention XML tags (e.g. `<at>BotName</at>`) from message text.
 * Teams embeds these tags when a user @-mentions the bot; stripping them gives
 * us the clean command string the user intended.
 */
function stripMention(text: string): string {
  return text.replace(/<at>[^<]*<\/at>/g, "").trim();
}
