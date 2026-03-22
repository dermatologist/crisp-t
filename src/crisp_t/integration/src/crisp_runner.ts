/**
 * crisp_runner.ts
 *
 * Thin wrapper around the CRISP-T command-line tools (crisp, crispt, crispviz).
 * Executes them as child processes and returns their combined stdout/stderr output.
 *
 * Allowed commands are kept in an explicit allowlist to prevent command injection.
 */

import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

/** Commands that the bot is permitted to execute. */
const ALLOWED_COMMANDS = ["crisp", "crispt", "crispviz"] as const;
type AllowedCommand = (typeof ALLOWED_COMMANDS)[number];

/** Result returned by {@link runCrispCommand}. */
export interface CrispResult {
  success: boolean;
  output: string;
}

/**
 * Validate that a command string is one of the permitted CRISP-T commands.
 *
 * @param command - The command name to check.
 * @returns `true` when the command is in the allowlist.
 */
export function isAllowedCommand(command: string): command is AllowedCommand {
  return (ALLOWED_COMMANDS as readonly string[]).includes(command);
}

/**
 * Parse a raw user message into a command name and its arguments.
 *
 * The expected format is:  `crisp <args>`, `crispt <args>`, or `crispviz <args>`.
 * Returns `null` when the message does not start with a recognised command.
 *
 * @param text - The raw message text sent by the Teams user.
 * @returns Parsed `{ command, args }` or `null`.
 */
export function parseCrispCommand(
  text: string
): { command: AllowedCommand; args: string[] } | null {
  const trimmed = text.trim();

  for (const cmd of ALLOWED_COMMANDS) {
    // Match "<command>" at the start, optionally followed by a space and arguments.
    if (trimmed === cmd || trimmed.startsWith(cmd + " ")) {
      const rest = trimmed.slice(cmd.length).trim();
      // Split remaining text into argv-style tokens (respects quoted strings).
      const args = rest.length > 0 ? splitArgs(rest) : [];
      return { command: cmd, args };
    }
  }

  return null;
}

/**
 * Execute an allowed CRISP-T CLI command with the given arguments.
 *
 * @param command   - One of: crisp | crispt | crispviz
 * @param args      - Argument list for the command.
 * @param dataPath  - Working directory in which to run the command.
 * @param timeoutMs - Maximum execution time in milliseconds (default 300 000).
 * @returns A {@link CrispResult} containing success flag and combined output.
 */
export async function runCrispCommand(
  command: AllowedCommand,
  args: string[],
  dataPath: string,
  timeoutMs = 300_000
): Promise<CrispResult> {
  if (!isAllowedCommand(command)) {
    return {
      success: false,
      output: `Error: '${command}' is not a permitted CRISP-T command. Allowed: ${ALLOWED_COMMANDS.join(", ")}`,
    };
  }

  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      cwd: dataPath,
      timeout: timeoutMs,
      // Limit output size to prevent memory issues with large result sets.
      maxBuffer: 10 * 1024 * 1024, // 10 MB
    });

    let output = stdout ?? "";
    if (stderr) {
      output += `\n\nWarnings / Errors:\n${stderr}`;
    }

    return { success: true, output: output.trim() || "Command completed successfully (no output)." };
  } catch (err: unknown) {
    const e = err as NodeJS.ErrnoException & { stdout?: string; stderr?: string; killed?: boolean };

    // Surface as much context as possible for the Teams user.
    if (e.killed) {
      return {
        success: false,
        output: `Error: command '${command}' timed out after ${timeoutMs / 1000} seconds.`,
      };
    }

    if (e.code === "ENOENT") {
      return {
        success: false,
        output: `Error: '${command}' was not found. Make sure CRISP-T is installed and on PATH.`,
      };
    }

    // Non-zero exit: include stdout/stderr if available.
    const output =
      [e.stdout, e.stderr].filter(Boolean).join("\n").trim() ||
      e.message;

    return { success: false, output: `Command failed:\n${output}` };
  }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Split a command-argument string into tokens, honouring single- and
 * double-quoted groups (e.g. `--title "my study"` → `["--title", "my study"]`).
 *
 * This avoids shell invocation and prevents shell-injection via argument values.
 */
function splitArgs(input: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let inSingle = false;
  let inDouble = false;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (ch === "'" && !inDouble) {
      inSingle = !inSingle;
    } else if (ch === '"' && !inSingle) {
      inDouble = !inDouble;
    } else if (ch === " " && !inSingle && !inDouble) {
      if (current) {
        tokens.push(current);
        current = "";
      }
    } else {
      current += ch;
    }
  }

  if (current) {
    tokens.push(current);
  }

  return tokens;
}
