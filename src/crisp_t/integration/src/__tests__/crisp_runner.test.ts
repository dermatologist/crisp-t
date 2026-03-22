/**
 * crisp_runner.test.ts
 *
 * Unit tests for the crisp_runner module.
 * These tests cover argument parsing and command validation without
 * spawning actual CRISP-T processes.
 */

import { isAllowedCommand, parseCrispCommand } from "../crisp_runner";

// ─── isAllowedCommand ─────────────────────────────────────────────────────────

describe("isAllowedCommand", () => {
  it("accepts 'crisp'", () => {
    expect(isAllowedCommand("crisp")).toBe(true);
  });

  it("accepts 'crispt'", () => {
    expect(isAllowedCommand("crispt")).toBe(true);
  });

  it("accepts 'crispviz'", () => {
    expect(isAllowedCommand("crispviz")).toBe(true);
  });

  it("rejects unknown commands", () => {
    expect(isAllowedCommand("rm")).toBe(false);
    expect(isAllowedCommand("bash")).toBe(false);
    expect(isAllowedCommand("")).toBe(false);
    expect(isAllowedCommand("CRISP")).toBe(false); // case-sensitive
  });
});

// ─── parseCrispCommand ────────────────────────────────────────────────────────

describe("parseCrispCommand", () => {
  it("parses a bare command with no arguments", () => {
    const result = parseCrispCommand("crisp");
    expect(result).not.toBeNull();
    expect(result!.command).toBe("crisp");
    expect(result!.args).toEqual([]);
  });

  it("parses a command with simple arguments", () => {
    const result = parseCrispCommand("crisp --help");
    expect(result).not.toBeNull();
    expect(result!.command).toBe("crisp");
    expect(result!.args).toEqual(["--help"]);
  });

  it("parses multiple arguments correctly", () => {
    const result = parseCrispCommand("crispt load-corpus --path data/study.csv");
    expect(result).not.toBeNull();
    expect(result!.command).toBe("crispt");
    expect(result!.args).toEqual(["load-corpus", "--path", "data/study.csv"]);
  });

  it("handles quoted arguments as a single token", () => {
    const result = parseCrispCommand('crispviz --title "My Study Results"');
    expect(result).not.toBeNull();
    expect(result!.command).toBe("crispviz");
    expect(result!.args).toContain("My Study Results");
  });

  it("strips leading/trailing whitespace before parsing", () => {
    const result = parseCrispCommand("  crisp --help  ");
    expect(result).not.toBeNull();
    expect(result!.command).toBe("crisp");
    expect(result!.args).toEqual(["--help"]);
  });

  it("returns null for non-CRISP messages", () => {
    expect(parseCrispCommand("hello world")).toBeNull();
    expect(parseCrispCommand("help")).toBeNull();
    expect(parseCrispCommand("rm -rf /")).toBeNull();
  });

  it("returns null for empty string", () => {
    expect(parseCrispCommand("")).toBeNull();
  });

  it("does not match a command that is only a prefix of an allowed command", () => {
    // 'cris' is not a valid command even though it starts with 'cri'
    expect(parseCrispCommand("cris --help")).toBeNull();
  });
});
