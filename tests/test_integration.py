"""Tests for the CRISP-T Microsoft Teams integration module.

These tests are structural / unit-level and do not require a running Teams
environment, a running crisp-ui server, or real Azure credentials.  They
validate:

- File and directory structure of the integration package
- TypeScript source correctness (via tsc --noEmit)
- Command routing logic (via subprocess calling Node)
- Configuration and environment-variable defaults
- package.json and tsconfig.json contents
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_DIR = (
    Path(__file__).parent.parent / "src" / "crisp_t" / "integration"
)
SRC_DIR = INTEGRATION_DIR / "src"


def node_available() -> bool:
    """Return True if Node.js ≥ 18 is available on PATH."""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False
        version = result.stdout.strip().lstrip("v")
        major = int(version.split(".")[0])
        return major >= 18
    except Exception:
        return False


def npm_available() -> bool:
    """Return True if npm is available on PATH."""
    try:
        result = subprocess.run(
            ["npm", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


requires_node = pytest.mark.skipif(
    not node_available(), reason="Node.js >=18 not available"
)
requires_npm = pytest.mark.skipif(
    not npm_available(), reason="npm not available"
)

# ─────────────────────────────────────────────────────────────────────────────
# Structure tests
# ─────────────────────────────────────────────────────────────────────────────


def test_integration_directory_exists():
    """The integration directory must exist."""
    assert INTEGRATION_DIR.is_dir(), (
        f"Integration directory not found: {INTEGRATION_DIR}"
    )


def test_src_directory_exists():
    """The src/ subdirectory must exist inside integration/."""
    assert SRC_DIR.is_dir(), f"src/ directory not found: {SRC_DIR}"


def test_index_ts_exists():
    """The main bot file src/index.ts must exist."""
    assert (SRC_DIR / "index.ts").is_file(), "src/index.ts not found"


def test_package_json_exists():
    """package.json must exist in the integration directory."""
    assert (INTEGRATION_DIR / "package.json").is_file(), "package.json not found"


def test_tsconfig_json_exists():
    """tsconfig.json must exist in the integration directory."""
    assert (INTEGRATION_DIR / "tsconfig.json").is_file(), "tsconfig.json not found"


def test_env_example_exists():
    """.env.example must exist in the integration directory."""
    assert (INTEGRATION_DIR / ".env.example").is_file(), ".env.example not found"


def test_node_modules_exist():
    """node_modules/ must exist (i.e. npm install has been run)."""
    assert (INTEGRATION_DIR / "node_modules").is_dir(), (
        "node_modules/ not found — run `npm install` inside src/crisp_t/integration/"
    )


def test_dist_directory_exists():
    """dist/ must exist (i.e. tsc has been run)."""
    assert (INTEGRATION_DIR / "dist").is_dir(), (
        "dist/ not found — run `npm run build` inside src/crisp_t/integration/"
    )


def test_compiled_index_js_exists():
    """The compiled JavaScript entry point dist/index.js must exist."""
    assert (INTEGRATION_DIR / "dist" / "index.js").is_file(), (
        "dist/index.js not found — run `npm run build`"
    )


# ─────────────────────────────────────────────────────────────────────────────
# package.json content tests
# ─────────────────────────────────────────────────────────────────────────────


def test_package_json_name():
    """package.json must have the expected package name."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    assert pkg.get("name") == "crisp-t-teams-bot"


def test_package_json_has_start_script():
    """package.json must have a 'start' script."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    assert "start" in pkg.get("scripts", {}), "Missing 'start' script in package.json"


def test_package_json_has_build_script():
    """package.json must have a 'build' script."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    assert "build" in pkg.get("scripts", {}), "Missing 'build' script in package.json"


def test_package_json_chat_dependency():
    """package.json must list 'chat' as a dependency."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = pkg.get("dependencies", {})
    assert "chat" in deps, "'chat' not found in dependencies"


def test_package_json_teams_adapter_dependency():
    """package.json must list '@chat-adapter/teams' as a dependency."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = pkg.get("dependencies", {})
    assert "@chat-adapter/teams" in deps, (
        "'@chat-adapter/teams' not found in dependencies"
    )


def test_package_json_state_memory_dependency():
    """package.json must list '@chat-adapter/state-memory' as a dependency."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = pkg.get("dependencies", {})
    assert "@chat-adapter/state-memory" in deps, (
        "'@chat-adapter/state-memory' not found in dependencies"
    )


def test_package_json_axios_dependency():
    """package.json must list 'axios' as a dependency."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = pkg.get("dependencies", {})
    assert "axios" in deps, "'axios' not found in dependencies"


def test_package_json_express_dependency():
    """package.json must list 'express' as a dependency."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = pkg.get("dependencies", {})
    assert "express" in deps, "'express' not found in dependencies"


def test_package_json_node_engine():
    """package.json should specify a minimum Node.js version of >=18."""
    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    engines = pkg.get("engines", {})
    node_req = engines.get("node", "")
    assert node_req, "No 'engines.node' field in package.json"
    assert "18" in node_req, f"Expected Node >=18 in engines.node, got: {node_req}"


# ─────────────────────────────────────────────────────────────────────────────
# tsconfig.json content tests
# ─────────────────────────────────────────────────────────────────────────────


def test_tsconfig_strict_mode():
    """tsconfig.json must enable strict mode."""
    tsconfig = json.loads((INTEGRATION_DIR / "tsconfig.json").read_text())
    assert tsconfig.get("compilerOptions", {}).get("strict") is True, (
        "tsconfig.json does not have strict=true"
    )


def test_tsconfig_outdir():
    """tsconfig.json must have outDir set to ./dist."""
    tsconfig = json.loads((INTEGRATION_DIR / "tsconfig.json").read_text())
    out_dir = tsconfig.get("compilerOptions", {}).get("outDir", "")
    assert "dist" in out_dir, f"Unexpected outDir in tsconfig.json: {out_dir}"


# ─────────────────────────────────────────────────────────────────────────────
# Source code content tests
# ─────────────────────────────────────────────────────────────────────────────


def _read_index_ts() -> str:
    return (SRC_DIR / "index.ts").read_text()


def test_index_ts_imports_chat():
    """src/index.ts must import from the 'chat' package."""
    src = _read_index_ts()
    assert 'from "chat"' in src, "src/index.ts does not import from 'chat'"


def test_index_ts_imports_teams_adapter():
    """src/index.ts must import from '@chat-adapter/teams'."""
    src = _read_index_ts()
    assert "@chat-adapter/teams" in src, (
        "src/index.ts does not import from '@chat-adapter/teams'"
    )


def test_index_ts_imports_state_memory():
    """src/index.ts must import from '@chat-adapter/state-memory'."""
    src = _read_index_ts()
    assert "@chat-adapter/state-memory" in src, (
        "src/index.ts does not import '@chat-adapter/state-memory'"
    )


def test_index_ts_has_on_new_mention():
    """src/index.ts must register an onNewMention handler."""
    src = _read_index_ts()
    assert "onNewMention" in src, "src/index.ts does not call onNewMention"


def test_index_ts_has_on_direct_message():
    """src/index.ts must register an onDirectMessage handler."""
    src = _read_index_ts()
    assert "onDirectMessage" in src, "src/index.ts does not call onDirectMessage"


def test_index_ts_has_list_command():
    """src/index.ts must handle the @list / /list command."""
    src = _read_index_ts()
    assert "@list" in src or "listModels" in src, (
        "src/index.ts does not handle the @list command"
    )


def test_index_ts_has_switch_command():
    """src/index.ts must handle the @switch / /switch command."""
    src = _read_index_ts()
    assert "@switch" in src or "switchModel" in src, (
        "src/index.ts does not handle the @switch command"
    )


def test_index_ts_has_crisp_command():
    """src/index.ts must handle the @crisp / /crisp command."""
    src = _read_index_ts()
    assert "@crisp" in src or "sendCrispMessage" in src, (
        "src/index.ts does not handle the @crisp command"
    )


def test_index_ts_has_clear_command():
    """src/index.ts must handle the @clear / /clear command."""
    src = _read_index_ts()
    assert "@clear" in src or "clearSession" in src, (
        "src/index.ts does not handle the @clear command"
    )


def test_index_ts_has_help_command():
    """src/index.ts must handle the @help / /help command."""
    src = _read_index_ts()
    assert "@help" in src or "getHelpText" in src, (
        "src/index.ts does not handle the @help command"
    )


def test_index_ts_has_health_check():
    """src/index.ts must include a health-check function for crisp-ui."""
    src = _read_index_ts()
    assert "isCrispUIRunning" in src, (
        "src/index.ts does not define isCrispUIRunning"
    )


def test_index_ts_has_start_server():
    """src/index.ts must attempt to start crisp-ui if not running."""
    src = _read_index_ts()
    assert "startCrispUI" in src or "crisp-ui" in src, (
        "src/index.ts does not reference crisp-ui startup"
    )


def test_index_ts_has_ensure_session():
    """src/index.ts must include session management (ensureSession)."""
    src = _read_index_ts()
    assert "ensureSession" in src, (
        "src/index.ts does not define ensureSession"
    )


def test_index_ts_has_error_handling():
    """src/index.ts must include try/catch error handling."""
    src = _read_index_ts()
    assert "try {" in src or "try{" in src, (
        "src/index.ts appears to have no try/catch error handling"
    )


def test_index_ts_has_webhook_route():
    """src/index.ts must expose the Teams webhook endpoint."""
    src = _read_index_ts()
    assert "/api/messages" in src, (
        "src/index.ts does not define the /api/messages webhook endpoint"
    )


def test_index_ts_has_health_endpoint():
    """src/index.ts must expose a /health HTTP endpoint."""
    src = _read_index_ts()
    assert "/health" in src, (
        "src/index.ts does not define a /health endpoint"
    )


def test_index_ts_exports_route_message():
    """src/index.ts must export the routeMessage function for testability."""
    src = _read_index_ts()
    assert "export async function routeMessage" in src, (
        "src/index.ts does not export routeMessage"
    )


def test_index_ts_exports_bot():
    """src/index.ts must export the bot instance."""
    src = _read_index_ts()
    assert "export const bot" in src, (
        "src/index.ts does not export the bot instance"
    )


def test_index_ts_exports_main():
    """src/index.ts must export a main() entry-point function."""
    src = _read_index_ts()
    assert "export async function main" in src, (
        "src/index.ts does not export main()"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TypeScript type-check test
# ─────────────────────────────────────────────────────────────────────────────


@requires_node
@requires_npm
def test_typescript_type_check():
    """TypeScript source must pass tsc --noEmit (no type errors)."""
    result = subprocess.run(
        ["npx", "tsc", "--noEmit"],
        capture_output=True,
        text=True,
        cwd=str(INTEGRATION_DIR),
        timeout=120,
    )
    assert result.returncode == 0, (
        f"TypeScript type errors detected:\n{result.stdout}\n{result.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routing logic tests (via compiled JS)
# ─────────────────────────────────────────────────────────────────────────────


@requires_node
def _run_routing_test(message: str, timeout: int = 15) -> dict:
    """
    Run a short Node.js ESM snippet that imports the compiled bot and calls
    routeMessage().  Returns a dict with 'result' key (or 'error'/'output').

    routeMessage() catches all internal errors and returns them as strings,
    so no HTTP server is required for these structural tests.
    """
    escaped = message.replace("\\", "\\\\").replace("'", "\\'")
    script = f"""
const {{ routeMessage }} = await import('./dist/index.js');
const result = await routeMessage('{escaped}');
process.stdout.write(JSON.stringify({{ result }}) + '\\n');
"""
    run = subprocess.run(
        ["node", "--input-type=module"],
        input=script,
        capture_output=True,
        text=True,
        cwd=str(INTEGRATION_DIR),
        timeout=timeout,
        env={**os.environ, "TEAMS_APP_ID": "test", "TEAMS_APP_PASSWORD": "test"},
    )
    # Find the last valid JSON line in stdout
    for line in reversed(run.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"output": run.stdout, "error": run.stderr}


@requires_node
def test_routing_list_command():
    """@list command should return a non-null response (models list or error)."""
    data = _run_routing_test("@list")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    # Should be non-null: either a model list or an error message (no server running)
    assert result is not None
    assert isinstance(result, str) and len(result) > 0


@requires_node
def test_routing_help_command():
    """@help command should return a non-empty help string (no HTTP needed)."""
    data = _run_routing_test("@help")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert "crisp" in result.lower() or "@list" in result or "@crisp" in result


@requires_node
def test_routing_clear_command():
    """@clear command should return a non-null response (no HTTP needed for clear)."""
    data = _run_routing_test("@clear")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert isinstance(result, str) and len(result) > 0


@requires_node
def test_routing_switch_command_no_model():
    """@switch without a model name should return an error message (no HTTP)."""
    data = _run_routing_test("@switch")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert "model" in result.lower() or "usage" in result.lower() or "❌" in result


@requires_node
def test_routing_crisp_no_payload():
    """/crisp without a payload should return an error message (no HTTP)."""
    data = _run_routing_test("/crisp")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert "message" in result.lower() or "usage" in result.lower() or "❌" in result


@requires_node
def test_routing_unknown_message_returns_none():
    """A plain message not starting with a command should return None (ignored)."""
    data = _run_routing_test("Hello, how are you?")
    assert "result" in data, f"Unexpected output: {data}"
    assert data["result"] is None, (
        f"Expected None for unaddressed message, got: {data['result']}"
    )


@requires_node
def test_routing_slash_list_command():
    """/list should return a non-null response (same routing as @list)."""
    data = _run_routing_test("/list")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert isinstance(result, str) and len(result) > 0


@requires_node
def test_routing_slash_help_command():
    """/help should return the same help text as @help."""
    data = _run_routing_test("/help")
    assert "result" in data, f"Unexpected output: {data}"
    result = data["result"]
    assert result is not None
    assert "crisp" in result.lower() or "@list" in result or "@crisp" in result


# ─────────────────────────────────────────────────────────────────────────────
# .env.example content tests
# ─────────────────────────────────────────────────────────────────────────────


def test_env_example_teams_app_id():
    """The .env.example must document TEAMS_APP_ID."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "TEAMS_APP_ID" in content


def test_env_example_teams_app_password():
    """The .env.example must document TEAMS_APP_PASSWORD."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "TEAMS_APP_PASSWORD" in content


def test_env_example_crisp_ui_url():
    """The .env.example must document CRISP_UI_URL."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "CRISP_UI_URL" in content


def test_env_example_port():
    """The .env.example must document PORT."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "PORT" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
