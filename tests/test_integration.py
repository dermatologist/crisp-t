"""Tests for the CRISP-T Microsoft Teams integration module.

These tests verify the static structure of the integration directory (source
files, configuration, documentation) without spawning a live Teams bot or
running npm commands.  This keeps CI fast and dependency-free for the Python
test suite.
"""

from pathlib import Path

import pytest

# Root of the integration directory
INTEGRATION_DIR = (
    Path(__file__).parent.parent / "src" / "crisp_t" / "integration"
)


# ─── Directory structure ──────────────────────────────────────────────────────


def test_integration_directory_exists():
    """The integration directory must be present."""
    assert INTEGRATION_DIR.is_dir(), (
        f"Expected integration directory at {INTEGRATION_DIR}"
    )


def test_integration_src_directory_exists():
    """The TypeScript source directory must be present."""
    assert (INTEGRATION_DIR / "src").is_dir()


def test_integration_tests_directory_exists():
    """The TypeScript unit-test directory must be present."""
    assert (INTEGRATION_DIR / "src" / "__tests__").is_dir()


# ─── Required files ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "relative_path",
    [
        "package.json",
        "tsconfig.json",
        ".env.example",
        "README.md",
        "src/index.ts",
        "src/crisp_runner.ts",
        "src/__tests__/crisp_runner.test.ts",
    ],
)
def test_required_file_exists(relative_path: str):
    """Every required file must be present in the integration directory."""
    target = INTEGRATION_DIR / relative_path
    assert target.exists(), f"Required file not found: {target}"


# ─── package.json content ─────────────────────────────────────────────────────


def test_package_json_has_chat_sdk_dependency():
    """package.json must declare the chat npm package as a dependency."""
    import json

    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "chat" in deps, "package.json must declare 'chat' as a dependency"


def test_package_json_has_teams_adapter():
    """package.json must declare the @chat-adapter/teams package."""
    import json

    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "@chat-adapter/teams" in deps, (
        "package.json must declare '@chat-adapter/teams' as a dependency"
    )


def test_package_json_has_build_script():
    """package.json must define a 'build' script (TypeScript compilation)."""
    import json

    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    scripts = pkg.get("scripts", {})
    assert "build" in scripts, "package.json must define a 'build' script"


def test_package_json_has_start_script():
    """package.json must define a 'start' script."""
    import json

    pkg = json.loads((INTEGRATION_DIR / "package.json").read_text())
    scripts = pkg.get("scripts", {})
    assert "start" in scripts, "package.json must define a 'start' script"


# ─── TypeScript source content ────────────────────────────────────────────────


def test_index_ts_uses_teams_adapter():
    """index.ts must import and use the Teams adapter."""
    source = (INTEGRATION_DIR / "src" / "index.ts").read_text()
    assert "createTeamsAdapter" in source, (
        "index.ts must import createTeamsAdapter from @chat-adapter/teams"
    )


def test_index_ts_uses_chat_sdk():
    """index.ts must import Chat from the 'chat' package."""
    source = (INTEGRATION_DIR / "src" / "index.ts").read_text()
    assert 'from "chat"' in source or "from 'chat'" in source, (
        "index.ts must import Chat from 'chat'"
    )


def test_index_ts_handles_mentions():
    """index.ts must register an onNewMention handler."""
    source = (INTEGRATION_DIR / "src" / "index.ts").read_text()
    assert "onNewMention" in source


def test_index_ts_handles_direct_messages():
    """index.ts must register an onDirectMessage handler."""
    source = (INTEGRATION_DIR / "src" / "index.ts").read_text()
    assert "onDirectMessage" in source


def test_index_ts_handles_subscribed_messages():
    """index.ts must register an onSubscribedMessage handler."""
    source = (INTEGRATION_DIR / "src" / "index.ts").read_text()
    assert "onSubscribedMessage" in source


def test_crisp_runner_ts_exports_parse_function():
    """crisp_runner.ts must export parseCrispCommand."""
    source = (INTEGRATION_DIR / "src" / "crisp_runner.ts").read_text()
    assert "parseCrispCommand" in source


def test_crisp_runner_ts_exports_run_function():
    """crisp_runner.ts must export runCrispCommand."""
    source = (INTEGRATION_DIR / "src" / "crisp_runner.ts").read_text()
    assert "runCrispCommand" in source


def test_crisp_runner_ts_has_allowlist():
    """crisp_runner.ts must define an ALLOWED_COMMANDS list."""
    source = (INTEGRATION_DIR / "src" / "crisp_runner.ts").read_text()
    assert "ALLOWED_COMMANDS" in source


def test_crisp_runner_uses_execfile_not_exec():
    """crisp_runner.ts must use execFile (not exec) to avoid shell injection."""
    source = (INTEGRATION_DIR / "src" / "crisp_runner.ts").read_text()
    assert "execFile" in source, (
        "crisp_runner.ts must use execFile to avoid shell injection vulnerabilities"
    )


# ─── Environment template ─────────────────────────────────────────────────────


def test_env_example_has_app_id():
    """.env.example must document the TEAMS_APP_ID variable."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "TEAMS_APP_ID" in content


def test_env_example_has_app_password():
    """.env.example must document the TEAMS_APP_PASSWORD variable."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "TEAMS_APP_PASSWORD" in content


def test_env_example_has_crisp_data_path():
    """.env.example must document the CRISP_DATA_PATH variable."""
    content = (INTEGRATION_DIR / ".env.example").read_text()
    assert "CRISP_DATA_PATH" in content


# ─── Documentation ────────────────────────────────────────────────────────────


def test_readme_exists_in_integration():
    """A README.md must exist inside the integration directory."""
    assert (INTEGRATION_DIR / "README.md").exists()


def test_docs_integration_md_exists():
    """docs/integration.md must exist at the repository level."""
    docs_path = (
        Path(__file__).parent.parent / "docs" / "integration.md"
    )
    assert docs_path.exists(), f"Expected docs/integration.md at {docs_path}"


def test_notes_integration_md_exists():
    """notes/integration.md must exist at the repository level."""
    notes_path = (
        Path(__file__).parent.parent / "notes" / "integration.md"
    )
    assert notes_path.exists(), f"Expected notes/integration.md at {notes_path}"
