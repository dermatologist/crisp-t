"""Tests for CRISP-T Web UI module."""

import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


def _module_available(module_name: str) -> bool:
    """Check if a module is available for import."""
    return find_spec(module_name) is not None


def test_ui_module_exists():
    """Test that the UI module can be imported."""
    import crisp_t.ui

    assert crisp_t.ui is not None


def test_ui_cli_module_exists():
    """Test that the CLI module can be imported."""
    import crisp_t.ui.cli

    assert crisp_t.ui.cli is not None


def test_ui_cli_has_main():
    """Test that the CLI module has a main function."""
    from crisp_t.ui.cli import main

    assert callable(main), "main should be callable"


def test_ui_static_files_exist():
    """Test that static files exist in the correct locations."""
    ui_dir = Path(__file__).parent.parent / "src" / "crisp_t" / "ui"

    # Check templates
    assert (ui_dir / "templates" / "index.html").exists(), "index.html should exist"

    # Check static files
    assert (ui_dir / "static" / "css" / "style.css").exists(), "style.css should exist"
    assert (ui_dir / "static" / "js" / "app.js").exists(), "app.js should exist"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
def test_ui_server_module_structure():
    """Test that the server module has expected structure (when Quart is available)."""
    from crisp_t.ui import server

    # Check that key components exist
    assert hasattr(server, "app"), "Quart app should exist"
    assert hasattr(server, "start_server"), "start_server function should exist"
    assert hasattr(server, "COPILOT_AVAILABLE"), "COPILOT_AVAILABLE flag should exist"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
def test_quart_routes_registered():
    """Test that Quart routes are properly registered (when Quart is available)."""
    from crisp_t.ui.server import app

    # Get list of registered routes
    routes = [rule.rule for rule in app.url_map.iter_rules()]

    # Check that expected routes exist
    assert "/" in routes, "Index route should exist"
    assert "/api/health" in routes, "Health check route should exist"
    assert any(
        "/api/session" in route for route in routes
    ), "Session routes should exist"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
def test_quart_app_configuration():
    """Test that Quart app is configured correctly (when Quart is available)."""
    from crisp_t.ui.server import app

    assert app is not None, "Quart app should be initialized"
    assert app.static_folder is not None and app.static_folder.endswith(
        "static"
    ), f"Static folder should end with 'static', got {app.static_folder}"
    assert app.template_folder is not None and app.template_folder.endswith(
        "templates"
    ), f"Template folder should end with 'templates', got {app.template_folder}"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
def test_copilot_import_graceful_failure():
    """Test that missing copilot SDK is handled gracefully (when Quart is available)."""
    from crisp_t.ui import server

    # Should not crash, just set COPILOT_AVAILABLE = False or True
    assert isinstance(server.COPILOT_AVAILABLE, bool)


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.skipif(
    not _module_available("copilot"),
    reason="Copilot SDK not installed",
)
def test_copilot_tool_definition():
    """Test that the CRISP command tool is properly defined (when both Quart and Copilot are available)."""
    from crisp_t.ui.server import execute_crisp_command

    # Accept either a callable or a Tool object with a callable handler
    try:
        from copilot import Tool

        is_tool = isinstance(execute_crisp_command, Tool)
    except ImportError:
        is_tool = False
    assert callable(execute_crisp_command) or (
        is_tool and callable(getattr(execute_crisp_command, "handler", None))
    ), "execute_crisp_command should be callable or a Tool with a callable handler"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.asyncio
async def test_health_endpoint_response():
    """Test the health check endpoint response format (when Quart is available)."""
    from crisp_t.ui.server import app

    client = app.test_client()
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = await response.get_json()

    assert "status" in data
    assert "copilot_available" in data
    assert "version" in data
    assert data["status"] == "ok"


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.asyncio
async def test_index_route_serves_html():
    """Test that the index route serves HTML (when Quart is available)."""
    from crisp_t.ui.server import app

    client = app.test_client()
    response = await client.get("/")

    assert response.status_code == 200
    assert response.content_type.startswith("text/html")


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.asyncio
async def test_get_messages_requires_session():
    """Test that getting messages requires a valid session (when Quart is available)."""
    from crisp_t.ui.server import app

    client = app.test_client()
    response = await client.get("/api/session/nonexistent/messages")

    assert response.status_code == 404
    data = await response.get_json()
    assert "error" in data


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.asyncio
async def test_models_endpoint_exists():
    """Test that the models endpoint returns a response (when Quart is available)."""
    from crisp_t.ui.server import app

    client = app.test_client()
    response = await client.get("/api/models")

    # Should return 200 if copilot available, or 500 if not
    assert response.status_code in [200, 500]
    data = await response.get_json()

    # Should have either models list or error
    assert "models" in data or "error" in data


@pytest.mark.skipif(
    not _module_available("quart"),
    reason="Quart not installed",
)
@pytest.mark.asyncio
async def test_index_html_has_dynamic_model_loading():
    """Test that the index.html template supports dynamic model loading (when Quart is available)."""
    from crisp_t.ui.server import app

    client = app.test_client()
    response = await client.get("/")

    assert response.status_code == 200
    html_content = await response.get_data(as_text=True)

    # Check that the dropdown exists
    assert 'id="modelSelect"' in html_content

    # Should not have hardcoded model options like before
    # (or should have only a loading placeholder)
    assert (
        "Loading models..." in html_content
        or '<option value="">Loading models...</option>' in html_content
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
