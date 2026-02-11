"""Tests for CRISP-T Web UI module."""

import sys
from pathlib import Path

import pytest


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
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_ui_server_module_structure():
    """Test that the server module has expected structure (when Flask is available)."""
    from crisp_t.ui import server
    
    # Check that key components exist
    assert hasattr(server, 'app'), "Flask app should exist"
    assert hasattr(server, 'start_server'), "start_server function should exist"
    assert hasattr(server, 'COPILOT_AVAILABLE'), "COPILOT_AVAILABLE flag should exist"


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_flask_routes_registered():
    """Test that Flask routes are properly registered (when Flask is available)."""
    from crisp_t.ui.server import app
    
    # Get list of registered routes
    routes = [rule.rule for rule in app.url_map.iter_rules()]
    
    # Check that expected routes exist
    assert "/" in routes, "Index route should exist"
    assert "/api/health" in routes, "Health check route should exist"
    assert any("/api/session" in route for route in routes), "Session routes should exist"


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_flask_app_configuration():
    """Test that Flask app is configured correctly (when Flask is available)."""
    from crisp_t.ui.server import app
    
    assert app is not None, "Flask app should be initialized"
    assert app.static_folder == "static", "Static folder should be configured"
    assert app.template_folder == "templates", "Template folder should be configured"


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_copilot_import_graceful_failure():
    """Test that missing copilot SDK is handled gracefully (when Flask is available)."""
    from crisp_t.ui import server
    
    # Should not crash, just set COPILOT_AVAILABLE = False or True
    assert isinstance(server.COPILOT_AVAILABLE, bool)


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
@pytest.mark.skipif(
    not pytest.importorskip("copilot", reason="copilot not installed"),
    reason="Copilot SDK not installed"
)
def test_copilot_tool_definition():
    """Test that the CRISP command tool is properly defined (when both Flask and Copilot are available)."""
    from crisp_t.ui.server import execute_crisp_command
    assert callable(execute_crisp_command), "execute_crisp_command should be callable"


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_health_endpoint_response():
    """Test the health check endpoint response format (when Flask is available)."""
    from crisp_t.ui.server import app
    
    client = app.test_client()
    response = client.get('/api/health')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert "status" in data
    assert "copilot_available" in data
    assert "version" in data
    assert data["status"] == "ok"


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_index_route_serves_html():
    """Test that the index route serves HTML (when Flask is available)."""
    from crisp_t.ui.server import app
    
    client = app.test_client()
    response = client.get('/')
    
    assert response.status_code == 200
    assert response.content_type.startswith('text/html')


@pytest.mark.skipif(
    not pytest.importorskip("flask", reason="flask not installed"),
    reason="Flask not installed"
)
def test_get_messages_requires_session():
    """Test that getting messages requires a valid session (when Flask is available)."""
    from crisp_t.ui.server import app
    
    client = app.test_client()
    response = client.get('/api/session/nonexistent/messages')
    
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

