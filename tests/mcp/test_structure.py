"""
Simple integration test to verify MCP server structure without needing mcp package.
"""

import importlib.util
import os
import sys


def test_server_module_structure():
    # Only run in GitHub Actions due to hardcoded paths
    if not os.environ.get("GITHUB_ACTIONS"):
        print(
            "Skipping test_server_module_structure: Only runs in GitHub Actions due to hardcoded paths."
        )
        return True
    """Test that the server module has the required structure."""
    # Determine the correct home directory prefix based on OS
    home = os.environ.get("HOME")
    # Fallback for Windows GitHub Actions runner
    if not home and os.name == "nt":
        home = os.environ.get("USERPROFILE", "C:/Users/runneradmin")
    if not home:
        raise RuntimeError(
            "Could not determine home directory for test path construction."
        )
    server_path = os.path.join(
        home, "work", "crisp-t", "crisp-t", "src", "crisp_t", "mcp", "server.py"
    )
    if not os.path.exists(server_path):
        raise FileNotFoundError(f"Server module not found at {server_path}")
    # Import the server module
    spec = importlib.util.spec_from_file_location(
        "crisp_t.mcp.server",
        server_path,
    )
    if spec is None:
        raise ImportError(f"Could not create module spec for {server_path}")
    server_module = importlib.util.module_from_spec(spec)

    # Check that it would define the expected functions if mcp was available
    # This is a basic smoke test

    # Read the file and check for key patterns
    with open(server_path, "r") as f:
        content = f.read()

    # Check for required decorators and functions
    assert "@app.list_tools()" in content
    assert "@app.call_tool()" in content
    assert "@app.list_resources()" in content
    assert "@app.read_resource()" in content
    assert "@app.list_prompts()" in content
    assert "@app.get_prompt()" in content

    # Check for key tool names
    assert '"load_corpus"' in content
    assert '"save_corpus"' in content
    assert '"regression_analysis"' in content
    assert '"add_relationship"' in content

    # Check for prompt names
    assert '"analysis_workflow"' in content
    assert '"triangulation_guide"' in content

    print("✓ Server module structure is valid")
    return True


def test_main_entry_point():
    # Only run in GitHub Actions due to hardcoded paths
    if not os.environ.get("GITHUB_ACTIONS"):
        print(
            "Skipping test_main_entry_point: Only runs in GitHub Actions due to hardcoded paths."
        )
        return True
    """Test that __main__.py has the correct structure."""
    home = os.environ.get("HOME")
    if not home and os.name == "nt":
        home = os.environ.get("USERPROFILE", "C:/Users/runneradmin")
    if not home:
        raise RuntimeError(
            "Could not determine home directory for test path construction."
        )
    main_path = os.path.join(
        home, "work", "crisp-t", "crisp-t", "src", "crisp_t", "mcp", "__main__.py"
    )
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Main entry point not found at {main_path}")
    with open(main_path, "r") as f:
        content = f.read()

    assert "def run_server()" in content
    assert "asyncio.run(main())" in content

    print("✓ Main entry point is valid")
    return True


def test_init_module():
    # Only run in GitHub Actions due to hardcoded paths
    if not os.environ.get("GITHUB_ACTIONS"):
        print(
            "Skipping test_init_module: Only runs in GitHub Actions due to hardcoded paths."
        )
        return True
    """Test that __init__.py exports correctly."""
    home = os.environ.get("HOME")
    if not home and os.name == "nt":
        home = os.environ.get("USERPROFILE", "C:/Users/runneradmin")
    if not home:
        raise RuntimeError(
            "Could not determine home directory for test path construction."
        )
    init_path = os.path.join(
        home, "work", "crisp-t", "crisp-t", "src", "crisp_t", "mcp", "__init__.py"
    )
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init module not found at {init_path}")
    with open(init_path, "r") as f:
        content = f.read()

    assert "from .server import app, main" in content
    assert '__all__ = ["app", "main"]' in content

    print("✓ Init module is valid")
    return True


if __name__ == "__main__":
    if not os.environ.get("GITHUB_ACTIONS"):
        print("Skipping all tests: Only runs in GitHub Actions due to hardcoded paths.")
        sys.exit(0)
    try:
        test_server_module_structure()
        test_main_entry_point()
        test_init_module()
        print("\n✅ All structural tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
