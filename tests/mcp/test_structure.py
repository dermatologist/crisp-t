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
    import platform

    system = platform.system()
    if system == "Windows":
        return True
    elif system == "Darwin":
        return True
    else:
        prefix = "/home/runner/work/crisp-t/crisp-t"
    server_path = f"{prefix}/src/crisp_t/mcp/server.py"
    # Import the server module
    spec = importlib.util.spec_from_file_location(
        "crisp_t.mcp.server",
        server_path,
    )
    server_module = importlib.util.module_from_spec(spec) # type: ignore

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
    import platform

    system = platform.system()
    if system == "Windows":
        return True
    elif system == "Darwin":
        return True
    else:
        prefix = "/home/runner/work/crisp-t/crisp-t"
    main_path = f"{prefix}/src/crisp_t/mcp/__main__.py"
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
    import platform

    system = platform.system()
    if system == "Windows":
        return True
    elif system == "Darwin":
        return True
    else:
        prefix = "/home/runner/work/crisp-t/crisp-t"
    init_path = f"{prefix}/src/crisp_t/mcp/__init__.py"
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
