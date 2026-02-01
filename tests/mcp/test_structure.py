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

    # Check for modular tool imports
    assert "from .tools import get_all_tools, handle_tool_call" in content
    
    # Check that server.py uses the modular approach
    assert "get_all_tools()" in content
    assert "handle_tool_call(" in content

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


def test_tool_modules_exist():
    # Only run in GitHub Actions due to hardcoded paths
    if not os.environ.get("GITHUB_ACTIONS"):
        print(
            "Skipping test_tool_modules_exist: Only runs in GitHub Actions due to hardcoded paths."
        )
        return True
    """Test that all tool modules exist and have correct structure."""
    import platform

    system = platform.system()
    if system == "Windows":
        return True
    elif system == "Darwin":
        return True
    else:
        prefix = "/home/runner/work/crisp-t/crisp-t"
    
    tools_dir = f"{prefix}/src/crisp_t/mcp/tools"
    
    # Check that tools directory exists
    assert os.path.exists(tools_dir), f"Tools directory not found: {tools_dir}"
    
    # Check for expected tool modules
    expected_modules = [
        "corpus_management.py",
        "nlp_analysis.py",
        "corpus_filtering.py",
        "dataframe_operations.py",
        "column_operations.py",
        "ml_tools.py",
        "semantic_search.py",
        "topological_analysis.py",
        "temporal_analysis.py",
        "embedding_linking.py",
        "misc_tools.py",
    ]
    
    for module in expected_modules:
        module_path = os.path.join(tools_dir, module)
        assert os.path.exists(module_path), f"Tool module not found: {module_path}"
        
        # Check that each module has required functions
        with open(module_path, "r") as f:
            content = f.read()
            assert "def get_" in content, f"Module {module} missing get_ function"
            assert "def handle_" in content, f"Module {module} missing handle_ function"
    
    # Check tools/__init__.py
    init_path = os.path.join(tools_dir, "__init__.py")
    assert os.path.exists(init_path), f"Tools __init__.py not found"
    
    with open(init_path, "r") as f:
        content = f.read()
        assert "def get_all_tools()" in content
        assert "def handle_tool_call(" in content
    
    print("✓ All tool modules exist and have correct structure")
    return True


if __name__ == "__main__":
    if not os.environ.get("GITHUB_ACTIONS"):
        print("Skipping all tests: Only runs in GitHub Actions due to hardcoded paths.")
        sys.exit(0)
    try:
        test_server_module_structure()
        test_main_entry_point()
        test_init_module()
        test_tool_modules_exist()
        print("\n✅ All structural tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
