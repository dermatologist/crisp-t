"""
Simple integration test to verify MCP server structure without needing mcp package.
"""

import sys
import importlib.util


def test_server_module_structure():
    """Test that the server module has the required structure."""
    # Import the server module
    spec = importlib.util.spec_from_file_location(
        "crisp_t.mcp.server",
        "/home/runner/work/crisp-t/crisp-t/src/crisp_t/mcp/server.py"
    )
    server_module = importlib.util.module_from_spec(spec)
    
    # Check that it would define the expected functions if mcp was available
    # This is a basic smoke test
    
    # Read the file and check for key patterns
    with open("/home/runner/work/crisp-t/crisp-t/src/crisp_t/mcp/server.py", "r") as f:
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
    assert '"topic_modeling"' in content
    assert '"regression_analysis"' in content
    assert '"add_relationship"' in content
    
    # Check for prompt names
    assert '"analysis_workflow"' in content
    assert '"triangulation_guide"' in content
    
    print("✓ Server module structure is valid")
    return True


def test_main_entry_point():
    """Test that __main__.py has the correct structure."""
    with open("/home/runner/work/crisp-t/crisp-t/src/crisp_t/mcp/__main__.py", "r") as f:
        content = f.read()
    
    assert "def run_server()" in content
    assert "asyncio.run(main())" in content
    
    print("✓ Main entry point is valid")
    return True


def test_init_module():
    """Test that __init__.py exports correctly."""
    with open("/home/runner/work/crisp-t/crisp-t/src/crisp_t/mcp/__init__.py", "r") as f:
        content = f.read()
    
    assert "from .server import app, main" in content
    assert '__all__ = ["app", "main"]' in content
    
    print("✓ Init module is valid")
    return True


if __name__ == "__main__":
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
