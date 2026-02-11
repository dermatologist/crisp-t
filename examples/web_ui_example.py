"""
Example script demonstrating the CRISP-T Web UI usage.

This script shows how to programmatically interact with the Web UI server
if you need to extend or integrate it with other tools.
"""

from crisp_t.ui.server import start_server


def example_basic_usage():
    """Start the server with default settings."""
    print("Starting CRISP-T Web UI on default host and port...")
    start_server()


def example_custom_host_port():
    """Start the server on a custom host and port."""
    print("Starting CRISP-T Web UI on custom host and port...")
    start_server(host="0.0.0.0", port=8080, debug=False)


def example_debug_mode():
    """Start the server in debug mode for development."""
    print("Starting CRISP-T Web UI in debug mode...")
    start_server(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    # Choose one of the examples to run
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Example 2: Custom host and port (uncomment to use)
    # example_custom_host_port()
    
    # Example 3: Debug mode (uncomment to use)
    # example_debug_mode()
