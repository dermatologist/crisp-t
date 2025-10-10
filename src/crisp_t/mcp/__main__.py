"""
Entry point for running the CRISP-T MCP server.
"""

import asyncio


def run_server():
    """Run the MCP server."""
    from .server import main
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
