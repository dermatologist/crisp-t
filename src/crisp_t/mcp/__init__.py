"""
MCP Server module for CRISP-T

Provides Model Context Protocol server for interacting with CRISP-T
via tools, resources, and prompts.
"""

from .server import app, main

__all__ = ["app", "main"]
