"""CLI command to start CRISP-T Web UI."""

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", default=5000, help="Port to bind to (default: 5000)")
@click.option("--debug", is_flag=True, help="Run in debug mode")
def main(host: str, port: int, debug: bool):
    """Start the CRISP-T Web UI server.

    This command starts a web server that provides a browser-based interface
    for interacting with CRISP-T using the GitHub Copilot SDK.

    Example usage:
        crisp-ui                    # Start on default host:port (127.0.0.1:5000)
        crisp-ui --host 0.0.0.0     # Allow external connections
        crisp-ui --port 8080        # Use custom port
        crisp-ui --debug            # Run in debug mode
    """
    from crisp_t.ui.server import start_server

    start_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
