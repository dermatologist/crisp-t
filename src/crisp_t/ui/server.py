"""Quart web server for CRISP-T UI with Copilot SDK integration.

This module provides a web-based interface for CRISP-T qualitative research tools,
powered by the GitHub Copilot SDK. It allows researchers to interact with CRISP-T
through natural language conversations with AI assistants.

Key Features:
- Async REST API for session management (using Quart/ASGI)
- Real-time chat interface with streaming responses
- Integration with CRISP-T CLI tools (crisp, crispt, crispviz)
- Support for multiple AI models (GPT-5, Claude, etc.)
- Custom provider support (Ollama, Azure OpenAI, etc.)

Architecture:
- Quart ASGI web server handles HTTP requests asynchronously
- Copilot SDK manages AI sessions with custom tools
- execute_crisp_command tool allows AI to run CRISP-T commands
- Frontend polls for message updates in real-time

Dependencies:
- quart: Async web framework (ASGI)
- quart-cors: Cross-origin resource sharing
- github-copilot-sdk: AI integration (optional)
- pydantic: Type validation (optional, used with copilot)

Note: Migrated from Flask to Quart to resolve event loop issues with async operations.
"""

import asyncio
import subprocess
from typing import Dict

from quart import Quart, jsonify, render_template, request
from quart_cors import cors

# Check if copilot SDK is available
try:
    from copilot import CopilotClient, define_tool
    from pydantic import BaseModel, Field

    COPILOT_AVAILABLE = True
except ImportError:
    COPILOT_AVAILABLE = False

app = Quart(__name__, static_folder="static", template_folder="templates")
app = cors(app)

# Global state for managing copilot clients and sessions
clients: Dict[str, dict] = {}
clients_lock = asyncio.Lock()


# Define tool and model classes only if Copilot is available
if COPILOT_AVAILABLE:

    class CrispCommandParams(BaseModel):
        """Parameters for CRISP command execution."""

        command: str = Field(description="The CRISP CLI command to execute (crisp, crispt, or crispviz)")
        args: str = Field(description="Command line arguments for the CRISP command")

    @define_tool(description="Execute CRISP-T CLI commands for qualitative research analysis")
    async def execute_crisp_command(params: CrispCommandParams) -> str:
        """
        Execute CRISP-T CLI commands.

        This tool allows the agent to run CRISP-T commands for qualitative and mixed-methods research.
        Available commands: crisp, crispt, crispviz
        """
        valid_commands = ["crisp", "crispt", "crispviz"]
        if params.command not in valid_commands:
            return f"Error: Invalid command '{params.command}'. Must be one of: {', '.join(valid_commands)}"

        try:
            # Build the full command
            full_command = [params.command] + params.args.split()

            # Execute the command
            result = subprocess.run(
                full_command, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            # Combine stdout and stderr for complete output
            output = result.stdout
            if result.stderr:
                output += f"\n\nErrors/Warnings:\n{result.stderr}"

            if result.returncode != 0:
                return f"Command failed with exit code {result.returncode}:\n{output}"

            return output or "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (exceeded 5 minutes)"
        except FileNotFoundError:
            return f"Error: Command '{params.command}' not found. Is CRISP-T installed?"
        except Exception as e:
            return f"Error executing command: {str(e)}"


async def create_copilot_session(session_id: str, model: str, config: dict) -> dict:
    """Create a new Copilot client and session."""
    if not COPILOT_AVAILABLE:
        raise RuntimeError("Copilot SDK is not installed. Install with: pip install crisp-t[copilot]")

    # Create client
    client_config = {"log_level": "info", "auto_start": True, "auto_restart": True}

    # Add GitHub token if provided
    if config.get("github_token"):
        client_config["github_token"] = config["github_token"]

    client = CopilotClient(client_config)
    await client.start()

    # Prepare session configuration
    session_config = {"model": model, "tools": [execute_crisp_command], "streaming": True}

    # Add custom provider if specified
    if config.get("use_custom_provider"):
        provider_config = {"type": config.get("provider_type", "openai"), "base_url": config.get("provider_base_url")}

        if config.get("provider_api_key"):
            provider_config["api_key"] = config["provider_api_key"]

        session_config["provider"] = provider_config

    # Add system message emphasizing CRISP-T expertise
    system_message = {
        "role": "system",
        "content": """You are an expert CRISP-T qualitative research assistant. You help researchers perform 
        mixed-methods analysis using CRISP-T CLI tools (crisp, crispt, crispviz).
        
        You have access to the execute_crisp_command tool to run CRISP-T commands. Use this tool to:
        - Import and analyze qualitative and quantitative data
        - Generate coding dictionaries and perform topic modeling
        - Create visualizations
        - Link textual findings to numeric outcomes
        - Perform semantic search and temporal analysis
        
        Always explain what you're doing and interpret the results for the user in the context of their research.""",
    }
    session_config["system_message"] = system_message

    # Note: Temperature and max_tokens are typically controlled at the provider/model level
    # These settings from the config are captured but not currently used
    # Future enhancement: Pass these to the provider configuration if supported

    # Create session
    session = await client.create_session(session_config)

    # Store message history
    messages = []

    # Event handler for session events
    def on_event(event):
        event_type = event.type.value
        print(f"[DEBUG] Event received: {event_type}")
        if event_type == "assistant.message":
            content = event.data.content
            print(f"[DEBUG] assistant.message: content_length={len(content)}")
            messages.append({"role": "assistant", "content": content, "timestamp": event.data.created_at})
        elif event_type == "user.message":
            content = event.data.content
            print(f"[DEBUG] user.message: content={content[:50]}...")
            messages.append({"role": "user", "content": content, "timestamp": event.data.created_at})
        elif event_type == "assistant.message_delta":
            # Handle streaming chunks
            delta = event.data.delta_content or ""
            print(f"[DEBUG] assistant.message_delta: delta_length={len(delta)}")
            if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("complete"):
                messages.append({"role": "assistant", "content": "", "complete": False})
            messages[-1]["content"] += delta
        elif event_type == "session.idle":
            # Mark last message as complete
            print(f"[DEBUG] session.idle: messages_count={len(messages)}")
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["complete"] = True
                print(f"[DEBUG] Marked message as complete, content_length={len(messages[-1]['content'])}")

    session.on(on_event)

    return {"client": client, "session": session, "messages": messages, "model": model, "config": config}


@app.route("/")
async def index():
    """Serve the main UI page."""
    return await render_template("index.html")


@app.route("/api/health", methods=["GET"])
async def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "copilot_available": COPILOT_AVAILABLE,
            "version": "1.0.0",
        }
    )


@app.route("/api/models", methods=["GET"])
async def list_models():
    """List available models from Copilot."""
    if not COPILOT_AVAILABLE:
        return jsonify({"error": "Copilot SDK not available"}), 500

    try:
        # Create a temporary client to get model list
        client = CopilotClient()
        await client.start()
        models = await client.list_models()
        await client.stop()

        return jsonify({"models": [model["id"] for model in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/create", methods=["POST"])
async def create_session():
    """Create a new Copilot session."""
    if not COPILOT_AVAILABLE:
        return jsonify({"error": "Copilot SDK not available. Install with: pip install crisp-t[copilot]"}), 500

    data = await request.json
    session_id = data.get("session_id")
    model = data.get("model", "gpt-5")
    config = data.get("config", {})

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        # Create the session asynchronously
        session_data = await create_copilot_session(session_id, model, config)

        # Store in global state
        async with clients_lock:
            clients[session_id] = session_data

        return jsonify({"status": "ok", "session_id": session_id, "model": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/send", methods=["POST"])
async def send_message(session_id: str):
    """Send a message to a Copilot session."""
    if not COPILOT_AVAILABLE:
        return jsonify({"error": "Copilot SDK not available"}), 500

    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients[session_id]

    data = await request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        session = session_data["session"]
        await session.send({"prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


@app.route("/api/session/<session_id>/messages", methods=["GET"])
async def get_messages(session_id: str):
    """Get message history for a session."""
    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients[session_id]
    
    messages = session_data["messages"]
    print(f"[DEBUG] get_messages: session={session_id}, count={len(messages)}")
    if messages:
        print(f"[DEBUG] Latest message: role={messages[-1].get('role')}, content_length={len(messages[-1].get('content', ''))}, complete={messages[-1].get('complete')}")
    
    return jsonify({"messages": messages})


@app.route("/api/session/<session_id>/destroy", methods=["POST"])
async def destroy_session(session_id: str):
    """Destroy a Copilot session."""
    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients.pop(session_id)

    try:
        await session_data["session"].destroy()
        await session_data["client"].stop()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"status": "ok"})


def start_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Start the Quart web server.
    
    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 5000)
        debug: Run in debug mode (WARNING: Only use in development, not production)
    """
    if not COPILOT_AVAILABLE:
        print("WARNING: Copilot SDK is not installed. Install with: pip install crisp-t[copilot]")
        print("The server will start but Copilot features will not be available.")

    if debug:
        print("\n⚠️  WARNING: Debug mode is enabled. This should only be used in development!")
        print("    Debug mode allows arbitrary code execution and should NEVER be used in production.")

    print(f"\n🚀 CRISP-T Web UI starting on http://{host}:{port}")
    print(f"📖 Open your browser and navigate to: http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")

    # Run the Quart app using the built-in ASGI server
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    start_server()
