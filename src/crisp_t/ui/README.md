# CRISP-T Web UI

This directory contains the web-based user interface for CRISP-T, powered by the GitHub Copilot SDK.

## Directory Structure

```
ui/
├── __init__.py           # Package initialization
├── cli.py                # CLI command for starting the server
├── server.py             # Flask web server and API endpoints
├── templates/            # HTML templates
│   └── index.html        # Main UI page
└── static/               # Static assets
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── app.js        # Frontend JavaScript
```

## Features

- **Flask Web Server**: Provides REST API and serves the UI
- **Copilot SDK Integration**: Creates sessions with custom tools
- **CRISP-T CLI Tool**: Allows AI to execute CRISP commands
- **Real-time Chat**: Streaming responses from AI models
- **Multi-Model Support**: GPT-5, GPT-4.1, Claude, and custom providers

## Development

### Running the Server

```bash
# From the repository root
python -m crisp_t.ui.cli

# Or after installation
crisp-ui
```

### Adding New Features

1. **New API Endpoints**: Add to `server.py`
2. **UI Changes**: Modify `templates/index.html` and `static/`
3. **Custom Tools**: Add new tools in `server.py` using `@define_tool`

### Testing

Test the server manually:

```bash
# Start the server
crisp-ui --port 5000

# Open browser to http://127.0.0.1:5000
```

## Architecture

### Backend (server.py)

The Flask server provides these endpoints:

- `GET /` - Serves the main UI
- `GET /api/health` - Health check
- `GET /api/models` - List available models
- `POST /api/session/create` - Create new session
- `POST /api/session/<id>/send` - Send message
- `GET /api/session/<id>/messages` - Get message history
- `POST /api/session/<id>/destroy` - Destroy session

### Frontend (app.js)

The JavaScript client:
- Manages session lifecycle
- Polls for new messages (500ms interval)
- Handles user input and displays responses
- Supports markdown-like formatting

### Copilot Integration

The `execute_crisp_command` tool allows the AI to run:
- `crisp` - Main analysis engine
- `crispt` - Corpus management
- `crispviz` - Visualization generation

## Dependencies

- `flask` - Web server
- `flask-cors` - CORS support
- `github-copilot-sdk` - Copilot integration
- `pydantic` - Type validation

Install with:
```bash
pip install crisp-t[copilot]
```

## Configuration

The UI supports:

- **Model Selection**: Choose from available models
- **Data Path**: Specify data directory
- **Custom Providers**: Use Ollama or custom OpenAI-compatible APIs
- **GitHub Token**: Optional authentication

## Security Notes

- Server binds to `127.0.0.1` by default (localhost only)
- GitHub tokens are stored in memory only (not persisted)
- The AI can execute file system operations via CRISP commands
- Use caution when exposing externally

## Contributing

When adding features:

1. Follow existing code patterns
2. Add error handling for all operations
3. Update documentation in `docs/ui.md`
4. Test with multiple models and configurations
5. Ensure the UI remains simple and intuitive

## License

Part of CRISP-T - GPL-3.0 License
