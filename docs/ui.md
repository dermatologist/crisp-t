# CRISP-T Web UI Documentation

## Overview

The CRISP-T Web UI provides a browser-based interface for interacting with CRISP-T's qualitative research tools using the GitHub Copilot SDK. It allows researchers to perform mixed-methods analysis through a conversational AI interface without needing to use the command line directly.

## Features

- **AI-Powered Chat Interface**: Interact with CRISP-T through natural language conversations
- **Multiple Model Support**: Choose from various AI models (GPT-5, GPT-4.1, Claude Sonnet 4.5, Claude Opus 4)
- **Custom Provider Support**: Use local models (Ollama) or custom OpenAI-compatible APIs
- **Real-time Streaming**: Get responses as they're generated
- **CRISP-T CLI Integration**: Full access to all CRISP-T commands (crisp, crispt, crispviz)
- **Easy Configuration**: Simple web interface for setting up your research environment

## Installation

### Prerequisites

1. **Python 3.10+** installed
2. **CRISP-T** installed with Copilot support:
   ```bash
   pip install crisp-t[copilot]
   ```
3. **GitHub Copilot CLI** installed and configured:
   - Follow the [Copilot CLI installation guide](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli)
   - Login using: `gh auth login` or `copilot auth login`

### Verify Installation

Check that everything is installed correctly:

```bash
# Verify CRISP-T CLI tools
crisp --help
crispt --help
crispviz --help

# Verify Copilot CLI
copilot --version

# Verify Web UI command
crisp-ui --help
```

## Starting the Web UI

### Basic Usage

Start the server on the default host and port (127.0.0.1:5000):

```bash
crisp-ui
```

Then open your browser and navigate to: `http://127.0.0.1:5000`

### Advanced Usage

```bash
# Start on a custom port
crisp-ui --port 8080

# Allow external connections (use 0.0.0.0 to bind to all interfaces)
crisp-ui --host 0.0.0.0 --port 8080

# Run in debug mode (for development)
crisp-ui --debug
```

## User Interface Guide

### Configuration Panel

The left panel contains configuration options:

#### 1. Model Selection
Choose the AI model to use for your research assistant:
- **GPT-5**: Latest OpenAI model with advanced reasoning
- **GPT-4.1**: Improved GPT-4 with better context handling
- **Claude Sonnet 4.5**: Anthropic's balanced model
- **Claude Opus 4**: Anthropic's most capable model

#### 2. Data Source
Specify the path to your research data:
- Enter the path to a folder containing your data files
- Supported formats: `.txt`, `.pdf`, `.csv`
- Example: `./data` or `/home/user/research_data`

#### 3. Advanced Settings

##### Custom Provider
Check this box to use a custom AI provider:
- **Provider Type**: OpenAI, Azure OpenAI, or Anthropic
- **Base URL**: API endpoint (e.g., `http://localhost:11434/v1` for Ollama)
- **API Key**: Your API key (optional for local providers like Ollama)

**Example: Using Ollama locally**
```
Provider Type: OpenAI
Base URL: http://localhost:11434/v1
API Key: (leave empty)
```

##### GitHub Token
Optionally provide a GitHub token for authentication:
- Required if not using the logged-in Copilot CLI session
- Generate a token at: https://github.com/settings/tokens

### Chat Interface

The right panel contains the chat interface:

#### Starting a Session
1. Configure your settings in the left panel
2. Click **"Start Session"**
3. Wait for the connection indicator to turn green
4. Start chatting with the AI assistant

#### Sending Messages
1. Type your message in the text box at the bottom
2. Press **Enter** or click **"Send"**
3. Wait for the AI response

#### Example Conversations

**Importing Data:**
```
You: Import data from ./data folder and show me what's in the corpus
AI: I'll import your data using CRISP-T...
[Executes: crisp --source ./data --out corpus]
```

**Topic Modeling:**
```
You: Perform topic modeling with 5 topics on the imported data
AI: I'll run topic modeling analysis...
[Executes: crisp --inp corpus --topics --num 5 --assign --out corpus]
```

**Visualization:**
```
You: Generate a word cloud of the most frequent terms
AI: I'll create a word cloud visualization...
[Executes: crispviz --inp corpus --wordcloud --out visualizations]
```

**Complex Analysis:**
```
You: Import the CSV data with "comments" as the text column, 
     perform sentiment analysis, and create a correlation heatmap
AI: I'll perform a comprehensive analysis...
[Executes multiple commands in sequence]
```

## Common Workflows

### 1. Basic Qualitative Analysis

```
1. You: "Import data from ./interviews folder"
2. You: "Generate a coding dictionary"
3. You: "Perform topic modeling with 5 topics"
4. You: "Show me the main themes"
5. You: "Create a word cloud visualization"
```

### 2. Mixed-Methods Triangulation

```
1. You: "Import CSV data from ./survey with 'open_response' as text column"
2. You: "Run topic modeling and sentiment analysis"
3. You: "Link the topic findings to the satisfaction_score variable"
4. You: "Run a regression analysis to see which topics predict satisfaction"
5. You: "Create a visualization showing the relationships"
```

### 3. Temporal Analysis

```
1. You: "Import time-stamped data from ./longitudinal"
2. You: "Link documents to rows by nearest timestamp"
3. You: "Filter data to the first quarter of 2025"
4. You: "Analyze sentiment changes over time"
5. You: "Create a temporal summary by week"
```

## Authentication Methods

The Web UI supports multiple authentication methods for GitHub Copilot:

### 1. Logged-in User (Default)
If you've logged in using `gh auth login` or `copilot auth login`, the Web UI will use your stored credentials automatically.

### 2. GitHub Token
Provide a personal access token in the Advanced Settings:
1. Generate a token at https://github.com/settings/tokens
2. Paste it in the "GitHub Token" field
3. Start your session

### 3. Custom Provider (BYOK)
Use your own API keys without GitHub authentication:
1. Check "Use Custom Provider"
2. Select provider type
3. Enter base URL and API key
4. No GitHub token needed

## Troubleshooting

### "Copilot SDK not available" Error

**Problem**: The github-copilot-sdk package is not installed.

**Solution**: Install CRISP-T with Copilot support:
```bash
pip install crisp-t[copilot]
```

### "Command not found" Errors

**Problem**: CRISP-T CLI tools are not in the PATH.

**Solution**: Ensure CRISP-T is installed and accessible:
```bash
which crisp
# If not found, reinstall:
pip install crisp-t[ml]
```

### "Failed to create session" Error

**Problem**: The Copilot CLI is not installed or not logged in.

**Solution**:
1. Install Copilot CLI: Follow [installation guide](https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli)
2. Login: `gh auth login` or provide a GitHub token in the UI

### Session Not Responding

**Problem**: The session appears stuck or not responding.

**Solution**:
1. Wait a few moments - some operations take time
2. Check the browser console for errors (F12)
3. Stop and restart the session
4. Restart the server if issues persist

### Port Already in Use

**Problem**: Cannot start server on default port 5000.

**Solution**: Use a different port:
```bash
crisp-ui --port 8080
```

## Advanced Topics

### Using Ollama for Local Models

1. Install and start Ollama:
   ```bash
   # Install Ollama (see https://ollama.ai)
   # Pull a model
   ollama pull deepseek-coder-v2:16b
   ```

2. Configure in the Web UI:
   - Check "Use Custom Provider"
   - Provider Type: OpenAI
   - Base URL: `http://localhost:11434/v1`
   - API Key: (leave empty)
   - Model: `deepseek-coder-v2:16b`

3. Start your session and chat!

### Integrating with Azure OpenAI

1. Configure in the Web UI:
   - Check "Use Custom Provider"
   - Provider Type: Azure
   - Base URL: `https://your-resource.openai.azure.com`
   - API Key: Your Azure OpenAI key
   - Model: Your deployment name (e.g., `gpt-4`)

2. Start your session

### Running Behind a Reverse Proxy

If you're running the Web UI behind a reverse proxy (like nginx):

1. Start the server on localhost:
   ```bash
   crisp-ui --host 127.0.0.1 --port 5000
   ```

2. Configure your proxy to forward requests to `http://127.0.0.1:5000`

3. Ensure WebSocket support is enabled in your proxy configuration

## Architecture

The CRISP-T Web UI consists of three main components:

### 1. Flask Web Server (`server.py`)
- Serves the HTML interface
- Provides REST API endpoints for session management
- Manages Copilot SDK client instances
- Handles async operations in a synchronous Flask context

### 2. Frontend Interface (`templates/index.html`, `static/*`)
- Configuration panel for settings
- Chat interface for conversations
- Real-time message updates via polling
- Responsive design for various screen sizes

### 3. Copilot SDK Integration
- Creates sessions with custom tools
- Provides `execute_crisp_command` tool to the AI
- Handles streaming responses
- Manages session lifecycle

## API Reference

The Web UI exposes the following REST API endpoints:

### Health Check
```
GET /api/health
Response: {"status": "ok", "copilot_available": true, "version": "1.0.0"}
```

### List Models
```
GET /api/models
Response: {"models": ["gpt-5", "gpt-4.1", "claude-sonnet-4.5", ...]}
```

### Create Session
```
POST /api/session/create
Body: {
  "session_id": "session-123",
  "model": "gpt-5",
  "config": {
    "data_path": "./data",
    "github_token": "ghp_...",  // optional
    "use_custom_provider": true,  // optional
    "provider_type": "openai",
    "provider_base_url": "http://localhost:11434/v1",
    "provider_api_key": "..."  // optional
  }
}
Response: {"status": "ok", "session_id": "session-123", "model": "gpt-5"}
```

### Send Message
```
POST /api/session/{session_id}/send
Body: {"prompt": "Import data from ./data"}
Response: {"status": "ok"}
```

### Get Messages
```
GET /api/session/{session_id}/messages
Response: {
  "messages": [
    {"role": "user", "content": "Hello", "timestamp": "..."},
    {"role": "assistant", "content": "Hi there!", "timestamp": "..."}
  ]
}
```

### Destroy Session
```
POST /api/session/{session_id}/destroy
Response: {"status": "ok"}
```

## Security Considerations

1. **Local Deployment**: By default, the server binds to `127.0.0.1` (localhost only)
2. **Token Storage**: GitHub tokens are not stored on the server - they're kept in memory only
3. **Data Access**: The AI agent can execute CRISP-T commands, which can read/write files
4. **Network Access**: When using external models, your data may be sent to third-party APIs

**Best Practices:**
- Run on localhost for personal use
- Use HTTPS when deploying externally
- Be cautious about data privacy when using cloud models
- Review the CRISP CLI skill to understand what commands can be executed

## Performance Tips

1. **Large Datasets**: Use `--num` and `--rec` flags to limit data during testing
2. **Model Selection**: Faster models like GPT-4.1 for quick iterations
3. **Streaming**: Enabled by default for real-time feedback
4. **Session Management**: Stop sessions when not in use to free resources

## Contributing

To contribute to the Web UI:

1. The UI code is in `src/crisp_t/ui/`
2. Follow the existing code structure
3. Test with multiple models and configurations
4. Update documentation for new features
5. Ensure error handling is comprehensive

## Support

For issues and questions:
- GitHub Issues: https://github.com/dermatologist/crisp-t/issues
- Documentation: https://dermatologist.github.io/crisp-t/
- CRISP-T CLI Skill: `.agents/skills/crisp-cli/`

## License

CRISP-T Web UI is part of the CRISP-T project and is licensed under GPL-3.0.
