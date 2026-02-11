# CRISP-T Web UI - Quick Start Guide

## Installation

### Step 1: Install CRISP-T with Copilot support

```bash
pip install crisp-t[copilot]
```

This installs:
- CRISP-T core package
- Quart web framework
- GitHub Copilot SDK
- All necessary dependencies

### Step 2: Install and configure GitHub Copilot CLI

Follow the official guide: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli

Quick setup:
```bash
# Install GitHub CLI if not already installed
# See: https://cli.github.com/

# Login to GitHub
gh auth login

# Verify Copilot CLI
copilot --version
```

## Quick Start

### Launch the Web UI

```bash
# Start on default port (5000)
crisp-ui

# Or specify a custom port
crisp-ui --port 8080
```

Open your browser to: `http://127.0.0.1:5000`

### First Session

1. **Configure Settings** (left panel):
   - Select AI model (e.g., GPT-5)
   - Set data path (e.g., `./data`)
   - Optionally configure advanced settings

2. **Start Session**:
   - Click "Start Session" button
   - Wait for green status indicator

3. **Start Chatting**:
   - Type your request in the chat box
   - Press Enter or click Send
   - Watch the AI respond in real-time

## Example Workflow

### Basic Analysis

```
You: Import data from ./interviews folder

AI: I'll import your data using CRISP-T...
[Runs: crisp --source ./interviews --out corpus]

You: What's in the corpus?

AI: Let me check...
[Runs: crispt --inp corpus --print]

You: Run topic modeling with 5 topics

AI: I'll perform topic modeling...
[Runs: crisp --inp corpus --topics --num 5 --assign --out corpus]

You: Create a word cloud

AI: Creating visualization...
[Runs: crispviz --inp corpus --wordcloud --out viz]
```

### Advanced Analysis

```
You: Import CSV data from ./survey with "feedback" as the text column,
     run sentiment analysis, and link it to the satisfaction_score variable

AI: I'll perform a comprehensive mixed-methods analysis...
[Runs multiple commands in sequence]
```

## Using Local Models (Ollama)

### Setup

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull deepseek-coder-v2:16b
   ```

### Configure in UI

1. Check "Use Custom Provider"
2. Set:
   - Provider Type: OpenAI
   - Base URL: `http://localhost:11434/v1`
   - API Key: (leave empty)
3. Model: Enter `deepseek-coder-v2:16b`
4. Start Session

## Troubleshooting

### "Copilot SDK not available"

**Solution**: Install with copilot support
```bash
pip install crisp-t[copilot]
```

### "Failed to create session"

**Cause**: Copilot CLI not installed or not logged in

**Solution**:
```bash
# Check if installed
copilot --version

# If not, install GitHub CLI and Copilot CLI
# Then login
gh auth login
```

### Port Already in Use

**Solution**: Use different port
```bash
crisp-ui --port 8080
```

### Commands Not Found

**Cause**: CRISP-T CLI tools not in PATH

**Solution**: Reinstall CRISP-T
```bash
pip install crisp-t[ml]
```

## Tips

1. **Start Simple**: Begin with basic commands to familiarize yourself
2. **Be Specific**: Provide clear instructions to the AI
3. **Use Examples**: The welcome screen shows example prompts
4. **Check Results**: Review command outputs in the chat
5. **Save Work**: Use `--out` flags to save intermediate results

## Next Steps

- Read the full documentation: `docs/ui.md`
- Explore CRISP-T CLI: `docs/cheatsheet.md`
- Try example workflows: `docs/DEMO.md`
- Learn about the crisp-cli skill: `.agents/skills/crisp-cli/SKILL.md`

## Getting Help

- GitHub Issues: https://github.com/dermatologist/crisp-t/issues
- Documentation: https://dermatologist.github.io/crisp-t/
- CRISP-T Wiki: https://github.com/dermatologist/crisp-t/wiki

---

**Happy analyzing! 🔍**
