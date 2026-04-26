# CRISP-T Chat Bot Integration — Microsoft Teams & Slack

This guide explains how to set up and use the CRISP-T chat bot, which lets researchers interact with CRISP-T from **Microsoft Teams** and **Slack** using natural-language commands.

## Overview

The bot acts as a bridge between Teams/Slack and the CRISP-T web UI server (`crisp-ui`). It uses the [Chat SDK](https://github.com/nicholasgasior/chat-sdk) (`@chat-adapter/teams` + `@chat-adapter/slack`) to handle messages from both platforms and forwards commands to the `crisp-ui` REST API.

```
Teams User ──► Teams Chat  ──┐
                               ├──► CRISP-T Bot (Node.js) ──► crisp-ui (Python) ──► CRISP-T Engine
Slack User  ──► Slack Channel ─┘
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | ≥ 18 |
| npm | ≥ 9 |
| Python | ≥ 3.10 |
| `crisp-t[copilot]` | latest |
| Microsoft Azure account | *(for Teams)* |
| Slack workspace with admin access | *(for Slack)* |

---

## Installation

### 1. Install the CRISP-T bot dependencies

```bash
cd integration
npm install
npm run build
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TEAMS_APP_ID` | Microsoft App ID from Azure Bot registration | *(required for Teams)* |
| `TEAMS_APP_PASSWORD` | Client secret from Azure App registration | *(required for Teams)* |
| `SLACK_BOT_TOKEN` | Slack Bot User OAuth Token (`xoxb-...`) | *(required for Slack)* |
| `SLACK_SIGNING_SECRET` | Slack App signing secret | *(required for Slack)* |
| `TEAMS_APP_TENANT_ID` | Tenant ID (leave blank for multi-tenant) | *(optional)* |
| `CRISP_UI_URL` | URL where `crisp-ui` is running | `http://127.0.0.1:5000` |
| `CRISP_DEFAULT_MODEL` | Default AI model for new sessions | `gpt-4.1` |
| `PORT` | Port the bot webhook server listens on | `3978` |

---

## Microsoft Teams Setup

### Step 1 — Create an Azure Bot resource

1. Sign in to the [Azure Portal](https://portal.azure.com).
2. Search for **Azure Bot** and click **Create**.
3. Fill in:
   - **Bot handle**: choose a unique name (e.g., `crisp-t-bot`)
   - **Subscription / Resource Group**: your existing group
   - **Microsoft App Type**: `Multi-tenant`
4. Under **Microsoft App ID**, select **Create new Microsoft App ID**.
5. Click **Review + Create**, then **Create**.

### Step 2 — Record your credentials

1. Open the new Azure Bot resource.
2. Under **Configuration**, copy the **Microsoft App ID** → set as `TEAMS_APP_ID`.
3. Click **Manage password** (links to the App registration).
4. Under **Certificates & secrets**, create a new **Client secret** → set as `TEAMS_APP_PASSWORD`.

### Step 3 — Enable the Teams channel

1. In the Azure Bot resource, select **Channels**.
2. Click **Microsoft Teams** and follow the prompts to enable it.
3. Save the configuration.

### Step 4 — Set the messaging endpoint

Once your bot is reachable from the internet (see [Exposing the bot](#exposing-the-bot-with-ngrok)):

1. In the Azure Bot resource, select **Configuration**.
2. Set the **Messaging endpoint** to `https://<your-public-domain>/api/messages`.
3. Save.

### Step 5 — Add the bot to a Teams workspace

1. In the Azure Bot resource, select **Channels → Microsoft Teams → Open in Teams**.
2. In Teams, click **Add** to install the bot.
3. You can now @-mention the bot in any channel or chat with it directly.

---

## Slack Setup

### Step 1 — Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App**.
2. Select **From scratch**, give it a name (e.g., `crisp-t-bot`) and choose your workspace.

### Step 2 — Configure permissions

1. Under **OAuth & Permissions**, add the following **Bot Token Scopes**:
   - `app_mentions:read`
   - `chat:write`
   - `im:history`
   - `im:read`
   - `im:write`
2. Click **Install to Workspace** and copy the **Bot User OAuth Token** → set as `SLACK_BOT_TOKEN`.

### Step 3 — Record the signing secret

1. Under **Basic Information**, find the **Signing Secret** → set as `SLACK_SIGNING_SECRET`.

### Step 4 — Enable Events API

Once your bot is reachable from the internet (see [Exposing the bot](#exposing-the-bot-with-ngrok)):

1. Under **Event Subscriptions**, toggle **Enable Events** on.
2. Set the **Request URL** to `https://<your-public-domain>/slack/events`.
3. Under **Subscribe to bot events**, add:
   - `app_mention`
   - `message.im`
4. Save changes and reinstall the app if prompted.

---

## Running the Bot

### Start crisp-ui first (recommended)

```bash
# In one terminal
crisp-ui
```

The bot will also attempt to auto-start `crisp-ui` on startup if it is not detected, but it is more reliable to start it manually.

### Start the bot

```bash
cd integration
npm start
```

You should see:

```
[crisp-t-bot] Bot listening on port 3978
[crisp-t-bot] Teams webhook:  http://localhost:3978/api/messages
[crisp-t-bot] Slack webhook:  http://localhost:3978/slack/events
[crisp-t-bot] CRISP-T session ready (model: gpt-4.1)
```

---

## Exposing the Bot with ngrok

During development you can use [ngrok](https://ngrok.com) to create a public tunnel:

```bash
ngrok http 3978
```

Copy the `https://` forwarding URL and use it for both platforms:
- **Teams**: append `/api/messages` → paste as Azure Bot **Messaging endpoint**
- **Slack**: append `/slack/events` → paste as Slack App **Request URL**

```
https://abc123.ngrok.io/api/messages   # Teams
https://abc123.ngrok.io/slack/events   # Slack
```

---

## Available Commands

Commands work in Teams channels (via @-mention), Teams DMs, Slack channels (via @-mention), and Slack DMs.

| Command | Description |
|---------|-------------|
| `@list` or `/list` | List available AI models |
| `@switch <model>` or `/switch <model>` | Switch to a different AI model |
| `@crisp <message>` or `/crisp <message>` | Send a message to the CRISP-T AI |
| `@clear` or `/clear` | Clear the current CRISP-T session |
| `@help` or `/help` | Show all available commands |

> **Note:** In channels the bot must be @-mentioned.  In DMs commands work without a mention prefix.

---

## Example Interactions

### List available models

Teams:
```
@crisp-t-bot @list
```

Slack:
```
@crisp-t-bot /list
```

Response:
```
**Available models:**
1. gpt-4.1
2. gpt-5
3. claude-sonnet-4.5

_Current model: gpt-4.1_
```

### Switch model

```
@crisp-t-bot @switch claude-sonnet-4.5
```

Response:
```
Switched to model: **claude-sonnet-4.5**
```

### Run a CRISP-T analysis

```
@crisp-t-bot @crisp Import the CSV file from ./data using the "review" column and analyse topics
```

Response (streamed):
```
I'll import the CSV file and perform topic analysis...
[full CRISP-T response]
```

### Clear the session

```
@crisp-t-bot @clear
```

Response:
```
Session cleared. A new session will be created automatically on your next `@crisp` command.
```

### Get help

```
@crisp-t-bot @help
```

Response:
```
**CRISP-T Teams Bot — available commands:**

• `@list` or `/list` — List available AI models
• `@switch <model>` or `/switch <model>` — Switch to a different AI model
• `@crisp <message>` or `/crisp <message>` — Send a message to CRISP-T
• `@clear` or `/clear` — Clear the current CRISP-T session
• `@help` or `/help` — Show this help message
...
```

---

## Architecture

```
integration/
├── src/
│   └── index.ts          # Main bot logic (Chat SDK + Express)
├── dist/                 # Compiled JavaScript (auto-generated)
├── node_modules/         # npm dependencies (auto-generated)
├── package.json          # npm project config
├── tsconfig.json         # TypeScript config
└── .env.example          # Environment variable documentation
```

### Key functions in `src/index.ts`

| Function | Description |
|----------|-------------|
| `isCrispUIRunning()` | Health-checks the crisp-ui server |
| `startCrispUI()` | Spawns `crisp-ui` as a background process |
| `ensureCrispUIRunning()` | Combines the two above; logs if it fails |
| `ensureSession()` | Creates a CRISP-T session if none exists |
| `destroySession()` | Destroys the active CRISP-T session |
| `listModels()` | Calls `GET /api/models` and formats the result |
| `switchModel(name)` | Destroys session and recreates with new model |
| `sendCrispMessage(msg)` | Sends a message and polls for the reply |
| `clearSession()` | Alias for `destroySession()` with a friendly message |
| `getHelpText(platform?)` | Returns formatted help (includes platform name if given) |
| `routeMessage(text, platform?)` | Top-level router; returns `null` for ignored messages |
| `main()` | Starts the Express server and initialises the session |

---

## Health Check

The bot exposes a `/health` endpoint:

```bash
curl http://localhost:3978/health
```

```json
{
  "status": "ok",
  "model": "gpt-4.1",
  "sessionActive": true
}
```

---

## Troubleshooting

### Bot does not respond in Teams

- Verify the **Messaging endpoint** in the Azure Bot **Configuration** is correct and publicly accessible.
- Check that `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD` are correct.
- Inspect the bot's console output for errors.

### Bot does not respond in Slack

- Verify the **Request URL** in the Slack App **Event Subscriptions** is correct and publicly accessible.
- Check that `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` are correct.
- Ensure the app is reinstalled in the workspace after any permission changes.

### "CRISP-T server is not running"

- Start `crisp-ui` manually in a separate terminal:
  ```bash
  crisp-ui
  ```
- Verify `CRISP_UI_URL` points to the correct host/port.

### "Copilot SDK not available"

- Install the copilot extra: `pip install crisp-t[copilot]`

### Session errors

- Run `@clear` to reset the session and try again.

---

## Security Notes

- Never commit your `.env` file.
- Use short-lived client secrets and rotate them regularly.
- Restrict `TEAMS_APP_TENANT_ID` to your organisation's tenant in production.
- Store Slack tokens securely and rotate them if compromised.
- For production, use Redis (`@chat-adapter/state-redis`) instead of in-memory state.
