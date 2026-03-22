# Microsoft Teams Integration

CRISP-T ships a **Microsoft Teams bot** that lets researchers run qualitative
research commands directly from a Teams chat.  The bot is built with the
[Chat SDK](https://www.npmjs.com/package/chat) (`chat` npm package) and the
[`@chat-adapter/teams`](https://www.npmjs.com/package/@chat-adapter/teams)
adapter, keeping Teams-specific complexity separate from the CRISP-T core.

---

## Architecture

```
Teams User
    │  @mention / DM
    ▼
Microsoft Teams
    │  HTTPS POST  (Bot Framework messages)
    ▼
Chat SDK Teams Adapter  (webhook on port 3978)
    │
    ├─ onNewMention ──────────────────┐
    ├─ onDirectMessage ───────────────┤
    └─ onSubscribedMessage ───────────┤
                                      ▼
                              handleUserMessage()
                                      │
                              parseCrispCommand()
                                      │
                              runCrispCommand()   (execFile – no shell)
                                      │
                              crisp / crispt / crispviz  (CRISP-T CLI)
                                      │
                              result posted back to Teams thread
```

The bot process runs **outside** the Python CRISP-T package—it is a Node.js
process that executes CRISP-T commands as child processes using `execFile`.
This keeps the Python and Node.js runtimes fully isolated.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python ≥ 3.10 | CRISP-T must be installed and its CLI tools on `PATH` |
| Node.js ≥ 18 | Required to run the bot |
| npm ≥ 9 | Package management |
| Azure account | Free tier is sufficient for development |
| Microsoft 365 tenant | For adding the bot to a Teams workspace |

---

## Installation

### 1 – Install CRISP-T

```bash
pip install crisp-t
# Verify the CLI tools are available:
crisp --version
crispt --version
crispviz --version
```

### 2 – Install bot dependencies

```bash
cd src/crisp_t/integration
npm install
```

### 3 – Build TypeScript

```bash
npm run build
```

---

## Azure Bot Registration

### Create the bot

1. Sign in to [portal.azure.com](https://portal.azure.com).
2. Click **Create a resource** → search for **Azure Bot** → **Create**.
3. Fill in:
   - **Bot handle** – e.g. `crisp-research-bot`
   - **Subscription / Resource group** – choose or create
   - **Pricing tier** – F0 (free)
   - **Microsoft App ID** – select *Create new Microsoft App ID*
4. Click **Review + create** → **Create**.

### Obtain credentials

After deployment open the bot resource:

1. **Configuration** → copy the **Microsoft App ID** → this is `TEAMS_APP_ID`.
2. **Manage** (link next to App ID) → **Certificates & secrets** →
   **New client secret** → copy the value → this is `TEAMS_APP_PASSWORD`.

---

## Configuration

Copy the environment template and fill in your values:

```bash
cd src/crisp_t/integration
cp .env.example .env
```

Edit `.env`:

```dotenv
TEAMS_APP_ID=<your-azure-app-id>
TEAMS_APP_PASSWORD=<your-azure-app-secret>
PORT=3978
CRISP_DATA_PATH=./workspace
CRISP_COMMAND_TIMEOUT=300
LOG_LEVEL=info
```

| Variable | Description | Default |
|----------|-------------|---------|
| `TEAMS_APP_ID` | Azure Bot App ID | *(required)* |
| `TEAMS_APP_PASSWORD` | Azure Bot client secret | *(required)* |
| `PORT` | Webhook server port | `3978` |
| `CRISP_DATA_PATH` | Working directory for CRISP commands | `./workspace` |
| `CRISP_COMMAND_TIMEOUT` | Max execution time per command (seconds) | `300` |
| `LOG_LEVEL` | Log verbosity (`error`\|`warn`\|`info`\|`debug`) | `info` |

---

## Running the Bot

### Development (with tunnel)

```bash
# Terminal 1 – start the bot
npm run dev   # uses ts-node, no compile step

# Terminal 2 – expose it publicly
ngrok http 3978
```

Copy the `https://` URL from ngrok.

### Production

```bash
npm run build
npm start
```

For production deployments consider running behind a reverse proxy (nginx) with
a valid TLS certificate instead of a tunnel.

---

## Connecting to Microsoft Teams

### Set the messaging endpoint

In **Azure Portal** → your bot → **Configuration**:

* **Messaging endpoint:** `https://<your-public-url>/api/messages`

Teams calls this URL for every message the bot receives.

### Create a Teams app manifest

Use the [Teams Developer Portal](https://dev.teams.microsoft.com/):

1. **Apps** → **New app**.
2. Fill in the basic information (name, description, icons).
3. Under **App features** → **Bot** → enter your App ID.
4. Enable **Personal** and/or **Team** scopes as needed.
5. **Publish** → **Download app package** (`.zip`).

### Sideload the app

1. In Microsoft Teams → **Apps** → **Manage your apps** →
   **Upload an app** → **Upload a custom app**.
2. Select the `.zip` you downloaded.
3. Add it to a team or personal chat.

---

## Interacting with the Bot

Once installed, users can:

### @mention in a channel

```
@crisp-bot crisp --help
@crisp-bot crispt load-corpus --path data/study.csv
@crisp-bot crispviz --wordcloud --out results/
```

### Direct message

Simply send the command text without the @mention:

```
crisp --topics --sentiment
crispt --semantic "patient experience" --num 10
help
```

### Getting help

Send `help` or @mention without a command to see the built-in usage card.

---

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `crisp <args>` | Main CRISP-T analysis tool | `crisp --topics` |
| `crispt <args>` | Corpus management and tagging | `crispt --meta "phase=1"` |
| `crispviz <args>` | Visualisation and reporting | `crispviz --wordcloud` |

Pass `--help` to any command for a full list of options:

```
@crisp-bot crisp --help
@crisp-bot crispt --help
@crisp-bot crispviz --help
```

---

## Example Interaction

```
User: @crisp-bot crisp --help

Bot: ⏳ Running `crisp --help`…

Bot: ✅ `crisp` completed

     Usage: crisp [OPTIONS]

       CRISP-T - Qualitative Research Analysis Framework

     Options:
       --inp TEXT      Input corpus path
       --source TEXT   Source data directory
       --topics        Run LDA topic modelling
       --sentiment     Run sentiment analysis
       ...
```

---

## Error Handling

The bot provides clear error messages for common failure scenarios:

| Scenario | Bot response |
|----------|-------------|
| Unknown command | Suggests `help` and `crisp --help` |
| CRISP-T not installed | `Error: 'crisp' was not found. Make sure CRISP-T is installed and on PATH.` |
| Command times out | `Error: command 'crisp' timed out after 300 seconds.` |
| Command exits with error | Full stderr/stdout output included |
| Unexpected exception | `⚠️ An unexpected error occurred…` with log reference |

---

## Security Considerations

* **Command allowlist** – only `crisp`, `crispt`, and `crispviz` may be
  executed; any other input is rejected before a subprocess is spawned.
* **No shell invocation** – `execFile` is used instead of `exec` or
  `spawn('sh', ['-c', …])`, eliminating shell-injection vectors.
* **Argument tokenisation** – the `splitArgs` helper tokenises arguments
  without passing them through a shell, respecting quoted strings.
* **Workspace isolation** – all file I/O is scoped to `CRISP_DATA_PATH`.
* **Credentials** – `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD` are read from
  environment variables and never logged or committed to source control.

---

## Troubleshooting

### Bot does not respond in Teams

1. Confirm the messaging endpoint URL is set correctly in Azure Portal.
2. Check the bot process is running (`npm start`) and the port is reachable.
3. Verify `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD` are correct.

### "CRISP-T not found" error

Ensure CRISP-T is installed in the same environment that runs the bot:

```bash
pip install crisp-t
which crisp   # should print a path
```

If using a virtual environment, activate it before starting the bot, or set
`PATH` explicitly in the bot's process environment.

### "Command timed out" error

Increase `CRISP_COMMAND_TIMEOUT` in `.env` for long-running analyses.

---

## Further Reading

* [Chat SDK documentation](https://www.npmjs.com/package/chat)
* [@chat-adapter/teams documentation](https://www.npmjs.com/package/@chat-adapter/teams)
* [CRISP-T CLI Cheatsheet](cheatsheet.md)
* [CRISP-T Demo](DEMO.md)
* [Notes and future plans](../notes/integration.md)
