# CRISP-T Microsoft Teams Integration

This directory contains a **Microsoft Teams bot** for CRISP-T, built with the
[Chat SDK](https://github.com/your-org/chat) (`chat` npm package).  Users can
@mention the bot or send it a direct message to run CRISP-T analysis commands
directly from within a Teams workspace.

---

## Directory Structure

```
integration/
├── src/
│   ├── index.ts            # Bot entry point – Chat SDK setup, event handlers
│   ├── crisp_runner.ts     # CRISP-T command executor (argument parsing, execFile)
│   └── __tests__/
│       └── crisp_runner.test.ts  # Unit tests for the command runner
├── package.json            # Node.js project configuration
├── tsconfig.json           # TypeScript compiler settings
├── .env.example            # Environment variable template
└── README.md               # This file
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Node.js | ≥ 18 |
| npm | ≥ 9 |
| CRISP-T | installed and on `PATH` (`pip install crisp-t`) |
| Microsoft Azure account | – |

---

## Quick Start

### 1 – Register a Bot in Azure

1. Go to the [Azure Portal](https://portal.azure.com/) and navigate to
   **Azure Bot** → **Create**.
2. Choose a bot handle (e.g. `crisp-bot`) and a resource group.
3. Under **Microsoft App ID** choose *Create new Microsoft App ID*.
4. After creation, open **Configuration** and note the **App ID**.
5. Under **Manage** → **Certificates & secrets**, create a new **client
   secret** and note the value.

### 2 – Install Dependencies

```bash
cd src/crisp_t/integration
npm install
```

### 3 – Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in TEAMS_APP_ID and TEAMS_APP_PASSWORD
```

### 4 – Build and Run

```bash
npm run build   # compile TypeScript → dist/
npm start       # launch the webhook server (default port 3978)
```

For local development with hot-reload:

```bash
npx ts-node src/index.ts
```

### 5 – Expose the Webhook

Teams needs a publicly accessible HTTPS URL.  During development use a
tunnelling tool:

```bash
# ngrok (https://ngrok.com/)
ngrok http 3978
# Copy the https URL, e.g. https://abc123.ngrok.io
```

### 6 – Configure the Messaging Endpoint

Back in the Azure Portal → your bot → **Configuration**:

* **Messaging endpoint:** `https://<your-tunnel-url>/api/messages`

### 7 – Add the Bot to a Teams Channel

1. In Teams, click **Apps** → **Manage your apps** → **Upload an app**.
2. Upload the Teams app manifest (create one via the
   [Developer Portal](https://dev.teams.microsoft.com/)) pointing at your
   bot's App ID.
3. Add the app to the desired team or chat.

---

## Usage

Once the bot is installed in Teams, interact with it as follows:

| Interaction | Example |
|-------------|---------|
| @mention in a channel | `@crisp-bot crisp --help` |
| Direct message | `crisp load-corpus --path data/study.csv` |
| Ask for help | `@crisp-bot help` |

Available CRISP-T commands:

```
crisp <args>      # Main analysis tool
crispt <args>     # Corpus management
crispviz <args>   # Visualisation
```

---

## Running Tests

```bash
npm test
```

---

## Further Reading

* **[Full integration guide](../../../docs/integration.md)**
* **[Additional notes and future plans](../../../notes/integration.md)**
* **[Chat SDK documentation](https://github.com/your-org/chat)**
* **[CRISP-T documentation](https://dermatologist.github.io/crisp-t/)**
