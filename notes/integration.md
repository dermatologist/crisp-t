# CRISP-T Chat Bot Integration — Notes

## Design Decisions

### In-memory state adapter
The integration uses `@chat-adapter/state-memory` for simplicity and zero-dependency setup.  In a production deployment with multiple bot instances or across restarts, replace it with `@chat-adapter/state-redis` (or another persistent state adapter) so subscribed threads survive process restarts.

### Single-session model
The current implementation uses a single, shared CRISP-T session (`crisp-bot-session`) for all users across both Teams and Slack.  This is appropriate for small research teams where all users share the same analysis context.  A future enhancement could create per-user or per-thread sessions keyed by platform user ID or conversation ID.

### Platform-aware help text
The `getHelpText(platform?)` function accepts an optional platform name (`'Teams'` or `'Slack'`).  When provided, the help header includes the platform name (e.g., "CRISP-T Slack Bot — available commands").  The `routeMessage(text, platform?)` function passes this through so users see contextual help.

### Polling for responses
After sending a message to `crisp-ui`, the bot polls the `/api/session/<id>/messages` endpoint in 1-second intervals for up to 30 seconds.  This is a practical workaround because the `crisp-ui` REST API does not yet expose Server-Sent Events or WebSockets for real-time streaming.  Future versions could adopt streaming once the API supports it.

### Express + Web Fetch API bridging
The Chat SDK webhook handlers use the standard Web Fetch API (`Request`/`Response`).  The bot uses Express for the HTTP server and converts between Express's Node.js-style `req`/`res` and the Web Fetch API types via two small helper functions (`buildWebRequest` / `sendWebResponse`).  This pattern avoids introducing additional HTTP framework dependencies.

### Shared webhook server
Both the Teams and Slack webhooks are served from the same Express process on the same port.  Teams uses `POST /api/messages` and Slack uses `POST /slack/events`.  This simplifies deployment and reduces the number of processes to manage.

---

## Future Plans

### Per-user / per-thread sessions
Allow each user (or conversation thread) to maintain an independent CRISP-T session so that multiple researchers can work simultaneously without sharing context.  Session IDs could be derived from the platform-specific user ID or conversation ID.

### Streaming responses
Integrate with Chat SDK's streaming API (`thread.stream()`) once the `crisp-ui` backend exposes real-time streaming.  This will give users incremental feedback instead of waiting for the full response.

### Redis-backed state
Switch to `@chat-adapter/state-redis` for production deployments to persist thread subscriptions across bot restarts.

### Slash command support
Register platform-specific slash commands (`/crisp`, `/list`, etc.) via the Azure Bot manifest (Teams) and Slack manifest so users get auto-complete suggestions and inline help.

### Adaptive Cards (Teams) and Block Kit (Slack)
Use Chat SDK JSX cards (`<Card>`, `<Actions>`, etc.) to render CRISP-T results as rich interactive messages with buttons for common follow-up actions (e.g., "Analyse topics", "Run regression", "Save corpus").

### Additional platforms
The Chat SDK makes it trivial to extend the bot further.  Adding Google Chat, Discord, or Telegram support only requires importing the corresponding adapter and providing credentials — the same `routeMessage` logic works unchanged.

### Authentication
Integrate Azure Active Directory (AAD) authentication for Teams and Slack OAuth for Slack to ensure only authorised users can interact with the bot and run CRISP-T analyses.

### Deployment
Document and automate deployment to Azure App Service, Azure Container Apps, or a Docker container for production use.

---

## Development Notes

### Running tests
```bash
# From the repository root
pytest tests/test_integration.py -v
```

### Type-checking without building
```bash
cd src/crisp_t/integration
npx tsc --noEmit
```

### Rebuilding after code changes
```bash
cd src/crisp_t/integration
npm run build
```

### Local end-to-end testing with Bot Framework Emulator
1. Download the [Bot Framework Emulator](https://github.com/microsoft/BotFramework-Emulator).
2. Start `crisp-ui` and the Teams bot locally.
3. Open the emulator and connect to `http://localhost:3978/api/messages`.
4. Type commands like `/list`, `/help`, `/crisp Hello`.

---

## References

- [Chat SDK documentation](https://github.com/nicholasgasior/chat-sdk)
- [`@chat-adapter/teams` adapter](https://www.npmjs.com/package/@chat-adapter/teams)
- [Azure Bot Framework](https://dev.botframework.com/)
- [Microsoft Teams developer documentation](https://learn.microsoft.com/en-us/microsoftteams/platform/)
- [CRISP-T Web UI documentation](docs/ui.md)
- [CRISP-T MCP Server documentation](docs/MCP_SERVER.md)
