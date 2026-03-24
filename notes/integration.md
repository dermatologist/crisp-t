# CRISP-T Microsoft Teams Integration — Notes

## Design Decisions

### In-memory state adapter
The integration uses `@chat-adapter/state-memory` for simplicity and zero-dependency setup.  In a production deployment with multiple bot instances or across restarts, replace it with `@chat-adapter/state-redis` (or another persistent state adapter) so subscribed threads survive process restarts.

### Single-session model
The current implementation uses a single, shared CRISP-T session (`teams-bot-session`) for all Teams users.  This is appropriate for small research teams where all users share the same analysis context.  A future enhancement could create per-user or per-thread sessions keyed by Teams user ID or conversation ID.

### Polling for responses
After sending a message to `crisp-ui`, the bot polls the `/api/session/<id>/messages` endpoint in 1-second intervals for up to 30 seconds.  This is a practical workaround because the `crisp-ui` REST API does not yet expose Server-Sent Events or WebSockets for real-time streaming.  Future versions could adopt streaming once the API supports it.

### Express + Web Fetch API bridging
The Chat SDK webhook handler uses the standard Web Fetch API (`Request`/`Response`).  The bot uses Express for the HTTP server and manually converts between Express's Node.js-style `req`/`res` and the Web Fetch API types.  This is a widely-used compatibility pattern that avoids introducing additional dependencies (e.g., Hono).

---

## Future Plans

### Per-user / per-thread sessions
Allow each Teams user (or conversation thread) to maintain an independent CRISP-T session so that multiple researchers can work simultaneously without sharing context.

### Streaming responses
Integrate with Chat SDK's streaming API (`thread.stream()`) once the `crisp-ui` backend exposes real-time streaming.  This will give users incremental feedback instead of waiting for the full response.

### Redis-backed state
Switch to `@chat-adapter/state-redis` for production deployments to persist thread subscriptions across bot restarts.

### Slash command support
Register Teams slash commands (`/crisp`, `/list`, etc.) via the Azure Bot manifest so users get auto-complete suggestions and inline help.

### Adaptive Cards
Use Chat SDK JSX cards (`<Card>`, `<Actions>`, etc.) to render CRISP-T results as rich Adaptive Cards with buttons for common follow-up actions (e.g., "Analyse topics", "Run regression", "Save corpus").

### Multi-channel support
The Chat SDK makes it trivial to extend the bot to additional platforms.  Adding Slack support, for example, only requires importing `createSlackAdapter` and providing the Slack credentials — the same event handlers work unchanged.

### Authentication
Integrate Azure Active Directory (AAD) authentication to ensure only authorised users can interact with the bot and run CRISP-T analyses.

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
