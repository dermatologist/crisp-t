# Teams Integration – Notes and Future Plans

This document captures design decisions, lessons learned, and ideas for
extending the Microsoft Teams integration beyond its initial implementation.

---

## Design Decisions

### Why Chat SDK?

The [Chat SDK](https://www.npmjs.com/package/chat) provides a unified,
adapter-based API for building bots across multiple chat platforms.  Choosing
it means:

* The same bot logic could be extended to Slack, Google Chat, Discord, or
  Telegram with minimal code changes (swap the adapter).
* Event routing (`onNewMention`, `onDirectMessage`, `onSubscribedMessage`) is
  handled by the SDK, keeping `index.ts` clean.
* Webhook server wiring is a single call to `bot.webhooks.teams`.

### Why `execFile` instead of `exec`?

`child_process.execFile` spawns the executable directly without invoking a
shell.  This eliminates shell-injection vulnerabilities that would be possible
if user-supplied arguments were interpolated into a shell command string.
Combined with the `ALLOWED_COMMANDS` allowlist, the attack surface is minimal.

### In-memory state

The initial implementation uses `@chat-adapter/state-memory` for simplicity.
This is appropriate for a single-node development deployment but does **not**
persist subscriptions across bot restarts.

---

## Current Limitations

| Limitation | Impact | Planned fix |
|------------|--------|-------------|
| In-memory state | Thread subscriptions lost on restart | Swap to Redis or PostgreSQL state adapter |
| Single-node only | Cannot scale horizontally | Persistent state adapter + sticky sessions or stateless design |
| No authentication | Any Teams user can run CRISP commands | Add an allowlist of permitted user IDs / email domains |
| File output not surfaced | Files written by CRISP-T are not sent back to Teams | Implement file upload via Chat SDK files API |
| No streaming output | Long commands show no progress until complete | Stream stdout lines as Teams message edits |
| No adaptive cards | Responses are plain Markdown | Render results as Teams Adaptive Cards for richer UX |

---

## Roadmap / Future Plans

### Near-term

- [ ] **Persistent state** – replace `createMemoryState()` with
  `createRedisState()` or `createPostgresState()` so subscriptions survive
  restarts.
- [ ] **User allowlist** – check `message.author.id` against a configured list
  of permitted Teams user IDs or email domains before executing commands.
- [ ] **Rich response cards** – use the Chat SDK JSX card API to format
  command output as structured Adaptive Cards with collapsible sections.

### Medium-term

- [ ] **Streaming output** – pipe `stdout` lines from the child process back to
  Teams in real time using message edits (the Chat SDK `thread.stream()` API).
- [ ] **File attachments** – after CRISP-T writes output files (e.g. word-cloud
  images, CSV exports), upload them to Teams using the Chat SDK files API.
- [ ] **Slash commands** – register `/crisp`, `/crispt`, `/crispviz` as Teams
  slash commands so users get auto-completion and parameter hints.
- [ ] **Multi-platform** – with adapters already abstracted, add Slack support
  by registering a `createSlackAdapter()` alongside the Teams adapter.

### Longer-term

- [ ] **MCP bridge** – instead of shelling out to the CLI, communicate with the
  CRISP-T MCP server over stdio/HTTP for richer tool invocation and structured
  response data.
- [ ] **Conversational AI** – pipe messages through the Copilot SDK (as the web
  UI does) so the bot can answer natural-language research questions, not just
  execute commands.
- [ ] **OAuth / SSO** – integrate Azure AD single-sign-on so the bot can
  associate Teams users with CRISP-T project workspaces automatically.
- [ ] **Docker image** – publish an official Docker image that bundles both the
  Python CRISP-T CLI tools and the Node.js bot, simplifying deployment.

---

## Deployment Notes

### Docker

A minimal two-stage Dockerfile for the bot:

```dockerfile
# Stage 1 – build TypeScript
FROM node:20-alpine AS builder
WORKDIR /app
COPY src/crisp_t/integration/package*.json ./
RUN npm ci
COPY src/crisp_t/integration/ ./
RUN npm run build

# Stage 2 – runtime
FROM python:3.12-slim
RUN pip install crisp-t
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
ENV NODE_ENV=production
CMD ["node", "dist/index.js"]
```

### Environment Variables in Production

Use a secrets manager (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault)
instead of `.env` files for `TEAMS_APP_ID` and `TEAMS_APP_PASSWORD` in
production deployments.

### Azure App Service

The bot can be hosted on Azure App Service (Node.js runtime):

1. Create an App Service plan and Web App.
2. Set **Configuration** → **Application settings** with the environment
   variables from `.env`.
3. Deploy via GitHub Actions, Azure CLI, or VS Code extension.
4. Update the messaging endpoint in Azure Bot to the App Service URL.

---

## References

* [Chat SDK npm package](https://www.npmjs.com/package/chat)
* [@chat-adapter/teams](https://www.npmjs.com/package/@chat-adapter/teams)
* [Microsoft Bot Framework](https://dev.botframework.com/)
* [Teams Developer Portal](https://dev.teams.microsoft.com/)
* [Azure Bot Service documentation](https://docs.microsoft.com/en-us/azure/bot-service/)
* [CRISP-T integration docs](../docs/integration.md)
