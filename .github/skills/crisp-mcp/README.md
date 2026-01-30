# crisp-mcp

**A reusable qualitative + mixed‑methods analysis skill for the CRISP‑T MCP Server**

`crisp-mcp` is a Model Context Protocol (MCP) skill designed to orchestrate qualitative, quantitative, and triangulated analysis workflows using the tools exposed by the **CRISP‑T MCP server**.
It encodes a transparent, reproducible, and methodologically grounded process for computational qualitative research — including open coding, focused coding, category development, memoing, and triangulation with numeric data.

This skill is ideal for:

- Mixed‑methods researchers
- Teaching‑focused institutions building reproducible analytic capacity
- Students learning grounded theory, CRISP‑T, or computational triangulation
- Research teams working with text + numeric datasets (e.g., narratives + survey/EHR data)

---

## ✨ What this skill does

`crisp-mcp` provides a structured workflow that:

1. **Discovers available CRISP‑T MCP tools** and adapts to the server’s capabilities
2. **Loads and inspects qualitative and quantitative data sources**
3. **Performs open and focused coding** using MCP text‑analysis tools
4. **Builds categories and codebooks**
5. **Links coded text to numeric variables** for triangulation
6. **Generates triangulation matrices and summaries**
7. **Produces analytic memos and exports** (Markdown, CSV, JSON, depending on server support)
8. **Documents every analytic decision** for transparency and reproducibility

The skill does *not* assume a specific tool naming scheme — it dynamically maps to whatever tools the CRISP‑T MCP server exposes.

---


If you are integrating this skill into a larger MCP ecosystem, place it in your agent’s skills directory or register it through your MCP client’s configuration.

---

## 🧠 Methodological foundations

This skill encodes best practices from:

- **Computational grounded theory**
- **CRISP‑T** (Collect → Reflect → Inspect → Sense‑make → Present → Transform)
- **Triangulation of data types** (text ↔ numeric)
- **Memo‑driven qualitative analysis**
- **Reproducible mixed‑methods workflows**

It is intentionally explicit about analytic decisions, making it suitable for teaching, collaboration, and publication‑ready documentation.

---

## 🛠️ MCP server requirements

The skill expects the CRISP‑T MCP server to expose tools for:

- Project management
- Listing and loading text sources
- Retrieving text segments
- Applying and listing codes
- Listing and loading numeric datasets
- Linking cases across data types
- Generating triangulation outputs
- Creating or updating memos
- Exporting artifacts

Because MCP supports tool discovery, the skill automatically adapts to the server’s actual tool list.

---

## 🚀 How to use this skill

### 1. Load the skill in your MCP‑compatible client

Depending on your client (Claude Desktop, LangChain MCP, FastMCP, etc.), load the skill by pointing to the `SKILL.md` file.

### 2. Connect to the CRISP‑T MCP server


### 3. Ask the model to begin an analysis

Examples:

- “Use the `crisp-mcp` skill to analyze my patient narratives and link them to the survey dataset.”
- “Run a full CRISP‑T workflow on the interviews in this project.”
- “Triangulate themes from the open‑ended responses with the numeric indicators.”

### 4. Provide optional parameters

The skill accepts optional inputs such as:

- Research question
- Project name
- Text source pattern
- Numeric dataset name
- Coding preferences
- Output format

If you omit them, the model will ask concise clarifying questions.

---

## 📄 Outputs

Depending on the tools available in the CRISP‑T MCP server, the skill may produce:

- Analytic memos (Markdown)
- Codebooks
- Triangulation matrices
- Summary tables
- JSON/CSV exports
- Reproducible traces of tool calls and analytic decisions

These outputs are designed to be publication‑ready or easily integrated into manuscripts, reports, and teaching materials.

---

## 🧩 Why this skill matters

`crisp-mcp` helps small‑to‑medium research teams:

- Reduce prompt overhead
- Standardize analytic workflows
- Improve reproducibility
- Teach computational qualitative methods
- Integrate text + numeric data without custom scripting
- Build sustainable analytic capacity aligned with CRISP‑T and mixed‑methods best practices

It’s a practical bridge between **domain expertise**, **methodological rigor**, and **agentic AI workflows**.

---

## 🤝 Contributing

Contributions are welcome — especially from researchers, educators, and developers working on:

- CRISP‑T
- Mixed‑methods pedagogy
- MCP server extensions
- Domain‑specific analytic workflows
- Reproducible research infrastructure

Please open an issue or pull request with your ideas.

---

## 📜 License

This skill inherits the license of the parent repository unless otherwise specified.

---

If you want, I can also generate:

- A **diagram** showing how the skill interacts with the CRISP‑T MCP server
- A **demo notebook** showing how to use the skill in LangChain or FastMCP
- A **teaching handout** explaining the workflow for students

Just tell me what direction you want to take this next.


