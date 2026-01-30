# Skill: crisp-mcp

## Summary

**Goal:**
Use the CRISP‑T MCP server to run a full, *documented* qualitative analysis workflow—linking textual narratives and numeric data—using computational triangulation and grounded theory–inspired coding.

This skill assumes:

- The **CRISP‑T MCP server** is available as an MCP server (e.g., `crisp-t`)
- The server exposes tools for **project management, text coding, numeric analysis, triangulation, and export**
- The model can **discover tools at runtime** and adapt to the actual tool list

Use this skill when the user wants to:

- Explore and code qualitative data (e.g., interviews, narratives, open‑ended survey responses)
- Triangulate qualitative codes with quantitative variables
- Produce **transparent, reproducible outputs** (memos, codebooks, matrices, summaries)

---

## Intended users and scenarios

- **Qualitative / mixed‑methods researchers** working with text + numeric data
- **Small‑to‑medium research teams** (e.g., teaching‑focused universities) who need reproducible workflows
- **Students** learning computational grounded theory, CRISP‑T, or computational triangulation

Typical scenarios:

- Coding COVID‑era patient narratives and linking them to open‑source epidemiological data
- Triangulating interview themes with survey scores or EHR‑derived indicators
- Building theory from mixed text–numeric corpora using CRISP‑T stages

---

## MCP server and tools

### Server

- **Server name (example):** `crisp-t`
- **Protocol:** Model Context Protocol (MCP) via stdio (e.g., `crisp-mcp`)

> **Important:**
> Always **discover** the available tools from the CRISP‑T MCP server first, then map them to the workflow below.
> Tool names in this file are **illustrative**; prefer the *actual* tool names and schemas exposed by the server.

### Expected tool capabilities

When you discover tools from the `crisp-t` MCP server, look for tools that correspond to the following capabilities:

1. **Project and configuration**
   - **List / load projects** (e.g., `list_projects`, `load_project`)
   - **List / configure data sources** (text + numeric)

2. **Textual data operations**
   - **List text sources** (e.g., documents, transcripts, narrative files)
   - **Fetch text segments** (e.g., by document, case, or segment ID)
   - **Apply or update codes** to segments
   - **List codes / categories / codebook**

3. **Numeric / structured data operations**
   - **List numeric datasets** (e.g., CSVs, tables)
   - **Fetch rows / variables** for a given case or group
   - **Run basic summaries or models** (e.g., descriptive stats, correlations)

4. **Triangulation / integration**
   - **Link coded text segments to numeric data** by case ID or key
   - **Generate triangulation matrices** (codes × variables, cases × themes, etc.)
   - **Compute simple indicators** (e.g., frequency, co‑occurrence, association)

5. **Memoing and export**
   - **Create / update analytic memos**
   - **Export codebooks, matrices, and summaries** (e.g., CSV, JSON, Markdown)
   - **Save session context or analysis snapshots**

When using this skill, you should:

- **Inspect tool descriptions and schemas**
- **Prefer higher‑level CRISP‑T tools** (e.g., “triangulate”, “inspect”, “sense‑make”) over low‑level primitives when available
- **Avoid guessing tool parameters**—use the schema and ask the user when needed

---

## Inputs

The user may provide some or all of the following. If missing, ask concise clarification questions.

- **Research question / focus**
  - `research_question`: short description of what they want to understand or explain
- **Project / dataset selection**
  - `project_name` or `project_id`
  - `text_source_pattern` (e.g., “patient_narratives”, “interviews_*”)
  - `numeric_dataset_name` (e.g., “covid_metrics.csv”, “survey_scores”)
- **Methodological preferences**
  - `coding_style`: open coding, focused coding, or using an existing codebook
  - `triangulation_focus`: e.g., “compare themes by age group”, “link narratives to hospitalization status”
- **Output preferences**
  - `output_format`: Markdown, CSV, JSON, or mixed
  - `audience`: “methods‑savvy colleague”, “student”, “grant reviewer”, etc.

If the user is vague, default to:

- Open coding → focused coding
- Simple triangulation (themes × one or two key numeric variables)
- Markdown summaries plus CSV exports where supported

---

## Outputs

This skill should aim to produce:

1. **Analytic memo**
   - Research question and context
   - Data sources used (text + numeric)
   - Coding approach and key categories
   - Triangulation findings (convergences, divergences, surprises)
   - Limitations and next steps

2. **Structured artifacts** (when tools support them)
   - Codebook (codes, definitions, examples)
   - Triangulation matrix (e.g., codes × variables)
   - Exported files (CSV/JSON/Markdown) referenced in the memo

3. **Reproducible trace**
   - Clear description of which MCP tools were called
   - Parameters used (e.g., filters, variables, thresholds)
   - Any important analytic decisions made during the session

---

## High‑level workflow

You should follow this workflow, adapting to the actual CRISP‑T tools you discover.

### 1. Clarify research focus and context

1. **Summarize the user’s goal** in 2–3 sentences.
2. If unclear, ask **one or two** focused questions:
   - **Example:** “Which is more important right now: exploring themes, or linking themes to numeric outcomes?”
3. Record the clarified research question in the analytic memo.

### 2. Discover CRISP‑T MCP tools and project context

1. **List available tools** from the `crisp-t` MCP server.
2. Identify tools that correspond to:
   - Project management
   - Text operations
   - Numeric operations
   - Triangulation
   - Memoing / export
3. If available, **list projects** and either:
   - Load the project specified by the user, or
   - Present a short list and ask the user to choose.

Document in the memo:

- Which project was loaded
- Which tools will be used and for what purpose

### 3. Inspect data sources (text + numeric)

1. Use project / data tools to:
   - **List text sources** (e.g., narrative collections, interview sets)
   - **List numeric datasets** (e.g., survey tables, EHR‑derived tables)
2. For each candidate source:
   - Fetch a **small sample** (e.g., a few narratives, a few rows of numeric data)
   - Briefly describe:
     - What the data represent
     - How they relate to the research question
3. Confirm with the user which sources to prioritize.

Update the memo with:

- Selected text sources and why
- Selected numeric datasets and why

### 4. Initial coding of qualitative data (open / exploratory)

1. Select a **manageable subset** of text (e.g., 10–20 narratives or segments).
2. Apply **open coding** using the CRISP‑T coding tools:
   - Generate **short, grounded codes** that stay close to the data
   - Avoid premature theory; focus on what participants are saying/doing/experiencing
3. Use MCP tools to:
   - Save codes to the project
   - Attach codes to segments
   - Retrieve and refine the emerging code list

In the memo:

- Describe the coding approach (e.g., open coding, line‑by‑line, incident‑focused)
- Provide **examples**: quote → code → brief interpretation

### 5. Focused coding and category development

1. Review the emerging code list using CRISP‑T tools (e.g., list codes, code frequencies).
2. Group related codes into **higher‑level categories** or themes:
   - Merge overlapping codes
   - Distinguish closely related but conceptually different codes
3. If the server supports it, create or update:
   - A **codebook** (code name, definition, inclusion/exclusion criteria, example quotes)
   - **Category structures** (e.g., parent/child relationships)

In the memo:

- Describe key categories and how they were derived
- Note any tensions, ambiguities, or alternative interpretations

### 6. Computational triangulation with numeric data

Using CRISP‑T tools that link text and numeric data:

1. **Identify the linkage key** (e.g., participant ID, case ID, encounter ID).
2. For each case or group:
   - Retrieve **coded qualitative data** (codes, categories, or themes)
   - Retrieve **numeric variables** (e.g., age, scores, outcomes, counts)
3. Construct **triangulation views**, such as:
   - Code frequencies by group (e.g., theme prevalence by age band or outcome)
   - Associations between codes and numeric indicators (e.g., high vs. low scores)
   - Patterns of convergence (where qualitative and quantitative tell a similar story)
   - Patterns of divergence (where they conflict or complicate each other)

When tools support it, generate:

- **Triangulation matrices** (e.g., codes × variables, cases × themes)
- **Summary statistics** (e.g., counts, proportions, simple associations)

In the memo:

- Describe **how** triangulation was done (data sources, linkage, variables)
- Highlight **convergences, divergences, and surprises**
- Emphasize that triangulation is about **enriching and challenging** interpretations, not “proving” them

### 7. Synthesis and theory‑building

1. Integrate insights from:
   - Qualitative categories and themes
   - Numeric patterns and contrasts
   - Triangulation matrices and summaries
2. Propose **tentative theoretical statements** or propositions:
   - “In this context, participants who X tend to Y, especially when Z…”
3. Explicitly note:
   - Alternative explanations
   - Data limitations
   - What additional data or analysis would strengthen the claims

Update the memo with:

- A **coherent narrative** that links data, codes, categories, and numeric patterns
- Clear statements about **what the data support** and **what remains uncertain**

### 8. Export and documentation

1. Use CRISP‑T export tools to save:
   - Codebook(s)
   - Triangulation matrices
   - Any generated summaries or tables
2. If supported, export:
   - The analytic memo as Markdown or another requested format
   - A machine‑readable representation of the analysis (e.g., JSON snapshot)

Provide the user with:

- A **human‑readable summary** (memo)
- A **list of exported artifacts** and where they are stored (paths, IDs, or URLs)

---

## Model behavior and style guidelines

When using this skill, you should:

- **Be methodologically explicit**
  - Name what you are doing (open coding, focused coding, triangulation, memoing)
  - Briefly justify key analytic decisions
- **Stay grounded in the data**
  - Use short quotes and concrete examples
  - Avoid over‑generalizing beyond what the data support
- **Respect the user’s context**
  - If they are teaching, highlight steps and rationale for students
  - If they are writing a paper or grant, emphasize clarity and traceability
- **Limit questions**
  - Ask at most **one or two** focused questions at a time
  - Default to reasonable assumptions when the user is not specific

---

## Limitations and cautions

- This skill **does not replace** human qualitative judgment; it supports and documents it.
- Triangulation results should be treated as **aids to interpretation**, not definitive proof.
- Always encourage the user to:
  - Review codes and categories
  - Challenge interpretations
  - Consider ethical and contextual factors not visible in the data

---
