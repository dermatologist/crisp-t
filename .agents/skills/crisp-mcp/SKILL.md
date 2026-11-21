---
name: crisp-mcp
description: qualitative and mixed-data research analysis using crisp-t mcp tools. use when analyzing interviews, focus groups, field notes, observations, documents, open-ended surveys, or linked numeric data; selecting or applying grounded theory, thematic analysis, reflexive thematic analysis, qualitative content analysis, framework analysis, or mixed methods; and producing traceable computationally assisted qualitative findings via mcp server.
---

# CRISP-T Qualitative and Mixed-Data Analysis via MCP

## Purpose

Use CRISP-T MCP tools to support methodologically explicit qualitative and mixed-data research. Treat computation as an aid to interpretation, not a substitute for qualitative reasoning.

**If the CRISP-T MCP server is not available, inform the user and use crisp-cli skill instead.**

## Start by selecting the methodology

Before substantive analysis, ask the user which methodology to adopt:

1. Constructivist grounded theory (Charmaz default for grounded theory)
2. Thematic analysis
3. Reflexive thematic analysis
4. Qualitative content analysis
5. Framework analysis
6. Mixed-methods analysis
7. Unsure: recommend a method from the research question and intended output

Also confirm the following:

1. Research question.
2. Expected output.
3. Data types.
4. Unit of analysis.
5. Whether data collection is complete.
6. Role of numeric data.
7. If external web or scholarly research is relevant to the analysis, explain why and obtain the user's permission before using it.

Use this decision guide when recommending a method:

- **Constructivist grounded theory**: develop an explanatory account of a process through iterative coding, comparison, memoing, sampling, and category integration.
- **Thematic analysis**: identify and interpret patterned meaning across a dataset.
- **Reflexive thematic analysis**: construct themes through recursive interpretation while treating researcher subjectivity as an analytic resource.
- **Qualitative content analysis**: systematically describe or interpret content using inductive, deductive, or combined categories; counts may support but not replace interpretation.
- **Framework analysis**: compare cases and predefined/emergent dimensions in a matrix, especially for applied, policy, clinical, implementation, or evaluation questions.
- **Mixed methods**: integrate qualitative and numeric evidence for convergence, explanation, expansion, development, sampling, or contradiction analysis.

Declare the selected methodology and intended analytic product in the report. If a complete methodology cannot be followed, describe the work accurately as an exploratory analysis or as using selected techniques.

## Qualitative sensemaking rules

- Read a purposeful, varied sample before broad automation.
- Define the unit of storage, coding, comparison, case, linkage, and inference.
- Keep every accepted code, category, or theme traceable to source excerpts and context.
- Compare cases, groups, sources, time periods, supporting evidence, and negative or deviant cases.
- Write analytic memos throughout; a report is not a substitute for memoing.
- Separate participant accounts, researcher interpretations, computational outputs, numeric findings, and integrated inferences.
- Do not select findings solely by frequency, topic weight, model probability, centrality, sentiment magnitude, or statistical significance.
- Record alternative interpretations, missing perspectives, contradictions, and limitations.

For constructivist grounded theory, use Charmaz's tradition unless the user requests another. Apply initial and focused coding, constant comparison, memo writing, theoretical sampling or documented corpus-based theoretical retrieval, category development, negative-case analysis, theoretical integration, and theoretical sufficiency. Do not present open/axial/selective coding as mandatory for Charmaz.

## Computational guardrails

Automated topics, clusters, keywords, entities, sentiment labels, embeddings, similarities, classifications, and regressions are computational outputs. Use them as candidate patterns, retrieval aids, comparison prompts, sampling aids, negative-case searches, triangulation evidence, or challenges to interpretation.

Do not automatically label computational outputs as qualitative codes, themes, categories, core categories, or theory. A machine suggestion becomes an accepted qualitative code only after reviewing the source in context, considering alternatives, comparing incidents where possible, and documenting the human decision. Preserve output origin and rejected suggestions.

Topic assignment is required when downstream CRISP-T document-level topic analysis depends on it; it is not a methodological requirement for all qualitative research.

## Summary

**Goal:**
Use the CRISP‑T MCP server to run a full, _documented_ qualitative analysis workflow—linking textual narratives and numeric data—using computational triangulation and the selected qualitative methodology.

This skill assumes:

- The **CRISP‑T MCP server** is available as an MCP server called crisp.
- The server exposes several tools for **corpus management, text coding, numeric analysis, triangulation, and export**
- The model can **discover tools at runtime** and adapt to the actual tool list

Use this skill when the user wants to:

- Explore and code qualitative data (e.g., interviews, narratives, open‑ended survey responses)
- Triangulate qualitative codes with quantitative variables
- Produce **transparent, reproducible outputs** (memos, codebooks, matrices, summaries)
- Follow a methodologically explicit qualitative or mixed-methods approach

---

## Intended users and scenarios

- **Qualitative / mixed‑methods researchers** working with text + numeric data

Typical scenarios:

- Coding COVID‑era patient narratives and linking them to open‑source epidemiological data
- Triangulating interview themes with survey scores or EHR‑derived indicators
- Building theory from mixed text–numeric corpora using CRISP‑T stages

---

## MCP server and tools

### Server

- **Server name (example):** `crisp-t`
- **Protocol:** Model Context Protocol (MCP) via stdio (e.g., `crisp-mcp`)

### Required artifact locations

- Use `<repository-root>/workspace/` for every corpus, result JSON, report, and visualization.
- Create it once with `mkdir -p workspace`.
- Keep the unmodified imported corpus in `workspace/corpus/` and reuse it with MCP load tools.
- Use `workspace/reports/` only for human-readable reports.
- Use `workspace/visualizations/` only for generated images or HTML.
- Keep qualitative artifacts under `workspace/qualitative/`, including codes, excerpts, memos, comparisons, categories, sampling decisions, and audit records.
- Intermediate corpora may use descriptive directories under `workspace/`, such as `workspace/corpus_topics/` or `workspace/corpus_assigned/`.
- Do not create top-level `analysis_output`, `results`, or `visualizations` directories unless the user asks.
- If source CSVs are in `./downloads`, import from there but write generated artifacts to `workspace/`.
- Inspect existing filenames before overwriting.

> **Important:**
> Always **discover** the available tools from the CRISP‑T MCP server first, then map them to the workflow below.
> Tool names in this file are **illustrative**; prefer the _actual_ tool names and schemas exposed by the server.

## Staged workflow

### 1. Establish design

- Select and declare the methodology.
- Confirm research question, intended output, data types, cases, analytic units, and integration purpose.
- For mixed methods, state whether the design is QUAL->quan, QUAN->qual, concurrent, embedded, or multiphase and whether integration seeks convergence, explanation, expansion, development, sampling, or contradiction.

### 2. Discover MCP tools and import data

- **List available tools** from the CRISP-T MCP server.
- Identify tools for corpus loading, text operations, numeric operations, triangulation, and export.
- Use corpus loading tools to import data from source directories.
- Mark free-text columns appropriately for text analysis.
- Save the preserved corpus to `workspace/corpus/`.
- If the source folder contains multiple CSV files, warn that only one CSV is supported.

### 3. Verify the corpus

- Use corpus inspection tools to list documents and DataFrame columns.
- Confirm document and row counts, text fields, metadata, identifiers, timestamps, outcome names, missingness, and the relationship between documents and DataFrame rows.

### 4. Conduct qualitative interpretation

- Read a varied sample before modeling.
- Create source-linked codes and analytic memos appropriate to the selected methodology.
- Compare evidence across cases, sources, groups, and time.
- Search for negative cases and alternative explanations.
- Save method decisions and developing findings under `workspace/qualitative/`.

### 5. Run text analysis selectively

- Use topic modeling, keyword extraction, sentiment analysis, NLP, semantic search, graphs, or visualizations for a stated analytic purpose.
- Use topic modeling tools only when a broad first pass is appropriate.
- For topic data, run topic modeling, then run clear cache and assign topics as a separate explicit step when document-level topic assignment is needed.
- Validate every meaningful computational pattern against source excerpts.

### 6. Run numeric analysis

- Inspect columns before modeling.
- Always specify outcome variable for supervised ML tasks.
- Use include and ignore parameters to keep features intentional.
- Explore numeric data with summary statistics, classification, clustering, regression, association, visualizations, or TDA as appropriate.
- Do not treat statistical association as qualitative meaning or causation.

### 7. Link and integrate text and numeric data

- Confirm the text-to-row relationship before linkage.
- Use relationship tools for theory-driven or validated manual connections.
- Use linkage tools with appropriate aggregation when linking text metadata to numeric outcomes.
- Supported linkage methods include `id`, `embedding`, `temporal`, and `keyword`.
- Supported aggregation strategies include `majority`, `mean`, `first`, and `mode`.
- State why data are linked, what the linkage unit is, and how aggregation changes interpretation.
- Perform cross-modal analysis when justified, including text features predicting numeric outcomes or numeric features contextualizing text outcomes.

### 8. Visualize and report

- Generate visualizations using MCP visualization tools, saving to `workspace/visualizations/`.
- Write the evolving and final report under `workspace/reports/`.
- Document corpus path, methodology, analytic units, tools used, parameters, filters, linkage, computational provenance, qualitative findings, numeric findings, integrated inferences, contradictions, and limitations.
- Treat visualizations as analytic aids, not findings by themselves.

### Expected tool capabilities (illustrative)

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

- **Research question and methodology**
  - `research_question`: short description of what they want to understand or explain
  - `methodology`: grounded theory, thematic analysis, reflexive thematic analysis, qualitative content analysis, framework analysis, or mixed methods
  - `expected_output`: report, theory, themes, categories, or other analytic product
- **Data types and units**
  - `data_types`: interviews, focus groups, field notes, documents, surveys, or mixed
  - `unit_of_analysis`: individual, group, case, document, or other unit
  - `data_collection_complete`: whether data collection is finished
  - `role_of_numeric_data`: how numeric data will be used in the analysis
- **Project / dataset selection**
  - `project_name` or `project_id`
  - `text_source_pattern` (e.g., “patient*narratives”, “interviews*\*”)
  - `numeric_dataset_name` (e.g., “covid_metrics.csv”, “survey_scores”)
- **Methodological preferences**
  - `coding_style`: initial coding, focused coding, or using an existing codebook
  - `triangulation_focus`: e.g., “compare themes by age group”, “link narratives to hospitalization status”
- **Output preferences**
  - `output_format`: Markdown, CSV, JSON, or mixed
  - `audience`: “methods‑savvy colleague”, “student”, “grant reviewer”, etc.

If the user is vague, default to:

- Initial coding → focused coding
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
2. **Select the methodology** using the decision guide above.
3. If unclear, ask **one or two** focused questions:
   - **Example:** "Which is more important right now: exploring themes, or linking themes to numeric outcomes?"
4. Record the clarified research question and methodology in the analytic memo.

### 2. Discover CRISP‑T MCP tools and project context

1. **List available tools** from the `crisp-t` MCP server.
2. Identify tools that correspond to:
   - Corpus loading and management
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

1. Use corpus loading tools to import data from source directories.
2. Use corpus inspection tools to:
   - **List documents** (e.g., narrative collections, interview sets)
   - **List DataFrame columns** (e.g., survey tables, EHR‑derived tables)
3. For each candidate source:
   - Fetch a **small sample** (e.g., a few narratives, a few rows of numeric data)
   - Briefly describe:
     - What the data represent
     - How they relate to the research question
4. Confirm with the user which sources to prioritize.
5. Confirm document and row counts, text fields, metadata, identifiers, timestamps, outcome names, missingness, and the relationship between documents and DataFrame rows.

Update the memo with:

- Selected text sources and why
- Selected numeric datasets and why

### 4. Initial qualitative interpretation and coding

1. Select a **manageable subset** of text (e.g., 10–20 narratives or segments).
2. **Read the sample** before running computational analyses.
3. Apply **initial coding** using CRISP‑T coding tools or manual annotation:
   - Generate **short, grounded codes** that stay close to the data
   - Avoid premature theory; focus on what participants are saying/doing/experiencing
4. Use MCP tools to:
   - Save codes to the project
   - Attach codes to segments
   - Retrieve and refine the emerging code list

In the memo:

- Describe the coding approach (e.g., initial coding, line‑by‑line, incident‑focused)
- Provide **examples**: quote → code → brief interpretation
- Save codes and memos under `workspace/qualitative/`

### 5. Focused coding and category development

1. Review the emerging code list using CRISP‑T tools (e.g., list codes, code frequencies).
2. Group related codes into **higher‑level categories** or themes:
   - Merge overlapping codes
   - Distinguish closely related but conceptually different codes
3. If the server supports it, create or update:
   - A **codebook** (code name, definition, inclusion/exclusion criteria, example quotes)
   - **Category structures** (e.g., parent/child relationships)
4. **Compare cases** across groups, sources, time periods, and negative cases.

In the memo:

- Describe key categories and how they were derived
- Note any tensions, ambiguities, or alternative interpretations
- Document sampling decisions and theoretical retrieval strategies

### 6. Computational text analysis (as analytic aid)

Using CRISP‑T tools for text analysis:

1. **Topic modeling** (when appropriate):
   - Run topic modeling with a reasonable number of topics (3-5 for testing).
   - Clear cache and assign topics to documents as a separate step.
   - Validate topic coherence against source excerpts.
2. **Sentiment analysis** (as labeled computational evidence):
   - Run sentiment analysis and treat results as computational outputs.
   - Do not treat sentiment scores as qualitative findings without validation.
3. **Semantic search and keyword extraction**:
   - Use for retrieval, sampling, or negative-case searches.
   - Validate extracted patterns against source context.

In the memo:

- Document which computational tools were used and why
- Note how computational outputs informed or challenged qualitative interpretation
- Preserve rejected computational suggestions and alternative interpretations

### 7. Numeric analysis and triangulation

Using CRISP‑T tools that link text and numeric data:

1. **Inspect numeric columns** before modeling.
2. **Identify the linkage key** (e.g., participant ID, case ID, encounter ID).
3. For each case or group:
   - Retrieve **coded qualitative data** (codes, categories, or themes)
   - Retrieve **numeric variables** (e.g., age, scores, outcomes, counts)
4. Construct **triangulation views**, such as:
   - Code frequencies by group (e.g., theme prevalence by age band or outcome)
   - Associations between codes and numeric indicators (e.g., high vs. low scores)
   - Patterns of convergence (where qualitative and quantitative tell a similar story)
   - Patterns of divergence (where they conflict or complicate each other)

When tools support it, generate:

- **Triangulation matrices** (e.g., codes × variables, cases × themes)
- **Summary statistics** (e.g., counts, proportions, simple associations)

6. Use linkage tools with appropriate aggregation when linking text metadata to numeric outcomes:
   - Supported linkage methods include `id`, `embedding`, `temporal`, and `keyword`.
   - Supported aggregation strategies include `majority`, `mean`, `first`, and `mode`.

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

### 9. Export and documentation

1. Use CRISP‑T export tools to save:
   - Codebook(s)
   - Triangulation matrices
   - Any generated summaries or tables
2. If supported, export:
   - The analytic memo as Markdown or another requested format
   - A machine‑readable representation of the analysis (e.g., JSON snapshot)
3. Save visualizations to `workspace/visualizations/`.
4. Save reports to `workspace/reports/`.
5. Document corpus path, methodology, analytic units, tools used, parameters, filters, linkage, computational provenance, qualitative findings, numeric findings, integrated inferences, contradictions, and limitations.

Provide the user with:

- A **human‑readable summary** (memo)
- A **list of exported artifacts** and where they are stored (paths, IDs, or URLs)

---

## Model behavior and style guidelines

When using this skill, you should:

- **Be methodologically explicit**
  - Name the selected methodology and what you are doing (initial coding, focused coding, triangulation, memoing)
  - Briefly justify key analytic decisions
  - Declare the methodology in the report
- **Stay grounded in the data**
  - Use short quotes and concrete examples
  - Avoid over‑generalizing beyond what the data support
  - Read a varied sample before broad automation
- **Treat computation as an aid**
  - Do not automatically label computational outputs as qualitative codes, themes, or theory
  - Validate computational patterns against source excerpts
  - Document rejected computational suggestions
- **Respect the user's context**
  - If they are teaching, highlight steps and rationale for students
  - If they are writing a paper or grant, emphasize clarity and traceability
- **Limit questions**
  - Ask at most **one or two** focused questions at a time
  - Default to reasonable assumptions when the user is not specific
- **Follow artifact conventions**
  - Use `workspace/` for all generated artifacts
  - Keep `workspace/corpus/` unchanged
  - Save qualitative artifacts under `workspace/qualitative/`

---

## Limitations and cautions

- Triangulation results should be treated as **aids to interpretation**, not definitive proof.
- Computational outputs (topics, sentiment, embeddings, etc.) are **candidate patterns**, not qualitative findings.
- Always encourage the user to:
  - Review codes and categories
  - Challenge interpretations
  - Consider ethical and contextual factors not visible in the data
- Do not select findings solely by frequency, topic weight, model probability, or statistical significance.
- Record alternative interpretations, missing perspectives, contradictions, and limitations.

## Error handling

| Error                           | Likely cause                    | Response                                              |
| ------------------------------- | ------------------------------- | ----------------------------------------------------- |
| `Cache error before assignment` | Cache from a previous run       | Use clear cache tools before assignment               |
| `Outcome not found`             | Incorrect column or field name  | Verify with corpus inspection tools                   |
| `ML features mismatch`          | Features changed after training | Clear cache and retrain                               |
| `Linkage failed`                | Insufficient data or metadata   | Verify IDs/timestamps or use a simpler linkage method |
| `Visualization empty`           | Required analysis was not run   | Run the relevant analysis step first                  |

## Performance notes

- **Large corpora (1000+ documents)**: Test with filters and smaller samples, then expand deliberately.
- **Topic modeling**: Use a smaller number of topics while testing; 3-5 topics is a practical initial diagnostic.
- **Graph/visualization tools**: Save intermediate results because these are more expensive.
- **Semantic search**: Initialization may make the first run slower.
- **ML training**: Sample or filter large datasets before full training.
- **Clear cache** if results appear stale or features changed.

## High-signal operating rules

- Work step by step; do not run unrelated analyses together.
- Save meaningful intermediate results to appropriate workspace directories.
- Preserve the original corpus unchanged.
- Use corpus filters and sampling while testing.
- Inspect metadata and relationships regularly.
- If a step is slow, reduce data with filters or sampling.
- If a step fails, report the exact error, try a simpler neighboring analysis, and continue when methodologically defensible.
- Generate the report as analysis proceeds, but keep analytic memos separate from the final narrative.

---
