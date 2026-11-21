---
name: crisp-cli
description: qualitative and mixed-data research analysis using crisp-t command-line tools. use when analyzing interviews, focus groups, field notes, observations, documents, open-ended surveys, or linked numeric data; selecting or applying grounded theory, thematic analysis, reflexive thematic analysis, qualitative content analysis, framework analysis, or mixed methods; and producing traceable computationally assisted qualitative findings.
---

# CRISP-T Qualitative and Mixed-Data Analysis

## Purpose

Use CRISP-T to support methodologically explicit qualitative and mixed-data research. Treat computation as an aid to interpretation, not a substitute for qualitative reasoning.

## Start by selecting the methodology

Before substantive analysis, ask the user which methodology to adopt:

1. Constructivist grounded theory (Charmaz default for grounded theory)
2. Thematic analysis
3. Reflexive thematic analysis
4. Qualitative content analysis
5. Framework analysis
6. Mixed-methods analysis
7. Unsure: recommend a method from the research question and intended output

Also confirm the research question, expected output, data types, unit of analysis, whether data collection is complete, and the role of numeric data. If external web or scholarly research may help, explain why and obtain the user's permission before using it.

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

## Core commands

Three CLI commands are available:

- **`crisp`**: Main analysis engine for text/NLP, ML, and visualization workflows.
- **`crispt`**: Corpus management, document manipulation, semantic search, and relationships.
- **`crispviz`**: Charts, word clouds, graphs, and LDA visualizations.

If a command is not found, prefix it with `uv run`. If that fails, use the project environment with `uv run` or `.venv/bin/crisp`. If CRISP-T is not installed, run:

```bash
uv pip install -e ".[ml,xg,dev]"
```

Before analysis, inspect the active command options:

```bash
uv run crisp --help
uv run crispt --help
uv run crispviz --help
```

Pay attention to all CLI tips and warnings.

## Required artifact locations

- Use `<repository-root>/workspace/` for every corpus, result JSON, report, and visualization.
- Create it once with `mkdir -p workspace`.
- Keep the unmodified imported corpus in `workspace/corpus/` and reuse it with `--inp`.
- Use `workspace/reports/` only for human-readable reports.
- Use `workspace/visualizations/` only for generated images or HTML.
- Keep qualitative artifacts under `workspace/qualitative/`, including codes, excerpts, memos, comparisons, categories, sampling decisions, and audit records.
- Intermediate corpora may use descriptive directories under `workspace/`, such as `workspace/corpus_topics/` or `workspace/corpus_assigned/`.
- Do not create top-level `analysis_output`, `results`, or `visualizations` directories unless the user asks.
- If source CSVs are in `./downloads`, import from there but write generated artifacts to `workspace/`.
- Inspect existing filenames before overwriting.

## Staged workflow

### 1. Establish design

- Select and declare the methodology.
- Confirm research question, intended output, data types, cases, analytic units, and integration purpose.
- For mixed methods, state whether the design is QUAL->quan, QUAN->qual, concurrent, embedded, or multiphase and whether integration seeks convergence, explanation, expansion, development, sampling, or contradiction.

### 2. Import once

- Use `--source` for a new source.
- Use `--unstructured <column>` for every free-text CSV column.
- Save the preserved corpus to `workspace/corpus/`.
- If the source folder contains multiple CSV files, warn that only one CSV is supported.

### 3. Verify the corpus

```bash
uv run crispt --inp workspace/corpus --print
uv run crispt --inp workspace/corpus --df-cols --print
```

Confirm document and row counts, text fields, metadata, identifiers, timestamps, outcome names, missingness, and the relationship between documents and DataFrame rows.

### 4. Conduct qualitative interpretation

- Read a varied sample before modeling.
- Create source-linked codes and analytic memos appropriate to the selected methodology.
- Compare evidence across cases, sources, groups, and time.
- Search for negative cases and alternative explanations.
- Save method decisions and developing findings under `workspace/qualitative/`.

### 5. Run text analysis selectively

- Use topic modeling, keyword extraction, sentiment analysis, NLP, semantic search, graphs, or visualizations for a stated analytic purpose.
- Use `--nlp` only when the corpus is manageable and a broad first pass is appropriate.
- For topic data, run `--topics`, then run `--clear --assign` as a separate explicit step when document-level topic assignment is needed.
- Use `--clear` before `--assign` whenever corpora, data, or filters change.
- Validate every meaningful computational pattern against source excerpts.

### 6. Run numeric analysis

- Inspect columns before modeling.
- Always specify `--outcome` for supervised ML tasks.
- Use `--include` and `--ignore` to keep features intentional.
- Explore numeric data with summary statistics, classification, clustering, regression, association, visualizations, or TDA as appropriate.
- Do not treat statistical association as qualitative meaning or causation.

### 7. Link and integrate text and numeric data

- Confirm the text-to-row relationship before linkage.
- Use `crispt --add-rel` for theory-driven or validated manual connections.
- Use `--linkage` plus `--aggregation` when linking text metadata to numeric outcomes.
- Supported linkage methods include `id`, `embedding`, `temporal`, and `keyword`.
- Supported aggregation strategies include `majority`, `mean`, `first`, and `mode`.
- State why data are linked, what the linkage unit is, and how aggregation changes interpretation.
- Perform cross-modal analysis when justified, including text features predicting numeric outcomes or numeric features contextualizing text outcomes.

### 8. Visualize and report

- Generate visualizations with `crispviz --out workspace/visualizations/`.
- Write the evolving and final report under `workspace/reports/`.
- Document corpus path, methodology, analytic units, commands, parameters, filters, linkage, computational provenance, qualitative findings, numeric findings, integrated inferences, contradictions, and limitations.
- Treat visualizations as analytic aids, not findings by themselves.

## High-signal operating rules

- Work step by step; do not run unrelated analyses together.
- Save meaningful intermediate results with `--out` under `workspace/`.
- Preserve `workspace/corpus/` unchanged.
- Use `--num` for document/topic limits and `--rec` for row/result limits while testing.
- Use `--filters` only after verifying that the field exists.
- Use `crispt --relationships` and `crispt --print` to inspect metadata and explicit links.
- Clear cache with `--clear` if results appear stale or features changed.
- If a step is slow, reduce data with filters, `--num`, or `--rec`.
- If a step fails, report the exact error, try a simpler neighboring analysis, and continue when methodologically defensible.
- Generate the report as analysis proceeds, but keep analytic memos separate from the final narrative.

## Key concepts and flags

### Corpus structure

- **Documents**: Interviews, focus groups, field notes, observations, documents, open-ended responses, or other text entries.
- **DataFrame**: Numeric or structured attributes such as age, income, ratings, survey responses, or outcomes.
- **Relationships**: Explicit links among documents, text findings, and numeric rows or variables.
- **Metadata**: Tags, timestamps, source information, case identifiers, and analytic provenance.

### Linkage methods

- **`id`**: Direct document-to-row matching by identifier.
- **`embedding`**: Semantic-similarity linkage.
- **`temporal`**: Time-based linkage using nearest, window, or sequence logic.
- **`keyword`**: Linkage through extracted keywords or topics.

### Aggregation strategies

- **`majority`**: Most common class for classification.
- **`mean`**: Average value for regression.
- **`first`**: First encountered value.
- **`mode`**: Most frequent value.

### Important flags

- **`--clear`**: Use before `--assign` if filters or data changed.
- **`--linkage`**: Required when the outcome is a text field.
- **`--unstructured`**: Mark every free-text CSV column for proper analysis.

### File formats

- **Corpus**: `corpus.json` and `corpus_df.csv` in the selected `--out` directory.
- **Visualizations**: PNG or HTML in the selected visualization directory.
- **Metadata**: Embedded in `corpus.json`; inspect with `--print`.

## Error handling

| Error | Likely cause | Response |
|---|---|---|
| `Cache error before --assign` | Cache from a previous run | Use `--clear` |
| `Outcome not found` | Incorrect column or field name | Verify with `crispt --df-cols` or `crispt --print` |
| `ML features mismatch` | Features changed after training | Clear cache and retrain |
| `Linkage failed` | Insufficient data or metadata | Verify IDs/timestamps or use a simpler linkage method |
| `Visualization empty` | Required analysis was not run | Run the relevant `--topics`, `--tdabm`, or `--graph` step first |

## Performance notes

- **Large corpora (1000+ documents)**: Test with `--num` and filters, then expand deliberately.
- **Topic modeling**: Use a smaller `--num` while testing; 3-5 topics is a practical initial diagnostic, not a methodological default.
- **TDABM/graphs**: Save intermediate results because these are more expensive.
- **Semantic search**: Initialization may make the first run slower.
- **ML training**: Sample or filter large datasets before full training.

## Example workflow

```bash
# Create directories and import one CSV.
mkdir -p workspace/reports workspace/visualizations workspace/qualitative
uv run crisp --source downloads --unstructured review --out workspace/corpus

# Verify corpus and numeric columns.
uv run crispt --inp workspace/corpus --print
uv run crispt --inp workspace/corpus --df-cols --print

# After reading and initial qualitative interpretation, run topic analysis.
uv run crisp --inp workspace/corpus --topics --num 5 --out workspace/corpus_topics
uv run crisp --inp workspace/corpus_topics --clear --assign --out workspace/corpus_assigned

# Use sentiment and numeric analysis as labeled computational evidence.
uv run crisp --inp workspace/corpus_assigned --sentiment --out workspace/corpus_text_results
uv run crisp --inp workspace/corpus_assigned --regression --outcome rating --include 100g_USD --out workspace/regression_results

# Generate visualizations.
uv run crispviz --inp workspace/corpus_assigned --out workspace/visualizations --wordcloud --ldavis

# Filter only after verifying the field.
uv run crisp --inp workspace/corpus_assigned --clear --filters "region=North" --assign --out workspace/north_region
```

For a CSV with a free-text column named `review`, the shortest CRISP-T computational workflow is:

```bash
mkdir -p workspace/reports workspace/visualizations workspace/qualitative
uv run crisp --source downloads --unstructured review --out workspace/corpus
uv run crispt --inp workspace/corpus --print
uv run crisp --inp workspace/corpus --topics --num 5 --out workspace/corpus_topics
uv run crisp --inp workspace/corpus_topics --clear --assign --sentiment --out workspace/corpus_assigned
uv run crispviz --inp workspace/corpus_assigned --out workspace/visualizations --wordcloud --ldavis
```

This is not, by itself, a complete qualitative methodology. Interpret and validate outputs according to the method selected with the user. Do not leave artifacts in `analysis_output/`, the source folder, or an ad hoc top-level directory.
