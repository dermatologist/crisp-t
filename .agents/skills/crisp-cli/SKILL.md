---
name: crisp-cli
description: Skill for qualitative mixed data research analysis using CRISP-T command-line tools.
---

## Overview

This skill enables agents to perform qualitative mixed data (text and numeric) research analysis using **CRISP-T command-line tools**.

## Core Commands

### Three main CLI commands are available:

- **`crisp`** - Main analysis engine (text/NLP, ML, visualization workflows)
- **`crispt`** - Corpus management (document manipulation, semantic search, relationships)
- **`crispviz`** - Visualization generation (charts, word clouds, graphs, LDA)

* If the command is not found in your environment, prefix it with `uv run`.
* If it fails, use the project environment with `uv run` or `.venv/bin/crisp`.
* If it is not installed, run `uv pip install -e ".[ml,xg,dev]"`.

## Required Artifact Location

- Use `<repository-root>/workspace/` for every corpus, result JSON, report, and visualization.
- Create `workspace/` once if it does not exist: `mkdir -p workspace`.
- Never create a separate `analysis_output`, `results`, or `visualizations` folder unless the user explicitly asks for it.
- Keep the main corpus in `workspace/corpus/` and reuse that path for every later `--inp` call.
- Use `workspace/reports/` only for human-readable reports and `workspace/visualizations/` only for generated images or HTML. Pass these directories with `--out` when a command supports it.
- If source CSVs live in `./downloads`, import from `./downloads` but write all generated artifacts to `workspace/`.
- Existing files may be reused; inspect their names before overwriting them.

```bash
mkdir -p workspace/reports workspace/visualizations
uv run crisp --source downloads --unstructured review --out workspace/corpus
```

Before running analyses, internalize the options for the active command:

```bash
uv run crisp --help
uv run crispt --help
uv run crispviz --help
```

## Efficient Workflow

Use a staged workflow rather than running unrelated analyses together.

1. **Import once**
   - Use `--source` for a new source and `--unstructured <column>` for every free-text column.
   - Save the corpus to `workspace/corpus/`.
   - If the corpus already exists, reuse `workspace/corpus/` with `--inp`.

2. **Verify the corpus**
   - Inspect document and DataFrame structure before choosing outcomes or features.
   - Use `uv run crispt --inp workspace/corpus --print` and `uv run crispt --inp workspace/corpus --df-cols --print`.
   - Confirm row counts, text metadata, and outcome names.

3. **Run text analysis**
   - For topic data, run `--topics`, then always run `--clear --assign` in a separate step.
   - Use `--nlp` only when the corpus is manageable and a broad first-pass analysis is appropriate.
   - Save meaningful stage results under `workspace/corpus_topics/` or `workspace/results/`.

4. **Run numeric analysis**
   - Always specify `--outcome` for ML tasks.
   - Verify the outcome with `--df-cols` before regression or classification.
   - Use `--include` and `--ignore` to keep features intentional.

5. **Integrate text and numeric findings**
   - Add relationships with `crispt --add-rel` for theory-driven or validated connections.
   - Use `--linkage` plus `--aggregation` when text metadata is being linked to numeric outcomes.
   - Do not attempt linkage until the text field and DataFrame relationship are confirmed.

6. **Visualize and report**
   - Generate visualizations with `crispviz --out workspace/visualizations/`.
   - Write the final report under `workspace/reports/`.
   - Document the corpus path, methods, limitations, and key cross-modal findings.

### High-Signal Rules

- Use `--num` for document/topic limits and `--rec` for row/result limits while testing.
- Use `--clear` before `--assign` after changing corpora or filters.
- Run `--topics` and `--assign` as separate, explicit steps; topic assignment is mandatory for document-level text analysis.
- Use `--filters` to analyze subsets, but first verify the filter field exists.
- Use `crispt --relationships` and `crispt --print` to inspect metadata and explicit links.
- Save intermediate corpora under `workspace/`, not beside source files or in ad hoc folders.
- Preserve at least one unmodified imported corpus at `workspace/corpus/`.
- If a step fails or produces no useful result, report the exact error and continue with a simpler neighboring analysis.

## Important Guidelines for Agents

- Perform multi-step workflows STEP-BY-STEP, saving intermediate results with `--out` for analytical flexibility.
- Do not run all analyses at once; break into smaller steps to isolate issues.
- If analysis results seem off, clear cache with `--clear` before re-running.
- If a particular analysis fails or takes too long, try reducing dataset size with filters or `--num` or `--rec` or both.
- If errors persist or if it still takes too long, skip the step and proceed to the next analysis.
- Document level TOPIC assignment using `--assign` is a VERY important step different from just running `--topics`. THIS STEP MUST be for TEXT DATA.
- Generate a report as you go, documenting insights from each step.
- If the source folder contains multiple CSV files, warn the user that only one CSV file is supported.
- All tools generate tips and warnings in the console output. Pay attention to these messages for guidance on next steps and troubleshooting.

## Important steps

- Import data into CRISP-T corpus and dataframe.
- Perform linking between text and numeric data using various methods (id based, keyword based, time based, embedding based).
- Explore text data using various methods (e.g., topic modeling, keyword extraction, sentiment analysis, visualizations).
- Explore numeric data using various methods (e.g., summary statistics, classification, clustering, regression, association, visualizations, TDA, etc.).
- Perform cross modal analysis using linked text and numeric data (e.g., text features as predictors for numeric outcomes, numeric features as predictors for text outcomes, etc.).
- Add manual connections between text documents and numeric rows if needed to support theory driven analysis.
- Derive insights from the analysis and document them.

## Key Concepts for Agents

### Corpus Structure

- **Documents**: Text entries (interviews, field notes, etc.)
- **DataFrame**: Numeric data (age, income, survey responses, etc.)
- **Relationships**: Explicit links between text findings and numeric variables
- **Metadata**: Tags, timestamps, source information

### Linkage Methods

- **id**: Direct document-to-row matching by ID
- **embedding**: Semantic similarity-based linking
- **temporal**: Time-based linking (nearest, window, sequence)
- **keyword**: Linking via extracted keywords/topics

### Aggregation Strategies

- **majority**: Most common value (classification)
- **mean**: Average value (regression)
- **first**: First value encountered
- **mode**: Most frequent value

### Important Flags

- `--clear`: Always use before `--assign` if filters/data changed
- `--linkage`: Required when outcome is a text field
- `--unstructured`: Mark free-text columns in CSV for proper analysis

### File Formats

- **Corpus files**: `corpus.json` + `corpus_df.csv` (created in `--out` folder)
- **Visualizations**: PNG/HTML (saved to `--out` folder)
- **Metadata**: Embedded in corpus.json (view with `--print`)

## Error Handling

| Error                         | Cause                           | Solution                                                   |
| ----------------------------- | ------------------------------- | ---------------------------------------------------------- |
| `Cache error before --assign` | Cache from previous run         | Use `--clear` flag                                         |
| `Outcome not found`           | Wrong column/field name         | Use `crispt --df-cols` or `crispt --print` to verify       |
| `ML features mismatch`        | Features changed after training | Clear cache and retrain                                    |
| `Linkage failed`              | Insufficient data/metadata      | Verify timestamps or use simpler linkage method            |
| `Visualization empty`         | Analysis not run                | Ensure `--topics`, `--tdabm`, or `--graph` completed first |

## Performance Notes

- **Large corpora** (1000+ docs): Use `--num` to limit imports, use filters
- **Topic modeling**: Adjust `--num` lower for faster processing (3-5 recommended)
- **TDABM/graphs**: More expensive; save intermediate results
- **Semantic search**: Requires initialization; slower on first run
- **ML training**: Very slow on large datasets; use sampling/filtering

## Examples

```bash
# Create the analysis directories and import one source CSV.
mkdir -p workspace/reports workspace/visualizations
uv run crisp --source downloads --unstructured review --out workspace/corpus

# Verify corpus structure and available numeric columns.
uv run crispt --inp workspace/corpus --print
uv run crispt --inp workspace/corpus --df-cols

# Run topic analysis and create document-level assignments.
uv run crisp --inp workspace/corpus --topics --num 5 --out workspace/corpus_topics
uv run crisp --inp workspace/corpus_topics --clear --assign --out workspace/corpus_assigned

# Run sentiment and numeric analysis using the assigned corpus.
uv run crisp --inp workspace/corpus_assigned --sentiment --out workspace/corpus_text_results
uv run crisp --inp workspace/corpus_assigned --regression --outcome rating --include 100g_USD --out workspace/regression_results

# Save visuals and reports beneath workspace/.
uv run crispviz --inp workspace/corpus_assigned --out workspace/visualizations --wordcloud --ldavis
# Write the human-readable report to workspace/reports/ANALYSIS_REPORT.md.

# Filter a subset only after confirming that "region" is a DataFrame column.
uv run crisp --inp workspace/corpus_assigned --clear --filters "region=North" --assign --out workspace/north_region
```

For a CSV with a text column named `review`, the shortest complete workflow is:

```bash
mkdir -p workspace/reports workspace/visualizations
uv run crisp --source downloads --unstructured review --out workspace/corpus
uv run crispt --inp workspace/corpus --print
uv run crisp --inp workspace/corpus --topics --num 5 --out workspace/corpus_topics
uv run crisp --inp workspace/corpus_topics --clear --assign --sentiment --out workspace/corpus_assigned
uv run crispviz --inp workspace/corpus_assigned --out workspace/visualizations --wordcloud --ldavis
```

Do not leave generated artifacts in `analysis_output/`, the source folder, or a new folder named after a one-off analysis.
