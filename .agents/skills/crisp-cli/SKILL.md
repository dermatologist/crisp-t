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

* If the command is not found in your environment, try prefixing with `uv run`
* If it fails use the python environment in .venv folder, if available.
* If it still fails, ensure CRISP-T is installed: `pip install crisp-t[ml]`

## Understand and INTERNALIZE the options and flags for each command by using `--help`. DO THIS BEFORE RUNNING ANY ANALYSES.

crisp --help
crispt --help
crispviz --help

## Tips for Effective Use
* **Use ./workspace** as the working directory for all artifacts, if not explicitly specified. Create it if it does not exist.
* **Always start with `--source`** to import data into a corpus structure, if not already done with `--out corpus`. If the corpus folder already exists, use `--inp corpus` to load it for analysis.
* **Use `--unstructured`** to specify free-text columns in CSV files. If only a single CSV file is present in the source folder, it will be automatically loaded as a corpus with the specified unstructured columns. If free-text columns are not specified, GUESS which columns are text based on content.
* **Limit dataset size** during testing with `--num` (documents) and `--rec` (rows)
* **Use `--clear`** when switching datasets or modifying filters
* **Use --assign** after `--topics` to assign documents to topics always. REMEMBER to use `--clear` before `--assign` if the corpus or filters have changed. THIS STEP MUST be done for TEXT DATA.
* **Combine `--nlp`** to run all text analyses at once. NOTE: May be slow for large corpora.
* **For ML tasks**, always specify the outcome variable with `--outcome`
* **Use `--linkage`** to connect text metadata to numeric outcomes in ML analyses
* **Do cross-modal linkage** when needed with `--linkage`.
* **Use `--aggregation`** to define how to combine multiple documents for a single outcome
* **Use `--include` and `--ignore`** to control features used in ML analyses
* **Use `crispt`** to manage corpus structure, add/remove documents, and define relationships
* **Use `crispviz`** to generate visualizations after analysis steps
* **Save intermediate results** using `--out` at each major step
* **Use filtering** (`--filters`) to analyze subsets.
* **Link early**: Add relationships after text analysis for mixed-methods validation
* **Visualize often**: Use `crispviz` after each major analysis step
* **Check metadata**: Use `crispt --print` to inspect corpus structure

## Important Guidelines for Agents
* Perform multi-step workflows STEP-BY-STEP, saving intermediate results with `--out` for analytical flexibility.
* Do not run all analyses at once; break into smaller steps to isolate issues.
* If analysis results seem off, clear cache with `--clear` before re-running.
* If a particular analysis fails or takes too long, try reducing dataset size with filters or `--num` or `--rec` or both.
* If errors persist or if it still takes too long, skip the step and proceed to the next analysis.
* Document level TOPIC assignment using `--assign` is a VERY important step different from just running `--topics`. THIS STEP MUST be for TEXT DATA.
* Generate a report as you go, documenting insights from each step.
* If the source folder contains multiple CSV files, warn the user that only one CSV file is supported.
* All tools generate tips and warnings in the console output. Pay attention to these messages for guidance on next steps and troubleshooting.

## Important steps
* Import data into CRISP-T corpus and dataframe.
* Perform linking between text and numeric data using various methods (id based, keyword based, time based, embedding based).
* Explore text data using various methods (e.g., topic modeling, keyword extraction, sentiment analysis, visualizations).
* Explore numeric data using various methods (e.g., summary statistics, classification, clustering, regression, association, visualizations, TDA, etc.).
* Perform cross modal analysis using linked text and numeric data (e.g., text features as predictors for numeric outcomes, numeric features as predictors for text outcomes, etc.).
* Add manual connections between text documents and numeric rows if needed to support theory driven analysis.
* Derive insights from the analysis and document them.

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

| Error | Cause | Solution |
|-------|-------|----------|
| `Cache error before --assign` | Cache from previous run | Use `--clear` flag |
| `Outcome not found` | Wrong column/field name | Use `crispt --df-cols` or `crispt --print` to verify |
| `ML features mismatch` | Features changed after training | Clear cache and retrain |
| `Linkage failed` | Insufficient data/metadata | Verify timestamps or use simpler linkage method |
| `Visualization empty` | Analysis not run | Ensure `--topics`, `--tdabm`, or `--graph` completed first |

## Performance Notes

- **Large corpora** (1000+ docs): Use `--num` to limit imports, use filters
- **Topic modeling**: Adjust `--num` lower for faster processing (3-5 recommended)
- **TDABM/graphs**: More expensive; save intermediate results
- **Semantic search**: Requires initialization; slower on first run
- **ML training**: Very slow on large datasets; use sampling/filtering

## Examples
```
crisp --source data_folder --out corpus
crisp --inp corpus --clear --assign --out results_v2
crisp --inp results_v2 --clear --filters "region=North" --assign --out results_filtered
crisp --inp results_filtered --regression --outcome satisfaction_score --include age,income --out results_v3
crisp --inp corpus --filters "embedding:text" --filters "temporal:df" --sentiment
```