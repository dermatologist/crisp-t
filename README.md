# 🔍 CRISP-T (**CRoss** **I**ndustry **S**tandard **P**rocess for **T**riangulation)

**Work in progress.** See the older version [here](https://github.com/dermatologist/crisp-t)

Qualitative research involves the collection and analysis of textual data, such as interview transcripts, open-ended survey responses, and field notes. It is often used in social sciences, humanities, and health research to explore complex phenomena and understand human experiences. In addition to textual data, qualitative researchers may also collect quantitative data, such as survey responses or demographic information, to complement their qualitative findings.

Qualitative research is often characterized by its inductive approach, where researchers aim to generate theories or concepts from the data rather than testing pre-existing hypotheses. This process is known as Grounded Theory, which emphasizes the importance of data-driven analysis and theory development.

We are developing a framework that integrates **textual data** (as a list of documents) and **numeric data** (as Pandas DataFrame) into structured classes that retain **metadata** from various analytical processes, such as **topic modeling** and **decision trees**. Researchers, with or without GenAI assistance, can define relationships between textual and numerical datasets based on their chosen **theoretical lens**.  A final analytical phase ensures that proposed relationships actually hold true.

[![crisp-t](https://github.com/dermatologist/crisp-t/blob/develop/notes/arch.drawio.svg)](https://github.com/dermatologist/crisp-t/blob/develop/notes/arch.drawio.svg)

## Installation

```bash
pip install crisp-t
```

For machine learning features:
```bash
pip install crisp-t[ml]
```

You'll also need to download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```


## Command Line Scripts

CRISP-T now provides three main command-line scripts:

- `crisp` — Main CLI for qualitative triangulation and analysis (see below)
- `crispviz` — Visualization CLI for corpus data (word frequencies, topic charts, wordclouds, etc.)
- `crispt` — Corpus manipulation CLI (create, edit, query, and manage corpus objects)

All scripts are installed as entry points and can be run directly from the command line after installation.

### crisp (Triangulation CLI)

```bash
crisp [OPTIONS]
```

#### Input/Output Options

- `--inp, -i PATH`: Load an existing corpus from a folder containing `corpus.json` (and optional `corpus_df.csv`)
- `--out, -o PATH`: When saving the corpus, provide a folder path; the CLI writes `corpus.json` (and `corpus_df.csv` if available) into that folder. When saving analysis results (topics, sentiment, etc.), this acts as a base path: files are written with suffixes, e.g., `results_topics.json`.
- `--unstructured, -t TEXT`: Text CSV column(s) to analyze/compare (can be used multiple times)
- `--ignore TEXT`: Comma-separated words to ignore during ingestion (applies to `--source/--sources`)

#### Analysis Options

- `--codedict`: Generate qualitative coding dictionary
- `--topics`: Generate topic model using LDA
- `--assign`: Assign documents to topics
- `--cat`: List categories of entire corpus or individual documents
- `--summary`: Generate extractive text summary
- `--sentiment`: Generate sentiment scores using VADER
- `--sentence`: Generate sentence-level scores when applicable
- `--nlp`: Generate all NLP reports (combines above text analyses)
- `--nnet`, `--cls`, `--knn`, `--kmeans`, `--cart`, `--pca`, `--ml`: Machine learning and clustering options (requires `crisp-t[ml]`)
- `--visualize`: Generate visualizations (word clouds, topic charts, etc.)
- `--num, -n INTEGER`: Number parameter (clusters, topics, epochs, etc.) - default: 3
- `--rec, -r INTEGER`: Record parameter (top N results, recommendations) - default: 3
- `--filters, -f TEXT`: Filters to apply as `key=value` (can be used multiple times); keeps only documents where `document.metadata[key] == value`. Invalid formats raise an error.
- `--verbose, -v`: Print verbose messages for debugging

#### Data Sources

- `--source, -s PATH|URL`: Read source data from a directory (reads .txt and .pdf) or from a URL
- `--sources PATH|URL`: Provide multiple sources; can be used multiple times

### crispviz (Visualization CLI)

```bash
crispviz [OPTIONS]
```

- `--inp, --source, --sources`: Input corpus or sources
- `--out`: Output directory for PNG images
- Visualization flags: `--freq`, `--by-topic`, `--wordcloud`, `--top-terms`, `--corr-heatmap`
- Optional params: `--bins`, `--top-n`, `--columns`

### crispt (Corpus Manipulation CLI)

```bash
crispt [OPTIONS]
```

- `--id`, `--name`, `--description`: Corpus metadata
- `--doc`: Add document as `id|name|text` or `id|text` (repeatable)
- `--remove-doc`: Remove document by ID (repeatable)
- `--meta`: Add/update corpus metadata as `key=value` (repeatable)
- `--add-rel`: Add relationship as `first|second|relation` (repeatable)
- `--clear-rel`: Clear all relationships
- `--out`: Save corpus to folder/file as `corpus.json`
- `--inp`: Load corpus from folder/file containing `corpus.json`
- Query options:
	- `--df-cols`: Print DataFrame column names
	- `--df-row-count`: Print DataFrame row count
	- `--df-row INDEX`: Print DataFrame row by index
	- `--doc-ids`: Print all document IDs
	- `--doc-id ID`: Print document by ID
	- `--relationships`: Print all relationships
	- `--relationships-for-keyword KEYWORD`: Print relationships involving a keyword

### Example Usage

#### Corpus Creation and Manipulation
```bash
# Create a new corpus and add documents
crispt --id mycorpus --name "My Corpus" --doc "d1|Doc 1|Text" --doc "d2|Doc 2|More text" --out ./corpus_folder

# Load and query corpus
crispt --inp ./corpus_folder --doc-ids --relationships

# Add relationships and query by keyword
crispt --inp ./corpus_folder --add-rel "text:foo|num:bar|correlates" --relationships-for-keyword foo
```

#### Triangulation and Analysis
```bash
# Load corpus and perform text analysis
crisp --inp ./corpus_folder --codedict --sentiment

# Topic modeling and clustering
crisp --inp ./corpus_folder --topics --assign --num 4 --rec 10
```

#### Visualization
```bash
# Generate word frequency and topic visualizations
crispviz --inp ./corpus_folder --freq --by-topic --out ./viz_out
```

### Output Formats

Results can be saved in multiple formats:
- **JSON**: Default format, preserves all data structures
- **CSV**: For tabular data (DataFrames, topic assignments, etc.)
- **Text**: For readable reports and summaries

When saving analysis outputs via `--out`, files are automatically named with suffixes indicating the analysis type:
- `*_coding_dictionary.json`: Qualitative coding results
- `*_topics.json`: Topic modeling results
- `*_sentiment.json`: Sentiment analysis results
- `*_kmeans.json`: Clustering results
- `*_ml_results.json`: Classification results

When saving the corpus via `--out`, the CLI writes `corpus.json` (and `corpus_df.csv` if present) into the specified folder. If you pass a file path, only its parent directory is used for writing `corpus.json`.

## Example use

A company collects:
- **Textual** feedback from customer support interactions.
- **Numerical** data on customer retention and sales performance.
Using this framework, business analysts can investigate how recurring concerns in feedback correspond to measurable business outcomes.

## Framework Documentation

For detailed information about available functions, metadata handling, and theoretical frameworks, see the [comprehensive user instructions](/notes/INSTRUCTION.md).

## Author

* [Bell Eapen](https://nuchange.ca) ([UIS](https://www.uis.edu/directory/bell-punneliparambil-eapen)) |  [Contact](https://nuchange.ca/contact) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)


## Citation

Under review.

## Give us a star ⭐️
If you find this project useful, give us a star. It helps others discover the project.
