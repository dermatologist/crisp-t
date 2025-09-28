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

## Command Line Interface

CRISP-T provides a comprehensive command-line interface for analyzing textual and numerical data. The CLI supports various analysis types and can handle different input formats.

### Basic Usage

```bash
crisp-t [OPTIONS]
```

### Input/Output Options

- `--inp, -i TEXT`: Input file in text format (.txt, .json)
- `--csv TEXT`: CSV file name for numerical data analysis  
- `--out, -o TEXT`: Output file name for saving results
- `--titles, -t TEXT`: Document(s) or CSV column(s) to analyze/compare (can be used multiple times)
- `--ignore TEXT`: Comma-separated words to ignore during analysis

### Analysis Options

#### Text Analysis
- `--codedict`: Generate qualitative coding dictionary
- `--topics`: Generate topic model using LDA
- `--assign`: Assign documents to topics
- `--cat`: List categories of entire corpus or individual documents
- `--summary`: Generate extractive text summary
- `--sentiment`: Generate sentiment scores using VADER
- `--sentence`: Generate sentence-level scores when applicable
- `--nlp`: Generate all NLP reports (combines above text analyses)

#### Machine Learning (requires `crisp-t[ml]`)
- `--nnet`: Display neural network model accuracy
- `--svm`: Display SVM confusion matrix  
- `--knn`: Display K-nearest neighbors analysis
- `--kmeans`: Display K-Means clustering results
- `--cart`: Display association rules (CART)
- `--pca`: Display Principal Component Analysis

#### Visualization and Configuration
- `--visualize`: Generate visualizations (word clouds, topic charts, etc.)
- `--num, -n INTEGER`: Number parameter (clusters, topics, epochs, etc.) - default: 3
- `--rec, -r INTEGER`: Record parameter (top N results, recommendations) - default: 3
- `--filters, -f TEXT`: Filters to apply (can be used multiple times)
- `--verbose, -v`: Print verbose messages for debugging

### Examples

#### Text Analysis from File
```bash
# Basic text analysis
crisp-t --inp interview.txt --codedict --sentiment

# Comprehensive NLP analysis with output
crisp-t --inp documents.txt --nlp --num 5 --out results

# Topic modeling with custom parameters
crisp-t --inp corpus.txt --topics --assign --num 4 --rec 10
```

#### CSV Data Analysis
```bash
# Analyze text columns from CSV
crisp-t --csv survey_data.csv --titles "comments,feedback" --sentiment --codedict

# Full analysis of CSV with both text and numerical data
crisp-t --csv research_data.csv --titles "open_responses" --nlp --kmeans --pca --num 3

# Machine learning analysis on numerical data
crisp-t --csv dataset.csv --titles target_variable --svm --nnet --kmeans --num 5
```

#### Combined Analysis
```bash
# Comprehensive triangulation analysis
crisp-t --csv mixed_data.csv --titles "qualitative_data,comments" --nlp --kmeans --svm --visualize --out comprehensive_results
```

#### Advanced Usage
```bash
# Custom analysis with filters and ignored words
crisp-t --inp interviews.txt --nlp --filters "demographic:adult" --ignore "um,uh,like" --verbose

# Export results in different formats
crisp-t --csv data.csv --titles responses --topics --out results.json
crisp-t --csv data.csv --titles responses --topics --out results.csv
```

### Output Formats

Results can be saved in multiple formats:
- **JSON**: Default format, preserves all data structures
- **CSV**: For tabular data (DataFrames, topic assignments, etc.)
- **Text**: For readable reports and summaries

Output files are automatically named with suffixes indicating the analysis type:
- `*_coding_dictionary.json`: Qualitative coding results
- `*_topics.json`: Topic modeling results  
- `*_sentiment.json`: Sentiment analysis results
- `*_kmeans.json`: Clustering results
- `*_svm_results.json`: Classification results

## Example use

A company collects:
- **Textual** feedback from customer support interactions.
- **Numerical** data on customer retention and sales performance.
Using this framework, business analysts can investigate how recurring concerns in feedback correspond to measurable business outcomes.

## Framework Documentation

For detailed information about available functions, metadata handling, and theoretical frameworks, see the [comprehensive user instructions](notes/INSTRUCTION.md).

## Author

* [Bell Eapen](https://nuchange.ca) ([UIS](https://www.uis.edu/directory/bell-punneliparambil-eapen)) |  [Contact](https://nuchange.ca/contact) | [![Twitter Follow](https://img.shields.io/twitter/follow/beapen?style=social)](https://twitter.com/beapen)


## Citation

Under review.

## Give us a star ⭐️
If you find this project useful, give us a star. It helps others discover the project.
