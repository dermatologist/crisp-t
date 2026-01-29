# Demo

## Setup

* CRISP-T is a python package that can be installed using pip and used from the command line. Your system should have python 3.11 or higher and pip installed. You can download and install Python for your operating system [here](https://www.python.org/downloads/). Optionally, CRISP-T can be imported in python scripts or jupyter notebooks, but this is not covered in this demo. See [documentation](https://dermatologist.github.io/crisp-t/) for more details.
* Install [Crisp-T](https://github.com/dermatologist/crisp-t) with `pip install crisp-t[ml]` or `uv pip install crisp-t[ml]`
* (Optional) Download covid narratives data to  `crisp_source` folder in home directory or current directory using `crisp --covid covidstories.omeka.net --source crisp_source`. You may use any other source of textual data (e.g. journal articles, interview transcripts) in .txt or .pdf format in the `crisp_source` folder or the folder you specify with --source option.
* (Optional) Download [Psycological Effects of COVID](https://www.kaggle.com/datasets/hemanthhari/psycological-effects-of-covid) dataset to `crisp_source` folder. You may use any other numeric dataset in .csv format in the `crisp_source` folder or the folder you specify with --source option.
* Create a `crisp_input` folder in home directory or current directory for keeping imported data for analysis.

## Import data

* Run the following command to import data from `crisp_source` folder to `crisp_input` folder.
* `--source` reads data from a directory (reads .txt, .pdf and a single .csv) or from a URL

```bash
crisp --source crisp_source --out crisp_input
```
* Ignore warnings related to pdf files.

## Perform Exploratory tasks using NLP

* Run the following command to perform a topic modelling and assign topics(keywords) to each narrative.
* *--inp crisp_input* below is optional as it defaults to `crisp_input` folder.

```bash
crisp --inp crisp_input --assign --out crisp_input
```

* The results will be saved in the same `crisp_input` folder, overwriting the corpus file.
* You may run several other analyses ([see documentation](https://dermatologist.github.io/crisp-t/) for details) and tweak parameters as needed.
* Hints will be provided in the terminal.

**From now on, we will use `crisp_input` folder as input folder unless specified otherwise as that is the default.**

## Explore results

```bash
crisp --print "documents 10"
```

* Notice that we have omitted --inp as it defaults to `crisp_input` folder. If you want to use a different folder, use --inp to specify it. The *--out* option helps to save intermediate results in a different folder.
* The above command prints first 10 documents in the corpus.

* Next, let us see the metadata assigned to each document.

```bash
crisp --print "documents metadata"
```

* Notice keywords/topics assigned to each narrative.
* You will notice *interviewee* and *interviewer* keywords. These are assigned based on the presence of these words in the narratives and may not be useful.
* You may remove these keywords by using --ignore with assign and check the results again.

```bash
crisp --clear --assign --ignore interviewee,interviewer --out crisp_input
crisp --print "documents metadata"
```

* *--clear* option clears the cache before running the analysis.
⚠️ **While analysing multiple datasets, use `crisp --clear` option to clear cache before switching datasets.** ⚠️
* Now you will see that these keywords are removed from the results.
* It prints the first 5 documents by default.

```bash
crisp --print "metadata clusters"
```
* Prints the clusters assigned to each document based on keywords.
* There are many other options to explore the results. See documentation for details.
* Let us choose narratives that contain 'mask' keyword and show the concepts/topics in these narratives.

```bash
crisp --inp crisp_input --clear --filters keywords=mask --topics
```

* The above results will not be saved as --out is not specified.
* Notice *time*, *people* as topics in this subset of narratives.
* If --filters is used, only the filtered documents are used for the analysis. When using filters you should explicitly specify --inp and --out options with different folders to avoid overwriting the input data.

## Quantitative exploratory analysis

* Let us see do a kmeans clustering of the csv dataset of covid data.

```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --kmeans
```

* Notice 3 clusters with different centroids. (number of clusters can be changed with --num option).

## Confirmation

* Let us add a relationship between numb:self_time and text:work in the corpus for future confirmation with LLMs.

```bash
crispt --add-rel "text:work|numb:self_time|correlates"
```

* Let us do a regression analysis to see how `relaxed` is affected by other variables.

```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --regression --outcome relaxed
```

* self_time has a positive correlation with relaxed.
* What about a decision tree analysis?

```bash
crisp --include relaxed,self_time,sleep_bal,time_dp,travel_time,home_env --cls --outcome relaxed
```

* Relaxed is converted to binary variable internally for classification.
* Ideally, you should do the binary conversion externally based on domain knowledge.
* Notice that self_time is the most important variable in predicting relaxed.

### [Topological Data Analysis](https://www.arxiv.org/abs/2504.14081) Rudkin, S., & Dlotko, P. (2024)
* Let us do a TDA analysis to see the shape of the data.
* parameters to --tdabm are specified as follows: *outcome:varables:radius*

```bash

crispt --tdabm relaxed:self_time,sleep_bal,time_dp,travel_time:0.6 --out crisp_input

```
* Let us visualize the TDA network.

```bash
crispviz --tdabm --out viz_out/
```

<p align="center">
  <img src="https://github.com/dermatologist/crisp-t/blob/develop/notes/tdabm.jpg" />
</p>

## [Sense-making by triangulation](INSTRUCTION.md)

## Now let us try out a csv dataset with text and numeric data.

* Download SMS Smishing Collection Data Set from [Kaggle](https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set) and convert the text file to csv adding the headers id, **CLASS** and **SMS**. Convert CLASS to numeric 0 and 1 for ham and smish respectively and add id as serial numbers.
* Place the csv file in a **new** `crisp_source` folder.
* Import the csv file to `crisp_input` folder using the following command.

```bash
crisp --source crisp_source/ --unstructured SMS
```

* Notice that the text column SMS is specified with --unstructured option. This creates CRISP documents from the text column.
* Now assign topics to the documents. Note that this also assigns clusters.

```bash
crisp --assign
```

* Now print the results to examine.
```bash
crisp --print "metadata clusters"
```

* Let us choose the cluster 1 and see the SMS classes in this cluster. (0=ham, 1=smish)
```bash
crisp --filters cluster=1 --print "dataframe stats"
```

* Next, let us check if the SMS texts converge towards predicting the CLASS (ham/ smish) variable with LSTM model.

```bash
crisp --lstm --outcome CLASS
```

## MCP Server for agentic AI. (Optional, but LLMs may be better at sense-making!)

### Try out the MCP server with the following command. (LLMs will offer course corrections and suggestions)


* load corpus from /Users/your-user-id/crisp_input
* use available tools
* What are the columns in df?
* Do a regression using time_bp,time_dp,travel_time,self_time with relaxed as outcome
* Interpret the results
* Is self_time or related concepts occur frequently in documents?
* can you ignore "interviewer,interviewee" and assign topics again? Yes.
* What are the topics in documents with keyword "work"?

<p align="center">
  <img src="https://github.com/dermatologist/crisp-t/blob/develop/notes/crisp.gif" />
</p>

## Visualization


### Let's [visualize the clusters in 2D space using PCA.](https://htmlpreview.github.io/?https://github.com/dermatologist/crisp-t/blob/develop/notes/lda_visualization.html)

```bash
crispviz --ldavis --out viz_out/
```

* The visualization will be saved in `viz_out` folder. Open the html file in a browser to explore.

### Let's generate a word cloud of keywords in the corpus.

```bash
crispviz --wordcloud --out viz_out/
```
* The word cloud will be saved in `viz_out` folder.

<p align="center">
  <img src="https://github.com/dermatologist/crisp-t/blob/develop/notes/wordcloud.jpg" />
</p>

## More examples — comprehensive CLI usage

The following grouped examples show common and advanced usage patterns for the three CLIs: `crisp`, `crispt`, and `crispviz`. These are practical, copy-pasteable command lines that demonstrate option combinations and formats discussed in this demo and cheatsheet.

### A. Data import & basic workflow (`crisp`)

# Import a folder with text files and a CSV; specify unstructured text column
```bash
crisp --source ./raw_data --unstructured "comments" --out ./crisp_input
```

# Import but limit text files and CSV rows when ingesting large sources
```bash
crisp --source ./raw_data --out ./crisp_input --num 10 --rec 500
```

# Import CSV placed in the source folder; ignore specific stopwords/columns
```bash
crisp --source ./survey --unstructured "comments" --ignore "interviewer,interviewee" --out ./survey_corpus
```

### B. Filtering and linking (`crisp` + `crispt`) — examples

# Exact-match filters (both `=` and `:` separators supported)
```bash
crisp --inp ./crisp_input --filters category=Health --topics
crisp --inp ./crisp_input --filters category:Health --topics
```

# Special link filters (text→df and df→text)
```bash
# Filter dataframe rows that are linked from documents via embeddings
crispt --inp ./crisp_input --filters embedding:text --out ./linked_by_embedding

# Filter documents that are linked from dataframe rows via temporal links
crispt --inp ./crisp_input --filters temporal:df --out ./linked_docs
```

# Legacy shorthand mappings — both map to `embedding:text` or `temporal:text`
```bash
crispt --inp ./crisp_input --filters =embedding
crispt --inp ./crisp_input --filters :temporal
```

# ID linkage: filter to a single ID, or sync remaining docs↔rows with blank value
```bash
# Filter to specific id
crisp --inp ./crisp_input --filters id=12345 --nlp

# Sync documents and dataframe rows by ID after other filters
crisp --inp ./crisp_input --filters id: --out ./synced_output
```

### C. Text analysis quick examples (`crisp`)

# Topic modeling and then assign topics to documents
```bash
crisp --inp ./crisp_input --topics --assign --out ./crisp_input_analyzed
```

# Run sentiment and summary together
```bash
crisp --inp ./crisp_input --sentiment --summary --num 5
```

# Run all NLP analyses (coding dictionary, topics, categories, summary, sentiment)
```bash
crisp --inp ./crisp_input --nlp
```

### D. Machine learning & cross-modal examples (`crisp`)

# Run k-means clustering on numeric CSV columns
```bash
crisp --inp ./survey_corpus --kmeans --num 4 --include age,income,score
```

# Classification (SVM + Decision Tree) using a DataFrame outcome column
```bash
crisp --inp ./survey_corpus --cls --outcome satisfaction_binary --include a,b,c --aggregation majority
```

# Neural net (requires `crisp-t[ml]`)
```bash
crisp --inp ./survey_corpus --nnet --outcome target_col --include feat1,feat2
```

# LSTM using text documents aligned by `id` column in CSV
```bash
crisp --inp ./survey_corpus --lstm --outcome CLASS
```

### E. Corpus management & inspection (`crispt`) — examples

# Create a new corpus and add documents
```bash
crispt --id my_corpus --name "Study A" --doc "1|Intro|This is the first document" --out ./my_corpus
```

# Add metadata and a relationship
```bash
crispt --inp ./my_corpus --meta "source=field" --add-rel "text:work|numb:self_time|correlates" --out ./my_corpus
```

# Remove a document and clear relationships
```bash
crispt --inp ./my_corpus --remove-doc 1 --clear-rel --out ./my_corpus
```

# Inspect dataset columns, row counts, or specific rows
```bash
crispt --inp ./my_corpus --df-cols
crispt --inp ./my_corpus --df-row-count
crispt --inp ./my_corpus --df-row 12
```

# Print usage: two supported formats
```bash
# Multi-flag form
crispt --inp ./my_corpus --print documents --print 10

# Single-string form
crispt --inp ./my_corpus --print "dataframe metadata"
```

### F. Semantic & embedding features (`crispt`) — examples

# Semantic search for similar documents (requires embedding backend)
```bash
crispt --inp ./my_corpus --semantic "patient anxiety" --num 8 --rec 0.45
```

# Find documents similar to a list of document IDs
```bash
crispt --inp ./my_corpus --similar-docs "1,2,3" --num 5
```

# Semantic-chunks: search within specific document chunks (use with --doc-id)
```bash
crispt --inp ./my_corpus --doc-id 5 --semantic-chunks "query phrase" --rec 0.6
```

# Embedding linking and stats
```bash
crispt --inp ./my_corpus --embedding-link "cosine:3:0.7" --embedding-stats --out ./emb_links
crispt --inp ./emb_links --filters embedding:df --out ./docs_linked_to_rows
```

### G. Temporal utilities (`crispt`) — examples

# Link by time (nearest, window with seconds, or sequence)
```bash
crispt --inp ./my_corpus --temporal-link "nearest:timestamp"
crispt --inp ./my_corpus --temporal-link "window:timestamp:300"  # ±300 seconds
```

# Temporal summaries, sentiment trends, and topics over periods
```bash
crispt --inp ./my_corpus --temporal-summary W
crispt --inp ./my_corpus --temporal-sentiment W:mean
crispt --inp ./my_corpus --temporal-topics W:5
```

### H. Visualization examples (`crispviz`)

# Word frequency + topic wordcloud + LDA interactive visualization
```bash
crispviz --inp ./crisp_input_analyzed --out viz_out --freq --wordcloud --ldavis
```

# Top terms with custom top-n and bins
```bash
crispviz --inp ./crisp_input --out viz_out --top-terms --top-n 30 --bins 80
```

# Correlation heatmap with selected numeric columns
```bash
crispviz --inp ./survey_corpus --out viz_out --corr-heatmap --corr-columns "age,income,score"
```

# Graph visualization filtered by node types and a different layout
```bash
crispviz --inp ./my_corpus --out viz_out --graph --graph-nodes document,keyword --graph-layout circular
```

### I. Small tips & parameter semantics

- `--rec` for `crispt` semantic commands can be a similarity threshold (float, default 0.4), while `--rec` for some `crisp` commands is used as an integer count — check the command context.
- `--num` defaults differ by context (e.g., `crispt` search default is 5; `crisp` analysis default is 3).
- `--aggregation` accepts `majority|mean|first|mode` and controls how multiple documents map to one numeric row are aggregated for ML tasks.

### J. Full-run example (import → analyze → visualize)

```bash
# 1) Import
crisp --source ./raw_data --unstructured "comments" --out ./crisp_input

# 2) Run NLP + sentiment + save
crisp --inp ./crisp_input --topics --assign --sentiment --out ./crisp_input_analyzed

# 3) Link by embedding and run regression on linked set
crispt --inp ./crisp_input_analyzed --embedding-link "cosine:1:0.7" --out ./linked
crisp --inp ./linked --outcome satisfaction_score --regression --out ./final_results

# 4) Create visualizations
crispviz --inp ./final_results --out viz_out --ldavis --wordcloud --corr-heatmap
```

