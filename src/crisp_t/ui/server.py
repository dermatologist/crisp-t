"""Quart web server for CRISP-T UI with Copilot SDK integration.

This module provides a web-based interface for CRISP-T qualitative research tools,
powered by the GitHub Copilot SDK. It allows researchers to interact with CRISP-T
through natural language conversations with AI assistants.

Key Features:
- Async REST API for session management (using Quart/ASGI)
- Real-time chat interface with streaming responses
- Integration with CRISP-T CLI tools (crisp, crispt, crispviz)
- Support for multiple AI models (GPT-5, Claude, etc.)
- Custom provider support (Ollama, Azure OpenAI, etc.)

Architecture:
- Quart ASGI web server handles HTTP requests asynchronously
- Copilot SDK manages AI sessions with custom tools
- execute_crisp_command tool allows AI to run CRISP-T commands
- Frontend polls for message updates in real-time

Dependencies:
- quart: Async web framework (ASGI)
- quart-cors: Cross-origin resource sharing
- github-copilot-sdk: AI integration (optional)
- pydantic: Type validation (optional, used with copilot)

Note: Migrated from Flask to Quart to resolve event loop issues with async operations.
"""

import asyncio
import subprocess
from typing import Dict

from quart import Quart, jsonify, render_template, request
from quart_cors import cors

# Check if copilot SDK is available
try:
    from copilot import CopilotClient, define_tool
    from pydantic import BaseModel, Field

    COPILOT_AVAILABLE = True
except ImportError:
    COPILOT_AVAILABLE = False

app = Quart(__name__, static_folder="static", template_folder="templates")
app = cors(app)

# Global state for managing copilot clients and sessions
clients: Dict[str, dict] = {}
clients_lock = asyncio.Lock()


# Define tool and model classes only if Copilot is available
if COPILOT_AVAILABLE:

    class CrispCommandParams(BaseModel):
        """Parameters for CRISP command execution."""

        command: str = Field(
            description="The CRISP CLI command to execute (crisp, crispt, or crispviz)"
        )
        args: str = Field(description="Command line arguments for the CRISP command")

    @define_tool(
        description="Execute CRISP-T CLI commands for qualitative research analysis"
    )
    async def execute_crisp_command(params: CrispCommandParams) -> str:
        """
        Execute CRISP-T CLI commands.

        This tool allows the agent to run CRISP-T commands for qualitative and mixed-methods research.
        Available commands: crisp, crispt, crispviz
        """
        valid_commands = ["crisp", "crispt", "crispviz"]
        if params.command not in valid_commands:
            return f"Error: Invalid command '{params.command}'. Must be one of: {', '.join(valid_commands)}"

        try:
            # Build the full command
            full_command = [params.command] + params.args.split()

            # Execute the command
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Combine stdout and stderr for complete output
            output = result.stdout
            if result.stderr:
                output += f"\n\nErrors/Warnings:\n{result.stderr}"

            if result.returncode != 0:
                return f"Command failed with exit code {result.returncode}:\n{output}"

            return output or "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (exceeded 5 minutes)"
        except FileNotFoundError:
            return f"Error: Command '{params.command}' not found. Is CRISP-T installed?"
        except Exception as e:
            return f"Error executing command: {str(e)}"


async def create_copilot_session(session_id: str, model: str, config: dict) -> dict:
    """Create a new Copilot client and session."""
    if not COPILOT_AVAILABLE:
        raise RuntimeError(
            "Copilot SDK is not installed. Install with: pip install crisp-t[copilot]"
        )

    # Create client
    client_config = {"log_level": "info", "auto_start": True, "auto_restart": True}

    # Add GitHub token if provided
    if config.get("github_token"):
        client_config["github_token"] = config["github_token"]

    client = CopilotClient(client_config)
    await client.start()

    # Prepare session configuration
    session_config = {
        "model": model,
        "tools": [execute_crisp_command],
        "streaming": True,
    }

    # Add custom provider if specified
    if config.get("use_custom_provider"):
        provider_config = {
            "type": config.get("provider_type", "openai"),
            "base_url": config.get("provider_base_url"),
        }

        if config.get("provider_api_key"):
            provider_config["api_key"] = config["provider_api_key"]

        session_config["provider"] = provider_config

    # Add system message emphasizing CRISP-T expertise
    system_message = {
        "role": "system",
        "content": """You are an expert CRISP-T qualitative research assistant. You help researchers perform
        mixed-methods analysis using CRISP-T CLI tools (crisp, crispt, crispviz).

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

## Tips for Effective Use
* **Use ./workspace** as the working directory for all artifacts, if not explicitly specified. Create it if it does not exist.
* **Use `--help`** with any command to see available options
* **Always start with `--source`** to import data into a corpus structure, if not already done
* **Use `--unstructured`** to specify free-text columns in CSV files
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


## Important steps
* Import data into CRISP-T corpus and dataframe.
* Perform linking between text and numeric data using various methods (id based, keyword based, time based, embedding based).
* Explore text data using various methods (e.g., topic modeling, keyword extraction, sentiment analysis, visualizations).
* Explore numeric data using various methods (e.g., summary statistics, classification, clustering, regression, association, visualizations, TDA, etc.).
* Perform cross modal analysis using linked text and numeric data (e.g., text features as predictors for numeric outcomes, numeric features as predictors for text outcomes, etc.).
* Add manual connections between text documents and numeric rows if needed to support theory driven analysis.
* Derive insights from the analysis and document them.
---

### Reference Guide: https://r.jina.ai/https://github.com/dermatologist/crisp-t/wiki


### 1. CRISP - Main Analysis Engine

**Command**: `crisp [options]`

**Essential Workflow**:
```bash
# Step 1: Import data from a source directory
crisp --source data_folder --out corpus_output

# Step 2: Run analysis on imported corpus
crisp --inp corpus_output [analysis options]

# Step 3: Save results
crisp --inp corpus_output --out results_folder
```

#### Data Loading Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--source/-s` | directory/URL | Import new data (creates corpus). Source folder should contain .txt, .pdf, .csv files |
| `--inp/-i` | path | Load existing corpus (corpus.json + corpus_df.csv) |
| `--out/-o` | path | Save corpus after analysis |
| `--unstructured/-t` | column_name | Mark CSV columns as free-text. Use multiple times for multiple columns |
| `--num/-n` | integer (default: 3) | When importing: max number of text/PDF files. When analyzing: numerical parameter (clusters, topics, etc.) |
| `--rec/-r` | integer (default: 3) | When importing: max CSV rows. When analyzing: top results to display |

**Example**:
```bash
crisp --source interview_data --unstructured responses --unstructured notes \
      --out my_corpus --num 50 --rec 10
```

#### Text Analysis Options

| Option | Flag | Purpose |
|--------|------|---------|
| `--codedict` | ✓ | Generate coding dictionary (verbs, nouns, adjectives/adverbs) |
| `--topics` | ✓ | Perform LDA topic modeling |
| `--assign` | ✓ | Assign documents to topics (run after --topics) |
| `--cat` | ✓ | Extract categories/concepts |
| `--summary` | ✓ | Generate extractive summary |
| `--sentiment` | ✓ | VADER sentiment analysis (corpus-level) |
| `--sentence` | ✓ | Document-level sentiment scores |
| `--nlp` | ✓ | Run ALL text analysis (codedict, topics, categories, summary, sentiment) |

**Important Notes**:
- Use `--clear` before `--assign` if corpus/filters changed
- VADER sentiment is corpus-level by default; combine with `--sentence` for document-level
- `--nlp` runs all text analyses sequentially

**Example**:
```bash
# Run all text analysis at once (not recommended for large corpora)
crisp --inp my_corpus --nlp --out results

# Topic analysis workflow
crisp --inp my_corpus --topics --num 5 --assign --out results
crisp --inp results --clear --assign --out results_v2
crisp --inp results_v2 --clear --filters "region=North" --assign --out results_filtered

# Sentiment with document-level scores
crisp --inp my_corpus --sentiment --sentence --out results
```

#### Machine Learning Options

| Option | Flag | Requires ML | Purpose |
|--------|------|------------|---------|
| `--cls` | ✓ | Yes | Classification (SVM + Decision Tree) |
| `--nnet` | ✓ | Yes | Neural Network classifier |
| `--knn` | ✓ | Yes | K-Nearest Neighbors search |
| `--kmeans` | ✓ | Yes | K-Means clustering |
| `--pca` | ✓ | Yes | Principal Component Analysis |
| `--regression` | ✓ | Yes | Linear/Logistic regression (auto-detect) |
| `--lstm` | ✓ | Yes | LSTM neural network on text |
| `--cart` | ✓ | Yes | Association rules (Apriori algorithm) |
| `--ml` | ✓ | Yes | Run ALL ML analyses |

**ML-Related Options**:
- `--outcome` (column_name or text_field) - Target variable for prediction
- `--linkage` (id/embedding/temporal/keyword) - Link text metadata to outcome
- `--aggregation` (majority/mean/first/mode) - How to combine multiple documents
- `--include` (columns) - Specific features to use
- `--ignore` (columns) - Features to exclude

**Example**:
```bash
# Classification with text metadata outcome
crisp --inp my_corpus --cls --outcome topic_name
crisp --inp my_corpus --cls --outcome topic_name --linkage keyword --aggregation majority
# Regression with numeric outcome
crisp --inp my_corpus --regression --outcome satisfaction_score --include age,income

# Neural network with auto-detected outcome type
crisp --inp my_corpus --nnet --outcome survey_response --linkage temporal:df --aggregation mean

# K-Means clustering with specific features
crisp --inp my_corpus --kmeans --num 4 --include age,income,years_experience
```

#### Filtering & Processing

| Option | Format | Purpose |
|--------|--------|---------|
| `--filters/-f` | key=value or link | Filter documents/rows. Multiple filters use AND logic |
| `--ignore` | comma-separated | Exclude words/columns from analysis |
| `--include` | comma-separated | Include specific columns |
| `--clear` | ✓ | Clear cache before analysis (use when switching datasets) |
| `--verbose/-v` | ✓ | Show detailed debugging information |

**Filter Examples**:
```bash
# Metadata filters
crisp --inp corpus --filters "region=North" --filters "source=Interview" --nlp

# Link-based filters (requires prior linking)
crisp --inp corpus --filters "embedding:text" --filters "temporal:df" --sentiment

# Combined
crisp --inp corpus --filters "sentiment=positive" --topics --num 5
```

#### Output & Display

| Option | Format | Purpose |
|--------|--------|---------|
| `--print/-p` | documents/N | Display corpus info. Examples: --print documents, --print 10 |
| `--sources` | path | Load from multiple source folders (used multiple times) |

---

### 2. CRISPT - Corpus Management

**Command**: `crispt [options]`

**Purpose**: Manipulate corpus structure, documents, metadata, and analyze semantic relationships.

#### Corpus Creation & Management

| Option | Format | Purpose |
|--------|--------|---------|
| `--id` | text (required for new) | Unique corpus identifier |
| `--name` | text | Descriptive corpus name |
| `--description` | text | Detailed corpus description |
| `--inp` | path | Load existing corpus |
| `--out` | path | Save corpus |
| `--print` | ✓ | Display full corpus |
| `--clear-rel` | ✓ | Remove all relationships |
| `--verbose/-v` | ✓ | Debug mode |

**Example**:
```bash
# Create new corpus
crispt --id my_study --name "Health Interview Study" --description "2025 interviews" \
       --out corpus_folder

# Load and display
crispt --inp corpus_folder --print

# Clear relationships
crispt --inp corpus_folder --clear-rel --out corpus_folder
```

#### Document Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--doc` | id\|name\|text | Add document (name optional) |
| `--remove-doc` | doc_id | Remove document by ID |
| `--doc-ids` | ✓ | List all document IDs |
| `--doc-id` | doc_id | Display specific document details |
| `--meta` | key=value | Add corpus metadata |

**Document Format**: `id|name|text` or `id|text`

**Example**:
```bash
# Add documents
crispt --id study --doc "interview1|Interview with Jane|Interview transcript..." \
       --doc "interview2|Interview with Bob|..." --out corpus_folder

# View documents
crispt --inp corpus_folder --doc-ids
crispt --inp corpus_folder --doc-id interview1

# Remove document
crispt --inp corpus_folder --remove-doc interview2 --out corpus_folder
```

#### Relationship Management

| Option | Format | Purpose |
|--------|--------|---------|
| `--add-rel` | first\|second\|relation | Add text↔numeric relationship |
| `--print-relationships` | ✓ | Show all relationships |
| `--relationships-for-keyword` | keyword | Find relationships involving keyword |

**Relationship Format**: `first|second|relation`
- **first**: `text:keyword` or `num:column`
- **second**: `text:keyword` or `num:column`
- **relation**: `correlates`, `predicts`, `contrasts`, etc.

**Example**:
```bash
# Add relationships after topic modeling
crispt --inp corpus_folder \
       --add-rel "text:healthcare|num:satisfaction_score|predicts" \
       --add-rel "text:cost_barriers|num:income_level|correlates" \
       --out corpus_folder

# Display relationships
crispt --inp corpus_folder --print-relationships
crispt --inp corpus_folder --relationships-for-keyword healthcare
```

#### Semantic Search Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--semantic` | query_text | Find documents similar to query |
| `--similar-docs` | doc_id1,doc_id2 | Find docs similar to reference docs |
| `--semantic-chunks` | query_text | Search within document chunks |
| `--doc-id` | doc_id | Specify document for chunk search |
| `--num` | integer (default: 5) | Results to return |
| `--rec` | float (default: 0.4) | Similarity threshold 0-1 |
| `--metadata-df` | ✓ | Export search metadata to DataFrame |
| `--metadata-keys` | keys | Specific metadata to export |

**Example**:
```bash
# Semantic search
crispt --inp corpus_folder --semantic "healthcare barriers" --num 10

# Find similar documents (literature review snowballing)
crispt --inp corpus_folder --similar-docs "doc1,doc2" --num 20 --rec 0.7

# Search within document
crispt --inp corpus_folder --semantic-chunks "cost barriers" \
       --doc-id interview1 --rec 0.5

# Export metadata
crispt --inp corpus_folder --metadata-df --metadata-keys "source,date,region"
```

#### DataFrame Operations

| Option | Format | Purpose |
|--------|--------|---------|
| `--df-cols` | ✓ | Show all DataFrame column names |
| `--df-row-count` | ✓ | Show row count |
| `--df-row` | index | Display specific row |

**Example**:
```bash
crispt --inp corpus_folder --df-cols
crispt --inp corpus_folder --df-row-count
crispt --inp corpus_folder --df-row 0
```

#### Temporal Analysis

| Option | Format | Purpose |
|--------|--------|---------|
| `--temporal-link` | method:column[:param] | Link documents to rows by time |
| `--temporal-filter` | start:end | Filter by time range (ISO 8601) |
| `--temporal-summary` | period | Summarize by time period |

**Methods**:
- `nearest:column` - Nearest timestamp
- `window:column:seconds` - Within time window
- `sequence:column:period` - By periods (D/W/M/Y)

**Example**:
```bash
# Link by nearest time
crispt --inp corpus_folder --temporal-link "nearest:timestamp" --out corpus_folder

# Link with 5-minute window
crispt --inp corpus_folder --temporal-link "window:timestamp:300" --out corpus_folder

# Link weekly
crispt --inp corpus_folder --temporal-link "sequence:timestamp:W" --out corpus_folder

# Filter time range
crispt --inp corpus_folder --temporal-filter "2025-01-01:2025-06-30" --out filtered_corpus

# Weekly summary
crispt --inp corpus_folder --temporal-summary "W"
```

#### Advanced Analysis

| Option | Format | Purpose |
|--------|--------|---------|
| `--tdabm` | y:x_vars[:radius] | Topological Data Analysis Ball Mapper |
| `--graph` | ✓ | Generate corpus relationship graph |

**Example**:
```bash
# TDABM analysis
crispt --inp corpus_folder --tdabm "satisfaction:age,income:0.3" --out corpus_folder

# Generate graph
crispt --inp corpus_folder --graph --out corpus_folder
```

---

### 3. CRISPVIZ - Visualization Engine

**Command**: `crispviz [options]`

**Purpose**: Generate charts, word clouds, LDA visualizations, and relationship graphs.

#### Basic Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--inp/-i` | path | Load corpus for visualization |
| `--out/-o` | path | Output directory for images |
| `--bins` | integer (default: 100) | Bins for frequency histograms |
| `--topics-num` | integer (default: 8) | Number of LDA topics |
| `--top-n` | integer (default: 20) | Top terms to display |
| `--verbose/-v` | ✓ | Debug output |

#### Visualization Types

| Option | Flag | Requires | Purpose |
|--------|------|----------|---------|
| `--freq` | ✓ | None | Word frequency distribution |
| `--top-terms` | ✓ | None | Top terms bar chart |
| `--wordcloud` | ✓ | LDA topics | Topic word cloud |
| `--ldavis` | ✓ | LDA topics | Interactive LDA visualization (HTML) |
| `--by-topic` | ✓ | LDA topics | Distribution by dominant topic |
| `--corr-heatmap` | ✓ | Numeric data | Correlation matrix heatmap |
| `--tdabm` | ✓ | TDABM analysis | TDABM topology visualization |
| `--graph` | ✓ | Graph data | Relationship network graph |

#### Graph Visualization Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--graph-nodes` | node_types | Node types: document, keyword, cluster, metadata |
| `--graph-layout` | algorithm | Layout: spring (default), circular, kamada_kawai, spectral |

#### Correlation Heatmap Options

| Option | Format | Purpose |
|--------|--------|---------|
| `--corr-columns` | column_list | Specific numeric columns (auto-selected if empty) |

**Example Usage**:
```bash
# Word frequency
crispviz --inp corpus_folder --out viz_output --freq

# Top terms chart
crispviz --inp corpus_folder --out viz_output --top-terms --top-n 30

# LDA visualizations (requires prior topic modeling)
crispviz --inp corpus_folder --out viz_output --wordcloud
crispviz --inp corpus_folder --out viz_output --ldavis --topics-num 5
crispviz --inp corpus_folder --out viz_output --by-topic

# Correlation analysis
crispviz --inp corpus_folder --out viz_output --corr-heatmap \
         --corr-columns age,income,satisfaction_score

# Network graph
crispviz --inp corpus_folder --out viz_output --graph \
         --graph-nodes document,keyword --graph-layout spring

# TDABM topology
crispviz --inp corpus_folder --out viz_output --tdabm

# All visualizations
crispviz --inp corpus_folder --out viz_output --freq --top-terms \
         --wordcloud --corr-heatmap --graph
```

---

## Common Workflows

### Workflow 1: Basic Qualitative Analysis

```bash
# 1. Import data
crisp --source research_data --out corpus --num 100 --unstructured "open_ended_q"

# 2. Generate coding dictionary
crisp --inp corpus --codedict --out corpus_v1

# 3. Topic modeling
crisp --inp corpus_v1 --topics --num 5 --assign --out corpus_v2

# 4. Sentiment analysis
crisp --inp corpus_v2 --sentiment --sentence --out corpus_v3

# 5. Visualizations
crispviz --inp corpus_v3 --out visualizations --freq --wordcloud --by-topic

# 6. Save final corpus
crispt --inp corpus_v3 --out final_corpus --print
```

### Workflow 2: Mixed-Methods Triangulation

```bash
# 1. Create corpus with CSV data
crisp --source data --unstructured comments --out corpus

# 2. Generate text analysis
crisp --inp corpus --topics --num 4 --assign --sentiment --out corpus_analyzed

# 3. Add relationships linking text findings to numeric outcomes
crispt --inp corpus_analyzed \
       --add-rel "text:healthcare|num:satisfaction_score|predicts" \
       --add-rel "text:cost_concerns|num:household_income|correlates" \
       --out corpus_linked

# 4. ML analysis linking text to numeric
crisp --inp corpus_linked --regression --outcome satisfaction_score \
      --linkage keyword --aggregation mean --out results

# 5. Visualize relationships
crispviz --inp results --out viz --graph --corr-heatmap
crispt --inp results --print-relationships
```

### Workflow 3: Temporal Analysis

```bash
# 1. Import time-stamped data
crisp --source time_series_data --out corpus

# 2. Link documents by time
crispt --inp corpus --temporal-link "sequence:timestamp:W" --out corpus_temporal

# 3. Generate temporal summary
crispt --inp corpus_temporal --temporal-summary "W"

# 4. Filter to specific period
crispt --inp corpus_temporal --temporal-filter "2025-01-01:2025-06-30" --out corpus_period

# 5. Analyze sentiment over time
crisp --inp corpus_period --sentiment --sentence --out results_period

# 6. Visualize time series
crispviz --inp results_period --out viz --freq --by-topic
```

### Workflow 4: ML Classification with Mixed Data

```bash
# 1. Prepare corpus
crisp --source data --unstructured text_col --out corpus

# 2. Generate text features
crisp --inp corpus --codedict --topics --num 3 --assign --out corpus_features

# 3. Train classifier with text metadata
crisp --inp corpus_features --cls --outcome satisfaction_level \
      --linkage keyword --aggregation majority --include age,income \
      --out classifier_results

# 4. View feature importance
crispt --inp classifier_results --print-relationships

# 5. Visualize results
crispviz --inp classifier_results --out viz --corr-heatmap --graph
```

---

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
- `--verbose`: Essential for debugging multi-step workflows

### File Formats
- **Corpus files**: `corpus.json` + `corpus_df.csv` (created in `--out` folder)
- **Visualizations**: PNG/HTML (saved to `--out` folder)
- **Metadata**: Embedded in corpus.json (view with `--print`)
---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Cache error before --assign` | Cache from previous run | Use `--clear` flag |
| `Outcome not found` | Wrong column/field name | Use `crispt --df-cols` or `crispt --print` to verify |
| `ML features mismatch` | Features changed after training | Clear cache and retrain |
| `Linkage failed` | Insufficient data/metadata | Verify timestamps or use simpler linkage method |
| `Visualization empty` | Analysis not run | Ensure `--topics`, `--tdabm`, or `--graph` completed first |

---

## Performance Notes

- **Large corpora** (1000+ docs): Use `--num` to limit imports, use filters
- **Topic modeling**: Adjust `--num` lower for faster processing (3-5 recommended)
- **TDABM/graphs**: More expensive; save intermediate results
- **Semantic search**: Requires initialization; slower on first run
- **ML training**: Very slow on large datasets; use sampling/filtering

""",
    }
    session_config["system_message"] = system_message

    # Note: Temperature and max_tokens are typically controlled at the provider/model level
    # These settings from the config are captured but not currently used
    # Future enhancement: Pass these to the provider configuration if supported

    # Create session
    session = await client.create_session(session_config)

    # Store message history
    messages = []

    # Event handler for session events
    def on_event(event):
        event_type = event.type.value
        print(f"[DEBUG] Event received: {event_type}")
        if event_type == "assistant.message":
            content = event.data.content
            print(f"[DEBUG] assistant.message: content_length={len(content)}")
            timestamp = getattr(event.data, "created_at", None)
            messages.append(
                {"role": "assistant", "content": content, "timestamp": timestamp}
            )
        elif event_type == "user.message":
            content = event.data.content
            print(f"[DEBUG] user.message: content={content[:50]}...")
            timestamp = getattr(event.data, "created_at", None)
            messages.append(
                {"role": "user", "content": content, "timestamp": timestamp}
            )
        elif event_type == "assistant.message_delta":
            # Handle streaming chunks
            delta = event.data.delta_content or ""
            print(f"[DEBUG] assistant.message_delta: delta_length={len(delta)}")
            if (
                not messages
                or messages[-1].get("role") != "assistant"
                or messages[-1].get("complete")
            ):
                messages.append({"role": "assistant", "content": "", "complete": False})
            messages[-1]["content"] += delta
        elif event_type == "session.idle":
            # Mark last message as complete
            print(f"[DEBUG] session.idle: messages_count={len(messages)}")
            if messages and messages[-1].get("role") == "assistant":
                messages[-1]["complete"] = True
                print(
                    f"[DEBUG] Marked message as complete, content_length={len(messages[-1]['content'])}"
                )

    session.on(on_event)

    return {
        "client": client,
        "session": session,
        "messages": messages,
        "model": model,
        "config": config,
    }


@app.route("/")
async def index():
    """Serve the main UI page."""
    return await render_template("index.html")


@app.route("/api/health", methods=["GET"])
async def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "copilot_available": COPILOT_AVAILABLE,
            "version": "1.0.0",
        }
    )


@app.route("/api/models", methods=["GET"])
async def list_models():
    """List available models from Copilot."""
    if not COPILOT_AVAILABLE:
        return jsonify({"error": "Copilot SDK not available"}), 500

    try:
        # Create a temporary client to get model list
        client = CopilotClient()
        await client.start()
        models = await client.list_models()
        await client.stop()

        return jsonify({"models": [model["id"] for model in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/create", methods=["POST"])
async def create_session():
    """Create a new Copilot session."""
    if not COPILOT_AVAILABLE:
        return (
            jsonify(
                {
                    "error": "Copilot SDK not available. Install with: pip install crisp-t[copilot]"
                }
            ),
            500,
        )

    data = await request.json
    session_id = data.get("session_id")
    model = data.get("model", "gpt-5")
    config = data.get("config", {})

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    try:
        # Create the session asynchronously
        session_data = await create_copilot_session(session_id, model, config)

        # Store in global state
        async with clients_lock:
            clients[session_id] = session_data

        return jsonify({"status": "ok", "session_id": session_id, "model": model})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/send", methods=["POST"])
async def send_message(session_id: str):
    """Send a message to a Copilot session."""
    if not COPILOT_AVAILABLE:
        return jsonify({"error": "Copilot SDK not available"}), 500

    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients[session_id]

    data = await request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        session = session_data["session"]
        await session.send({"prompt": prompt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


@app.route("/api/session/<session_id>/messages", methods=["GET"])
async def get_messages(session_id: str):
    """Get message history for a session."""
    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients[session_id]

    messages = session_data["messages"]
    print(f"[DEBUG] get_messages: session={session_id}, count={len(messages)}")
    if messages:
        print(
            f"[DEBUG] Latest message: role={messages[-1].get('role')}, content_length={len(messages[-1].get('content', ''))}, complete={messages[-1].get('complete')}"
        )

    return jsonify({"messages": messages})


@app.route("/api/session/<session_id>/destroy", methods=["POST"])
async def destroy_session(session_id: str):
    """Destroy a Copilot session."""
    async with clients_lock:
        if session_id not in clients:
            return jsonify({"error": "Session not found"}), 404
        session_data = clients.pop(session_id)

    try:
        await session_data["session"].destroy()
        await session_data["client"].stop()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"})


def start_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Start the Quart web server.

    Args:
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        port: Port to bind to (default: 5000)
        debug: Run in debug mode (WARNING: Only use in development, not production)
    """
    if not COPILOT_AVAILABLE:
        print(
            "WARNING: Copilot SDK is not installed. Install with: pip install crisp-t[copilot]"
        )
        print("The server will start but Copilot features will not be available.")

    if debug:
        print(
            "\n⚠️  WARNING: Debug mode is enabled. This should only be used in development!"
        )
        print(
            "    Debug mode allows arbitrary code execution and should NEVER be used in production."
        )

    print(f"\n🚀 CRISP-T Web UI starting on http://{host}:{port}")
    print(f"📖 Open your browser and navigate to: http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")

    # Run the Quart app using the built-in ASGI server
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    start_server()
