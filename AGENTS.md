# Agent Instructions for CRISP-T Repository

## Repository Overview

**CRISP-T** (Cross Industry Standard Process for Triangulation) is a qualitative research method and toolkit for analyzing mixed datasets containing both textual and numerical data. It enables computational triangulation and sense-making through:

- Text analytics (topic modeling, sentiment analysis, NLP)
- Numerical analysis (regression, clustering, decision trees)
- Integration of qualitative and quantitative findings
- MCP (Model Context Protocol) server for AI agent integration

## Key Capabilities

### Core Functionality
- **Textual Analysis**: Document management, topic modeling, sentiment analysis, coding dictionaries, categorization
- **Numerical Analysis**: Regression, clustering, classification, PCA, association rules
- **Triangulation**: Link textual findings with numerical variables through relationships
- **MCP Server**: Full functionality exposed as tools, resources, and prompts for AI agents

### Command Line Tools
1. `crisp` - Main analytical CLI for triangulation and analysis
2. `crispviz` - Visualization CLI for corpus data
3. `crispt` - Corpus manipulation CLI
4. `crisp-mcp` - MCP server for AI agent integration

## Repository Structure

```
crisp-t/
├── src/crisp_t/           # Main source code
│   ├── mcp/               # MCP server implementation
│   ├── helpers/           # Helper modules
│   └── utils.py           # Utility functions
├── tests/                 # Test files
├── docs/                  # Documentation (MkDocs)
├── notes/                 # User guides and demos
│   ├── DEMO.md           # Usage examples
│   └── INSTRUCTION.md    # Comprehensive user instructions
├── examples/              # Example scripts
├── pyproject.toml        # Project configuration
└── README.md             # Main readme

Key files you should familiarize yourself with:
- README.md - Main documentation
- notes/INSTRUCTION.md - Detailed function reference and workflows
- notes/DEMO.md - Practical usage examples
- docs/MCP_SERVER.md - MCP server documentation
- src/crisp_t/mcp/server.py - MCP server implementation
```

## Development Workflow

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   cd /path/to/workspace
   git clone https://github.com/dermatologist/crisp-t.git
   cd crisp-t
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[ml,xg,dev]"
   # or using uv
   uv pip install -e ".[ml,xg,dev]"
   ```

3. **Run tests**:
   ```bash
   pytest
   # or with coverage
   pytest --cov=src/crisp_t
   ```

### Code Style and Conventions

- **Python Version**: Python 3.8+
- **Code Style**: Follow PEP 8; use existing code patterns as reference
- **Testing**: Write tests for new functionality; maintain existing test coverage
- **Documentation**: Update relevant docs (README, INSTRUCTION.md, MCP_SERVER.md) for user-facing changes
- **Comments**: Add comments only when necessary to explain complex logic; code should be self-documenting

### Git Workflow

- **Branch**: Work on the `develop` branch or feature branches
- **Commits**: Use meaningful commit messages describing the change
- **Pull Requests**: Submit PRs to the `develop` branch

## Common Tasks for Agents

### 1. Adding New MCP Tools

When adding a new MCP tool:

1. **Update `src/crisp_t/mcp/server.py`**:
   - Add tool definition in `list_tools()` function
   - Add implementation in `call_tool()` function
   - Follow existing patterns for error handling

2. **Update documentation**:
   - Add tool description to `docs/MCP_SERVER.md`
   - Include example usage

3. **Test the tool**:
   - Run the MCP server: `crisp-mcp`
   - Test with an MCP client (e.g., Claude Desktop)

### 2. Adding New CLI Features

For new CLI features in `crisp`, `crispviz`, or `crispt`:

1. **Update CLI implementation** in `src/crisp_t/cli/`
2. **Update help text** in the CLI code
3. **Update documentation**:
   - README.md for main features
   - notes/INSTRUCTION.md for detailed instructions
   - notes/DEMO.md for examples

### 3. Fixing Bugs

1. **Understand the issue**: Review issue description and reproduce the bug
2. **Locate the code**: Use grep, find, or IDE search to locate relevant code
3. **Make minimal changes**: Fix only what's necessary; don't refactor unrelated code
4. **Test thoroughly**: Ensure the fix works and doesn't break existing functionality
5. **Update tests**: Add regression tests if appropriate

### 4. Updating Documentation

When updating documentation:

- **README.md**: High-level overview, installation, basic usage
- **notes/INSTRUCTION.md**: Detailed function reference, workflows, best practices
- **notes/DEMO.md**: Step-by-step examples with commands
- **docs/MCP_SERVER.md**: MCP server tools, resources, prompts

## Testing Strategy

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/crisp_t --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run specific test
pytest tests/test_specific.py::test_function_name
```

### Writing Tests

- Follow existing test patterns in `tests/`
- Use pytest fixtures for common setup
- Mock external dependencies (files, networks, etc.)
- Test both success and error cases

## MCP Server Usage

### Starting the Server

```bash
crisp-mcp
```

### Available MCP Tools

**Corpus Management**: load_corpus, save_corpus, add_document, remove_document, get_document, list_documents, add_relationship, get_relationships

**Text Analysis**: generate_coding_dictionary, topic_modeling, assign_topics, extract_categories, generate_summary, sentiment_analysis

**Numeric Analysis**: get_df_columns, get_df_row_count, get_df_row

**Machine Learning** (requires crisp-t[ml]): kmeans_clustering, decision_tree_classification, svm_classification, neural_network_classification, regression_analysis, pca_analysis, association_rules, knn_search

### MCP Prompts

- `analysis_workflow` - Step-by-step analysis guide
- `triangulation_guide` - Guide for triangulating qualitative and quantitative findings

## Common Patterns

### Loading and Saving Corpus

```python
# Load from existing corpus
from crisp_t.helpers.initializer import initialize_corpus
corpus = initialize_corpus(inp="/path/to/corpus_folder")

# Load from source directory
corpus = initialize_corpus(source="/path/to/source_data")

# Save corpus
from crisp_t.helpers.io import write_data
write_data.write_corpus_to_json(corpus, "/path/to/output")
```

### Text Analysis

```python
from crisp_t.text import Text
text = Text(corpus=corpus)
text.document_count()
text.common_words(index=20)
```

### Numeric Analysis with ML

```python
from crisp_t.csv import Csv
from crisp_t.ml import ML

csv = Csv(corpus=corpus)
csv.read_csv("data.csv")

ml = ML(csv=csv)
ml.get_regression(y="outcome_column")
ml.get_kmeans(number_of_clusters=5)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed; ML features require `crisp-t[ml]`
2. **File Path Issues**: Use absolute paths; relative paths may not work as expected
3. **MCP Server Not Found**: Ensure `crisp-mcp` command is in PATH after installation
4. **Test Failures**: Some tests may depend on external data or ML libraries; install all dev dependencies

### Environment Variables

- `HOME` - Used for default data paths in some examples (not hardcoded in main code)

## Best Practices for AI Agents

1. **Understand Before Changing**: Review existing code patterns before making changes
2. **Minimal Changes**: Make the smallest possible changes to achieve the goal
3. **Test Early and Often**: Run tests after each significant change
4. **Document User-Facing Changes**: Update README.md, INSTRUCTION.md, or MCP_SERVER.md for features users will interact with
5. **Follow Existing Patterns**: Match the style and structure of existing code
6. **Preserve Metadata**: The corpus structure includes metadata; don't discard it
7. **Handle Errors Gracefully**: Follow existing error handling patterns

## Example Workflows

### Typical Analysis Workflow

1. Load corpus from source or existing data
2. Perform text analysis (topics, sentiment, coding)
3. Perform numeric analysis (regression, clustering)
4. Link findings through relationships
5. Save corpus with metadata

### MCP Server Workflow

1. User requests analysis through AI assistant
2. Agent uses MCP tools to:
   - Load corpus
   - Perform requested analyses
   - Interpret results
   - Add relationships if patterns found
   - Save corpus
3. Agent provides interpretation and insights

## Key Concepts

### Corpus
The core data structure containing:
- Documents (textual data)
- DataFrame (numerical data)
- Metadata (analysis results, relationships)

### Relationships
Links between textual findings and numerical variables:
- Format: `first|second|relation`
- Example: `text:healthcare|num:satisfaction_score|predicts`

### Triangulation
Process of validating findings across multiple data sources and analytical methods:
1. Text analysis reveals patterns
2. Numeric analysis reveals correlations
3. Relationships document connections
4. Validation confirms findings hold true

## Resources

- **Main Documentation**: https://dermatologist.github.io/crisp-t/
- **GitHub Repository**: https://github.com/dermatologist/crisp-t
- **Issue Tracker**: https://github.com/dermatologist/crisp-t/issues
- **MCP Protocol**: https://modelcontextprotocol.io/

## Contact

For questions or contributions, see CONTRIBUTING.md or contact the maintainer through GitHub issues.
