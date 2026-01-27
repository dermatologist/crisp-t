# Agent Instructions for CRISP-T Repository


## Agent development and contribution guidelines for CRISP-T.

- Follow the repository structure and coding conventions
- All new classes should accept the corpus as their input unless otherwise justified
- When a corpus is modified set the self.corpus to the modified corpus
- When a dataframe is modified set self.corpus.df to the modified dataframe
- Add appropriate command-line interface (CLI) commands for new features
- Add appropriate MCP tools for new analysis functions
- Update README and documentation for user-facing changes and CLI commands
- Write tests for new functionality and maintain existing test coverage

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
   uv pip install -e ".[ml,xg,dev]"
   ```

3. **Run tests**:
   ```bash
   uv run pytest
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


