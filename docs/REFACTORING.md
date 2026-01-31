# MCP Server Refactoring Documentation

This document describes the refactoring of `mcp/server.py` into a modular structure.

## Overview

The `mcp/server.py` file was refactored from a monolithic 2560-line file into a modular architecture with 11 focused tool modules, reducing the main server file to just 233 lines (91% reduction).

## Goals

1. **Improve Maintainability**: Break down the large server file into logical, manageable components
2. **Enhance Readability**: Group related tools together in dedicated modules
3. **Facilitate Testing**: Enable targeted testing of individual tool categories
4. **Enable Scalability**: Make it easier to add new tools in appropriate categories
5. **Better Organization**: Separate concerns between server infrastructure and tool implementations

## New Structure

### Directory Layout

```
src/crisp_t/mcp/
├── __init__.py
├── __main__.py
├── server.py (233 lines, down from 2560)
├── prompts/
│   ├── __init__.py
│   ├── analysis_workflow.txt
│   └── triangulation_guide.txt
├── tools/
│   ├── __init__.py (Tool registry and router)
│   ├── corpus_management.py (9 tools)
│   ├── nlp_analysis.py (6 tools)
│   ├── corpus_filtering.py (2 tools)
│   ├── dataframe_operations.py (3 tools)
│   ├── column_operations.py (8 tools)
│   ├── ml_tools.py (9 tools, conditional)
│   ├── semantic_search.py (4 tools)
│   ├── topological_analysis.py (1 tool)
│   ├── temporal_analysis.py (5 tools)
│   ├── embedding_linking.py (2 tools)
│   └── misc_tools.py (2 tools)
└── utils/
    ├── __init__.py
    ├── responses.py
    └── validators.py
```

### Tool Categories

#### 1. Corpus Management (`corpus_management.py`)
- **Tools**: 9
- **Purpose**: Core corpus operations (loading, saving, document management, relationships)
- **Key Tools**: `load_corpus`, `save_corpus`, `add_document`, `remove_document`, `get_document`, `list_documents`, `add_relationship`, `get_relationships`, `get_relationships_for_keyword`

#### 2. NLP/Text Analysis (`nlp_analysis.py`)
- **Tools**: 6
- **Purpose**: Natural language processing and text analysis
- **Key Tools**: `generate_coding_dictionary`, `topic_modeling`, `assign_topics`, `extract_categories`, `generate_summary`, `sentiment_analysis`

#### 3. Corpus Filtering (`corpus_filtering.py`)
- **Tools**: 2
- **Purpose**: Document filtering and counting operations
- **Key Tools**: `filter_documents`, `document_count`

#### 4. DataFrame Operations (`dataframe_operations.py`)
- **Tools**: 3
- **Purpose**: Basic DataFrame inspection and access
- **Key Tools**: `get_df_columns`, `get_df_row_count`, `get_df_row`

#### 5. Column Operations (`column_operations.py`)
- **Tools**: 8
- **Purpose**: DataFrame column manipulation and transformation
- **Key Tools**: `bin_a_column`, `one_hot_encode_column`, `filter_rows_by_column_value`, `oversample`, `restore_oversample`, `get_column_types`, `get_column_values`, `retain_numeric_columns_only`

#### 6. ML Tools (`ml_tools.py`)
- **Tools**: 9 (conditional on ML dependencies)
- **Purpose**: Machine learning algorithms and analysis
- **Key Tools**: `kmeans_clustering`, `decision_tree_classification`, `svm_classification`, `neural_network_classification`, `regression_analysis`, `pca_analysis`, `association_rules`, `knn_search`, `lstm_text_classification`

#### 7. Semantic Search (`semantic_search.py`)
- **Tools**: 4
- **Purpose**: Embedding-based semantic search operations
- **Key Tools**: `semantic_search`, `find_similar_documents`, `semantic_chunk_search`, `export_metadata_df`

#### 8. Topological Analysis (`topological_analysis.py`)
- **Tools**: 1
- **Purpose**: Topological data analysis using Ball Mapper
- **Key Tools**: `tdabm_analysis`

#### 9. Temporal Analysis (`temporal_analysis.py`)
- **Tools**: 5
- **Purpose**: Time-based analysis and temporal linking
- **Key Tools**: `temporal_link_by_time`, `temporal_filter`, `temporal_summary`, `temporal_sentiment_trend`, `temporal_topics`

#### 10. Embedding Linking (`embedding_linking.py`)
- **Tools**: 2
- **Purpose**: Linking documents to dataframe rows using embeddings
- **Key Tools**: `embedding_link`, `embedding_link_stats`

#### 11. Miscellaneous Tools (`misc_tools.py`)
- **Tools**: 2
- **Purpose**: Utility operations (state management, cache clearing)
- **Key Tools**: `reset_corpus_state`, `clear_cache`

## Implementation Pattern

Each tool module follows a consistent pattern:

### Module Structure

```python
"""
[Module Name] Tools for MCP Server

Brief description of the tool category.
"""

import logging
from typing import Any

from mcp.types import TextContent, Tool

from ..utils.responses import (
    error_response,
    success_response,
    # ... other response helpers
)

logger = logging.getLogger(__name__)


def get_[category]_tools() -> list[Tool]:
    """Return list of [category] tool definitions."""
    return [
        Tool(
            name="tool_name",
            description="Tool description...",
            inputSchema={
                "type": "object",
                "properties": {
                    # ... schema properties
                },
                "required": [...]
            }
        ),
        # ... more tools
    ]


def handle_[category]_tool(
    name: str,
    arguments: dict[str, Any],
    corpus: Any,
    text_analyzer: Any,
    csv_analyzer: Any,
    ml_analyzer: Any,
) -> tuple[list[TextContent], Any, Any] | None:
    """
    Handle [category] tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Current corpus
        text_analyzer: Current text analyzer
        csv_analyzer: Current CSV analyzer
        ml_analyzer: Current ML analyzer
        
    Returns:
        Tuple of (response, updated_corpus, updated_ml_analyzer) or None if tool not handled
    """
    if name == "tool_name":
        # Implementation
        return response, updated_corpus, updated_ml_analyzer
    
    # ... more handlers
    
    # Tool not handled by this module
    return None
```

### Special Cases

#### ML Tools Module
The ML tools module handles conditional imports:

```python
# Try to import ML if available
try:
    from ...ml import ML
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available")
```

#### Misc Tools Module
Returns a 5-tuple to allow resetting all analyzers:

```python
return (response, corpus, text_analyzer, csv_analyzer, ml_analyzer)
```

## Tool Registry (`tools/__init__.py`)

The tool registry aggregates all tools and routes handler calls:

### Key Functions

#### `get_all_tools()`
Collects tools from all modules:

```python
def get_all_tools():
    """Get all available tools from all categories."""
    tools = []
    tools.extend(get_corpus_management_tools())
    tools.extend(get_nlp_analysis_tools())
    # ... other categories
    
    if ML_TOOLS_AVAILABLE:
        tools.extend(get_ml_tools())
    
    return tools
```

#### `handle_tool_call()`
Routes tool calls to appropriate handlers:

```python
def handle_tool_call(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer):
    """Route tool calls to appropriate handlers based on tool name."""
    
    # Try misc_tool first (returns 5-tuple)
    result = handle_misc_tool(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer)
    if result is not None:
        return result
    
    # Try other handlers (return 3-tuple)
    handlers = [
        handle_corpus_management_tool,
        handle_nlp_analysis_tool,
        # ... other handlers
    ]
    
    for handler in handlers:
        result = handler(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer)
        if result is not None:
            response, updated_corpus, updated_ml_analyzer = result
            return response, updated_corpus, text_analyzer, csv_analyzer, updated_ml_analyzer
    
    return None
```

## Server Integration

The refactored `server.py` imports and uses the modular tools:

```python
from .tools import get_all_tools, handle_tool_call
from .prompts import ANALYSIS_WORKFLOW, TRIANGULATION_GUIDE

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools from all tool modules."""
    return get_all_tools()

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls by routing to appropriate tool modules."""
    global _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer

    try:
        result = handle_tool_call(
            name, arguments, _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer
        )
        
        if result is not None:
            response, _corpus, _text_analyzer, _csv_analyzer, _ml_analyzer = result
            return response
        else:
            return error_response(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return error_response(str(e))
```

## Testing

### Structure Tests (`tests/mcp/test_structure.py`)
- Validates server.py uses modular imports
- Checks for tool module existence
- Verifies each module has required functions

### Module Tests (`tests/mcp/test_tool_modules.py`)
- Tests each tool module individually
- Validates tool counts
- Checks tool schemas
- Verifies handler function signatures

### Integration Tests (`tests/mcp/test_server.py`)
- Tests the full MCP server integration
- Validates tool listing
- Tests tool execution

## Benefits

1. **Maintainability**: 
   - Each module is focused on a single category
   - Easy to locate and modify specific tools
   - Reduced cognitive load

2. **Testability**:
   - Can test individual modules in isolation
   - Easier to mock dependencies
   - Faster test execution for specific categories

3. **Scalability**:
   - Clear pattern for adding new tools
   - Easy to add new categories
   - Minimal changes to core server code

4. **Code Quality**:
   - Better separation of concerns
   - Cleaner imports
   - More modular dependencies

5. **Team Collaboration**:
   - Multiple developers can work on different tool categories
   - Reduced merge conflicts
   - Clearer code ownership

## Migration Guide

### Adding a New Tool

1. Identify the appropriate tool category module
2. Add the tool definition to `get_[category]_tools()`
3. Add the handler to `handle_[category]_tool()`
4. Add test to `tests/mcp/test_tool_modules.py`

### Adding a New Category

1. Create new file in `src/crisp_t/mcp/tools/[category].py`
2. Implement `get_[category]_tools()` and `handle_[category]_tool()`
3. Import and add to `tools/__init__.py`:
   - Import functions
   - Add to `get_all_tools()`
   - Add to handlers list in `handle_tool_call()`
4. Add tests to `tests/mcp/test_tool_modules.py`

## Backward Compatibility

The refactoring maintains 100% backward compatibility:

- All 51 tools remain available
- Tool names unchanged
- Tool schemas unchanged
- Handler behavior unchanged
- API surface unchanged

The only changes are internal organization - external consumers see no difference.

## Performance Considerations

The modular structure has minimal performance impact:

- Tool definitions are created once at import time
- Handler routing adds negligible overhead (simple if/else chain)
- No additional network or I/O operations
- Memory usage similar (all tools still loaded)

## Future Improvements

Potential future enhancements:

1. **Lazy Loading**: Load tool modules on-demand
2. **Plugin System**: Allow external tool modules
3. **Tool Versioning**: Support multiple versions of tools
4. **Tool Categories API**: Allow querying tools by category
5. **Dynamic Tool Registration**: Register tools at runtime

## Conclusion

The refactoring successfully transformed a monolithic 2560-line server file into a well-organized modular architecture with 11 focused tool modules. This improves maintainability, testability, and scalability while maintaining 100% backward compatibility.
