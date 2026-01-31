"""
Tests for individual tool modules.

These tests verify that each tool module has the correct structure
and can be imported without errors.
"""

import pytest


def test_corpus_management_module():
    """Test corpus management module structure."""
    from src.crisp_t.mcp.tools.corpus_management import (
        get_corpus_management_tools,
        handle_corpus_management_tool,
    )
    
    # Test that we can get tools
    tools = get_corpus_management_tools()
    assert len(tools) == 9
    
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "load_corpus",
        "save_corpus",
        "add_document",
        "remove_document",
        "get_document",
        "list_documents",
        "add_relationship",
        "get_relationships",
        "get_relationships_for_keyword",
    ]
    
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"


def test_nlp_analysis_module():
    """Test NLP analysis module structure."""
    from src.crisp_t.mcp.tools.nlp_analysis import (
        get_nlp_analysis_tools,
        handle_nlp_analysis_tool,
    )
    
    tools = get_nlp_analysis_tools()
    assert len(tools) == 6
    
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "generate_coding_dictionary",
        "topic_modeling",
        "assign_topics",
        "extract_categories",
        "generate_summary",
        "sentiment_analysis",
    ]
    
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"


def test_corpus_filtering_module():
    """Test corpus filtering module structure."""
    from src.crisp_t.mcp.tools.corpus_filtering import (
        get_corpus_filtering_tools,
        handle_corpus_filtering_tool,
    )
    
    tools = get_corpus_filtering_tools()
    assert len(tools) == 2
    
    tool_names = [tool.name for tool in tools]
    assert "filter_documents" in tool_names
    assert "document_count" in tool_names


def test_dataframe_operations_module():
    """Test dataframe operations module structure."""
    from src.crisp_t.mcp.tools.dataframe_operations import (
        get_dataframe_operations_tools,
        handle_dataframe_operations_tool,
    )
    
    tools = get_dataframe_operations_tools()
    assert len(tools) == 3
    
    tool_names = [tool.name for tool in tools]
    assert "get_df_columns" in tool_names
    assert "get_df_row_count" in tool_names
    assert "get_df_row" in tool_names


def test_column_operations_module():
    """Test column operations module structure."""
    from src.crisp_t.mcp.tools.column_operations import (
        get_column_operations_tools,
        handle_column_operations_tool,
    )
    
    tools = get_column_operations_tools()
    assert len(tools) == 8
    
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "bin_a_column",
        "one_hot_encode_column",
        "filter_rows_by_column_value",
        "oversample",
        "restore_oversample",
        "get_column_types",
        "get_column_values",
        "retain_numeric_columns_only",
    ]
    
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"


def test_semantic_search_module():
    """Test semantic search module structure."""
    from src.crisp_t.mcp.tools.semantic_search import (
        get_semantic_search_tools,
        handle_semantic_search_tool,
    )
    
    tools = get_semantic_search_tools()
    assert len(tools) == 4
    
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "semantic_search",
        "find_similar_documents",
        "semantic_chunk_search",
        "export_metadata_df",
    ]
    
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"


def test_topological_analysis_module():
    """Test topological analysis module structure."""
    from src.crisp_t.mcp.tools.topological_analysis import (
        get_topological_analysis_tools,
        handle_topological_analysis_tool,
    )
    
    tools = get_topological_analysis_tools()
    assert len(tools) == 1
    
    tool_names = [tool.name for tool in tools]
    assert "tdabm_analysis" in tool_names


def test_temporal_analysis_module():
    """Test temporal analysis module structure."""
    from src.crisp_t.mcp.tools.temporal_analysis import (
        get_temporal_analysis_tools,
        handle_temporal_analysis_tool,
    )
    
    tools = get_temporal_analysis_tools()
    assert len(tools) == 5
    
    tool_names = [tool.name for tool in tools]
    expected_names = [
        "temporal_link_by_time",
        "temporal_filter",
        "temporal_summary",
        "temporal_sentiment_trend",
        "temporal_topics",
    ]
    
    for name in expected_names:
        assert name in tool_names, f"Missing tool: {name}"


def test_embedding_linking_module():
    """Test embedding linking module structure."""
    from src.crisp_t.mcp.tools.embedding_linking import (
        get_embedding_linking_tools,
        handle_embedding_linking_tool,
    )
    
    tools = get_embedding_linking_tools()
    assert len(tools) == 2
    
    tool_names = [tool.name for tool in tools]
    assert "embedding_link" in tool_names
    assert "embedding_link_stats" in tool_names


def test_misc_tools_module():
    """Test misc tools module structure."""
    from src.crisp_t.mcp.tools.misc_tools import (
        get_misc_tools,
        handle_misc_tool,
    )
    
    tools = get_misc_tools()
    assert len(tools) == 2
    
    tool_names = [tool.name for tool in tools]
    assert "reset_corpus_state" in tool_names
    assert "clear_cache" in tool_names


def test_ml_tools_module():
    """Test ML tools module structure (if available)."""
    try:
        from src.crisp_t.mcp.tools.ml_tools import (
            get_ml_tools,
            handle_ml_tool,
            ML_AVAILABLE,
        )
        
        if ML_AVAILABLE:
            tools = get_ml_tools()
            assert len(tools) == 9
            
            tool_names = [tool.name for tool in tools]
            expected_names = [
                "kmeans_clustering",
                "decision_tree_classification",
                "svm_classification",
                "neural_network_classification",
                "regression_analysis",
                "pca_analysis",
                "association_rules",
                "knn_search",
                "lstm_text_classification",
            ]
            
            for name in expected_names:
                assert name in tool_names, f"Missing tool: {name}"
        else:
            # If ML not available, get_ml_tools should return empty list
            tools = get_ml_tools()
            assert len(tools) == 0
    except ImportError:
        # ML module might not be available
        pytest.skip("ML tools module not available")


def test_tools_init_module():
    """Test tools __init__ module integration."""
    from src.crisp_t.mcp.tools import (
        get_all_tools,
        handle_tool_call,
        ML_TOOLS_AVAILABLE,
    )
    
    # Test that get_all_tools combines all tools
    all_tools = get_all_tools()
    assert len(all_tools) >= 42  # At least 42 tools (non-ML)
    
    # Check that all expected tool names are present
    tool_names = [tool.name for tool in all_tools]
    
    # Check for some key tools from different categories
    assert "load_corpus" in tool_names
    assert "sentiment_analysis" in tool_names
    assert "filter_documents" in tool_names
    assert "get_df_columns" in tool_names
    assert "semantic_search" in tool_names
    assert "temporal_summary" in tool_names
    
    # Test that handle_tool_call returns None for unknown tools
    result = handle_tool_call(
        "nonexistent_tool", {}, None, None, None, None
    )
    assert result is None


def test_all_tools_have_required_fields():
    """Test that all tools have required schema fields."""
    from src.crisp_t.mcp.tools import get_all_tools
    
    all_tools = get_all_tools()
    
    for tool in all_tools:
        # Check required fields
        assert tool.name, f"Tool missing name: {tool}"
        assert tool.description, f"Tool {tool.name} missing description"
        assert tool.inputSchema, f"Tool {tool.name} missing inputSchema"
        
        # Check inputSchema structure
        schema = tool.inputSchema
        assert "type" in schema, f"Tool {tool.name} schema missing type"
        assert schema["type"] == "object", f"Tool {tool.name} schema type should be object"
        assert "properties" in schema, f"Tool {tool.name} schema missing properties"


def test_handler_functions_exist():
    """Test that all handler functions exist and have correct signature."""
    import inspect
    
    from src.crisp_t.mcp.tools.corpus_management import handle_corpus_management_tool
    from src.crisp_t.mcp.tools.nlp_analysis import handle_nlp_analysis_tool
    from src.crisp_t.mcp.tools.corpus_filtering import handle_corpus_filtering_tool
    
    handlers = [
        handle_corpus_management_tool,
        handle_nlp_analysis_tool,
        handle_corpus_filtering_tool,
    ]
    
    for handler in handlers:
        # Check that handler is callable
        assert callable(handler)
        
        # Check signature has expected parameters
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        assert "name" in params
        assert "arguments" in params
        assert "corpus" in params
