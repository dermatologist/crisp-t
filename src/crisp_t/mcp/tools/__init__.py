"""
Tools module for CRISP-T MCP Server.

This module organizes MCP tools into logical categories for better maintainability.
"""

from .corpus_management import get_corpus_management_tools, handle_corpus_management_tool
from .nlp_analysis import get_nlp_analysis_tools, handle_nlp_analysis_tool
from .corpus_filtering import get_corpus_filtering_tools, handle_corpus_filtering_tool
from .dataframe_operations import get_dataframe_operations_tools, handle_dataframe_operations_tool
from .column_operations import get_column_operations_tools, handle_column_operations_tool
from .data_analysis import get_data_analysis_tools, handle_data_analysis_tool
from .semantic_search import get_semantic_search_tools, handle_semantic_search_tool
from .topological_analysis import get_topological_analysis_tools, handle_topological_analysis_tool
from .temporal_analysis import get_temporal_analysis_tools, handle_temporal_analysis_tool
from .embedding_linking import get_embedding_linking_tools, handle_embedding_linking_tool
from .misc_tools import get_misc_tools, handle_misc_tool

# ML tools are optional
try:
    from .ml_tools import get_ml_tools, handle_ml_tool
    ML_TOOLS_AVAILABLE = True
except ImportError:
    ML_TOOLS_AVAILABLE = False


def get_all_tools():
    """Get all available tools from all categories."""
    tools = []
    tools.extend(get_corpus_management_tools())
    tools.extend(get_nlp_analysis_tools())
    tools.extend(get_corpus_filtering_tools())
    tools.extend(get_dataframe_operations_tools())
    tools.extend(get_column_operations_tools())
    tools.extend(get_data_analysis_tools())
    tools.extend(get_semantic_search_tools())
    tools.extend(get_topological_analysis_tools())
    tools.extend(get_temporal_analysis_tools())
    tools.extend(get_embedding_linking_tools())
    tools.extend(get_misc_tools())
    
    if ML_TOOLS_AVAILABLE:
        tools.extend(get_ml_tools())
    
    return tools


def handle_tool_call(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer):
    """
    Route tool calls to appropriate handlers based on tool name.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        corpus: Global corpus instance
        text_analyzer: Global text analyzer instance
        csv_analyzer: Global CSV analyzer instance
        ml_analyzer: Global ML analyzer instance
        
    Returns:
        tuple: (response, updated_corpus, updated_text_analyzer, updated_csv_analyzer, updated_ml_analyzer) 
    """
    # Try misc_tool first (it returns 5-tuple to reset all state)
    result = handle_misc_tool(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer)
    if result is not None:
        return result
    
    # Try other category handlers (they return 3-tuple: response, corpus, ml_analyzer)
    handlers = [
        handle_corpus_management_tool,
        handle_nlp_analysis_tool,
        handle_corpus_filtering_tool,
        handle_dataframe_operations_tool,
        handle_column_operations_tool,
        handle_data_analysis_tool,
        handle_semantic_search_tool,
        handle_topological_analysis_tool,
        handle_temporal_analysis_tool,
        handle_embedding_linking_tool,
    ]
    
    if ML_TOOLS_AVAILABLE:
        handlers.append(handle_ml_tool)
    
    for handler in handlers:
        result = handler(name, arguments, corpus, text_analyzer, csv_analyzer, ml_analyzer)
        if result is not None:
            # Convert 3-tuple to 5-tuple by keeping text_analyzer and csv_analyzer unchanged
            response, updated_corpus, updated_ml_analyzer = result
            return response, updated_corpus, text_analyzer, csv_analyzer, updated_ml_analyzer
    
    # Tool not found in any handler
    return None


__all__ = [
    "get_all_tools",
    "handle_tool_call",
    "ML_TOOLS_AVAILABLE",
]
