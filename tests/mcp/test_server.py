"""
Tests for the MCP server
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.server import ServerRequestContext
from mcp.types import (
    CallToolRequestParams,
    GetPromptRequestParams,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    TextContent,
)
from pydantic import AnyUrl

from src.crisp_t.mcp.server import _init_corpus, app
from src.crisp_t.mcp.server import handle_call_tool as call_tool
from src.crisp_t.mcp.server import handle_get_prompt as get_prompt
from src.crisp_t.mcp.server import handle_list_prompts as list_prompts
from src.crisp_t.mcp.server import handle_list_resources as list_resources
from src.crisp_t.mcp.server import handle_list_tools as list_tools
from src.crisp_t.mcp.server import handle_read_resource as read_resource


@pytest.mark.asyncio
async def test_list_tools():
    """Test that tools are listed correctly."""
    # Create mock ctx and params
    mock_ctx = Mock(spec=ServerRequestContext)
    result = await list_tools(mock_ctx, None)
    tools = result.tools

    assert len(tools) > 0
    tool_names = [tool.name for tool in tools]

    # Check for essential corpus management tools
    assert "load_corpus" in tool_names
    assert "save_corpus" in tool_names
    assert "add_document" in tool_names
    assert "remove_document" in tool_names
    assert "list_documents" in tool_names

    # Check for NLP tools
    # assert "generate_coding_dictionary" in tool_names
    # assert "topic_modeling" in tool_names
    assert "assign_topics" in tool_names
    assert "sentiment_analysis" in tool_names

    # Check for relationship tools
    assert "add_relationship" in tool_names
    assert "get_relationships" in tool_names

    # Check that each tool has required fields
    for tool in tools:
        assert tool.name
        assert tool.description
        assert tool.input_schema


@pytest.mark.asyncio
async def test_list_resources_empty():
    """Test listing resources when no corpus is loaded."""
    # Reset global state
    import src.crisp_t.mcp.server as server_module

    server_module._corpus = None

    mock_ctx = Mock(spec=ServerRequestContext)
    result = await list_resources(mock_ctx, None)
    assert result.resources == []


@pytest.mark.asyncio
async def test_list_prompts():
    """Test that prompts are listed correctly."""
    mock_ctx = Mock(spec=ServerRequestContext)
    result = await list_prompts(mock_ctx, None)
    prompts = result.prompts

    assert len(prompts) > 0
    prompt_names = [prompt.name for prompt in prompts]

    assert "analysis_workflow" in prompt_names
    assert "triangulation_guide" in prompt_names


@pytest.mark.asyncio
async def test_get_prompt_analysis_workflow():
    """Test getting the analysis workflow prompt."""
    mock_ctx = Mock(spec=ServerRequestContext)
    params = GetPromptRequestParams(name="analysis_workflow", arguments={})
    result = await get_prompt(mock_ctx, params)

    assert result.description
    assert len(result.messages) > 0
    assert result.messages[0].role == "user"
    assert "Follow these steps" in result.messages[0].content.text


@pytest.mark.asyncio
async def test_get_prompt_triangulation_guide():
    """Test getting the triangulation guide prompt."""
    mock_ctx = Mock(spec=ServerRequestContext)
    params = GetPromptRequestParams(name="triangulation_guide", arguments={})
    result = await get_prompt(mock_ctx, params)

    assert result.description
    assert len(result.messages) > 0
    assert "Triangulation" in result.messages[0].content.text


@pytest.mark.asyncio
async def test_get_prompt_unknown():
    """Test getting an unknown prompt raises error."""
    mock_ctx = Mock(spec=ServerRequestContext)
    params = GetPromptRequestParams(name="nonexistent_prompt", arguments={})
    with pytest.raises(ValueError, match="Unknown prompt"):
        await get_prompt(mock_ctx, params)


@pytest.mark.asyncio
async def test_call_tool_no_corpus():
    """Test calling tools without a loaded corpus."""
    import src.crisp_t.mcp.server as server_module

    server_module._corpus = None
    server_module._text_analyzer = None

    # Test that tools requiring corpus return appropriate message
    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="list_documents", arguments={})
    result = await call_tool(mock_ctx, params)
    assert len(result.content) == 1
    assert "No corpus loaded" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_add_document():
    """Test adding a document."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    # Create a minimal corpus
    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], metadata={}
    )

    # Add a document
    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(
        name="add_document",
        arguments={"doc_id": "doc1", "text": "Test document text", "name": "Test Doc"},
    )
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    assert "added" in result.content[0].text.lower()
    assert len(server_module._corpus.documents) == 1
    assert server_module._corpus.documents[0].id == "doc1"


@pytest.mark.asyncio
async def test_call_tool_list_documents():
    """Test listing documents."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    # Create a corpus with documents
    doc1 = Document(id="doc1", text="Text 1", name="Doc 1", score=0.0, metadata={})
    doc2 = Document(id="doc2", text="Text 2", name="Doc 2", score=0.0, metadata={})

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[doc1, doc2], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="list_documents", arguments={})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    doc_ids = json.loads(result.content[0].text)
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


@pytest.mark.asyncio
async def test_call_tool_remove_document():
    """Test removing a document."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    # Create a corpus with a document
    doc1 = Document(id="doc1", text="Text 1", name="Doc 1", score=0.0, metadata={})

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[doc1], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="remove_document", arguments={"doc_id": "doc1"})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    assert "removed" in result.content[0].text.lower()
    assert len(server_module._corpus.documents) == 0


@pytest.mark.asyncio
async def test_call_tool_add_relationship():
    """Test adding a relationship."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(
        name="add_relationship",
        arguments={
            "first": "text:health",
            "second": "num:age",
            "relation": "correlates",
        },
    )
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    assert "added" in result.content[0].text.lower()


@pytest.mark.asyncio
async def test_call_tool_get_relationships():
    """Test getting relationships."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], metadata={}
    )

    # Add a relationship first
    server_module._corpus.add_relationship("text:health", "num:age", "correlates")

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="get_relationships", arguments={})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    rels = json.loads(result.content[0].text)
    assert len(rels) > 0


@pytest.mark.asyncio
async def test_call_tool_get_df_columns():
    """Test getting DataFrame columns."""
    import pandas as pd

    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    # Create corpus with DataFrame
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], df=df, metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="get_df_columns", arguments={})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    cols = json.loads(result.content[0].text)
    assert "col1" in cols
    assert "col2" in cols


@pytest.mark.asyncio
async def test_call_tool_get_df_row_count():
    """Test getting DataFrame row count."""
    import pandas as pd

    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    df = pd.DataFrame({"col1": [1, 2, 3]})
    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], df=df, metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="get_df_row_count", arguments={})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    assert "3" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_unknown():
    """Test calling an unknown tool."""
    mock_ctx = Mock(spec=ServerRequestContext)
    params = CallToolRequestParams(name="nonexistent_tool", arguments={})
    result = await call_tool(mock_ctx, params)

    assert len(result.content) == 1
    assert "Unknown tool" in result.content[0].text


@pytest.mark.asyncio
async def test_read_resource_invalid_uri():
    """Test reading an invalid resource URI."""
    mock_ctx = Mock(spec=ServerRequestContext)
    params = ReadResourceRequestParams(uri="invalid://uri")
    with pytest.raises(ValueError, match="Unknown resource URI"):
        await read_resource(mock_ctx, params)


@pytest.mark.asyncio
async def test_read_resource_no_corpus():
    """Test reading a resource when no corpus is loaded."""
    import src.crisp_t.mcp.server as server_module

    server_module._corpus = None

    mock_ctx = Mock(spec=ServerRequestContext)
    params = ReadResourceRequestParams(uri="corpus://document/doc1")
    with pytest.raises(ValueError, match="No corpus loaded"):
        await read_resource(mock_ctx, params)


@pytest.mark.asyncio
async def test_read_resource_document_not_found():
    """Test reading a non-existent document."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = ReadResourceRequestParams(uri="corpus://document/nonexistent")
    with pytest.raises(ValueError, match="Document not found"):
        await read_resource(mock_ctx, params)


@pytest.mark.asyncio
async def test_read_resource_success():
    """Test successfully reading a document resource."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    doc = Document(
        id="doc1",
        text="This is test document text",
        name="Test Doc",
        score=0.0,
        metadata={},
    )

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[doc], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = ReadResourceRequestParams(uri="corpus://document/doc1")
    result = await read_resource(mock_ctx, params)
    assert result is not None
    assert len(result.contents) == 1


@pytest.mark.asyncio
async def test_list_resources_with_documents():
    """Test listing resources when documents are present."""
    import src.crisp_t.mcp.server as server_module
    from src.crisp_t.model.corpus import Corpus
    from src.crisp_t.model.document import Document

    doc1 = Document(id="doc1", text="Text 1", name="Doc 1", score=0.0, metadata={})
    doc2 = Document(
        id="doc2",
        text="Text 2",
        name="Doc 2",
        description="Desc 2",
        score=0.0,
        metadata={},
    )

    server_module._corpus = Corpus(
        id="test", name="Test Corpus", documents=[doc1, doc2], metadata={}
    )

    mock_ctx = Mock(spec=ServerRequestContext)
    params = PaginatedRequestParams()
    result = await list_resources(mock_ctx, params)
    resources = result.resources

    assert len(resources) == 2
    assert str(resources[0].uri) == "corpus://document/doc1"
    assert str(resources[1].uri) == "corpus://document/doc2"
    assert "Doc 1" in resources[0].name
    assert "Doc 2" in resources[1].name
