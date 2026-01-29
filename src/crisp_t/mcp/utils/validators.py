"""Validation utilities for MCP server handlers."""
from collections.abc import Callable
from functools import wraps

from .responses import no_corpus_response, no_csv_analyzer_response, no_text_analyzer_response


def require_corpus(func: Callable) -> Callable:
    """Decorator to ensure corpus is loaded before executing handler."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Import here to avoid circular dependency
        from ..server import _corpus

        if _corpus is None:
            return no_corpus_response()
        return await func(*args, **kwargs)
    return wrapper


def require_csv_analyzer(func: Callable) -> Callable:
    """Decorator to ensure CSV analyzer is available before executing handler."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        from ..server import _csv_analyzer

        if _csv_analyzer is None:
            return no_csv_analyzer_response()
        return await func(*args, **kwargs)
    return wrapper


def require_text_analyzer(func: Callable) -> Callable:
    """Decorator to ensure text analyzer is available before executing handler."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        from ..server import _text_analyzer

        if _text_analyzer is None:
            return no_text_analyzer_response()
        return await func(*args, **kwargs)
    return wrapper
