"""Execution utilities for CLI analysis operations."""
from collections.abc import Callable
from typing import Any

import click

from .ui import format_error, format_success


def execute_analysis_with_save(
    analysis_func: Callable[[], Any],
    output_path: str | None = None,
    suffix: str = "results",
    success_message: str = "Analysis completed successfully",
    error_message_prefix: str = "Error performing analysis",
) -> Any:
    """
    Execute an analysis function with automatic error handling and optional output saving.

    Args:
        analysis_func: The analysis function to execute (no arguments)
        output_path: Optional path to save results
        suffix: Suffix for the output filename
        success_message: Message to display on success
        error_message_prefix: Prefix for error messages

    Returns:
        The result of the analysis function, or None if an error occurred
    """
    try:
        result = analysis_func()

        if output_path:
            # Import from cli module where _save_output is defined
            from ...cli import _save_output
            _save_output(result, output_path, suffix)
            click.echo(format_success(f"{success_message} - Saved to {output_path}"))
        else:
            click.echo(format_success(success_message))

        return result

    except Exception as e:
        click.echo(format_error(f"{error_message_prefix}: {e}"))
        return None
