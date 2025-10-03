import click
from ..text import Text
from ..csv import Csv

def get_text_analyzer(corpus, filters=None):
    """Initialize Text analyzer with corpus and apply optional filters.
    Args:
        corpus (Corpus): The text corpus to analyze.
        filters (list, optional): List of filters in key=value or key:value format to apply on documents.
    Returns:
        Text: Initialized Text analyzer with applied filters.
    """
    text_analyzer = Text(corpus=corpus)
    # Apply filters if provided
    if filters:
        try:
            for flt in filters:
                if "=" in flt:
                    key, value = flt.split("=", 1)
                elif ":" in flt:
                    key, value = flt.split(":", 1)
                else:
                    raise ValueError("Filter must be in key=value or key:value format")
                text_analyzer.filter_documents(key.strip(), value.strip())
            click.echo(
                f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
            )
        except Exception as e:
            # Surface as CLI error with non-zero exit code
            click.echo(f"Probably no document metadata to filter, but let me check numeric metadata: {e}")
    return text_analyzer

def get_csv_analyzer(
    corpus,
    comma_separated_unstructured_text_columns=None,
    comma_separated_ignore_columns=None,
    filters=None
):
    if corpus and corpus.df is not None:
        click.echo("Loading CSV data from corpus.df")
        csv_analyzer = Csv(corpus=corpus)
        csv_analyzer.df = corpus.df
        text_columns, ignore_columns = _process_csv(
            csv_analyzer, comma_separated_unstructured_text_columns, comma_separated_ignore_columns, filters
        )
        click.echo(f"Loaded CSV with shape: {csv_analyzer.get_shape()}")
        return csv_analyzer
    else:
        raise ValueError("Corpus or corpus.df is not set")


def _process_csv(
    csv_analyzer,
    comma_separated_unstructured_text_columns=None,
    comma_separated_ignore_columns=None,
    filters=None
):
    text_columns = comma_separated_unstructured_text_columns if comma_separated_unstructured_text_columns else ""
    ignore_columns = comma_separated_ignore_columns if comma_separated_ignore_columns else ""
    csv_analyzer.comma_separated_text_columns = text_columns
    csv_analyzer.comma_separated_ignore_columns = ignore_columns
    if filters:
        try:
            for flt in filters:
                if "=" in flt:
                    key, value = flt.split("=", 1)
                elif ":" in flt:
                    key, value = flt.split(":", 1)
                else:
                    raise ValueError("Filter must be in key=value or key:value format")
                csv_analyzer.filter_rows_by_column_value(key.strip(), value.strip())
            click.echo(
                f"Applied filters {list(filters)}; remaining rows: {csv_analyzer.get_shape()[0]}"
            )
        except Exception as e:
            # Surface as CLI error with non-zero exit code
            click.echo(
                f"Probably no numeric metadata to filter, but let me check document metadata: {e}"
            )
    return text_columns, ignore_columns