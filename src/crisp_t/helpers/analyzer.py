import click
import pandas as pd

from ..csv import Csv
from ..text import Text


def get_analyzers(
    corpus,
    comma_separated_unstructured_text_columns=None,
    comma_separated_ignore_columns=None,
    filters=None,
):
    """Initialize both Text and Csv analyzers with unified filter logic.

    Supports special filter keywords for linking:
    - =embedding or :embedding - Filter dataframe rows linked via embedding_links
    - =temporal or :temporal - Filter dataframe rows linked via temporal_links

    Args:
        corpus (Corpus): The corpus to analyze.
        comma_separated_unstructured_text_columns (str, optional): CSV columns with free-text.
        comma_separated_ignore_columns (str, optional): Columns to ignore.
        filters (list, optional): List of filters in key=value or key:value format.
            Special filters: =embedding, :embedding, =temporal, :temporal

    Returns:
        tuple: (text_analyzer, csv_analyzer) - Both initialized and filtered analyzers
    """
    text_analyzer = None
    csv_analyzer = None

    # Initialize text analyzer
    if corpus and hasattr(corpus, "documents") and corpus.documents:
        text_analyzer = Text(corpus=corpus)

    # Initialize csv analyzer
    if corpus and corpus.df is not None:
        csv_analyzer = Csv(corpus=corpus)
        csv_analyzer.df = corpus.df
        text_columns = (
            comma_separated_unstructured_text_columns
            if comma_separated_unstructured_text_columns
            else ""
        )
        ignore_columns = (
            comma_separated_ignore_columns if comma_separated_ignore_columns else ""
        )
        csv_analyzer.comma_separated_text_columns = text_columns
        csv_analyzer.comma_separated_ignore_columns = ignore_columns
        click.echo(f"Loaded CSV with shape: {csv_analyzer.get_shape()}")

    # Apply filters
    if filters and (text_analyzer or csv_analyzer):
        # Separate regular filters from special link filters
        regular_filters = []
        link_filters = []

        for flt in filters:
            # Check for special linking filters
            if flt in ["=embedding", ":embedding", "=temporal", ":temporal"]:
                link_filters.append(flt)
            else:
                regular_filters.append(flt)

        # Apply regular filters first
        if regular_filters:
            _apply_regular_filters(text_analyzer, csv_analyzer, regular_filters)

        # Apply link-based filters
        if link_filters:
            _apply_link_filters(corpus, text_analyzer, csv_analyzer, link_filters)

    return text_analyzer, csv_analyzer


def _apply_regular_filters(text_analyzer, csv_analyzer, filters):
    """Apply regular key=value or key:value filters to analyzers."""
    try:
        for flt in filters:
            if "=" in flt:
                key, value = flt.split("=", 1)
            elif ":" in flt:
                key, value = flt.split(":", 1)
            else:
                raise ValueError("Filter must be in key=value or key:value format")

            key = key.strip()
            value = value.strip()

            # Apply to text analyzer
            if text_analyzer:
                try:
                    text_analyzer.filter_documents(key, value)
                except Exception as e:
                    click.echo(f"Could not apply text filter {key}={value}: {e}")

            # Apply to csv analyzer
            if csv_analyzer:
                try:
                    csv_analyzer.filter_rows_by_column_value(key, value)
                except Exception as e:
                    click.echo(f"Could not apply CSV filter {key}={value}: {e}")

        # Report results
        if text_analyzer:
            click.echo(
                f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
            )
        if csv_analyzer:
            click.echo(
                f"Applied filters {list(filters)}; remaining rows: {csv_analyzer.get_shape()[0]}"
            )

    except Exception as e:
        click.echo(f"Error applying filters: {e}")


def _apply_link_filters(corpus, text_analyzer, csv_analyzer, link_filters):
    """Apply embedding_links or temporal_links based filters."""
    if not csv_analyzer or csv_analyzer.df is None:
        click.echo("⚠️  Cannot apply link filters: no dataframe available")
        return

    linked_df_indices = set()

    for flt in link_filters:
        link_type = None
        if flt in ["=embedding", ":embedding"]:
            link_type = "embedding_links"
        elif flt in ["=temporal", ":temporal"]:
            link_type = "temporal_links"

        if not link_type:
            continue

        # Get documents (use filtered documents from text_analyzer if available)
        documents = (
            text_analyzer.corpus.documents if text_analyzer else corpus.documents
        )

        # Collect dataframe indices from linked documents
        for doc in documents:
            if link_type in doc.metadata:
                links = doc.metadata[link_type]
                if isinstance(links, list):
                    for link in links:
                        if isinstance(link, dict) and "df_index" in link:
                            linked_df_indices.add(link["df_index"])

        if linked_df_indices:
            click.echo(
                f"Found {len(linked_df_indices)} linked dataframe rows for {link_type}"
            )
        else:
            click.echo(f"⚠️  No {link_type} found in documents")

    # Filter dataframe to only include linked rows
    if linked_df_indices:
        df = csv_analyzer.df
        # Filter by index values that are in the linked set
        linked_indices = [idx for idx in df.index if idx in linked_df_indices]
        if linked_indices:
            csv_analyzer.df = df.loc[linked_indices]
            # Update corpus dataframe as well
            if text_analyzer:
                text_analyzer.corpus.df = csv_analyzer.df
            click.echo(
                f"Filtered dataframe to {len(linked_indices)} rows based on link filters"
            )
        else:
            click.echo("⚠️  No dataframe rows match the linked indices")


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
            click.echo(
                f"Probably no document metadata to filter, but let me check numeric metadata: {e}"
            )
    return text_analyzer


def get_csv_analyzer(
    corpus,
    comma_separated_unstructured_text_columns=None,
    comma_separated_ignore_columns=None,
    filters=None,
):
    if corpus and corpus.df is not None:
        click.echo("Loading CSV data from corpus.df")
        csv_analyzer = Csv(corpus=corpus)
        csv_analyzer.df = corpus.df
        text_columns, ignore_columns = _process_csv(
            csv_analyzer,
            comma_separated_unstructured_text_columns,
            comma_separated_ignore_columns,
            filters,
        )
        click.echo(f"Loaded CSV with shape: {csv_analyzer.get_shape()}")
        return csv_analyzer
    else:
        raise ValueError("Corpus or corpus.df is not set")


def _process_csv(
    csv_analyzer,
    comma_separated_unstructured_text_columns=None,
    comma_separated_ignore_columns=None,
    filters=None,
):
    text_columns = (
        comma_separated_unstructured_text_columns
        if comma_separated_unstructured_text_columns
        else ""
    )
    ignore_columns = (
        comma_separated_ignore_columns if comma_separated_ignore_columns else ""
    )
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
