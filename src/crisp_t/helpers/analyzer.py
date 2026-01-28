import click

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
    - embedding:text or embedding=text - Filter dataframe rows linked via embedding_links (text→df)
    - embedding:df or embedding=df - Filter documents linked via dataframe reverse links (df→text)
    - temporal:text or temporal=text - Filter dataframe rows linked via temporal_links (text→df)
    - temporal:df or temporal=df - Filter documents linked via dataframe reverse links (df→text)
    - id or id: or id= - Synchronize filtering between documents and DataFrame by ID:
        - id=<value> - Filter to specific ID
        - id: or id= (no value) - Sync all remaining documents to matching DataFrame rows by ID

    Args:
        corpus (Corpus): The corpus to analyze.
        comma_separated_unstructured_text_columns (str, optional): CSV columns with free-text.
        comma_separated_ignore_columns (str, optional): Columns to ignore.
        filters (list, optional): List of filters in key=value or key:value format.
            Special filters: embedding:text, embedding=text, embedding:df, embedding=df,
                           temporal:text, temporal=text, temporal:df, temporal=df,
                           id, id:, id=, id=<value>, id:<value>
            Legacy filters: =embedding, :embedding, =temporal, :temporal (mapped to :text variants)

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
            # Check for special linking filters (new explicit format or legacy format)
            if flt in [
                "embedding:text",
                "embedding=text",
                "embedding:df",
                "embedding=df",
                "temporal:text",
                "temporal=text",
                "temporal:df",
                "temporal=df",
                "=embedding",
                ":embedding",
                "=temporal",
                ":temporal",
            ]:
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
    """Apply regular key=value or key:value filters to analyzers.

    Special handling for 'id' filter: when filtering on id, both documents
    and DataFrame rows are synchronized by ID for implicit ID linkage.
    If id filter has no value (id: or id=), syncs after other filters are applied.
    """
    try:
        id_filter_value = None
        has_id_filter = False

        # First pass: apply all non-id filters and detect id filter
        for flt in filters:
            if "=" in flt:
                key, value = flt.split("=", 1)
            elif ":" in flt:
                key, value = flt.split(":", 1)
            else:
                raise ValueError("Filter must be in key=value or key:value format")

            key = key.strip()
            value = value.strip()

            # Track id filter for later processing
            if key.lower() == "id":
                has_id_filter = True
                id_filter_value = value
                # Only apply specific id filtering if value is not empty
                if value:
                    # Filter documents by ID
                    if text_analyzer:
                        try:
                            text_analyzer.filter_documents(key, value)
                        except Exception as e:
                            click.echo(
                                f"Could not apply text filter {key}={value}: {e}"
                            )

                    # Filter DataFrame by ID column if it exists (implicit ID linkage)
                    if (
                        csv_analyzer
                        and csv_analyzer.df is not None
                        and "id" in csv_analyzer.df.columns
                    ):
                        try:
                            csv_analyzer.df = csv_analyzer.df[
                                csv_analyzer.df["id"].astype(str) == str(value)
                            ]
                            click.echo(
                                f"Applied ID linkage: synced DataFrame to {len(csv_analyzer.df)} rows matching id={value}"
                            )
                        except Exception as e:
                            click.echo(f"Could not apply ID filter to DataFrame: {e}")
            else:
                # Regular filter: apply independently to both analyzers
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

        # Second pass: if id filter exists with no value, sync after other filters applied
        if (
            has_id_filter
            and not id_filter_value
            and text_analyzer
            and csv_analyzer
            and csv_analyzer.df is not None
            and "id" in csv_analyzer.df.columns
        ):
            try:
                # Get all current document IDs
                doc_ids = {doc.id for doc in text_analyzer.corpus.documents}
                # Filter DataFrame to rows where id matches a document ID
                csv_analyzer.df = csv_analyzer.df[
                    csv_analyzer.df["id"]
                    .astype(str)
                    .isin([str(did) for did in doc_ids])
                ]
                click.echo(
                    f"Applied ID linkage: synced DataFrame to {len(csv_analyzer.df)} rows matching remaining document IDs"
                )
            except Exception as e:
                click.echo(f"Could not apply ID sync to DataFrame: {e}")

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
    """Apply bidirectional link filters.

    Supports two directions:
    - text→df: Filter dataframe rows based on documents' embedding/temporal links
    - df→text: Filter documents based on dataframe rows that have reverse links
    """
    if not csv_analyzer or csv_analyzer.df is None:
        click.echo("⚠️  Cannot apply link filters: no dataframe available")
        return

    linked_df_indices = set()
    linked_doc_ids = set()

    for flt in link_filters:
        # Map legacy filters to new explicit format
        if flt == "=embedding" or flt == ":embedding":
            flt = "embedding:text"
        elif flt == "=temporal" or flt == ":temporal":
            flt = "temporal:text"

        # Parse filter to determine direction and link type
        direction = None  # "text" or "df"
        link_type = None  # "embedding_links" or "temporal_links"

        if flt in ["embedding:text", "embedding=text"]:
            direction = "text"
            link_type = "embedding_links"
        elif flt in ["embedding:df", "embedding=df"]:
            direction = "df"
            link_type = "embedding_links"
        elif flt in ["temporal:text", "temporal=text"]:
            direction = "text"
            link_type = "temporal_links"
        elif flt in ["temporal:df", "temporal=df"]:
            direction = "df"
            link_type = "temporal_links"

        if not direction or not link_type:
            continue

        if direction == "text":
            # text→df: Filter dataframe rows based on document links
            _apply_text_to_df_filter(
                corpus, text_analyzer, csv_analyzer, link_type, linked_df_indices
            )
        else:  # direction == "df"
            # df→text: Filter documents based on dataframe row reverse links
            _apply_df_to_text_filter(
                corpus, text_analyzer, csv_analyzer, link_type, linked_doc_ids
            )

    # Apply dataframe filtering
    if linked_df_indices:
        df = csv_analyzer.df
        linked_indices = [idx for idx in df.index if idx in linked_df_indices]
        if linked_indices:
            csv_analyzer.df = df.loc[linked_indices]
            if text_analyzer:
                text_analyzer.corpus.df = csv_analyzer.df
            click.echo(
                f"Filtered dataframe to {len(linked_indices)} rows based on link filters"
            )
        else:
            click.echo("⚠️  No dataframe rows match the linked indices")

    # Apply document filtering
    if linked_doc_ids and text_analyzer:
        documents = text_analyzer.corpus.documents
        filtered_documents = [doc for doc in documents if doc.id in linked_doc_ids]
        text_analyzer.corpus.documents = filtered_documents
        corpus.documents = filtered_documents
        click.echo(
            f"Filtered documents to {len(filtered_documents)} based on link filters"
        )


def _apply_text_to_df_filter(
    corpus, text_analyzer, csv_analyzer, link_type, linked_df_indices
):
    """Filter dataframe rows based on documents' embedding/temporal links.

    Collects all df_indices from document metadata and uses them to filter dataframe.
    """
    # Get documents (use filtered documents from text_analyzer if available)
    documents = text_analyzer.corpus.documents if text_analyzer else corpus.documents

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


def _apply_df_to_text_filter(
    corpus, text_analyzer, csv_analyzer, link_type, linked_doc_ids
):
    """Filter documents based on reverse links from dataframe rows.

    Creates reverse mapping: if document D links to row R via link_type,
    then row R can select document D.
    """
    # Get documents
    documents = text_analyzer.corpus.documents if text_analyzer else corpus.documents

    # Build a map of df_index → doc_ids for reverse lookup
    df_index_to_docs = {}
    for doc in documents:
        if link_type in doc.metadata:
            links = doc.metadata[link_type]
            if isinstance(links, list):
                for link in links:
                    if isinstance(link, dict) and "df_index" in link:
                        df_idx = link["df_index"]
                        if df_idx not in df_index_to_docs:
                            df_index_to_docs[df_idx] = set()
                        df_index_to_docs[df_idx].add(doc.id)

    # Collect document IDs that are linked from dataframe rows
    df = csv_analyzer.df
    for df_idx in df.index:
        if df_idx in df_index_to_docs:
            linked_doc_ids.update(df_index_to_docs[df_idx])

    if linked_doc_ids:
        click.echo(
            f"Found {len(linked_doc_ids)} documents linked from dataframe rows via {link_type}"
        )
    else:
        click.echo(f"⚠️  No documents found linked from dataframe rows via {link_type}")


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
