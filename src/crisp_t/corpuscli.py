import logging
import warnings

import click

from .helpers.initializer import initialize_corpus
from .model.corpus import Corpus
from .model.document import Document
from .tdabm import Tdabm

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _parse_kv(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise click.ClickException(f"Invalid metadata '{value}'. Use key=value format.")
    key, val = value.split("=", 1)
    return key.strip(), val.strip()


def _parse_doc(value: str) -> tuple[str, str | None, str]:
    # id|name|text (name optional -> id||text)
    parts = value.split("|", 2)
    if len(parts) == 2:
        doc_id, text = parts
        name = None
    elif len(parts) == 3:
        doc_id, name, text = parts
    else:
        raise click.ClickException(
            "Invalid --doc value. Use 'id|name|text' or 'id|text'."
        )
    return doc_id.strip(), (name.strip() if name else None), text


def _parse_relationship(value: str) -> tuple[str, str, str]:
    # first|second|relation
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise click.ClickException("Invalid relationship. Use 'first|second|relation'.")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


@click.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress and debugging information.",
)
@click.option(
    "--id", help="Unique identifier for the corpus (required when creating new corpus)."
)
@click.option("--name", default=None, help="Descriptive name for the corpus.")
@click.option(
    "--description",
    default=None,
    help="Detailed description of the corpus content and purpose.",
)
@click.option(
    "--doc",
    "docs",
    multiple=True,
    help=(
        "Add a document to the corpus. Format: 'id|name|text' or 'id|text' (name optional). "
        "Can be used multiple times to add several documents."
    ),
)
@click.option(
    "--remove-doc",
    "remove_docs",
    multiple=True,
    help="Remove a document from the corpus by its ID. Can be used multiple times.",
)
@click.option(
    "--meta",
    "metas",
    multiple=True,
    help="Add or update corpus metadata. Format: key=value. Can be used multiple times.",
)
@click.option(
    "--add-rel",
    "relationships",
    multiple=True,
    help=(
        "Add a relationship between textual and numerical data. "
        "Format: 'first|second|relation' (e.g., text:term|num:column|correlates). "
        "Can be used multiple times."
    ),
)
@click.option(
    "--clear-rel",
    is_flag=True,
    help="Remove all relationships from the corpus metadata.",
)
@click.option(
    "--print",
    "print_corpus",
    is_flag=True,
    help="Display the corpus in a formatted view.",
)
@click.option(
    "--out", default=None, help="Save the corpus to a folder or file as corpus.json."
)
@click.option(
    "--inp",
    default=None,
    help="Load an existing corpus from a folder or file containing corpus.json.",
)
# New options for Corpus methods
@click.option(
    "--df-cols", is_flag=True, help="Display all column names in the DataFrame."
)
@click.option(
    "--df-row-count", is_flag=True, help="Display the number of rows in the DataFrame."
)
@click.option(
    "--df-row",
    default=None,
    type=int,
    help="Display a specific DataFrame row by its index.",
)
@click.option("--doc-ids", is_flag=True, help="Display all document IDs in the corpus.")
@click.option(
    "--doc-id", default=None, help="Display details of a specific document by its ID."
)
@click.option(
    "--relationships",
    "print_relationships",
    is_flag=True,
    help="Display all relationships defined in the corpus.",
)
@click.option(
    "--relationships-for-keyword",
    default=None,
    help="Display all relationships involving a specific keyword.",
)
@click.option(
    "--semantic",
    default=None,
    help="Perform semantic search using a query string. Returns documents similar to the query.",
)
@click.option(
    "--similar-docs",
    default=None,
    help="Find documents similar to a comma-separated list of document IDs. Useful for literature reviews. Use with --num and --rec.",
)
@click.option(
    "--num",
    default=5,
    type=int,
    help="Number of results to return for search operations (default: 5).",
)
@click.option(
    "--semantic-chunks",
    default=None,
    help="Perform semantic search on document chunks within a specific document. Use with --doc-id and --rec for threshold.",
)
@click.option(
    "--rec",
    default=0.4,
    type=float,
    help="Similarity threshold for semantic operations (0-1, default: 0.4). Higher values = more similar results only.",
)
@click.option(
    "--metadata-df",
    is_flag=True,
    help="Export semantic search collection metadata as a DataFrame. Requires semantic search initialization.",
)
@click.option(
    "--metadata-keys",
    default=None,
    help="Comma-separated list of metadata keys to include when exporting to DataFrame.",
)
@click.option(
    "--tdabm",
    default=None,
    help="Perform TDABM analysis. Format: 'y_variable:x_variables:radius' (e.g., 'satisfaction:age,income:0.3'). Radius defaults to 0.3 if omitted.",
)
@click.option(
    "--graph",
    is_flag=True,
    help="Generate a graph representation of the corpus. Requires documents to have keywords assigned (run text analysis first).",
)
@click.option(
    "--temporal-link",
    default=None,
    help="Link documents to dataframe rows by time. Format: 'method:column:param' where method is 'nearest', 'window', or 'sequence'. Examples: 'nearest:timestamp', 'window:timestamp:300' (±300 seconds), 'sequence:timestamp:W' (weekly).",
)
@click.option(
    "--temporal-filter",
    default=None,
    help="Filter corpus by time range. Format: 'start:end' where start and end are ISO 8601 timestamps. Either can be omitted. Examples: '2025-01-01:', ':2025-12-31', '2025-01-01:2025-06-30'.",
)
@click.option(
    "--temporal-summary",
    default=None,
    help="Generate temporal summary. Format: 'period' where period is 'D' (day), 'W' (week), 'M' (month), or 'Y' (year). Example: 'W' for weekly summary.",
)
@click.option(
    "--temporal-sentiment",
    default=None,
    help="Analyze sentiment trends over time. Format: 'period:aggregation' where period is 'D'/'W'/'M' and aggregation is 'mean'/'median'. Example: 'W:mean' for weekly average sentiment.",
)
@click.option(
    "--temporal-topics",
    default=None,
    help="Extract topics over time periods. Format: 'period:top_n' where period is 'D'/'W'/'M' and top_n is number of topics. Example: 'W:5' for top 5 topics per week.",
)
@click.option(
    "--temporal-subgraphs",
    default=None,
    help="Create time-sliced subgraphs. Format: 'period' where period is 'D'/'W'/'M'. Example: 'W' for weekly subgraphs.",
)
@click.option(
    "--embedding-link",
    default=None,
    help="Link documents to dataframe rows by embedding similarity. Format: 'metric:top_k:threshold' where metric is 'cosine' or 'euclidean', top_k is number of links per document, threshold is min similarity (0-1). Example: 'cosine:1:0.7'.",
)
@click.option(
    "--embedding-stats",
    is_flag=True,
    help="Display statistics about embedding-based links in the corpus.",
)
@click.option(
    "--embedding-viz",
    default=None,
    help="Visualize embedding space using dimensionality reduction. Format: 'method:output_path' where method is 'tsne', 'pca', or 'umap'. Example: 'tsne:embedding_viz.png'.",
)
def main(
    verbose: bool,
    id: str | None,
    name: str | None,
    description: str | None,
    docs: tuple[str, ...],
    remove_docs: tuple[str, ...],
    metas: tuple[str, ...],
    relationships: tuple[str, ...],
    clear_rel: bool,
    print_corpus: bool,
    out: str | None,
    inp: str | None,
    df_cols: bool,
    df_row_count: bool,
    df_row: int | None,
    doc_ids: bool,
    doc_id: str | None,
    print_relationships: bool,
    relationships_for_keyword: str | None,
    semantic: str | None,
    similar_docs: str | None,
    num: int,
    semantic_chunks: str | None,
    rec: float,
    metadata_df: bool,
    metadata_keys: str | None,
    tdabm: str | None,
    graph: bool,
    temporal_link: str | None,
    temporal_filter: str | None,
    temporal_summary: str | None,
    temporal_sentiment: str | None,
    temporal_topics: str | None,
    temporal_subgraphs: str | None,
    embedding_link: str | None,
    embedding_stats: bool,
    embedding_viz: str | None,
):
    """
    CRISP-T Corpus CLI: Create and manipulate corpus data from the command line.

    This tool allows you to create, modify, query, and manage corpus objects
    without writing any Python code. Perfect for quick data exploration and
    corpus management tasks.

    \b
    📚 COMMON TASKS:

    Create a new corpus:
       crispt --id my_corpus --name "My Research" --doc "1|Doc1|Hello world"

    Load and modify an existing corpus:
       crispt --inp crisp_input --doc "2|Doc2|More text" --out crisp_input

    Query corpus information:
       crispt --inp crisp_input --doc-ids --df-cols

    Perform semantic search:
       crispt --inp crisp_input --semantic "find this topic" --num 10

    \b
    💡 TIPS:
    • Use --print to view the full corpus structure
    • Combine multiple operations in a single command
    • Use --out to save changes after modifications
    • TDABM and graph features require prior analysis

    \b
    📖 For more examples, see: notes/DEMO.md
    """
    logging.basicConfig(level=(logging.DEBUG if verbose else logging.WARNING))
    logger = logging.getLogger(__name__)

    if verbose:
        click.echo(click.style("✓ Verbose mode enabled", fg="cyan"))

    # Display banner
    click.echo(click.style("\n" + "=" * 60, fg="blue", bold=True))
    click.echo(click.style("CRISP-T Corpus CLI", fg="green", bold=True))
    click.echo(click.style("=" * 60 + "\n", fg="blue", bold=True))

    # Load corpus from --inp if provided
    corpus = initialize_corpus(inp=inp)
    if not corpus:
        # Build initial corpus from CLI args
        if not id:
            raise click.ClickException(
                click.style("❌ Error: ", fg="red", bold=True)
                + "--id is required when creating a new corpus (not using --inp)."
            )
        corpus = Corpus(
            id=id,
            name=name,
            description=description,
            score=None,
            documents=[],
            df=None,
            visualization={},
            metadata={},
        )
        click.echo(click.style("✓ New corpus created", fg="green"))

    # Add documents
    for d in docs:
        doc_id, doc_name, doc_text = _parse_doc(d)
        document = Document(
            id=doc_id,
            name=doc_name,
            description=None,
            score=0.0,
            text=doc_text,
            metadata={},
        )
        corpus.add_document(document)
    if docs:
        click.echo(click.style(f"✓ Added {len(docs)} document(s)", fg="green"))

    # Remove documents
    for rid in remove_docs:
        corpus.remove_document_by_id(rid)
    if remove_docs:
        click.echo(click.style(f"✓ Removed {len(remove_docs)} document(s)", fg="green"))

    # Update metadata
    for m in metas:
        k, v = _parse_kv(m)
        corpus.update_metadata(k, v)
    if metas:
        count = len(metas)
        entry_word = "entry" if count == 1 else "entries"
        click.echo(click.style(f"✓ Updated {count} metadata {entry_word}", fg="green"))

    # Relationships
    for r in relationships:
        first, second, relation = _parse_relationship(r)
        corpus.add_relationship(first, second, relation)
    if relationships:
        click.echo(
            click.style(f"✓ Added {len(relationships)} relationship(s)", fg="green")
        )
    if clear_rel:
        corpus.clear_relationships()
        click.echo(click.style("✓ Cleared all relationships", fg="green"))

    # Print DataFrame column names
    if df_cols:
        cols = corpus.get_all_df_column_names()
        click.echo(
            click.style("📊 DataFrame columns: ", fg="cyan", bold=True) + str(cols)
        )

    # Print DataFrame row count
    if df_row_count:
        count = corpus.get_row_count()
        click.echo(
            click.style("📊 DataFrame row count: ", fg="cyan", bold=True) + str(count)
        )

    # Print DataFrame row by index
    if df_row is not None:
        row = corpus.get_row_by_index(df_row)
        if row is not None:
            click.echo(click.style(f"📊 DataFrame row {df_row}:", fg="cyan", bold=True))
            click.echo(row.to_dict())
        else:
            click.echo(click.style(f"⚠️  No row found at index {df_row}", fg="yellow"))

    # Print all document IDs
    if doc_ids:
        ids = corpus.get_all_document_ids()
        click.echo(click.style("📄 Document IDs: ", fg="cyan", bold=True) + str(ids))

    # Print document by ID
    if doc_id:
        doc = corpus.get_document_by_id(doc_id)
        if doc:
            click.echo(click.style(f"📄 Document {doc_id}:", fg="cyan", bold=True))
            click.echo(doc.model_dump())
        else:
            click.echo(
                click.style(f"⚠️  No document found with ID {doc_id}", fg="yellow")
            )
            exit(0)

    # Print relationships
    if print_relationships:
        rels = corpus.get_relationships()
        click.echo(click.style("🔗 Relationships: ", fg="cyan", bold=True) + str(rels))

    # Print relationships for keyword
    if relationships_for_keyword:
        rels = corpus.get_all_relationships_for_keyword(relationships_for_keyword)
        click.echo(
            click.style(
                f"🔗 Relationships for keyword '{relationships_for_keyword}': ",
                fg="cyan",
                bold=True,
            )
            + str(rels)
        )

    # Semantic search
    if semantic:
        try:
            from .semantic import Semantic

            click.echo(
                click.style("\n🔍 Performing semantic search for: ", fg="yellow")
                + click.style(f"'{semantic}'", fg="cyan", bold=True)
            )
            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    click.echo(
                        click.style(
                            "ℹ️  Note: Using simple embeddings (network unavailable)",
                            fg="blue",
                        )
                    )
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            corpus = semantic_analyzer.get_similar(semantic, n_results=num)
            click.echo(
                click.style(
                    f"✓ Found {len(corpus.documents)} similar document(s)", fg="green"
                )
            )
            click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the filtered corpus"
            )
            click.echo(f"   • Use {click.style('--print', fg='green')} to view results")
        except ImportError as e:
            click.echo(click.style(f"❌ Error: {e}", fg="red"))
            click.echo(
                click.style("   Install with: ", fg="white")
                + click.style("pip install chromadb", fg="cyan", bold=True)
            )
        except Exception as e:
            click.echo(
                click.style("❌ Error during semantic search: ", fg="red", bold=True)
                + str(e)
            )

    # Find similar documents
    if similar_docs:
        try:
            from .semantic import Semantic

            click.echo(f"\nFinding documents similar to: '{similar_docs}'")
            click.echo(f"Number of results: {num}")
            # Convert rec to 0-1 range if needed (for similar_docs, threshold is 0-1)
            threshold = rec / 10.0 if rec > 1.0 else rec
            click.echo(f"Similarity threshold: {threshold}")

            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    click.echo("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise

            # Get similar document IDs
            similar_doc_ids = semantic_analyzer.get_similar_documents(
                document_ids=similar_docs, n_results=num, threshold=threshold
            )

            click.echo(f"✓ Found {len(similar_doc_ids)} similar documents")
            if similar_doc_ids:
                click.echo("\nSimilar Document IDs:")
                for doc_id in similar_doc_ids:
                    doc = corpus.get_document_by_id(doc_id)
                    doc_name = f" ({doc.name})" if doc and doc.name else ""
                    click.echo(f"  - {doc_id}{doc_name}")
                click.echo("\nHint: Use --doc-id to view individual documents")
                click.echo(
                    "Hint: This feature is useful for literature reviews to find similar documents"
                )
            else:
                click.echo("No similar documents found above the threshold.")
                click.echo("Hint: Try lowering the threshold with --rec")

        except ImportError as e:
            click.echo(f"Error: {e}")
            click.echo("Install chromadb with: pip install chromadb")
        except Exception as e:
            click.echo(f"Error finding similar documents: {e}")

    # Semantic chunk search
    if semantic_chunks:
        if not doc_id:
            click.echo("Error: --doc-id is required when using --semantic-chunks")
        else:
            try:
                from .semantic import Semantic

                click.echo(
                    f"\nPerforming semantic chunk search for: '{semantic_chunks}'"
                )
                click.echo(f"Document ID: {doc_id}")
                click.echo(f"Threshold: {rec}")

                # Try with default embeddings first, fall back to simple embeddings
                try:
                    semantic_analyzer = Semantic(corpus)
                except Exception as network_error:
                    # If network error or download fails, try simple embeddings
                    if (
                        "address" in str(network_error).lower()
                        or "download" in str(network_error).lower()
                    ):
                        click.echo(
                            "Note: Using simple embeddings (network unavailable)"
                        )
                        semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                    else:
                        raise

                # Get similar chunks
                chunks = semantic_analyzer.get_similar_chunks(
                    query=semantic_chunks,
                    doc_id=doc_id,
                    threshold=rec,
                    n_results=20,  # Get more chunks to filter by threshold
                )

                click.echo(f"✓ Found {len(chunks)} matching chunks")
                click.echo("\nMatching chunks:")
                click.echo("=" * 60)
                for i, chunk in enumerate(chunks, 1):
                    click.echo(f"\nChunk {i}:")
                    click.echo(chunk)
                    click.echo("-" * 60)

                if len(chunks) == 0:
                    click.echo("No chunks matched the query above the threshold.")
                    click.echo(
                        "Hint: Try lowering the threshold with --rec or use a different query."
                    )
                else:
                    click.echo(
                        f"\nHint: These {len(chunks)} chunks can be used for coding/annotating the document."
                    )
                    click.echo(
                        "Hint: Adjust --rec threshold to get more or fewer results."
                    )

            except ImportError as e:
                click.echo(f"Error: {e}")
                click.echo("Install chromadb with: pip install chromadb")
            except Exception as e:
                click.echo(f"Error during semantic chunk search: {e}")

    # Export metadata as DataFrame
    if metadata_df:
        try:
            from .semantic import Semantic

            click.echo("\nExporting metadata as DataFrame...")
            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    click.echo("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            # Parse metadata_keys if provided
            keys_list = None
            if metadata_keys:
                keys_list = [k.strip() for k in metadata_keys.split(",")]
            corpus = semantic_analyzer.get_df(metadata_keys=keys_list)
            click.echo("✓ Metadata exported to DataFrame")
            if corpus.df is not None:
                click.echo(f"DataFrame shape: {corpus.df.shape}")
                click.echo(f"Columns: {list(corpus.df.columns)}")
            click.echo("Hint: Use --out to save the corpus with the updated DataFrame")
        except ImportError as e:
            click.echo(f"Error: {e}")
            click.echo("Install chromadb with: pip install chromadb")
        except Exception as e:
            click.echo(f"Error exporting metadata: {e}")

    # TDABM analysis
    if tdabm:
        try:
            # Parse tdabm parameter: y_variable:x_variables:radius
            parts = tdabm.split(":")
            if len(parts) < 2:
                raise click.ClickException(
                    click.style("❌ Invalid format. ", fg="red", bold=True)
                    + "Use 'y_variable:x_variables:radius' "
                    + "(e.g., 'satisfaction:age,income:0.3'). Radius defaults to 0.3 if omitted."
                )

            y_var = parts[0].strip()
            x_vars = parts[1].strip()
            radius = 0.3  # default

            if len(parts) >= 3:
                try:
                    radius = float(parts[2].strip())
                except ValueError:
                    raise click.ClickException(
                        click.style("❌ Invalid radius: ", fg="red", bold=True)
                        + f"'{parts[2]}'. Must be a number."
                    ) from None

            click.echo(click.style("\n📊 Performing TDABM analysis...", fg="yellow"))
            click.echo(f"   Y variable: {click.style(y_var, fg='cyan')}")
            click.echo(f"   X variables: {click.style(x_vars, fg='cyan')}")
            click.echo(f"   Radius: {click.style(str(radius), fg='cyan')}")

            tdabm_analyzer = Tdabm(corpus)
            result = tdabm_analyzer.generate_tdabm(
                y=y_var, x_variables=x_vars, radius=radius
            )

            click.echo("\n" + result)
            click.echo(click.style("\n✓ TDABM analysis complete", fg="green"))
            click.echo(click.style("\n💡 Next steps:", fg="cyan", bold=True))
            click.echo("   • Results stored in corpus metadata['tdabm']")
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the corpus"
            )
            click.echo(
                f"   • Use {click.style('crispviz --tdabm', fg='green')} to visualize results"
            )

        except ValueError as e:
            click.echo(click.style(f"\n❌ Error: {e}", fg="red", bold=True))
            click.echo(click.style("\n💡 Troubleshooting:", fg="cyan", bold=True))
            click.echo(
                "   • Ensure your corpus has a DataFrame with the specified variables"
            )
            click.echo("   • Y variable must be continuous (not binary)")
            click.echo("   • X variables must be numeric or ordinal")
        except Exception as e:
            click.echo(
                click.style("\n❌ Error during TDABM analysis: ", fg="red", bold=True)
                + str(e)
            )

    # Temporal analysis operations
    if temporal_link:
        try:
            from datetime import timedelta

            from .temporal import TemporalAnalyzer

            click.echo(click.style("\n⏰ Performing temporal linking...", fg="yellow"))

            # Parse temporal_link parameter
            parts = temporal_link.split(":")
            if len(parts) < 2:
                raise click.ClickException(
                    "Invalid format. Use 'method:column' or 'method:column:param' "
                    "(e.g., 'nearest:timestamp', 'window:timestamp:300', 'sequence:timestamp:W')"
                )

            method = parts[0].strip()
            time_column = parts[1].strip()
            analyzer = TemporalAnalyzer(corpus)

            if method == "nearest":
                max_gap = None
                if len(parts) >= 3:
                    max_gap = timedelta(seconds=float(parts[2]))
                corpus = analyzer.link_by_nearest_time(
                    time_column=time_column, max_gap=max_gap
                )
                click.echo(
                    click.style(
                        "✓ Documents linked to nearest dataframe rows", fg="green"
                    )
                )

            elif method == "window":
                window_seconds = 300  # Default ±5 minutes
                if len(parts) >= 3:
                    window_seconds = float(parts[2])
                window = timedelta(seconds=window_seconds)
                corpus = analyzer.link_by_time_window(
                    time_column=time_column, window_before=window, window_after=window
                )
                click.echo(
                    click.style(
                        f"✓ Documents linked within ±{window_seconds}s window",
                        fg="green",
                    )
                )

            elif method == "sequence":
                period = "W"  # Default to weekly
                if len(parts) >= 3:
                    period = parts[2].strip()
                corpus = analyzer.link_by_sequence(
                    time_column=time_column, period=period
                )
                click.echo(
                    click.style(f"✓ Documents linked by {period} sequences", fg="green")
                )

            else:
                raise click.ClickException(
                    f"Unknown method: {method}. Use 'nearest', 'window', or 'sequence'"
                )

            click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
            click.echo(
                "   • Temporal links stored in document metadata['temporal_links']"
            )
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the corpus"
            )

        except Exception as e:
            click.echo(
                click.style("\n❌ Error in temporal linking: ", fg="red", bold=True)
                + str(e)
            )

    if temporal_filter:
        try:
            from .temporal import TemporalAnalyzer

            click.echo(click.style("\n⏰ Filtering by time range...", fg="yellow"))

            # Parse temporal_filter parameter
            parts = temporal_filter.split(":")
            if len(parts) != 2:
                raise click.ClickException(
                    "Invalid format. Use 'start:end' (e.g., '2025-01-01:2025-12-31')"
                )

            start_time = parts[0].strip() if parts[0].strip() else None
            end_time = parts[1].strip() if parts[1].strip() else None

            analyzer = TemporalAnalyzer(corpus)
            corpus = analyzer.filter_by_time_range(
                start_time=start_time, end_time=end_time
            )

            click.echo(
                click.style(
                    f"✓ Corpus filtered to {len(corpus.documents)} documents",
                    fg="green",
                )
            )
            if corpus.df is not None:
                click.echo(f"   • DataFrame rows: {len(corpus.df)}")

        except Exception as e:
            click.echo(
                click.style("\n❌ Error in temporal filtering: ", fg="red", bold=True)
                + str(e)
            )

    if temporal_summary:
        try:
            from .temporal import TemporalAnalyzer

            click.echo(click.style("\n⏰ Generating temporal summary...", fg="yellow"))

            period = temporal_summary.strip()
            analyzer = TemporalAnalyzer(corpus)
            summary = analyzer.get_temporal_summary(period=period)

            if not summary.empty:
                click.echo(click.style("\n✓ Temporal summary:", fg="green"))
                click.echo(summary)
                click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
                click.echo("   • Summary stored in corpus metadata")
            else:
                click.echo(
                    click.style("⚠ No temporal data available for summary", fg="yellow")
                )

        except Exception as e:
            click.echo(
                click.style("\n❌ Error in temporal summary: ", fg="red", bold=True)
                + str(e)
            )

    if temporal_sentiment:
        try:
            from .temporal import TemporalAnalyzer

            click.echo(click.style("\n⏰ Analyzing sentiment trends...", fg="yellow"))

            # Parse temporal_sentiment parameter
            parts = temporal_sentiment.split(":")
            period = parts[0].strip() if len(parts) > 0 else "W"
            aggregation = parts[1].strip() if len(parts) > 1 else "mean"

            analyzer = TemporalAnalyzer(corpus)
            trend = analyzer.get_temporal_sentiment_trend(
                period=period, aggregation=aggregation
            )

            if not trend.empty:
                click.echo(click.style("\n✓ Sentiment trend:", fg="green"))
                click.echo(trend)
                click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
                click.echo("   • Sentiment trend stored in corpus metadata")
                click.echo(
                    f"   • Use {click.style('crispviz --temporal-sentiment', fg='green')} to visualize"
                )
            else:
                click.echo(
                    click.style(
                        "⚠ No sentiment data available for trend analysis", fg="yellow"
                    )
                )
                click.echo(
                    "   • Run sentiment analysis first with: crisp --inp <folder> --sentiment"
                )

        except Exception as e:
            click.echo(
                click.style("\n❌ Error in temporal sentiment: ", fg="red", bold=True)
                + str(e)
            )

    if temporal_topics:
        try:
            from .temporal import TemporalAnalyzer

            click.echo(click.style("\n⏰ Extracting temporal topics...", fg="yellow"))

            # Parse temporal_topics parameter
            parts = temporal_topics.split(":")
            period = parts[0].strip() if len(parts) > 0 else "W"
            top_n = int(parts[1].strip()) if len(parts) > 1 else 5

            analyzer = TemporalAnalyzer(corpus)
            topics = analyzer.get_temporal_topics(period=period, top_n=top_n)

            if topics:
                click.echo(
                    click.style(f"\n✓ Topics over time (top {top_n}):", fg="green")
                )
                for period_key, topic_list in topics.items():
                    click.echo(
                        f"\n{click.style(period_key, fg='cyan')}: {', '.join(topic_list)}"
                    )
                click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
                click.echo("   • Temporal topics stored in corpus metadata")
            else:
                click.echo(
                    click.style(
                        "⚠ No temporal data available for topic extraction", fg="yellow"
                    )
                )

        except Exception as e:
            click.echo(
                click.style("\n❌ Error in temporal topics: ", fg="red", bold=True)
                + str(e)
            )

    if temporal_subgraphs:
        try:
            from .graph import CrispGraph

            click.echo(click.style("\n⏰ Creating temporal subgraphs...", fg="yellow"))

            period = temporal_subgraphs.strip()

            # Ensure graph exists
            if "graph" not in corpus.metadata:
                click.echo(click.style("⚠ Creating base graph first...", fg="yellow"))
                graph_gen = CrispGraph(corpus)
                graph_gen.create_graph()

            graph_gen = CrispGraph(corpus)
            subgraphs = graph_gen.create_temporal_subgraphs(period=period)

            click.echo(
                click.style(
                    f"\n✓ Created {len(subgraphs)} temporal subgraphs", fg="green"
                )
            )
            for period_key, subgraph in subgraphs.items():
                click.echo(
                    f"   • {period_key}: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
                )

            click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
            click.echo("   • Temporal subgraphs stored in corpus metadata")
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the corpus"
            )

        except Exception as e:
            click.echo(
                click.style(
                    "\n❌ Error creating temporal subgraphs: ", fg="red", bold=True
                )
                + str(e)
            )

    # Embedding-based linking
    if embedding_link:
        try:
            from .embedding_linker import EmbeddingLinker

            click.echo(
                click.style(
                    "\n🔗 Performing embedding-based cross-modal linking...",
                    fg="yellow",
                )
            )

            # Parse embedding_link parameter
            parts = embedding_link.split(":")
            metric = parts[0].strip() if len(parts) > 0 else "cosine"
            top_k = int(parts[1].strip()) if len(parts) > 1 else 1
            threshold = float(parts[2].strip()) if len(parts) > 2 else None

            if metric not in ["cosine", "euclidean"]:
                raise click.ClickException(
                    f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'"
                )

            click.echo(f"   • Metric: {click.style(metric, fg='cyan')}")
            click.echo(f"   • Top-k: {click.style(str(top_k), fg='cyan')}")
            if threshold:
                click.echo(
                    f"   • Threshold: {click.style(f'{threshold:.2f}', fg='cyan')}"
                )

            linker = EmbeddingLinker(
                corpus,
                similarity_metric=metric,
                use_simple_embeddings=False,  # Use default embeddings
            )

            corpus = linker.link_by_embedding_similarity(
                threshold=threshold, top_k=top_k
            )

            stats = linker.get_link_statistics()
            click.echo(click.style("\n✓ Embedding-based linking complete", fg="green"))
            click.echo(
                f"   • Linked documents: {click.style(str(stats['linked_documents']), fg='cyan')}/{stats['total_documents']}"
            )
            click.echo(
                f"   • Total links: {click.style(str(stats['total_links']), fg='cyan')}"
            )
            click.echo(
                f"   • Avg similarity: {click.style('{:.3f}'.format(stats['avg_similarity']), fg='cyan')}"
            )

            click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
            click.echo(
                "   • Embedding links stored in document metadata['embedding_links']"
            )
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the corpus"
            )
            click.echo(
                f"   • Use {click.style('--embedding-stats', fg='green')} to view detailed statistics"
            )

        except ImportError as e:
            click.echo(click.style(f"\n❌ Error: {e}", fg="red", bold=True))
            click.echo(click.style("\n💡 Install ChromaDB:", fg="cyan", bold=True))
            click.echo(f"   {click.style('pip install chromadb', fg='green')}")
        except Exception as e:
            click.echo(
                click.style(
                    "\n❌ Error in embedding-based linking: ", fg="red", bold=True
                )
                + str(e)
            )

    if embedding_stats:
        try:
            from .embedding_linker import EmbeddingLinker

            click.echo(click.style("\n📊 Embedding Link Statistics", fg="yellow"))

            # Check if corpus has embedding links
            has_links = any(
                "embedding_links" in doc.metadata and doc.metadata["embedding_links"]
                for doc in corpus.documents
            )

            if not has_links:
                click.echo(
                    click.style("\n⚠ No embedding links found in corpus", fg="yellow")
                )
                click.echo(
                    f"   Run {click.style('--embedding-link', fg='green')} first to create links"
                )
                return

            # Create linker to get stats
            linker = EmbeddingLinker(corpus, use_simple_embeddings=True)
            stats = linker.get_link_statistics()

            click.echo(click.style("\n✓ Statistics:", fg="green"))
            click.echo(
                f"   • Total documents: {click.style(str(stats['total_documents']), fg='cyan')}"
            )
            click.echo(
                f"   • Linked documents: {click.style(str(stats['linked_documents']), fg='cyan')}"
            )
            click.echo(
                f"   • Total links: {click.style(str(stats['total_links']), fg='cyan')}"
            )
            click.echo(
                f"   • Average similarity: {click.style('{:.3f}'.format(stats['avg_similarity']), fg='cyan')}"
            )
            click.echo(
                f"   • Min similarity: {click.style('{:.3f}'.format(stats['min_similarity']), fg='cyan')}"
            )
            click.echo(
                f"   • Max similarity: {click.style('{:.3f}'.format(stats['max_similarity']), fg='cyan')}"
            )

        except Exception as e:
            click.echo(
                click.style(
                    "\n❌ Error getting embedding statistics: ", fg="red", bold=True
                )
                + str(e)
            )

    if embedding_viz:
        try:
            from .embedding_linker import EmbeddingLinker

            click.echo(click.style("\n📈 Visualizing embedding space...", fg="yellow"))

            # Parse embedding_viz parameter
            parts = embedding_viz.split(":")
            method = parts[0].strip() if len(parts) > 0 else "tsne"
            output_path = parts[1].strip() if len(parts) > 1 else "embedding_viz.png"

            if method not in ["tsne", "pca", "umap"]:
                raise click.ClickException(
                    f"Unknown method: {method}. Use 'tsne', 'pca', or 'umap'"
                )

            click.echo(f"   • Method: {click.style(method, fg='cyan')}")
            click.echo(f"   • Output: {click.style(output_path, fg='cyan')}")

            linker = EmbeddingLinker(corpus, use_simple_embeddings=True)

            # Generate embeddings first
            linker._get_text_embeddings()
            linker._get_numeric_embeddings()

            # Create visualization
            linker.visualize_embedding_space(output_path=output_path, method=method)

            click.echo(
                click.style(f"\n✓ Visualization saved to {output_path}", fg="green")
            )

        except ImportError as e:
            click.echo(
                click.style("\n❌ Error: Missing dependencies", fg="red", bold=True)
            )
            click.echo(f"   {e!s}")
            click.echo(
                click.style("\n💡 Install required packages:", fg="cyan", bold=True)
            )
            click.echo(
                f"   {click.style('pip install matplotlib scikit-learn', fg='green')}"
            )
        except Exception as e:
            click.echo(
                click.style("\n❌ Error creating visualization: ", fg="red", bold=True)
                + str(e)
            )

    # Graph generation
    if graph:
        try:
            from .graph import CrispGraph

            click.echo(
                click.style("\n🕸️  Generating graph representation...", fg="yellow")
            )
            graph_gen = CrispGraph(corpus)
            graph_data = graph_gen.create_graph()

            click.echo(click.style("\n✓ Graph created successfully", fg="green"))
            click.echo(
                f"   • Nodes: {click.style(str(graph_data['num_nodes']), fg='cyan')}"
            )
            click.echo(
                f"   • Edges: {click.style(str(graph_data['num_edges']), fg='cyan')}"
            )
            click.echo(
                f"   • Documents: {click.style(str(graph_data['num_documents']), fg='cyan')}"
            )
            click.echo(
                f"   • Has keywords: {click.style(str(graph_data['has_keywords']), fg='cyan')}"
            )
            click.echo(
                f"   • Has clusters: {click.style(str(graph_data['has_clusters']), fg='cyan')}"
            )
            click.echo(
                f"   • Has metadata: {click.style(str(graph_data['has_metadata']), fg='cyan')}"
            )

            click.echo(click.style("\n💡 Next steps:", fg="cyan", bold=True))
            click.echo("   • Graph data stored in corpus metadata['graph']")
            click.echo(
                f"   • Use {click.style('--out <folder>', fg='green')} to save the corpus"
            )
            click.echo(
                f"   • Use {click.style('crispviz --graph', fg='green')} to visualize the graph"
            )

        except ValueError as e:
            click.echo(click.style(f"\n❌ Error: {e}", fg="red", bold=True))
            click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
            click.echo("   Documents need keywords assigned first")
            click.echo(
                f"   Run text analysis with: {click.style('crisp --inp <folder> --assign', fg='green')}"
            )
        except Exception as e:
            click.echo(
                click.style("\n❌ Error generating graph: ", fg="red", bold=True)
                + str(e)
            )
            logger.error(f"Graph generation error: {e}", exc_info=True)

    # Save corpus to --out if provided
    if out:
        from .read_data import ReadData

        rd = ReadData(corpus=corpus)
        rd.write_corpus_to_json(out, corpus=corpus)
        click.echo(
            click.style("\n✓ Corpus saved to: ", fg="green", bold=True)
            + click.style(str(out), fg="cyan")
        )

    if print_corpus:
        click.echo(
            click.style(
                "\n╔═══════════════════════════════════════════════╗",
                fg="blue",
                bold=True,
            )
        )
        click.echo(
            click.style(
                "║  📊 CORPUS DETAILS                           ║", fg="blue", bold=True
            )
        )
        click.echo(
            click.style(
                "╚═══════════════════════════════════════════════╝\n",
                fg="blue",
                bold=True,
            )
        )
        corpus.pretty_print()

    logger.info("Corpus CLI finished")
