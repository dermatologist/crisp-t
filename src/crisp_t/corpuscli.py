import json
import logging
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

from . import __version__
from .helpers.initializer import initialize_corpus
from .model.corpus import Corpus
from .model.document import Document
from .tdabm import Tdabm


def _parse_kv(value: str) -> tuple[str, str]:
    if "=" not in value:
        console.print(
            f"[red]Error:[/red] Invalid metadata '{value}'. Use key=value format."
        )
        raise typer.Exit(code=1)
    key, val = value.split("=", 1)
    return key.strip(), val.strip()


def _parse_doc(value: str) -> tuple[str, Optional[str], str]:
    # id|name|text (name optional -> id||text)
    parts = value.split("|", 2)
    if len(parts) == 2:
        doc_id, text = parts
        name = None
    elif len(parts) == 3:
        doc_id, name, text = parts
    else:
        console.print(
            "[red]Error:[/red] Invalid --doc value. Use 'id|name|text' or 'id|text'."
        )
        raise typer.Exit(code=1)
    return doc_id.strip(), (name.strip() if name else None), text


def _parse_relationship(value: str) -> tuple[str, str, str]:
    # first|second|relation
    parts = value.split("|", 2)
    if len(parts) != 3:
        console.print(
            "[red]Error:[/red] Invalid relationship. Use 'first|second|relation'."
        )
        raise typer.Exit(code=1)
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable detailed logging for debugging"
    ),
    id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Unique identifier for the corpus (required when creating new corpus)",
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Human-readable name for the corpus"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Brief description of the corpus and its contents"
    ),
    docs: List[str] = typer.Option(
        [],
        "--doc",
        help="Add document as 'id|name|text' or 'id|text'. Use multiple times for multiple documents",
    ),
    remove_docs: List[str] = typer.Option(
        [],
        "--remove-doc",
        help="Remove document by ID. Use multiple times to remove multiple documents",
    ),
    metas: List[str] = typer.Option(
        [],
        "--meta",
        help="Add/update metadata as key=value pairs. Use multiple times for multiple entries",
    ),
    relationships: List[str] = typer.Option(
        [],
        "--add-rel",
        help="Add relationship as 'first|second|relation' (e.g., text:term|numb:col|correlates)",
    ),
    clear_rel: bool = typer.Option(
        False, "--clear-rel", help="Clear all relationships from corpus metadata"
    ),
    print_corpus: bool = typer.Option(
        False, "--print", help="Display the complete corpus in a formatted view"
    ),
    out: Optional[str] = typer.Option(
        None, "--out", help="Save corpus to folder or file (corpus.json)"
    ),
    inp: Optional[str] = typer.Option(
        None, "--inp", help="Load existing corpus from folder or file (corpus.json)"
    ),
    df_cols: bool = typer.Option(
        False, "--df-cols", help="Display all DataFrame column names"
    ),
    df_row_count: bool = typer.Option(
        False, "--df-row-count", help="Display total number of rows in DataFrame"
    ),
    df_row: Optional[int] = typer.Option(
        None, "--df-row", help="Display specific DataFrame row by index number"
    ),
    doc_ids: bool = typer.Option(
        False, "--doc-ids", help="List all document IDs in the corpus"
    ),
    doc_id: Optional[str] = typer.Option(
        None, "--doc-id", help="Display specific document by its ID"
    ),
    print_relationships: bool = typer.Option(
        False, "--relationships", help="Display all relationships in the corpus"
    ),
    relationships_for_keyword: Optional[str] = typer.Option(
        None,
        "--relationships-for-keyword",
        help="Find all relationships involving a specific keyword",
    ),
    semantic: Optional[str] = typer.Option(
        None,
        "--semantic",
        help="Perform semantic search with query string to find similar documents",
    ),
    similar_docs: Optional[str] = typer.Option(
        None,
        "--similar-docs",
        help="Find documents similar to comma-separated document IDs (useful for literature reviews)",
    ),
    num: int = typer.Option(
        5, "--num", help="Number of results to return for searches (default: 5)"
    ),
    semantic_chunks: Optional[str] = typer.Option(
        None,
        "--semantic-chunks",
        help="Search document chunks semantically. Requires --doc-id and --rec (threshold)",
    ),
    rec: float = typer.Option(
        0.4,
        "--rec",
        help="Similarity threshold for semantic search (0-1 range, default: 0.4)",
    ),
    metadata_df: bool = typer.Option(
        False, "--metadata-df", help="Export collection metadata as DataFrame"
    ),
    metadata_keys: Optional[str] = typer.Option(
        None,
        "--metadata-keys",
        help="Comma-separated metadata keys to include in DataFrame export",
    ),
    tdabm: Optional[str] = typer.Option(
        None,
        "--tdabm",
        help="Perform TDABM analysis: 'y_variable:x_variables:radius' (e.g., 'satisfaction:age,income:0.3')",
    ),
    graph: bool = typer.Option(
        False,
        "--graph",
        help="Generate graph representation (documents must have keywords assigned first)",
    ),
):
    """
    [bold cyan]CRISP-T Corpus CLI[/bold cyan]

    Create, manipulate, and query corpus data quickly from the command line.
    Manage documents, metadata, relationships, and perform semantic searches.

    [bold]Quick Start Examples:[/bold]
      [dim]# Create new corpus[/dim]
      crispt --id my_corpus --name "My Research" --doc "1|First doc|Text here"

      [dim]# Load and inspect corpus[/dim]
      crispt --inp data/ --print

      [dim]# Semantic search[/dim]
      crispt --inp data/ --semantic "healthcare policy" --num 10

      [dim]# Add relationships[/dim]
      crispt --inp data/ --add-rel "text:theme|num:score|correlates" --out data/
    """
    logging.basicConfig(level=(logging.DEBUG if verbose else logging.WARNING))
    logger = logging.getLogger(__name__)

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]CRISP-T[/bold cyan]: Corpus CLI\n"
            f"Version: [yellow]{__version__}[/yellow]",
            border_style="cyan",
        )
    )
    console.print()

    # Load corpus from --inp if provided
    corpus = initialize_corpus(inp=inp)
    if not corpus:
        # Build initial corpus from CLI args
        if not id:
            console.print(
                "[red]Error:[/red] Corpus ID is required when creating a new corpus",
                style="bold red",
            )
            console.print(
                "[yellow]Hint:[/yellow] Use --id <corpus_id> to specify an ID, or --inp <directory> to load an existing corpus"
            )
            raise typer.Exit(code=1)
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
        console.print(f"[green]✓[/green] Added {len(docs)} document(s)")

    # Remove documents
    for rid in remove_docs:
        corpus.remove_document_by_id(rid)
    if remove_docs:
        console.print(f"[green]✓[/green] Removed {len(remove_docs)} document(s)")

    # Update metadata
    for m in metas:
        k, v = _parse_kv(m)
        corpus.update_metadata(k, v)
    if metas:
        console.print(f"[green]✓[/green] Updated metadata entries: {len(metas)}")

    # Relationships
    for r in relationships:
        first, second, relation = _parse_relationship(r)
        corpus.add_relationship(first, second, relation)
    if relationships:
        console.print(f"[green]✓[/green] Added {len(relationships)} relationship(s)")
    if clear_rel:
        corpus.clear_relationships()
        console.print("[green]✓[/green] Cleared relationships")

    # Print DataFrame column names
    if df_cols:
        cols = corpus.get_all_df_column_names()
        console.print(f"DataFrame columns: {cols}")

    # Print DataFrame row count
    if df_row_count:
        count = corpus.get_row_count()
        console.print(f"DataFrame row count: {count}")

    # Print DataFrame row by index
    if df_row is not None:
        row = corpus.get_row_by_index(df_row)
        if row is not None:
            console.print(f"DataFrame row {df_row}: {row.to_dict()}")
        else:
            console.print(f"No row at index {df_row}")

    # Print all document IDs
    if doc_ids:
        ids = corpus.get_all_document_ids()
        console.print(f"Document IDs: {ids}")

    # Print document by ID
    if doc_id:
        doc = corpus.get_document_by_id(doc_id)
        if doc:
            console.print(f"Document {doc_id}: {doc.model_dump()}")
        else:
            console.print(f"No document found with ID {doc_id}")
            exit(0)

    # Print relationships
    if print_relationships:
        rels = corpus.get_relationships()
        print(f"Relationships: {rels}")

    # Print relationships for keyword
    if relationships_for_keyword:
        rels = corpus.get_all_relationships_for_keyword(relationships_for_keyword)
        console.print(
            f"Relationships for keyword '{relationships_for_keyword}': {rels}"
        )

    # Semantic search
    if semantic:
        try:
            from .semantic import Semantic

            console.print(f"\nPerforming semantic search for: '{semantic}'")
            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            corpus = semantic_analyzer.get_similar(semantic, n_results=num)
            console.print(
                f"[green]✓[/green] Found {len(corpus.documents)} similar documents"
            )
            console.print(
                f"Hint: Use --out to save the filtered corpus, or --print to view results"
            )
        except ImportError as e:
            console.print(f"Error: {e}")
            console.print("Install chromadb with: pip install chromadb")
        except Exception as e:
            console.print(f"Error during semantic search: {e}")

    # Find similar documents
    if similar_docs:
        try:
            from .semantic import Semantic

            console.print(f"\nFinding documents similar to: '{similar_docs}'")
            console.print(f"Number of results: {num}")
            # Convert rec to 0-1 range if needed (for similar_docs, threshold is 0-1)
            threshold = rec / 10.0 if rec > 1.0 else rec
            console.print(f"Similarity threshold: {threshold}")

            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise

            # Get similar document IDs
            similar_doc_ids = semantic_analyzer.get_similar_documents(
                document_ids=similar_docs, n_results=num, threshold=threshold
            )

            console.print(
                f"[green]✓[/green] Found {len(similar_doc_ids)} similar documents"
            )
            if similar_doc_ids:
                console.print("\nSimilar Document IDs:")
                for doc_id in similar_doc_ids:
                    doc = corpus.get_document_by_id(doc_id)
                    doc_name = f" ({doc.name})" if doc and doc.name else ""
                    console.print(f"  - {doc_id}{doc_name}")
                console.print("\nHint: Use --doc-id to view individual documents")
                console.print(
                    "Hint: This feature is useful for literature reviews to find similar documents"
                )
            else:
                console.print("No similar documents found above the threshold.")
                console.print("Hint: Try lowering the threshold with --rec")

        except ImportError as e:
            console.print(f"Error: {e}")
            console.print("Install chromadb with: pip install chromadb")
        except Exception as e:
            console.print(f"Error finding similar documents: {e}")

    # Semantic chunk search
    if semantic_chunks:
        if not doc_id:
            console.print("Error: --doc-id is required when using --semantic-chunks")
        else:
            try:
                from .semantic import Semantic

                console.print(
                    f"\nPerforming semantic chunk search for: '{semantic_chunks}'"
                )
                console.print(f"Document ID: {doc_id}")
                console.print(f"Threshold: {rec}")

                # Try with default embeddings first, fall back to simple embeddings
                try:
                    semantic_analyzer = Semantic(corpus)
                except Exception as network_error:
                    # If network error or download fails, try simple embeddings
                    if (
                        "address" in str(network_error).lower()
                        or "download" in str(network_error).lower()
                    ):
                        console.print(
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

                console.print(f"[green]✓[/green] Found {len(chunks)} matching chunks")
                console.print("\nMatching chunks:")
                console.print("=" * 60)
                for i, chunk in enumerate(chunks, 1):
                    console.print(f"\nChunk {i}:")
                    console.print(chunk)
                    console.print("-" * 60)

                if len(chunks) == 0:
                    console.print("No chunks matched the query above the threshold.")
                    console.print(
                        "Hint: Try lowering the threshold with --rec or use a different query."
                    )
                else:
                    console.print(
                        f"\nHint: These {len(chunks)} chunks can be used for coding/annotating the document."
                    )
                    console.print(
                        "Hint: Adjust --rec threshold to get more or fewer results."
                    )

            except ImportError as e:
                console.print(f"Error: {e}")
                console.print("Install chromadb with: pip install chromadb")
            except Exception as e:
                console.print(f"Error during semantic chunk search: {e}")

    # Export metadata as DataFrame
    if metadata_df:
        try:
            from .semantic import Semantic

            console.print("\nExporting metadata as DataFrame...")
            # Try with default embeddings first, fall back to simple embeddings
            try:
                semantic_analyzer = Semantic(corpus)
            except Exception as network_error:
                # If network error or download fails, try simple embeddings
                if (
                    "address" in str(network_error).lower()
                    or "download" in str(network_error).lower()
                ):
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            # Parse metadata_keys if provided
            keys_list = None
            if metadata_keys:
                keys_list = [k.strip() for k in metadata_keys.split(",")]
            corpus = semantic_analyzer.get_df(metadata_keys=keys_list)
            console.print("[green]✓[/green] Metadata exported to DataFrame")
            if corpus.df is not None:
                console.print(f"DataFrame shape: {corpus.df.shape}")
                console.print(f"Columns: {list(corpus.df.columns)}")
            console.print(
                "Hint: Use --out to save the corpus with the updated DataFrame"
            )
        except ImportError as e:
            console.print(f"Error: {e}")
            console.print("Install chromadb with: pip install chromadb")
        except Exception as e:
            console.print(f"Error exporting metadata: {e}")

    # TDABM analysis
    if tdabm:
        try:
            # Parse tdabm parameter: y_variable:x_variables:radius
            parts = tdabm.split(":")
            if len(parts) < 2:
                raise click.ClickException(
                    "Invalid --tdabm format. Use 'y_variable:x_variables:radius' "
                    "(e.g., 'satisfaction:age,income:0.3'). Radius defaults to 0.3 if omitted."
                )

            y_var = parts[0].strip()
            x_vars = parts[1].strip()
            radius = 0.3  # default

            if len(parts) >= 3:
                try:
                    radius = float(parts[2].strip())
                except ValueError:
                    console.print(
                        f"[red]Error:[/red] Invalid radius value: '{parts[2]}'. Must be a number."
                    )
                    raise typer.Exit(code=1)

            console.print(f"\nPerforming TDABM analysis...")
            console.print(f"  Y variable: {y_var}")
            console.print(f"  X variables: {x_vars}")
            console.print(f"  Radius: {radius}")

            tdabm_analyzer = Tdabm(corpus)
            result = tdabm_analyzer.generate_tdabm(
                y=y_var, x_variables=x_vars, radius=radius
            )

            console.print("\n" + result)
            console.print("\nHint: TDABM results stored in corpus metadata['tdabm']")
            console.print("Hint: Use --out to save the corpus with TDABM metadata")
            console.print("Hint: Use 'crispviz --tdabm' to visualize the results")

        except ValueError as e:
            console.print(f"Error: {e}")
            console.print(
                "Hint: Ensure your corpus has a DataFrame with the specified variables"
            )
            console.print("Hint: Y variable must be continuous (not binary)")
            console.print("Hint: X variables must be numeric/ordinal")
        except Exception as e:
            console.print(f"Error during TDABM analysis: {e}")

    # Graph generation
    if graph:
        try:
            from .graph import CrispGraph

            console.print("\nGenerating graph representation...")
            graph_gen = CrispGraph(corpus)
            graph_data = graph_gen.create_graph()

            console.print(f"[green]✓[/green] Graph created successfully")
            console.print(f"  Nodes: {graph_data['num_nodes']}")
            console.print(f"  Edges: {graph_data['num_edges']}")
            console.print(f"  Documents: {graph_data['num_documents']}")
            console.print(f"  Has keywords: {graph_data['has_keywords']}")
            console.print(f"  Has clusters: {graph_data['has_clusters']}")
            console.print(f"  Has metadata: {graph_data['has_metadata']}")

            console.print("\nHint: Graph data stored in corpus metadata['graph']")
            console.print("Hint: Use --out to save the corpus with graph metadata")
            console.print("Hint: Use 'crispviz --graph' to visualize the graph")

        except ValueError as e:
            console.print(f"Error: {e}")
            console.print("Hint: Make sure documents have keywords assigned first")
            console.print("Hint: You can assign keywords using text analysis features")
        except Exception as e:
            console.print(f"Error generating graph: {e}")
            logger.error(f"Graph generation error: {e}", exc_info=True)

    # Save corpus to --out if provided
    if out:
        from .read_data import ReadData

        rd = ReadData(corpus=corpus)
        rd.write_corpus_to_json(out, corpus=corpus)
        console.print(f"[green]✓[/green] Corpus saved to {out}")

    if print_corpus:
        console.print("\n=== Corpus Details ===")
        corpus.pretty_print()

    logger.info("Corpus CLI finished")


if __name__ == "__main__":
    app()
