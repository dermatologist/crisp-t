import json
import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import typer
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel

console = Console()

from .model.corpus import Corpus
from .model.document import Document
from .helpers.initializer import initialize_corpus
from .tdabm import Tdabm


def _parse_kv(value: str) -> tuple[str, str]:
    if "=" not in value:
        console.print(f"[red]Error:[/red] {Invalid metadata '{value}'. Use key=value format.}")
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
        raise click.ClickException(
            "Invalid --doc value. Use 'id|name|text' or 'id|text'."
        )
    return doc_id.strip(), (name.strip() if name else None), text


def _parse_relationship(value: str) -> tuple[str, str, str]:
    # first|second|relation
    parts = value.split("|", 2)
    if len(parts) != 3:
        console.print(f"[red]Error:[/red] "Invalid relationship. Use 'first|second|relation'."")
        raise typer.Exit(code=1)
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


app = typer.Typer(add_completion=False, rich_markup_mode="rich")

@app.command()
@click.option("--verbose", "-v", is_flag=True, help="Print verbose messages.")
@click.option("--id", help="Unique identifier for the corpus.")
@click.option("--name", default=None, help="Name of the corpus.")
@click.option("--description", default=None, help="Description of the corpus.")
@click.option(
    "--doc",
    "docs",
    multiple=True,
    help=(
        "Add a document as 'id|name|text' (or 'id|text' if name omitted). "
        "Can be used multiple times."
    ),
)
@click.option(
    "--remove-doc",
    "remove_docs",
    multiple=True,
    help="Remove a document by its ID (can be used multiple times).",
)
@click.option(
    "--meta",
    "metas",
    multiple=True,
    help="Add or update corpus metadata as key=value (can be used multiple times).",
)
@click.option(
    "--add-rel",
    "relationships",
    multiple=True,
    help=(
        "Add a relationship as 'first|second|relation' (e.g., text:term|numb:col|correlates)."
    ),
)
@click.option(
    "--clear-rel",
    is_flag=True,
    help="Clear all relationships in the corpus metadata.",
)
@click.option("--print", "print_corpus", is_flag=True, help="Pretty print the corpus")
@click.option(
    "--out", default=None, help="Write corpus to a folder or file as corpus.json (save)"
)
@click.option(
    "--inp",
    default=None,
    help="Load corpus from a folder or file containing corpus.json (load)",
)
# New options for Corpus methods
@click.option("--df-cols", is_flag=True, help="Print all DataFrame column names.")
@click.option("--df-row-count", is_flag=True, help="Print number of rows in DataFrame.")
@click.option("--df-row", default=None, type=int, help="Print DataFrame row by index.")
@click.option("--doc-ids", is_flag=True, help="Print all document IDs in the corpus.")
@click.option("--doc-id", default=None, help="Print document by ID.")
@click.option(
    "--relationships",
    "print_relationships",
    is_flag=True,
    help="Print all relationships in the corpus.",
)
@click.option(
    "--relationships-for-keyword",
    default=None,
    help="Print all relationships involving a specific keyword.",
)
@click.option(
    "--semantic",
    default=None,
    help="Perform semantic search with the given query string. Returns similar documents.",
)
@click.option(
    "--similar-docs",
    default=None,
    help="Find documents similar to a comma-separated list of document IDs. Use with --num and --rec. Useful for literature reviews.",
)
@click.option(
    "--num",
    default=5,
    type=int,
    help="Number of results to return (default: 5). Used for semantic search and similar documents search.",
)
@click.option(
    "--semantic-chunks",
    default=None,
    help="Perform semantic search on document chunks. Returns matching chunks for a specific document. Use with --doc-id and --rec (threshold).",
)
@click.option(
    "--rec",
    default=0.4,
    type=float,
    help="Threshold for semantic search (0-1, default: 0.4). Only chunks with similarity above this value are returned.",
)
@click.option(
    "--metadata-df",
    is_flag=True,
    help="Export collection metadata as DataFrame. Requires semantic search to be initialized first.",
)
@click.option(
    "--metadata-keys",
    default=None,
    help="Comma-separated list of metadata keys to include in DataFrame export.",
)
@click.option(
    "--tdabm",
    default=None,
    help="Perform TDABM analysis. Format: 'y_variable:x_variables:radius' (e.g., 'satisfaction:age,income:0.3'). Radius defaults to 0.3 if omitted.",
)
@click.option(
    "--graph",
    is_flag=True,
    help="Generate graph representation of the corpus. Requires documents to have keywords assigned (run with --assign first).",
)
def main(
    verbose: bool,
    id: Optional[str],
    name: Optional[str],
    description: Optional[str],
    docs: tuple[str, ...],
    remove_docs: tuple[str, ...],
    metas: tuple[str, ...],
    relationships: tuple[str, ...],
    clear_rel: bool,
    print_corpus: bool,
    out: Optional[str],
    inp: Optional[str],
    df_cols: bool,
    df_row_count: bool,
    df_row: Optional[int],
    doc_ids: bool,
    doc_id: Optional[str],
    print_relationships: bool,
    relationships_for_keyword: Optional[str],
    semantic: Optional[str],
    similar_docs: Optional[str],
    num: int,
    semantic_chunks: Optional[str],
    rec: float,
    metadata_df: bool,
    metadata_keys: Optional[str],
    tdabm: Optional[str],
    graph: bool,
):
    """
    CRISP-T Corpus CLI: create and manipulate a corpus quickly from the command line.
    """
    logging.basicConfig(level=(logging.DEBUG if verbose else logging.WARNING))
    logger = logging.getLogger(__name__)

    if verbose:
        console.print("Verbose mode enabled")

    console.print("_________________________________________")
    console.print("CRISP-T: Corpus CLI")
    console.print("_________________________________________")

    # Load corpus from --inp if provided
    corpus = initialize_corpus(inp=inp)
    if not corpus:
        # Build initial corpus from CLI args
        if not id:
            console.print(f"[red]Error:[/red] "--id is required when not using --inp."")
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
        console.print(f"✓ Added {len(docs)} document(s)")

    # Remove documents
    for rid in remove_docs:
        corpus.remove_document_by_id(rid)
    if remove_docs:
        console.print(f"✓ Removed {len(remove_docs)} document(s)")

    # Update metadata
    for m in metas:
        k, v = _parse_kv(m)
        corpus.update_metadata(k, v)
    if metas:
        console.print(f"✓ Updated metadata entries: {len(metas)}")

    # Relationships
    for r in relationships:
        first, second, relation = _parse_relationship(r)
        corpus.add_relationship(first, second, relation)
    if relationships:
        console.print(f"✓ Added {len(relationships)} relationship(s)")
    if clear_rel:
        corpus.clear_relationships()
        console.print("✓ Cleared relationships")

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
        console.print(f"Relationships: {rels}")

    # Print relationships for keyword
    if relationships_for_keyword:
        rels = corpus.get_all_relationships_for_keyword(relationships_for_keyword)
        console.print(f"Relationships for keyword '{relationships_for_keyword}': {rels}")

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
                if "address" in str(network_error).lower() or "download" in str(network_error).lower():
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            corpus = semantic_analyzer.get_similar(semantic, n_results=num)
            console.print(f"✓ Found {len(corpus.documents)} similar documents")
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
                if "address" in str(network_error).lower() or "download" in str(network_error).lower():
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise

            # Get similar document IDs
            similar_doc_ids = semantic_analyzer.get_similar_documents(
                document_ids=similar_docs,
                n_results=num,
                threshold=threshold
            )

            console.print(f"✓ Found {len(similar_doc_ids)} similar documents")
            if similar_doc_ids:
                console.print("\nSimilar Document IDs:")
                for doc_id in similar_doc_ids:
                    doc = corpus.get_document_by_id(doc_id)
                    doc_name = f" ({doc.name})" if doc and doc.name else ""
                    console.print(f"  - {doc_id}{doc_name}")
                console.print("\nHint: Use --doc-id to view individual documents")
                console.print("Hint: This feature is useful for literature reviews to find similar documents")
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

                console.print(f"\nPerforming semantic chunk search for: '{semantic_chunks}'")
                console.print(f"Document ID: {doc_id}")
                console.print(f"Threshold: {rec}")

                # Try with default embeddings first, fall back to simple embeddings
                try:
                    semantic_analyzer = Semantic(corpus)
                except Exception as network_error:
                    # If network error or download fails, try simple embeddings
                    if "address" in str(network_error).lower() or "download" in str(network_error).lower():
                        console.print("Note: Using simple embeddings (network unavailable)")
                        semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                    else:
                        raise

                # Get similar chunks
                chunks = semantic_analyzer.get_similar_chunks(
                    query=semantic_chunks,
                    doc_id=doc_id,
                    threshold=rec,
                    n_results=20  # Get more chunks to filter by threshold
                )

                console.print(f"✓ Found {len(chunks)} matching chunks")
                console.print("\nMatching chunks:")
                console.print("=" * 60)
                for i, chunk in enumerate(chunks, 1):
                    console.print(f"\nChunk {i}:")
                    console.print(chunk)
                    console.print("-" * 60)

                if len(chunks) == 0:
                    console.print("No chunks matched the query above the threshold.")
                    console.print("Hint: Try lowering the threshold with --rec or use a different query.")
                else:
                    console.print(f"\nHint: These {len(chunks)} chunks can be used for coding/annotating the document.")
                    console.print("Hint: Adjust --rec threshold to get more or fewer results.")

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
                if "address" in str(network_error).lower() or "download" in str(network_error).lower():
                    console.print("Note: Using simple embeddings (network unavailable)")
                    semantic_analyzer = Semantic(corpus, use_simple_embeddings=True)
                else:
                    raise
            # Parse metadata_keys if provided
            keys_list = None
            if metadata_keys:
                keys_list = [k.strip() for k in metadata_keys.split(",")]
            corpus = semantic_analyzer.get_df(metadata_keys=keys_list)
            console.print("✓ Metadata exported to DataFrame")
            if corpus.df is not None:
                console.print(f"DataFrame shape: {corpus.df.shape}")
                console.print(f"Columns: {list(corpus.df.columns)}")
            console.print("Hint: Use --out to save the corpus with the updated DataFrame")
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
                    console.print(f"[red]Error:[/red] {Invalid radius value: '{parts[2]}'. Must be a number.}")
        raise typer.Exit(code=1)

            console.print(f"\nPerforming TDABM analysis...")
            console.print(f"  Y variable: {y_var}")
            console.print(f"  X variables: {x_vars}")
            console.print(f"  Radius: {radius}")

            tdabm_analyzer = Tdabm(corpus)
            result = tdabm_analyzer.generate_tdabm(y=y_var, x_variables=x_vars, radius=radius)

            console.print("\n" + result)
            console.print("\nHint: TDABM results stored in corpus metadata['tdabm']")
            console.print("Hint: Use --out to save the corpus with TDABM metadata")
            console.print("Hint: Use 'crispviz --tdabm' to visualize the results")

        except ValueError as e:
            console.print(f"Error: {e}")
            console.print("Hint: Ensure your corpus has a DataFrame with the specified variables")
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

            console.print(f"✓ Graph created successfully")
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
        console.print(f"✓ Corpus saved to {out}")

    if print_corpus:
        console.print("\n=== Corpus Details ===")
        corpus.pretty_print()

    logger.info("Corpus CLI finished")
