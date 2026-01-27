import logging
import warnings
from collections import Counter
from pathlib import Path

import click
import pandas as pd

from . import __version__
from .cluster import Cluster
from .helpers.initializer import initialize_corpus
from .visualize import QRVisualize

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress and debugging information.")
@click.option("--inp", "-i", help="Load an existing corpus from a folder containing corpus.json.")
@click.option(
    "--out",
    "-o",
    help="Output directory where visualization images (PNG/HTML) will be saved.",
)
@click.option(
    "--bins", default=100, show_default=True, help="Number of bins for frequency distribution charts."
)
@click.option(
    "--topics-num",
    default=8,
    show_default=True,
    help="Number of topics for LDA analysis when required (Mettler et al. 2025 recommends 8).",
)
@click.option(
    "--top-n",
    default=20,
    show_default=True,
    help="Number of top terms to display in the top-terms bar chart.",
)
@click.option(
    "--corr-columns",
    default="",
    help="Comma-separated list of numeric columns for correlation heatmap. Auto-selected if empty.",
)
@click.option("--freq", is_flag=True, help="Generate word frequency distribution visualization.")
@click.option(
    "--by-topic",
    is_flag=True,
    help="Generate distribution by dominant topic (requires LDA topic modeling first).",
)
@click.option(
    "--wordcloud", is_flag=True, help="Generate topic word cloud visualization (requires LDA first)."
)
@click.option(
    "--ldavis",
    is_flag=True,
    help="Generate interactive LDA visualization as HTML (requires LDA and pyLDAvis).",
)
@click.option(
    "--top-terms", is_flag=True, help="Generate top terms bar chart based on word frequencies."
)
@click.option(
    "--corr-heatmap",
    is_flag=True,
    help="Generate correlation heatmap from numeric columns in your CSV data.",
)
@click.option(
    "--tdabm",
    is_flag=True,
    help="Generate TDABM visualization (requires TDABM analysis in corpus metadata). Run 'crispt --tdabm' first.",
)
@click.option(
    "--graph",
    is_flag=True,
    help="Generate graph visualization (requires graph data in corpus metadata). Run 'crispt --graph' first.",
)
@click.option(
    "--graph-nodes",
    default="",
    help=(
        "Comma-separated node types to include: document,keyword,cluster,metadata. "
        "Example: --graph-nodes document,keyword. Leave empty or use 'all' for all types."
    ),
)
@click.option(
    "--graph-layout",
    default="spring",
    show_default=True,
    help="Layout algorithm for graph: spring (default), circular, kamada_kawai, or spectral.",
)
def main(
    verbose: bool,
    inp: str | None,
    out: str,
    bins: int,
    topics_num: int,
    top_n: int,
    corr_columns: str,
    freq: bool,
    by_topic: bool,
    wordcloud: bool,
    ldavis: bool,
    top_terms: bool,
    corr_heatmap: bool,
    tdabm: bool,
    graph: bool,
    graph_nodes: str,
    graph_layout: str,
):
    """CRISP-T: Visualization CLI

    Generate publication-quality visualizations from your corpus data.
    Supports word clouds, topic distributions, correlation heatmaps, and more.

    \b
    📊 GETTING STARTED:

    Step 1: Ensure you have an analyzed corpus (created with 'crisp' or 'crispt')

    Step 2: Create an output directory for your visualizations:
       mkdir visualizations

    Step 3: Generate visualizations:
       crispviz --inp crisp_input --out visualizations --freq --wordcloud --topics-num 8

    \b
    💡 TIPS:
    • Some visualizations require prior analysis (e.g., --wordcloud needs topics)
    • Use --ldavis for interactive HTML visualization
    • Combine multiple flags to generate several visualizations at once
    • Results are saved as PNG images (or HTML for --ldavis)

    \b
    📖 For examples and detailed usage, see: notes/DEMO.md
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo(click.style("✓ Verbose mode enabled", fg="cyan"))

    # Display banner with colors
    click.echo(click.style("\n" + "=" * 60, fg="blue", bold=True))
    click.echo(click.style("CRISP-T Visualizations", fg="green", bold=True))
    click.echo(click.style(f"Version: {__version__}", fg="cyan"))
    click.echo(click.style("=" * 60 + "\n", fg="blue", bold=True))

    try:
        out_dir = Path(out)
    except TypeError:
        click.echo(click.style("❌ Error: ", fg="red", bold=True) +
                  "No output directory specified.")
        click.echo(click.style("\n💡 Tip: ", fg="cyan") +
                  "Use " + click.style("--out <directory>", fg="green") +
                  " to specify where visualizations should be saved")
        raise click.Abort() from None
    out_dir.mkdir(parents=True, exist_ok=True)
    click.echo(click.style("✓ Output directory: ", fg="green") +
              click.style(str(out_dir), fg="cyan"))

    # Initialize components
    corpus = None

    click.echo(click.style("\n📂 Loading corpus...", fg="yellow"))
    corpus = initialize_corpus(inp=inp)

    if not corpus:
        raise click.ClickException(
            click.style("❌ Error: ", fg="red", bold=True) +
            "No input provided. Use " + click.style("--inp <corpus_folder>", fg="green")
        )

    click.echo(click.style("✓ Corpus loaded: ", fg="green") +
              f"{len(corpus.documents)} document(s)")

    viz = QRVisualize(corpus=corpus)

    # Helper: build LDA if by-topic or wordcloud requested
    cluster_instance = None

    def ensure_topics():
        nonlocal cluster_instance
        if cluster_instance is None:
            click.echo(click.style("\n⚙️  Building topic model...", fg="yellow"))
            cluster_instance = Cluster(corpus=corpus)
            cluster_instance.build_lda_model(topics=topics_num)
            # Populate visualization structures used by QRVisualize
            cluster_instance.format_topics_sentences(visualize=True)
            click.echo(click.style(f"✓ Topic model ready ({topics_num} topics)", fg="green"))
        return cluster_instance

    click.echo(click.style("\n╔═══════════════════════════════════════════════╗", fg="blue", bold=True))
    click.echo(click.style("║  🎨 GENERATING VISUALIZATIONS                ║", fg="blue", bold=True))
    click.echo(click.style("╚═══════════════════════════════════════════════╝\n", fg="blue", bold=True))

    # 1) Word frequency distribution
    if freq:
        click.echo(click.style("▸ Creating word frequency distribution...", fg="yellow"))
        df_text = pd.DataFrame(
            {"Text": [getattr(doc, "text", "") or "" for doc in corpus.documents]}
        )
        out_path = out_dir / "word_frequency.png"
        viz.plot_frequency_distribution_of_words(
            df=df_text, folder_path=str(out_path), bins=bins, show=False
        )
        click.echo(click.style("  ✓ Saved: ", fg="green") +
                  click.style(str(out_path), fg="cyan"))

    # 2) Distribution by topic (requires topics)
    if by_topic:
        click.echo(click.style("▸ Creating distribution by topic...", fg="yellow"))
        ensure_topics()
        out_path = out_dir / "by_topic.png"
        viz.plot_distribution_by_topic(
            df=None, folder_path=str(out_path), bins=bins, show=False
        )
        click.echo(click.style("  ✓ Saved: ", fg="green") +
                  click.style(str(out_path), fg="cyan"))

    # 3) Topic wordcloud (requires topics)
    if wordcloud:
        click.echo(click.style("▸ Creating topic word cloud...", fg="yellow"))
        ensure_topics()
        out_path = out_dir / "wordcloud.png"
        viz.plot_wordcloud(topics=None, folder_path=str(out_path), show=False)
        click.echo(click.style("  ✓ Saved: ", fg="green") +
                  click.style(str(out_path), fg="cyan"))

    # 3.5) LDA visualization (requires topics)
    if ldavis:
        click.echo(click.style("▸ Creating interactive LDA visualization...", fg="yellow"))
        cluster = ensure_topics()
        out_path = out_dir / "lda_visualization.html"
        try:
            viz.get_lda_viz(
                lda_model=cluster._lda_model,
                corpus_bow=cluster._bag_of_words,
                dictionary=cluster._dictionary,
                folder_path=str(out_path),
                show=False,
            )
            click.echo(click.style("  ✓ Saved: ", fg="green") +
                      click.style(str(out_path), fg="cyan"))
        except ImportError as e:
            click.echo(click.style(f"  ⚠️  Warning: {e}", fg="yellow"))
        except Exception as e:
            click.echo(click.style("  ❌ Error: ", fg="red") + str(e))

    # 4) Top terms (compute from text directly)
    if top_terms:
        click.echo(click.style("▸ Creating top terms bar chart...", fg="yellow"))
        texts = [getattr(doc, "text", "") or "" for doc in corpus.documents]
        tokens = []
        for t in texts:
            tokens.extend((t or "").lower().split())
        freq_map = Counter(tokens)
        if not freq_map:
            click.echo(click.style("  ⚠️  No tokens found to plot", fg="yellow"))
        else:
            df_terms = pd.DataFrame(
                {
                    "term": list(freq_map.keys()),
                    "frequency": list(freq_map.values()),
                }
            )
            # QRVisualize sorts internally; we just pass full DF
            out_path = out_dir / "top_terms.png"
            viz.plot_top_terms(
                df=df_terms, top_n=top_n, folder_path=str(out_path), show=False
            )
            click.echo(click.style("  ✓ Saved: ", fg="green") +
                      click.style(str(out_path), fg="cyan"))

    # 5) Correlation heatmap
    if corr_heatmap:
        click.echo(click.style("▸ Creating correlation heatmap...", fg="yellow"))
        if getattr(corpus, "df", None) is None or corpus.df.empty:
            click.echo(click.style("  ⚠️  No CSV data available for correlation heatmap; skipping.", fg="yellow"))
        else:
            df0 = corpus.df.copy()
            # If user specified columns, attempt to use them; else let visualize auto-select
            cols = (
                [c.strip() for c in corr_columns.split(",") if c.strip()]
                if corr_columns
                else None
            )
            out_path = out_dir / "corr_heatmap.png"
            if cols:
                # Pass subset to avoid rename ambiguity
                sub = (
                    df0[cols].copy().select_dtypes(include=["number"])
                )  # ensure numeric
                viz.plot_correlation_heatmap(
                    df=sub, columns=None, folder_path=str(out_path), show=False
                )
            else:
                viz.plot_correlation_heatmap(
                    df=df0, columns=None, folder_path=str(out_path), show=False
                )
            click.echo(click.style("  ✓ Saved: ", fg="green") +
                      click.style(str(out_path), fg="cyan"))

    # TDABM visualization
    if tdabm:
        click.echo(click.style("▸ Creating TDABM visualization...", fg="yellow"))
        if "tdabm" not in corpus.metadata:
            click.echo(click.style("  ⚠️  No TDABM data found in corpus metadata.", fg="yellow"))
            click.echo(click.style("  💡 Tip: ", fg="cyan") +
                      "Run TDABM analysis first with: " +
                      click.style("crispt --tdabm y_var:x_vars:radius --inp <corpus_dir>", fg="green"))
        else:
            out_path = out_dir / "tdabm.png"
            try:
                viz.draw_tdabm(corpus=corpus, folder_path=str(out_path), show=False)
                click.echo(click.style("  ✓ Saved: ", fg="green") +
                          click.style(str(out_path), fg="cyan"))
            except Exception as e:
                click.echo(click.style("  ❌ Error: ", fg="red") + str(e))
                logger.error(f"TDABM visualization error: {e}", exc_info=True)

    # Graph visualization (filtered by node types if provided)
    if graph or graph_nodes:
        click.echo(click.style("▸ Creating graph visualization...", fg="yellow"))
        if "graph" not in corpus.metadata:
            click.echo(click.style("  ⚠️  No graph data found in corpus metadata.", fg="yellow"))
            click.echo(click.style("  💡 Tip: ", fg="cyan") +
                      "Run graph generation first with: " +
                      click.style("crispt --graph --inp <corpus_dir>", fg="green"))
        else:
            raw_types = (graph_nodes or "").strip().lower()
            include_all = raw_types in ("", "all", "*")
            allowed_types = {"document", "keyword", "cluster", "metadata"}
            requested_types = set()
            if not include_all:
                for part in raw_types.split(","):
                    p = part.strip()
                    if not p:
                        continue
                    if p in allowed_types:
                        requested_types.add(p)
                    else:
                        click.echo(click.style(
                            f"  ⚠️  Unknown node type '{p}' ignored. ", fg="yellow") +
                            f"Allowed: {', '.join(sorted(allowed_types))}"
                        )
                if not requested_types:
                    click.echo(click.style("  ℹ️  No valid node types specified; defaulting to all.", fg="blue"))
                    include_all = True

            graph_data = corpus.metadata.get("graph", {})
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            if include_all:
                filtered_nodes = nodes
                filtered_edges = edges
            else:
                filtered_nodes = [n for n in nodes if n.get("label") in requested_types]
                kept_ids = {str(n.get("id")) for n in filtered_nodes}
                filtered_edges = [
                    e
                    for e in edges
                    if str(e.get("source")) in kept_ids
                    and str(e.get("target")) in kept_ids
                ]

            # Build a shallow copy of graph metadata with filtered components
            filtered_graph_meta = dict(graph_data)
            filtered_graph_meta["nodes"] = filtered_nodes
            filtered_graph_meta["edges"] = filtered_edges
            filtered_graph_meta["num_nodes"] = len(filtered_nodes)
            filtered_graph_meta["num_edges"] = len(filtered_edges)
            filtered_graph_meta["num_documents"] = sum(
                1 for n in filtered_nodes if n.get("label") == "document"
            )

            # Inject temporary filtered metadata for visualization
            original_graph_meta = corpus.metadata.get("graph")
            corpus.metadata["graph"] = filtered_graph_meta
            out_path = out_dir / "graph.png"
            try:
                viz.draw_graph(
                    corpus=corpus,
                    folder_path=str(out_path),
                    show=False,
                    layout=graph_layout,
                )
                click.echo(click.style("  ✓ Saved: ", fg="green") +
                          click.style(str(out_path), fg="cyan"))
                if not include_all:
                    click.echo(click.style(
                        f"  ℹ️  Filtered to node types: {', '.join(sorted(requested_types))}",
                        fg="blue"
                    ))
            except Exception as e:
                click.echo(click.style("  ❌ Error: ", fg="red") + str(e))
                logger.error(f"Graph visualization error: {e}", exc_info=True)
            finally:
                # Restore original metadata (avoid side-effects)
                corpus.metadata["graph"] = original_graph_meta

    click.echo(click.style("\n" + "=" * 60, fg="green", bold=True))
    click.echo(click.style("✓ Visualization Complete!", fg="green", bold=True))
    click.echo(click.style("✓ All visualizations saved to: ", fg="green") +
              click.style(str(out_dir), fg="cyan"))
    click.echo(click.style("=" * 60 + "\n", fg="green", bold=True))


if __name__ == "__main__":
    main()
