import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import click
import pandas as pd

from . import __version__
from .cluster import Cluster
from .helpers.initializer import initialize_corpus
from .read_data import ReadData
from .visualize import QRVisualize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Print verbose messages.")
@click.option("--inp", "-i", help="Load corpus from a folder containing corpus.json")
@click.option(
    "--out",
    "-o",
    help="Output directory where PNG images will be written",
)
@click.option(
    "--bins", default=100, show_default=True, help="Number of bins for distributions"
)
@click.option(
    "--topics-num",
    default=8,
    show_default=True,
    help="Number of topics for LDA when required (default 8 as per Mettler et al. 2025)",
)
@click.option(
    "--top-n",
    default=20,
    show_default=True,
    help="Top N terms to show in top-terms chart",
)
@click.option(
    "--corr-columns",
    default="",
    help="Comma separated numeric columns for correlation heatmap; if empty, auto-select",
)
@click.option("--freq", is_flag=True, help="Export: word frequency distribution")
@click.option(
    "--by-topic",
    is_flag=True,
    help="Export: distribution by dominant topic (requires LDA)",
)
@click.option(
    "--wordcloud", is_flag=True, help="Export: topic wordcloud (requires LDA)"
)
@click.option(
    "--ldavis", is_flag=True, help="Export: interactive LDA visualization HTML (requires LDA)"
)
@click.option(
    "--top-terms", is_flag=True, help="Export: top terms bar chart (computed from text)"
)
@click.option(
    "--corr-heatmap",
    is_flag=True,
    help="Export: correlation heatmap (from CSV numeric columns)",
)
@click.option(
    "--tdabm",
    is_flag=True,
    help="Export: TDABM visualization (requires TDABM analysis in corpus metadata)",
)
def main(
    verbose: bool,
    inp: Optional[str],
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
):
    """CRISP-T: Visualization CLI

    Build corpus (source preferred over inp), optionally handle multiple sources,
    and export selected visualizations as PNG files into the output directory.
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Verbose mode enabled")

    click.echo("_________________________________________")
    click.echo("CRISP-T: Visualizations")
    click.echo(f"Version: {__version__}")
    click.echo("_________________________________________")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    read_data = ReadData()
    corpus = None

    corpus = initialize_corpus(inp=inp)

    # if --inp provided, and not --out, set out to inp
    if inp and not out:
        out = inp
        click.echo(f"Output path not provided. Using input path as output: {out}")

    if not corpus:
        raise click.ClickException("No input provided. Use --source/--sources or --inp")

    viz = QRVisualize(corpus=corpus)

    # Helper: build LDA if by-topic or wordcloud requested
    cluster_instance = None
    def ensure_topics():
        nonlocal cluster_instance
        if cluster_instance is None:
            cluster_instance = Cluster(corpus=corpus)
            cluster_instance.build_lda_model(topics=topics_num)
            # Populate visualization structures used by QRVisualize
            cluster_instance.format_topics_sentences(visualize=True)
        return cluster_instance

    # 1) Word frequency distribution
    if freq:
        df_text = pd.DataFrame(
            {"Text": [getattr(doc, "text", "") or "" for doc in corpus.documents]}
        )
        out_path = out_dir / "word_frequency.png"
        viz.plot_frequency_distribution_of_words(
            df=df_text, folder_path=str(out_path), bins=bins, show=False
        )
        click.echo(f"Saved: {out_path}")

    # 2) Distribution by topic (requires topics)
    if by_topic:
        ensure_topics()
        out_path = out_dir / "by_topic.png"
        viz.plot_distribution_by_topic(
            df=None, folder_path=str(out_path), bins=bins, show=False
        )
        click.echo(f"Saved: {out_path}")

    # 3) Topic wordcloud (requires topics)
    if wordcloud:
        ensure_topics()
        out_path = out_dir / "wordcloud.png"
        viz.plot_wordcloud(topics=None, folder_path=str(out_path), show=False)
        click.echo(f"Saved: {out_path}")

    # 3.5) LDA visualization (requires topics)
    if ldavis:
        cluster = ensure_topics()
        out_path = out_dir / "lda_visualization.html"
        try:
            viz.get_lda_viz(
                lda_model=cluster._lda_model,
                corpus_bow=cluster._bag_of_words,
                dictionary=cluster._dictionary,
                folder_path=str(out_path),
                show=False
            )
            click.echo(f"Saved: {out_path}")
        except ImportError as e:
            click.echo(f"Warning: {e}")
        except Exception as e:
            click.echo(f"Error generating LDA visualization: {e}")

    # 4) Top terms (compute from text directly)
    if top_terms:
        texts = [getattr(doc, "text", "") or "" for doc in corpus.documents]
        tokens = []
        for t in texts:
            tokens.extend((t or "").lower().split())
        freq_map = Counter(tokens)
        if not freq_map:
            click.echo("No tokens found to plot top terms.")
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
            click.echo(f"Saved: {out_path}")

    # 5) Correlation heatmap
    if corr_heatmap:
        if getattr(corpus, "df", None) is None or corpus.df.empty:
            click.echo("No CSV data available for correlation heatmap; skipping.")
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
            click.echo(f"Saved: {out_path}")

    # TDABM visualization
    if tdabm:
        if 'tdabm' not in corpus.metadata:
            click.echo("Warning: No TDABM data found in corpus metadata.")
            click.echo("Hint: Run TDABM analysis first with: crispt --tdabm y_var:x_vars:radius --inp <corpus_dir>")
        else:
            out_path = out_dir / "tdabm.png"
            try:
                viz.draw_tdabm(corpus=corpus, folder_path=str(out_path), show=False)
                click.echo(f"Saved: {out_path}")
            except Exception as e:
                click.echo(f"Error generating TDABM visualization: {e}")
                logger.error(f"TDABM visualization error: {e}", exc_info=True)

    click.echo("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()
