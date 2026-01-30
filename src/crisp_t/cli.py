import logging
import pathlib
import warnings
from pathlib import Path

import click

from . import __version__
from .cluster import Cluster
from .helpers.analyzer import get_analyzers
from .helpers.clib import clear_cache
from .helpers.clib.ui import (
    format_error,
    format_info,
    format_success,
    print_section_header,
)
from .helpers.initializer import initialize_corpus
from .read_data import ReadData
from .sentiment import Sentiment

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .ml import ML

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning(
        "ML dependencies not available. Install with: pip install crisp-t[ml]"
    )


@click.command()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress and debugging information.",
)
@click.option(
    "--covid",
    "-cf",
    default="",
    help="Download COVID-19 narratives from the specified website.",
)
@click.option(
    "--inp",
    "-i",
    help="Load an existing corpus from a folder containing corpus.json and corpus_df.csv files.",
)
@click.option(
    "--out",
    "-o",
    help="Save the corpus to a folder. The corpus will be saved as corpus.json and corpus_df.csv.",
)
@click.option(
    "--csv",
    default="",
    help="[Deprecated] CSV file name. Please place CSV in --source folder instead.",
)
@click.option(
    "--num",
    "-n",
    default=3,
    help="Numerical parameter for analysis (e.g., number of clusters, topics, or epochs). When used with --source, limits the maximum number of text/PDF files to import. Default: 3.",
)
@click.option(
    "--rec",
    "-r",
    default=3,
    help="Number of top results to display or record index for specific operations. When used with --source, limits the maximum number of CSV rows to import. Default: 3.",
)
@click.option(
    "--unstructured",
    "-t",
    multiple=True,
    help="Specify CSV columns containing free-form text (e.g., open-ended survey responses). Can be used multiple times.",
)
@click.option(
    "--filters",
    "-f",
    multiple=True,
    help="Filter documents/rows by metadata or links. Format: key=value (regular) or embedding:text/embedding:df/temporal:text/temporal:df (link filters). Can be used multiple times.",
)
@click.option(
    "--codedict",
    is_flag=True,
    help="Generate a qualitative coding dictionary from your text data.",
)
@click.option(
    "--topics",
    is_flag=True,
    help="Perform topic modeling using Latent Dirichlet Allocation (LDA).",
)
@click.option(
    "--assign", is_flag=True, help="Assign each document to its most relevant topic."
)
@click.option(
    "--cat",
    is_flag=True,
    help="Extract and list common categories or themes from the corpus.",
)
@click.option(
    "--summary",
    is_flag=True,
    help="Generate an extractive summary of the text data.",
)
@click.option(
    "--sentiment",
    is_flag=True,
    help="Analyze sentiment in the text using VADER sentiment analysis.",
)
@click.option(
    "--sentence",
    is_flag=True,
    default=False,
    help="Generate sentence-level analysis (use with --sentiment for document-level scores).",
)
@click.option(
    "--nlp",
    is_flag=True,
    help="Run all available natural language processing analyses (coding dictionary, topics, categories, summary, sentiment).",
)
@click.option(
    "--ml",
    is_flag=True,
    help="Run all available machine learning analyses (requires crisp-t[ml] installation).",
)
@click.option(
    "--nnet", is_flag=True, help="Train and evaluate a neural network classifier."
)
@click.option(
    "--cls",
    is_flag=True,
    help="Perform classification using Support Vector Machine (SVM) and Decision Tree algorithms.",
)
@click.option(
    "--knn",
    is_flag=True,
    help="Perform K-Nearest Neighbors search to find similar records.",
)
@click.option(
    "--kmeans",
    is_flag=True,
    help="Perform K-Means clustering to group similar records.",
)
@click.option(
    "--cart",
    is_flag=True,
    help="Generate association rules using the Apriori algorithm (CART).",
)
@click.option(
    "--pca",
    is_flag=True,
    help="Perform Principal Component Analysis (PCA) for dimensionality reduction.",
)
@click.option(
    "--regression",
    is_flag=True,
    help="Perform regression analysis (linear or logistic, automatically detected).",
)
@click.option(
    "--lstm",
    is_flag=True,
    help="Train an LSTM (Long Short-Term Memory) neural network on text data to predict outcomes.",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualizations for words, topics, and word clouds.",
)
@click.option(
    "--ignore",
    default="",
    help="Comma-separated list of words or columns to exclude from analysis.",
)
@click.option(
    "--include",
    default="",
    help="Comma-separated list of columns to include in the analysis.",
)
@click.option(
    "--outcome",
    default="",
    help="Specify the target variable (outcome) for machine learning tasks. Can be a DataFrame column name OR a text metadata field name (when --linkage is specified).",
)
@click.option(
    "--linkage",
    type=click.Choice(["id", "embedding", "temporal", "keyword"], case_sensitive=False),
    default=None,
    help="Linkage method to use when outcome is a text metadata field. Choices: id, embedding, temporal, keyword.",
)
@click.option(
    "--aggregation",
    type=click.Choice(["majority", "mean", "first", "mode"], case_sensitive=False),
    default="majority",
    help="Aggregation strategy when multiple documents link to one row. Default: majority (for classification) or mean (for regression).",
)
@click.option(
    "--source",
    "--import",
    "-s",
    help="Source directory or URL containing your data files (.txt, .pdf, and .csv).",
)
@click.option(
    "--print",
    "-p",
    "print_args",
    multiple=True,
    help="Display corpus information. Examples: --print documents --print 10, or --print 'documents 10'",
)
@click.option(
    "--sources",
    multiple=True,
    help="Load data from multiple sources (directories or URLs). Can be used multiple times.",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear the cache before running analysis. Use when switching between datasets.",
)
def main(
    verbose,
    covid,
    inp,
    out,
    csv,
    num,
    rec,
    unstructured,
    filters,
    codedict,
    topics,
    assign,
    cat,
    summary,
    sentiment,
    sentence,
    nlp,
    nnet,
    cls,
    knn,
    kmeans,
    cart,
    pca,
    regression,
    lstm,
    ml,
    visualize,
    ignore,
    include,
    outcome,
    linkage,
    aggregation,
    source,
    sources,
    print_args,
    clear,
):
    """CRISP-T: Cross Industry Standard Process for Triangulation.

    A comprehensive framework for analyzing textual and numerical data using
    advanced NLP, machine learning, and statistical techniques. Designed for
    researchers and practitioners working with mixed-methods data.

    \b
    📚 GETTING STARTED - DATA PREPARATION:

    Step 1: Create a source directory (e.g., crisp_source) in your workspace

    Step 2: Add your data files to this directory:
       • Text files: Interview transcripts, field notes (.txt or .pdf format)
       • Numeric data: One CSV file with quantitative data

    Step 3: Import your data to create a corpus:
       crisp --source crisp_source --out crisp_input

    Step 4: Run analyses on your imported corpus:
       crisp --inp crisp_input [analysis options]

    \b
    💡 TIPS:
    • For CSV files with free-text columns, use --unstructured <column_name>
    • Use --help to see all available analysis options
    • Use --clear when switching between different datasets
    • Results can be saved at any stage using --out

    \b
    📖 For detailed examples, see: notes/DEMO.md
    📖 For complete documentation, visit: https://github.com/dermatologist/crisp-t/wiki
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo(click.style("✓ Verbose mode enabled", fg="cyan"))

    # Display banner with colors
    click.echo(click.style("\n" + "=" * 60, fg="blue", bold=True))
    click.echo(
        click.style("CRISP-T", fg="green", bold=True)
        + click.style(" - Qualitative Research Analysis Framework", fg="white")
    )
    click.echo(click.style(f"Version: {__version__}", fg="cyan"))
    click.echo(click.style("=" * 60 + "\n", fg="blue", bold=True))

    # Initialize components
    read_data = ReadData()
    corpus = None
    text_analyzer = None
    csv_analyzer = None
    ml_analyzer = None

    if clear:
        clear_cache()

    try:
        # Handle COVID data download
        if covid:
            if not source:
                raise click.ClickException(
                    format_error(
                        "--source (output folder) is required when using --covid."
                    )
                )
            click.echo(click.style("\n📥 Downloading COVID narratives...", fg="yellow"))
            click.echo(f"   From: {click.style(covid, fg='cyan')}")
            click.echo(f"   To: {click.style(source, fg='cyan')}")
            try:
                from .utils import QRUtils

                QRUtils.read_covid_narratives(source, covid)
                click.echo(
                    format_success(
                        f"Successfully downloaded COVID narratives to {source}"
                    )
                )
            except Exception as e:
                raise click.ClickException(format_error(f"Download failed: {e}")) from e

        # Build corpus using helpers (source preferred over inp)
        # if not source or inp, use default folders or env vars
        try:
            text_cols = ",".join(unstructured) if unstructured else ""
            # When using --source, num and rec are used for import limits
            max_text_files = num if source and num > 3 else None
            max_csv_rows = rec if source and rec > 3 else None
            corpus = initialize_corpus(
                source=source,
                inp=inp,
                comma_separated_text_columns=text_cols,
                comma_separated_ignore_words=(ignore if ignore else None),
                max_text_files=max_text_files,
                max_csv_rows=max_csv_rows,
            )
            # If filters were provided with ':' while using --source, emit guidance message
            if (
                source
                and filters
                and any(":" in flt and "=" not in flt for flt in filters)
            ):
                click.echo(format_info("Filters are not supported when using --source"))
        except click.ClickException:
            raise
        except Exception as e:
            click.echo(
                format_error(f"Error initializing corpus: {e}"),
                err=True,
            )
            logger.exception(f"Failed to initialize corpus: {e}")
            return

        # Handle multiple sources (unchanged behavior, but no filters applied here)
        if sources and not corpus:
            click.echo(
                click.style("\n📥 Loading data from multiple sources...", fg="yellow")
            )
            loaded_any = False
            # When using --sources, num and rec are used for import limits
            max_text_files = num if sources and num > 3 else None
            max_csv_rows = rec if sources and rec > 3 else None
            for src in sources:
                click.echo(f"   • Reading from: {click.style(src, fg='cyan')}")
                try:
                    read_data.read_source(
                        src,
                        comma_separated_ignore_words=ignore if ignore else None,
                        max_text_files=max_text_files,
                        max_csv_rows=max_csv_rows,
                    )
                    loaded_any = True
                    click.echo(format_success("Loaded successfully", indent=5))
                except Exception as e:
                    logger.exception(f"Failed to read source {src}: {e}")
                    raise click.ClickException(
                        format_error(f"Failed to load source: {e}")
                    ) from e

            if loaded_any:
                corpus = read_data.create_corpus(
                    name="Corpus from multiple sources",
                    description=f"Data loaded from {len(sources)} sources",
                )
                click.echo(
                    click.style(
                        f"\n✓ Successfully loaded {len(corpus.documents)} document(s) from {len(sources)} source(s)",
                        fg="green",
                        bold=True,
                    )
                )
                # Filters are not applied for --sources in bulk mode

        # Initialize analyzers with unified filter logic
        if corpus:
            try:
                text_cols = ",".join(unstructured) if unstructured else ""
                text_analyzer, csv_analyzer = get_analyzers(
                    corpus,
                    comma_separated_unstructured_text_columns=text_cols,
                    comma_separated_ignore_columns=(ignore if ignore else ""),
                    filters=filters,
                )
            except Exception as e:
                click.echo(
                    click.style(
                        "❌ Error initializing analyzers: ", fg="red", bold=True
                    )
                    + str(e),
                    err=True,
                )
                logger.exception(f"Failed to initialize analyzers: {e}")
                return

        # Load CSV data (deprecated)
        if csv:
            click.echo(
                click.style("⚠️  Warning: ", fg="yellow", bold=True)
                + "--csv option has been deprecated. Put csv file in --source folder instead."
            )

        # Initialize ML analyzer if available and ML functions are requested
        if (
            ML_AVAILABLE
            and (
                nnet or cls or knn or kmeans or cart or pca or regression or lstm or ml
            )
            and csv_analyzer
        ):
            if include:
                csv_analyzer.comma_separated_include_columns(include)
            ml_analyzer = ML(csv=csv_analyzer)  # type: ignore
        else:
            if (
                nnet or cls or knn or kmeans or cart or pca or regression or lstm or ml
            ) and not ML_AVAILABLE:
                click.echo(
                    click.style(
                        "⚠️  Machine Learning features require additional dependencies.",
                        fg="yellow",
                    )
                )
                click.echo(
                    click.style("   Install with: ", fg="white")
                    + click.style("pip install crisp-t[ml]", fg="cyan", bold=True)
                )
            if (
                nnet or cls or knn or kmeans or cart or pca or regression or lstm or ml
            ) and not csv_analyzer:
                click.echo(
                    click.style("⚠️  ML analysis requires CSV data. ", fg="yellow")
                    + "Use --csv to provide a data file or include CSV in --source folder."
                )

        # Ensure we have data to work with
        if not corpus and not csv_analyzer:
            click.echo(
                click.style("\n⚠️  No input data provided.", fg="yellow", bold=True)
            )
            click.echo(click.style("\n💡 Quick Start Guide:", fg="cyan", bold=True))
            click.echo("   1. Place your data files in a folder (e.g., crisp_source)")
            click.echo("   2. Import the data:")
            click.echo(
                click.style(
                    "      crisp --source crisp_source --out crisp_input", fg="green"
                )
            )
            click.echo("   3. Run analyses:")
            click.echo(
                click.style(
                    "      crisp --inp crisp_input --topics --sentiment", fg="green"
                )
            )
            click.echo(
                "\n   Run "
                + click.style("crisp --help", fg="cyan")
                + " for all options\n"
            )
            return

        # Text Analysis Operations
        if text_analyzer:
            if nlp or codedict:
                print_section_header(
                    "CODING DICTIONARY GENERATION", emoji="📖", color="blue"
                )
                click.echo(
                    click.style("\nWhat is a Coding Dictionary?", fg="cyan", bold=True)
                )
                click.echo(
                    "   A structured representation of your qualitative data organized into:"
                )
                click.echo(
                    click.style("   • CATEGORY:", fg="yellow")
                    + " Main actions or themes (common verbs)"
                )
                click.echo(
                    click.style("   • PROPERTY:", fg="yellow")
                    + " Concepts associated with each category (common nouns)"
                )
                click.echo(
                    click.style("   • DIMENSION:", fg="yellow")
                    + " Characteristics of each property (adjectives, adverbs)"
                )
                click.echo(
                    click.style("\n💡 Tips for Better Results:", fg="cyan", bold=True)
                )
                click.echo(
                    f"   • Use {click.style('--ignore', fg='green')} with common words you want to exclude"
                )
                click.echo(
                    f"   • Use {click.style('--filters', fg='green')} to focus on specific document subsets"
                )
                click.echo(
                    f"   • Use {click.style('--num', fg='green')} to control the number of categories shown"
                )
                click.echo(
                    f"   • Use {click.style('--rec', fg='green')} to control items displayed per section\n"
                )
                try:
                    text_analyzer.make_spacy_doc()
                    coding_dict = text_analyzer.print_coding_dictionary(
                        num=num, top_n=rec
                    )
                    if out:
                        _save_output(coding_dict, out, "coding_dictionary")
                        click.echo(
                            click.style(
                                "\n✓ Coding dictionary saved successfully", fg="green"
                            )
                        )
                except Exception as e:
                    click.echo(
                        click.style(
                            "\n❌ Error generating coding dictionary: ",
                            fg="red",
                            bold=True,
                        )
                        + str(e)
                    )

            if nlp or topics:
                print_section_header("TOPIC MODELING (LDA)", emoji="🎯", color="blue")
                click.echo(
                    click.style("\nWhat is Topic Modeling?", fg="cyan", bold=True)
                )
                click.echo("   Discovers hidden thematic structure in your text data.")
                click.echo(
                    "   Each topic is represented as a weighted combination of words."
                )
                click.echo(click.style("\n📊 Output Format:", fg="cyan", bold=True))
                click.echo(
                    '   Topic 0: 0.116*"category" + 0.093*"comparison" + 0.070*"incident" + ...'
                )
                click.echo("   (Higher weights = more important words for that topic)")
                click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
                click.echo(
                    f"   • Use {click.style('--num', fg='green')} to set the number of topics (default: 3)"
                )
                click.echo(
                    f"   • Use {click.style('--rec', fg='green')} to control words shown per topic"
                )
                click.echo(
                    f"   • Use {click.style('--filters', fg='green')} to analyze specific document subsets\n"
                )
                try:
                    cluster_analyzer = Cluster(corpus=corpus)
                    cluster_analyzer.build_lda_model(topics=num)
                    topics_result = cluster_analyzer.print_topics(num_words=rec)
                    click.echo(
                        click.style(
                            f"\n✓ Generated {len(topics_result)} topics with weights shown above",
                            fg="green",
                        )
                    )
                    if out:
                        _save_output(topics_result, out, "topics")
                        click.echo(format_success("Topics saved successfully"))
                except Exception as e:
                    click.echo(format_error(f"Error generating topics: {e}"))

            if nlp or assign:
                print_section_header(
                    "DOCUMENT-TOPIC ASSIGNMENTS", emoji="📌", color="blue"
                )
                click.echo(click.style("\nWhat does this do?", fg="cyan", bold=True))
                click.echo(
                    "   Assigns each document to its most relevant topic based on content similarity."
                )
                click.echo("   Shows the contribution percentage for each assignment.")
                click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
                click.echo(
                    f"   Use {click.style('--visualize', fg='green')} to prepare data for visualization\n"
                )
                try:
                    if "cluster_analyzer" not in locals():
                        cluster_analyzer = Cluster(corpus=corpus)
                        cluster_analyzer.build_lda_model(topics=num)
                    assignments = cluster_analyzer.format_topics_sentences(
                        visualize=visualize
                    )
                    document_assignments = cluster_analyzer.print_clusters()
                    click.echo(
                        click.style(
                            f"\n✓ Assigned {len(assignments)} documents to topics",
                            fg="green",
                        )
                    )
                    if out:
                        _save_output(assignments, out, "topic_assignments")
                        click.echo(format_success("Assignments saved successfully"))
                except Exception as e:
                    click.echo(format_error(f"Error assigning topics: {e}"))

            if nlp or cat:
                print_section_header("CATEGORY ANALYSIS", emoji="🏷️", color="blue")
                click.echo(
                    click.style("\nWhat is Category Analysis?", fg="cyan", bold=True)
                )
                click.echo("   Extracts common concepts and themes from your corpus.")
                click.echo(
                    "   Results shown as a 'bag of terms' with importance weights."
                )
                click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
                click.echo(
                    f"   • Use {click.style('--num', fg='green')} to adjust the number of categories"
                )
                click.echo(
                    f"   • Use {click.style('--filters', fg='green')} to focus on specific documents\n"
                )
                try:
                    text_analyzer.make_spacy_doc()
                    categories = text_analyzer.print_categories(num=num)
                    if out:
                        _save_output(categories, out, "categories")
                        click.echo(format_success("Categories saved successfully"))
                except Exception as e:
                    click.echo(
                        click.style(
                            "\n❌ Error generating categories: ", fg="red", bold=True
                        )
                        + str(e)
                    )

            if nlp or summary:
                print_section_header("TEXT SUMMARIZATION", emoji="📝", color="blue")
                click.echo(
                    click.style("\nWhat is Text Summarization?", fg="cyan", bold=True)
                )
                click.echo(
                    "   Generates an extractive summary by selecting the most important"
                )
                click.echo("   sentences that represent the main points of your text.")
                click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
                click.echo(
                    f"   • Use {click.style('--num', fg='green')} to control summary length (number of sentences)"
                )
                click.echo(
                    f"   • Use {click.style('--filters', fg='green')} to summarize specific document subsets\n"
                )
                try:
                    text_analyzer.make_spacy_doc()
                    summary_result = text_analyzer.generate_summary(weight=num)
                    click.echo(summary_result)
                    if out:
                        _save_output(summary_result, out, "summary")
                        click.echo(format_success("Summary saved successfully"))
                except Exception as e:
                    click.echo(format_error(f"Error generating summary: {e}"))

            if nlp or sentiment:
                print_section_header(
                    "SENTIMENT ANALYSIS (VADER)", emoji="😊", color="blue"
                )
                click.echo(
                    click.style("\nWhat is Sentiment Analysis?", fg="cyan", bold=True)
                )
                click.echo("   Analyzes the emotional tone of your text using VADER")
                click.echo("   (Valence Aware Dictionary and sEntiment Reasoner).")
                click.echo(click.style("\n📊 Output Scores:", fg="cyan", bold=True))
                click.echo(
                    click.style("   • neg:", fg="red")
                    + " Negative sentiment (0.0 to 1.0)"
                )
                click.echo(
                    click.style("   • neu:", fg="yellow")
                    + " Neutral sentiment (0.0 to 1.0)"
                )
                click.echo(
                    click.style("   • pos:", fg="green")
                    + " Positive sentiment (0.0 to 1.0)"
                )
                click.echo(
                    click.style("   • compound:", fg="cyan")
                    + " Overall sentiment (-1.0 to +1.0)"
                )
                click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
                click.echo(
                    f"   • Use {click.style('--sentence', fg='green')} for document-level scores"
                )
                click.echo(
                    f"   • Use {click.style('--filters', fg='green')} to analyze specific documents\n"
                )
                try:
                    sentiment_analyzer = Sentiment(corpus=corpus)  # type: ignore
                    sentiment_results = sentiment_analyzer.get_sentiment(
                        documents=sentence, verbose=verbose
                    )
                    click.echo(sentiment_results)
                    if out:
                        _save_output(sentiment_results, out, "sentiment")
                        click.echo(
                            format_success("Sentiment analysis saved successfully")
                        )
                except Exception as e:
                    click.echo(
                        format_error(f"Error generating sentiment analysis: {e}")
                    )

        # Machine Learning Operations
        if ml_analyzer and ML_AVAILABLE:
            target_col = outcome

            if kmeans or ml:
                print_section_header("K-MEANS CLUSTERING", emoji="🔍", color="magenta")
                click.echo(
                    click.style("\nWhat is K-Means Clustering?", fg="cyan", bold=True)
                )
                click.echo(
                    "   Groups similar records together based on numeric features."
                )
                click.echo(
                    "   Automatically removes non-numeric columns and missing values."
                )
                click.echo(click.style("\n⚠️  Important:", fg="yellow", bold=True))
                click.echo(
                    "   Data preprocessing may affect compatibility with other ML analyses."
                )
                click.echo(click.style("\n💡 Tip:", fg="cyan", bold=True))
                click.echo(
                    f"   Use {click.style('--num', fg='green')} to set the number of clusters (default: 3)\n"
                )
                csv_analyzer.retain_numeric_columns_only()
                csv_analyzer.drop_na()
                _ml_analyzer = ML(csv=csv_analyzer)
                clusters, members = _ml_analyzer.get_kmeans(
                    number_of_clusters=num, verbose=verbose
                )
                _ml_analyzer.profile(members, number_of_clusters=num)
                click.echo(
                    click.style(
                        f"\n✓ Clustering complete: {num} clusters generated", fg="green"
                    )
                )
                if out:
                    _save_output(
                        {"clusters": clusters, "members": members}, out, "kmeans"
                    )
                    click.echo(format_success("Results saved successfully"))

            if (cls or ml) and target_col:
                print_section_header(
                    "CLASSIFICATION MODELS", emoji="🎯", color="magenta"
                )
                click.echo(
                    click.style(
                        "\nWhat are Classification Models?", fg="cyan", bold=True
                    )
                )
                click.echo(
                    "   Predict categorical outcomes using machine learning algorithms:"
                )
                click.echo(
                    click.style("   • SVM:", fg="yellow")
                    + " Support Vector Machine with confusion matrix"
                )
                click.echo(
                    click.style("   • Decision Tree:", fg="yellow")
                    + " Tree-based classifier with feature importance"
                )
                click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
                click.echo(
                    f"   • Use {click.style('--outcome', fg='green')} to specify your target variable"
                )
                click.echo(
                    f"   • Use {click.style('--rec', fg='green')} to control top features displayed"
                )
                click.echo(
                    f"   • Use {click.style('--include', fg='green')} to select specific features\n"
                )
                if not target_col:
                    raise click.ClickException(
                        click.style("❌ Error: ", fg="red", bold=True)
                        + "--outcome is required for classification tasks"
                    )
                click.echo(
                    click.style("\n▸ Running SVM Classification...", fg="yellow")
                )
                try:
                    confusion_matrix = ml_analyzer.svm_confusion_matrix(
                        y=target_col,
                        test_size=0.25,
                        linkage_method=linkage,
                        aggregation=aggregation,
                    )
                    click.echo(
                        ml_analyzer.format_confusion_matrix_to_human_readable(
                            confusion_matrix
                        )
                    )
                    click.echo(format_success("SVM classification complete", indent=2))
                    if out:
                        _save_output(confusion_matrix, out, "svm_results")
                except Exception as e:
                    click.echo(format_error(f"Error in SVM: {e}", indent=2))
                click.echo(
                    click.style(
                        "\n▸ Running Decision Tree Classification...", fg="yellow"
                    )
                )
                try:
                    cm, importance = ml_analyzer.get_decision_tree_classes(
                        y=target_col,
                        top_n=rec,
                        linkage_method=linkage,
                        aggregation=aggregation,
                    )
                    click.echo(
                        "\n" + click.style("Feature Importance:", fg="cyan", bold=True)
                    )
                    click.echo(
                        ml_analyzer.format_confusion_matrix_to_human_readable(cm)
                    )
                    click.echo(
                        format_success(
                            "Decision tree classification complete", indent=2
                        )
                    )
                    if out:
                        _save_output(cm, out, "decision_tree_results")
                except Exception as e:
                    click.echo(format_error(f"Error in Decision Tree: {e}", indent=2))

            if nnet or ml:
                click.echo("\n=== Neural Network Classification Accuracy ===")
                click.echo(
                    """
                            Neural Network classifier with accuracy output.
                Hint:   Use --outcome to specify the target variable for classification.
                        Use --include to specify columns to include in the analysis (comma separated).
                """
                )
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for neural network tasks"
                    )
                try:
                    predictions = ml_analyzer.get_nnet_predictions(
                        y=target_col, linkage_method=linkage, aggregation=aggregation
                    )
                    if out:
                        _save_output(predictions, out, "nnet_results")
                except Exception as e:
                    click.echo(f"Error performing Neural Network classification: {e}")

            if knn or ml:
                click.echo("\n=== K-Nearest Neighbors ===")
                click.echo(
                    """
                           K-Nearest Neighbors search results.
                Hint:   Use --outcome to specify the target variable for KNN search.
                        Use --rec to specify the record number to search from (1-based index).
                        Use --num to specify the number of nearest neighbors to retrieve.
                        Use --include to specify columns to include in the analysis (comma separated).
                """
                )
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for KNN search tasks"
                    )
                if rec < 1:
                    raise click.ClickException(
                        "--rec must be a positive integer (1-based index)"
                    )
                try:
                    knn_results = ml_analyzer.knn_search(
                        y=target_col,
                        n=num,
                        r=rec,
                        linkage_method=linkage,
                        aggregation=aggregation,
                    )
                    if out:
                        _save_output(knn_results, out, "knn_results")
                except Exception as e:
                    click.echo(f"Error performing K-Nearest Neighbors search: {e}")

            if cart or ml:
                click.echo("\n=== Association Rules (CART) ===")
                click.echo(
                    """
                           Association Rules using the Apriori algorithm.
                Hint:   Use --outcome to specify the target variable to remove from features.
                        Use --num to specify the minimum support (between 1 and 99).
                        Use --rec to specify the minimum threshold for the rules (between 1 and 99).
                        Use --include to specify columns to include in the analysis (comma separated).
                """
                )
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for association rules tasks"
                    )
                if not (1 <= num <= 99):
                    raise click.ClickException(
                        "--num must be between 1 and 99 for min_support"
                    )
                if not (1 <= rec <= 99):
                    raise click.ClickException(
                        "--rec must be between 1 and 99 for min_threshold"
                    )
                _min_support = float(num / 100)
                _min_threshold = float(rec / 100)
                click.echo(
                    f"Using min_support={_min_support:.2f} and min_threshold={_min_threshold:.2f}"
                )
                try:
                    apriori_results = ml_analyzer.get_apriori(
                        y=target_col,
                        min_support=_min_support,
                        min_threshold=_min_threshold,
                    )
                    click.echo(apriori_results)
                    if out:
                        _save_output(apriori_results, out, "association_rules")
                except Exception as e:
                    click.echo(f"Error generating association rules: {e}")

            if (pca or ml) and target_col:
                click.echo("\n=== Principal Component Analysis ===")
                click.echo(
                    """
                           Principal Component Analysis (PCA) results.
                Hint:   Use --outcome to specify the target variable to remove from features.
                        Use --num to specify the number of principal components to generate.
                        Use --include to specify columns to include in the analysis (comma separated).
                """
                )
                try:
                    pca_results = ml_analyzer.get_pca(
                        y=target_col,
                        n=num,
                        linkage_method=linkage,
                        aggregation=aggregation,
                    )
                    if out:
                        _save_output(pca_results, out, "pca_results")
                except Exception as e:
                    click.echo(f"Error performing Principal Component Analysis: {e}")

            if (regression or ml) and target_col:
                click.echo("\n=== Regression Analysis ===")
                click.echo(
                    """
                           Regression Analysis (Linear or Logistic Regression).
                           Automatically detects binary outcomes for logistic regression.
                           Otherwise uses linear regression for continuous outcomes.
                Hint:   Use --outcome to specify the target variable for regression.
                        Use --include to specify columns to include in the analysis (comma separated).
                """
                )
                try:
                    regression_results = ml_analyzer.get_regression(
                        y=target_col, linkage_method=linkage, aggregation=aggregation
                    )
                    if out:
                        _save_output(regression_results, out, "regression_results")
                except Exception as e:
                    click.echo(f"Error performing regression analysis: {e}")

            if lstm or ml:
                click.echo("\n=== LSTM Text Classification ===")
                click.echo(
                    """
                           LSTM (Long Short-Term Memory) model for text-based prediction.
                           Tests if text documents converge towards predicting the outcome variable.
                           Requires both text documents and an 'id' column to align texts with outcome.
                Hint:   Use --outcome to specify the target variable for LSTM prediction.
                        The outcome should be binary (two classes).
                        Ensure documents have IDs matching the 'id' column in your data.
                """
                )
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for LSTM prediction tasks"
                    )
                try:
                    lstm_results = ml_analyzer.get_lstm_predictions(y=target_col)
                    if out:
                        _save_output(lstm_results, out, "lstm_results")
                except Exception as e:
                    click.echo(f"Error performing LSTM prediction: {e}")

        elif (
            nnet or cls or knn or kmeans or cart or pca or regression or lstm or ml
        ) and not ML_AVAILABLE:
            click.echo(
                click.style(
                    "\n⚠️  Machine Learning features are not installed.",
                    fg="yellow",
                    bold=True,
                )
            )
            click.echo(
                click.style("   Install with: ", fg="white")
                + click.style("pip install crisp-t[ml]", fg="cyan", bold=True)
            )

        # Save corpus and csv if output path is specified
        if out and corpus:
            if filters and inp and out and inp == out:
                raise click.ClickException(
                    click.style("❌ Error: ", fg="red", bold=True)
                    + "--out cannot be the same as --inp when using --filters. "
                    + "Please specify a different output folder to avoid overwriting input data."
                )
            if filters and ((not inp) or (not out)):
                raise click.ClickException(
                    format_error(
                        "Both --inp and --out must be specified when using --filters."
                    )
                )
            output_path = pathlib.Path(out)
            # Allow both directory and a file path '.../corpus.json'
            if output_path.suffix:
                # Ensure parent exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_base = output_path
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                save_base = output_path / "corpus.json"
            read_data.write_corpus_to_json(str(save_base), corpus=corpus)
            click.echo(
                format_success(
                    f"Corpus saved to: {click.style(str(save_base), fg='cyan')}"
                )
            )

        if print_args and corpus:
            print_section_header("CORPUS DETAILS", emoji="📊", color="blue")
            # Join the print arguments into a single string
            print_command = " ".join(print_args) if print_args else None
            if print_command:
                click.echo(corpus.pretty_print(show=print_command))

        click.echo(click.style("\n" + "=" * 60, fg="green", bold=True))
        click.echo(format_success("Analysis Complete!"))
        click.echo(click.style("=" * 60 + "\n", fg="green", bold=True))

    except click.ClickException:
        # Let Click handle and set non-zero exit code
        raise
    except Exception as e:
        # Convert unexpected exceptions to ClickException for non-zero exit code
        if verbose:
            import traceback

            traceback.print_exc()
        raise click.ClickException(str(e)) from e


def _save_output(data, base_path: str, suffix: str):
    """Helper function to save analysis output to files."""
    try:
        import json

        import pandas as pd

        output_path = pathlib.Path(base_path)
        if output_path.suffix:
            # Use provided extension
            save_path = output_path / f"{output_path.stem}_{suffix}{output_path.suffix}"
        else:
            # Default to JSON
            save_path = output_path / f"{output_path.stem}_{suffix}.json"

        if isinstance(data, pd.DataFrame):
            if save_path.suffix == ".csv":
                data.to_csv(save_path, index=False)
            else:
                data.to_json(save_path, orient="records", indent=2)
        elif isinstance(data, (dict, list)):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            # For other types, try JSON serialization first, fall back to string
            with open(save_path, "w", encoding="utf-8") as f:
                try:
                    json.dump(data, f, indent=2, default=str)
                except (TypeError, ValueError):
                    # If JSON fails, write as string
                    f.write(str(data))

        click.echo(
            format_success(
                f"Results saved to: {click.style(str(save_path), fg='cyan')}", indent=3
            )
        )

    except Exception as e:
        click.echo(
            click.style(
                f"⚠️  Warning: Could not save output to {base_path}_{suffix}: ",
                fg="yellow",
            )
            + str(e)
        )


def _process_csv(csv_analyzer, unstructured, ignore, filters):
    text_columns = ",".join(unstructured) if unstructured else ""
    ignore_columns = ignore if ignore else ""
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


if __name__ == "__main__":
    main()
