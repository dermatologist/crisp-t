import logging
import pathlib
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import click

from . import __version__
from .cluster import Cluster
from .csv import Csv
from .read_data import ReadData
from .sentiment import Sentiment
from .text import Text
from .visualize import QRVisualize

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
@click.option("--verbose", "-v", is_flag=True, help="Print verbose messages.")
@click.option(
    "--covid", "-cf", default="", help="Download COVID narratives from the website"
)
@click.option("--inp", "-i", help="Load corpus from a folder containing corpus.json")
@click.option("--out", "-o", default="", help="Write corpus to a folder as corpus.json")
@click.option("--csv", default="", help="CSV file name")
@click.option(
    "--num", "-n", default=3, help="N (clusters/epochs, etc, depending on context)"
)
@click.option("--rec", "-r", default=3, help="Record or top_n (based on context)")
@click.option(
    "--unstructured",
    "-t",
    multiple=True,
    help="Csv columns with text data that needs to be treated as text. (Ex. Free text comments)",
)
@click.option(
    "--filters",
    "-f",
    multiple=True,
    help="Filters to apply as key=value (can be used multiple times)",
)
@click.option("--codedict", is_flag=True, help="Generate coding dictionary")
@click.option("--topics", is_flag=True, help="Generate topic model")
@click.option("--assign", is_flag=True, help="Assign documents to topics")
@click.option(
    "--cat", is_flag=True, help="List categories of entire corpus or individual docs"
)
@click.option(
    "--summary",
    is_flag=True,
    help="Generate summary for entire corpus or individual docs",
)
@click.option(
    "--sentiment",
    is_flag=True,
    help="Generate sentiment score for entire corpus or individual docs",
)
@click.option(
    "--sentence",
    is_flag=True,
    default=False,
    help="Generate sentence-level scores when applicable",
)
@click.option("--nlp", is_flag=True, help="Generate all NLP reports")
@click.option("--ml", is_flag=True, help="Generate all ML reports")
@click.option("--nnet", is_flag=True, help="Display accuracy of a neural network model")
@click.option(
    "--cls", is_flag=True, help="Display confusion matrix from classifiers (SVM, Decision Tree)"
)
@click.option("--knn", is_flag=True, help="Display nearest neighbours")
@click.option("--kmeans", is_flag=True, help="Display KMeans clusters")
@click.option("--cart", is_flag=True, help="Display Association Rules")
@click.option("--pca", is_flag=True, help="Display PCA")
@click.option("--visualize", is_flag=True, help="Visualize words, topics or wordcloud")
@click.option("--ignore", default="", help="Comma separated ignore words or columns depending on context")
@click.option("--include", default="", help="Comma separated columns to include from csv")
@click.option("--outcome", default="", help="Outcome variable for ML tasks")
@click.option("--source", "-s", help="Source URL or directory path to read data from")
@click.option("--print", "-p", is_flag=True, help="Pretty print the corpus to console")
@click.option(
    "--sources",
    multiple=True,
    help="Multiple sources (URLs or directories) to read data from; can be used multiple times",
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
    ml,
    visualize,
    ignore,
    include,
    outcome,
    source,
    sources,
    print,
):
    """CRISP-T: Cross Industry Standard Process for Triangulation.

    A comprehensive framework for analyzing textual and numerical data using
    advanced NLP, machine learning, and statistical techniques.
    """

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        click.echo("Verbose mode enabled")

    click.echo("_________________________________________")
    click.echo("CRISP-T: Qualitative Research Analysis Framework")
    click.echo(f"Version: {__version__}")
    click.echo("_________________________________________")

    # Initialize components
    read_data = ReadData()
    corpus = None
    text_analyzer = None
    csv_analyzer = None
    ml_analyzer = None

    try:
        # Handle COVID data download
        if covid:
            if not source:
                raise click.ClickException("--source (output folder) is required when using --covid.")
            click.echo(f"Downloading COVID narratives from: {covid} to {source}")
            try:
                from .utils import QRUtils
                QRUtils.read_covid_narratives(source, covid)
                click.echo(f"✓ COVID narratives downloaded to {source}")
            except Exception as e:
                raise click.ClickException(f"COVID download failed: {e}")

        # Load corpus from input file if provided
        if inp:
            click.echo(f"Loading corpus from: {inp}")
            corpus = read_data.read_corpus_from_json(inp, comma_separated_ignore_words=ignore if ignore else "")
            text_analyzer = Text(corpus=corpus)
            click.echo(f"Loaded corpus with {text_analyzer.document_count()} documents")
            # Apply filters if provided
            if filters:
                try:
                    for flt in filters:
                        if "=" not in flt:
                            raise ValueError("Filter must be in key=value format")
                        key, value = flt.split("=", 1)
                        text_analyzer.filter_documents(key.strip(), value.strip())
                    click.echo(
                        f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
                    )
                except Exception as e:
                    # Surface as CLI error with non-zero exit code
                    raise click.ClickException(str(e))

        # Handle source option (URL or directory)
        if source:
            click.echo(f"Reading data from source: {source}")
            try:
                read_data.read_source(
                    source, comma_separated_ignore_words=ignore if ignore else None
                )
                corpus = read_data.create_corpus(
                    name=f"Corpus from {source}",
                    description=f"Data loaded from {source}",
                )
                click.echo(
                    f"✓ Successfully loaded {len(corpus.documents)} document(s) from {source}"
                )
                # Apply filters if provided
                if filters:
                    try:
                        text_analyzer = Text(corpus=corpus)
                        for flt in filters:
                            if "=" not in flt:
                                raise ValueError("Filter must be in key=value format")
                            key, value = flt.split("=", 1)
                            text_analyzer.filter_documents(key.strip(), value.strip())
                        click.echo(
                            f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
                        )
                    except Exception as e:
                        # Surface as CLI error with non-zero exit code
                        raise click.ClickException(str(e))

            except click.ClickException as e:
                logger.error(f"Failed to read source {source}: {e}")
                raise
            except Exception as e:
                click.echo(f"✗ Error reading from source: {e}", err=True)
                logger.error(f"Failed to read source {source}: {e}")
                return

        # Handle multiple sources
        if sources:
            loaded_any = False
            for src in sources:
                click.echo(f"Reading data from source: {src}")
                try:
                    read_data.read_source(
                        src, comma_separated_ignore_words=ignore if ignore else None
                    )
                    loaded_any = True
                except Exception as e:
                    logger.error(f"Failed to read source {src}: {e}")
                    raise click.ClickException(str(e))

            if loaded_any:
                corpus = read_data.create_corpus(
                    name="Corpus from multiple sources",
                    description=f"Data loaded from {len(sources)} sources",
                )
                click.echo(
                    f"✓ Successfully loaded {len(corpus.documents)} document(s) from {len(sources)} sources"
                )
                # Apply filters if provided
                if filters:
                    try:
                        text_analyzer = Text(corpus=corpus)
                        for flt in filters:
                            if "=" not in flt:
                                raise ValueError("Filter must be in key=value format")
                            key, value = flt.split("=", 1)
                            text_analyzer.filter_documents(key.strip(), value.strip())
                        click.echo(
                            f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
                        )
                    except Exception as e:
                        raise click.ClickException(str(e))

        # Load csv from corpus.df if available
        if corpus and corpus.df is not None:
            click.echo("Loading CSV data from corpus.df")
            csv_analyzer = Csv(corpus=corpus)
            csv_analyzer.df = corpus.df
            text_columns = ",".join(unstructured) if unstructured else ""
            ignore_columns = ignore if ignore else ""

            csv_analyzer.comma_separated_text_columns = text_columns
            csv_analyzer.comma_separated_ignore_columns = ignore_columns
            click.echo(f"Loaded CSV with shape: {csv_analyzer.get_shape()}")
            if verbose:
                click.echo(f"Columns: {csv_analyzer.get_columns()}")

        # Load CSV data
        if csv:
            click.echo(f"Loading CSV data from: {csv}")
            csv_path = pathlib.Path(csv)
            if csv_path.exists():
                csv_analyzer = Csv()
                text_columns = ",".join(unstructured) if unstructured else ""
                ignore_columns = ignore if ignore else ""

                csv_analyzer.comma_separated_text_columns = text_columns
                csv_analyzer.comma_separated_ignore_columns = ignore_columns
                csv_analyzer.read_csv(str(csv_path))

                click.echo(f"Loaded CSV with shape: {csv_analyzer.get_shape()}")
                if verbose:
                    click.echo(f"Columns: {csv_analyzer.get_columns()}")

                # Add df to corpus if available
                if corpus:
                    corpus.df = csv_analyzer.df

                # Create corpus from CSV text columns if specified
                if text_columns and not corpus:
                    df = csv_analyzer.df
                    text_cols = [
                        col.strip() for col in text_columns.split(",") if col.strip()
                    ]

                    # Add documents to read_data
                    from .model import Document

                    for idx, row in df.iterrows():
                        combined_text = " ".join(
                            [str(row[col]) for col in text_cols if col in df.columns]
                        )
                        if combined_text.strip() and combined_text.lower() != "nan":
                            metadata = {
                                col: row[col]
                                for col in df.columns
                                if col not in text_cols
                            }
                            document = Document(
                                text=combined_text,
                                metadata=metadata,
                                id=str(idx),
                                score=0.0,
                                name=f"doc_{idx}",
                                description=f"Document from CSV row {idx}",
                            )
                            read_data._documents.append(document)

                    if read_data._documents:
                        corpus = read_data.create_corpus(
                            name="CSV Corpus", description=f"Loaded from {csv}"
                        )
                        text_analyzer = Text(corpus=corpus)
                        click.echo(
                            f"Created corpus from CSV with {text_analyzer.document_count()} documents"
                        )
                        # Apply filters if provided
                        if filters:
                            try:
                                for flt in filters:
                                    if "=" not in flt:
                                        raise ValueError(
                                            "Filter must be in key=value format"
                                        )
                                    key, value = flt.split("=", 1)
                                    text_analyzer.filter_documents(
                                        key.strip(), value.strip()
                                    )
                                click.echo(
                                    f"Applied filters {list(filters)}; remaining documents: {text_analyzer.document_count()}"
                                )
                            except Exception as e:
                                # Surface as CLI error with non-zero exit code
                                raise click.ClickException(str(e))
                    else:
                        click.echo("No valid text data found in specified columns")

            else:
                click.echo(f"CSV file not found: {csv}")
                return

        # Initialize ML analyzer if available and ML functions are requested
        if ML_AVAILABLE and (nnet or cls or knn or kmeans or cart or pca or ml) and csv_analyzer:
            if include:
                csv_analyzer.comma_separated_include_columns(include)
            ml_analyzer = ML(csv=csv_analyzer)
        else:
            if (nnet or cls or knn or kmeans or cart or pca or ml) and not ML_AVAILABLE:
                click.echo("Machine learning features require additional dependencies.")
                click.echo("Install with: pip install crisp-t[ml]")
            if (nnet or cls or knn or kmeans or cart or pca or ml) and not csv_analyzer:
                click.echo("ML analysis requires CSV data. Use --csv to provide a data file.")

        # Ensure we have data to work with
        if not corpus and not csv_analyzer:
            click.echo(
                "No input data provided. Use --inp for text files or --csv for data files."
            )
            return

        # Text Analysis Operations
        if text_analyzer:
            if nlp or codedict:
                click.echo("\n=== Generating Coding Dictionary ===")
                click.echo("""
                Coding Dictionary Format:
                - CATEGORY: Common verbs representing main actions or themes.
                - PROPERTY: Common nouns associated with each CATEGORY.
                - DIMENSION: Common adjectives, adverbs, or verbs associated with each PROPERTY.

                Hint:   Use --ignore with a comma-separated list of words to exclude common but uninformative words.
                        Use --filters to narrow down documents based on metadata.
                        Use --num to adjust the number of categories displayed.
                        Use --rec to adjust the number of top items displayed per section.
                """)
                try:
                    text_analyzer.make_spacy_doc()
                    coding_dict = text_analyzer.print_coding_dictionary(num=num, top_n=rec)
                    if out:
                        _save_output(coding_dict, out, "coding_dictionary")
                except Exception as e:
                    click.echo(f"Error generating coding dictionary: {e}")

            if nlp or topics:
                click.echo("\n=== Topic Modeling ===")
                click.echo("""
                Topic Modeling Output Format:
                Each topic is represented as a list of words with associated weights indicating their importance within the topic.
                Example:
                Topic 0: 0.116*"category" + 0.093*"comparison" + 0.070*"incident" + ...
                Hint:   Use --num to adjust the number of topics generated.
                        Use --filters to narrow down documents based on metadata.
                        Use --rec to adjust the number of words displayed per topic.
                """)
                try:
                    cluster_analyzer = Cluster(corpus=corpus)
                    cluster_analyzer.build_lda_model(topics=num)
                    topics_result = cluster_analyzer.print_topics(
                        num_words=rec
                    )
                    click.echo(f"Generated {len(topics_result)} topics as above with the weights in brackets.")
                    if out:
                        _save_output(topics_result, out, "topics")
                except Exception as e:
                    click.error(f"Error generating topics: {e}")

            if nlp or assign:
                click.echo("\n=== Document-Topic Assignments ===")
                click.echo(
                    """
                Document-Topic Assignment Format:
                Each document is assigned to the topic it is most associated with, along with the contribution percentage.
                Hint: --visualize adds a DataFrame to corpus.visualization["assign_topics"] for visualization.
                """
                )
                try:
                    if "cluster_analyzer" not in locals():
                        cluster_analyzer = Cluster(corpus=corpus)
                        cluster_analyzer.build_lda_model(topics=num)
                    assignments = cluster_analyzer.format_topics_sentences(
                        visualize=visualize
                    )
                    click.echo(f"Assigned {len(assignments)} documents to topics")
                    if out:
                        _save_output(assignments, out, "topic_assignments")
                except Exception as e:
                    click.error(f"Error assigning topics: {e}")

            if nlp or cat:
                click.echo("\n=== Category Analysis ===")
                click.echo("""
                Category Analysis Output Format:
                           A list of common concepts or themes in "bag_of_terms" with corresponding weights.
                Hint:   Use --num to adjust the number of categories displayed.
                        Use --filters to narrow down documents based on metadata.
                """)
                try:
                    text_analyzer.make_spacy_doc()
                    categories = text_analyzer.print_categories(num=num)
                    if out:
                        _save_output(categories, out, "categories")
                except Exception as e:
                    click.error(f"Error generating categories: {e}")

            if nlp or summary:
                click.echo("\n=== Text Summarization ===")
                click.echo("""
                Text Summarization Output Format: A list of important sentences representing the main points of the text.
                Hint:   Use --num to adjust the number of sentences in the summary.
                        Use --filters to narrow down documents based on metadata.
                """)
                try:
                    text_analyzer.make_spacy_doc()
                    summary_result = text_analyzer.generate_summary(weight=num)
                    click.echo(summary_result)
                    if out:
                        _save_output(summary_result, out, "summary")
                except Exception as e:
                    click.error(f"Error generating summary: {e}")

            if nlp or sentiment:
                click.echo("\n=== Sentiment Analysis ===")
                click.echo("""
                Sentiment Analysis Output Format:
                           neg, neu, pos, compound scores.
                Hint:   Use --filters to narrow down documents based on metadata.
                        Use --sentence to get document-level sentiment scores.
                """)
                try:
                    sentiment_analyzer = Sentiment(corpus=corpus)
                    sentiment_results = sentiment_analyzer.get_sentiment(
                        documents=sentence, verbose=verbose
                    )
                    click.echo(sentiment_results)
                    if out:
                        _save_output(sentiment_results, out, "sentiment")
                except Exception as e:
                    click.error(f"Error generating sentiment analysis: {e}")

        # Machine Learning Operations
        if ml_analyzer and ML_AVAILABLE:
            target_col = outcome

            if kmeans or ml:
                click.echo("\n=== K-Means Clustering ===")
                click.echo("""
                           K-Means clustering removes non-numeric columns.
                           Additionally it removes NaN values.
                           So combining with other ML options may not work as expected.
                Hint:   Use --num to adjust the number of clusters generated.
                """)
                csv_analyzer.retain_numeric_columns_only()
                csv_analyzer.drop_na()
                _ml_analyzer = ML(csv=csv_analyzer)
                clusters, members = _ml_analyzer.get_kmeans(
                    number_of_clusters=num, verbose=verbose
                )
                _ml_analyzer.profile(members, number_of_clusters=num)
                if out:
                    _save_output(
                        {"clusters": clusters, "members": members}, out, "kmeans"
                    )

            if (cls or ml) and target_col:
                click.echo("\n=== Classifier Evaluation ===")
                click.echo("""
                           Classifier
                            - SVM: Support Vector Machine classifier with confusion matrix output.
                            - Decision Tree: Decision Tree classifier with feature importance output.
                Hint:   Use --outcome to specify the target variable for classification.
                        Use --rec to adjust the number of top important features displayed.
                        Use --include to specify columns to include in the analysis (comma separated).
                """)
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for classification tasks"
                    )
                click.echo("\n=== SVM ===")
                try:
                    confusion_matrix = ml_analyzer.svm_confusion_matrix(
                        y=target_col, test_size=0.25
                    )
                    click.echo(ml_analyzer.format_confusion_matrix_to_human_readable(confusion_matrix))
                    if out:
                        _save_output(confusion_matrix, out, "svm_results")
                except Exception as e:
                    click.error(f"Error performing SVM classification: {e}")
                click.echo("\n=== Decision Tree Classification ===")
                try:
                    cm, importance = ml_analyzer.get_decision_tree_classes(y=target_col, top_n=rec)
                    click.echo("\n=== Feature Importance ===")
                    click.echo(
                        ml_analyzer.format_confusion_matrix_to_human_readable(
                            cm
                        )
                    )
                    if out:
                        _save_output(cm, out, "decision_tree_results")
                except Exception as e:
                    click.error(f"Error performing Decision Tree classification: {e}")

            if (nnet or ml) and target_col:
                click.echo("\n=== Neural Network Classification Accuracy ===")
                click.echo("""
                            Neural Network classifier with accuracy output.
                Hint:   Use --outcome to specify the target variable for classification.
                        Use --include to specify columns to include in the analysis (comma separated).
                """)
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for neural network tasks"
                    )
                try:
                    predictions = ml_analyzer.get_nnet_predictions(y=target_col)
                    if out:
                        _save_output(predictions, out, "nnet_results")
                except Exception as e:
                    click.error(f"Error performing Neural Network classification: {e}")

            if (knn or ml) and target_col:
                click.echo("\n=== K-Nearest Neighbors ===")
                click.echo("""
                           K-Nearest Neighbors search results.
                Hint:   Use --outcome to specify the target variable for KNN search.
                        Use --rec to specify the record number to search from (1-based index).
                        Use --num to specify the number of nearest neighbors to retrieve.
                        Use --include to specify columns to include in the analysis (comma separated).
                """)
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for KNN search tasks"
                    )
                if rec < 1:
                    raise click.ClickException("--rec must be a positive integer (1-based index)")
                try:
                    knn_results = ml_analyzer.knn_search(y=target_col, n=num, r=rec)
                    if out:
                        _save_output(knn_results, out, "knn_results")
                except Exception as e:
                    click.error(f"Error performing K-Nearest Neighbors search: {e}")

            if (cart or ml) and target_col:
                click.echo("\n=== Association Rules (CART) ===")
                click.echo("""
                           Association Rules using the Apriori algorithm.
                Hint:   Use --outcome to specify the target variable to remove from features.
                        Use --num to specify the minimum support (between 1 and 99).
                        Use --rec to specify the minimum threshold for the rules (between 1 and 99).
                        Use --include to specify columns to include in the analysis (comma separated).
                """)
                if not target_col:
                    raise click.ClickException(
                        "--outcome is required for association rules tasks"
                    )
                if not (1 <= num <= 99):
                    raise click.ClickException("--num must be between 1 and 99 for min_support")
                if not (1 <= rec <= 99):
                    raise click.ClickException("--rec must be between 1 and 99 for min_threshold")
                _min_support = float(num / 100)
                _min_threshold = float(rec / 100)
                click.echo(f"Using min_support={_min_support:.2f} and min_threshold={_min_threshold:.2f}")
                try:
                    apriori_results = ml_analyzer.get_apriori(
                        y=target_col, min_support=_min_support, min_threshold=_min_threshold
                    )
                    click.echo(apriori_results)
                    if out:
                        _save_output(apriori_results, out, "association_rules")
                except Exception as e:
                    click.error(f"Error generating association rules: {e}")

            if (pca or ml) and target_col:
                click.echo("\n=== Principal Component Analysis ===")
                click.echo("""
                           Principal Component Analysis (PCA) results.
                Hint:   Use --outcome to specify the target variable to remove from features.
                        Use --num to specify the number of principal components to generate.
                        Use --include to specify columns to include in the analysis (comma separated).
                """)
                try:
                    pca_results = ml_analyzer.get_pca(y=target_col, n=num)
                    if out:
                        _save_output(pca_results, out, "pca_results")
                except Exception as e:
                    click.error(f"Error performing Principal Component Analysis: {e}")

        elif (nnet or cls or knn or kmeans or cart or pca or ml) and not ML_AVAILABLE:
            click.echo("Machine learning features require additional dependencies.")
            click.echo("Install with: pip install crisp-t[ml]")

        # Visualization
        if visualize and corpus:
            click.echo("\n=== Generating Visualizations ===")
            viz = QRVisualize()
            # This would generate appropriate visualizations based on the analysis performed
            click.echo("Visualization functionality integrated with analysis results")

        # Save corpus and csv if output directory is specified
        if out and corpus and not filters:
            output_path = pathlib.Path(out)
            output_path.mkdir(parents=True, exist_ok=True)
            read_data.write_corpus_to_json(str(output_path))
            click.echo(f"✓ Corpus and csv saved to {output_path}")

        if print and corpus:
            click.echo("\n=== Corpus Details ===")
            click.echo(corpus.pretty_print())

        click.echo("\n=== Analysis Complete ===")

    except click.ClickException:
        # Let Click handle and set non-zero exit code
        raise
    except Exception as e:
        # Convert unexpected exceptions to ClickException for non-zero exit code
        if verbose:
            import traceback

            traceback.print_exc()
        raise click.ClickException(str(e))


def _save_output(data, base_path: str, suffix: str):
    """Helper function to save analysis output to files."""
    try:
        import json

        import pandas as pd

        output_path = pathlib.Path(base_path)
        if output_path.suffix:
            # Use provided extension
            save_path = (
                output_path / f"{output_path.stem}_{suffix}{output_path.suffix}"
            )
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
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(data))

        click.echo(f"Results saved to: {save_path}")

    except Exception as e:
        click.echo(f"Warning: Could not save output to {base_path}_{suffix}: {str(e)}")


if __name__ == "__main__":
    main()
