import json
import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

import click

from .model.corpus import Corpus
from .model.document import Document


def _parse_kv(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise click.ClickException(f"Invalid metadata '{value}'. Use key=value format.")
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
        raise click.ClickException("Invalid relationship. Use 'first|second|relation'.")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


@click.command()
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
        "Add a relationship as 'first|second|relation' (e.g., text:term|num:col|correlates)."
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
):
    """
    CRISP-T Corpus CLI: create and manipulate a corpus quickly from the command line.
    """
    logging.basicConfig(level=(logging.DEBUG if verbose else logging.WARNING))
    logger = logging.getLogger(__name__)

    if verbose:
        click.echo("Verbose mode enabled")

    click.echo("_________________________________________")
    click.echo("CRISP-T: Corpus CLI")
    click.echo("_________________________________________")

    # Load corpus from --inp if provided
    corpus = None
    if inp:
        from .helpers.initializer import initialize_corpus

        corpus = initialize_corpus(inp=inp)
        if corpus:
            click.echo(f"✓ Loaded corpus from {inp}")
        else:
            raise click.ClickException(f"Failed to load corpus from {inp}")
    else:
        # Build initial corpus from CLI args
        if not id:
            raise click.ClickException("--id is required when not using --inp.")
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
        click.echo(f"✓ Added {len(docs)} document(s)")

    # Remove documents
    for rid in remove_docs:
        corpus.remove_document_by_id(rid)
    if remove_docs:
        click.echo(f"✓ Removed {len(remove_docs)} document(s)")

    # Update metadata
    for m in metas:
        k, v = _parse_kv(m)
        corpus.update_metadata(k, v)
    if metas:
        click.echo(f"✓ Updated metadata entries: {len(metas)}")

    # Relationships
    for r in relationships:
        first, second, relation = _parse_relationship(r)
        corpus.add_relationship(first, second, relation)
    if relationships:
        click.echo(f"✓ Added {len(relationships)} relationship(s)")
    if clear_rel:
        corpus.clear_relationships()
        click.echo("✓ Cleared relationships")

    # Save corpus to --out if provided
    if out:
        from src.crisp_t.read_data import ReadData

        rd = ReadData(corpus=corpus)
        rd.write_corpus_to_json(out, corpus=corpus)
        click.echo(f"✓ Corpus saved to {out}")

    if print_corpus:
        click.echo("\n=== Corpus Details ===")
        corpus.pretty_print()

    logger.info("Corpus CLI finished")
