import logging
import os

import click

from ..read_data import ReadData

logger = logging.getLogger(__name__)


def initialize_corpus(
    source=None,
    inp=None,
    comma_separated_text_columns="",
    comma_separated_ignore_words="i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,same,so,than,too,very,s,t,can,will,just,don,should,now",
    max_text_files=None,
    max_csv_rows=None,
):
    """Initialize a corpus from source or input file.
    Priority is given to source if both are provided.
    If neither is provided, check for default folders 'crisp_source' or 'crisp_input' in current directory.
    If still none, check environment variables CRISP_T_SOURCE or CRISP_T_INPUT.

    Args:
        source (str, optional): URL or directory to read data from.
        inp (str, optional): Path to input text file or directory.
        comma_separated_text_columns (str, optional): Comma-separated list of unstructured text columns in CSV files to be treated as text documents.
        comma_separated_ignore_words (str, optional): Comma-separated stop words to ignore.
        max_text_files (int, optional): Maximum number of text/PDF files to import when using source.
        max_csv_rows (int, optional): Maximum number of CSV rows to import when using source.
    """
    # Handle source option (URL or directory)
    read_data = ReadData()
    inp = inp or os.getenv("CRISP_T_INPUT") or None
    source = source or os.getenv("CRISP_T_SOURCE") or None

    if inp and source:
        click.echo("Both --source and --inp options provided; prioritizing --source.", err=True)

    if source:
        click.echo(f"Reading data from source: {source}")
        try:
            read_data.read_source(
                source,
                comma_separated_text_columns=comma_separated_text_columns,
                comma_separated_ignore_words=(
                    comma_separated_ignore_words
                    if comma_separated_ignore_words
                    else None
                ),
                max_text_files=max_text_files,
                max_csv_rows=max_csv_rows,
            )
            corpus = read_data.create_corpus(
                name=f"Corpus from {source}",
                description=f"Data loaded from {source}",
            )
            click.echo(
                f"✓ Successfully loaded {len(corpus.documents)} document(s) from {source}"  # type: ignore
            )

        except click.ClickException as e:
            logger.exception(f"Failed to read source {source}: {e}")
            raise
        except Exception as e:
            click.echo(f"✗ Error reading from source: {e}", err=True)
            logger.exception(f"Failed to read source {source}: {e}")
            return
        return corpus

    # Handle inp option (text file or directory). It can be in the home directory or current directory
    if not inp and os.path.exists(os.path.join(os.path.expanduser("~"), "crisp_input")):
        inp = os.path.join(os.path.expanduser("~"), "crisp_input")
    if not inp and os.path.exists(os.path.join(os.getcwd(), "crisp_input")):
        inp = os.path.join(os.getcwd(), "crisp_input")
    if not inp and os.path.exists("crisp_input"):
        inp = "crisp_input"
    # Load corpus from input file if provided
    if inp:
        click.echo(f"Loading corpus from: {inp}")
        corpus = read_data.read_corpus_from_json(
            inp,
            comma_separated_ignore_words=(
                comma_separated_ignore_words if comma_separated_ignore_words else ""
            ),
        )
        if corpus:
            click.echo(
                f"✓ Successfully loaded {len(corpus.documents)} document(s) from {inp}"
            )
            return corpus
        else:
            click.echo(f"✗ No documents found in {inp}", err=True)
            return
    # check if crisp folder exists in the current directory or home directory
    # if not source and os.path.exists(os.path.join(os.path.expanduser("~"), "crisp_source")):
    #     source = os.path.join(os.path.expanduser("~"), "crisp_source")
    # if not source and os.path.exists(os.path.join(os.getcwd(), "crisp_source")):
    #     source = os.path.join(os.getcwd(), "crisp_source")
    # if not source and os.path.exists("crisp_source"):
    #     source = "crisp_source"
    # source = source or os.getenv("CRISP_T_SOURCE")

    click.echo("✗ No input source provided. Please specify --source or --inp option, or set CRISP_T_SOURCE or CRISP_T_INPUT environment variables.", err=True)
    return
