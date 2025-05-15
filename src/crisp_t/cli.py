import click
from . import __version__


def main():
    click.echo("_________________________________________")
    click.echo("QRMine(TM) Qualitative Research Miner. v" + __version__)
    pass


if __name__ == "__main__":
    main()
