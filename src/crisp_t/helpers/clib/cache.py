"""Cache management utilities."""

import shutil
from pathlib import Path

import click


def clear_cache():
    """Delete cache folder if it exists."""
    cache_dir = Path("cache")
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        click.echo(click.style("✓ Cache cleared successfully", fg="green"))
    else:
        click.echo(click.style("ℹ️  No cache found to clear", fg="blue"))
