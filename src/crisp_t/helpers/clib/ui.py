"""UI utilities for CLI commands."""
import click


def print_section_header(title: str, emoji: str = "", color: str = "blue", bold: bool = True):
    """
    Print a styled section header with a box border.

    Args:
        title: The title text to display
        emoji: Optional emoji to prefix the title
        color: Click color name (blue, green, yellow, red, cyan, magenta)
        bold: Whether to make the text bold
    """
    # Calculate padding to center title
    box_width = 41
    title_with_emoji = f"{emoji}  {title}" if emoji else title
    padding_left = (box_width - len(title_with_emoji) - 4) // 2
    padding_right = box_width - len(title_with_emoji) - 4 - padding_left

    click.echo(click.style("\n╔═══════════════════════════════════════╗", fg=color, bold=bold))
    click.echo(click.style(f"║{' ' * padding_left}  {title_with_emoji}  {' ' * padding_right}║", fg=color, bold=bold))
    click.echo(click.style("╚═══════════════════════════════════════╝", fg=color, bold=bold))


def print_tips(tips: dict[str, str]):
    """
    Print formatted tips with styled parameter names.

    Args:
        tips: Dictionary mapping parameter names to descriptions
    """
    if not tips:
        return

    click.echo(click.style("\n💡 Tips:", fg="cyan", bold=True))
    for param, description in tips.items():
        click.echo(f"Use {click.style(param, fg='green')} {description}")


def format_success(message: str) -> str:
    """Format a success message with checkmark."""
    return click.style(f"✓ {message}", fg="green")


def format_error(message: str) -> str:
    """Format an error message with X mark."""
    return click.style(f"❌ {message}", fg="red", bold=True)


def format_info(message: str) -> str:
    """Format an info message with info icon."""
    return click.style(f"ℹ️  {message}", fg="blue")


def format_warning(message: str) -> str:
    """Format a warning message."""
    return click.style(f"⚠️  {message}", fg="yellow")
