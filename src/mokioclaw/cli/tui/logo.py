"""ASCII Art Logo for MokioClaw TUI."""

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

LOGO_ASCII = r"""
   __  __      _    _       _____ _
  |  \/  |    | |  (_)     / ____| |
  | \  / | ___| | ___  ___| |    | | ___ __ ___      __
  | |\/| |/ _ \ |/ / |/ _ \ |    | |/ _ \ \ \ /\ / /
  | |  | |  __/   <| | (_) | |____| | (_) |\ V  V /
  |_|  |_|\___|_|\_\_|\___/ \_____|_|\___/  \_/\_/
"""

TAGLINE = "Stage 6 · MultiAgent + Context + Harness"


def build_logo() -> RenderableType:
    """Build a Rich renderable for the MokioClaw logo.

    Returns a ``Panel`` containing the ASCII art with a tagline.
    """
    body = Text()
    body.append("\n")
    body.append("🐾 ", style="bold")
    body.append("MokioClaw", style="bold cyan")
    body.append("\n")
    body.append("━" * 38, style="dim")
    body.append("\n")
    body.append(f"  {TAGLINE}", style="italic yellow")
    body.append("\n")
    body.append("━" * 38, style="dim")

    return Panel(
        body,
        border_style="blue",
        padding=(0, 2),
    )