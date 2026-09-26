"""Command-line interface: ``tailsafe --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tailsafe import DISCLAIMER, __version__
from tailsafe.config import Params

app = typer.Typer(
    help="TailSafe — tail-risk evacuation stress-testing for high-rise Hong Kong.",
    no_args_is_help=True,
    add_completion=False,
)
params_app = typer.Typer(help="Inspect the parameter registry (config/params.yaml).")
app.add_typer(params_app, name="params")


@app.command()
def version() -> None:
    """Print the TailSafe version and the responsible-use notice."""
    typer.echo(f"tailsafe {__version__}")
    typer.echo(DISCLAIMER)


@params_app.command("check")
def params_check(
    path: Annotated[Path | None, typer.Option(help="Registry file to check.")] = None,
) -> None:
    """Validate the registry and summarise how many values still need citations."""
    params = Params.load(path)
    leaves = list(params.leaves())
    pending = params.assumption_report()
    typer.echo(f"{params.origin}: {len(leaves)} parameters, all with a source field.")
    typer.echo(f"{len(pending)} are ASSUMPTIONs that still need a citation.")


@params_app.command("list")
def params_list(
    assumptions: Annotated[
        bool, typer.Option("--assumptions", help="Only values that still need a citation.")
    ] = False,
    path: Annotated[Path | None, typer.Option(help="Registry file to read.")] = None,
) -> None:
    """List parameters with their units and sources."""
    params = Params.load(path)
    rows = params.assumption_report() if assumptions else list(params.leaves())
    for p in rows:
        unit = f" [{p.unit}]" if p.unit else ""
        typer.echo(f"{p.path}{unit}  ({p.dist})\n    source: {p.source}")


if __name__ == "__main__":  # pragma: no cover
    app()
