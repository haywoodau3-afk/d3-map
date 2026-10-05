"""Canonical d3map command-line entry point."""

from collections.abc import Sequence

from dmap.cli import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv, prog="d3map")
