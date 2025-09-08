"""Simple logging setup helper.

Usage:
    from scripts.logging_setup import setup_logging
    setup_logging(verbose=True)  # or False
"""
from __future__ import annotations

import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


__all__ = ["setup_logging"]

