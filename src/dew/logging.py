"""Dew's stderr diagnostics, with the training display's console when live.

Set levels or handlers on ``logging.getLogger("dew")`` to configure diagnostics.
To route them through an application's root handlers, remove Dew's handlers
and set ``propagate = True``. Dew never changes the root logger.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler

_lock = threading.RLock()
_handler: RichHandler | None = None


def configure() -> None:
    """Attach one default handler, leaving a caller's Dew handlers untouched."""
    global _handler
    logger = logging.getLogger("dew")
    with _lock:
        if logger.handlers:
            return
        console = Console(stderr=True, force_terminal=None if sys.stderr.isatty() else False)
        diagnostics = RichHandler(console=console,
                                  show_time=True, show_level=True, show_path=False,
                                  log_time_format="%H:%M:%S", markup=False, highlighter=None)
        diagnostics.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logger.addHandler(diagnostics)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.WARNING)
        logger.propagate = False
        _handler = diagnostics


@contextmanager
def display_console(console: Console) -> Iterator[None]:
    """Use the live panel's console only for Dew's own default handler."""
    diagnostics = _handler
    if diagnostics is None or diagnostics not in logging.getLogger("dew").handlers:
        yield
        return
    diagnostics.acquire()
    try:
        previous, diagnostics.console = diagnostics.console, console
    finally:
        diagnostics.release()
    try:
        yield
    finally:
        diagnostics.acquire()
        try:
            diagnostics.console = previous
        finally:
            diagnostics.release()
