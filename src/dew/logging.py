"""Dew's stderr diagnostics, with the training display's console when live.

Set levels or handlers on ``logging.getLogger("dew")`` to configure diagnostics.
To route them through an application's root handlers, remove Dew's handlers
and set ``propagate = True``. Dew never changes the root logger.
"""

from __future__ import annotations

import copy
import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from rich.console import Console
from rich.file_proxy import FileProxy
from rich.text import Text
from rich.traceback import Traceback

_lock = threading.RLock()


class _TerminalDiagnostics(logging.Handler):
    """One styled record on the terminal, without inserted wrapping newlines."""

    def __init__(self, console: Console):
        super().__init__()
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        message = record
        if record.exc_info and record.exc_info[0] is not None:
            message = copy.copy(record)
            message.exc_info = message.exc_text = None
        line = Text(datetime.fromtimestamp(record.created).strftime("%H:%M:%S "), style="log.time")
        line.append(record.levelname, style=f"logging.level.{record.levelname.lower()}")
        line.append(" " + self.format(message))
        self.console.print(line, soft_wrap=True)
        if record.exc_info and record.exc_info[0] is not None:
            exception_type, exception, backtrace = record.exc_info
            self.console.print(Traceback.from_exception(exception_type, exception, backtrace))


_handler: _TerminalDiagnostics | logging.StreamHandler | None = None


def configure() -> None:
    """Attach one default handler, leaving a caller's Dew handlers untouched."""
    global _handler
    logger = logging.getLogger("dew")
    with _lock:
        if logger.handlers:
            return
        if sys.stderr.isatty():
            diagnostics = _TerminalDiagnostics(Console(stderr=True, soft_wrap=True))
            diagnostics.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        else:
            diagnostics = logging.StreamHandler()
            diagnostics.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
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
    if isinstance(diagnostics, _TerminalDiagnostics):
        diagnostics.acquire()
        try:
            previous_console, diagnostics.console = diagnostics.console, console
        finally:
            diagnostics.release()
        try:
            yield
        finally:
            diagnostics.acquire()
            try:
                diagnostics.console = previous_console
            finally:
                diagnostics.release()
    else:
        previous_stream = diagnostics.stream
        previous_wrap, console.soft_wrap = console.soft_wrap, True
        diagnostics.setStream(FileProxy(console, previous_stream))
        try:
            yield
        finally:
            diagnostics.setStream(previous_stream)
            console.soft_wrap = previous_wrap
