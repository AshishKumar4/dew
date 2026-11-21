"""The truths store's switch, apart from its machinery (tests/truths_ahead.py, tests/arm_store.py): every
check-1 site reads whether a store is open, and only a GPU tier part's prepare or run opens one, so the CPU
lane's tests load none of the machinery, whose churn would re-key every one of their rows (the test cache
keys a row on each module its test loads)."""

import contextlib
import contextvars
from collections.abc import Iterator

COMPILED: contextvars.ContextVar = contextvars.ContextVar("truths_ahead_store", default=None)
"""The store open (`truths_ahead.compiled_truths`, a `truths_ahead.CompiledTruths`), None outside one."""

SITE_IN_PROGRESS: contextvars.ContextVar[str | None] = contextvars.ContextVar("truths_ahead_site",
                                                                              default=None)
"""The label of the check-1 site whose truths are computed (`judged`), None outside one."""


@contextlib.contextmanager
def judged(label: str) -> Iterator[None]:
    """Within the block, site `label`'s truths are computed (`truths_ahead.running`)."""
    token = SITE_IN_PROGRESS.set(label)
    try:
        yield
    finally:
        SITE_IN_PROGRESS.reset(token)


def running():
    """The open store computing the site in progress's truths, and its label (`truths_ahead.running`); None
    where no store is open, its machinery not loaded."""
    if COMPILED.get() is None:
        return None
    import truths_ahead

    return truths_ahead.running()


def reading():
    """The open store a run reads the site in progress from, and its label (`truths_ahead.reading`); None
    where no store is open, its machinery not loaded."""
    if COMPILED.get() is None:
        return None
    import truths_ahead

    return truths_ahead.reading()
