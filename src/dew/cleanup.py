"""Running every step of a teardown, whichever fail, and raising one error that names them all."""

from collections.abc import Callable, Sequence
from typing import Any


def attempted(steps: Sequence[tuple[str, Callable[[], Any]]], *,
              primary: BaseException | None = None) -> None:
    """Run every step, even after one fails, its result unread. Each failure is noted on `primary`,
    the error its caller is already raising, which stays the one raised; without one the first
    failure is raised with the later ones noted on it."""
    error = primary
    for label, step in steps:
        try:
            step()
        except BaseException as failure:
            if error is None:
                error = failure
            else:
                error.add_note(f"{label} failed: {failure!r}")
    if primary is None and error is not None:
        raise error
