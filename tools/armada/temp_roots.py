"""A temp root of one pytest process's own (its PYTEST_DEBUG_TEMPROOT, row 8298): pytest's default,
/tmp/pytest-of-<user>, is one per node, and a process's exit, pruning it, resolves the `pytest-current` link
another process is replacing (a GPU lane runs several at once). Outside the modules every test file's key
covers (ci.py's SHARED), so the root changes no row's key."""

import contextlib
import shutil
import tempfile
from collections.abc import Iterator


@contextlib.contextmanager
def temp_root() -> Iterator[str]:
    """A fresh directory for one pytest process's temp root. Its name is short, and its numbered directory is
    always pytest-0, so a test's Unix socket under tmp_path stays within sun_path's 108 bytes. Removed once the
    process has returned; a supervisor interrupted mid-run leaves it, under a process its lifeline has yet to
    stop."""
    root = tempfile.mkdtemp(prefix="p")
    yield root
    shutil.rmtree(root, ignore_errors=True)
