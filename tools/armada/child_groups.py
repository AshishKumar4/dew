"""A command's processes owned as one group (`start`, `stop`), as src/dew/_cleanup.py owns them: started
as the leader of a session and process group of its own with a lifeline that kills the group once its
owner is gone however it went, stopped with SIGKILL to the whole group. Dew's tests
(tests/process_support.py) and CI (tools/armada/ci.py) own their children through it. dew-ops keeps a
reaper of its own beside it (dewops/process.py), which also follows descendants that leave the group,
as a subreaper, on hosts without a Dew checkout.

CI must not `import dew`, which sets the process's XLA flags for every test process it starts, so this
file loads src/dew/_cleanup.py (the standard library alone) by its path, under the name `dew._cleanup`; where
the process already imported it, under that name, that module is the one used. A process therefore has
one module and one lifeline table, whichever loaded it first.
"""

import importlib.util
import sys
import threading
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "dew" / "_cleanup.py"
"""The file of `dew._cleanup`, which a process that has not imported it loads by its path (`_cleanup`)."""
_LOADING = threading.Lock()
"""`_cleanup`'s, as CI's threads start their first groups at once (ci.py's gpu-lane): one loads the module,
the rest wait for it whole."""


def _cleanup():
    """The module `dew._cleanup`: the process's own where it is loaded, else `SOURCE`'s, published once it
    has run, so no thread reads it half made."""
    with _LOADING:
        if "dew._cleanup" in sys.modules:
            return sys.modules["dew._cleanup"]
        spec = importlib.util.spec_from_file_location("dew._cleanup", SOURCE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
        return module


def start(command, **options):
    """`dew._cleanup.start`, the module loaded at the first group's start, so a process that starts none (CI's
    planner, in a checkout of tools and tests alone) never reads src/."""
    return _cleanup().start(command, **options)


def stop(process, **options):
    """`dew._cleanup.stop`."""
    return _cleanup().stop(process, **options)
