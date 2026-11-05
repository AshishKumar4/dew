"""A command's processes owned as one group (`start`, `stop`), as src/dew/cleanup.py owns them: started
as the leader of a session and process group of its own with a lifeline that kills the group once its
owner is gone however it went, stopped with SIGKILL to the whole group. Dew's tests
(tests/process_support.py) and CI (tools/armada/ci.py) own their children through it. dew-ops keeps a
reaper of its own beside it (dewops/process.py), which also follows descendants that leave the group,
as a subreaper, on hosts without a Dew checkout.

CI must not `import dew`, which sets the process's XLA flags for every test process it starts, so this
file loads src/dew/cleanup.py (the standard library alone) by its path, under the name `dew.cleanup`; where
the process already imported it, under that name, that module is the one used. A process therefore has
one module and one lifeline table, whichever loaded it first.
"""

import importlib.util
import sys
from pathlib import Path


def _cleanup():
    """The module `dew.cleanup`: the process's own where it is loaded, else src/dew/cleanup.py's file."""
    if "dew.cleanup" in sys.modules:
        return sys.modules["dew.cleanup"]
    path = Path(__file__).resolve().parents[2] / "src" / "dew" / "cleanup.py"
    spec = importlib.util.spec_from_file_location("dew.cleanup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_owner = _cleanup()
start, stop = _owner.start, _owner.stop
