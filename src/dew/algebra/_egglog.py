"""egglog's first import, on a thread of its own.

egglog's builtin declarations keep the frame of the code that first imports egglog, and
through it every caller's, until each is first used. Imported first on a thread of its own,
they hold that thread's bootstrap frame alone, never the locals of a call that compiles a
process's first program through the pass, whoever first imports `dew.algebra` (a training
step's first trace, a kernel's options). A process that never loads the pass never loads
egglog."""

import importlib
import threading

_importing = threading.Thread(target=importlib.import_module, args=("egglog",), name="dew-egglog-import")
_importing.start()
_importing.join()
