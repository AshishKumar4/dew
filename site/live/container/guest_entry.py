"""The kernel process enters its limits before IPython can run a cell."""

import os
import sys

from guest_limits import install

if os.environ.get("DEW_GUEST_TRACE") == "1":
    print("GUEST_ENTRY started",flush=True)
install(os.environ.get("DEW_GUEST_PROFILE", "cell"))
# The launcher's host PID is outside this private PID namespace; unshare owns the child lifetime.
os.environ.pop("JPY_PARENT_PID", None)
if os.environ.get("DEW_GUEST_TRACE") == "1":
    print("GUEST_ENTRY limits installed",flush=True)
if os.environ.get("DEW_GUEST_TRACE") == "1":
    import faulthandler
    trace = os.fdopen(os.dup(2), 'w')
    faulthandler.dump_traceback_later(2, repeat=True, file=trace)
sys.argv[0] = "ipykernel_launcher"


def launch():
    """IPython's kernel, imported only once the limits above hold, so all it imports runs under them."""
    from ipykernel.kernelapp import launch_new_instance

    launch_new_instance()


try:
    launch()
except BaseException:
    if os.environ.get('DEW_GUEST_TRACE') == '1':
        import traceback
        traceback.print_exc(file=trace)
        trace.flush()
    raise
