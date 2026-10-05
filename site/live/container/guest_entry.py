"""The kernel process enters its limits before IPython can run a cell."""

import os
import sys

from guest_limits import install

if os.environ.get("DEW_GUEST_TRACE") == "1":
    print("GUEST_ENTRY started",flush=True)
install()
if os.environ.get("DEW_GUEST_TRACE") == "1":
    print("GUEST_ENTRY limits installed",flush=True)
if os.environ.get("DEW_GUEST_TRACE") == "1":
    import faulthandler
    faulthandler.dump_traceback_later(2, repeat=True)
sys.argv[0] = "ipykernel_launcher"
from ipykernel.kernelapp import launch_new_instance  # noqa: E402 - install policy first

launch_new_instance()
