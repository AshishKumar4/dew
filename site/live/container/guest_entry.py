"""The kernel process enters its limits before IPython can run a cell."""

import sys

from guest_limits import install

install()
sys.argv[0] = "ipykernel_launcher"
from ipykernel.kernelapp import launch_new_instance  # noqa: E402 - install policy first

launch_new_instance()
