"""A process of its own (`dew._isolated.Isolated`), where Dew runs what must not share its caller's
process: its replies come over pipes of their own."""

import os
import sys

import pytest

from dew._isolated import Isolated, IsolatedFailure


def noisy(setting: str) -> dict:
    """A reply, its process printing around it as a C library's debug log does: Python's stdout and the
    process's file descriptor 1 directly, before and after the value is made."""
    print("NCCL INFO before", flush=True)
    os.write(1, b"node:1:2 [0] NCCL INFO a native line\n")
    reply = {"setting": os.environ.get(setting), "gpus": [0, 1]}
    print("{\"not\": \"the reply\"}", flush=True)
    sys.stderr.write("a warning\n")
    return reply


def failing() -> None:
    raise ValueError("refused here")


def test_a_reply_reaches_its_caller_whatever_its_process_prints_and_its_settings_apply():
    """A function run in the process (`Isolated.call`) replies with its value however much the process
    prints to its stdout, Python's or a native library's on its file descriptor, around it; the process's
    environment holds the caller's settings (`environment`); a raised error comes back as one. A child's
    reply read from its stdout's last line was refused for any line printed after it: NCCL_DEBUG=INFO
    ended a 2xH100 start in `dew.communicators` (row 6497)."""
    with Isolated({"DEW_ISOLATED_PROBE": "set"}) as process:
        assert process.call(noisy, "DEW_ISOLATED_PROBE") == {"setting": "set", "gpus": [0, 1]}
        with pytest.raises(IsolatedFailure):
            process.call(failing)
        assert process.call(noisy, "DEW_ISOLATED_PROBE")["setting"] == "set"
