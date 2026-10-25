# A guest's reach into another context, which benchmark_gateway.py's isolated() calls with that context's id.
# Each attempt below must be refused: the AssertionError after it runs only if it went through.
import contextlib
import socket
from pathlib import Path


def reach(foreign):
    for path in [f'/sessions/connections/kernel-{foreign}.json', f'/sessions/ipc/kernel-{foreign}/kernel-1']:
        with contextlib.suppress(PermissionError):
            Path(path).read_bytes()
            raise AssertionError('guest read another context')
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with contextlib.suppress(PermissionError):
            sock.connect(f'/sessions/ipc/kernel-{foreign}/kernel-1')
            raise AssertionError('guest connected to another context')
    finally:
        sock.close()
