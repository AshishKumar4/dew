# What a guest context may not see or do, run in each context by benchmark_gateway.py's isolated().
# Each attempt below must be refused: the AssertionError after it runs only if it went through.
import contextlib
import os
import resource
import socket
from pathlib import Path

for proc in Path('/proc').glob('[0-9]*'):
    try:
        uid = next(line for line in (proc / 'status').read_text().splitlines()
                   if line.startswith('Uid:')).split()[1]
    except (OSError, StopIteration):
        continue
    assert int(uid) == os.getuid(), 'guest saw a foreign process'
links = {line.split(':')[0].strip() for line in Path('/proc/net/dev').read_text().splitlines()[2:]}
assert links == {'lo'}
status = Path('/proc/self/status').read_text().splitlines()
assert all(line.split(':')[1].strip() == '0000000000000000' for line in status if line.startswith('Cap'))
assert next(line for line in status if line.startswith('NoNewPrivs:')).split()[1] == '1'
with contextlib.suppress(OSError):
    socket.create_connection(('127.0.0.1', 8890), timeout=1)
    raise AssertionError('guest reached gateway TCP')
with contextlib.suppress(MemoryError):
    bytearray(1024 * 1024 * 1024)
    raise AssertionError('guest exceeded memory allowance')
with contextlib.suppress(PermissionError, ValueError):
    resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    raise AssertionError('guest raised hard memory limit')
with contextlib.suppress(PermissionError):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    raise AssertionError('guest forked')
