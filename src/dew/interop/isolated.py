"""A process of its own for code that sets torch's defaults process-wide: tier 2's float64 twin of a source
(`dew.interop.verify`), whose build and runs set the default dtype and widen the float32 casts and pins a
source's code carries (`_dew_torch.precision.float64`), which no other thread of a load's process may see.

The process is a fresh interpreter running this module, which calls the module-level functions it is sent
and replies with their values, both pickled over pipes of their own, so nothing the functions print reaches
the replies. It imports what the load's process imports (that process's `sys.path`), so a class a module
defines, sent by reference, is found there. It imports the load's script itself only where it is asked to
(`main`): a class the script defines is found there as `multiprocessing`'s spawn finds it, the script run
as `__mp_main__`, its `if __name__ == "__main__":` block skipped, which any other load leaves unrun; and a
load inside a daemon process (a pool's worker) starts it, which a `multiprocessing` child of a daemon may
not.
"""

import os
import pickle
import subprocess
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection


class Isolated:
    """A process of its own (this module's): `call` runs a module-level function there. It ends with the
    context."""

    def __init__(self):
        asked, asks = os.pipe()
        replies, replied = os.pipe()
        environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(path for path in sys.path if path)}
        self._process = subprocess.Popen([sys.executable, '-m', __name__, str(asked), str(replied)],
                                         pass_fds=(asked, replied), env=environment)
        os.close(asked)
        os.close(replied)
        self._sent = Connection(asks, readable=False)
        self._read = Connection(replies, writable=False)

    def __enter__(self) -> 'Isolated':
        return self

    def __exit__(self, failed: type[BaseException] | None, *_) -> None:
        try:
            if failed is None:
                # An empty message ends the process's loop (`_serve`).
                self._sent.send_bytes(b'')
            else:
                # An exchange broken off (an interrupt, a failed call): the process may be mid-reply, and
                # its reply is not read.
                self._process.kill()
        finally:
            self._sent.close()
            self._read.close()
            self._process.wait()

    def call(self, function: Callable, *arguments):
        """`function(*arguments)` in the process, its value: `function` a module-level function, which the
        process imports by its module and name, and `arguments` and the value pickled. Raises
        `IsolatedFailure` naming its error where it raised, or where the process ended before replying."""
        self._sent.send_bytes(pickle.dumps((function, arguments)))
        try:
            failed, value = pickle.loads(self._read.recv_bytes())
        except EOFError as ended:
            raise IsolatedFailure(f"its process ended with code {self._process.wait()} before replying "
                                  f"to {function.__qualname__}") from ended
        if failed:
            raise IsolatedFailure(value)
        return value


def main() -> dict:
    """The load's script as `multiprocessing`'s spawn reads it to import it in another process, which
    `Isolated.call(multiprocessing.spawn.prepare, main())` imports there: all but the process's
    authentication key, which no process but spawn's own child may be sent (an `AuthenticationString`
    refuses to pickle elsewhere) and which this process, connecting to none, has no use for."""
    import multiprocessing.spawn

    prepared = multiprocessing.spawn.get_preparation_data('isolated')
    return {key: value for key, value in prepared.items() if key != 'authkey'}


class IsolatedFailure(RuntimeError):
    """A function `Isolated.call` ran raised in its process: its error's type and message."""


def _serve(requests: Connection, replies: Connection) -> None:
    """The process's loop: each function sent read and called and its value, or its error's type and
    message, replied, until the load's process sends an empty message. A request that does not unpickle
    here (a class an interactive session defined, which no file holds) is an error replied like any
    other."""
    while message := requests.recv_bytes():
        try:
            function, arguments = pickle.loads(message)
            reply = (False, function(*arguments))
        except Exception as error:
            # The process boundary: any error a request raises is relayed whole to the caller.
            reply = (True, f"{type(error).__name__}: {error}")
        replies.send_bytes(pickle.dumps(reply))


if __name__ == '__main__':
    _serve(Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False))
