"""A process of its own for code that sets torch's defaults process-wide: tier 2's float64 twin of a source
(`dew.interop.verify`), whose build and runs set the default dtype and widen the float32 casts and pins a
source's code carries (`_dew_torch.precision.float64`), which no other thread of a load's process may see.

The process is a fresh interpreter running this module, which calls the module-level functions it is sent
and replies with their values, both pickled over pipes of their own, so nothing the functions print reaches
the replies. It is not `multiprocessing`'s spawn, which imports a script's `__main__` again in its child
and so runs a script's own load there; it imports what the load's process imports (that process's
`sys.path`), so a class a script or a test registered at runtime, sent by reference, builds there from its
module.
"""

import os
import pickle
import subprocess
import sys
from collections.abc import Callable
from typing import BinaryIO


class Isolated:
    """A process of its own (this module's): `call` runs a module-level function there. It ends with the
    context."""

    def __init__(self):
        asked, self._asks = os.pipe()
        self._replies, replied = os.pipe()
        environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(path for path in sys.path if path)}
        self._process = subprocess.Popen([sys.executable, '-m', __name__, str(asked), str(replied)],
                                         pass_fds=(asked, replied), env=environment)
        os.close(asked)
        os.close(replied)
        # Unbuffered, so a request is written whole as it is sent and closing holds nothing to flush into
        # a pipe whose reader ended.
        self._sent: BinaryIO = os.fdopen(self._asks, 'wb', buffering=0)
        self._read: BinaryIO = os.fdopen(self._replies, 'rb')

    def __enter__(self) -> 'Isolated':
        return self

    def __exit__(self, failed: type[BaseException] | None, *_) -> None:
        try:
            if failed is None:
                # None ends the process's loop (`_serve`).
                pickle.dump(None, self._sent)
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
        `IsolatedFailure` naming its error where it raised."""
        pickle.dump((function, arguments), self._sent)
        try:
            failed, value = pickle.load(self._read)
        except EOFError as ended:
            raise IsolatedFailure(f"its process ended with code {self._process.wait()} before replying "
                                  f"to {function.__qualname__}") from ended
        if failed:
            raise IsolatedFailure(value)
        return value


class IsolatedFailure(RuntimeError):
    """A function `Isolated.call` ran raised in its process: its error's type and message."""


def _serve(asks: BinaryIO, replies: BinaryIO) -> None:
    """The process's loop: each function sent called and its value, or its error's type and message,
    replied, until the load's process sends None."""
    while (asked := pickle.load(asks)) is not None:
        function, arguments = asked
        try:
            reply = (False, function(*arguments))
        except Exception as error:
            # The process boundary: any error a function raises is relayed whole to the caller.
            reply = (True, f"{type(error).__name__}: {error}")
        pickle.dump(reply, replies)
        replies.flush()


if __name__ == '__main__':
    with os.fdopen(int(sys.argv[1]), 'rb') as asks, os.fdopen(int(sys.argv[2]), 'wb') as replies:
        _serve(asks, replies)
