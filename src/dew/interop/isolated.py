"""A process of its own for code that sets torch's defaults process-wide: tier 2's float64 twin of a source
(`dew.interop.verify`), whose build and runs set the default dtype and widen the float32 casts and pins a
source's code carries (`_dew_torch.precision.float64`), which no other thread of a load's process may see.

The process is `multiprocessing`'s spawn: a fresh interpreter that imports the load's script as
`__mp_main__` (its `if __name__ == "__main__":` block skipped, as `multiprocessing` asks of any script that
starts a process), so a class the script defines and registered at runtime, sent by reference, is found
there. It calls the module-level functions it is sent and replies with their values over a pipe of its own.
"""

import multiprocessing
import pickle
from collections.abc import Callable
from multiprocessing.connection import Connection


class Isolated:
    """A process of its own (this module's): `call` runs a module-level function there. It ends with the
    context."""

    def __init__(self):
        context = multiprocessing.get_context('spawn')
        self._ours, theirs = context.Pipe()
        self._process = context.Process(target=_serve, args=(theirs,), daemon=True)
        self._process.start()
        theirs.close()

    def __enter__(self) -> 'Isolated':
        return self

    def __exit__(self, failed: type[BaseException] | None, *_) -> None:
        try:
            if failed is None:
                # An empty message ends the process's loop (`_serve`).
                self._ours.send_bytes(b'')
            else:
                # An exchange broken off (an interrupt, a failed call): the process may be mid-reply, and
                # its reply is not read.
                self._process.kill()
        finally:
            self._ours.close()
            self._process.join()

    def call(self, function: Callable, *arguments):
        """`function(*arguments)` in the process, its value: `function` a module-level function, which the
        process imports by its module and name, and `arguments` and the value pickled. Raises
        `IsolatedFailure` naming its error where it raised, or where the process ended before replying."""
        self._ours.send_bytes(pickle.dumps((function, arguments)))
        try:
            failed, value = self._ours.recv()
        except EOFError as ended:
            self._process.join()
            raise IsolatedFailure(f"its process ended with code {self._process.exitcode} before replying "
                                  f"to {function.__qualname__}") from ended
        if failed:
            raise IsolatedFailure(value)
        return value


class IsolatedFailure(RuntimeError):
    """A function `Isolated.call` ran raised in its process: its error's type and message."""


def _serve(requests: Connection) -> None:
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
        requests.send(reply)
