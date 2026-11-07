"""A test's choice of a recurrence's port (`dew.nn.kernels.generation.chosen`), off a GPU interpreted.

An interpreted port's products are exact under any declared precision (`recurrence_chunk._port_precision`),
so a test names it at the run's own, the precision the rest of its arm runs at too.
"""

import contextlib
from collections.abc import Iterator

from dew.nn.kernels import generation


@contextlib.contextmanager
def row_names_port(choice: str, kernel: str, *, port: bool = True) -> Iterator[None]:
    """`choice`'s port `kernel` where `port`, else XLA."""
    with generation.chosen(**{choice: kernel if port else "xla"}):
        yield
