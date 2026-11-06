"""A test's choice of a recurrence's port (`dew.nn.kernels.generation.chosen`), off a GPU interpreted.

A port's fp32 products are TF32 (fla's and mamba_ssm's Triton dots), which the run's declaration takes at
the default product precision alone (`recurrence_chunk._port_precision`): under the suite's lanes'
'highest' (`lane_environment`) a port a test names hands an fp32 call to XLA. A test of the port runs it at
the default, the precision its products are declared at.
"""

import contextlib
from collections.abc import Iterator

import jax

from dew.nn.kernels import generation


@contextlib.contextmanager
def row_names_port(choice: str, kernel: str, *, port: bool = True) -> Iterator[None]:
    """`choice`'s port `kernel` where `port`, at the default product precision, else XLA at the run's."""
    with (generation.chosen(**{choice: kernel if port else "xla"}),
          jax.default_matmul_precision("default") if port else contextlib.nullcontext()):
        yield
