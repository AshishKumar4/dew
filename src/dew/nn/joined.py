"""Projections of one input run as one GEMM, their kernels joined per call.

A layer's projections that read one input stay separate parameters, as its
checkpoints store them. Where a generation's kernel row measured the join
faster (`KERNELS`, 'joined'), a call stacks their kernels and runs one
product: one wide GEMM fills the device where narrower ones over the same
rows do not, and reads the input once. Stacking copies the kernels once a
call, small beside the activations, and their gradients unstack the same way.
"""

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import meta

from dew.cache import INTERCEPTED
from dew.nn.dense import _plain_dense
from dew.nn.precision import biased, float32_dot_general

from .kernels.generation import INHERENT, measured_kernel
from .kernels.selection import Kernel, Op, Requirement, select


def _direct_qwix_interception() -> bool:
    """Direct Qwix calls expose no public running-provider query. Keep their active ancestor check
    here until Flax or Qwix exposes it; lifted transforms retain this stack when parents change.
    A quantized child may run below an ordinary parent, so every active ancestor matters."""
    from flax.linen.module import _context

    return any("_unquantized_type" in vars(layer) for module in _context.module_stack if module is not None
               for layer in type(module).__mro__)


def _intercepted() -> bool:
    """A Dew interception scope, or a direct Qwix scope outside Dew's wrapper entry points."""
    return INTERCEPTED.value or _direct_qwix_interception()


@dataclasses.dataclass(frozen=True)
class JoinCase:
    """The projections of one input as the join's selector reads them (`joins`): whether each is a plain
    Dense (`dense`) on the join's dot_general (`dot_general`), whether they agree in width, bias, dtype,
    precision and promotion (`uniform`), whether a wrapper intercepts their own calls (`intercepted`),
    and whether each holds its own kernel (`held`)."""

    dense: bool
    dot_general: bool
    uniform: bool
    intercepted: bool
    held: bool


JOIN_SUPPORTS: tuple[Requirement[JoinCase], ...] = (
    Requirement('dense', lambda case: case.dense,
                "a projection is not a plain Dense, whose call the join reproduces", row=INHERENT,
                kind='handoff'),
    Requirement('dot_general', lambda case: case.dot_general,
                "a projection takes a dot_general other than the join's (`float32_dot_general`)",
                row=INHERENT, kind='handoff'),
    Requirement('uniform', lambda case: case.uniform,
                "the projections differ in width, bias, dtype, precision or promotion", row=INHERENT,
                kind='handoff'),
    Requirement('intercepted', lambda case: not case.intercepted,
                "the model computes under a wrapper (an adapter, a quantization) that intercepts each "
                "projection's own call", row=INHERENT, kind='handoff'),
    Requirement('kernel', lambda case: case.held, "a projection holds no kernel of its own", row=INHERENT,
                kind='handoff'),
)
"""What one GEMM needs to stand in for the projections' own calls: each a handoff, so a call outside
them runs its projections apart even where the join is chosen, as a wrapped model's projections keep their
own calls (`selection.Requirement`)."""


def join_case(projections: Sequence[nn.Module]) -> JoinCase:
    """The `JoinCase` of `projections`."""
    return JoinCase(
        dense=all(_plain_dense(projection) for projection in projections),
        dot_general=all(projection.dot_general in (None, float32_dot_general)
                        and projection.dot_general_cls is None for projection in projections),
        uniform=len({(projection.features, projection.use_bias, projection.dtype, projection.precision,
                      projection.promote_dtype) for projection in projections}) == 1,
        intercepted=_intercepted(),
        held=all(projection.has_variable('params', 'kernel') for projection in projections))


def _join(choice: str) -> Op[JoinCase]:
    """The join's selector for `choice` ('gate_up', 'delta_gates'): the projections joined where the
    generation's row measured the join faster and its requirements hold, else apart."""
    return Op(choice, (Kernel('joined', JOIN_SUPPORTS), Kernel('apart', (), generic=True)),
              lambda case: measured_kernel(choice, 'apart'))


JOINS = {choice: _join(choice) for choice in ('gate_up', 'delta_gates')}
"""The join's selector of each choice a layer joins projections for."""


def joins(choice: str, projections: Sequence[nn.Module]) -> bool:
    """Whether `projections` run as one GEMM (`joined_dense`), as the join's selector for `choice`
    reports it (`JOINS`): where the generation's row measured `choice` 'joined' and their calls take
    nothing the join does not reproduce. Initialization creates each projection's parameters through
    its own call."""
    if projections[0].is_initializing():
        return False
    return select(JOINS[choice], join_case(projections)).kernel == 'joined'


def joined_dense(x: jax.Array, projections: Sequence[nn.Module], *, axis: int) -> jax.Array:
    """`projections`' outputs of `x` from one GEMM, stacked on output axis `axis`: -2 gives
    `[..., n, features]`, -1 `[..., features, n]`. Each kernel and bias is read and cast as its
    own call casts it; `joins` says when the result is theirs."""
    if axis not in (-2, -1):
        raise ValueError(f"the projections stack on output axis -2 or -1, not {axis}")
    promoted = [projection.promote_dtype(
        x, meta.unbox(projection.get_variable('params', 'kernel')),
        meta.unbox(projection.get_variable('params', 'bias')) if projection.use_bias else None,
        dtype=projection.dtype) for projection in projections]
    inputs = promoted[0][0]
    # A kernel is [inputs, features]; the stacked axis sits beside its features as it will in the output.
    kernel = jnp.stack([kernel for _, kernel, _ in promoted], axis=axis + 3)
    biases = (jnp.stack([bias for _, _, bias in promoted], axis=axis + 2) if projections[0].use_bias
              else None)
    dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
    y = float32_dot_general(inputs, kernel, dimensions, precision=projections[0].precision)
    if biases is None:
        return y
    # Taken and biased as each Dense takes and biases its own (`dew.nn.dense`).
    return biased(y, biases)
