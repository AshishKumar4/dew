"""Projections of one input run as one GEMM, their kernels joined per call.

A layer's projections that read one input stay separate parameters, as its
checkpoints store them. Where the join measured faster on a device (`JOINS`'
declared cells), a call stacks their kernels and runs one
product: one wide GEMM fills the device where narrower ones over the same
rows do not, and reads the input once. Stacking copies the kernels once a
call, small beside the activations, and their gradients unstack the same way.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeGuard

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import meta

from dew.nn.dense import Dense, _biased, _plain_dense, _promoted, intercepted
from dew.nn.precision import float32_dot_general

from .kernels.generation import INHERENT, Choice, device_generation
from .kernels.projections import per_shard, rows_split
from .kernels.selection import Kernel, Op, Requirement, select


@dataclasses.dataclass(frozen=True)
class Projections:
    """A layer's projections as a selector that stands in for their own calls reads them (the join's, the
    fused gated MLP's, `dew.nn.gated_mlp`): whether each is a plain Dense (`dense`) on the default product
    (`dot_general`), whether a wrapper intercepts their own calls (`intercepted`), and whether each holds
    its own kernel (`held`)."""

    dense: bool
    dot_general: bool
    intercepted: bool
    held: bool


def projections_of(projections: Sequence[nn.Module]) -> Projections:
    """The `Projections` of `projections`: whether a wrapper owns their products each asks within its own
    scope (`Dense.product_owned`), a rule over its module path matching it there."""
    dense = all(_plain_dense(projection) for projection in projections)
    return Projections(
        dense=dense,
        dot_general=all(projection.dot_general in (None, float32_dot_general)
                        and projection.dot_general_cls is None for projection in projections),
        intercepted=(any(projection.product_owned() for projection in projections) if dense
                     else intercepted()),
        held=all(projection.has_variable('params', 'kernel') for projection in projections))


def projection_requirements[C](of: Callable[[C], Projections], stands_in: str) -> tuple[Requirement[C], ...]:
    """What a kernel that `stands_in` for projections' own calls needs of them (`of(case)`'s
    `Projections`): each a handoff, so a call outside them runs its projections apart even where the kernel
    is chosen, as a wrapped model's projections keep their own calls (`selection.Requirement`)."""
    return (
        Requirement('dense', lambda case: of(case).dense,
                    f"a projection is not a plain Dense, whose call {stands_in} reproduces", row=INHERENT,
                    kind='handoff'),
        Requirement('dot_general', lambda case: of(case).dot_general,
                    f"a projection takes a dot_general other than {stands_in}'s (`float32_dot_general`)",
                    row=INHERENT, kind='handoff'),
        Requirement('intercepted', lambda case: not of(case).intercepted,
                    "the model computes under a wrapper (an adapter, a quantization) that intercepts each "
                    "projection's own call", row=INHERENT, kind='handoff'),
        Requirement('kernel', lambda case: of(case).held, "a projection holds no kernel of its own",
                    row=INHERENT, kind='handoff'),
    )


@dataclasses.dataclass(frozen=True)
class JoinCase:
    """The projections of one input as the join's selector reads them (`joins`): their `Projections`, and
    whether they agree in width, bias, dtype, precision, promotion and product (`uniform`: the join takes the
    first's product for all)."""

    projections: Projections
    uniform: bool

    @property
    def cell(self) -> tuple[str, str]:
        """The cell a receipt measures (`selection.Celled`): the device's generation, and the one kind of
        call a join is."""
        return device_generation(), 'join'


JOIN_SUPPORTS: tuple[Requirement[JoinCase], ...] = (
    *projection_requirements(lambda case: case.projections, "the join"),
    Requirement('uniform', lambda case: case.uniform,
                "the projections differ in width, bias, dtype, precision, promotion or product", row=INHERENT,
                kind='handoff'),
)
"""What one GEMM needs to stand in for the projections' own calls (`projection_requirements`), and their
agreement in what one GEMM's columns share."""


def join_case(projections: Sequence[nn.Module]) -> JoinCase:
    """The `JoinCase` of `projections`."""
    return JoinCase(
        projections=projections_of(projections),
        uniform=len({(projection.features, projection.use_bias, projection.dtype, projection.precision,
                      projection.promote_dtype, projection.dot_general) for projection in projections}) == 1)


DELTA_GATES_FASTER = ("e5198da4e: a gated delta net's b and a, 24,576 tokens, forward and backward 5.44 -> "
                      "5.29 ms on an A100 40GB and 2.006 -> 1.945 on an H100")
"""The receipt behind the delta net's gates' joined cells."""

JOINED_FASTER = {
    (Choice.GATE_UP, 'sm80'): (
        "dbf402f36: Qwen3.5-0.8B's packed step, 24,576 tokens, one 1024 -> 7168 GEMM for two 1024 -> 3584 "
        "ones, 0.36 ms a layer less on an A100 80GB (forward 1.67 -> 1.54, input gradient 1.55 -> 1.49, "
        "weight gradient 1.70 -> 1.51 ms); none on an H100"),
    (Choice.DELTA_GATES, 'sm80'): DELTA_GATES_FASTER,
    (Choice.DELTA_GATES, 'sm90'): DELTA_GATES_FASTER,
}
"""Each join's cells, (the join, the device generation it measured faster on) to its receipt: a literal of
them, which each join's selector reads and thresh's declared-set seam registers (.thresh/thresh.cue)."""


def _join(choice: Choice) -> Op[JoinCase]:
    """The join's selector for `choice` (`CHOICES`): the projections apart, but joined where
    the join measured faster (`JOINED_FASTER`) and its requirements hold."""
    joined = Kernel[JoinCase]('joined', JOIN_SUPPORTS)
    return Op(choice, (joined, Kernel('apart', (), generic=True)), lambda case: 'apart',
              declared={(device, 'join'): (joined, receipt)
                        for (join, device), receipt in JOINED_FASTER.items() if join == choice})


CHOICES = (Choice.GATE_UP, Choice.DELTA_GATES)
"""The joins layers ask for (`joins`): named apart from their cells, so a join whose every cell is retired
keeps its selector and runs its projections apart."""

JOINS = {choice: _join(choice) for choice in CHOICES}
"""The join's selector of each choice a layer joins projections for."""


def joins(projections: Sequence[nn.Module], choice: Choice) -> TypeGuard[Sequence[Dense]]:
    """Whether `projections` run as one GEMM (`joined_dense`), as the join's selector for `choice` reports it
    (`JOINS`): where the join measured faster on the device and their calls take nothing the join does not
    reproduce, so each a bare `Dense` (the join's `dense` requirement), which the answer states of them for
    `joined_dense` (a guard narrows its first argument). Initialization creates each projection's parameters
    through its own call."""
    if projections[0].is_initializing():
        return False
    return select(JOINS[choice], join_case(projections)).kernel == 'joined'


def joined_dense(x: jax.Array, projections: Sequence[Dense], *, axis: int) -> jax.Array:
    """`projections`' outputs of `x` from one GEMM, stacked on output axis `axis`: -2 gives
    `[..., n, features]`, -1 `[..., features, n]`. Each kernel and bias is read and cast as its
    own call casts it, and the product taken and biased by a Dense's own arithmetic (`dense._biased`: a
    16-bit product with an fp32-stored bias rounded once with it, as each projection apart rounds its own);
    `joins` says when the result is theirs."""
    if axis not in (-2, -1):
        raise ValueError(f"the projections stack on output axis -2 or -1, not {axis}")
    first = projections[0]
    stored = [meta.unbox(projection.get_variable('params', 'bias')) if first.use_bias else None
              for projection in projections]
    promoted = [_promoted(projection, x, meta.unbox(projection.get_variable('params', 'kernel')), bias)
                for projection, bias in zip(projections, stored, strict=True)]
    inputs = promoted[0][0]
    # A kernel is [inputs, features]; the stacked axis sits beside its features as it will in the output,
    # and the biases beside theirs, so the kernels side by side and the biases flattened alike pair each
    # output column with its own bias.
    def flat(biases: list) -> jax.Array | None:
        return None if biases[0] is None else jnp.stack(biases, axis=axis + 2).reshape(-1)

    dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
    kernels = tuple(kernel for _, kernel, _ in promoted)

    def taken(product):
        # One matrix product of the kernels side by side, as a Dense takes its own (`dense._biased`). On a
        # mesh whose automatic axes split the rows alone, each device's rows are their own product, the
        # kernels entering apart and joined there, so each one's gradient is summed at its own shape
        # (`projections.per_shard`); the bias is added past it, its gradient summed as a Dense's.
        def joined(inputs, *held):
            return product(inputs, jnp.stack(held, axis=axis + 3).reshape(inputs.shape[-1], -1))

        if rows_split(inputs) is not None:
            return per_shard(joined, inputs, kernels)
        return joined(inputs, *kernels)

    y = _biased(first, inputs, kernels[0], dimensions, flat([bias for _, _, bias in promoted]),
                lambda bias: bias, flat(stored), taken=taken)
    stacked = (len(kernels), kernels[0].shape[1]) if axis == -2 else (kernels[0].shape[1], len(kernels))
    return y.reshape(*y.shape[:-1], *stacked)
