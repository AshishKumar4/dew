"""Projections of one input run as one GEMM, their kernels joined per call.

A layer's projections that read one input stay separate parameters, as its
checkpoints store them. Where the join measured faster on a device (`JOINS`'
declared cells), a call stacks their kernels and runs one
product: one wide GEMM fills the device where narrower ones over the same
rows do not, and reads the input once. Stacking copies the kernels once a
call, small beside the activations, and their gradients unstack the same way.

Projections of unequal widths (a layer's query, key and value) run side by
side (`beside_dense`): one launch over their columns, each its own output,
and their input gradient chained in one accumulator (`gemm.gemm`'s
`beside`), where a launch takes that product (`side_by_side`).

A caller that reads a projection's own call (any Flax method interceptor: a
watch, node perturbation; Flax's captured intermediates) keeps it apart, as a
wrapper that intercepts it does.
"""

import contextlib
import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from typing import TypeGuard

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import meta
from flax.linen.dtypes import promote_dtype
from flax.typing import PrecisionLike

from dew.nn.dense import Dense, _biased, _plain_dense, _promoted, _rounds_once, intercepting, observed
from dew.nn.precision import biased, float32_dot_general

from .kernels import gemm, generation
from .kernels.generation import INHERENT, Choice, device_generation
from .kernels.projections import kernel_path, mapped_shards, recipe_takes, shard_placement
from .kernels.selection import Kernel, KernelRefused, Op, Requirement, forced, select
from .sharding import automatic_axes, on_shards


@dataclasses.dataclass(frozen=True)
class Projections:
    """A layer's projections as a selector that stands in for their own calls reads them (the join's, the
    fused gated MLP's, `dew.nn.gated_mlp`): whether each is a plain Dense (`dense`) on the default product
    (`dot_general`), whether a wrapper intercepts their own calls (`intercepted`), whether each holds
    its kernel and bias as its own call declares them (`held`, `_held`), whether the run's low-precision
    recipe takes any one's product apart or their joined product (`recipe`: `projections.recipe_takes`),
    each quantized as its own, which a join of them would not be, and whether a caller reads any one's own
    call (`observed`, `dense.observed`); `leaves` the held leaves, the one read of them a join's operands are
    prepared from (`prepared`), None where any is not held."""

    dense: bool
    dot_general: bool
    intercepted: bool
    held: bool
    recipe: bool
    observed: bool
    leaves: "Leaves | None" = dataclasses.field(default=None, compare=False)


@dataclasses.dataclass(frozen=True)
class Leaves:
    """Projections' leaves as their own calls declare them (`_leaves`), unboxed: each one's kernel `[depth,
    features]` and its bias as stored (None where it takes none)."""

    kernels: tuple[jax.Array, ...]
    stored: tuple[jax.Array | None, ...]


def plain_denses(projections: Sequence[nn.Module]) -> TypeGuard[Sequence[Dense]]:
    """Whether every one of `projections` is a plain Dense (`dense._plain_dense`)."""
    return all(_plain_dense(projection) for projection in projections)


def projections_of(projections: Sequence[nn.Module], depth: int | None = None) -> Projections:
    """The `Projections` of `projections`: whether a wrapper owns their products each asks within its own
    scope (`Dense.product_owned`), a rule over its module path matching it there, or any Flax method
    interceptor is installed (`dense.intercepting`). Of projections not every
    one a plain Dense nothing more is read (a module that is no projection, a Dropout, has none of a
    Dense's fields): none stands in for them."""
    if not plain_denses(projections):
        return Projections(dense=False, dot_general=False, intercepted=False, held=False, recipe=False,
                           observed=False)
    leaves = _leaves(projections, depth)
    return Projections(
        dense=True,
        dot_general=all(projection.dot_general in (None, float32_dot_general)
                        and projection.dot_general_cls is None for projection in projections),
        intercepted=any(intercepting(projection) or projection.product_owned() for projection in projections),
        held=leaves is not None, recipe=leaves is not None and _recipe_differs(leaves.kernels),
        observed=any(observed(projection) for projection in projections), leaves=leaves)


def _leaves(projections: Sequence[Dense], depth: int | None) -> Leaves | None:
    """`projections`' leaves (`Leaves`) where each holds its kernel, and its bias where it takes one, of the
    shapes its own call declares them (`Dense._linear`'s params: `[depth, features]`, `depth` its input's
    width where given, and `[features]`), else None: its own call reports the tree's error, which a read of
    its leaves would pass by (a missing bias left out, a one-wide one broadcast)."""
    kernels, stored = [], []
    for projection in projections:
        kernel = projection.get_variable('params', 'kernel')
        bias = projection.get_variable('params', 'bias') if projection.use_bias else None
        if kernel is None or (projection.use_bias and bias is None):
            return None
        kernel, bias = meta.unbox(kernel), None if bias is None else meta.unbox(bias)
        width = projection.features
        if not (kernel.ndim == 2 and kernel.shape[1] == width and depth in (None, kernel.shape[0])
                and (bias is None or bias.shape == (width,))):
            return None
        kernels.append(kernel)
        stored.append(bias)
    return Leaves(tuple(kernels), tuple(stored))


def _recipe_differs(kernels: Sequence[jax.Array]) -> bool:
    """Whether the run's low-precision recipe takes any of projections' `kernels` apart or their joined
    product (`projections.recipe_takes`, the one decision `recipe_projection` reads, of each kernel's `[depth,
    width]` and of the joined `[depth, their widths' sum]` the join multiplies by): either way one GEMM of
    them would not be quantized as they are apart (train 928 F1: two 16-wide gates no MXFP8 block takes,
    joined 32 wide, which one does)."""
    shapes = [kernel.shape[-2:] for kernel in kernels]
    joined = (shapes[0][0], sum(width for _, width in shapes))
    return any(recipe_takes(*shape) for shape in (*shapes, joined))


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
        Requirement('kernel', lambda case: of(case).held,
                    "a projection holds no kernel or bias of the width its own call declares, whose error "
                    "its own call reports", row=INHERENT, kind='handoff'),
        Requirement('observed', lambda case: not of(case).observed,
                    f"a caller reads a projection's own output as Flax captures it "
                    f"(`capture_intermediates`), which {stands_in} does not make", row=INHERENT,
                    kind='handoff'),
        Requirement('recipe', lambda case: not of(case).recipe,
                    f"the run's low-precision recipe takes the projections' products (`projections."
                    f"recipe_projection`), each quantized as its own, which {stands_in} does not",
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
    return JoinCase(projections=projections_of(projections), uniform=_agree(projections, widths=True))


def _agree(projections: Sequence[nn.Module], *, widths: bool) -> bool:
    """Whether `projections` agree in bias, dtype, precision, promotion and product, and in width where
    `widths`: what one GEMM's columns share."""
    return len({(projection.features if widths else None, projection.use_bias, projection.dtype,
                 projection.precision, projection.promote_dtype, projection.dot_general)
                for projection in projections}) == 1


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


@dataclasses.dataclass(frozen=True)
class Prepared:
    """Projections' operands as each one's own call casts them (`dense._promoted`): each one's input, kernel
    and bias as promoted, and each bias as stored (`dense._biased`'s `stored`: its rounding reads it)."""

    inputs: tuple[jax.Array, ...]
    kernels: tuple[jax.Array, ...]
    biases: tuple[jax.Array | None, ...]
    stored: tuple[jax.Array | None, ...]


def prepared(x: jax.Array, projections: Sequence[Dense], leaves: Leaves) -> Prepared:
    """`projections`' operands over `x` (`Prepared`) from their held `leaves` (`_leaves`), each kernel and
    bias cast as its own call casts it (`dense._promoted`)."""
    promoted = [_promoted(projection, x, kernel, bias)
                for projection, kernel, bias in zip(projections, leaves.kernels, leaves.stored, strict=True)]
    inputs, kernels, biases = (tuple(each) for each in zip(*promoted, strict=True))
    return Prepared(inputs, kernels, biases, leaves.stored)


def _mapped[R](product: Callable[..., R], inputs: jax.Array, kernels: tuple,
               shards: tuple[jax.sharding.PartitionSpec, dict] | None) -> R:
    """`product(inputs, *kernels)`, or, with `shards` (a split and the kernels' placements, `mapped_shards`'
    or `shard_placement`'s), each device's rows its own call, the kernels entering apart, each placed by the
    layout and gathered whole there (`on_shards`), so each one's gradient leaves the map at its own shape."""
    if shards is None:
        return product(inputs, *kernels)
    split, gathered = shards
    return on_shards(product, split, (inputs, *kernels), gathered=gathered)


def joined_dense(x: jax.Array, projections: Sequence[Dense], *, axis: int) -> jax.Array:
    """`projections`' outputs of `x` from one GEMM, stacked on output axis `axis`: -2 gives
    `[..., n, features]`, -1 `[..., features, n]`. Each kernel and bias is read and cast as its
    own call casts it (`prepared`), and the product taken and biased by a Dense's own arithmetic
    (`dense._biased`: a 16-bit product with an fp32-stored bias rounded once with it, as each projection
    apart rounds its own); `joins` says when the result is theirs."""
    if axis not in (-2, -1):
        raise ValueError(f"the projections stack on output axis -2 or -1, not {axis}")
    leaves = _leaves(projections, x.shape[-1])
    if leaves is None:
        raise ValueError("the joined projections hold no kernel or bias of the shapes their calls declare: "
                         "`joins` takes none such")
    first, operands = projections[0], prepared(x, projections, leaves)
    inputs, kernels = operands.inputs[0], operands.kernels
    # A kernel is [inputs, features]; the stacked axis sits beside its features as it will in the output,
    # and the biases beside theirs, so the kernels side by side and the biases flattened alike pair each
    # output column with its own bias.
    def flat(biases: tuple) -> jax.Array | None:
        return None if biases[0] is None else jnp.stack(biases, axis=axis + 2).reshape(-1)

    dimensions = (((inputs.ndim - 1,), (0,)), ((), ()))
    # Where a launch takes a device's rows' product (`projections.mapped_shards`), each device's rows are
    # their own product (`_mapped`), joined there; the bias is added past it, its gradient summed as a
    # Dense's. A product its layer rounds once with an fp32 bias (`_rounds_once`) is XLA's widened one, as
    # a Dense's is, and stays with the partitioner.
    widened = first.use_bias and _rounds_once(first, inputs, kernels[0], flat(operands.biases),
                                              flat(operands.stored))
    shards = None if widened else mapped_shards(inputs, kernels, tuple(map(kernel_path, projections)),
                                                inputs.dtype, first.precision)

    def taken(product):
        # One matrix product of the kernels side by side, as a Dense takes its own (`dense._biased`).
        def joined(inputs, *held):
            return product(inputs, jnp.stack(held, axis=axis + 3).reshape(inputs.shape[-1], -1))

        return _mapped(joined, inputs, kernels, shards)

    y = _biased(first, inputs, kernels[0], dimensions, flat(operands.biases), lambda bias: bias,
                flat(operands.stored), taken=taken)
    stacked = (len(kernels), kernels[0].shape[1]) if axis == -2 else (kernels[0].shape[1], len(kernels))
    return y.reshape(*y.shape[:-1], *stacked)


@dataclasses.dataclass(frozen=True)
class BesidePlan:
    """Projections' outputs side by side as their selector admitted them (`side_by_side`): their operands
    (`prepared`), the placement a mesh splitting their rows alone gives each device's own (`shard_placement`:
    its split, the kernels' placements and a device's rows; None elsewhere), and their product's precision;
    called, each projection's output (`beside_dense`)."""

    operands: Prepared
    placement: tuple[jax.sharding.PartitionSpec, dict, jax.ShapeDtypeStruct] | None
    precision: PrecisionLike
    named: bool = False
    """Whether the call names the side-by-side product (`qkv` `joined`): its launches the template's where
    legal, whatever a receipt measured (`_launching`; base review 760be77fe F2)."""
    forward: generation.Route | None = None
    """The forward's route the selector admitted them by (`gemm.side_by_side`'s, `BesideCase.route`), which
    their product runs and reports (`gemm`'s `selected`; base review b68fc2382 F5)."""

    def __call__(self) -> tuple[jax.Array, ...]:
        return beside_dense(self)


def _launching(named: bool):
    """The scope a named side-by-side product's launches run in: the grouped GEMM's template named where its
    caller names no grouped choice, its products' launches taken where legal and refused by name where not
    (`generation.chosen`); the caller's own where it names the template's, another refused before any
    launch (`BESIDE_SUPPORTS`' `grouped`); else the selectors' own (an unnamed call's measured routes)."""
    if named and forced(Choice.GROUPED_MATMUL) is None:
        return generation.chosen(grouped_matmul='mosaic')
    return contextlib.nullcontext()


class BesideCase:
    """The projections of one input `x` as the side-by-side selector reads them (`side_by_side`), each fact
    formed where a requirement first reads it, in the requirements' order (`BESIDE_SUPPORTS`), so a
    structural handoff refuses them before their operands are prepared, and every requirement before a
    launch is asked: their `Projections` (`found`); whether their calls agree in all their columns share but
    their widths and promote by dtype alone (`agreed`); whether they give one input and kernels of its one
    dtype (`uniform`); whether any one's 16-bit product rounds once with its fp32 bias (`widened`: XLA's
    widened product, `dense._rounds_once`, which no launch takes); and whether a launch takes their product
    side by side (`route`: `gemm.side_by_side`'s forward route, of a device's own rows where a mesh splits
    them alone, `plan`), its forward and its input gradient independent choices (`gemm.side_by_side`)."""

    def __init__(self, projections: Sequence[Dense], x: jax.Array):
        self.projections, self.x = projections, x
        self.refused: KernelRefused | None = None
        """Why a kernel the call names refuses their forward side by side (`route`), where one does."""

    @functools.cached_property
    def found(self) -> Projections:
        return projections_of(self.projections, self.x.shape[-1])

    @functools.cached_property
    def agreed(self) -> bool:
        return _agree(self.projections, widths=False) and self.projections[0].promote_dtype is promote_dtype

    @functools.cached_property
    def plan(self) -> BesidePlan:
        # The leaves `found` read and checked, once (`Projections.leaves`): read only past its handoffs.
        leaves = self.found.leaves
        assert leaves is not None, "a plan of projections that hold no leaves their calls declare"
        operands = prepared(self.x, self.projections, leaves)
        paths = tuple(map(kernel_path, self.projections))
        placement = shard_placement(operands.inputs[0], operands.kernels, paths)
        named = forced(Choice.QKV) == 'joined'
        return BesidePlan(operands, placement, self.projections[0].precision, named)

    @functools.cached_property
    def uniform(self) -> bool:
        operands = self.plan.operands
        return len({each.dtype for each in (*operands.inputs, *operands.kernels)}) == 1

    @functools.cached_property
    def kept(self) -> bool:
        # The input as its calls take it, uncast: a cast of `x` (fp32 activations into fp16 projections) has
        # each call's input cotangent widened back before they sum, where one product's sums them in its
        # dtype (base review b68fc2382 F1: 40,000 and 30,000 summed to fp16's infinity, 70,000 apart).
        return self.plan.operands.inputs[0].dtype == jnp.asarray(self.x).dtype

    @functools.cached_property
    def widened(self) -> bool:
        operands = self.plan.operands
        return any(bias is not None and _rounds_once(projection, inputs, kernel, bias, stored)
                   for projection, inputs, kernel, bias, stored in zip(
                       self.projections, operands.inputs, operands.kernels, operands.biases, operands.stored,
                       strict=True))

    @functools.cached_property
    def route(self) -> generation.Route | None:
        inputs, kernels = self.plan.operands.inputs[0], self.plan.operands.kernels
        placement = self.plan.placement
        rows = (placement[2] if placement is not None
                else jax.ShapeDtypeStruct((math.prod(inputs.shape[:-1]), inputs.shape[-1]), inputs.dtype))
        matrices = tuple(jax.ShapeDtypeStruct(kernel.shape, kernel.dtype) for kernel in kernels)
        # Seen whole where a placement maps the rows, else where the rows and every kernel are, as `gemm`'s
        # forward and input gradient read them (head review of b68fc2382: replicated rows by a varying kernel
        # in a shard_map took no compound launch there).
        whole = placement is not None or not automatic_axes()
        with _launching(self.plan.named):
            try:
                return gemm.side_by_side(rows, matrices, inputs.dtype, self.plan.precision, whole=whole)
            except KernelRefused as refused:
                # A kernel the call names refusing their forward side by side: they run apart, the refusal
                # the selector's reason (`BESIDE_SUPPORTS`' `launched`).
                self.refused = refused
                return None


BESIDE_SUPPORTS: tuple[Requirement[BesideCase], ...] = (
    *projection_requirements(lambda case: case.found, "the side-by-side product"),
    Requirement('agreed', lambda case: case.agreed,
                "the projections differ in bias, dtype, precision, promotion or product, or promote other "
                "than by dtype", row=INHERENT, kind='handoff'),
    Requirement('uniform', lambda case: case.uniform,
                "the projections' calls give their input or kernels different dtypes", row=INHERENT,
                kind='handoff'),
    Requirement('kept', lambda case: case.kept,
                "the projections' calls cast their input, each one's cotangent widened back before they sum",
                row=INHERENT, kind='handoff'),
    Requirement('widened', lambda case: not case.widened,
                "a 16-bit product rounds once with its fp32 bias, XLA's widened product", row=INHERENT,
                kind='handoff'),
    Requirement('grouped',
                lambda case: not case.plan.named or forced(Choice.GROUPED_MATMUL) in (None, 'mosaic'),
                lambda case: (f"grouped_matmul {forced(Choice.GROUPED_MATMUL)!r} is named beside qkv "
                              "'joined', whose product runs the grouped GEMM template's launches"),
                row=INHERENT),
    Requirement('launched', lambda case: case.route is not None,
                lambda case: ("no launch takes the projections' product side by side, its forward or its "
                              "input gradient (`gemm.side_by_side`)"
                              + ("" if case.refused is None else f": {case.refused}")), row=INHERENT),
)
"""What one side-by-side launch needs to stand in for the projections' own calls
(`projection_requirements`), what its columns share, and, last, a launch that takes it (head review of
4d6c7a69c F4: the launch asked before the structural refusals)."""

BESIDE = Op(Choice.QKV, (Kernel[BesideCase]('joined', BESIDE_SUPPORTS), Kernel('apart', (), generic=True)))
"""The side-by-side product's selector: the projections side by side where every requirement holds, a launch
taking that product the last (`BesideCase.route`), else each its own call."""


def side_by_side(projections: Sequence[nn.Module], x: jax.Array) -> BesidePlan | None:
    """Several projections of `x` side by side as `BESIDE` admits them (`BesidePlan`: called, their
    outputs), else None, each its own call: one projection, or initialization, which creates each one's
    parameters through its own call, takes none."""
    if len(projections) < 2 or projections[0].is_initializing() or not plain_denses(projections):
        return None
    case = BesideCase(projections, x)
    if select(BESIDE, case).kernel != 'joined':
        return None
    return dataclasses.replace(case.plan, forward=case.route)


def beside_dense(plan: BesidePlan) -> tuple[jax.Array, ...]:
    """The projections' outputs `plan` admitted, each its own, from one `gemm` of their columns side by side
    (`gemm.gemm`'s `beside`): its forward one launch where the grouped selector took it so as it admitted
    them (`plan.forward`, run as decided), else each product its own call (Hopper's measured route: the
    forward apart, the input gradient one chained launch, `grouped_matmul._grouped_preferred`); each bias
    added past its product as a Dense adds it (`precision.biased`). Where a mesh splits the rows alone, each
    device's rows are their own call (`_mapped`, `plan.placement`), as the selector asked of them. A named
    side-by-side product runs the template's launches, its pullback too (`_launching`, `GROUPED.bound`)."""
    operands = plan.operands

    def side_by_side(inputs, *kernels):
        rows = inputs.reshape(-1, inputs.shape[-1])
        with _launching(plan.named):
            outs = gemm.gemm(rows, kernels[0], beside=kernels[1:], out_dtype=inputs.dtype,
                             precision=plan.precision, selected=plan.forward)
        return tuple(out.reshape(*inputs.shape[:-1], out.shape[-1]) for out in outs)

    placement = None if plan.placement is None else plan.placement[:2]
    outs = _mapped(side_by_side, operands.inputs[0], operands.kernels, placement)
    return tuple(out if bias is None else biased(out, bias)
                 for out, bias in zip(outs, operands.biases, strict=True))
