"""A gated MLP's three products composed on the GEMM template's hooks (docs/design/kernel-templates.md,
Appendix C rows 17 and 18).

The gate and up projections are one paired product (`gemm`'s `paired`) whose epilogue writes the two
projections and their gated product (`activations.gated_hook`: the activation and its roundings), so the
product is formed as the tiles are written and never in a pass of its own. The down projection multiplies
that product as its prologue's formed operand (`gemm`'s `applied`): the prologue is the same hook, over the
gate as its rows and the up projection as its side, so the down projection's reverse mode forms the hook's
VJP, which JAX derives from the traced hook, in the epilogue of its input gradient's product, writing the
gate's and the up projection's cotangents where the hidden activation's would have been. No derivative is
written here: each is `gemm`'s rule for its hooks.

The fused MLP runs where its selector (`GATED_MLP`) takes it: named (`generation.chosen(gated_mlp=
'fused')`), or on a generation whose receipt declares it faster (`FUSED_CELLS`); its projections apart
otherwise, as `GatedMLP` ran them.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeGuard

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import meta
from flax.linen.dtypes import canonicalize_dtype
from flax.typing import PrecisionLike
from jax.ad_checkpoint import checkpoint_name

from dew.nn import activations
from dew.nn.dense import Dense, _promoted
from dew.nn.joined import Projections, projection_requirements, projections_of
from dew.nn.kernels import gemm, grouped_matmul, projections
from dew.nn.kernels.generation import INHERENT, Choice, device_generation, gpu_target
from dew.nn.kernels.selection import Kernel, Op, Requirement, select
from dew.nn.precision import recipe_role
from dew.nn.sharding import seen_whole

FUSION_ROW = 'ledger 1958'
"""The row tracking the fused MLP's reach: a projection it does not reproduce, and its receipts."""


@dataclasses.dataclass(frozen=True)
class GatedCase:
    """A gated MLP's three projections (gate, up, down) as the fused MLP's selector reads them
    (`GATED_MLP`): their `joined.Projections`, whether they agree in dtype, precision and promotion
    (`uniform`), whether any adds a bias (`biased`), whether the run's low-precision recipe takes the
    products (`recipe`, `precision.recipe_role`), the rows the products multiply in the dtype they run at
    (`rows`, flattened, `fused_products`'), the gate's and the down projection's kernels in it (`kernels`,
    None where a projection holds none), whether the rows and kernels are seen whole (`whole`, `seen_whole`)
    and the products' `precision`."""

    projections: Projections
    uniform: bool
    biased: bool
    recipe: bool
    rows: jax.ShapeDtypeStruct
    kernels: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct] | None
    whole: bool
    precision: PrecisionLike
    hook: Callable[[jax.Array, jax.Array], jax.Array]
    """The gated product's hook (`activations.gated_hook` of the activation and its limit), which the paired
    product's epilogue and the down product's prologue trace."""

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype the products run at, the promotion `gated_mlp` runs (`promote_dtype`'s)."""
        return jnp.dtype(self.rows.dtype)

    @property
    def cell(self) -> tuple[str, jnp.dtype]:
        """The cell a receipt measures (`selection.Celled`): the device's generation and the products'
        dtype."""
        return device_generation(), self.dtype

    def launched(self) -> bool:
        """Whether the template launches the up product and the down product (`projections.launches`), asked
        only where the fused MLP is being chosen (its last requirement), so a call that runs its products
        apart reports no route of a product it never launches."""
        if self.kernels is None:
            return False
        gate, down = self.kernels
        hidden = jax.ShapeDtypeStruct((self.rows.shape[0], gate.shape[1]), self.dtype)
        tile = jax.ShapeDtypeStruct((128, 128), self.dtype)
        return (all(projections.launches(rows, kernel, self.dtype, self.precision, whole=self.whole)
                    for rows, kernel in ((self.rows, gate), (hidden, down)))
                and grouped_matmul.hook_lowers(self.hook, (tile, tile), gpu_target()))


FUSED_SUPPORTS: tuple[Requirement[GatedCase], ...] = (
    *projection_requirements(lambda case: case.projections, "the fused MLP"),
    Requirement('uniform', lambda case: case.uniform,
                "the projections differ in dtype, precision or promotion", row=INHERENT, kind='handoff'),
    Requirement('biased', lambda case: not case.biased,
                "a projection adds a bias, which no hook of the fused MLP adds yet", row=FUSION_ROW,
                kind='handoff'),
    Requirement('recipe', lambda case: not case.recipe,
                "the run's low-precision recipe takes the projections' products (`projections."
                "recipe_projection`), which the fused MLP's products do not", row=FUSION_ROW, kind='handoff'),
    Requirement('launched', lambda case: case.launched(),
                "no launch of the GEMM template takes the up or the down product here (a mesh axis splitting "
                "them, a CPU, a declared XLA cell), or its hooks the gated product's operations (an exact "
                "GELU's erf behind an optimization barrier)", row=FUSION_ROW, kind='handoff'),
)
"""What the fused MLP needs to stand in for the projections' own calls: each a handoff, so a call outside
them runs its projections apart even where the fused MLP is named."""

FUSED = Kernel[GatedCase]('fused', FUSED_SUPPORTS)
"""The three products on the template's hooks (`gated_mlp`)."""

FUSED_CELLS: dict[tuple[str | None, jnp.dtype], tuple[Kernel[GatedCase], str]] = {}
"""The fused MLP's declared cells, (device generation, activations' dtype) to the fused kernel and its
receipt: none yet. Where the fused Pallas SiLU product holds a cell (`norm_swiglu.SWIGLU_CELLS`), a cell here
is declared only from the fused MLP's own receipt on that generation."""

GATED_MLP: Op[GatedCase] = Op(Choice.GATED_MLP, (FUSED, Kernel('apart', (), generic=True)),
                              lambda case: 'apart', declared=FUSED_CELLS)
"""The fused MLP's selector: the projections apart, but fused where a receipt declares it faster
(`FUSED_CELLS`) or the block names it, and its requirements hold."""


def gated_case(x: jax.Array, projections_: Sequence[nn.Module], activation: str,
               limit: float | None = None) -> GatedCase:
    """The `GatedCase` of a gated MLP's `projections_` (gate, up, down) over its input `x`, its gated product
    `activation`'s clamped at `limit`."""
    gate, _, down = projections_
    found = projections_of(projections_)
    # A kernel a wrapper owns the products of (a served model's quantized kernel, `found.intercepted`) is
    # the wrapper's to read, never an array of the products' dtype.
    kernels = ([meta.unbox(projection.get_variable('params', 'kernel')) for projection in (gate, down)]
               if found.held and not found.intercepted else None)
    dtype = canonicalize_dtype(x, *(kernels or ()), dtype=gate.dtype)
    return GatedCase(
        projections=found,
        uniform=len({(projection.dtype, projection.precision, projection.promote_dtype)
                     for projection in projections_}) == 1,
        biased=any(projection.use_bias for projection in projections_),
        recipe=recipe_role() is not None,
        rows=jax.ShapeDtypeStruct((x.size // x.shape[-1], x.shape[-1]), dtype),
        kernels=None if kernels is None else (jax.ShapeDtypeStruct(kernels[0].shape, dtype),
                                              jax.ShapeDtypeStruct(kernels[1].shape, dtype)),
        whole=seen_whole(x) and all(seen_whole(kernel) for kernel in kernels or ()),
        precision=gate.precision, hook=activations.gated_hook(activation, limit))


def fuses(projections_: Sequence[nn.Module], rows: jax.Array, activation: str,
          limit: float | None = None) -> TypeGuard[Sequence[Dense]]:
    """Whether a gated MLP's projections (gate, up, down) of `rows`, its gated product `activation`'s clamped
    at `limit`, run fused (`gated_mlp`), as the fused MLP's selector reports it (`GATED_MLP`), so each a
    bare `Dense` (its `dense` requirement), which the answer states of them for `gated_mlp`. Initialization
    creates each projection's parameters through its own call."""
    if projections_[0].is_initializing():
        return False
    return select(GATED_MLP, gated_case(rows, projections_, activation, limit)).kernel == 'fused'


def gated_mlp(x: jax.Array, projections_: Sequence[nn.Dense], activation: str, *,
              limit: float | None = None, names: tuple[str, str, str] | None = None) -> jax.Array:
    """A gated MLP's output of `x` through its projections (gate, up, down), fused (`fused_products`), each
    kernel read and cast as its projection's call casts it."""
    promoted = [_promoted(projection, x, meta.unbox(projection.get_variable('params', 'kernel')), None)
                for projection in projections_]
    inputs = promoted[0][0]
    gate, up, down = (kernel for _, kernel, _ in promoted)
    out = fused_products(inputs.reshape(-1, inputs.shape[-1]), (gate, up, down), activation, limit=limit,
                         precision=projections_[0].precision, names=names)
    return out.reshape(*inputs.shape[:-1], out.shape[-1])


def fused_products(rows: jax.Array, kernels: tuple[jax.Array, jax.Array, jax.Array], activation: str, *,
                   limit: float | None = None, precision=None, names: tuple[str, str, str] | None = None,
                   interpret_on_cpu: bool = False) -> jax.Array:
    """`rows`' output through a gated MLP's kernels (gate, up, down; `[in, hidden]`, `[in, hidden]`,
    `[hidden, out]`), fused (the module's docstring): the gate and up projections and their gated product
    (`activations.gated_hook` of `activation` and `limit`) one paired product, the down projection's prologue
    that product, formed. `names`, the checkpoint names of the gate's, the up projection's and the down
    projection's outputs, which a remat policy reads, as `GatedMLP` names them. `interpret_on_cpu`,
    `gemm`'s."""
    dtype = rows.dtype
    hook = activations.gated_hook(activation, limit)

    def projected(gate_acc, up_acc):
        gate, up = gate_acc.astype(dtype), up_acc.astype(dtype)
        return gate, up, hook(gate, up)

    gate, up, hidden = gemm.gemm(rows, kernels[0], paired=kernels[1], epilogue=projected, out_dtype=dtype,
                                 out_dtypes=(dtype,) * 3, precision=precision,
                                 interpret_on_cpu=interpret_on_cpu)
    if names is not None:
        gate, up = checkpoint_name(gate, names[0]), checkpoint_name(up, names[1])
    out = gemm.gemm(gate, kernels[2], prologue=hook, prologue_operands=((up, ('rows', 'depth')),),
                    applied=hidden, out_dtype=dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
    return out if names is None else checkpoint_name(out, names[2])
