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
import math
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
from dew.nn.kernels import gemm, projections
from dew.nn.kernels.generation import INHERENT, Choice, device_generation, gpu_target
from dew.nn.kernels.selection import Kernel, Op, Requirement, select, selected
from dew.nn.kernels.templates.gemm import wide
from dew.nn.kernels.templates.gemm.hooks import hook_lowers
from dew.nn.residuals import RECOMPUTED, Residual
from dew.nn.sharding import automatic_axes, on_shards

FUSION_ROW = 'ledger 1958'
"""The row tracking the fused MLP's reach: a projection it does not reproduce, and its receipts."""


@dataclasses.dataclass(frozen=True)
class GatedCase:
    """A gated MLP's three projections (gate, up, down) as the fused MLP's selector reads them (`GATED_MLP`):
    their `joined.Projections`, whether they agree in dtype, precision and promotion (`uniform`), whether any
    adds a bias (`biased`), the rows the products multiply in the dtype they run at (`rows`, flattened,
    `fused_products`'), the gate's and the down projection's kernels in it (`kernels`, `[in, hidden]` /
    `[hidden, out]` for dense or `[experts, in, hidden]` / `[experts, hidden, out]` for grouped, None where
    unavailable), whether a launch sees the rows and kernels whole (`whole`: no automatic mesh axis splits
    them), the products' `precision`, the gated product's elementwise `hook` (None where the activation has
    none, e.g. `Situ`), whether the call is grouped (`grouped`: expert-sorted rows in `moe.gated_experts`,
    measured in its own `(dtype, 'grouped')` cell), whether the gate and up projections pair (`paired`), and
    whether the down projection accepts a prologue (`down_prologue`)."""

    projections: Projections
    uniform: bool
    biased: bool
    rows: jax.ShapeDtypeStruct
    kernels: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct] | None
    whole: bool
    precision: PrecisionLike
    hook: Callable[[jax.Array, jax.Array], jax.Array] | None
    """The gated product's hook (`activations.gated_hook` of the activation and its limit), which the paired
    product's epilogue and the down product's prologue trace; None where the activation has no hook."""
    grouped: bool = False
    paired: bool = True
    down_prologue: bool = True
    group_sizes: jax.ShapeDtypeStruct | None = None
    gather: jax.ShapeDtypeStruct | None = None
    scatter: jax.ShapeDtypeStruct | None = None
    layer: jax.ShapeDtypeStruct | None = None

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype the products run at, the promotion `gated_mlp` runs (`promote_dtype`'s)."""
        return jnp.dtype(self.rows.dtype)

    @property
    def cell(self) -> tuple[str, jnp.dtype | tuple[jnp.dtype, str]]:
        """The cell a receipt measures (`selection.Celled`): the device's generation and the products'
        dtype (or `(dtype, 'grouped')` for grouped expert MLPs, whose receipts are measured apart)."""
        return device_generation(), (self.dtype, 'grouped') if self.grouped else self.dtype

    def launched(self) -> bool:
        """Whether the template launches the up product and the down product (`projections.launches` for
        dense, or the paired-epilogue and down-prologue grouped launches via `_grouped_launched` for grouped
        experts), asked only where the fused MLP is being chosen (its last requirement), so a call that runs
        its products apart reports no route of a product it never launches."""
        from dew.nn.kernels.selection import KernelRefused

        if self.kernels is None or self.hook is None:
            return False
        gate, down = self.kernels
        if self.grouped:
            return _grouped_launched(
                self.rows, gate, down, self.dtype, self.precision, self.hook, whole=self.whole,
                group_sizes=self.group_sizes, gather=self.gather, scatter=self.scatter, layer=self.layer)
        hidden = jax.ShapeDtypeStruct((self.rows.shape[0], gate.shape[-1]), self.dtype)
        try:
            launched = all(projections.launches(rows, kernel, self.dtype, self.precision, whole=self.whole)
                           for rows, kernel in ((self.rows, gate), (hidden, down)))
        except KernelRefused:
            return False
        if not launched:
            return False
        tile = jax.ShapeDtypeStruct((128, 128), self.dtype)
        return hook_lowers(self.hook, (tile, tile), gpu_target())

    def wide_launched(self) -> bool:
        """Whether the GEMM template launches the wide split-epilogue product (`wide.gated_product`) and
        plain down product (`projections.launches`) on this device, asked only where `wide` is chosen."""
        from dew.nn.kernels import generation, grouped_matmul, selection
        from dew.nn.kernels.selection import KernelRefused

        if self.grouped or self.kernels is None or self.hook is None or not self.whole:
            return False
        gate, down = self.kernels
        if gate.shape[-1] % wide.GROUP:
            return False
        saved = not _recomputes_gate_up()
        wide_kernel = jax.ShapeDtypeStruct((gate.shape[0], 2 * gate.shape[-1]), self.dtype)
        hidden = jax.ShapeDtypeStruct((self.rows.shape[0], gate.shape[-1]), self.dtype)
        epilogue = wide.Gated(self.hook, self.dtype, saved)
        with generation.routes(alone=True):
            try:
                call, product = gemm._gemm_call(
                    self.rows, wide_kernel, None, None, (), (), epilogue, (), (),
                    (self.dtype,) * (1 + saved), several=True, transpose_rhs=False, interpret=False,
                    precision=self.precision, dense=True, paired=None, first=None, gather=None,
                    scatter=None, summed_as=())
                case = grouped_matmul.grouped_case(*call, product=product, whole=self.whole)
                if selection.selected(grouped_matmul.GROUPED, case).name != grouped_matmul.MOSAIC.name:
                    return False
                if not projections.launches(hidden, down, self.dtype, self.precision, whole=self.whole):
                    return False
            except KernelRefused:
                return False
        tile = jax.ShapeDtypeStruct((128, 128), self.dtype)
        return hook_lowers(self.hook, (tile, tile), gpu_target())


def _recomputes_gate_up() -> bool:
    """Whether an enclosing remat rung recomputes the gate or up projection (`RECOMPUTED`)."""
    recomputed = RECOMPUTED.get()
    return Residual.GATE_PROJ in recomputed or Residual.UP_PROJ in recomputed


def _grouped_stack(matrix: jax.ShapeDtypeStruct, dtype: jnp.dtype,
                   layered: bool) -> jax.ShapeDtypeStruct | None:
    """Flatten a 4D stacked expert bank `[layers, experts, in, out]` to 3D when `layered`, or keep 3D."""
    if layered and matrix.ndim == 4:
        return jax.ShapeDtypeStruct((matrix.shape[0] * matrix.shape[1], *matrix.shape[2:]), dtype)
    return matrix if matrix.ndim == 3 else None


def _grouped_launched(rows: jax.ShapeDtypeStruct, gate: jax.ShapeDtypeStruct, down: jax.ShapeDtypeStruct,
                      dtype: jnp.dtype, precision: PrecisionLike,
                      hook: Callable[[jax.Array, jax.Array], jax.Array], *, whole: bool,
                      group_sizes: jax.ShapeDtypeStruct | None = None,
                      gather: jax.ShapeDtypeStruct | None = None,
                      scatter: jax.ShapeDtypeStruct | None = None,
                      layer: jax.ShapeDtypeStruct | None = None) -> bool:
    """Whether `grouped_matmul.GROUPED` selects Mosaic's fused launch (`MOSAIC`, never CUTLASS/XLA which
    form the down prologue in XLA) for both the paired gate+up epilogue and down-projection prologue,
    building each call through `gemm._gemm_call` with the actual group sizes, gather/scatter tables, and
    layer offset (row 12847)."""
    from dew.nn.kernels import generation, grouped_matmul, selection
    from dew.nn.kernels.selection import KernelRefused

    layered = layer is not None
    gate_3d = _grouped_stack(gate, dtype, layered)
    down_3d = _grouped_stack(down, dtype, layered)
    if gate_3d is None or down_3d is None:
        return False
    sizes = (jax.ShapeDtypeStruct((gate.shape[1 if layered and gate.ndim == 4 else 0],), jnp.int32)
             if group_sizes is None else jax.ShapeDtypeStruct(group_sizes.shape, jnp.int32))
    if sizes.ndim != 1:
        return False
    sorted_rows = rows.shape[0] if gather is None else gather.shape[0]
    hidden = jax.ShapeDtypeStruct((sorted_rows, gate_3d.shape[-1]), dtype)
    first = jax.ShapeDtypeStruct((), jnp.int32) if layered else None

    def selected_name(call, product) -> str:
        case = grouped_matmul.grouped_case(*call, product=product, whole=whole)
        return selection.selected(grouped_matmul.GROUPED, case).name

    def pair_name(lhs: jax.ShapeDtypeStruct, table: jax.ShapeDtypeStruct | None) -> str:
        call, product = gemm._gemm_call(
            lhs, gate_3d, sizes, None, (), (), activations.rounded_gated(hook, dtype), (), (), (dtype,) * 3,
            several=True, transpose_rhs=False, interpret=False, precision=precision, dense=False,
            paired=gate_3d, first=first, gather=table, scatter=None, summed_as=())
        return selected_name(call, product)

    def down_name(table: jax.ShapeDtypeStruct | None) -> str:
        call, product = gemm._gemm_call(
            hidden, down_3d, sizes, hook, (hidden,), (('rows', 'depth'),), None, (), (), (dtype,),
            several=False, transpose_rhs=False, interpret=False, precision=precision, dense=False,
            paired=None, first=first, gather=None, scatter=table, summed_as=())
        return selected_name(call, product)

    mosaic, xla = grouped_matmul.MOSAIC.name, grouped_matmul.XLA.name
    with generation.routes(alone=True):
        try:
            # Mirror `gemm._gemm`: only when no tabled launch takes `gather`/`scatter` (`XLA`), `gemm._gemm`
            # gathers/scatters around the untabled grouped launch; CUTLASS runs with an XLA-formed prologue.
            pair_picked = pair_name(rows, gather)
            if pair_picked != mosaic and (
                    pair_picked != xla or gather is None or pair_name(
                        jax.ShapeDtypeStruct((sorted_rows, rows.shape[-1]), dtype), None) != mosaic):
                return False
            down_picked = down_name(scatter)
            if down_picked != mosaic and (
                    down_picked != xla or scatter is None or down_name(None) != mosaic):
                return False
        except KernelRefused:
            return False
    return True


FUSED_SUPPORTS: tuple[Requirement[GatedCase], ...] = (
    *projection_requirements(lambda case: case.projections, "the fused MLP"),
    Requirement('uniform', lambda case: case.uniform,
                "the projections differ in dtype, precision or promotion", row=INHERENT, kind='handoff'),
    Requirement('biased', lambda case: not case.biased,
                "a projection adds a bias, which no hook of the fused MLP adds yet", row=FUSION_ROW,
                kind='handoff'),
    Requirement('paired', lambda case: case.paired,
                "the gate and up projections cannot run as one paired product", row=FUSION_ROW,
                kind='handoff'),
    Requirement('down_prologue', lambda case: case.down_prologue,
                "the down projection takes no prologue (an NVFP4 bank, a recipe taking the experts, or "
                "another layer stack)", row=FUSION_ROW, kind='handoff'),
    Requirement('hook', lambda case: case.hook is not None,
                "the gated activation has no elementwise hook to lower on the template", row=FUSION_ROW,
                kind='handoff'),
    Requirement('launched', lambda case: case.launched(),
                "no launch of the GEMM template takes the up or the down product here (a mesh axis splitting "
                "them, a CPU, a declared XLA cell), or its hooks the gated product's operations (an exact "
                "GELU's erf behind an optimization barrier)", row=FUSION_ROW, kind='handoff'),
)
"""What the fused MLP needs to stand in for the projections' own calls: each a handoff, so a call outside
them runs its projections apart even where the fused MLP is named."""

FUSED = Kernel[GatedCase]('fused', FUSED_SUPPORTS)
"""The three products on the template's hooks (`fused_products`)."""

WIDE_SUPPORTS: tuple[Requirement[GatedCase], ...] = (
    *projection_requirements(lambda case: case.projections, "the wide fused MLP"),
    Requirement('dense', lambda case: not case.grouped,
                "the wide split-epilogue product takes dense projections alone", row=FUSION_ROW,
                kind='handoff'),
    Requirement('uniform', lambda case: case.uniform,
                "the projections differ in dtype, precision or promotion", row=INHERENT, kind='handoff'),
    Requirement('biased', lambda case: not case.biased,
                "a projection adds a bias, which no hook of the fused MLP adds yet", row=FUSION_ROW,
                kind='handoff'),
    Requirement('aligned_hidden', lambda case: (
                    case.kernels is not None and case.kernels[0].shape[-1] % wide.GROUP == 0),
                f"the hidden width is not a multiple of {wide.GROUP} columns", row=FUSION_ROW,
                kind='handoff'),
    Requirement('hook', lambda case: case.hook is not None,
                "the gated activation has no elementwise hook to lower on the template", row=FUSION_ROW,
                kind='handoff'),
    Requirement('wide_launched', lambda case: case.wide_launched(),
                "no launch of the GEMM template stores the narrowed wide gated product or takes the down "
                "product here", row=FUSION_ROW, kind='handoff'),
)
"""What the wide split-epilogue MLP needs to stand in for the projections' own calls."""

WIDE = Kernel[GatedCase]('wide', WIDE_SUPPORTS)
"""One wide interleaved gate+up product with a narrowed split-epilogue and plain down (`wide_products`)."""

FUSED_CELLS: dict[tuple[str | None, jnp.dtype | tuple[jnp.dtype, str]], tuple[Kernel[GatedCase], str]] = {}
"""The fused MLP's declared cells, `(device generation, dtype)` for dense or `(device generation, (dtype,
'grouped'))` for grouped experts to the fused kernel and its receipt: none yet. Where the fused Pallas SiLU
product holds a cell (`norm_swiglu.SWIGLU_CELLS`), a cell here is declared only from the fused MLP's own
receipt on that generation."""

GATED_MLP: Op[GatedCase] = Op(Choice.GATED_MLP, (FUSED, WIDE, Kernel('apart', (), generic=True)),
                              lambda case: 'apart', declared=FUSED_CELLS)
"""The fused MLP's selector: the projections apart, or `fused`/`wide` where a receipt declares it on par or
faster (`FUSED_CELLS`) or the block names it, and its requirements hold."""


def gated_case(x: jax.Array, projections_: Sequence[nn.Module], activation: str,
               limit: float | None = None) -> GatedCase:
    """The `GatedCase` of a gated MLP's `projections_` (gate, up, down) over its input `x`, its gated product
    `activation`'s clamped at `limit`."""
    gate = projections_[0]
    found = projections_of(projections_)
    # A kernel a wrapper owns the products of (a served model's quantized kernel, `found.intercepted`) is
    # the wrapper's to read, never an array of the products' dtype.
    kernels = (tuple(meta.unbox(projection.get_variable('params', 'kernel')) for projection in projections_)
               if found.held and not found.intercepted else None)
    # The gate's and the down projection's kernels promote with the rows, as `gated_mlp` promotes them.
    dtype = canonicalize_dtype(x, *(() if kernels is None else (kernels[0], kernels[2])), dtype=gate.dtype)
    # Where the rows run per shard (`_placement`), the products are each shard's, of its rows and the
    # kernels gathered whole in the map.
    placement = None if kernels is None else _placement(x, projections_, kernels)
    rows = (x.size // x.shape[-1], x.shape[-1]) if placement is None else placement[2].shape
    return GatedCase(
        projections=found,
        uniform=len({(projection.dtype, projection.precision, projection.promote_dtype)
                     for projection in projections_}) == 1,
        biased=any(projection.use_bias for projection in projections_),
        rows=jax.ShapeDtypeStruct(rows, dtype),
        kernels=None if kernels is None else (jax.ShapeDtypeStruct(kernels[0].shape, dtype),
                                              jax.ShapeDtypeStruct(kernels[2].shape, dtype)),
        whole=placement is not None or not automatic_axes(),
        precision=gate.precision, hook=activations.gated_hook(activation, limit))


def _placement(x: jax.Array, projections_: Sequence[nn.Module], kernels: tuple[jax.Array, ...]):
    """Where a gated MLP of `x` through `projections_` (gate, up, down), of `kernels`, runs per shard
    (`projections.shard_placement`): its rows split alone on the mesh, each kernel gathered whole in the
    map from where the traced program's layout places it, as a projection's product maps its rows
    (`projections.mapped_shards`); None elsewhere."""
    return projections.shard_placement(x, kernels, tuple(map(projections.kernel_path, projections_)))


def fuses(projections_: Sequence[nn.Module], rows: jax.Array, activation: str,
          limit: float | None = None) -> TypeGuard[Sequence[Dense]]:
    """Whether a gated MLP's projections (gate, up, down) of `rows`, its gated product `activation`'s clamped
    at `limit`, run fused (`gated_mlp`), as the fused MLP's selector reports it (`GATED_MLP`), so each a
    bare `Dense` (its `dense` requirement), which the answer states of them for `gated_mlp`. Initialization
    creates each projection's parameters through its own call."""
    if projections_[0].is_initializing():
        return False
    return select(GATED_MLP, gated_case(rows, projections_, activation, limit)).kernel in ('fused', 'wide')


_GROUPED_PROJECTIONS = Projections(
    dense=True, dot_general=True, intercepted=False, held=True, recipe=False, observed=False)
"""`Projections` stand-in for grouped expert banks (`moe.gated_experts`), which hold no Flax `Dense` modules
and validate pairing, recipes, and down-projection prologue eligibility directly via `GatedCase`."""


def _shape_struct(value: jax.Array | jax.ShapeDtypeStruct | None,
                  dtype=None) -> jax.ShapeDtypeStruct | None:
    """Normalize an array or `ShapeDtypeStruct` operand descriptor for `GatedCase`."""
    if value is None:
        return None
    return jax.ShapeDtypeStruct(value.shape, value.dtype if dtype is None else dtype)


def grouped_case(rows: jax.Array | jax.ShapeDtypeStruct,
                 kernels: tuple[jax.Array | jax.ShapeDtypeStruct, jax.Array | jax.ShapeDtypeStruct] | None,
                 precision: PrecisionLike,
                 hook: Callable[[jax.Array, jax.Array], jax.Array] | None, *,
                 paired: bool = True, down_prologue: bool = True,
                 whole: bool | None = None,
                 group_sizes: jax.Array | jax.ShapeDtypeStruct | None = None,
                 gather: jax.Array | jax.ShapeDtypeStruct | None = None,
                 scatter: jax.Array | jax.ShapeDtypeStruct | None = None,
                 layer: int | jax.Array | jax.ShapeDtypeStruct | None = None) -> GatedCase:
    """The `GatedCase` of a grouped gated expert MLP (`moe.gated_experts`, row 12847) over `rows` in its
    compute dtype and `(gate, down)` expert kernel shapes (`[experts, in, hidden]` / `[experts, hidden, out]`
    or stacked `[layers, experts, ...]` with `layer`, None where unavailable), plus optional `group_sizes`,
    `gather`, `scatter`, and `layer` launch descriptors."""
    dtype = jnp.dtype(rows.dtype)
    flat_rows = jax.ShapeDtypeStruct((math.prod(rows.shape[:-1]), rows.shape[-1]), dtype)
    typed_kernels = (None if kernels is None else (
        jax.ShapeDtypeStruct(kernels[0].shape, dtype),
        jax.ShapeDtypeStruct(kernels[1].shape, dtype)))
    return GatedCase(
        projections=_GROUPED_PROJECTIONS,
        uniform=True,
        biased=False,
        rows=flat_rows,
        kernels=typed_kernels,
        whole=not automatic_axes() if whole is None else whole,
        precision=precision,
        hook=hook,
        grouped=True,
        paired=paired,
        down_prologue=down_prologue,
        group_sizes=_shape_struct(group_sizes, jnp.int32),
        gather=_shape_struct(gather, jnp.int32),
        scatter=_shape_struct(scatter, jnp.int32),
        layer=None if layer is None else jax.ShapeDtypeStruct((), jnp.int32))


def fuses_grouped(rows: jax.Array | jax.ShapeDtypeStruct,
                  kernels: tuple[jax.Array | jax.ShapeDtypeStruct, jax.Array | jax.ShapeDtypeStruct] | None,
                  precision: PrecisionLike,
                  hook: Callable[[jax.Array, jax.Array], jax.Array] | None, *,
                  paired: bool = True, down_prologue: bool = True,
                  whole: bool | None = None,
                  group_sizes: jax.Array | jax.ShapeDtypeStruct | None = None,
                  gather: jax.Array | jax.ShapeDtypeStruct | None = None,
                  scatter: jax.Array | jax.ShapeDtypeStruct | None = None,
                  layer: int | jax.Array | jax.ShapeDtypeStruct | None = None) -> bool:
    """Whether a grouped gated expert MLP (`moe.gated_experts`) runs fused on the GEMM template's hooks, as
    `GATED_MLP` selects it."""
    return select(GATED_MLP, grouped_case(
        rows, kernels, precision, hook, paired=paired, down_prologue=down_prologue, whole=whole,
        group_sizes=group_sizes, gather=gather, scatter=scatter, layer=layer)).kernel == 'fused'


def gated_mlp(x: jax.Array, projections_: Sequence[nn.Dense], activation: str, *,
              limit: float | None = None, names: tuple[str, str, str] | None = None) -> jax.Array:
    """A gated MLP's output of `x` through its projections (gate, up, down), fused (`fused_products` or
    `wide_products` per `GATED_MLP`), each kernel read and cast as its projection's call casts it."""
    form = selected(GATED_MLP, gated_case(x, projections_, activation, limit)).name
    runner = wide_products if form == 'wide' else fused_products
    promoted = [_promoted(projection, x, meta.unbox(projection.get_variable('params', 'kernel')), None)
                for projection in projections_]
    inputs = promoted[0][0]
    kernels = tuple(kernel for _, kernel, _ in promoted)

    def products(inputs, gate, up, down):
        out = runner(inputs.reshape(-1, inputs.shape[-1]), (gate, up, down), activation, limit=limit,
                     precision=projections_[0].precision, names=names)
        return out.reshape(*inputs.shape[:-1], out.shape[-1])

    # On a mesh splitting its rows alone, each device's rows are its own products' (`_placement`, as
    # `fuses` read them), its kernels gathered whole there, so each one's gradient leaves the map
    # reduce-scattered back to its placement.
    placement = _placement(x, projections_, kernels)
    if placement is None:
        return products(inputs, *kernels)
    split, gathered, _ = placement
    return on_shards(products, split, (inputs, *kernels), gathered=gathered)


def wide_products(rows: jax.Array, kernels: tuple[jax.Array, jax.Array, jax.Array], activation: str, *,
                  limit: float | None = None, precision=None, names: tuple[str, str, str] | None = None,
                  interpret_on_cpu: bool = False) -> jax.Array:
    """`rows`' output through a gated MLP's kernels (gate, up, down; `[in, hidden]`, `[in, hidden]`,
    `[hidden, out]`), using one wide interleaved gate+up product with a narrowed split-epilogue
    (`wide.gated_product`, formulation (C), docs/design/kernel-templates.md §3) and a plain down product
    (`gemm.gemm`). Recomputes the gate+up residual (`saved=False`) when an enclosing remat rung names
    `Residual.GATE_PROJ` or `Residual.UP_PROJ` (`RECOMPUTED`)."""
    dtype = rows.dtype
    hook = activations.gated_hook(activation, limit)
    if hook is None:
        raise ValueError(f"gated activation {activation!r} has no elementwise hook for wide_products")
    saved = not _recomputes_gate_up()
    name = names[0] if names is not None else Residual.GATE_PROJ
    combined = wide.interleaved(kernels[0], kernels[1])
    result = wide.gated_product(rows, combined, hook, out_dtype=dtype, saved=saved, name=str(name),
                                precision=precision, interpret_on_cpu=interpret_on_cpu)
    hidden = result[1] if isinstance(result, tuple) else result
    out = gemm.gemm(hidden, kernels[2], out_dtype=dtype, precision=precision,
                    interpret_on_cpu=interpret_on_cpu)
    return out if names is None else checkpoint_name(out, names[2])


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
    gate, up, hidden = gemm.gemm(rows, kernels[0], paired=kernels[1],
                                 epilogue=activations.rounded_gated(hook, dtype), out_dtype=dtype,
                                 out_dtypes=(dtype,) * 3, precision=precision,
                                 interpret_on_cpu=interpret_on_cpu)
    if names is not None:
        gate, up = checkpoint_name(gate, names[0]), checkpoint_name(up, names[1])
    out = gemm.gemm(gate, kernels[2], prologue=hook, prologue_operands=((up, ('rows', 'depth')),),
                    applied=hidden, out_dtype=dtype, precision=precision, interpret_on_cpu=interpret_on_cpu)
    return out if names is None else checkpoint_name(out, names[2])
