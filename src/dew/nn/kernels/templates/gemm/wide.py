"""Dew: a gated MLP's gate and up projections as one product over one wide matrix
(docs/design/kernel-templates.md, Appendix C row 18's formulation (C)): the two matrices' columns
interleaved a `GROUP` at a time (`interleaved`), so each output tile's accumulator holds its gate and up
columns in the same threads' registers, and the epilogue writes only their gated product (`gated`), half
the tile's width: one accumulator where the paired product holds two, and the hidden activation the one
output a forward writes.

`GROUP` is the least run whose gate and up halves every GPU family's accumulator holds in one thread, in whole
tiles of its layout's outermost column tile (WGMMA's and TCGEN05's 8; mma.sync's 8, 16 or 32 as its warps
split a share's columns, `mosaic_sm80.MmaIssuer.specialized`), and the most a tcgen05 epilogue's 64-column
read-out holds a pair of: the epilogue splits its accumulator by selecting registers
(`inline_forms.split_groups`), no instruction, and stores the narrowed output as computed
(`quantized.Form.narrowed`, `stores.Output.narrowed`).
"""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.kernels import launches
from dew.nn.kernels.templates.gemm import quantized

GROUP = 32
"""The columns of each run of gate or up columns in a wide matrix (the module's docstring)."""

UNIT = 2 * GROUP
"""The columns a tile, or a tcgen05 epilogue's read-out, holds whole pairs of runs in."""


def interleaved(gate: jax.Array, up: jax.Array) -> jax.Array:
    """`gate` and `up` (`[..., in, hidden]` each, `hidden` a multiple of `GROUP`) as one wide matrix
    `[..., in, 2 * hidden]`, their columns alternating a `GROUP` at a time, gate's first."""
    if gate.shape != up.shape or gate.shape[-1] % GROUP:
        raise ValueError(f"a wide matrix interleaves two matrices of one shape whose columns are whole "
                         f"{GROUP}-column runs, not {gate.shape} and {up.shape}")
    *lead, depth, hidden = gate.shape
    runs = (*lead, depth, hidden // GROUP, 1, GROUP)
    return jnp.concatenate((gate.reshape(runs), up.reshape(runs)), axis=-2).reshape(*lead, depth, 2 * hidden)


def deinterleaved(wide: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A wide matrix's gate and up matrices (`interleaved`'s inverse), its columns' runs apart."""
    *lead, depth, width = wide.shape
    if width % UNIT:
        raise ValueError(f"a wide matrix's {width} columns are no whole pairs of {GROUP}-column runs")
    runs = wide.reshape(*lead, depth, width // UNIT, 2, GROUP)
    halves = (*lead, depth, width // 2)
    return runs[..., 0, :].reshape(halves), runs[..., 1, :].reshape(halves)


def split(acc: jax.Array) -> tuple[jax.Array, jax.Array]:
    """An accumulator tile of a wide product (`[rows, columns]`, whole `UNIT`s) as its gate's and up's
    columns (`[rows, columns / 2]` each): in a compiled GPU epilogue the registers holding each run selected
    where they lie (`inline_forms.split_groups`), elsewhere (interpreted, XLA's form of the product) its runs'
    view."""
    from dew.nn.kernels.templates import inline_forms

    layout = quantized.epilogue_register_layout() if launches.interpreting() is False else None
    if layout is not None:
        return inline_forms.split_groups(acc, GROUP, layout)
    return deinterleaved(acc)


@dataclasses.dataclass(frozen=True)
class Gated:
    """The epilogue of a wide product (`gated_product`): its accumulator rounded to `dtype`, as the paired
    form rounds the gate's and up's products (`gated_mlp.fused_products`), and their gated product
    `hook(gate, up)` (`activations.gated_hook`) its one output, half its tile's width; `saved`, the rounded
    product first, whole, a reverse pass's residual alone (`hooks.Residualized`: written where a derivative
    is taken, `unsaved` elsewhere). It states its outputs' forms (`out_forms`), which a launch stores them
    in."""

    hook: Callable
    dtype: jnp.dtype
    saved: bool = False

    def __call__(self, acc: jax.Array) -> tuple[jax.Array, ...]:
        rounded = acc.astype(self.dtype)
        hidden = self.hook(*split(rounded)).astype(self.dtype)
        return (rounded, hidden) if self.saved else (hidden,)

    @property
    def residuals(self) -> int:
        """Its outputs a reverse pass alone reads (`hooks.Residualized`): the rounded product, where saved."""
        return int(self.saved)

    def unsaved(self) -> "Gated":
        """The epilogue where no derivative is taken: its gated product alone."""
        return dataclasses.replace(self, saved=False)

    @property
    def out_forms(self) -> tuple:
        """Each output's form (`Hooks.out_forms`): the rounded product a tile of its own, the gated product
        narrowed to half its tile (`quantized.Form.narrowed`)."""
        return (*((None,) if self.saved else ()), quantized.Form(2, narrowed=GROUP))


def gated_product(rows: jax.Array, wide: jax.Array, hook: Callable, *, out_dtype, saved: bool = False,
                  name: str = "gate_up", precision=None, interpret_on_cpu: bool = False) -> jax.Array:
    """`rows` (`[tokens, in]`) through a gated MLP's gate and up projections held as one wide matrix (`[in, 2
    * hidden]`, their columns interleaved a `GROUP` at a time, `interleaved`): one product, one accumulator,
    whose epilogue writes their gated product `hook(gate, up)` alone (`[tokens, hidden]` in `out_dtype`, each
    projection rounded to it first, as the paired form rounds them, `gated_mlp.fused_products`;
    `activations.gated_hook`), half its tile's width (`Gated`; formulation (C),
    docs/design/kernel-templates.md §3). `saved`: the rounded wide product (`[tokens, 2 * hidden]`, gate and
    up interleaved) stored too where a derivative is taken, never returned (`Gated.residuals`), the reverse
    pass's residual under `name` (`gemm.gemm`'s `named`, a remat policy's label), from which it forms the
    gate's and up's cotangents (the hook's VJP) for one input gradient and one kernel gradient of the wide
    matrix; an inference's launch stores the gated product alone. Unsaved, the reverse pass forms the
    product again.
    Its route is the dense product's (`grouped_matmul.GROUPED`): the template's launch where it stores the
    narrowed output (`quantized.forms_refusal`), XLA's otherwise, the same function of the product."""
    from dew.nn.kernels import gemm

    dtype = jnp.dtype(out_dtype)
    named = (name, None) if saved else ()
    outs = gemm.gemm(rows, wide, None, epilogue=Gated(hook, dtype, saved), out_dtype=dtype,
                     out_dtypes=(dtype,) * (1 + saved), precision=precision,
                     interpret_on_cpu=interpret_on_cpu, named=named)
    return outs[0]
