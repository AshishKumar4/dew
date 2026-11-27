"""A frozen bfloat16 matrix's lossless 13-bit window's product (`window_product`, row 5387): its words and
meta halves (`formats.Bf16Window`) streamed into the Mosaic GEMM mainloop and each stage decoded in place
into the bfloat16 weights it stores, bit for bit (`MosaicWindowPackedGmm`, `packed.weight_format('window')`),
where a receipt finds that faster than the decoded matrix's GEMM; else the decoded matrix's XLA dot."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.core import Tracer

from dew.nn.derivatives import KernelGlue, contracted
from dew.nn.kernels import gemm, generation, grouped_quantized, selection, xla_products
from dew.nn.kernels.formats import Bf16Window
from dew.nn.sharding import automatic_axes

WINDOW_ROW = "ledger 5387"


@dataclasses.dataclass(frozen=True)
class MosaicWindowPackedGmm(grouped_quantized.MosaicWordPackedGmm):
    """`MosaicWordPackedGmm` of a lossless bfloat16 window (`formats.Bf16Window`, row 5387): its stages
    decoded in place, a staged matrix alone (its meta halves are its columns')."""
    DEFAULT_FMT: ClassVar[str] = "window"


WINDOW = gemm.word_packed_op("window", WINDOW_ROW, "M-NG-49")


def admitted(window: Bf16Window) -> Bf16Window:
    """`window` where its form is this Dew's encoder's, checked where it is concrete (`Bf16Window.checked`;
    a traced one's is row 13700's): the one entry every product of a window passes, its own launch's and the
    decoded matrix's (`gemm.gemm`'s fallback) alike."""
    if not isinstance(window.form, Tracer):
        window.checked()
    return window


def window_product(lhs: jax.Array, window: Bf16Window, group_sizes: jax.Array | None = None, *,
                   out_dtype=None, interpret_on_cpu: bool = False) -> jax.Array:
    """`lhs @ window` for a lossless bfloat16 window (`formats.Bf16Window`, row 5387: `[depth, width]`
    dense, or `[groups, depth, width]` grouped by `group_sizes`): `lhs` rounded to bfloat16, each product of
    a decoded weight (its stored bits again) exact, summed in fp32 and rounded once to `out_dtype` (`lhs`'s
    where None); decoded a stage at a time in the GEMM mainloop where the selector takes it (`WINDOW`), else
    the decoded matrix's XLA dot (`xla_products.decoded_dot`). Frozen: its gradient is the rows' alone."""
    admitted(window)
    dense = group_sizes is None
    if (dense and len(window.shape) != 2) or (not dense and len(window.shape) != 3):
        raise ValueError(f"window_product weight rank mismatch for dense={dense}: {window.shape}")
    rows = lhs.astype(jnp.bfloat16)
    # The launch's bank is grouped: a dense window its one group.
    words, meta = (jnp.asarray(plane)[None] if dense else jnp.asarray(plane)
                   for plane in (window.words, window.meta))
    sizes = jnp.asarray([lhs.shape[0]], jnp.int32) if dense else group_sizes.astype(jnp.int32)
    launch = MosaicWindowPackedGmm()

    def _spec(operand):
        return jax.ShapeDtypeStruct(jnp.shape(operand), jnp.result_type(operand))

    bank_spec = jax.tree.map(_spec, Bf16Window(words, meta, window.form))
    bound = launch.bind(_spec(rows), bank_spec, _spec(sizes), (), (), None, None, None, out_dtype="float32",
                        trans_rhs=False, interpret=interpret_on_cpu, dense=dense, exact_sums=False)
    with xla_products.interpreted_context(interpret_on_cpu):
        case = gemm.WordPackedCase(functools.partial(gemm._planned, launch, bound), not automatic_axes())
        mosaic = selection.select(WINDOW, case).kernel == "mosaic"
    if mosaic:
        form = jnp.asarray(window.form)
        rules = gemm._packed_weight_rules(launch, lambda w, m: Bf16Window(w, m, form), interpret_on_cpu,
                                          dense=dense, finite=False)
        product = contracted("test_a_window_product_holds_every_order")(
            generation.linear_tangent(*rules, live_cotangents=True, kind=KernelGlue()))
        operands = (words, sizes, meta)
    else:
        product = xla_products.decoded_dot
        operands = (window.decoded(), group_sizes)
    summed = gemm._interpreted(product, interpret_on_cpu, rows, *operands)
    return summed.astype(lhs.dtype if out_dtype is None else out_dtype)
