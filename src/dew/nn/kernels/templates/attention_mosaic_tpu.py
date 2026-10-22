"""The attention template's TPU mainloop (Pallas Mosaic TPU), the design's first TPU rung
(docs/design/kernel-templates.md §6): the flash structure jax's and splash's TPU kernels hold, with the
template's algebra (`attention_algebra`) between the two products.

One program a row, a head, a query block and a key block, the key blocks innermost and sequential: the
softmax's running state (`algebra.Partial`) lives in VMEM scratch across them, starts at the first and
finishes at the last. Each key block issues S = Q·Kᵀ on the MXU, hands it to the algebra (`modify`,
`peak`, `weights`, `total`, `rescale`, `narrow`) and issues P·V. A causal call skips the blocks above its
query block's diagonal, their keys' index held at the diagonal's so the pipeline fetches nothing new, and
masks only the blocks the diagonal or the keys' padding cuts. The kernel states its own scoped VMEM limit,
the device's VMEM (`TpuInfo.vmem_capacity_bytes`), so what it may hold is known by construction.

The forward over full and causal calls, no hooks: the rung's first piece."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from dew.nn.kernels.templates import attention_algebra as algebra
from dew.nn.kernels.templates.attention import NATURAL, Hooks, KeySpans, padded_call
from dew.nn.precision import float32_products

LANES = 128
"""A TPU vector register's lanes: a key block and a head width are whole multiples of it."""


@dataclasses.dataclass(frozen=True)
class Tiles:
    """A launch's query block (`block_q`, a multiple of 8, a vector register's sublanes) and key block
    (`block_k`, a multiple of `LANES`)."""

    block_q: int = 512
    block_k: int = 512

    def __post_init__(self):
        if self.block_q % 8 or self.block_k % LANES:
            raise ValueError(f"{self}: a query block of whole sublanes and a key block of whole lanes")


TILES = Tiles()
"""The tiles a call takes where it names none."""


def forward(query: jax.Array, key: jax.Array, value: jax.Array, spans: KeySpans, hooks: Hooks | None = None,
            *, scale: float | None = None, natural: bool = NATURAL, tiles: Tiles = TILES,
            interpret: bool = False) -> jax.Array:
    """The template's forward on this mainloop: `query` `[B, Q, H, D]` over `key` and `value`
    `[B, K, Hk, D]` and the keys `spans` leaves each query (full or causal), as
    `dew.nn.kernels.templates.attention.reference` computes it. `natural` keeps the logits natural
    (`algebra.Forward.base2`); `interpret` runs it on the CPU (Pallas TPU's interpret mode)."""
    if hooks is not None and hooks != Hooks():
        raise NotImplementedError("the TPU mainloop takes no hooks yet")
    if spans.form not in ("full", "causal"):
        raise NotImplementedError(f"the TPU mainloop takes full and causal calls, not {spans.form!r} ones")
    value_width = value.shape[-1]
    call = padded_call(query, key, value, spans, Hooks(), max(tiles.block_q, tiles.block_k))
    spec = algebra.Forward(spans.form, spans.keys, ids=False,
                           scale=query.shape[3] ** -0.5 if scale is None else scale, natural=natural,
                           wide=jnp.dtype(query.dtype).itemsize >= 4)
    # Heads ahead of tokens, so a block's two minor axes are its tokens and a head width of whole lanes.
    query, key, value = (jnp.pad(jnp.swapaxes(array, 1, 2), ((0, 0),) * 3 + ((0, -array.shape[3] % LANES),))
                         for array in (call.query, call.key, call.value))
    out = _forward(query, key, value, spec=spec, tiles=tiles, interpret=interpret)
    return jnp.swapaxes(out, 1, 2)[:, :call.queries, :, :value_width]


def _kernel(q_ref, k_ref, v_ref, out_ref, acc_ref, m_ref, total_ref, *, spec: algebra.Forward, tiles: Tiles,
            blocks: int):
    block_q, block_k = tiles.block_q, tiles.block_k
    query_block, key_block = pl.program_id(2), pl.program_id(3)
    first_query, first_key = query_block * block_q, key_block * block_k
    causal = spec.form == "causal"
    padded = bool(spec.keys % block_k)

    @pl.when(key_block == 0)
    def _():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        total_ref[...] = jnp.zeros(total_ref.shape, jnp.float32)

    def step(*, masked: bool):
        query = q_ref[...]
        # Products of the operands' own dtype at its policy's precision: fp32 ones are fp32 products
        # (HIGHEST, several MXU passes), where the TPU's default takes one bf16 pass.
        precision = float32_products(query.dtype)
        s = jnp.dot(query, k_ref[...].T, precision=precision, preferred_element_type=jnp.float32)
        coords = algebra.Coords(first_query + lax.broadcasted_iota(jnp.int32, s.shape, 0),
                                first_key + lax.broadcasted_iota(jnp.int32, s.shape, 1))
        logits = algebra.modify(spec, s, coords, masked=masked)
        m, alpha = algebra.peak(spec, m_ref[...], logits)
        weights = algebra.weights(spec, logits, m)
        total_ref[...] = algebra.total(total_ref[...], alpha, weights)
        value = v_ref[...]
        acc_ref[...] = algebra.rescale(acc_ref[...], alpha) + jnp.dot(
            algebra.narrow(weights, value.dtype), value, precision=precision,
            preferred_element_type=jnp.float32)
        m_ref[...] = m

    # A block every query of the query block sees whole, before the diagonal and the keys' padding, takes
    # no mask; the rest that any query sees are masked.
    last_key = first_key + block_k - 1
    whole = (last_key <= first_query) if causal else True
    if padded:
        whole = jnp.logical_and(whole, key_block < blocks - 1)
    seen = (first_key <= first_query + block_q - 1) if causal else True
    pl.when(jnp.logical_and(seen, whole))(functools.partial(step, masked=False))
    pl.when(jnp.logical_and(seen, jnp.logical_not(whole)))(functools.partial(step, masked=True))

    @pl.when(key_block == blocks - 1)
    def _():
        state = algebra.Partial(acc_ref[...], m_ref[...], total_ref[...])
        out, _, _ = algebra.finish(spec, state, initial=False)
        out_ref[...] = out.astype(out_ref.dtype)


def _forward(query, key, value, *, spec: algebra.Forward, tiles: Tiles, interpret: bool):
    batch, heads, queries, width = query.shape
    kv_heads, keys, value_width = key.shape[1], key.shape[2], value.shape[-1]
    group = heads // kv_heads
    block_q, block_k = tiles.block_q, tiles.block_k
    blocks = keys // block_k

    def key_index(b, h, i, j):
        # Past a causal query block's diagonal no block is read: hold the diagonal's, fetching nothing.
        if spec.form == "causal":
            j = jnp.minimum(j, lax.div(i * block_q + block_q - 1, block_k))
        return b, h // group, j, 0

    vmem = None if interpret else pltpu.get_tpu_info().vmem_capacity_bytes
    return pl.pallas_call(
        functools.partial(_kernel, spec=spec, tiles=tiles, blocks=blocks),
        grid=(batch, heads, queries // block_q, blocks),
        in_specs=[pl.BlockSpec((None, None, block_q, width), lambda b, h, i, j: (b, h, i, 0)),
                  pl.BlockSpec((None, None, block_k, width), key_index),
                  pl.BlockSpec((None, None, block_k, value_width), key_index)],
        out_specs=pl.BlockSpec((None, None, block_q, value_width), lambda b, h, i, j: (b, h, i, 0)),
        out_shape=jax.ShapeDtypeStruct((batch, heads, queries, value_width), query.dtype),
        scratch_shapes=[pltpu.VMEM((block_q, value_width), jnp.float32), pltpu.VMEM((block_q,), jnp.float32),
                        pltpu.VMEM((block_q,), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"), vmem_limit_bytes=vmem),
        interpret=interpret, name="attention_forward_tpu",
    )(query, key, value)
