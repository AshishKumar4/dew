"""The recurrence template's algebra (docs/design/kernel-templates.md §3): the chunked form of the update as
named, pure functions on values, which every implementation of the chunked op calls and none rewrites. This
rung covers `S' = T S + W` with T a diagonal decay times the delta rule's erase, E the identity, and a chunk's
transition `(T, W)` dense (`transition`); the value-side decay E and the factored transitions follow with the
families that need them. A token updates and reads, per head over a state `S` `[Dk, Dv]`,

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q

(`recurrence.update` is its one-token form). Within a chunk of C tokens:

- `decays(g, documents) -> (grown, pairs)`: the decay from the chunk's start `exp(gc)`, the log decay `gc`
  cumulated within the chunk, and each pair of positions' decay `exp(gc[s] - gc[t])`, every exponent
  compensated;
- `intra(k, kb, pairs) -> A`: the delta rule's strictly lower key-key system, `-(beta k k^T)` weighed by the
  pairs;
- `factor(A) -> L`: its triangular inverse `(I - A)^-1` (WY/UT), skipped where nothing is corrected;
- `apply(L, kb, vb, entering) -> (W, U)`: the corrected write's two products, `L (beta k exp(gc))` and `L
  (beta v)`;
- `carry(S, step) -> (S', (S, v_new))`: one chunk across the memory, the corrected values `v_new = U - W S`
  and the write `grown[-1] S + (k exp(gc[-1] - gc))^T v_new`;
- `readout(q, entering, entered, k, pairs, v_new) -> o`: the memory entered read through the decayed
  queries, plus the chunk's own pairs;
- `compose(first, second)`: two spans' transitions `S -> T S + W` (`Span`) as one, associatively, for a scan
  over shards or chunks run in parallel; `entering(span, axis)` the memory each shard of a sequence enters its
  span with, their exclusive prefix across the mesh axis.

Each works over any leading dimensions: the portable math (`recurrence_chunk.chunk_delta_rule`) calls them on
`[B, H, NC, ...]` arrays, and a kernel's body on one chunk's tiles. Packed `documents` restart the decays at
each document's first position (varlen).

The algebra declares its products and issues none (the design's layering): every product goes through the
caller's `issue(name, left, right)`, the mainloop's, by its name in `PRODUCTS`, which says what it multiplies;
the portable math's issuer is XLA's (`portable`), a kernel's its own dot."""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from dew.nn.kernels.generation import linear_map
from dew.nn.kernels.templates.attention_algebra import power2
from dew.nn.kernels.templates.product import Exact, Transposed
from dew.nn.precision import compensated_add, float32_einsum, float32_matmul

from . import recurrence

Issue = Callable[[str, "Operand", "Operand"], jax.Array]
"""A mainloop's product: `issue(name, left, right)` is `left @ right` over the last two axes, the product
`PRODUCTS[name]` declares, in the mainloop's own dot."""

PRODUCTS = {
    "intra": "the write-gated keys against the keys, `kb k^T`, under a decay a head (`intra`)",
    "scores": "the queries against the keys, `q k^T`, under a decay a head (`readout`)",
    "merge": "two diagonal blocks' inverses joined across the block below them (`factor`)",
    "apply_w": "the inverse against the decayed write-gated keys, `L (kb exp(gc))` (`apply`)",
    "apply_u": "the inverse against the write-gated values, `L (vb)` (`apply`)",
    "corrected": "the write's keys against the memory entering, `W S` (`carry`)",
    "write": "the decayed keys against the corrected values, `(k last)^T v_new` (`carry`)",
    "memory": "the decayed queries against the memory entered, `(q exp(gc)) S` (`readout`)",
    "local": "the chunk's scores against its corrected values, `A_qk v_new` (`readout`)",
    "compose": "a transition against another's, `T2 T1` and `T2 W1` (`compose`)",
    "local_t": "the keys against the queries, `k q^T`, the scores' transpose (`local_gradient`)",
    "local_dv": "the scores' transpose against the output's gradient, `A^T do` (`local_gradient`)",
    "reads": "the queries' transpose against the decayed output gradient, `q^T (do exp(gc))`",
    "values_dv": "the keys against the leaving memory's gradient, `k dS` (`reverse_carried`)",
    "weights_dv": "the weights' transpose against the values' gradient, `w^T dv` (`reverse_carried`)",
    "values_ds": "the output's gradient against the corrected values, `do v^T` (`scores_gradient`)",
    "memory_dq": "the output's gradient against the entered memory's transpose, `do S^T`",
    "memory_dk": "the corrected values against the leaving memory's gradient, `v dS^T`",
    "memory_dw": "the values' gradient against the entered memory's transpose, `dv S^T`",
    "memory_trace": "the entered memory's transpose against the leaving one's gradient, `S^T dS`, whose "
                    "diagonal sums `sum(S dS)` (a mainloop's `memory_gradients` launch)",
    "scores_dq": "the scores' gradient against the keys, `ds k`, and its transpose against the queries",
    "inverse_t": "the inverse's transpose against a gradient, `L^T dX` (`system_gradient`)",
    "inverse_dx": "a gradient against a weighted operand's transpose, `dw (k b exp(gc))^T`",
    "inverse_l": "the inverse's gradient through the inverse, `L^T dX L^T`",
    "keys_dl": "the system's gradient against the keys, `dL k`, and the keys against themselves, `k k^T`",
    "keys_dlt": "the system's gradient, its rows scaled by the write gates, transposed against the keys",
    "summed": "a decay's range terms against a triangle or the identity, each token's sum (`summed_back`)",
}
"""The algebra's products by name. A decay a key dimension weighs each term of `intra`'s and `scores`'
contractions (`paired`), which a mainloop issues as a product of the factored operands within the range its
decay's bound keeps exact, or as the weighed sum."""


def portable(name: str, left, right):
    """XLA's product, the portable math's issuer: `left @ right`, float32 products in float32 on a GPU, as
    torch's are, not TF32, summed as Dew sums a CPU's float32 products (`dew.nn.precision.float32_matmul`,
    which `accumulated`); the portable math takes its operands in at least float32
    (`recurrence_chunk.chunked`), a `Transposed` one transposed."""
    del name
    left, right = (jnp.swapaxes(x.value, -1, -2) if isinstance(x, Transposed) else
                   x.value if isinstance(x, Exact) else x for x in (left, right))
    return float32_matmul(left, right)


def _cumulated(g, documents=None):
    """`g` `[..., C, F]` cumulated over its C positions as a compensated pair `(hi, lo)` (`compensated_add`),
    restarting at each of packed `documents`' first positions."""
    zeros = jnp.zeros_like(g)
    if documents is None:
        return jax.lax.associative_scan(compensated_add, (g, zeros), axis=g.ndim - 2)

    def segmented(left, right):
        (left_start, *left_sum), (right_start, *right_sum) = left, right
        total = compensated_add(left_sum, right_sum)
        restarted = (jnp.where(right_start, kept, summed)
                     for kept, summed in zip(right_sum, total, strict=True))
        return left_start | right_start, *restarted

    changed = jnp.concatenate([jnp.ones_like(documents[..., :1, :], bool),
                               documents[..., 1:, :] != documents[..., :-1, :]], axis=-2)
    _, hi, lo = jax.lax.associative_scan(segmented, (jnp.broadcast_to(changed, g.shape), g, zeros),
                                         axis=g.ndim - 2)
    return hi, lo


@jax.custom_jvp
def decays(g, documents=None):
    """Cumulate per-chunk log decays and build the pairwise decay between positions.

    `g` is `[..., C, F]`: the C positions of a chunk and F decay channels. Returns the decay from the chunk's
    start `grown = exp(gc)`, `gc` the inclusive cumulative sum over C, `[..., C, F]`, and `decay[..., s, t, f]
    = exp(gc[s, f] - gc[t, f])` for s >= t, zero above the diagonal, `[..., C, C, F]`.

    The references subtract two fp32 cumulative sums, so every exponent carries the rounding of the chunk's
    whole accumulated magnitude, which exp turns into a relative error on every decay. Here the cumulative sum
    is compensated (`compensated_add`) and the difference is a `(whole, part)` pair too, exp taken of each, so
    a decay is as precise as its own exponent: the difference rounded to one fp32 lost up to half its ulp,
    which put the gradients of the decay and the write gate under aligned keys 1.6 to 2.1 times the
    token-by-token recurrence's distance from float64 on an A100; `grown` takes both halves too
    (`grown_from`), where exp of the rounded sum carried its rounding into the memory's decay and put the
    state's gradient near twice the reference's distance from float64. On a per-key-decay model's tiny fixture
    that error dominated the gradient: over 16 orderings of the residual stream (an exact symmetry, so only
    the rounding moves) the updated logits sat at an RMS 2.04 times the reference's distance from float64 with
    plain sums on CPU and 1.83 on an RTX 4080, and 0.92 and 0.72 with these, against the reference's own 0.98.
    Summing each pair's range instead (a segment sum) is as exact and made a per-key-decay layer's forward and
    backward 25% slower; this costs about 1% (on an RTX 4080, a 32-head per-key-decay layer over 1024 tokens
    47.6 against 48.0 ms, a 0.8B model's per-head-decay layer over 4096 16.3 against 16.5).

    The gradient is the pairs' own (`_decays_tangent`): differentiating the difference of two prefixes sums a
    pair's cotangent into its two ends with opposite signs, and the decay gradient, a row sum less a column
    sum of the chunk's pairs, cancelled to fp32 noise wherever the prefixes ran long.

    With packed `documents` `[..., C, 1]` (each position's document within its chunk), the sum restarts at
    each document's first position and pairs of positions in different documents weigh zero, masked before
    exp: a document's decays are its own tokens' alone, the ones it computes run by itself."""
    hi, lo = _cumulated(g, documents)
    chunk_size = hi.shape[-2]
    inclusive = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_))[..., None]
    if documents is not None:
        inclusive = inclusive & (documents[..., :, None, :] == documents[..., None, :, :])
    whole, part = compensated_add((hi[..., :, None, :], lo[..., :, None, :]),
                                  (-hi[..., None, :, :], -lo[..., None, :, :]))
    # Masked before exp, as the references do. The unused positive
    # differences can overflow, and an outer where alone leaves 0 * inf in
    # the decay gradient.
    whole, part = (jnp.where(inclusive, x, 0.0) for x in (whole, part))
    return grown_from(hi, lo, jnp.exp), jnp.where(inclusive, jnp.exp(whole) * jnp.exp(part), 0.0)


def raised(whole, part):
    """`exp2(whole + part)` of a compensated base-2 exponent to fp32's resolution at any size: its integer
    `n = floor(whole)` an exact power of two (`attention_algebra.power2`, the attention template's) and its
    fraction `(whole - n) + part`, near [0, 1), through exp2. The exponent rounded to one fp32 lost up to half
    its ulp, 1.9e-6 near -43 (a pair of tokens at -30 a token), a decay's relative 1.3e-6, which put the
    port's decay gradient under decays near zero 7 to 10 times the token-by-token recurrence's distance from
    float64; and exp2 of a whole exponent, which a CPU takes as `exp(x ln 2)`, rounds its product with ln 2
    the same way."""
    n = _floor(whole)
    return power2(jnp.exp2((whole - n) + part), n)


def _floor(x):
    """`floor(x)` of fp32 `x` from its truncation toward zero, which Mosaic GPU lowers (it lowers no floor),
    the same value (a zero's sign aside): every fp32 of magnitude 2^23 or more is an integer, and infinities
    and NaNs are their own floor."""
    truncated = jnp.clip(x, -2.0 ** 23, 2.0 ** 23).astype(jnp.int32).astype(x.dtype)
    floor = jnp.where(truncated > x, truncated - 1.0, truncated)
    return jnp.where(jnp.abs(x) < 2.0 ** 23, floor, x)


def decayed(rows, rows_lo, columns, columns_lo):
    """`exp2(rows - columns)` `[R, C]` of a compensated prefix's halves (`apart`) to fp32's resolution, as
    `raised` takes it, each side's integer part apart from its fraction on its vector: the tile takes the
    integers' difference (exact), the fractions' in (-1, 1) and the low halves', and one exp2 of their sum
    scaled by the integers' power of two (`_scaled`). The exponent rounds at the fractions' scale, at least as
    finely as the whole difference's, and the tile takes no floor and one integer add for its scaling: a
    mainloop holds a few of its working tiles, not `raised`'s many (wgmma's output and reverse launches
    spilled 128 and 260 bytes on them, and none with no pair decay, SE regs-probe90b)."""
    row_whole, column_whole = _floor(rows), _floor(columns)
    fraction = (((rows - row_whole)[:, None] - (columns - column_whole)[None, :])
                + (rows_lo[:, None] - columns_lo[None, :]))
    return _scaled(jnp.exp2(fraction), row_whole[:, None] - column_whole[None, :])


PAIR_FLOOR = -124.0
"""The least power of two `_scaled` scales by, past which a pair's decay is zero: `x 2^k` of an `x` at least
1/4 stays a normal fp32 down to it, and a decay under 2^-122 weighs nothing beside the chunk's own token."""


def _scaled(x, k):
    """`x * 2^k` of `x` in [1/4, 4) (`exp2` of a pair's fraction) and an integral `k`, exactly, by adding `k`
    to `x`'s exponent bits: zero where `k` is under `PAIR_FLOOR`, and `k` held at 120 above, where the pairs
    past a token (a decay's inverse, which every caller masks) stay finite."""
    held = jnp.clip(k, PAIR_FLOOR, 120.0).astype(jnp.int32)
    bits = jax.lax.bitcast_convert_type(x, jnp.int32) + jax.lax.shift_left(held, jnp.full_like(held, 23))
    return jnp.where(k < PAIR_FLOOR, 0.0, jax.lax.bitcast_convert_type(bits, jnp.float32))


def separated(hi, lo):
    """A compensated base-2 exponent `(hi, lo)` as its whole part and fraction `(n, f)`, `f` in [0, 1): the
    form whose differences are exact at any size, the wholes' an integer and the fractions' within one
    (`between`). Once a call, where its prefix is cumulated."""
    whole = _floor(hi)
    part = (hi - whole) + lo
    carry = _floor(part)
    return whole + carry, part - carry


def between(later, earlier):
    """`exp2(later - earlier)` of two separated exponents (`separated`, broadcasting), to fp32's resolution at
    any size: exp2 of the fractions' difference, in (1/2, 2), scaled by the wholes' power of two (`_scaled`),
    zero below 2^-124. A difference of the rounded sums lost the fraction's last bits at the exponent's size,
    1e-6 of the decay near -43, which put a port's decay gradient under decays near zero 9 to 14 times the
    token-by-token recurrence's error from float64 (ledger 2098)."""
    (whole, fraction), (other, its_fraction) = later, earlier
    return _scaled(jnp.exp2(fraction - its_fraction), whole - other)


def grown_from(hi, lo, exp, unit: float = 1.0):
    """`exp(hi + lo)` of a compensated exponent `(hi, lo)`, rounded about once: `exp(hi)` corrected by its
    first-order term in `lo` (`|lo|` is under half an ulp of `hi`, so the second is far below fp32's
    rounding), `exp` the base's exponential and `unit` the derivative of its log at one (ln 2 for `exp2`)."""
    grown = exp(hi)
    return grown + grown * (lo * unit)


def _ranged(decay, dg):
    """The decay's tangent: each pair's decay times its range's tangent, `dg` summed over `(t, s]`, pair by
    pair; a pair across documents weighs zero."""
    width = dg.shape[-2]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    return decay * jnp.cumsum(jnp.where(strict, dg[..., :, None, :], 0.0), axis=-3)


def _ranged_transpose(decay, cotangent):
    """`_ranged`'s transpose: position i's gradient is the weighted cotangent of every pair whose range holds
    it, `sum_{s >= i} sum_{t < i} cotangent[s, t] decay[s, t]`, two cumulative sums of the chunk's pairs, no
    pair entering any sum it does not span."""
    width = decay.shape[-3]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    weighted = jnp.cumsum(jnp.where(strict, cotangent * decay, 0.0), axis=-2)
    before = jnp.concatenate([jnp.zeros_like(weighted[..., :, :1, :]), weighted[..., :, :-1, :]], axis=-2)
    after = jnp.flip(jnp.cumsum(jnp.flip(before, axis=-3), axis=-3), axis=-3)
    return jnp.moveaxis(jnp.diagonal(after, axis1=-3, axis2=-2), -1, -2)


@decays.defjvp
def _decays_tangent(primals, tangents):
    """The pairs' own tangent, in forward and reverse mode alike: the decay's is each pair's range
    (`_ranged`), and its transpose sums only the pairs spanning a position (`_ranged_transpose`). On a tiny
    hybrid's packed decision rows (tests/test_decision_packing.py, joint) the prefix difference's gradient put
    the per-head-decay layer's A_log 3.58 and dt_bias 4.49 times as far from float64 as the rows run alone,
    against 2 allowed; with this and the memory write's decay row (`xla_chunk_states`) every backbone leaf sat
    at 0.77-0.93, on six CPUs of two models. It costs the chunked rule's forward and backward 4% with a decay
    per head and 26% with one per key dimension on CPU (2 x 4096 x 16 heads of 128: 1.508 against 1.564 s; 2 x
    1024 x 8 heads, 1.936 against 2.440 s)."""
    g, documents = primals
    dg, _ = tangents
    grown, decay = decays(g, documents)
    return (grown, decay), (grown * _cumulated(dg, documents)[0],
                            linear_map(_ranged, _ranged_transpose, decay, dg))


def factor(a, issue: Issue = portable):
    """Return (I - A)^-1 for strictly lower triangular `a` `[..., C, C]`, the chunked delta rules' `attn[i,
    :i] += sum_k attn[i, k] attn[k, :i]` loop.

    Built from its diagonal blocks, doubling their size: with X1 and X2 the inverses of two neighbouring
    blocks, their union's is X1 and X2 on the diagonal and X2 A21 X1 below. Each block is a diagonal block of
    the whole inverse, so every product stays the size of its entries. The series I + A + A^2 + ..., summed by
    doubling powers of A, did not: on a 0.8B model's aligned keys the powers grew as binomials (1.8e17 at
    C=64) and cancelled to an inverse whose entries are at most 1, 1e24 off in fp32. The first level is
    written out rather than as products with one-wide identity blocks: XLA TPU rewrites a matmul by a
    broadcast identity into a dilated convolution whose fusion fails register allocation on a v6e
    (`live_range_finder.cc:57 RET_CHECK`, jax 0.11.2)."""
    chunk_size = a.shape[-1]
    if chunk_size == 1:
        return jnp.ones_like(a)
    width = 1 << (chunk_size - 1).bit_length()
    if width != chunk_size:  # zero rows and columns extend it by the identity
        pad = [(0, 0)] * (a.ndim - 2) + [(0, width - chunk_size)] * 2
        return factor(jnp.pad(a, pad), issue)[..., :chunk_size, :chunk_size]
    lead = a.shape[:-2]

    def diagonal_blocks(size):
        """`a`'s `[..., width // size, size, size]` blocks on the diagonal."""
        count = width // size
        blocked = a.reshape(*lead, count, size, count, size)
        return jnp.moveaxis(jnp.diagonal(blocked, axis1=-4, axis2=-2), -1, -3)

    # Two-wide blocks: [[1, 0], [a, 1]].
    below = diagonal_blocks(2)[..., 1, 0]
    one, zero = jnp.ones_like(below), jnp.zeros_like(below)
    inverse = jnp.stack([jnp.stack([one, zero], -1), jnp.stack([below, one], -1)], -2)
    size = 2
    while size < width:
        pairs = inverse.reshape(*lead, width // (2 * size), 2, size, size)
        first, second = pairs[..., 0, :, :], pairs[..., 1, :, :]
        joined = issue("merge", issue("merge", second, diagonal_blocks(2 * size)[..., size:, :size]), first)
        top = jnp.concatenate([first, jnp.zeros_like(first)], -1)
        inverse = jnp.concatenate([top, jnp.concatenate([joined, second], -1)], -2)
        size *= 2
    return inverse.reshape(*lead, width, width)


def paired(name: str, left, right, decay, issue: Issue = portable):
    """`sum_d left[s, d] right[t, d] decay[s, t, d]` within each chunk, `[..., C, C]`: one decay channel
    scales the product `name` after it, one per key dimension weighs each term of it."""
    if decay.shape[-1] == 1:
        return issue(name, left, jnp.swapaxes(right, -1, -2)) * decay[..., 0]
    return float32_einsum('...sd,...td,...std->...st', left, right, decay)


def intra(key, keyed, pairs, issue: Issue = portable):
    """The delta rule's strictly lower system `A` `[..., C, C]`: `-sum_d keyed[s, d] key[t, d] pairs[s, t, d]`
    for s > t, zero on and above the diagonal, `keyed` the write-gated keys `beta k`."""
    chunk_size = pairs.shape[-2]
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), jnp.bool_), -1)
    return jnp.where(strict, -paired("intra", keyed, key, pairs, issue), 0.0)


def apply(inverse, keyed, valued, entering, issue: Issue = portable):
    """`(W, U)`: the corrected write's products `inverse @ (keyed * entering)` and `inverse @ valued`, or,
    without the delta rule (`inverse` None), no correction and the gated values as they are."""
    if inverse is None:
        return jnp.zeros_like(keyed), valued
    return issue("apply_w", inverse, keyed * entering), issue("apply_u", inverse, valued)


def carry(state, step, issue: Issue = portable):
    """One chunk across the memory `state` `[..., Dk, Dv]`: `step` is the chunk's keys, `W`, `U`, the decay
    from its start `grown` (`decays`), the decay's last row `last` (`exp(gc[-1] - gc)`, zero for a packed
    chunk's earlier documents) and `kept` (one where the memory entering passes through, zero where a document
    starts in the chunk). Returns the memory leaving and `(the memory entering, v_new)`."""
    k_i, w_i, u_i, grown_i, last_i, kept_i = step
    v_corrected = u_i - issue("corrected", w_i, state)
    return (state * jnp.swapaxes(grown_i[..., -1:, :], -1, -2) * kept_i
            + issue("write", jnp.swapaxes(k_i * last_i, -1, -2), v_corrected), (state, v_corrected))


def _last_token(state, token, delta: bool):
    """One formed token `(key, value, log_decay, beta, start)` through the recurrence's own update."""
    key, value, g, beta, start = token
    state = jnp.where(start[..., None, None], 0.0, state)
    return recurrence.update(state, jnp.zeros_like(key), key, value, jnp.exp(g), beta[..., None],
                             delta=delta)[1]


def peeled_carry(state, step, *, delta: bool, issue: Issue = portable):
    """A chunk's WY write through its penultimate real token, then its last through the recurrence.

    `step[:6]` is `carry`'s step, whose `grown`, `last` and `kept` stop at the penultimate token;
    its keys, W and U still span the whole chunk, so the returned corrected values stay `U - W S`
    for `readout`. `step[6:]` is the last real token (`_last_token`), not the chunk's padding.

    Under aligned keys the WY write cancels separately rounded inverse/value paths: over 40
    draws its final memory sat 1.567 and 1.541 times the scalar/vector recurrence's error from
    float64, predominantly generated within each chunk, not accumulated across chunks. One
    chronological erasure at its boundary holds those at 0.997 and 1.036, and the formed gates'
    at 1.001 over 320 draws. The readout remains WY. At 2 x 4096 x 16 heads of 128, CPU forward
    time was 0.914 against 0.711 s, and compiled peak memory 846.8 against 845.2 MB."""
    prefix, products = carry(state, step[:6], issue)
    return _last_token(prefix, step[6:], delta), products


def readout(query, entering, entered, key, pairs, corrected, issue: Issue = portable):
    """A chunk's output `[..., C, Dv]`: the memory it entered read through the decayed queries (`remembered`),
    and the chunk's own pairs over its corrected values (the inclusive lower `pairs` read the token itself).
    """
    memory: jax.Array = remembered(query * entering, entered, issue)
    local: jax.Array = remembered(paired("scores", query, key, pairs, issue), corrected, issue, "local")
    return memory + local


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def remembered(decayed, entered, issue: Issue = portable, name: str = "memory"):
    """A chunk's tokens reading through their decays, `decayed @ entered` (the product `name`): the memory the
    chunk entered read through its decayed queries ('memory'), or its corrected values through its scores
    ('local'). The gradient in `entered` sums the chunk's tokens from its last (`_remembered_transpose`):
    `decayed` falls along the chunk with its decay, so the smallest terms add first, as the recurrence a token
    at a time adds them. Summed from the first, the memory's gradient sat 1.76 to 1.89 times the
    token-by-token recurrence's distance from float64 under every delta family, and the corrected values' put
    the values' and the write gate's at 1.5 to 1.6 under aligned keys; from the last, the memory's sat at
    0.97-1.22 (tests/test_recurrence_stability.py)."""
    return issue(name, decayed, entered)


def _remembered_transpose(decayed, cotangent, issue: Issue, name: str):
    """`decayed^T @ cotangent`, the chunk's tokens summed from its last."""
    return issue(name, jnp.swapaxes(jnp.flip(decayed, -2), -1, -2), jnp.flip(cotangent, -2))


def readout_entered_gradient(decayed_query, weights, scores, cotangent, issue: Issue = portable):
    """`readout`'s gradient in the memory a chunk entered: `Q_entered^T do - W^T (scores^T do)`,
    because its corrected values are `U - W S`. The token reductions run from the last
    (`_remembered_transpose`), as the memory's adjoint a token at a time does."""
    corrected = _remembered_transpose(scores, cotangent, issue, "local")
    memory = _remembered_transpose(decayed_query, cotangent, issue, "memory")
    return memory - issue("corrected", jnp.swapaxes(weights, -1, -2), corrected)


@remembered.defjvp
def _remembered_tangent(issue, name, primals, tangents):
    """The product's tangent, its `entered` side linear in `entered` with the transpose
    `_remembered_transpose`, in forward and reverse mode alike."""
    decayed, entered = primals
    d_decayed, d_entered = tangents
    read = linear_map(lambda decayed, entered: issue(name, decayed, entered),
                      functools.partial(_remembered_transpose, issue=issue, name=name), decayed, d_entered)
    return remembered(decayed, entered, issue, name), issue(name, d_decayed, entered) + read


class Span(NamedTuple):
    """A span's affine map of the memory entering it, `S -> T S + W`: its `write` `W` `[..., Dk, Dv]` and its
    `transfer` `T`, a dense one `[..., Dk, Dk]` (the delta rule's, `transition`, or a rule's final memory from
    the identity, `recurrence_chunk.entering_memory`) or a head's decay as its log `[...]`, never expanded to
    a matrix: it scales the whole memory, which then may be laid either way round (Mamba-2's `[..., P,
    N]`)."""

    transfer: jax.Array
    write: jax.Array

    @property
    def dense(self) -> bool:
        return self.transfer.ndim == self.write.ndim

    def decayed(self, memory, issue: Issue = portable) -> jax.Array:
        """`T S`."""
        if self.dense:
            return issue("compose", self.transfer, memory)
        return jnp.exp(self.transfer)[..., None, None] * memory

    def applied(self, memory, issue: Issue = portable) -> jax.Array:
        """`T S + W`: the memory the span leaves from the memory `S` entering it."""
        return self.decayed(memory, issue) + self.write


def compose(first: Span, second: Span, issue: Issue = portable) -> Span:
    """Two spans' transitions as one (`Span`, each of the same form): `second` after `first` is `(T2 T1, T2 W1
    + W2)`, a decay's logs summed, associative, so a scan over spans may run as a tree."""
    first, second = Span(*first), Span(*second)
    transfer = (issue("compose", second.transfer, first.transfer) if second.dense
                else first.transfer + second.transfer)
    return Span(transfer, second.applied(first.write, issue))


def entering(span: Span, axis: str, initial=None, issue: Issue = portable) -> jax.Array:
    """The memory each shard of a sequence the mesh axis `axis` splits enters its `span` with, inside a
    `shard_map` that holds `axis` manual: the shards' spans before it composed (`compose`) and applied to
    `initial`, the memory the first shard enters (zero where None). A Kogge-Stone scan: shifts 1, 2, 4, ...
    for `ceil(log2 n)` rounds, one span a shard a round, each shard composing the one it is handed where a
    shard that far back exists, then one shift more to make the prefix exclusive; autodiff transposes each
    `ppermute`. A dense span's prefix composes its transfers too (`T2 T1`), where gathering every shard's
    summaries and applying them in turn (fla's context-parallel states) held n of them a shard."""
    shards, index = jax.lax.axis_size(axis), jax.lax.axis_index(axis)

    def shifted(value, shift: int):
        return jax.lax.ppermute(value, axis, [(r, r + shift) for r in range(shards - shift)])

    span, shift = Span(*span), 1
    while shift < shards:
        composed = compose(shifted(span, shift), span, issue)
        span = jax.tree.map(functools.partial(jnp.where, index >= shift), composed, span)
        shift *= 2
    before = shifted(span, 1)
    if initial is None:
        return jnp.where(index > 0, before.write, 0.0)
    return jnp.where(index > 0, before.applied(initial.astype(before.write.dtype), issue), initial)


def transition(key, w, u, grown, last, kept, issue: Issue = portable) -> Span:
    """A chunk's transition `(T, W)` (`Span`, `compose`): with `carry`'s operands, `T = grown[-1] kept - (k
    last)^T W`
    and `W = (k last)^T U`, so `carry`'s memory leaving is `T S + W`. Packed documents keep the memory
    entering only where `kept` because the operands are built so (`recurrence_chunk.chunked`): `W` is zero
    outside the document the memory entering belongs to and `last` outside the chunk's last document; `kept`
    alone gates only the decay's diagonal."""
    decayed = jnp.swapaxes(key * last, -1, -2)
    dk = key.shape[-1]
    diagonal = jnp.eye(dk, dtype=grown.dtype) * jnp.swapaxes(grown[..., -1:, :], -1, -2) * kept
    return Span(diagonal - issue("corrected", decayed, w), issue("write", decayed, u))


def peeled_transition(key, w, u, grown, last, kept, token, *, delta: bool, issue: Issue = portable):
    """The affine map of `peeled_carry`: the prefix's transition followed by its last real token.
    Its linear part carries memory cotangents between chunks in the custom backward."""
    linear, written = transition(key, w, u, grown, last, kept, issue)
    k, _, g, beta, start = token
    no_write = k, jnp.zeros_like(k), g, beta, start
    return Span(_last_token(linear, no_write, delta), _last_token(written, token, delta))


# A chunk's tiles. The functions below are the same algebra on one chunk of one head, as a mainloop's
# program holds it: sub-chunks of SUB rows and column blocks, each operand a tile or a `Load`, which a
# GPU mainloop reads from memory where it is used and the portable mainloop slices. Every mainloop calls
# them as they are, so two mainloops at the same tiles compute the same values.

SUB = 16
"""Rows of a sub-chunk: the diagonal blocks the triangular inverse is built from (fla's BC)."""

class Tile(Protocol):
    """A tile a mainloop holds in memory: an operand its products take as it is, transposed or read in
    another dtype, whose registers `value()` loads where the algebra needs them elementwise."""

    @property
    def shape(self) -> tuple: ...

    @property
    def T(self) -> "Tile": ...

    def astype(self, dtype) -> "Tile": ...

    def value(self) -> jax.Array: ...


Load = Callable[..., jax.Array] | Callable[..., Tile]
"""A tile read where it is used: `load(i, c)` sub-chunk `i`'s column block `c`, or `load(c)` a chunk's; the
backward's loads a `Tile`."""

TileLoad = Callable[..., Tile]
"""A load of a `Tile` (the backward's operands)."""

Operand = jax.Array | Tile | Transposed | Exact
"""A product's operand (`Issue`'s): registers, a tile in memory, or registers taken transposed."""


def apart(rows, rows_lo, columns, columns_lo):
    """A compensated prefix's difference between positions, `rows - columns` `[R, C]` from each side's `(hi,
    lo)` halves (`recurrence.cumulated(compensated=True)`): the his' difference and the los', so the exponent
    is as precise as its own range."""
    return (rows[:, None] - columns[None, :]) + (rows_lo[:, None] - columns_lo[None, :])


def _lags(shape) -> jax.Array:
    """A tile's row index less its column index `[R, C]`, how far each column's token lags its row's (its
    column 0 the row index): the one int tile a mask comparing a row with its column holds (`lag > 0` strictly
    lower), where `_grid`'s two stay live."""
    rows, cols = _grid(shape)
    return rows - cols


def _grid(shape) -> tuple[jax.Array, jax.Array]:
    """A tile's row and column indices `[R, C]` each, two iotas, so a mainloop lays each as its tile (a vector
    broadcast both ways is one value Mosaic GPU's layout inference cannot lay twice)."""
    return jax.lax.broadcasted_iota(jnp.int32, shape, 0), jax.lax.broadcasted_iota(jnp.int32, shape, 1)


def tril_inverse(diagonal, index):
    """`(I + D)^-1` of a strictly lower square block `D`, row by row: forward substitution. `index` is the
    block's `arange` of rows, which the caller's other masks share."""
    sub = diagonal.shape[0]
    inverse = -diagonal
    for i in range(2, sub):
        row = jnp.sum(jnp.where((index == i)[:, None], -diagonal, 0.0), axis=0)
        row = jnp.where(index < i, row, 0.0)
        row = row + jnp.sum(row[:, None] * inverse, axis=0)
        inverse = jnp.where((index == i)[:, None], row[None, :], inverse)
    return inverse + (index[:, None] == index[None, :])


def system(keys: Load, decay: Callable[[int, int], jax.Array], betas, count: int, columns: int,
           issue: Issue) -> dict:
    """The blocks `{(i, j): [SUB, SUB]}` for `j <= i` of one chunk's `A = tril(beta_s (k_s . k_t) exp2(g_s -
    g_t), -1)` over its `count` sub-chunks, a decay a head: `keys(i, c)` sub-chunk `i`'s key column block `c`
    of `columns`, `decay(i, j)` the pairs' decays from sub-chunk `j` to `i` and `betas` each sub-chunk's
    write gates, fp32. Every block of `k k^T` sums its column blocks' products ('intra'): fla's kkt (the
    GDN port's solve)."""

    def block(i, j):
        product = sum(issue("intra", keys(i, c), keys(j, c).T) for c in range(columns))
        pairs = decay(i, j)
        if i == j:
            pairs = jnp.where(_lags(pairs.shape) > 0, pairs, 0.0)
        return product * pairs * betas[i][:, None]

    return {(i, j): block(i, j) for i in range(count) for j in range(i + 1)}


def keyed(product, gate, low, beta, column_gate, column_low):
    """One chunk's `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)` `[C, C]`, a decay a head, from its keys'
    products `product` (`k k^T`) and its prefix halves and write gates as a row's vectors (`gate`, `low`,
    `beta`) and the halves as a column's (`column_gate`, `column_low`), where a mainloop's layouts keep the
    two apart."""
    lag = _lags(product.shape)
    return jnp.where(lag > 0, product * decayed(gate, low, column_gate, column_low) * beta[:, None], 0.0)


def doubled(a, issue: Issue):
    """`factor`'s doubling on one tile: `(I - A)^-1` `[C, C]` of a strictly lower `a`, every level's blocks
    held in place in the tile, for a mainloop whose products take whole tiles (Mosaic GPU's, a warpgroup's 64
    rows, which hold no block of their own, and whose element-wise steps take 32 of a tile's entries a
    thread, so a forward substitution's 16 steps over its 16-row blocks cost a 64-row tile more than the
    three levels of products they replace). From the two-wide blocks `[[1, 0], [a, 1]]`, each level joins
    pairs of `s`-blocks as `X_2s = X_s + (M_s * (X_s A)) X_s` ('merge' twice), `M_s` each pair's lower block:
    the block-diagonal `X_s` against the whole tile, masked there, is its odd blocks' rows of `A` over the
    even ones', so the join is `factor`'s `X2 A21 X1` there and zero elsewhere, the same terms, and a
    mainloop stages `A` once."""
    size = a.shape[-1]
    rows, cols = _grid(a.shape)
    inverse = jnp.where(rows == cols, 1.0, 0.0) + jnp.where((rows % 2 == 1) & (cols == rows - 1), a, 0.0)
    width = 2
    while width < size:
        lower = ((rows // width) % 2 == 1) & (cols // width == rows // width - 1)
        inverse = inverse + issue("merge", jnp.where(lower, issue("merge", inverse, a), 0.0), inverse)
        width *= 2
    return inverse


def inverted(blocks: dict, issue: Issue, *, neighbours_first: bool = False) -> dict:
    """`(I + A)^-1`'s blocks `{(i, j): [SUB, SUB]}` for `j <= i` from `A`'s strictly lower blocks `blocks`
    (the diagonal ones strictly lower themselves): forward substitution inverts each diagonal block
    (`tril_inverse`), and neighbouring inverses merge as `X_ij = -X_ii sum_m A_im X_mj` ('merge'), nearest the
    diagonal first: fla's solve (chunk_fwd.py's for the gated delta rule). With `neighbours_first`, a pair of
    neighbours merges as `-(X_ii A_ij) X_jj`, the product order fla's chunk_intra.py takes for Kimi delta
    attention."""
    count = max(i for i, _ in blocks) + 1
    index = jnp.arange(blocks[0, 0].shape[0])
    inverse = {(i, i): tril_inverse(blocks[i, i], index) for i in range(count)}
    for distance in range(1, count):
        for i in range(distance, count):
            j = i - distance
            if distance == 1 and neighbours_first:
                inverse[i, j] = -issue("merge", issue("merge", inverse[i, i], blocks[i, j]), inverse[j, j])
                continue
            inner = functools.reduce(jnp.add, (issue("merge", blocks[i, m], inverse[m, j])
                                               for m in range(j, i)))
            inverse[i, j] = -issue("merge", inverse[i, i], inner)
    return inverse


def weighted(name: str, lower, tile, beta, issue: Issue, decay=None):
    """`(I + L) (tile beta [decay])` of the inverse's strictly lower part `lower` (`L`, the inverse less its
    unit diagonal), the scaled operand rounded to `tile`'s dtype for the product and kept fp32 on the
    diagonal: 'apply_u' on the values, 'apply_w' on the keys under their decay `exp2(g)` (`[C, 1]` a head's,
    `[C, BK]` a key dimension's). The diagonal outside the product leaves it the off-diagonal terms alone,
    where a tensor core's truncating sum (mma.sync's) loses no more than their last bit. Without the delta
    rule (`lower` None) nothing is corrected and there is no product: the gated tile, fp32 (`apply`'s)."""
    scaled = tile * beta[:, None] if decay is None else tile * beta[:, None] * decay
    if lower is None:
        return scaled.astype(jnp.float32)
    return scaled.astype(jnp.float32) + issue(name, lower, scaled.astype(tile.dtype))


def corrected(values, weights: Load | None, state, dtype, issue: Issue):
    """A chunk's corrected values `u - w S` `[C, BV]`, fp32: `values` its `u`, `weights(i)` its `w`'s key
    slice `i` of the memory's (`state`, fp32), each slice cast to `w`'s `dtype` for its product; without the
    delta rule (`weights` None) `u` itself."""
    if weights is None:
        return values.astype(jnp.float32)
    predicted = sum(issue("corrected", weights(i), part.astype(dtype)) for i, part in enumerate(state))
    return values.astype(jnp.float32) - predicted


def carried(state, keys: Load, values, decay, dtype, issue: Issue, last: tuple | None = None) -> list:
    """The memory leaving a chunk, slice by key slice: `state[i] decay[i] + keys(i)^T values` ('write'), the
    values cast to the keys' `dtype`; `decay[i]` scales each slice's rows `[K_i, 1]` or the whole. `last`,
    `(key_rows, value_row, row)`, takes the chunk's last real token's write, its row `row`, out of the
    product, its key slice `key_rows(i)` as a row's vector and its values `value_row` as a column's: the
    write that weighs most under steep decays, added on its own where a tensor core's truncating sum
    (mma.sync's) would lose its last bit."""
    values = values.astype(dtype)
    if last is None:
        return [part * decay[i] + issue("write", keys(i).T, values) for i, part in enumerate(state)]
    key_rows, value_row, row = last
    rows, _ = _grid(values.shape)
    earlier = jnp.where(rows != row, values, jnp.zeros((), values.dtype))
    return [part * decay[i] + (issue("write", keys(i).T, earlier) + key_rows(i)[:, None] * value_row[None, :])
            for i, part in enumerate(state)]


def scalar_write(corrected_values, last, gate, last_lo, gate_lo):
    """A decay a head's write: the corrected values gated by `exp2(g_last - g)` over both halves, and the
    memory's decay `exp2(g_last)`."""
    gated = corrected_values * raised(last - gate, last_lo - gate_lo)[:, None]
    return gated, raised(last, last_lo)


def chunk_scores(queries: Load, keys: Load, columns: int, gate, low, dtype, issue: Issue, *,
                 laid: tuple | None = None):
    """A chunk's own pairs' scores `[C, C]` under a decay a head, `tril((q k^T) exp2(g_s - g_t))`, the key
    column blocks `queries(c)` and `keys(c)` summed into fp32 ('scores'), in `dtype` for `chunk_output`'s
    'local', which every value block of the chunk takes. `laid` is the prefix's halves `(gate, low)` as a
    column's vectors where a mainloop's layouts keep them apart from a row's (`system`'s). The pairs' decays
    come before the products, their exponents' working tiles done before the products' sum is live."""
    decay = decayed(gate, low, *((gate, low) if laid is None else laid))
    scores = jnp.zeros((gate.shape[0], gate.shape[0]), jnp.float32)
    for c in range(columns):
        scores += issue("scores", queries(c), keys(c).T)
    scores = scores * decay
    lag = _lags(scores.shape)
    return jnp.where(lag >= 0, scores, 0.0).astype(dtype)


def chunk_output(queries: Load, entered: Load, columns: int, gate, low, scores, value, scale: float,
                 issue: Issue):
    """A chunk's output `[C, BV]`, fp32, a decay a head: `scale ((q S) exp2(g) + scores v)`, the key column
    blocks `queries(c)` and `entered(c)` (the memory the chunk entered) summed into fp32 ('memory'), and the
    chunk's own pairs' `scores` (`chunk_scores`, registers or a tile in the values' dtype) against the value
    block ('local')."""
    out = jnp.zeros((gate.shape[0], value.shape[-1]), jnp.float32)
    for c in range(columns):
        out += issue("memory", queries(c), entered(c))
    out = out * raised(gate, low)[:, None]
    return out * scale + issue("local", scores, value) * scale


def factored_range(floor: float, sub: int = SUB) -> bool:
    """Whether a decay a key dimension whose every token's log decay is at least `floor` (natural log, a
    layer's declared lower bound: Kimi delta attention's -5) factors over `sub`-token sub-chunks
    (`key_decayed_pairs`): a sub-chunk's widest range in base 2, `-floor sub log2(e)`, under fp32's largest
    power of two's exponent, so a key gated up from its sub-chunk's start stays finite."""
    return -floor * sub * math.log2(math.e) < 126


def key_decayed_pairs(rows: Callable[[int], jax.Array], keys: Callable[[int], jax.Array], columns: int,
                      gates: Callable[[int], tuple],
                      reference: Callable[[int, int], tuple], issue: Issue, *, chunk: int, strict: bool,
                      sub: int = SUB):
    """A chunk's pairs' products under a decay a key dimension `[C, C]` fp32, `sum_k r_ik k_jk exp2(g_ik -
    g_jk)` for `j <= i` (`j < i` where `strict`), zero elsewhere: the rows `rows(c)` (the queries, or the
    write-gated keys) and keys `keys(c)` of each key block `c` of `columns`, fp32, `gates(c)` its prefix
    halves, over a chunk of `chunk` tokens. Each sub-chunk `s` of `sub` rows takes its first token's prefix
    `reference(s, c)` (a column's vectors) as its pairs' common point: its rows gated down to it,
    `r exp2(g - g_ref)`, against every key up to its end gated up from it, `k exp2(g_ref - g)`, one product a
    sub-chunk and key block ('pairs'): fla_kda's sub-chunk factoring, its diagonal sub-chunks' too, which a
    declared floor bounds (`factored_range`). An exponent is its halves' difference summed, within a
    sub-chunk's range of zero down the rows (and up the keys); past it every pair's true value underflows."""
    total = jnp.zeros((chunk, chunk), jnp.float32)
    for place in range(0, chunk, sub):
        for c in range(columns):
            # Each tile read where it is used, a sub-chunk and key block at a time: few live at once.
            gate, low = gates(c)
            start_gate, start_low = reference(place // sub, c)
            exponent = (gate - start_gate[None, :]) + (low - start_low[None, :])
            index, _ = _grid(exponent.shape)
            down = jnp.where((index >= place) & (index < place + sub), rows(c) * jnp.exp2(exponent), 0.0)
            up = jnp.where(index < place + sub, keys(c) * jnp.exp2(-exponent), 0.0)
            total = total + issue("pairs", down, Transposed(up))
    lag = _lags(total.shape)
    return jnp.where(lag > 0 if strict else lag >= 0, total, 0.0)


def key_decayed_write(key, gate, low, last, last_low):
    """A chunk's keys gated to its end under a decay a key dimension, `k exp2(g_last - g)` over both prefix
    halves, in the keys' arithmetic: the write the memory takes (`carried`)."""
    return key * raised(last - gate, last_low - low)


def key_gated(query, gates: tuple, scale: float):
    """A key block's queries gated by their decays to the chunk's start and scaled, `scale q exp2(g)`
    (`gates` the prefix's halves, `raised`), rounded to the queries' dtype: `key_decayed_output`'s left
    operand, as fla's chunk_gla output kernel scales its gated queries."""
    return (query * raised(*gates) * scale).astype(query.dtype)


def key_decayed_output(gated: Load, entered: Load, columns: int, scores, value,
                       summed: Callable[[list], jax.Array]):
    """A chunk's output `[C, BV]`, fp32, a decay a key dimension: `(scale q exp2(g))_narrow S + tril(Aqk) v`,
    each key block's gated and scaled queries `gated(c)` (`key_gated`'s) against the memory entered
    `entered(c)` ('memory') and the scores `scores` (`Aqk`, scaled) rounded to the values' dtype ('local'),
    every product into one accumulator (`summed`, the mainloop's: a list of `(name, left, right)`), as fla's
    chunk_kda sums its key blocks' into one."""
    lag = _lags(scores.shape)
    local = jnp.where(lag >= 0, scores, 0.0).astype(value.dtype)
    return summed([("memory", gated(c), entered(c)) for c in range(columns)] + [("local", local, value)])


# The fast backward of a decay a head under the delta rule, fla's arithmetic on a chunk's tiles (the port's
# `dew.nn.kernels.fla_gdn` backward kernels): the transposes of the forward launches' products, each taken as
# a product of tiles a mainloop holds as they are or reads transposed, no register tile transposed. A
# decay's gradient leaves as its terms, the pairs `[C, C]` (each `(i, j)` a term through the prefix's range
# `(j, i]`) and each token's terms through the range from its chunk's start `[C]`, which `summed_back`
# sums token by token. The accurate backward is the recurrence's own adjoint a token at a time
# (`recurrence_chunk.chunk_delta_rule`'s).


def local_gradient(keys: TileLoad, queries: TileLoad, columns: int, gate, low, out_grad, scale: float,
                   issue: Issue, *, laid: tuple | None = None):
    """The corrected values' gradient from the chunk's own output, `[C, BV]` fp32: `A^T do` for
    `chunk_output`'s scores `A[t, s] = scale (q_t . k_s) exp2(g_t - g_s)` (`s <= t`), its transpose taken as
    `k q^T` ('local_t') and cast to the gradient's dtype for its product ('local_dv'); `laid` as
    `chunk_output`'s. The pairs' decays come before the products, as `chunk_scores`'."""
    column_gate, column_low = (gate, low) if laid is None else laid
    decay = decayed(-gate, -low, -column_gate, -column_low)
    product = functools.reduce(jnp.add, (issue("local_t", keys(c), queries(c).T) for c in range(columns)))
    product = product * decay
    lag = _lags(product.shape)
    product = jnp.where(lag <= 0, product * scale, 0.0)
    return issue("local_dv", product.astype(out_grad.dtype), out_grad)


def reverse_carried(grad: list, keys: TileLoad, weights: TileLoad, queries: TileLoad, out_grad, local, gate,
                    low, last, last_low, scale: float, dtype, issue: Issue) -> tuple[list, jax.Array]:
    """One chunk of the memory's reverse recurrence (fla's dhu) under a decay a head: from the gradient of
    the memory leaving the chunk `grad`, its key slices `[K_i, BV]` fp32, the corrected values' gradient
    `dv = (k grad) exp2(g_last - g) + local` ('values_dv', the gradient cast to `dtype`) and the gradient of
    the memory the chunk entered, `grad exp2(g_last) + scale q^T (do exp2(g)) - w^T dv` ('reads', the
    decayed `do` fp32; 'weights_dv', `dv` cast to `dtype`). `last` and `last_low` are the chunk's last prefix
    halves as a row's vectors, `queries(i)` and `weights(i)` key slice `i`'s tiles."""
    dv = functools.reduce(jnp.add, (issue("values_dv", keys(i), part.astype(dtype))
                                    for i, part in enumerate(grad)))
    dv = dv * raised(last - gate, last_low - low)[:, None] + local
    read = out_grad.astype(jnp.float32) * raised(gate, low)[:, None]
    kept = raised(last, last_low)[:, None]
    entered = [part * kept + issue("reads", queries(i).T, read) * scale
               - issue("weights_dv", weights(i).T, dv.astype(dtype)) for i, part in enumerate(grad)]
    return entered, dv


def scores_gradient(queries: TileLoad, keys: TileLoad, columns: int, values: TileLoad, out_grads: TileLoad,
                    blocks: int, gate, low, scale: float, dtype, issue: Issue, *,
                    laid: tuple | None = None) -> tuple[jax.Array, jax.Array]:
    """A chunk and value head's scores' gradient under a decay a head, `ds = do v^T` over the value blocks
    `j` gated by its pairs' decays, causal and scaled, in `dtype` (which `memory_gradients`' `dq += ds k`
    and `dk += ds^T q` take), and the decay's pair terms `ds (q k^T)` (strictly lower, fp32,
    `summed_pairs`'): the part of fla's dqkwg the memory's products do not take, a launch of its own."""
    column_gate, column_low = (gate, low) if laid is None else laid
    pair_decay = decayed(gate, low, column_gate, column_low)
    ds = functools.reduce(jnp.add, (issue("values_ds", out_grads(j), values(j).T) for j in range(blocks)))
    lag = _lags(ds.shape)
    ds = jnp.where(lag >= 0, ds * pair_decay, 0.0) * scale
    scores = functools.reduce(jnp.add, (issue("local_t", queries(c), keys(c).T) for c in range(columns)))
    return ds.astype(dtype), jnp.where(lag > 0, ds * scores, 0.0)


def memory_gradients(queries: TileLoad, keys: TileLoad, columns: int, values: TileLoad, out_grads: TileLoad,
                     value_grads: TileLoad, entered: TileLoad, leaving: TileLoad, blocks: int, ds, gate, low,
                     last, last_low, memory_decay, scale: float, issue: Issue,
                     emit: Callable[[str, int, jax.Array], None]) -> tuple[jax.Array, jax.Array]:
    """The queries', keys' and weights' gradients of a chunk and value head (fla's dqkwg's memory products)
    under a decay a head, given its scores' gradient `ds` (`scores_gradient`'s, a tile its products read as
    they are, in the operands' dtype): over the value blocks `j`, `dq = do S^T`, `dk = v dS^T` and
    `dw = -dv S^T` for the corrected values `values(j)`, the output's gradient `out_grads(j)`, the corrected
    values' own `value_grads(j)` and the memories the chunk entered and left `entered(c, j)`, `leaving(c, j)`
    (key block `c`), then the gates and `dq += ds k`, `dk += ds^T q`, each key block's to `emit('dq' | 'dk'
    | 'dw', c, grad)` as it is final; and the decay's range terms `(after, leaving)` (`summed_back`'s),
    `leaving` `[C]` each token's write into the memory the chunk leaves. `memory_decay` is the decay's
    gradient through the memory the chunk leaves, `sum(S dS)` (a row's vector of it; None where a caller
    summing key blocks apart adds it once)."""
    row_decay = raised(gate, low)[:, None]
    leaving_decay = raised(last - gate, last_low - low)[:, None]
    leaving_terms, after = [], []
    for c in range(columns):
        dq = _summed(issue("memory_dq", out_grads(j), entered(c, j).T) for j in range(blocks))
        dq = dq * row_decay * scale
        after.append(jnp.sum(dq * queries(c).value().astype(jnp.float32), axis=1))
        emit("dq", c, dq + issue("scores_dq", ds, keys(c)))
        dk = _summed(issue("memory_dk", values(j), leaving(c, j).T) for j in range(blocks))
        dk = dk * leaving_decay
        leaving_terms.append(jnp.sum(dk * keys(c).value().astype(jnp.float32), axis=1))
        emit("dk", c, dk + issue("scores_dq", Transposed(ds), queries(c)))
        emit("dw", c, -_summed(issue("memory_dw", value_grads(j), entered(c, j).T) for j in range(blocks)))
    after = _summed(after)
    if memory_decay is not None:
        lag = _lags((after.shape[0],) * 2)
        after = after + jnp.where(lag[:, 0] == after.shape[0] - 1, memory_decay * raised(last, last_low), 0.0)
    return after, _summed(leaving_terms)


def system_gradient(lower, keys: TileLoad, columns: int, values: TileLoad, blocks: int, w_grads: TileLoad,
                    u_grads: TileLoad, beta, gate, low, dtype, issue: Issue, *,
                    laid: tuple | None = None) -> jax.Array:
    """The WY representation's backward (fla's prepare_wy_repr_bwd) under a decay a head, its system's part:
    with the inverse `X = I + lower` (`lower` its strictly lower part, a tile), `w = X (k beta exp2(g))` and
    `u = X (v beta)`, from `w`'s and `u`'s gradients `w_grads(c)` and `u_grads(j)`: `dX = dw (k beta
    exp2 g)^T + du (v beta)^T` ('inverse_dx') and `dL = -X^T tril(dX, -1) X^T` gated by `exp2(g_s - g_t)`
    ('inverse_t', 'inverse_l'), in `dtype`, which `keys_gradient` takes. `X^T dw` and `X^T du` and what
    they give are `keys_gradient`'s and `values_gradient`'s."""
    def transposed_inverse(m):
        """`X^T m`, the diagonal outside the product."""
        return m + issue("inverse_t", lower.T, m.astype(dtype))

    decay = raised(gate, low)
    d_x = None
    for c in range(columns):
        scaled = keys(c).value().astype(jnp.float32) * (beta * decay)[:, None]
        d_x = _added(d_x, issue("inverse_dx", w_grads(c), Transposed(scaled.astype(dtype))))
    for j in range(blocks):
        scaled = (values(j).value().astype(jnp.float32) * beta[:, None]).astype(dtype)
        d_x = _added(d_x, issue("inverse_dx", u_grads(j), Transposed(scaled)))
    assert d_x is not None, "a chunk's keys and values take at least a block each"
    lag = _lags(d_x.shape)
    strict = lag > 0
    d_x = jnp.where(strict, d_x, 0.0)
    d_x = d_x + issue("inverse_l", d_x.astype(dtype), lower.T)
    d_x = transposed_inverse(d_x)
    column_gate, column_low = (gate, low) if laid is None else laid
    return jnp.where(strict, -d_x * decayed(gate, low, column_gate, column_low), 0.0).astype(dtype)


def values_gradient(lower, values: TileLoad, blocks: int, u_grads: TileLoad, beta, dtype, issue: Issue, *,
                    emit: Callable[[str, int, jax.Array], None]) -> jax.Array:
    """The WY representation's backward under a decay a head, its values' part: each value block's gradient
    `(X^T du) beta` ('inverse_t') to `emit('values', j, dv)` as it is computed. Returns beta's gradient
    through `u`, fp32."""
    d_betas = []
    for j in range(blocks):
        u_grad = u_grads(j).value().astype(jnp.float32)
        d_scaled = u_grad + issue("inverse_t", lower.T, u_grad.astype(dtype))
        emit("values", j, d_scaled * beta[:, None])
        d_betas.append(jnp.sum(d_scaled * values(j).value().astype(jnp.float32), axis=1))
    return _summed(d_betas)


def keys_gradient(lower, keys: TileLoad, columns: int, w_grads: TileLoad, d_l, beta, gate, low, dtype,
                  issue: Issue, *,
                  emit: Callable[[str, int, jax.Array], None]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The WY representation's backward under a decay a head, its keys' part, given the system's gradient
    `d_l` (`system_gradient`'s, a tile its products read as they are, in `dtype`): each key block's gradient,
    through `w` (`X^T dw` scaled, 'inverse_t') and through the system (`dL k` and `dL^T (k beta)`,
    'keys_dl', 'keys_dlt'), to `emit('keys', c, dk)` as it is final. Returns beta's gradient through `w` and
    the system, the decay's range terms through `w` (`summed_back`'s `after`) and its pair terms
    `dL (k k^T) beta` (`summed_pairs`'), fp32."""
    decay = raised(gate, low)
    narrow = d_l if isinstance(d_l, jax.Array) else d_l.value()
    weighted_l = (narrow.astype(jnp.float32) * beta[:, None]).astype(dtype)
    d_betas, d_gate = [], None
    for c in range(columns):
        key = keys(c)
        w_grad = w_grads(c).value().astype(jnp.float32)
        # `X^T dw` decayed, then its sum with `dL k` (each term's own beta its row's): one tile of them live.
        through_w = (w_grad + issue("inverse_t", lower.T, w_grad.astype(dtype))) * decay[:, None]
        key_value = key.value().astype(jnp.float32)
        d_gate = _added(d_gate, beta * jnp.sum(through_w * key_value, axis=1))
        combined = through_w + issue("keys_dl", d_l, key)
        d_betas.append(jnp.sum(combined * key_value, axis=1))
        # `dL^T (k beta)` as `(beta dL)^T k`: one operand in registers, the keys a tile.
        emit("keys", c, combined * beta[:, None] + issue("keys_dlt", Transposed(weighted_l), key))
    # The keys' own pairs once every block's gradient is out: a tile not live beside theirs.
    products = None
    for c in range(columns):
        key = keys(c)
        products = _added(products, issue("keys_dl", key, key.T))
    assert products is not None and d_gate is not None, "a chunk's keys take at least a block"
    pairs = narrow.astype(jnp.float32) * (products * beta[:, None])
    return _summed(d_betas), d_gate, pairs


def _added(total, term):
    """`total + term`, `term` where `total` is None: a sum held as it accumulates, in `_summed`'s order, one
    tile live rather than every term."""
    return term if total is None else total + term


def _summed(terms) -> jax.Array:
    """`terms`' sum in their order."""
    return functools.reduce(jnp.add, terms)


def summed_back(pairs, after, leaving=None, issue: Issue = portable,
                columns: Callable[[jax.Array], jax.Array] | None = None):
    """The gradient of each of a chunk's per-token decays from the gradients of terms through ranges of its
    prefix: `pairs` `[C, C]` fp32, each `(i, j)` a term through `(j, i]` (strictly lower), `after` `[C]`,
    each `s` a term through the chunk's start to `s`, and `leaving` `[C]` where given, each `t` a term
    through `(t, C - 1]`, a pair of the chunk's last row. Each token sums only the terms whose range holds it,
    `sum_{j < t <= i} pairs + sum_{s >= t} after + sum_{j < t} leaving`: each row's pairs before `t` a
    product with the strictly upper ones ('summed', an `Exact` triangle: no pair added and taken back), each
    `leaving` term in its own row above the diagonal (a vector a row each, as a mainloop's registers lay
    it), and each token's sum its column's (`columns`, a mainloop's own reduction across its rows' warps,
    where given).
    Linear in its terms: a sum of launches' terms sums the same as their sums, and it is its pairs' sum
    (`summed_pairs`) and its ranges' (cumulative sums of `after` and `leaving`), which launches of their own
    may each take."""
    lag = _lags(pairs.shape)
    through = issue("summed", pairs, Exact(jnp.where(lag < 0, 1.0, 0.0)))
    above = 0.0 if leaving is None else leaving[:, None]
    terms = jnp.where(lag >= 0, through + after[:, None], above)
    return jnp.sum(terms, axis=0) if columns is None else columns(terms)


def summed_pairs(pairs, issue: Issue = portable, columns: Callable[[jax.Array], jax.Array] | None = None):
    """`summed_back`'s pairs' part, `sum_{j < t <= i} pairs` each token's."""
    lag = _lags(pairs.shape)
    terms = jnp.where(lag >= 0, issue("summed", pairs, Exact(jnp.where(lag < 0, 1.0, 0.0))), 0.0)
    return jnp.sum(terms, axis=0) if columns is None else columns(terms)
