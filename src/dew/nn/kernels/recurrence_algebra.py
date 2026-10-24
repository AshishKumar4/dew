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
- `compose(first, second)`: two spans' transitions `S -> T S + W` as one, associatively, for a scan over
  shards or chunks run in parallel.

Each works over any leading dimensions: the portable math (`recurrence_chunk.chunk_delta_rule`) calls them on
`[B, H, NC, ...]` arrays, and a kernel's body on one chunk's tiles. Packed `documents` restart the decays at
each document's first position (varlen).

The algebra declares its products and issues none (the design's layering): every product goes through the
caller's `issue(name, left, right)`, the mainloop's, by its name in `PRODUCTS`, which says what it multiplies;
the portable math's issuer is XLA's (`portable`), a kernel's its own dot."""

import functools
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from dew.nn.kernels.templates.attention_algebra import power2
from dew.nn.precision import compensated_add, float32_einsum, float32_matmul

from . import recurrence

Issue = Callable[[str, jax.Array, jax.Array], jax.Array]
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
    "values_ds": "the output's gradient against the corrected values, `do v^T` (`query_key_gradients`)",
    "memory_dq": "the output's gradient against the entered memory's transpose, `do S^T`",
    "memory_dk": "the corrected values against the leaving memory's gradient, `v dS^T`",
    "memory_dw": "the values' gradient against the entered memory's transpose, `dv S^T`",
    "scores_dq": "the scores' gradient against the keys, `ds k`, and its transpose against the queries",
    "inverse_t": "the inverse's transpose against a gradient, `L^T dX` (`weighted_gradient`)",
    "inverse_dx": "a gradient against a weighted operand's transpose, `dw (k b exp(gc))^T`",
    "inverse_l": "the inverse's gradient through the inverse, `L^T dX L^T`",
    "keys_dl": "the system's gradient against the keys, `dL k`, and the keys against themselves, `k k^T`",
    "keys_dlt": "the system's gradient, its rows scaled by the write gates, transposed against the keys",
    "summed": "a decay's range terms against a triangle or the identity, each token's sum (`summed_back`)",
}
"""The algebra's products by name. A decay a key dimension weighs each term of `intra`'s and `scores`'
contractions (`paired`), which a mainloop issues as a product of the factored operands within the range its
decay's bound keeps exact, or as the weighed sum."""


class Transposed:
    """A register operand a product takes transposed, `value^T`: a mainloop stages it and reads the tile
    transposed, where a transpose of registers would move every element between threads."""

    def __init__(self, value: jax.Array):
        self.value = value

    @property
    def shape(self) -> tuple:
        return (*self.value.shape[:-2], self.value.shape[-1], self.value.shape[-2])


def portable(name: str, left, right):
    """XLA's product, the portable math's issuer: `left @ right`, float32 products in float32 on a GPU, as
    torch's are, not TF32 (`dew.nn.precision.float32_matmul`); the portable math takes its operands in at
    least float32 (`recurrence_chunk.chunked`), a `Transposed` one transposed."""
    del name
    left, right = (jnp.swapaxes(x.value, -1, -2) if isinstance(x, Transposed) else x for x in (left, right))
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
    """`exp2(rows - columns)` `[R, C]` of a compensated prefix's halves (`apart`) to fp32's resolution
    (`raised`)."""
    return raised(rows[:, None] - columns[None, :], rows_lo[:, None] - columns_lo[None, :])


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
                            linear_call(_ranged, _ranged_transpose, decay, dg))


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
    read = linear_call(lambda decayed, entered: issue(name, decayed, entered),
                       functools.partial(_remembered_transpose, issue=issue, name=name), decayed, d_entered)
    return remembered(decayed, entered, issue, name), issue(name, d_decayed, entered) + read


def compose(first, second, issue: Issue = portable):
    """Two spans' transitions as one: each a pair `(T [..., Dk, Dk], W [..., Dk, Dv])` taking the memory
    entering to `T S + W`; `second` after `first` is `(T2 T1, T2 W1 + W2)`, associative, so a scan over spans
    may run as a tree."""
    (t1, w1), (t2, w2) = first, second
    return issue("compose", t2, t1), issue("compose", t2, w1) + w2


def transition(key, w, u, grown, last, kept, issue: Issue = portable):
    """A chunk's transition `(T, W)` (`compose`): with `carry`'s operands, `T = grown[-1] kept - (k last)^T W`
    and `W = (k last)^T U`, so `carry`'s memory leaving is `T S + W`. Packed documents keep the memory
    entering only where `kept` because the operands are built so (`recurrence_chunk.chunked`): `W` is zero
    outside the document the memory entering belongs to and `last` outside the chunk's last document; `kept`
    alone gates only the decay's diagonal."""
    decayed = jnp.swapaxes(key * last, -1, -2)
    dk = key.shape[-1]
    diagonal = jnp.eye(dk, dtype=grown.dtype) * jnp.swapaxes(grown[..., -1:, :], -1, -2) * kept
    return diagonal - issue("corrected", decayed, w), issue("write", decayed, u)


def peeled_transition(key, w, u, grown, last, kept, token, *, delta: bool, issue: Issue = portable):
    """The affine map of `peeled_carry`: the prefix's transition followed by its last real token.
    Its linear part carries memory cotangents between chunks in the custom backward."""
    linear, written = transition(key, w, u, grown, last, kept, issue)
    k, _, g, beta, start = token
    no_write = k, jnp.zeros_like(k), g, beta, start
    return _last_token(linear, no_write, delta), _last_token(written, token, delta)


# A chunk's tiles. The functions below are the same algebra on one chunk of one head, as a mainloop's
# program holds it: sub-chunks of SUB rows and column blocks, each operand a tile or a `Load`, which a
# GPU mainloop reads from memory where it is used and the portable mainloop slices. Every mainloop calls
# them as they are, so two mainloops at the same tiles compute the same values.

SUB = 16
"""Rows of a sub-chunk: the diagonal blocks the triangular inverse is built from (fla's BC)."""

class Tile(Protocol):
    """A tile a mainloop holds in memory: an operand its products take as it is or transposed, whose
    registers `value()` loads where the algebra needs them elementwise."""

    @property
    def shape(self) -> tuple: ...

    @property
    def T(self) -> "Tile": ...

    def value(self) -> jax.Array: ...


Load = Callable[..., jax.Array] | Callable[..., Tile]
"""A tile read where it is used: `load(i, c)` sub-chunk `i`'s column block `c`, or `load(c)` a chunk's; the
backward's loads a `Tile`."""

TileLoad = Callable[..., Tile]
"""A load of a `Tile` (the backward's operands)."""


def apart(rows, rows_lo, columns, columns_lo):
    """A compensated prefix's difference between positions, `rows - columns` `[R, C]` from each side's `(hi,
    lo)` halves (`recurrence.cumulated(compensated=True)`): the his' difference and the los', so the exponent
    is as precise as its own range."""
    return (rows[:, None] - columns[None, :]) + (rows_lo[:, None] - columns_lo[None, :])


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


def solved(keys: Load, gates, lows, betas, columns: int, issue: Issue) -> dict:
    """`(I + A)^-1`'s blocks `{(i, j): [SUB, SUB]}` for `j <= i` of one chunk's `A = tril(beta_s (k_s . k_t)
    exp2(g_s - g_t), -1)`, a decay a head: `system`'s blocks, inverted (`inverted`)."""
    return inverted(system(keys, gates, lows, betas, columns, issue), issue)


def system(keys: Load, gates, lows, betas, columns: int, issue: Issue, *,
           laid: tuple | None = None) -> dict:
    """The blocks `{(i, j): [SUB, SUB]}` for `j <= i` of one chunk's `A = tril(beta_s (k_s . k_t) exp2(g_s -
    g_t), -1)`, a decay a head: `keys(i, c)` sub-chunk `i`'s key column block `c` of `columns`, `gates`,
    `lows` and `betas` each sub-chunk's prefix halves and write gates, fp32, and `laid` the halves `(gates,
    lows)` as a column's vectors, where a mainloop's layouts keep a column's vector apart from a row's (Mosaic
    GPU's, whose vectors are one or the other of a tile's). Every block of `k k^T` sums its column blocks'
    products ('intra')."""
    count = len(gates)
    column_gates, column_lows = (gates, lows) if laid is None else laid

    def block(i, j):
        product = sum(issue("intra", keys(i, c), keys(j, c).T) for c in range(columns))
        decay = decayed(gates[i], lows[i], column_gates[j], column_lows[j])
        if i == j:
            rows, cols = _grid(decay.shape)
            decay = jnp.where(rows > cols, decay, 0.0)
        return product * decay * betas[i][:, None]

    return {(i, j): block(i, j) for i in range(count) for j in range(i + 1)}


def keyed(product, gate, low, beta, column_gate, column_low):
    """One chunk's `A = tril(beta_s (k_s . k_t) exp2(g_s - g_t), -1)` `[C, C]`, a decay a head, from its keys'
    products `product` (`k k^T`) and its prefix halves and write gates as a row's vectors (`gate`, `low`,
    `beta`) and the halves as a column's (`column_gate`, `column_low`), where a mainloop's layouts keep the
    two apart."""
    rows, cols = _grid(product.shape)
    return jnp.where(rows > cols, product * decayed(gate, low, column_gate, column_low) * beta[:, None], 0.0)


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


def inverted(blocks: dict, issue: Issue) -> dict:
    """`(I + A)^-1`'s blocks `{(i, j): [SUB, SUB]}` for `j <= i` from `A`'s strictly lower blocks `blocks`
    (the diagonal ones strictly lower themselves): forward substitution inverts each diagonal block
    (`tril_inverse`), and neighbouring inverses merge as `X_ij = -X_ii sum_m A_im X_mj` ('merge'), nearest the
    diagonal first."""
    count = max(i for i, _ in blocks) + 1
    index = jnp.arange(blocks[0, 0].shape[0])
    inverse = {(i, i): tril_inverse(blocks[i, i], index) for i in range(count)}
    for distance in range(1, count):
        for i in range(distance, count):
            j = i - distance
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
    `(key_rows, value_row)`, takes the chunk's last token's write out of the product, its key slice
    `key_rows(i)` as a row's vector and its values `value_row` as a column's: the write that weighs most
    under steep decays, added on its own where a tensor core's truncating sum (mma.sync's) would lose its
    last bit."""
    values = values.astype(dtype)
    if last is None:
        return [part * decay[i] + issue("write", keys(i).T, values) for i, part in enumerate(state)]
    key_rows, value_row = last
    rows, _ = _grid(values.shape)
    earlier = jnp.where(rows < values.shape[0] - 1, values, jnp.zeros((), values.dtype))
    return [part * decay[i] + (issue("write", keys(i).T, earlier) + key_rows(i)[:, None] * value_row[None, :])
            for i, part in enumerate(state)]


def scalar_write(corrected_values, last, gate, last_lo, gate_lo):
    """A decay a head's write: the corrected values gated by `exp2(g_last - g)` over both halves, and the
    memory's decay `exp2(g_last)`."""
    gated = corrected_values * raised(last - gate, last_lo - gate_lo)[:, None]
    return gated, raised(last, last_lo)


def chunk_output(queries: Load, entered: Load, keys: Load, columns: int, gate, low, value, scale: float,
                 issue: Issue, *, laid: tuple | None = None):
    """A chunk's output `[C, BV]`, fp32, a decay a head: `scale ((q S) exp2(g) + tril((q k^T) exp2(g_s - g_t))
    v)`, the key column blocks `queries(c)`, `entered(c)` (the memory the chunk entered) and `keys(c)` summed
    into fp32 ('memory', 'scores'), the scores cast to the values' dtype for 'local'. `laid` is the prefix's
    halves `(gate, low)` as a column's vectors where a mainloop's layouts keep them apart from a row's
    (`system`'s)."""
    out = jnp.zeros((gate.shape[0], value.shape[-1]), jnp.float32)
    scores = jnp.zeros((gate.shape[0], gate.shape[0]), jnp.float32)
    for c in range(columns):
        query = queries(c)
        out += issue("memory", query, entered(c))
        scores += issue("scores", query, keys(c).T)
    out = out * raised(gate, low)[:, None]
    scores = scores * decayed(gate, low, *((gate, low) if laid is None else laid))
    rows, cols = _grid(scores.shape)
    scores = jnp.where(rows >= cols, scores, 0.0)
    return out * scale + issue("local", scores.astype(value.dtype), value) * scale


def key_decayed_row(query, weighted_key, gate, low, earlier: Load, place, chunk: int, scale: float):
    """One token's rows of a chunk's scores and keys' products under a decay a key dimension, `[R, C]` each
    for its `R` value heads, fp32: `Aqk_j = scale sum_k q_k k_jk exp2(g_k - g_jk)` for `j <= place` and `Akk_j
    = sum_k (beta k)_k k_jk exp2(g_k - g_jk)` for `j < place`, every pair's exponent its own over both prefix
    halves (`gate`, `low`). `earlier(j)` reads the chunk's `j`-th token's prefix halves and key, fp32."""
    index = jnp.arange(chunk)

    def column(j, sums):
        gate_j, low_j, key_j = earlier(j)
        gated = key_j * raised(gate - gate_j, low - low_j)
        here = index == j
        # The earlier columns as a vector's mask: Mosaic GPU's layouts take no scalar condition over a tile.
        return (jnp.where(here[None, :], (jnp.sum(query * gated, axis=1) * scale)[:, None], sums[0]),
                jnp.where((here & (index < place))[None, :], jnp.sum(weighted_key * gated, axis=1)[:, None],
                          sums[1]))

    empty = jnp.zeros((query.shape[0], chunk), jnp.float32)
    return jax.lax.fori_loop(0, place + 1, column, (empty, empty))


def key_decayed_write(key, gate, low, last, last_low):
    """A chunk's keys gated to its end under a decay a key dimension, `k exp2(g_last - g)` over both prefix
    halves, in the keys' arithmetic: the write the memory takes (`carried`)."""
    return key * raised(last - gate, last_low - low)


def key_decayed_output(queries: Load, gates: Load, entered: Load, columns: int, scores, value, scale: float,
                       issue: Issue):
    """A chunk's output `[C, BV]`, fp32, a decay a key dimension: `scale (q exp2(g))_narrow S + tril(Aqk) v`,
    each key block's gated queries `queries(c) exp2(gates(c))` (`gates(c)` the prefix's halves, `raised`)
    rounded to their dtype against the memory entered `entered(c)` ('memory'), the scores `scores` (`Aqk`,
    scaled) rounded to the values' dtype ('local')."""
    out = jnp.zeros((scores.shape[0], value.shape[-1]), jnp.float32)
    for c in range(columns):
        query = queries(c)
        gated = (query * raised(*gates(c))).astype(query.dtype)
        out = out + issue("memory", gated, entered(c).astype(gated.dtype))
    out = out * scale
    rows, cols = _grid(scores.shape)
    local = jnp.where(rows >= cols, scores, 0.0).astype(value.dtype)
    return out + issue("local", local, value)


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
    `chunk_output`'s."""
    product = functools.reduce(jnp.add, (issue("local_t", keys(c), queries(c).T) for c in range(columns)))
    column_gate, column_low = (gate, low) if laid is None else laid
    product = product * decayed(-gate, -low, -column_gate, -column_low)
    rows, cols = _grid(product.shape)
    product = jnp.where(rows <= cols, product * scale, 0.0)
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


def query_key_gradients(queries: TileLoad, keys: TileLoad, columns: int, values: TileLoad,
                        out_grads: TileLoad, value_grads: TileLoad, entered: TileLoad, leaving: TileLoad,
                        blocks: int, gate, low, last, last_low, memory_decay, scale: float, dtype,
                        issue: Issue, *, laid: tuple | None = None):
    """The queries', keys', weights' and decay's gradients of a chunk and value head (fla's dqkwg) under a
    decay a head: over the value blocks `j`, `ds = do v^T`, `dq = do S^T`, `dk = v dS^T` and `dw = -dv S^T`
    for the corrected values `values(j)`, the output's gradient `out_grads(j)`, the corrected values' own
    `value_grads(j)` and the memories the chunk entered and left `entered(c, j)`, `leaving(c, j)` (key block
    `c`); then the gates, `ds`'s causal mask, `dq += ds k` and `dk += ds^T q`. `memory_decay` is the decay's
    gradient through the memory the chunk leaves, `sum(S dS)` (a row's vector of it; None where a caller
    summing key blocks apart adds it once). Returns each key block's `(dq, dk, dw)` fp32 and the decay's
    terms `(pairs, after, leaving)`, `leaving` `[C]` each token's write into the memory the chunk leaves,
    a term through `(t, C - 1]` (a vector a row each, as a mainloop holds it, where its place among the pairs
    is a column's)."""
    ds = functools.reduce(jnp.add, (issue("values_ds", out_grads(j), values(j).T) for j in range(blocks)))
    row_decay = raised(gate, low)[:, None]
    leaving_decay = raised(last - gate, last_low - low)[:, None]
    column_gate, column_low = (gate, low) if laid is None else laid
    pair_decay = decayed(gate, low, column_gate, column_low)
    rows, cols = _grid(ds.shape)
    ds = jnp.where(rows >= cols, ds * pair_decay, 0.0) * scale
    scores = functools.reduce(jnp.add, (issue("local_t", queries(c), keys(c).T) for c in range(columns)))
    grads, leaving_terms, after = [], [], []
    for c in range(columns):
        dq = _summed(issue("memory_dq", out_grads(j), entered(c, j).T) for j in range(blocks))
        dk = _summed(issue("memory_dk", values(j), leaving(c, j).T) for j in range(blocks))
        dw = -_summed(issue("memory_dw", value_grads(j), entered(c, j).T) for j in range(blocks))
        dq = dq * row_decay * scale
        dk = dk * leaving_decay
        query, key = queries(c).value().astype(jnp.float32), keys(c).value().astype(jnp.float32)
        leaving_terms.append(jnp.sum(dk * key, axis=1))
        after.append(jnp.sum(dq * query, axis=1))
        grads.append((dq, dk, dw))
    leaving_terms, after = _summed(leaving_terms), _summed(after)
    pairs = jnp.where(rows > cols, ds * scores, 0.0)
    if memory_decay is not None:
        after = after + jnp.where(rows[:, 0] == ds.shape[0] - 1, memory_decay * raised(last, last_low), 0.0)
    narrow = ds.astype(dtype)
    transposed = Transposed(narrow)
    grads = [(dq + issue("scores_dq", narrow, keys(c)), dk + issue("scores_dq", transposed, queries(c)), dw)
             for c, (dq, dk, dw) in enumerate(grads)]
    return grads, (pairs, after, leaving_terms)


def weighted_gradient(lower, keys: TileLoad, columns: int, values: TileLoad, blocks: int, w_grads: TileLoad,
                      u_grads: TileLoad, beta, gate, low, dtype, issue: Issue, *, laid: tuple | None = None):
    """The WY representation's backward (fla's prepare_wy_repr_bwd) under a decay a head: with the inverse
    `X = I + lower` (`lower` its strictly lower part, a tile), `w = X (k beta exp2(g))` and `u = X (v beta)`,
    the keys', values', write gate's and decay's gradients from `w`'s and `u`'s, `w_grads(c)` and
    `u_grads(j)`: `dX = dw (k beta exp2 g)^T + du (v beta)^T`, `dL = -X^T tril(dX, -1) X^T` gated by
    `exp2(g_s - g_t)` ('inverse_t', 'inverse_dx', 'inverse_l'), then the keys' gradient through the system
    ('keys_dl'). Returns each key block's `dk`, each value block's `dv`, beta's gradient and the decay's terms
    `(pairs, after)`, fp32."""
    def transposed_inverse(m):
        """`X^T m`, the diagonal outside the product."""
        return m + issue("inverse_t", lower.T, m.astype(dtype))

    decay = raised(gate, low)
    d_xs, d_betas, d_gates, through_w = [], [], [], []
    for c in range(columns):
        key = keys(c).value().astype(jnp.float32)
        scaled = key * (beta * decay)[:, None]
        w_grad = w_grads(c)
        d_scaled = transposed_inverse(w_grad.value().astype(jnp.float32))
        through_w.append(d_scaled)
        d_xs.append(issue("inverse_dx", w_grad, Transposed(scaled.astype(dtype))))
        d_betas.append(jnp.sum(d_scaled * key * decay[:, None], axis=1))
        d_gates.append(jnp.sum(d_scaled * scaled, axis=1))
    value_grads = []
    for j in range(blocks):
        value = values(j).value().astype(jnp.float32)
        scaled = (value * beta[:, None]).astype(dtype)
        u_grad = u_grads(j)
        d_scaled = transposed_inverse(u_grad.value().astype(jnp.float32))
        value_grads.append(d_scaled * beta[:, None])
        d_xs.append(issue("inverse_dx", u_grad, Transposed(scaled)))
        d_betas.append(jnp.sum(d_scaled * value, axis=1))
    d_x, d_gate = _summed(d_xs), _summed(d_gates)
    rows, cols = _grid(d_x.shape)
    strict = rows > cols
    d_x = jnp.where(strict, d_x, 0.0)
    d_x = d_x + issue("inverse_l", d_x.astype(dtype), lower.T)
    d_x = transposed_inverse(d_x)
    column_gate, column_low = (gate, low) if laid is None else laid
    d_l = jnp.where(strict, -d_x * decayed(gate, low, column_gate, column_low), 0.0).astype(dtype)
    weighted_l = (d_l.astype(jnp.float32) * beta[:, None]).astype(dtype)
    products, key_grads = [], []
    for c in range(columns):
        key = keys(c)
        products.append(issue("keys_dl", key, key.T))
        d_key = issue("keys_dl", d_l, key)
        # `dL^T (k beta)` as `(beta dL)^T k`: one operand in registers, the keys a tile.
        key_grads.append(through_w[c] * (decay * beta)[:, None] + d_key * beta[:, None]
                         + issue("keys_dlt", Transposed(weighted_l), key))
        d_betas.append(jnp.sum(d_key * key.value().astype(jnp.float32), axis=1))
    pairs = d_l.astype(jnp.float32) * (_summed(products) * beta[:, None])
    return key_grads, value_grads, _summed(d_betas), (pairs, d_gate)


def _summed(terms) -> jax.Array:
    """`terms`' sum in their order."""
    return functools.reduce(jnp.add, terms)


def summed_back(pairs, after, leaving=None, issue: Issue = portable):
    """The gradient of each of a chunk's per-token decays from the gradients of terms through ranges of its
    prefix: `pairs` `[C, C]` fp32, each `(i, j)` a term through `(j, i]` (strictly lower), `after` `[C]`,
    each `s` a term through the chunk's start to `s`, and `leaving` `[C]` where given, each `t` a term
    through `(t, C - 1]`, a pair of the chunk's last row. Each token sums only the terms whose range holds it,
    `sum_{j < t <= i} pairs + sum_{s >= t} after`: each row's pairs before `t` a product with the strictly
    upper ones ('summed', no pair added and taken back), and each column's sum the transpose's row sums (a
    product with the identity, exact)."""
    rows, cols = _grid(pairs.shape)
    if leaving is not None:
        pairs = pairs + jnp.where((rows == pairs.shape[0] - 1) & (rows > cols), leaving[None, :], 0.0)
    through = issue("summed", pairs, jnp.where(rows < cols, 1.0, 0.0))
    held = jnp.where(rows >= cols, through + after[:, None], 0.0)
    return jnp.sum(issue("summed", Transposed(held), jnp.where(rows == cols, 1.0, 0.0)), axis=1)
