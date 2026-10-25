"""The recurrence template's algebra (docs/design/kernel-templates.md §3): the chunked form of the update
as named, pure functions on values, which every implementation of the chunked op calls and none
rewrites. This rung covers `S' = T S + W` with T a diagonal decay times the delta rule's erase, E the
identity, and a chunk's transition `(T, W)` dense (`transition`); the value-side decay E and the factored
transitions follow with the families that need them. A token updates and reads, per head over a state
`S` `[Dk, Dv]`,

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q

(`recurrence.update` is its one-token form). Within a chunk of C tokens:

- `decays(g, documents) -> (grown, pairs)`: the decay from the chunk's start `exp(gc)`, the log decay `gc`
  cumulated within the chunk, and each pair of positions' decay `exp(gc[s] - gc[t])`, every exponent
  compensated;
- `intra(k, kb, pairs) -> A`: the delta rule's strictly lower key-key system, `-(beta k k^T)` weighed by the
  pairs;
- `factor(A) -> L`: its triangular inverse `(I - A)^-1` (WY/UT), skipped where nothing is corrected;
- `apply(L, kb, vb, entering) -> (W, U)`: the corrected write's two products, `L (beta k exp(gc))` and
  `L (beta v)`;
- `carry(S, step) -> (S', (S, v_new))`: one chunk across the memory, the corrected values `v_new = U - W S`
  and the write `grown[-1] S + (k exp(gc[-1] - gc))^T v_new`;
- `readout(q, entering, entered, k, pairs, v_new) -> o`: the memory entered read through the decayed
  queries, plus the chunk's own pairs;
- `compose(first, second)`: two spans' transitions `S -> T S + W` as one, associatively, for a scan over
  shards or chunks run in parallel.

Each works over any leading dimensions: the portable math (`recurrence_chunk.chunk_delta_rule`) calls them
on `[B, H, NC, ...]` arrays, and a kernel's body on one chunk's tiles. Packed `documents` restart the
decays at each document's first position (varlen).

The algebra declares its products and issues none (the design's layering): every product goes through the
caller's `issue(name, left, right)`, the mainloop's, by its name in `PRODUCTS`, which says what it
multiplies; the portable math's issuer is XLA's (`portable`), a kernel's its own dot.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

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
}
"""The algebra's products by name. A decay a key dimension weighs each term of `intra`'s and `scores`'
contractions (`paired`), which a mainloop issues as a product of the factored operands within the
range its decay's bound keeps exact, or as the weighed sum."""


def portable(name: str, left, right):
    """XLA's product, the portable math's issuer: `left @ right`, float32 products in float32 on a GPU,
    as torch's are, not TF32, summed as Dew sums a CPU's float32 products (`dew.nn.precision.float32_matmul`,
    which `accumulated`); the portable math takes its operands in at least float32
    (`recurrence_chunk.chunked`)."""
    del name
    return float32_matmul(left, right)


def _cumulated(g, documents=None):
    """`g` `[..., C, F]` cumulated over its C positions as a compensated pair `(hi, lo)`
    (`compensated_add`), restarting at each of packed `documents`' first positions."""
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

    `g` is `[..., C, F]`: the C positions of a chunk and F decay channels.
    Returns the decay from the chunk's start `grown = exp(gc)`, `gc` the
    inclusive cumulative sum over C, `[..., C, F]`, and
    `decay[..., s, t, f] = exp(gc[s, f] - gc[t, f])` for s >= t, zero above
    the diagonal, `[..., C, C, F]`.

    The references subtract two fp32 cumulative sums, so every exponent
    carries the rounding of the chunk's whole accumulated magnitude, which
    exp turns into a relative error on every decay. Here the cumulative sum
    is compensated (`compensated_add`) and the difference is a `(whole, part)` pair too, exp taken of
    each, so a decay is as precise as its own exponent: the difference rounded to one fp32 lost up to
    half its ulp, which put the gradients of the decay and the write gate under aligned keys 1.6 to
    2.1 times the token-by-token recurrence's distance from float64 on an A100; `grown` takes
    both halves too (`grown_from`), where exp of the rounded sum carried its
    rounding into the memory's decay and put the state's gradient near
    twice the reference's distance from float64. On
    a per-key-decay model's tiny fixture that error dominated the gradient:
    over 16 orderings of the residual stream (an exact symmetry, so only the
    rounding moves) the updated logits sat at an RMS 2.04 times the
    reference's distance from float64 with plain sums on CPU and 1.83 on an
    RTX 4080, and 0.92 and 0.72 with these, against the reference's own
    0.98. Summing each pair's range instead (a segment sum) is as
    exact and made a per-key-decay layer's forward and backward 25% slower; this
    costs about 1% (on an RTX 4080, a 32-head per-key-decay layer over 1024 tokens
    47.6 against 48.0 ms, a 0.8B model's per-head-decay layer over 4096 16.3
    against 16.5).

    The gradient is the pairs' own (`_decays_tangent`): differentiating
    the difference of two prefixes sums a pair's cotangent into its two ends
    with opposite signs, and the decay gradient, a row sum less a column sum
    of the chunk's pairs, cancelled to fp32 noise wherever the prefixes ran
    long.

    With packed `documents` `[..., C, 1]` (each position's document within
    its chunk), the sum restarts at each document's first position and pairs
    of positions in different documents weigh zero, masked before exp: a
    document's decays are its own tokens' alone, the ones it computes run by
    itself.
    """
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
    `n = floor(whole)` an exact power of two (`_power_of_two`) and its fraction `(whole - n) + part`, near
    [0, 1), through exp2. The exponent rounded to one fp32 lost up to half its ulp, 1.9e-6 near -43 (a
    pair of tokens at -30 a token), a decay's relative 1.3e-6, which put the port's decay gradient under
    decays near zero 7 to 10 times the token-by-token recurrence's distance from float64; and exp2 of a
    whole exponent, which a CPU takes as `exp(x ln 2)`, rounds its product with ln 2 the same way."""
    n = jnp.floor(whole)
    return jnp.exp2((whole - n) + part) * _power_of_two(n)


def _power_of_two(n):
    """`2^n` of an integral fp32 `n`, exactly: two halves' fp32 bits built from their exponents, whose
    product reaches fp32's subnormals and rounds past them to zero, and past fp32's range to inf."""
    n = jnp.clip(n, -300.0, 300.0).astype(jnp.int32)
    half = jax.lax.shift_right_arithmetic(n, jnp.int32(1))

    def bits(exponent):
        biased = jnp.clip(exponent, -127, 128) + 127
        return jax.lax.bitcast_convert_type(jax.lax.shift_left(biased, jnp.int32(23)), jnp.float32)

    return bits(half) * bits(n - half)


def decayed(rows, rows_lo, columns, columns_lo):
    """`exp2(rows - columns)` `[R, C]` of a compensated prefix's halves (`apart`) to fp32's resolution
    (`raised`)."""
    return raised(rows[:, None] - columns[None, :], rows_lo[:, None] - columns_lo[None, :])


def grown_from(hi, lo, exp, unit: float = 1.0):
    """`exp(hi + lo)` of a compensated exponent `(hi, lo)`, rounded about once: `exp(hi)` corrected by its
    first-order term in `lo` (`|lo|` is under half an ulp of `hi`, so the second is far below fp32's
    rounding), `exp` the base's exponential and `unit` the derivative of its log at one (ln 2 for
    `exp2`)."""
    grown = exp(hi)
    return grown + grown * (lo * unit)


def _ranged(decay, dg):
    """The decay's tangent: each pair's decay times its range's tangent, `dg` summed over `(t, s]`,
    pair by pair; a pair across documents weighs zero."""
    width = dg.shape[-2]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    return decay * jnp.cumsum(jnp.where(strict, dg[..., :, None, :], 0.0), axis=-3)


def _ranged_transpose(decay, cotangent):
    """`_ranged`'s transpose: position i's gradient is the weighted cotangent of every pair whose
    range holds it, `sum_{s >= i} sum_{t < i} cotangent[s, t] decay[s, t]`, two cumulative sums
    of the chunk's pairs, no pair entering any sum it does not span."""
    width = decay.shape[-3]
    strict = jnp.tril(jnp.ones((width, width), bool), -1)[..., None]
    weighted = jnp.cumsum(jnp.where(strict, cotangent * decay, 0.0), axis=-2)
    before = jnp.concatenate([jnp.zeros_like(weighted[..., :, :1, :]), weighted[..., :, :-1, :]], axis=-2)
    after = jnp.flip(jnp.cumsum(jnp.flip(before, axis=-3), axis=-3), axis=-3)
    return jnp.moveaxis(jnp.diagonal(after, axis1=-3, axis2=-2), -1, -2)


@decays.defjvp
def _decays_tangent(primals, tangents):
    """The pairs' own tangent, in forward and reverse mode alike: the decay's is each pair's range
    (`_ranged`), and its transpose sums only the pairs spanning a position (`_ranged_transpose`).
    On a tiny hybrid's packed decision rows (tests/test_decision_packing.py, joint) the prefix
    difference's gradient put the per-head-decay layer's A_log 3.58 and dt_bias 4.49 times as far from
    float64 as the rows run alone, against 2 allowed; with this and the memory write's decay row
    (`xla_chunk_states`) every backbone leaf sat at 0.77-0.93, on six CPUs of two models. It costs
    the chunked rule's forward and backward 4% with a decay per head and 26% with one per key
    dimension on CPU (2 x 4096 x 16 heads of 128: 1.508 against 1.564 s; 2 x 1024 x 8 heads,
    1.936 against 2.440 s)."""
    g, documents = primals
    dg, _ = tangents
    grown, decay = decays(g, documents)
    return (grown, decay), (grown * _cumulated(dg, documents)[0],
                            linear_call(_ranged, _ranged_transpose, decay, dg))


def factor(a, issue: Issue = portable):
    """Return (I - A)^-1 for strictly lower triangular `a` `[..., C, C]`, the
    chunked delta rules' `attn[i, :i] += sum_k attn[i, k] attn[k, :i]` loop.

    Built from its diagonal blocks, doubling their size: with X1 and X2 the
    inverses of two neighbouring blocks, their union's is X1 and X2 on the
    diagonal and X2 A21 X1 below. Each block is a diagonal block of the
    whole inverse, so every product stays the size of its entries. The
    series I + A + A^2 + ..., summed by doubling powers of A, did not: on
    a 0.8B model's aligned keys the powers grew as binomials (1.8e17 at C=64)
    and cancelled to an inverse whose entries are at most 1, 1e24 off in
    fp32. The first level is written out rather than as products with
    one-wide identity blocks: XLA TPU rewrites a matmul by a broadcast
    identity into a dilated convolution whose fusion fails register
    allocation on a v6e (`live_range_finder.cc:57 RET_CHECK`, jax 0.11.2).
    """
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
    chunk's earlier documents) and `kept`
    (one where the memory entering passes through, zero where a document starts in the chunk). Returns
    the memory leaving and `(the memory entering, v_new)`."""
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
    """A chunk's output `[..., C, Dv]`: the memory it entered read through the decayed queries
    (`remembered`), and the chunk's own pairs over its corrected values (the inclusive lower `pairs` read
    the token itself)."""
    memory: jax.Array = remembered(query * entering, entered, issue)
    local: jax.Array = remembered(paired("scores", query, key, pairs, issue), corrected, issue, "local")
    return memory + local


@functools.partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def remembered(decayed, entered, issue: Issue = portable, name: str = "memory"):
    """A chunk's tokens reading through their decays, `decayed @ entered` (the product `name`): the memory
    the chunk entered read through its decayed queries ('memory'), or its corrected values through its
    scores ('local'). The gradient in `entered` sums the chunk's tokens from its last
    (`_remembered_transpose`): `decayed` falls along the chunk with its decay, so the smallest terms add
    first, as the recurrence a token at a time adds them. Summed from the first, the memory's gradient sat
    1.76 to 1.89 times the token-by-token recurrence's distance from float64 under every delta family, and
    the corrected values' put the values' and the write gate's at 1.5 to 1.6 under aligned keys; from the
    last, the memory's sat at 0.97-1.22 (tests/test_recurrence_stability.py)."""
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
    entering to `T S + W`; `second` after `first` is `(T2 T1, T2 W1 + W2)`, associative, so a scan over
    spans may run as a tree."""
    (t1, w1), (t2, w2) = first, second
    return issue("compose", t2, t1), issue("compose", t2, w1) + w2


def transition(key, w, u, grown, last, kept, issue: Issue = portable):
    """A chunk's transition `(T, W)` (`compose`): with `carry`'s operands,
    `T = grown[-1] kept - (k last)^T W` and `W = (k last)^T U`, so `carry`'s memory leaving is `T S + W`.
    Packed documents keep the memory entering only where `kept` because the operands are built so
    (`recurrence_chunk.chunked`): `W` is zero outside the document the memory entering belongs to and
    `last` outside the chunk's last document; `kept` alone gates only the decay's diagonal."""
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

Load = Callable[..., jax.Array]
"""A tile read where it is used: `load(i, c)` sub-chunk `i`'s column block `c`, or `load(c)` a chunk's."""


def apart(rows, rows_lo, columns, columns_lo):
    """A compensated prefix's difference between positions, `rows - columns` `[R, C]` from each side's
    `(hi, lo)` halves (`recurrence.cumulated(compensated=True)`): the his' difference and the los', so
    the exponent is as precise as its own range."""
    return (rows[:, None] - columns[None, :]) + (rows_lo[:, None] - columns_lo[None, :])


def tril_inverse(diagonal, index):
    """`(I + D)^-1` of a strictly lower square block `D`, row by row: forward substitution. `index` is
    the block's `arange` of rows, which the caller's other masks share."""
    sub = diagonal.shape[0]
    inverse = -diagonal
    for i in range(2, sub):
        row = jnp.sum(jnp.where((index == i)[:, None], -diagonal, 0.0), axis=0)
        row = jnp.where(index < i, row, 0.0)
        row = row + jnp.sum(row[:, None] * inverse, axis=0)
        inverse = jnp.where((index == i)[:, None], row[None, :], inverse)
    return inverse + (index[:, None] == index[None, :])


def solved(keys: Load, gates, lows, betas, columns: int, issue: Issue) -> dict:
    """`(I + A)^-1`'s blocks `{(i, j): [SUB, SUB]}` for `j <= i` of one chunk's `A = tril(beta_s (k_s .
    k_t) exp2(g_s - g_t), -1)`, a decay a head: `keys(i, c)` sub-chunk `i`'s key column block `c` of
    `columns`, `gates`, `lows` and `betas` each sub-chunk's prefix halves and write gates, fp32. Every
    block of `k k^T` sums its column blocks' products ('intra'), and the blocks invert (`inverted`)."""
    count = len(gates)
    index = jnp.arange(gates[0].shape[0])
    strict = index[:, None] > index[None, :]

    def block(i, j):
        product = sum(issue("intra", keys(i, c), keys(j, c).T) for c in range(columns))
        decay = decayed(gates[i], lows[i], gates[j], lows[j])
        product = product * (jnp.where(strict, decay, 0.0) if i == j else decay)
        return product * betas[i][:, None]

    return inverted({(i, j): block(i, j) for i in range(count) for j in range(i + 1)}, issue)


def inverted(blocks: dict, issue: Issue) -> dict:
    """`(I + A)^-1`'s blocks `{(i, j): [SUB, SUB]}` for `j <= i` from `A`'s strictly lower blocks
    `blocks` (the diagonal ones strictly lower themselves): forward substitution inverts each diagonal
    block (`tril_inverse`), and neighbouring inverses merge as `X_ij = -X_ii sum_m A_im X_mj` ('merge'),
    nearest the diagonal first."""
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


def weighted(name: str, inverse, tile, beta, issue: Issue, decay=None):
    """`inverse (tile beta [decay])`, the scaled operand rounded to `tile`'s dtype before the product:
    'apply_u' on the values, 'apply_w' on the keys under their decay `exp2(g)` (`[C, 1]` a head's, `[C,
    BK]` a key dimension's)."""
    scaled = tile * beta[:, None] if decay is None else tile * beta[:, None] * decay
    return issue(name, inverse, scaled.astype(tile.dtype))


def corrected(values, weights: Load, state, dtype, issue: Issue):
    """A chunk's corrected values `u - w S` `[C, BV]`, fp32: `values` its `u`, `weights(i)` its `w`'s
    key slice `i` of the memory's (`state`, fp32), each slice cast to `w`'s `dtype` for its product."""
    predicted = sum(issue("corrected", weights(i), part.astype(dtype)) for i, part in enumerate(state))
    return values.astype(jnp.float32) - predicted


def carried(state, keys: Load, values, decay, dtype, issue: Issue) -> list:
    """The memory leaving a chunk, slice by key slice: `state[i] decay[i] + keys(i)^T values` ('write'),
    the values cast to the keys' `dtype`; `decay[i]` scales each slice's rows `[K_i, 1]` or the whole."""
    return [part * decay[i] + issue("write", keys(i).T, values.astype(dtype)) for i, part in enumerate(state)]


def scalar_write(corrected_values, last, gate, last_lo, gate_lo):
    """A decay a head's write: the corrected values gated by `exp2(g_last - g)` over both halves, and the
    memory's decay `exp2(g_last)`."""
    gated = corrected_values * raised(last - gate, last_lo - gate_lo)[:, None]
    return gated, raised(last, last_lo)


def chunk_output(queries: Load, entered: Load, keys: Load, columns: int, gate, low, value, scale: float,
                 issue: Issue):
    """A chunk's output `[C, BV]`, fp32, a decay a head: `scale ((q S) exp2(g) + tril((q k^T) exp2(g_s -
    g_t)) v)`, the key column blocks `queries(c)`, `entered(c)` (the memory the chunk entered) and
    `keys(c)` summed into fp32 ('memory', 'scores'), the scores cast to the values' dtype for 'local'."""
    out = jnp.zeros((gate.shape[0], value.shape[-1]), jnp.float32)
    scores = jnp.zeros((gate.shape[0], gate.shape[0]), jnp.float32)
    for c in range(columns):
        query = queries(c)
        out += issue("memory", query, entered(c))
        scores += issue("scores", query, keys(c).T)
    out = out * raised(gate, low)[:, None]
    scores = scores * decayed(gate, low, gate, low)
    index = jnp.arange(gate.shape[0])
    scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
    return out * scale + issue("local", scores.astype(value.dtype), value) * scale


def key_decayed_row(query, weighted_key, gate, low, earlier: Load, place, chunk: int, scale: float):
    """One token's rows of a chunk's scores and keys' products under a decay a key dimension, `[R, C]`
    each for its `R` value heads, fp32: `Aqk_j = scale sum_k q_k k_jk exp2(g_k - g_jk)` for `j <= place`
    and `Akk_j = sum_k (beta k)_k k_jk exp2(g_k - g_jk)` for `j < place`, every pair's exponent its own over
    both prefix halves (`gate`, `low`). `earlier(j)` reads the chunk's `j`-th token's prefix halves and
    key, fp32."""
    index = jnp.arange(chunk)

    def column(j, sums):
        gate_j, low_j, key_j = earlier(j)
        gated = key_j * raised(gate - gate_j, low - low_j)
        here = (index == j)[None, :]
        return (jnp.where(here, (jnp.sum(query * gated, axis=1) * scale)[:, None], sums[0]),
                jnp.where(here & (j < place), jnp.sum(weighted_key * gated, axis=1)[:, None], sums[1]))

    empty = jnp.zeros((query.shape[0], chunk), jnp.float32)
    return jax.lax.fori_loop(0, place + 1, column, (empty, empty))


def key_decayed_write(key, gate, low, last, last_low):
    """A chunk's keys gated to its end under a decay a key dimension, `k exp2(g_last - g)` over both
    prefix halves, in the keys' arithmetic: the write the memory takes (`carried`)."""
    return key * raised(last - gate, last_low - low)


def key_decayed_output(queries: Load, gates: Load, entered: Load, columns: int, scores, value, scale: float,
                       issue: Issue):
    """A chunk's output `[C, BV]`, fp32, a decay a key dimension: `scale (q exp2(g))_narrow S +
    tril(Aqk) v`, each key block's gated queries `queries(c) exp2(gates(c))` (`gates(c)` the prefix's
    halves, `raised`) rounded to their dtype
    against the memory entered `entered(c)` ('memory'), the scores `scores` (`Aqk`, scaled) rounded to the
    values' dtype ('local')."""
    out = jnp.zeros((scores.shape[0], value.shape[-1]), jnp.float32)
    for c in range(columns):
        query = queries(c)
        gated = (query * raised(*gates(c))).astype(query.dtype)
        out = out + issue("memory", gated, entered(c).astype(gated.dtype))
    out = out * scale
    index = jnp.arange(scores.shape[0])
    local = jnp.where(index[:, None] >= index[None, :], scores, 0.0).astype(value.dtype)
    return out + issue("local", local, value)
