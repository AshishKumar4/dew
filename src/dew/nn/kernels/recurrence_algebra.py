"""The recurrence template's algebra (bible §6; the template design's §3): the chunked form of the update
as named, pure functions on values, which every implementation of the chunked op calls and none
rewrites. This rung covers `S' = T S + W` with T a diagonal decay times the delta rule's erase, E the
identity, and a chunk's transition `(T, W)` dense (`transition`); the value-side decay E and the factored
transitions follow with the families that need them. A token updates and reads, per head over a state
`S` `[Dk, Dv]`,

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q

(`recurrence.update` is its one-token form). Within a chunk of C tokens:

- `decays(g, documents) -> (gc, pairs)`: the log decay cumulated within the chunk and each pair of positions'
  decay `exp(gc[s] - gc[t])`, its exponent compensated;
- `intra(k, kb, pairs) -> A`: the delta rule's strictly lower key-key system, `-(beta k k^T)` weighed by the
  pairs;
- `factor(A) -> L`: its triangular inverse `(I - A)^-1` (WY/UT), skipped where nothing is corrected;
- `apply(L, kb, vb, entering) -> (W, U)`: the corrected write's two products, `L (beta k exp(gc))` and
  `L (beta v)`;
- `carry(S, step) -> (S', (S, v_new))`: one chunk across the memory, the corrected values `v_new = U - W S`
  and the write `exp(gc[-1]) S + (k exp(gc[-1] - gc))^T v_new`;
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

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from dew.nn.precision import compensated_add, compensated_difference, float32_einsum, float32_matmul

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
    as torch's are, not TF32 (`dew.nn.precision.float32_matmul`, ledger row 1327); the portable math takes
    its operands in at least float32 (`recurrence_chunk.chunked`)."""
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
    Returns the inclusive cumulative sum `gc` over C, `[..., C, F]`, and
    `decay[..., s, t, f] = exp(gc[s, f] - gc[t, f])` for s >= t, zero above
    the diagonal, `[..., C, C, F]`.

    The references subtract two fp32 cumulative sums, so every exponent
    carries the rounding of the chunk's whole accumulated magnitude, which
    exp turns into a relative error on every decay. Here the cumulative sum
    is compensated (`compensated_add`) and the difference takes both
    halves, so an exponent is as precise as its own range. On
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
    diff = compensated_difference((hi[..., :, None, :], lo[..., :, None, :]),
                                  (hi[..., None, :, :], lo[..., None, :, :]))
    # Masked before exp, as the references do. The unused positive
    # differences can overflow, and an outer where alone leaves 0 * inf in
    # the decay gradient.
    diff = jnp.where(inclusive, diff, 0.0)
    return hi, jnp.where(inclusive, jnp.exp(diff), 0.0)


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
    gc, decay = decays(g, documents)
    return (gc, decay), (_cumulated(dg, documents)[0], linear_call(_ranged, _ranged_transpose, decay, dg))


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
    """One chunk across the memory `state` `[..., Dk, Dv]`: `step` is the chunk's keys, `W`, `U`, `gc`, the
    decay's last row `last` (`exp(gc[-1] - gc)`, zero for a packed chunk's earlier documents) and `kept`
    (one where the memory entering passes through, zero where a document starts in the chunk). Returns
    the memory leaving and `(the memory entering, v_new)`."""
    k_i, w_i, u_i, gc_i, last_i, kept_i = step
    v_corrected = u_i - issue("corrected", w_i, state)
    return (state * jnp.swapaxes(jnp.exp(gc_i[..., -1:, :]), -1, -2) * kept_i
            + issue("write", jnp.swapaxes(k_i * last_i, -1, -2), v_corrected), (state, v_corrected))


def readout(query, entering, entered, key, pairs, corrected, issue: Issue = portable):
    """A chunk's output `[..., C, Dv]`: the memory it entered read through the decayed queries, and the
    chunk's own pairs over its corrected values (the inclusive lower `pairs` read the token itself)."""
    return (issue("memory", query * entering, entered)
            + issue("local", paired("scores", query, key, pairs, issue), corrected))


def compose(first, second, issue: Issue = portable):
    """Two spans' transitions as one: each a pair `(T [..., Dk, Dk], W [..., Dk, Dv])` taking the memory
    entering to `T S + W`; `second` after `first` is `(T2 T1, T2 W1 + W2)`, associative, so a scan over
    spans may run as a tree."""
    (t1, w1), (t2, w2) = first, second
    return issue("compose", t2, t1), issue("compose", t2, w1) + w2


def transition(key, w, u, gc, last, kept, issue: Issue = portable):
    """A chunk's transition `(T, W)` (`compose`): with `carry`'s operands,
    `T = exp(gc[-1]) kept - (k last)^T W` and `W = (k last)^T U`, so `carry`'s memory leaving is `T S + W`.
    Packed documents keep the memory entering only where `kept` because the operands are built so
    (`recurrence_chunk.chunked`): `W` is zero outside the document the memory entering belongs to and
    `last` outside the chunk's last document; `kept` alone gates only the decay's diagonal."""
    decayed = jnp.swapaxes(key * last, -1, -2)
    dk = key.shape[-1]
    diagonal = jnp.eye(dk, dtype=gc.dtype) * jnp.swapaxes(jnp.exp(gc[..., -1:, :]), -1, -2) * kept
    return diagonal - issue("corrected", decayed, w), issue("write", decayed, u)
