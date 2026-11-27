"""The recurrence template's algebra (docs/design/kernel-templates.md §3): the chunked form of the update as
named, pure functions on a chunk's tiles, which the template's launches call and none rewrites. A token
updates and reads, per head over a state `S` `[Dk, Dv]`,

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q

(`recurrence.update` is its one-token form). The portable route is the same forms derived from that update
(`dew.algebra.scan`, run by the template's plain-JAX mainloop, `recurrence_chunk.xla_mainloop`), which issue
their products through `portable` and read a chunk's memory through `remembered`. Across a sequence's shards,
`compose(first, second)` joins two spans' transitions `S -> T S + W` (`Span`) associatively, and
`entering(span, axis)` gives the memory each shard of a sequence enters its span with, their exclusive prefix
across the mesh axis. Packed documents restart the decays at each document's first position (varlen).

The algebra declares its products and issues none (the design's layering): every product goes through the
caller's `issue(name, left, right)`, the mainloop's, by its name, which its launch's issuer dispatches to the
buffers it stages that product's operands in; the portable math's issuer is XLA's (`portable`), a kernel's its
own dot."""

import functools
import inspect
import math
import operator
from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn import derivatives
from dew.nn.float_bits import TWO_FACTOR_REACH, exact_power2, raised_exactly
from dew.nn.kernels.generation import linear_map, unbroadcast
from dew.nn.kernels.options import program
from dew.nn.kernels.templates.product import Blocks, Exact, Support, Transposed
from dew.nn.precision import (
    at_least_fp32,
    compensated_add,
    compensated_sum,
    exact_float32,
    float32_einsum,
    float32_matmul,
    power2,
    renormalized_add,
    rounding_alone,
    two_product,
    two_sum,
    uncontracted,
)

Issue = Callable[[str, "Operand", "Operand"], jax.Array]
"""A mainloop's product: `issue(name, left, right)` is `left @ right` over the last two axes, in the
mainloop's own dot, `name` the product its launch's issuer stages it as."""

STATE_PRECISION = exact_float32(None)
"""The precision the portable math's products take: a chunk's products carry the recurrent state, whose
rounding every later chunk reads compounded, so IEEE float32 (`dew.nn.precision.exact_float32`) whatever JAX's
configured precision, whose default runs an fp32 product as TF32 on a GPU (the derived scan's at post81 on an
A100: 1823 times the fp32 recurrence's distance from float64, M-JS-96). It is the reference the template's
forms are held to; TF32 stays where the run's default numerics put it, in the ports and the fast forms. A
`Precision`, not a dot algorithm derived for it (`dew.algebra.error.product_precision`): XLA routes a dot to
an emitter by its shape, and its CPU's elemental emitter takes no named algorithm (row 5473)."""


def portable(name: str, left, right):
    """XLA's product, the portable math's issuer: `left @ right` at `STATE_PRECISION`, summed as Dew sums a
    CPU's float32 products (`float32_matmul`, which `accumulated`); the portable math takes its operands in
    at least float32 (`recurrence_chunk.xla_mainloop`), a `Transposed` one transposed."""
    del name
    left, right = (jnp.swapaxes(x.value, -1, -2) if isinstance(x, Transposed) else
                   x.value if isinstance(x, Exact) else x for x in (left, right))
    return float32_matmul(left, right, STATE_PRECISION)


def portable_einsum(subscripts: str, *operands):
    """`portable`'s product written as an einsum (`float32_einsum`) at `STATE_PRECISION`: the portable math's
    contractions that carry or read the recurrent state (the derived scan's pairs, a selective scan's read, a
    sharded span's entering memory) where no `left @ right` states them."""
    return float32_einsum(subscripts, *operands, precision=STATE_PRECISION)


def suffix_sums(x: jax.Array) -> jax.Array:
    """Each position's sum over its chunk's positions at and after it, along `x`'s second-last axis (a chunk's
    tokens), in at least float32 (a float64 truth's own): a reversed cumulative sum, a term no position's sum
    takes meeting no arithmetic of it, where a product with the chunk's upper triangle of ones multiplied it
    by zero, a nonfinite one NaN in every sum (a turned gradient's entering-memory term of 2^126 times its
    cotangent, `templates.recurrence.turned`). Kimi delta attention's decay gradient's sums run in its own
    pairs' launch (`own_pairs._suffix_summed`)."""
    x = _wide(x)
    return jnp.flip(jnp.cumsum(jnp.flip(x, -2), axis=-2), -2)


def sums_before(x: jax.Array) -> jax.Array:
    """Each position's sum over its chunk's positions before it, along `x`'s second-last axis, in at least
    float32 (`suffix_sums`' exclusive prefix twin): a cumulative sum shifted a position, each
    sum of the earlier terms alone, a later term meeting none of its arithmetic, where an inclusive sum less
    its own term lost a small sum beside a large term to its rounding (a chunk's last writer's whole weight
    beside its earlier ones' exp(-30) each, Mamba-3's turns' gradient's worst row 3.75 from float64 over its
    size)."""
    summed = jnp.cumsum(_wide(x), axis=-2)
    return jnp.concatenate([jnp.zeros_like(summed[..., :1, :]), summed[..., :-1, :]], axis=-2)


def _wide(x: jax.Array) -> jax.Array:
    """`x` in at least float32, a float64 truth's own."""
    return x.astype(at_least_fp32(x.dtype))


LOG2_E = math.log2(math.e)
"""log2(e) to double precision, the prefix's base-2 scale (fla's RCP_LN2 is 1.93e-8 short of it)."""


@program
def prefix(g):
    """The decay's prefix within each chunk, compensated, in base 2: `g` `[..., C, H]` (natural log) to `(hi,
    lo)`, each token's `g log2(e)` an exact pair (its product with log2(e)'s fp32 rounding error-free,
    `precision.two_product`, plus its product with the rounding's remainder) summed compensated
    (`precision.renormalized_add` over the terms fenced once), so a difference of two positions is as precise
    as its own range. One variadic reduce-window over each position's chunk so far, which XLA fuses into the
    launches' layout of the pair: as an associative scan its log-depth levels were XLA launches of their own,
    fourteen of the gated delta net's 364 us forward on a GH200, 45 us of it (LF-GDN-NCU-gh200; four fusions
    in all compiled for an sm90 with the window, jsf-prefix-rw), the pair as precise either way (1.2e-14 to
    1.5e-14 relative from float64 over decays of 1e-3 to 30 a token). One program JAX traces once for `g`'s
    type: its exact sums traced afresh, each layer's forward, its kept forward in the backward and its
    gradient's preparation each paid them, 226 s of Qwen3.5-4B's 1208 s ahead-of-time compile (row 3142,
    FS2440-4B-STACKS)."""
    scale = np.float32(LOG2_E)
    scaled, error = two_product(g, jnp.full_like(g, scale), bit_split=True)
    error = error + g * float(np.float32(LOG2_E - float(scale)))
    axis, size = g.ndim - 2, g.shape[g.ndim - 2]
    window = tuple(size if dim == axis else 1 for dim in range(g.ndim))
    padding = tuple((size - 1, 0) if dim == axis else (0, 0) for dim in range(g.ndim))
    zero = jnp.zeros((), g.dtype)
    # As the one compensated pair reduction sums (`precision.summed_pairs`): each term's hi read as one value
    # where it is made (`uncontracted`), the window's additions marked to round alone, and the pairs combined
    # by `renormalized_add`, where a combine fencing its operands at every step (`compensated_add`) was a
    # second way of the same sum (row 10722).
    with rounding_alone():
        (scaled,) = uncontracted(scaled)
        return jax.lax.reduce_window((scaled, error), (zero, zero), renormalized_add, window,
                                     (1,) * g.ndim, padding)


def prefix_terms(g) -> tuple:
    """Each token's term of `prefix`'s pair, `g log2(e)` as a Dekker product and its residual plus `g` times
    the constant's own rounding, an fp32 pair `(hi, lo)` `g`'s shape: the factor split where it is known
    (`two_product` of a constant), `g` by its bits in the launch."""
    scale = np.float32(LOG2_E)
    scaled, error = two_product(g, scale, bit_split=True)
    return scaled, error + g * float(np.float32(LOG2_E - float(scale)))


def scanned_prefix(g, stage: Callable[[tuple], Callable[[int], tuple]], *, chunk: int) -> tuple:
    """`prefix`'s pair of a chunk's decay tile `g` `[C, BK]` (natural log) within a launch, its compensated
    sums taken by doubling (Hillis-Steele's scan: `log2(C)` steps, each row's running pair added to the one
    `e` rows above it, `e` doubling), each step's pair staged by `stage(pair)` in shared memory, whose reader
    gives it shifted down `e` rows, zeros above (a padded tile's): the same exact terms as `prefix`'s, summed
    in another association, a difference of two positions as precise as its own range either way. The
    prefix a launch's own: as its own XLA scan, thirteen launches took 0.30 ms of Kimi Linear's 1.57 ms
    template forward on an H100 (prefix-probe-h100-9e3be6213, kda-profile-h100-69233abb8). Each step adds the
    pairs as staged, its own rows read back (`stage(pair)(0)`) beside the shifted ones: a token's rounded
    product `g log2(e)` held in its registers met the first step's TwoSum, where the compiler contracted the
    product into the sum (`fma(g, log2(e), shifted)`), its rounding counted twice beside the Dekker residual
    that keeps it: the native prefix lay 1.9e-6 from exact where the interpreted one lay 1e-12 (M-MC-9), a
    key's gated write one bf16 step from the witness's at a tie (M-MC-8)."""
    pair = prefix_terms(g)
    shift = 1
    while shift < chunk:
        staged = stage(pair)
        pair = compensated_add(staged(0), staged(shift))
        shift *= 2
    return pair


def raised(whole, part: jax.Array | float = 0.0):
    """`exp2(whole + part)` of a compensated base-2 exponent (`part` zero where it is one fp32: a decay a key
    dimension's rounded prefix, `intra._prefix`) to fp32's resolution at any size: its integer
    `n = floor(whole)` an exact power of two (`precision.power2`) and its
    fraction `(whole - n) + part`, near [0, 1), through exp2. The exponent rounded to one fp32 lost up to half
    its ulp, 1.9e-6 near -43 (a pair of tokens at -30 a token), a decay's relative 1.3e-6, which put the
    port's decay gradient under decays near zero 7 to 10 times the token-by-token recurrence's distance from
    float64; and exp2 of a whole exponent, which a CPU takes as `exp(x ln 2)`, rounds its product with ln 2
    the same way. Its power of two within `TWO_FACTOR_REACH` (`power2`'s two factors, both normal powers): the
    fraction's exp2 lies in [1, 2), so `n` past fp32's exponents gives zero or infinity at any reach, and the
    two factors held an A100's query_key launch spill-free where three spilled it (ledger 2519), each set
    from `n` as a float (`power2`'s `integral`)."""
    n = _floor(whole)
    return power2(jnp.exp2((whole - n) + part), n, reach=TWO_FACTOR_REACH, integral=True)


def _floor(x):
    """`floor(x)` of fp32 `x` from its truncation toward zero, the same value (a zero's sign aside): every
    fp32 of magnitude 2^23 or more is an integer, and infinities and NaNs are their own floor, so their
    truncation, whatever its conversion gives past an int32 or of a NaN, is never taken: the conversion's
    own clamps (FS-CONV-SASS1) bound it, where a clip to 2^23 before it issued two more an element. Mosaic
    GPU lowers `jnp.floor` too (one cvt.rmi since post81), but there an H100's write-gated forked training
    certified a consumer past its first rung and its fp32 turned training step held in no config's compile
    (theme/jsf-intradiet e04820c37), as the sm100 Kimi delta attention two-role training launch spilled 8
    bytes at its residency's 248-register share (audit K2-1, jsf-calib-floor): ptxas is not monotone
    between them."""
    truncated = x.astype(jnp.int32).astype(x.dtype)
    floor = jnp.where(truncated > x, truncated - 1.0, truncated)
    return jnp.where(jnp.abs(x) < 2.0 ** 23, floor, x)


def decayed(rows, rows_lo, columns, columns_lo):
    """`exp2(rows - columns)` `[R, C]` of a compensated prefix's halves at the tile's rows and columns, each
    side's vector separated (`separated`, `between`): the tile takes the wholes' difference, exact, the
    fractions', and one exp2 scaled by a power of two."""
    (row_whole, row_fraction), (column_whole, column_fraction) = (separated(rows, rows_lo),
                                                                  separated(columns, columns_lo))
    return between((row_whole[:, None], row_fraction[:, None]),
                   (column_whole[None, :], column_fraction[None, :]))


def prefix_difference(later: tuple, earlier: tuple) -> tuple:
    """`later - earlier` of a decay's prefix at two positions (broadcasting), each its limbs (`(hi, lo)` a
    compensated pair's, `(hi,)` a decay a key dimension's rounded sum alone, `Call.prefix_limbs`), as a pair
    whose sum it is exactly: one limb's difference rounded and what the rounding dropped (`two_sum`), a
    pair's compensated (`compensated_add`), so no rounding of the difference reaches what it raises."""
    if len(later) == 1:
        (left,), (right,) = later, earlier
        return two_sum(left, -right)
    return compensated_add(later, tuple(-half for half in earlier))


def range_decay(later: tuple, earlier: tuple):
    """`exp2(later - earlier)` of a decay's prefix at two positions (broadcasting), each its limbs: `raised`
    of their difference, exact (`prefix_difference`): a range's decay, which from the difference rounded to
    one fp32 lost a fraction's last bits at the prefix's size (1e-6 of the decay near -43). The template's
    ranges, on its limbs' vectors; separating both sides for `between` held more registers, one of sm80's
    training launches spilling 48 bytes (a port's prefix, stored separated, takes `between`)."""
    return raised(*prefix_difference(later, earlier))


PAIR_CEILING = 120.0
"""The largest power of two `between` scales by, where it holds: the pairs past a token (a decay's inverse,
which every caller masks) stay finite."""


def separated(hi, lo: jax.Array | float = 0.0):
    """A compensated base-2 exponent `(hi, lo)` as its whole part and fraction `(n, f)`: `n` the integer
    `floor(hi)` (`hi` itself past 2^23, where every fp32 is one), `f` what `hi` holds past it and `lo` (zero
    where the exponent is one fp32, a decay a key dimension's rounded prefix), the form whose differences are
    exact at any size (`between`). Once a call, where its prefix is cumulated."""
    whole = _floor(hi)
    return whole, (hi - whole) + lo


def between(later, earlier):
    """`exp2(later - earlier)` of two separated exponents (`separated`, broadcasting), to fp32's resolution:
    exp2 of the fractions' difference times the wholes' difference's exact power of two (`exact_power2`), zero
    where the wholes differ by -127 or less (its zero exponent bits; at -126 and -125 the product may lie
    among fp32's subnormals, which a GPU's fast math flushes) and the difference held at `PAIR_CEILING` above.
    A difference of the rounded sums lost the fraction's last bits at the exponent's size, 1e-6 of the decay
    near -43, which put a port's decay gradient under decays near zero 9 to 14 times the token-by-token
    recurrence's error from float64 (ledger 2098). The fractions' difference stays within two, its exp2 within
    [1/4, 4), for exponents under 2^24, where a fraction's remainder `lo` is at most one: a chunk's prefix
    (its 64 tokens' at most 2^18 each). Truncating the difference into the scale, for any size, cost the
    template's forward 1% on an H100 and 3% on an A100 (JS-954-57bf6dc05). Every pair's decay whose exponent
    may be either sign, the ports' of their separated prefix and the template's of its halves (`decayed`), and
    the ports' ranges; a decay whose exponent is at most zero is `decay`'s; the template's ranges
    (`range_decay`) and a decay from the prefix's start are `raised`'s."""
    (whole, fraction), (other, its_fraction) = later, earlier
    return jnp.exp2(fraction - its_fraction) * exact_power2(jnp.clip(whole - other, -127.0, PAIR_CEILING))


def decay(later, earlier):
    """`between` of a `later` exponent at most `earlier` (a decay, its exponent at most zero: a log decay at
    most zero a token, the chunked forms' contract), held below alone: no ceiling a decay never reaches. The
    intra walk's factored pairs and own bands and their gradients' (`key_block_pairs`, `key_diagonal_bands`,
    `key_decayed_pairs_back`, `key_diagonal_pairs_back`), where the exponent's two clips and a zero's select
    were 5.5k of a compute warp's 28.9k issues in a B200's Kimi Linear intra walk once its prefix was
    separated (modelled at 191cb7975e4)."""
    (whole, fraction), (other, its_fraction) = later, earlier
    return jnp.exp2(fraction - its_fraction) * exact_power2(jnp.maximum(whole - other, -127.0))


def first_decays(dg, entered, d_entered, first, memory, takes, live, passed=None):
    """`dg` `[N, ...]`, a decay a key dimension's gradient summed back over each chunk (a token or a laid
    row a row), with each sequence's first token's taken from the memory its decay scales: the memory it
    enters (`entered` `[M, ..., Dv]`, its index `memory` among them) against that memory's gradient
    (`d_entered`), summed over the values, where the sequence takes a memory (`takes`), exactly zero where it
    enters none; a sequence's first token at row `first`, one with no tokens (not `live`) dropped. Summed
    back, that token's gradient is every term of its chunk, which cancel to fp32's residue where the memory
    is zero: rows of float64 zero 1.5e-7 from it against fla's 3.3e-9 (row 5194, M-JS-26-a100), 2.7e-11 on a
    zero initial memory, as a float64 recurrence holds it exactly. fla_kda's port and the template read
    it alike. Where `passed` (`dg`'s shape) is given, a token's value only where it holds: a clamped log
    decay's derivative zero (`templates.recurrence.mainloop.clamped_gradient`)."""
    shared = jnp.sum(entered.astype(jnp.float32) * d_entered, axis=-1).reshape(-1, *dg.shape[1:])
    taking = takes.reshape(-1, *(1,) * (dg.ndim - 1))
    value = jnp.where(taking, shared[memory], 0.0)
    index = jnp.where(live, first, dg.shape[0])
    if passed is not None:
        value = jnp.where(passed[jnp.minimum(index, dg.shape[0] - 1)], value, 0.0)
    return dg.at[index].set(value.astype(dg.dtype), mode="drop")


def _remembered(decayed, entered, issue: Issue = portable, name: str = "memory"):
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


LEFT_CHUNK = 8
"""The read width each piece of a readout's gradient in its decayed operand sums (`_remembered_left`)."""


def _remembered_left(entered, cotangent, issue: Issue, name: str, shape: tuple[int, ...]):
    """`cotangent @ entered^T`, the readout's gradient in its decayed operand: over the read width in pieces
    of LEFT_CHUNK, each piece's product issued and the pieces summed rounded once (`compensated_sum`), so
    its rounding is a piece's whatever order a backend's product sums in. As one product of the whole width,
    XLA:CPU's order put Mamba-2's C gradient 1.61 times the token scan's distance from float64 where a
    chunk's decays reach e^-45 (row 4543). Summed back to the decayed operand's `shape` where `entered`'s
    batch axes broadcast it (`unbroadcast`)."""
    width = entered.shape[-1]
    pieces = [issue(name, cotangent[..., start:start + LEFT_CHUNK],
                    jnp.swapaxes(entered[..., start:start + LEFT_CHUNK], -1, -2))
              for start in range(0, width, LEFT_CHUNK)]
    left = pieces[0] if len(pieces) == 1 else compensated_sum(jnp.stack(pieces), (0,))
    return left if left.shape == shape else unbroadcast(left, shape)


def _remembered_tangent(issue, name, primals, tangents):
    """The product's tangent, linear in each side: its `entered` side with the transpose
    `_remembered_transpose`, its decayed side with `_remembered_left`, in forward and reverse mode alike."""
    decayed, entered = primals
    d_decayed, d_entered = tangents
    read = linear_map(lambda decayed, entered: issue(name, decayed, entered),
                      functools.partial(_remembered_transpose, issue=issue, name=name), decayed, d_entered)
    moved = linear_map(lambda entered, decayed: issue(name, decayed, entered),
                       functools.partial(_remembered_left, issue=issue, name=name, shape=d_decayed.shape),
                       entered, d_decayed)
    return remembered(decayed, entered, issue, name), moved + read


remembered = derivatives.contracted("test_the_chunked_delta_rule_holds_every_order")(
    derivatives.declared_tangent(derivatives.Protected(), _remembered, _remembered_tangent,
                                 nondiff_argnums=(2, 3)))


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


# A chunk's tiles. The functions below are the same algebra on one chunk of one head, as a mainloop's
# program holds it: sub-chunks of SUB rows and column blocks, each operand a tile or a `Load`, which a
# GPU mainloop reads from memory where it is used and the portable mainloop slices. Every mainloop calls
# them as they are, so two mainloops at the same tiles compute the same values.

SUB = 16
"""Rows of a sub-chunk: the diagonal blocks the triangular inverse is built from, the reduction contract's,
not a tile's: fla's BC, whose substitution order the template's and the portable math's inverse take on every
GPU and every family, so check 1 pairs their numbers with the port's and the mainloops with each other (a
family's product rows set its tiles, not this: wgmma takes 64 rows, mma.sync 16)."""

class Staged(Protocol):
    """A tile staged once in shared memory (`key_block_pairs`' and `key_decayed_pairs_back`' `staged`), a
    block of whose rows a product reads at a time."""

    def rows(self, start, count: int) -> "Tile": ...


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
    lower), where `tile_indices`'s two stay live."""
    rows, cols = tile_indices(shape)
    return rows - cols


def tile_indices(shape) -> tuple[jax.Array, jax.Array]:
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
    two apart. The decays are the compensated prefix's (`decayed`, a representation the template keeps);
    the tile's algebra over them is its declaration's derived forms (`derived_tile`, `pairs_of`)."""
    decays = decayed(gate, low, column_gate, column_low)
    return derived_tile("pairs").mapped({"products": product, "decays": decays, "scales": beta})


def pairs_of(products, decays, scales):
    """A chunk's strictly lower pairs `tril(products decays scales_s, -1)` `[C, C]` (theme A's tile
    declaration of the delta rule's system, `derived_tile`): each pair's product times its decay, its row's
    token's scale (its write gate), zero on and above the diagonal."""
    return jnp.where(_lags(products.shape) > 0, products * decays * scales[:, None], 0.0)


def scores_of(products, decays, scales, apart: int = 0):
    """A chunk's own pairs' scores `tril(products decays scales_t)` `[C, C]` (theme A's tile declaration of
    the readout's scores, `derived_tile`): each pair's product times its decay and its column's token's scale
    (the keys' side's write gate, or 1), zero above the diagonal; a readout `apart` from its group's writes
    (`chunk_scores`') keeps the pairs of later groups of `apart` tokens alone (`_later_groups`)."""
    kept = _later_groups(products.shape, apart) if apart else _lags(products.shape) >= 0
    return jnp.where(kept, products * decays * scales[None, :], 0.0)


TILES = {"pairs": pairs_of, "scores": scores_of}
"""The recurrence's tile declarations (`derived_tile`): its system's pairs and its readout's scores."""


def _later_groups(shape, group: int, *, columns_later: bool = False) -> jax.Array:
    """Whether each row of a tile `[R, C]` is in a later group of `group` consecutive tokens than its column
    (its column in a later one than its row where `columns_later`, the transposed tile's, formed as it is,
    no tile transposed): the pairs a readout apart from its token's own writes keeps
    (`recurrence.Hooks.reads` 'apart', a token's ranks `group` tokens, mamba_ssm's Mamba-3 MIMO `i // R > j
    // R`, `mamba3_mimo_fwd.py`; strictly lower at one). `group` a power of two (the template's, dividing
    its role's rows, `template.Mosaic.eligible`): each index's group its bits past the group's, shifted, no
    division."""
    shift = group.bit_length() - 1
    if group != 1 << shift:
        raise ValueError(f"a group of {group} tokens is no power of two")
    rows, cols = (index >> shift for index in tile_indices(shape))
    return cols > rows if columns_later else rows > cols


@functools.cache
def derived_tile(name: str, apart: int = 0):
    """The derived forms of the tile declaration `name` (`TILES`, the fast class), the template's: a mainloop
    forms the products and its compensated prefix's decays (representations it keeps) and takes the tile's
    algebra over them from these forms, the hand forms' own arithmetic in their order (FS-DERIVED-ID1/2). A
    readout `apart` from its group's writes is the declaration's own form at that group (`scores_of`'s
    `apart`); a tile declaring none refuses it."""
    from dew.algebra import SATURATION, derive
    axes = {"products": ("rows", "columns"), "decays": ("rows", "columns"),
            "scales": ("rows",) if name == "pairs" else ("columns",)}
    declared = TILES[name]
    if apart:
        if "apart" not in inspect.signature(declared).parameters:
            raise ValueError(f"the {name} tile declares no readout apart from its group's writes")
        declared = functools.partial(declared, apart=apart)
    return derive.derive(derive.Reduction(declared, axes, walked="columns", exponent="e"), accurate=False,
                         saturation=SATURATION)


Supported = Callable[[str, "Operand", "Operand", Support], jax.Array]
"""A product of a `Support`'s terms alone (`operands.Issuer.supported`): `doubled`'s merges, whose terms lie
in its blocks."""


def doubled_level(a, inverse, rows, cols, level, issue: Issue, *, supported: Supported | None = None):
    """Join the width-`s` blocks of `X_s` at `s = 2**level`: `X_2s = X_s + (M_s * (X_s A)) X_s`.

    `M_s` takes each odd block's rows over the preceding block's columns: `factor`'s `X2 A21 X1` there,
    zero elsewhere. The two `"merge"` products use the cached supports when `supported`: `X_s` block
    diagonal, `A` strictly lower, and joins inside their blocks.
    """
    row_block, col_block = rows >> level, cols >> level
    lower = ((row_block & 1) == 1) & (col_block == row_block - 1)
    if supported is None:
        return inverse + issue("merge", jnp.where(lower, issue("merge", inverse, a), 0.0), inverse)
    joining, joined = _merges_support(a.shape[-1], 1 << level)
    joins = jnp.where(lower, supported("merge", inverse, a, joining), 0.0)
    return inverse + supported("merge", joins, inverse, joined)


def doubled(a, issue: Issue, *, supported: Supported | None = None):
    """`factor`'s whole-tile inverse `(I - A)^-1` of a strictly lower `A`, starting at two-wide blocks and
    calling `doubled_level` until they span the tile. A mainloop stages `A` once. A Mosaic warpgroup's 64
    rows hold no block of their own; elementwise steps take 32 entries a thread, so the 16 steps of
    forward substitution within each 16-row block cost more than the three product levels that replace
    those steps. Each level uses `factor`'s same terms.
    """
    rows, cols = tile_indices(a.shape)
    inverse = jnp.where(rows == cols, 1.0, 0.0) + jnp.where((rows % 2 == 1) & (cols == rows - 1), a, 0.0)
    for level in range(1, (a.shape[-1] - 1).bit_length()):
        inverse = doubled_level(a, inverse, rows, cols, level, issue, supported=supported)
    return inverse


@functools.cache
def _merges_support(size: int, width: int) -> tuple[Support, Support]:
    """`doubled`'s two merges' supports at blocks of `width` (`Support`): the inverse `X_s` lower-triangular
    in its `width`-blocks, `A` strictly lower, the joins `M_s * (X_s A)` read in each odd block's rows over
    the block before's columns; then those joins against the inverse, every output read (the sum's). Formed
    once a size and width, so every trace and part of a merge reads one support and its tiles
    (`Support.tiles`)."""
    rows, cols = np.indices((size, size))
    held = (rows // width == cols // width) & (rows >= cols)
    joins = ((rows // width) % 2 == 1) & (cols // width == rows // width - 1)
    masks = (held, rows > cols, joins, np.ones((size, size), bool))
    for mask in masks:
        mask.setflags(write=False)
    held, lower, joins, every = masks
    return Support(held, lower, joins), Support(joins, held, every)


def solved(system, issue: Issue, *, supported: Supported | None = None):
    """`(I + A)^-1 - I` `[C, C]` of a chunk's strictly lower system `A`: `doubled`'s inverse of `-A`, its unit
    diagonal taken off after the last level, as `weighted` takes it (it adds the diagonal outside its
    product), its merges of their supports alone where `supported` (`doubled`'s)."""
    rows, cols = tile_indices(system.shape)
    return doubled(-system, issue, supported=supported) - jnp.where(rows == cols, 1.0, 0.0)


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
    scaled = write_gated(tile, beta) if decay is None else write_gated(tile, beta) * decay
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
    rows, _ = tile_indices(values.shape)
    earlier = jnp.where(rows != row, values, jnp.zeros((), values.dtype))
    return [part * decay[i] + (issue("write", keys(i).T, earlier) + key_rows(i)[:, None] * value_row[None, :])
            for i, part in enumerate(state)]


def keys_gated_carried(state, keys: TileLoad, values, gate, decay, dtype, issue: Issue) -> list:
    """`carried` with the write's gate on the keys' side: `state[i] decay[i] + (keys(i) gate)^T values`, each
    key row scaled by its token's `gate` `[C]` (its write gate and its decay to the chunk's end) in fp32 and
    rounded to `dtype` for the product, the values (`values`, a tile) as they are: mamba_ssm's state write,
    `(B dt exp(dA_last - dA))^T x` (`_chunk_state_fwd`), whose rounding a 16-bit call's takes."""
    return [part * decay[i] + issue("write", Transposed(
        (keys(i).value().astype(jnp.float32) * gate[:, None]).astype(dtype)), values)
        for i, part in enumerate(state)]


def scalar_write(corrected_values, last, gate, last_lo, gate_lo):
    """A decay a head's write: the corrected values gated by `exp2(g_last - g)` over both halves."""
    return corrected_values * range_decay((last, last_lo), (gate, gate_lo))[:, None]


def chunk_scores(queries: Load, keys: Load, columns: int, gate, low, dtype, issue: Issue, *,
                 laid: tuple | None = None, column_scale=None, apart: int = 0):
    """A chunk's own pairs' scores `[C, C]` under a decay a head, `tril((q k^T) exp2(g_s - g_t))`, the key
    column blocks `queries(c)` and `keys(c)` summed into fp32 ('scores'), in `dtype` for `chunk_output`'s
    'local', which every value block of the chunk takes. `laid` is the prefix's halves `(gate, low)` as a
    column's vectors where a mainloop's layouts keep them apart from a row's (`system`'s). The pairs' decays
    come before the products, their exponents' working tiles done before the products' sum is live.
    `column_scale` `[C]`, where given, scales each key token's column before the scores are rounded (the
    write gate on the keys' side, mamba_ssm's `CB exp(dA_i - dA_j) dt_j`, `_chunk_scan_fwd`). A readout
    `apart` from its token's own writes (`recurrence.Hooks.reads` 'apart', its token's ranks `apart`
    consecutive tokens) keeps the pairs of later tokens alone (`_later_groups`)."""
    decay = decayed(gate, low, *((gate, low) if laid is None else laid))
    scores = jnp.zeros((gate.shape[0], gate.shape[0]), jnp.float32)
    for c in range(columns):
        scores += issue("scores", queries(c), keys(c).T)
    scales = jnp.ones((gate.shape[0],), jnp.float32) if column_scale is None else column_scale
    # The tile's algebra over its products and decays its declaration's derived forms (`scores_of`).
    tile = derived_tile("scores", apart)
    return tile.mapped({"products": scores, "decays": decay, "scales": scales}).astype(dtype)


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


def _laid_references(at: Callable[[int], tuple], index: jax.Array, chunk: int, sub: int) -> tuple:
    """The sub-chunks' common points laid over their rows, each half `[C, BK]` (`separated`): sub-chunk
    `s`'s first token's separated prefix (`at(s)`, a column's vectors) on its rows from its own on, a
    sub-chunk a rolled turn. Read at static rows, their addresses were hoisted to the kernel's start and held
    through it (260 of an H100's 475 registers at the peak)."""

    def laid(s, below):
        return tuple(jnp.where(index >= s * sub, half[None, :], under)
                     for half, under in zip(at(s), below, strict=True))

    return jax.lax.fori_loop(0, chunk // sub, laid, (jnp.zeros(index.shape, jnp.float32),) * 2)


def gated_up(prefix: tuple, reference: tuple, index: jax.Array, place) -> jax.Array:
    """`exp2(G - g_j)` `[C, BK]` for every token `j` before `place`, `G` the separated prefix at `place`
    (`reference`, a column's vectors) and `g_j` the block's (`prefix`, `separated`): each earlier key gated up
    to a sub-chunk's first token, its exponent at most zero (`decay`), zero from `place` on (selected, so a
    later token's exponent, past zero, weighs nothing). The factor a sub-chunk's pairs take every earlier key
    by (`key_block_pairs`, `key_decayed_pairs_back`), fla_kda's `exp(g_n - g_k)` of its inter-sub-chunk
    products."""
    return jnp.where(index < place, decay(tuple(half[None, :] for half in reference), prefix), 0.0)


def key_block_pairs(rows: Sequence[Callable[[], jax.Array]], keys: jax.Array, prefix: tuple,
                    at: Callable[[int], tuple], *, chunk: int, scales: Sequence[float],
                    sub: int = SUB) -> tuple[list[jax.Array], Callable[[int], jax.Array]]:
    """A chunk's pairs under a decay a key dimension, one key block's, as the operands of the products
    whose sum they are: `(gated, raised)`, each of `rows`' gated rows `[C, BK]` (each forming its row when
    called: the queries, the write-gated keys; fp32), and `raised(s)` sub-chunk `s`'s keys gated up `[C,
    BK]`. A caller stages each row set's gated rows once and each later sub-chunk's keys in turn, and
    issues for each sub-chunk `s` past the first one product a row set (`(name, lhs, rhs)`, 'pairs'): the
    `sub` rows of `s` of the staged rows (a `Staged` tile's `rows`) against the staged keys read
    transposed, `[sub, C]` a row set and sub-chunk, `sum_k r_ik k_jk exp2(g_ik - g_jk)` for row `i` in `s`
    and key `j` in an earlier sub-chunk, zero for keys from `s` on (a sub-chunk's own pairs are
    `key_diagonal_bands`', each its own decay, as fla's token-parallel diagonal). Each pair's decay splits
    at `s`'s first token `G_s`, as fla_kda's inter-sub-chunk products split it
    (chunk_kda_fwd_kernel_inter_solve_fused): `exp2(g_i - G_s) exp2(G_s - g_j)`, each exponent at most zero
    (`decay`), the rows gated down to `G_s` and every key gated up to it (`gated_up`). `prefix` is the
    block's separated prefix (`separated`) and `at(token)` it at the chunk's `token`, a column's vectors;
    each row set's gated rows are scaled by its of `scales` (the scores' the attention scale), so the sums
    a mainloop holds are its pairs scaled. A row set's products sum into one accumulator a sub-chunk, which
    a mainloop may hold across its key blocks, read once. The products take a sub-chunk's rows (mma.sync's,
    `intra.sub_chunked`): a whole chunk's rows masked to a sub-chunk's took 4.2 times fla's inter-sub-chunk
    launch on an A100 (1508 us against 356, M-JS-144-a100). A sub-chunk's own pairs factored too, taken on
    the tensor cores, put a packed document's first token's output, a row of 2.5e-5 where `q . k` cancels,
    at 0.83 of its size from float64 on an A100 against 7e-3 taken each its own decay (row 5194, M-JS-48w)."""
    index, _ = tile_indices(prefix[0].shape)
    # Every row gated down to its own sub-chunk's first token.
    down = decay(prefix, _laid_references(lambda s: at(s * sub), index, chunk, sub))
    # Scaled once gated down, so a large row its decay attenuates stays finite.
    gated = [row() * down * scale for row, scale in zip(rows, scales, strict=True)]
    return gated, lambda s: keys * gated_up(prefix, at(s * sub), index, s * sub)



def key_diagonal_bands(rows: list, shifted: Callable[[jax.Array], tuple], *, sub: int = SUB,
                       unit_pairs: bool = False) -> list:
    """Each sub-chunk's own pairs of one key block under a decay a key dimension, each its own decay, as the
    sub-chunks' diagonal blocks `[C, sub]` a product: its row `i`'s column `c` the pair `(i, j)` of the token
    `j` at `c` in `i`'s sub-chunk, `sum_k r_ik k_jk exp2(g_ik - g_jk)`, zero where `j` follows `i`
    (fla_kda's token-parallel diagonal, chunk_kda_fwd_intra_token_parallel), as the walk's sums take them in
    place (`own_pairs.intra_own`). `rows` the products' rows `[C, BK]` fp32, each against the
    block's keys; `shifted(e)` its keys and each token's own decay `(keys, decays)` (`token_decays`), each row
    `i` holding token `i - e`'s, any finite value where `i - e` leaves its sub-chunk (masked). A pair's decay
    is the product of its tokens' own, formed across the offsets as the token recurrence forms it (`pair_e =
    pair_(e - 1) a_(i - e + 1)`, at most `sub - 1` roundings, every factor at most one), each term the row
    times the key, then times the decay, in a rolled loop over the offsets, each offset's keys and decays read
    once for every product; each offset's sums selected into its band's column, written once, and a row before
    its offset selected out of every column once, at the end, not term by term nor offset by offset (its sum
    any value, a select's, never a product's: selected on every term, the mask was a quarter of a B200's own
    pairs' issues, an emulated B200 at b6a888850b2; an offset's two-part mask and the band's add on every
    offset were 12% of an emulated GH200's intra_own issues, M-JS-145h). Each term's decay
    from the separated prefix instead (`decay`: two differences, a
    clip, a power of two and an exp2), a B200's Kimi Linear training spent 0.96 ms of an emulated 2.86 in this
    form and its transpose, each issue-bound (an emulated B200 at 8aab1729e65). Summed over a chunk's key
    blocks, a launch writes its blocks in place (`own_pairs.intra_own`): placed a key block and an offset at
    a time, the `[C, C]` selects took an H100's Kimi Linear intra launch from 492 to 1982 us (M-JS-50d).
    Where the keys and rows are at most one in size (`unit_pairs`, `mainloop.Call.unit_pairs`), each key
    takes its pair's decay first instead, once for every product (`decayed_key`), and each row's term is a
    fused multiply-add of it (fla's token-parallel diagonal folds it so, into the key): all the decayed key
    loses is below normal in the exact term too. Each product's row decayed apart, a thread's own pairs
    issued twice the multiplies, an emulated GH200's intra_own issue-bound (M-JS-170). Other operands keep
    the row times the key first, which loses no term in range to the decay (row 7299; a decayed row lost
    2^-100 against a key of 2^60, base review 6ce025bc7 F1; a decayed key 2^-160 against a row of 2^100,
    #3703 review)."""
    index, _ = tile_indices(rows[0].shape)
    lane = jax.lax.broadcasted_iota(jnp.int32, (index.shape[0], sub), 1)
    token = jax.lax.broadcasted_iota(jnp.int32, lane.shape, 0)

    def column(e):
        # A row's pair `e` tokens back, at its column within the sub-chunk's block.
        return token % sub - e

    def offset(e, carried):
        bands, pair, following = carried
        keys, decays = shifted(e)
        pair = pair * following
        summed = [jnp.sum(band_term(row, keys, pair, unit_pairs=False), axis=1) for row in rows]
        return (tuple(jnp.where(lane == column(e), total[:, None], band)
                      for band, total in zip(bands, summed, strict=True)), pair, decays)

    def folded(e, carried):
        # Each key's decay folded in once for every product (`decayed_key`): a multiply an element and each
        # row's term a fused multiply-add, where keys and rows at most one in size (`unit_pairs`) make what
        # the decayed key loses below normal in the exact term too.
        bands, pair, following = carried
        keys, decays = shifted(e)
        pair = pair * following
        keyed = decayed_key(keys, pair)
        summed = [jnp.sum(band_term(row, keyed, None, unit_pairs=True), axis=1) for row in rows]
        return (tuple(jnp.where(lane == column(e), total[:, None], band)
                      for band, total in zip(bands, summed, strict=True)), pair, decays)

    zero, one = jnp.zeros(lane.shape, jnp.float32), jnp.ones(index.shape, jnp.float32)
    bands = jax.lax.fori_loop(0, sub, folded if unit_pairs else offset, ((zero,) * len(rows), one, one))[0]
    # Each column written once, by its offset's select; a column past its row selected out once, at the end.
    return [jnp.where(token % sub >= lane, band, 0.0) for band in bands]


def token_decays(prefix: tuple, earlier: tuple, *, sub: int = SUB) -> jax.Array:
    """Each token's own decay `exp2(g_i - g_(i - 1))` `[C, BK]` of a key block's separated prefix `prefix`
    (`separated`) and `earlier`, each row `i` holding token `i - 1`'s: a sub-chunk's own pairs' factors
    (`key_diagonal_bands`, `key_diagonal_pairs_back`), its exponent at most zero (`decay`). A sub-chunk's
    first token's is one, no pair of its own sub-chunk spanning it, whatever `earlier` holds there."""
    index, _ = tile_indices(prefix[0].shape)
    first = index % sub == 0
    return decay(prefix, tuple(jnp.where(first, half, before) for half, before in zip(prefix, earlier,
                                                                                       strict=True)))


OWN_REACH = 112
"""The most whole powers of two an anchored factor's exponent spans (`own_anchored`): within it `between` is
exact, past 120 it holds its scale and below -124 it may flush."""

OWN_BAND = 100
"""The magnitudes, as powers of two, every anchored operand's elements keep within or are zero as given
(`own_out_of_band`): both of an element's two bf16 parts normal, its narrowing losing no more of it than its
parts' 16 bits."""

MIDDLE = SUB // 2
"""A sub-chunk's token its own pairs' anchor is taken at (`own_anchors`): both factors within half a
sub-chunk's decays."""


def own_anchors(at: Callable[[int], tuple], shape: tuple[int, ...], sub: int = SUB) -> tuple:
    """Each row's sub-chunk's anchor, the separated prefix at its `MIDDLE` token (`at(token)`, a column's
    vectors), laid over its rows, each half `shape` `[C, BK]` (`_laid_references`): the point both factors of
    a sub-chunk's own pair are taken against (`own_anchored`), fla_kda's safe gate's shifted first token."""
    index, _ = tile_indices(shape)
    return _laid_references(lambda s: at(s * sub + MIDDLE), index, shape[0], sub)


def own_anchored(prefix: tuple, anchors: tuple) -> tuple[jax.Array, jax.Array]:
    """A key block's anchored factors `(exp2(g_i - A), exp2(A - g_i))` `[C, BK]` of its separated prefix
    `prefix` against its rows' anchors (`own_anchors`): a sub-chunk's own pair `(i, j)` decays by the first's
    on row `i` times the second's on key `j`, `exp2(g_i - g_j)` (`between`, its exponent either sign), so the
    own pairs are one product of rows and keys each scaled once (fla_kda's safe-gate diagonal, chunk_intra.py
    at 413ad2c), where `key_diagonal_bands` forms each pair's decay apart. The pairs past a token (`j > i`)
    take factors past one, which every caller masks."""
    return between(prefix, anchors), between(anchors, prefix)


def own_reach_exceeded(prefix: tuple, anchors: tuple) -> jax.Array:
    """Where a key block's anchored factors pass `OWN_REACH` (`own_anchored`), bool `[C, BK]`: a token's
    prefix more whole powers of two from its anchor than the reach, or a prefix not finite."""
    (whole, fraction), (other, its_fraction) = prefix, anchors
    return (jnp.abs(whole - other) > OWN_REACH) | ~jnp.isfinite(fraction + its_fraction)


def own_out_of_band(given: jax.Array, scaled: jax.Array) -> jax.Array:
    """Where an anchored operand's elements leave its band, bool, each its `scaled` value of its `given` one:
    not finite, or given nonzero and its magnitude past `2^OWN_BAND` or below `2^-OWN_BAND` (a value its two
    bf16 parts take less exactly, or a fast-math flush to zero, of a term whose pair may keep it)."""
    size = jnp.abs(scaled)
    return ~jnp.isfinite(scaled) | ((given != 0) & ((size < 2.0 ** -OWN_BAND) | (size > 2.0 ** OWN_BAND)))


def own_rows_failed(reached: jax.Array | None, operands: Sequence[tuple[jax.Array, jax.Array]] = ()
                    ) -> jax.Array:
    """The rows whose anchored products leave their guard, a row's vector `[C]` fp32, one where they do: an
    element of `reached` (`own_reach_exceeded`, `[C, BK]`; None: none) or of an operand `(given, scaled)` of
    `operands` out of its band (`own_out_of_band`), reduced along its row (a warp's own); a caller takes
    their largest across the warps once (a row's vector's share of scratch, where the whole tile's took a
    tile's)."""
    flags = ([] if reached is None else [reached]) + [own_out_of_band(*operand) for operand in operands]
    return jnp.max(jnp.where(functools.reduce(jnp.logical_or, flags), 1.0, 0.0), axis=1)


def own_compact(square: jax.Array, *, sub: int = SUB) -> jax.Array:
    """A chunk's `[C, C]` product's own sub-chunks' blocks compact `[C, sub]` (`key_diagonal_bands`' form, a
    row's column its earlier token's place in its sub-chunk's block): each row's pairs at or before it in its
    own block, zero past it (selected: an anchored pair past its row may be any value, `own_anchored`), the
    rest of the product unread."""
    rows, lanes = tile_indices((square.shape[0], sub))
    found = jnp.zeros(rows.shape, jnp.float32)
    for s in range(square.shape[1] // sub):
        found = jnp.where((rows // sub == s) & (lanes <= rows % sub), square[:, s * sub:(s + 1) * sub], found)
    return found


def own_bands(pairs: jax.Array, *, chunk: int, sub: int = SUB) -> jax.Array:
    """the own pairs' transpose, laid an offset a row: a call's pairs `[R, H * chunk]` (each chunk's square
    of a head's pairs) as their own pairs' bands `[H * sub, R]`, each head's row `e` the pairs `(i, i - e)` of
    every row `i` within its sub-chunk, zero past it: an offset's band a row a backward reads whole
    (`key_diagonal_pairs_back`'s strips), each column read out of a token's band a select and a sum an offset
    otherwise."""
    rows, heads = pairs.shape[0], pairs.shape[1] // chunk
    token = jnp.arange(rows)[:, None] % chunk
    offset = jnp.arange(sub)[None, :]
    found = jnp.take_along_axis(pairs.reshape(rows, heads, chunk),
                                jnp.clip(token - offset, 0, chunk - 1)[:, None, :], axis=2)
    banded = jnp.where((offset <= token % sub)[:, None, :], found, jnp.zeros((), pairs.dtype))
    return banded.reshape(rows, heads * sub).T


def write_gated(tile: jax.Array, strengths: jax.Array) -> jax.Array:
    """A tile `[C, D]` each row scaled by its token's write strength (`strengths` `[C]`, the delta rule's
    beta): the keys and values the delta rule writes, as their products take them (fla's `k * beta` before
    its `k k^T`, `v * beta` before its `A v`), gated on the operand, which scaling after the product is not
    at finite precision (kernel-templates.md §3). `gated` its one word's rounding, a native kernel's too."""
    return gated(tile, strengths[:, None])


def gated(word, strength):
    """A word scaled by its token's write strength, one rounding: `write_gated`'s arithmetic on any values
    that multiply (arrays, or a native kernel's emitted words, `templates.native.Word`)."""
    return word * strength


def band_term(row, key, pair, *, unit_pairs: bool):
    """A sub-chunk's own pair's term (`key_diagonal_bands`): the row times the key, then times the pair's
    decay; or where keys and rows are at most one in size (`unit_pairs`), the row times the decayed key `key`
    (`decayed_key`'s, formed once for every product's row), `pair` unread. On any values that multiply, as
    `gated`."""
    return row * key if unit_pairs else row * key * pair


def decayed_key(key, pair):
    """A key decayed by its pair's decay, rounded once (`band_term`'s `unit_pairs` key): what it loses below
    normal, a key at most one in size and every row at most one, is below normal in the exact term too."""
    return key * pair


def columns_before(found: jax.Array, start: int) -> jax.Array:
    """A sub-chunk's row block of a system's pairs `found` `[SUB, C]` (`key_block_pairs`', rows from `start`
    on) strictly below the chunk's diagonal across sub-chunks: its columns before `start`, zero from it on.
    `key_block_pairs`' are zero there as formed, but for a row whose write gate overflows, a NaN where it
    meets a zero factor."""
    return jnp.where(tile_indices(found.shape)[1] < start, found, 0.0)


def block_masked(block: jax.Array, *, strict: bool, scale: float, sub: int = SUB) -> jax.Array:
    """A block of own pairs (`key_diagonal_bands`') as its output takes it: a token's pair with itself (its
    row's column within the sub-chunk) dropped where `strict` (the system, strictly lower), else kept and
    `scale`d (the scores)."""
    lane = jax.lax.broadcasted_iota(jnp.int32, block.shape, 1)
    token = jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
    return jnp.where(lane < token % sub, block, 0.0) if strict else block * scale


def key_decayed_write(key, gates: tuple, last: tuple):
    """A chunk's keys gated to its end under a decay a key dimension, `k exp2(g_last - g)` of its prefix
    `gates` and the chunk's last token's `last` (each its limbs), their difference exact (`range_decay`), in
    the keys' arithmetic: the write the memory takes (`carried`)."""
    return key * range_decay(last, gates)


def key_gated(query, gates: tuple, scale: float):
    """A key block's queries gated by their decays to the chunk's start and scaled, `scale q exp2(g)`
    (`gates` the prefix's limbs, `raised`; `Call.prefix_limbs`), rounded to the queries'
    dtype: `key_decayed_output`'s left operand, as fla's chunk_gla output kernel scales its gated queries."""
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
# `(j, i]`) and each token's terms through the range from its chunk's start `[C]`, which `summed_pairs`
# and `summed_ranges` sum token by token.


def local_gradient(keys: TileLoad, queries: TileLoad, columns: int, gate, low, out_grads: TileLoad,
                   blocks: int, scale: float, dtype, issue: Issue, summed: Callable[[list], jax.Array], *,
                   emit: Callable[[int, jax.Array], None], laid: tuple | None = None,
                   apart: int = 0) -> None:
    """The corrected values' gradient from the chunk's own output, each value block `j`'s `[C, BV]` fp32 to
    `emit(j, dv)`: `A^T do` for `chunk_output`'s scores `A[t, s] = scale (q_t . k_s) exp2(g_t - g_s)`
    (`s <= t`), its transpose taken as `k q^T` ('local_t', the key blocks' sum in one accumulator,
    `summed`) and cast to the gradient's `dtype` once for every block's product with `out_grads(j)`
    ('local_dv'); `laid` and `apart` as `chunk_output`'s and `chunk_scores`'. The pairs' decays come before
    the products, as `chunk_scores`'."""
    column_gate, column_low = (gate, low) if laid is None else laid
    decay = decayed(-gate, -low, -column_gate, -column_low)
    product = summed([("local_t", keys(c), queries(c).T) for c in range(columns)]) * decay
    lag = _lags(product.shape)
    kept = _later_groups(product.shape, apart, columns_later=True) if apart else lag <= 0
    scores = jnp.where(kept, product * scale, 0.0).astype(dtype)
    for j in range(blocks):
        emit(j, issue("local_dv", scores, out_grads(j)))


def reverse_carried(grad: list, keys: TileLoad, weights: TileLoad | None, queries: TileLoad, out_grad, local,
                    gate, low, last, last_low, kept: list, scale: float, dtype, issue: Issue
                    ) -> tuple[list, jax.Array]:
    """One chunk of the memory's reverse recurrence (fla's dhu) under a decay a head: from the gradient of
    the memory leaving the chunk `grad`, its key slices `[K_i, BV]` fp32, the corrected values' gradient
    `dv = (k grad) exp2(g_last - g) + local` ('values_dv', the gradient cast to `dtype`) and the gradient of
    the memory the chunk entered, `grad exp2(g_last) + scale q^T (do exp2(g)) - w^T dv` ('reads', the
    decayed `do` fp32; 'weights_dv', `dv` cast to `dtype`). `last` and `last_low` are the chunk's last prefix
    halves as a row's vectors, `kept[i]` the memory's decay across the chunk as key slice `i`'s rows'
    (`exp2(g_last)` `[K_i, 1]`), `queries(i)` and `weights(i)` key slice `i`'s tiles. Without the delta rule
    (`weights` None) nothing was corrected, and the memory entered takes no `w^T dv`."""
    dv = functools.reduce(jnp.add, (issue("values_dv", keys(i), part.astype(dtype))
                                    for i, part in enumerate(grad)))
    dv = dv * range_decay((last, last_low), (gate, low))[:, None] + local
    read = out_grad.astype(jnp.float32) * raised(gate, low)[:, None]

    def slice_entered(i, part):
        # A slice's terms in one expression, so each slice's products are live alone (formed slice by slice
        # into a list first, every slice's partial sum stayed live past the products: 128 bytes spilled in
        # an H100's reverse walk of Qwen3.5-27B's gated delta net).
        entered = part * kept[i] + issue("reads", queries(i).T, read) * scale
        return entered if weights is None else entered - issue("weights_dv", weights(i).T, dv.astype(dtype))

    return [slice_entered(i, part) for i, part in enumerate(grad)], dv


def key_scores_back(scores: Tile, out_grads: TileLoad, corrected: TileLoad, blocks: int, issue: Issue, *,
                    emit: Callable[[int, jax.Array], None]) -> jax.Array:
    """A chunk and value head's local part of the backward under a decay a key dimension (fla_kda's dAv):
    the scores' gradient `dAqk = do v_new^T` `[C, C]` fp32, causal, summed over the value blocks `j` of the
    output's gradient `out_grads(j)` and the corrected values `corrected(j)` ('values_ds'), returned; and each
    block's corrected values' gradient from the chunk's own output, `scores^T do` `[C, BV]` fp32 to
    `emit(j, dv)` ('local_dv'), the scores (the intra launch's, scaled and decayed) as they are."""
    d_scores = functools.reduce(jnp.add, (issue("values_ds", out_grads(j), corrected(j).T)
                                          for j in range(blocks)))
    for j in range(blocks):
        emit(j, issue("local_dv", scores.T, out_grads(j)))
    return jnp.where(_lags(d_scores.shape) >= 0, d_scores, 0.0)


def key_reverse_carried(grad: list, keys: TileLoad, weights: TileLoad | None, queries: TileLoad,
                        out_grad: Tile, local, kept: list, dtype, issue: Issue) -> tuple[list, jax.Array]:
    """One chunk of the memory's reverse recurrence under a decay a key dimension (fla's dhu with its gated
    keys): `reverse_carried`'s, its keys already gated to the chunk's end (`key_decayed_write`'s) and its
    queries to its start and scaled (`key_gated`'s), so neither product takes a row's decay: the corrected
    values' gradient `dv = k grad + local` ('values_dv') and the memory entered's, each key slice decayed by
    its rows' decays across the chunk `kept[i]` `[K_i, 1]`, `grad kept + q^T do - w^T dv` ('reads' on the
    output's gradient as it is, 'weights_dv' on `dv` cast to `dtype`)."""
    dv = functools.reduce(jnp.add, (issue("values_dv", keys(i), part.astype(dtype))
                                    for i, part in enumerate(grad))) + local

    def slice_entered(i, part):
        entered = part * kept[i] + issue("reads", queries(i).T, out_grad)
        return entered if weights is None else entered - issue("weights_dv", weights(i).T, dv.astype(dtype))

    return [slice_entered(i, part) for i, part in enumerate(grad)], dv


def key_memory_decay(entered, leaving) -> jax.Array:
    """The gradient of a chunk's memory decay under a decay a key dimension before its `exp2(g_last)`, a
    key's row of each: `sum(h dh)` over the value columns of the memory the chunk entered `entered` `[BK,
    BV]` and the gradient of the one it leaves `leaving`, `[BK]` fp32, the log decay's term on the chunk's
    last token (`key_memory_block`'s other)."""
    if (generated := _generated_body("key_memory_decay", (entered, leaving))) is not None:
        return generated(entered, leaving)
    return jnp.sum(entered.astype(jnp.float32) * leaving.astype(jnp.float32), axis=1)


def key_values_back(value_grads: TileLoad, values: TileLoad, blocks: int, lower: Tile, betas, dtype,
                    issue: Issue, summed: Callable[..., jax.Array], *,
                    emit: Callable[[int, jax.Array], None], d_betas: jax.Array | None = None
                    ) -> tuple[jax.Array, jax.Array]:
    """The values' part of a decay a key dimension's backward (`templates.recurrence.key_backward`): through
    `u = (I + L) (v beta)`, each value block `j`'s `d(v beta) = (I + L)^T du` ('values_merge') of the
    corrected values' gradient `value_grads(j)`, its values' gradient `beta d(v beta)` `[C, BV]` fp32 to
    `emit(j, dv)`, and `(dA, dbeta)`, the inverse's gradient's part `du (v beta)^T` `[C, C]` and the write
    gates' `sum(v d(v beta))`, fp32. A walk of a row's value blocks a few at a time gives the write gates'
    sum its earlier steps left (`d_betas`) and a `summed` taking these blocks' products onto its system's."""
    d_betas = jnp.zeros(betas.shape, jnp.float32) if d_betas is None else d_betas
    for j in range(blocks):
        value = values(j).value().astype(jnp.float32)
        d_scaled = (value_grads(j).value().astype(jnp.float32)
                    + issue("values_merge", lower.T, value_grads(j)))
        d_betas = d_betas + jnp.sum(d_scaled * value, axis=1)
        emit(j, d_scaled * betas[:, None])
    # The inverse's part summed into one accumulator after (`summed`), each block's gated values formed as
    # its product stages them: an accumulator a block, summed beside each block's own gradient, an A100's
    # launch spilled 96 bytes a thread (kda-gate6 at 306006c01).
    d_system = summed([("values_merge", value_grads(j), lambda j=j: Transposed(
        write_gated(values(j).value().astype(jnp.float32), betas).astype(dtype))) for j in range(blocks)],
        shape=(lower.shape[0],) * 2)
    return d_system, d_betas


def key_memory_sums(out_grads: TileLoad, corrected: TileLoad, value_grads: TileLoad, entered: TileLoad,
                    leaving: TileLoad, blocks: int, summed: tuple[Callable[[list], jax.Array], ...]
                    ) -> tuple[Callable[[], jax.Array], ...]:
    """A key block's three memory products summed over value blocks `j` `[C, BK]` fp32
    (`key_memory_block`'s): `do h^T` ('memory_dq'), `v_new dh^T` ('memory_dk') and `dv h^T` ('memory_dw') of
    the output's gradient `out_grads(j)`, the corrected values `corrected(j)` and their gradient
    `value_grads(j)` against the memory entered `entered(j)` and its gradient leaving `leaving(j)` `[BK, BV]`,
    each a callable issuing its sum into one accumulator where it is read (its `summed`'s), so one sum is held
    at a time; a walk of a row's value blocks a few at a time gives `summed`s taking them onto the sums its
    earlier steps left."""
    def total(name, left, right, issued):
        return issued([(name, left(j), right(j).T) for j in range(blocks)])

    pairs = (("memory_dq", out_grads, entered), ("memory_dk", corrected, leaving),
             ("memory_dw", value_grads, entered))
    return tuple(functools.partial(total, *pair, issued) for pair, issued in zip(pairs, summed, strict=True))


def key_memory_block(query: Tile, key: Tile, gates: tuple, last: tuple, sums: tuple, lower: Tile, betas,
                     system, scale: float, dtype, issue: Issue, *, emit: Callable[[str, jax.Array], None],
                     columns: Callable[[jax.Array], jax.Array] | None = None,
                     read: Tile | None = None) -> tuple[jax.Array, jax.Array]:
    """One key block's part of a decay a key dimension's backward (`templates.recurrence.key_backward`), its
    queries `query` and keys `key` `[C, BK]`, their prefix from the chunk's start `gates` `[C, BK]` and the
    chunk's last token's `last` as a row's vectors (each its limbs, `Call.prefix_limbs`), and its memory
    products' sums over the value blocks (`key_memory_sums`' callables, each issued where it is read):
    `dq = scale exp2(g) do h^T`, `dk = exp2(g_last - g) v_new dh^T`, the weights' `dw = -dv h^T`
    ('memory_dq', 'memory_dk', 'memory_dw'); then `d(kg beta) = (I + L)^T dw` ('weights_merge'):
    `emit('dq', ...)`, `emit('dk', ...)` and the log decay's terms `emit('dg', ...)` `[C, BK]` fp32, the
    chunk's last prefix's `sum(k dk)` on its last token, and `(dA, dbeta)`: the inverse's gradient over the
    key blocks so far `system` `[C, C]` with this block's part `-dw (kg beta)^T` added as its product forms,
    so the part is held no longer, and the write gates' `sum(kg d(kg beta))`, fp32. The last prefix's other
    term, the memory's decay across the chunk (`key_memory_decay`), is the caller's: a key block's memory
    tiles' rows met the keys' columns here, which Mosaic GPU's layout inference ran out of fuel laying
    (kda-lower5-sm90 at 6e03eb57e). Each sum over the value blocks issued into one accumulator
    (`key_memory_sums`, `memory_gradients`'): a list of each block's products, their accumulators at once,
    spilled 744 bytes a thread on an H100 (kda-gate3 at ee3734584). `columns`, where given, sums a tile's
    columns (the mainloop's, `summed_pairs`'). Under an erase along reads of their own (`read`, the block's
    `[C, BK]`: `recurrence.Hooks.erase` 'read', the weights `X (b beta exp2 g)`), the weights' terms are the
    reads': `emit('dk', ...)` the memory's alone and `emit('dr', ...)` the reads' through the weights, the
    system's, the log decay's and the write gates' terms on the reads."""
    decays = raised(*gates)
    to_last = range_decay(tuple(limb[None, :] for limb in last), gates)
    query_sum, key_sum, weight_sum = sums
    # Each sum in turn, the log decay's terms gathered as each one's term is formed, so one sum and its
    # terms are live at a time: the three sums at once, beside the terms every one fed, spilled 240 bytes a
    # thread on an H100 (kda-gate5 at db546905d); the weights' first, 232 (kda-gate8 at b1e1fcb9b).
    d_query = query_sum() * decays * scale
    emit("dq", d_query)
    d_decay = query.value().astype(jnp.float32) * d_query
    keys = key.value().astype(jnp.float32)
    d_key = key_sum() * to_last
    # The chunk's last prefix's term on its last token through the gated keys' `exp2(g_last - g)` every
    # token's.
    keyed = keys * d_key
    at_last = jnp.sum(keyed, axis=0) if columns is None else columns(keyed)
    rows, _ = tile_indices(d_key.shape)
    d_decay = d_decay - keyed + jnp.where(rows == rows.shape[0] - 1, at_last[None, :], 0.0)
    d_weight = weight_sum()
    # `d(kg beta) = (I + L)^T dw` of `dw = -dv h^T`: cast `dw` once before both the diagonal and off-diagonal
    # terms (and `d_system`) so its narrow rounding cancels across `(I + L)^T` and between `dbeta_m` and
    # `dbeta_p` as fla's `dw = -dw.astype(dtype)` does (row 8716).
    d_weight_narrow = d_weight.astype(dtype)
    d_gated = -(d_weight_narrow.astype(jnp.float32) + issue("weights_merge", lower.T, d_weight_narrow))
    through = d_gated * decays * betas[:, None]
    if read is None:
        emit("dk", d_key + through)
    else:
        emit("dk", d_key)
        emit("dr", through)
    gated = (keys if read is None else read.value().astype(jnp.float32)) * decays
    gated_rows = Transposed(write_gated(gated, betas).astype(dtype))
    system = system - issue("weights_merge", d_weight_narrow, gated_rows)
    emit("dg", d_decay + gated * d_gated * betas[:, None])
    d_betas = jnp.sum(d_gated * gated, axis=1)
    return system, d_betas


def _wy_inverse_adjoint(d_system, lower, dtype, issue: Issue, *, right_label: str,
                        left_label: str) -> jax.Array:
    """Strictly lower `-(I + L)^T tril(d_system, -1) (I + L)^T`, casting each factor once before both its
    diagonal identity and off-diagonal `lower.T` product so bf16 rounding cancels (row 8716, base review
    b9f004f74 F5)."""
    strict = _lags(d_system.shape) > 0
    d_system = jnp.where(strict, d_system, 0.0).astype(dtype)
    right = (d_system.astype(jnp.float32) + issue(right_label, d_system, lower.T)).astype(dtype)
    return jnp.where(strict, -(right.astype(jnp.float32) + issue(left_label, lower.T, right)), 0.0)


def key_system_back(d_system, lower: Tile, dtype, issue: Issue) -> jax.Array:
    """The keys' system's gradient `[C, C]` fp32 of a chunk's from its inverse's `d_system` (`I + L`, `L`
    `lower`): `-(I + L)^T dA (I + L)^T`, strictly lower, each side's identity outside its product
    ('system_merge'), as fla_kda's wy_dqkg forms its `dA`."""
    return _wy_inverse_adjoint(d_system, lower, dtype, issue, right_label="system_merge",
                               left_label="system_merge")


def key_decayed_pairs_back(d_pairs: list, rows: list, keys: Callable[[int], jax.Array], columns: int,
                           gates: Callable[[int], tuple], reference: Callable[[int, int], tuple],
                           together: Callable[[list], tuple], staged: Callable[[jax.Array], Staged],
                           raised: Callable[[jax.Array], Tile], *, chunk: int,
                           emit: Callable[[str, int, int, jax.Array], None], sub: int = SUB) -> None:
    """The transpose of a chunk's factored pairs (`key_block_pairs`'), each at its sub-chunk's first token:
    for each pairs' gradient `d_pairs[p]` `[C, C]` (a tile, causal or strict as its pairs were, zero
    elsewhere, scaled as they were) of rows `rows[p](c)` against the same keys `keys(c)` fp32, each key
    block `c`'s gradients to `emit`: `('rows', p, c, d)` the rows' `sum_j dP_ij k_j exp2(g_i - g_j)` and
    `('keys', 0, c, d)` the keys', summed over the pairs, `sum_p sum_i dP_ij r_i exp2(g_i - g_j)`, each
    `[C, BK]` fp32. Each sub-chunk `s` past the first takes its first token's prefix as the common point, as
    the forward does: `dP` against every earlier key gated up to it (`gated_up`, the keys staged by
    `raised(value)`), kept on the rows of `s` and gated down by `exp2(g - g_ref)`; `dP^T` against the rows of
    `s` gated down to it, each key's then gated up to it ('pairs'): the pairs' gradients' rows of `s` against
    the gated rows' own, each pairs' rows gated down staged once a key block (`staged(value)`; `d_pairs`
    staged as they come), each a tile whose `rows(start, count)` a product takes, so the keys' products'
    depth is a sub-chunk's, not the chunk's of masked rows. A sub-chunk's products are issued together and
    awaited once (`together`, `Issuer.summed_together`'s groups, a reader each): each awaited where it was
    read, a sub-chunk's three waits stood on its chain as the forward's commits did (23% of a B200's intra
    walk, an emulated B200's line map at 37ef70629d7). The sub-chunks are a rolled loop, the rows' and keys'
    gradients carried across it: unrolled, every sub-chunk's tiles at once, an H100's launch spilled 1756
    bytes a thread (kda-gate3 at ee3734584).

    The factored products take the keys before each sub-chunk alone, as the forward takes them
    (`key_block_pairs`); each sub-chunk's own pairs' gradients are `key_diagonal_pairs_back`'s."""
    for c in range(columns):
        prefix = separated(*gates(c))
        index, _ = tile_indices(prefix[0].shape)
        down = decay(prefix, _laid_references(lambda s, c=c: separated(*reference(s, c)), index, chunk, sub))
        gated_rows = [staged(row(c) * down) for row in rows]

        def turn(s, carried, c=c, prefix=prefix, index=index, gated_rows=gated_rows):
            d_rows, d_keys = carried
            place = s * sub
            up = gated_up(prefix, separated(*reference(s, c)), index, place)
            # The keys read as the turn takes them, no tile of them live across the turns, and gated up
            # staged once, every pairs' product reading them.
            keyed = raised(keys(c) * up)
            *rowed, gated = together(
                [[("pairs", d_pair, keyed)] for d_pair in d_pairs]
                + [[("pairs", d_pair.rows(place, sub).T, down_rows.rows(place, sub))
                    for d_pair, down_rows in zip(d_pairs, gated_rows, strict=True)]])
            # Each read as it is added, one value live at a time.
            within = (index >= place) & (index < place + sub)
            d_rows = tuple(total + jnp.where(within, read(), 0.0)
                           for total, read in zip(d_rows, rowed, strict=True))
            return d_rows, d_keys + up * gated()

        zero = jnp.zeros(index.shape, jnp.float32)
        # The first sub-chunk has no keys before it.
        d_rows, d_keys = jax.lax.fori_loop(1, chunk // sub, turn, ((zero,) * len(d_pairs), zero))
        d_rows = tuple(total * down for total in d_rows)
        for p, total in enumerate(d_rows):
            emit("rows", p, c, total)
        emit("keys", 0, c, d_keys)


def key_diagonal_pairs_back(strips: list, own: list, behind: Callable[[jax.Array], tuple],
                            lower: Callable[[jax.Array, jax.Array], jax.Array], d_rows: tuple, d_keys,
                            *, sub: int = SUB, sheared: bool = False) -> tuple:
    """`key_diagonal_bands`' transpose on one key block, added onto its gradients so far `d_rows` (one a
    pairs' gradient) and `d_keys`: for each offset `e` the band `dP[i, i - e]` of each pairs' gradient
    `strips[p](e)` within `i`'s sub-chunk, a row's vector `[C]` (a row of the pairs' gradients' own bands
    laid an offset a row, `own_bands`, read once a launch: read from each chunk's `[C, C]` square at every
    offset and key block, an sm100's own pairs' backward took 0.844 ms of Kimi Linear's training,
    M-JS-119), and each pair's own decay, the product of its tokens' own formed across the offsets as the
    forward forms it (`behind(e)`: the keys and each token's own decay, each row holding the token `e`
    before its own's, as `key_diagonal_bands` reads them); the rows' `dP[i, i - e] k_{i - e}` decayed, and
    the keys' `sum_p dP[i, i - e] r_i` decayed (`own[p]()` the rows, read where weighed), formed on the
    rows `i` and moved up to the keys `i - e` (`lower(e, x)`: row `i` of `x` at row `i - e`, any value
    past the tile's end, masked). Every factor at most one (fla_kda's token-parallel diagonal's gradient,
    chunk_kda_bwd_intra's diagonal blocks). A row before its offset is zero on its strips (`own_bands`
    lays them so), its terms a zero strip's: its keys and decays read below its sub-chunk finite
    (`behind`'s, the staged rows below the chunk zeroed), where the gradients' every term held to its mask
    was a quarter of a B200's own pairs' backward's issues. A read erase's rows are its reads, declared at
    most one (`recurrence.Hooks.unit_reads`): undeclared, a read of `2^100` against a pairs' gradient of
    `-2^60` was `-2^160` before its decay, and the call takes the portable math. Its update `sheared` (its
    reads not its keys: no projection, which bounds the tied update's cotangents), the pairs' terms summed
    before their decay where their sum is finite, else each decayed first: cotangents about `2^127` summed
    to `2^128` where their decay takes them to `2^67` (head review of a84db53d5)."""
    index, _ = tile_indices(d_keys.shape)

    def offset(e, carried):
        d_rows, d_keys, pair, following = carried
        keys, decays = behind(e)
        pair = pair * following
        # A row before its offset reads below its sub-chunk, finite values (the staged rows below the chunk
        # zeroed), its strip zero there (`own_bands`) its terms'. The keys' terms are moved up from the rows
        # `e` below, each within its sub-chunk where selected.
        bands = [strip(e)[:, None] for strip in strips]
        # The rows read where they are weighed, not held across the offsets: held, an sm100's own pairs'
        # backward needed all of its 248 registers, read here 232 (row 5298).
        d_rows = tuple(total + strip * keys * pair for total, strip in zip(d_rows, bands, strict=True))
        # The pairs' terms summed before their decay (each one's decayed apart, two normal terms of
        # 3 * 2^-128 flushed where their sum decayed is normal); a sheared update's decayed first where
        # their sum overflows.
        terms = [strip * row() for strip, row in zip(bands, own, strict=True)]
        summed = functools.reduce(jnp.add, terms)
        weighted = summed * pair
        if sheared:
            weighted = jnp.where(jnp.isfinite(summed), weighted,
                                 functools.reduce(jnp.add, (term * pair for term in terms)))
        return (d_rows, d_keys + jnp.where(index % sub + e < sub, lower(e, weighted), 0.0), pair, decays)

    one = jnp.ones(index.shape, jnp.float32)
    d_rows, d_keys, _, _ = jax.lax.fori_loop(0, sub, offset, (d_rows, d_keys, one, one))
    return d_rows, d_keys


def scores_gradient(queries: TileLoad, keys: TileLoad, columns: int, values: TileLoad, out_grads: TileLoad,
                    blocks: int, gate, low, scale: float, dtype, summed: Callable[[list], jax.Array], *,
                    laid: tuple | None = None, column_scale=None,
                    apart: int = 0) -> tuple[jax.Array, jax.Array]:
    """A chunk and value head's scores' gradient under a decay a head, `ds = do v^T` over the value blocks
    `j` gated by its pairs' decays, causal and scaled, in `dtype` (which `memory_gradients`' `dq += ds k`
    and `dk += ds^T q` take), and the decay's pair terms `ds (q k^T)` (strictly lower, fp32,
    `summed_pairs`'): the part of fla's dqkwg the memory's products do not take, a launch of its own. Each
    of the two sums over blocks issued into one accumulator (`summed`, the mainloop's: a list of `(name,
    left, right)`): an accumulator a block, held two at a time, an fp32 call's two parts spilled 32 bytes a
    thread past 255 on an H100, where one each needed 216 (scores-ablate at a44dfe721). The scores' sum
    first, ds's after it and then the pairs' decays, so no part of ds is live through the scores' blocks:
    ds and its decays through them, an A100's launch at keys and values 256 wide spilled 192 bytes a thread
    where first it needs 224 registers, an H100's at Qwen3.5-27B's widths the same 192 either way
    (scores-order and scores-first at e43d62899). `column_scale` `[C]`, where given, scales each value
    token's column in fp32 after the products: the write gate on the keys' side (`chunk_scores`'), mamba_ssm's
    `dCB = (dout x^T) dt` (`_chunk_scan_bwd_dcb`). A readout `apart` from its token's own writes keeps the
    pairs of later tokens alone, as `chunk_scores`' (the pairs' terms of a token's own are zero either way:
    within a token no decay)."""
    scores = summed([Blocks("local_t", queries, lambda c: keys(c).T, columns)])
    ds = summed([Blocks("values_ds", out_grads, lambda j: values(j).T, blocks)])
    if column_scale is not None:
        ds = ds * column_scale[None, :]
    column_gate, column_low = (gate, low) if laid is None else laid
    lag = _lags(ds.shape)
    kept = _later_groups(ds.shape, apart) if apart else lag >= 0
    ds = jnp.where(kept, ds * decayed(gate, low, column_gate, column_low), 0.0) * scale
    return ds.astype(dtype), jnp.where(lag > 0, ds * scores, 0.0)


def memory_gradients(queries: TileLoad, keys: TileLoad, columns: int, values: TileLoad, out_grads: TileLoad,
                     value_grads: TileLoad | None, entered: TileLoad, leaving: TileLoad, blocks: int, ds,
                     gate, low, last, last_low, memory_decay, scale: float,
                     summed: Callable[[list], jax.Array], emit: Callable[[str, int, jax.Array], None],
                     row_scale=None, summed_all: Callable[[list], tuple] | None = None
                     ) -> tuple[jax.Array, jax.Array]:
    """The queries', keys' and weights' gradients of a chunk and value head (fla's dqkwg's memory products)
    under a decay a head, given its scores' gradient `ds` (`scores_gradient`'s, a tile its products read as
    they are, in the operands' dtype): over the value blocks `j`, `dq = do S^T`, `dk = v dS^T` and
    `dw = -dv S^T` for the corrected values `values(j)`, the output's gradient `out_grads(j)`, the corrected
    values' own `value_grads(j)` and the memories the chunk entered and left `entered(c, j)`, `leaving(c, j)`
    (key block `c`), then the gates and `dq += ds k`, `dk += ds^T q` (none where `ds` is None: a caller
    summing the scores' gradient over value heads first takes them once), each key block's to `emit('dq' |
    'dk' | 'dw', c, grad)` as it is final; and the decay's range terms `(after, leaving)` (`summed_ranges`'),
    `leaving` `[C]` each token's write into the memory the chunk leaves. `memory_decay` is the decay's
    gradient through the memory the chunk leaves, `sum(S dS)` (a row's vector of it; None where a caller
    summing key blocks apart adds it once). Without the delta rule (`value_grads` None) there are no weights
    and no `dw`. `row_scale` `[C]`, where given, scales each token's row of `dk`'s memory products in fp32:
    the write gate on the keys' side (`keys_gated_carried`), mamba_ssm's `dB = (x dstates^T) dt`. Each sum
    of products over the value blocks issued into one accumulator (`summed`, the mainloop's: a list of
    `(name, left, right)`), as `scores_gradient`'s are; a key block's sums, none reading another's, issued
    together where the mainloop gives `summed_all` (a list of such lists, each its own accumulator, one wait
    for all, each read where it is used: a reader a sum), each issued where it is read where not."""
    row_decay = raised(gate, low)[:, None]
    gathered = summed_all or (lambda groups: tuple(functools.partial(summed, group) for group in groups))
    leaving_decay = range_decay((last, last_low), (gate, low))[:, None]
    leaving_terms, after = [], []
    for c in range(columns):
        groups = [[("memory_dq", out_grads(j), entered(c, j).T) for j in range(blocks)],
                  [("memory_dk", values(j), leaving(c, j).T) for j in range(blocks)]]
        if value_grads is not None:
            groups.append([("memory_dw", value_grads(j), entered(c, j).T) for j in range(blocks)])
        if ds is not None:
            groups += [[("scores_dq", ds, keys(c))], [("scores_dq", Transposed(ds), queries(c))]]
        read_dq, read_dk, *reads = gathered(groups)
        read_dw = reads.pop(0) if value_grads is not None else None
        dq = read_dq() * row_decay * scale
        after.append(jnp.sum(dq * queries(c).value().astype(jnp.float32), axis=1))
        emit("dq", c, dq if ds is None else dq + reads[0]())
        dk = read_dk()
        dk = dk * leaving_decay if row_scale is None else dk * (leaving_decay * row_scale[:, None])
        leaving_terms.append(jnp.sum(dk * keys(c).value().astype(jnp.float32), axis=1))
        emit("dk", c, dk if ds is None else dk + reads[1]())
        if read_dw is not None:
            emit("dw", c, -read_dw())
    after = _summed(after)
    if memory_decay is not None:
        lag = _lags((after.shape[0],) * 2)
        after = after + jnp.where(lag[:, 0] == after.shape[0] - 1, memory_decay * raised(last, last_low), 0.0)
    return after, _summed(leaving_terms)


def system_gradient(lower, keys: TileLoad, columns: int, values: TileLoad, blocks: int, w_grads: TileLoad,
                    u_grads: TileLoad, beta, gate, low, dtype, issue: Issue,
                    summed: Callable[..., jax.Array], *, laid: tuple | None = None) -> jax.Array:
    """The WY representation's backward (fla's prepare_wy_repr_bwd) under a decay a head, its system's part:
    with the inverse `X = I + lower` (`lower` its strictly lower part, a tile), `w = X (k beta exp2(g))` and
    `u = X (v beta)`, from `w`'s and `u`'s gradients `w_grads(c)` and `u_grads(j)`: `dX = dw (k beta
    exp2 g)^T + du (v beta)^T` ('inverse_dx', 'inverse_du': a key block's and a value block's, each staged
    at its own width) and `dL = -X^T tril(dX, -1) X^T` gated by `exp2(g_s - g_t)` ('inverse_t',
    'inverse_l'), in `dtype`, which `keys_gradient` takes. `X^T dw` and `X^T du` and what
    they give are `keys_gradient`'s and `values_gradient`'s. `dX`'s key and value blocks issued into one
    accumulator (`summed`, the mainloop's): an accumulator a block, an H100's launch at keys and values 256
    wide spilled 264 bytes a thread (wide-sm90 at 8d1145e66)."""
    decay = raised(gate, low)

    def scaled(load, scale):
        return lambda: Transposed((load().value().astype(jnp.float32) * scale[:, None]).astype(dtype))

    keyed = [("inverse_dx", w_grads(c), scaled(functools.partial(keys, c), beta * decay))
             for c in range(columns)]
    valued = [("inverse_du", u_grads(j), scaled(functools.partial(values, j), beta)) for j in range(blocks)]
    d_x = summed(keyed + valued, shape=(lower.shape[0],) * 2)
    strict = _lags(d_x.shape) > 0
    d_l = _wy_inverse_adjoint(d_x, lower, dtype, issue, right_label="inverse_l", left_label="inverse_t")
    column_gate, column_low = (gate, low) if laid is None else laid
    return jnp.where(strict, d_l * decayed(gate, low, column_gate, column_low), 0.0).astype(dtype)


def _generated_body(name: str, operands: tuple):
    """The algebra's generated and checked body of this module's component `name` for a call over `operands`
    where the run asks for it (`dew.algebra.generated.body`); None, the written math to run, otherwise."""
    from dew.algebra import generated

    return generated.body(f"dew.algebra.component_cases.{name}", operands)


def selective_decay(step, rate, kept=None):
    """Mamba's selective scan's decay of a state by a token, `exp(dt A)` of its `rate` `A`, as `exp2((dt A)
    log2(e))`, zero at a document's first token (`kept` 0, where given): the one decay form of the scan's
    every path (the template's forward and backward, the portable scan), the token and adjoint functions'
    `a` (`selective_token`, `selective_adjoint`), which the scan's schedules take a token or a block of
    tokens at a time (`templates.recurrence.selective`). The rate is taken to base 2 after the step
    multiplies it, so a rate past float32's range times a padded token's zero step is zero, not `0 * -inf`
    (row 8491: A = -3 2^126 made `A log2(e)` -inf and a padded token's decay NaN), and a step past it times
    a small rate its decay: the step taken to base 2 first, `dt = 3 2^126` overflowed, `A = -2^-126`
    decaying to 0 for `exp(-3)` and `A = 0` to NaN for 1 (mamba1b's review)."""
    operands = (step, rate) if kept is None else (step, rate, kept)
    if (generated := _generated_body("selective_decay", operands)) is not None:
        return generated(*operands)
    decay = jnp.exp2((step * rate) * LOG2_E)
    return decay if kept is None else decay * kept


def selective_token(states: list, decays: list, written, b: list) -> list:
    """Each state's update by a token of the selective scan, `h_n = a_n h_n + (dt x) B_n`: `states` and
    `decays` (`selective_decay`'s) a list a state, `written` the token's `dt x` and `b` its `B_n`."""
    return [decay * state + written * b_n for state, decay, b_n in zip(states, decays, b, strict=True)]


def pairwise_sum(terms: Sequence, *, combine=operator.add):
    """Sum nonempty `terms` as an adjacent-pair balanced tree with odd-tail carry, shared by
    `selective_read` and decode (`step.py`, base review da734bf54 F6, row 9080)."""
    parts = list(terms)
    while len(parts) > 1:
        paired = [combine(x, y) for x, y in zip(parts[::2], parts[1::2], strict=False)]
        parts = [*paired, *parts[len(parts) // 2 * 2:]]
    return parts[0]


def selective_read(states: list, c: list, skip, value):
    """A token's output of the selective scan, `y = sum_n C_n h_n + D x`, the states' terms summed as a
    balanced tree, its depth log2 of their count: summed in turn, the readout's 16 states put an fp32 scan's
    output 1.68 times the portable scan's distance from float64 where every decay is near zero, its own
    write the whole state (the interpreted regime test)."""
    if (generated := _generated_body("selective_read", (*states, *c, skip, value))) is not None:
        return generated(*states, *c, skip, value)
    return pairwise_sum([c_n * state for c_n, state in zip(c, states, strict=True)]) + skip * value


_SELECTIVE_SUBNORMAL_STEP = 2.0 ** -100
_SELECTIVE_DX_SCALE = 2.0 ** 64


def selective_dx_scale(skip, dy, step):
    """Derive once per block `(dx_init, dx_step, inv_scale)` for a selective scan backward's `dx` state sum:
    on subnormal-step lanes (`|step| < 2^-100`) whose `skip * dy` fits below `2^60`, lift both `skip * dy`
    and `step` by `2^64` via `float_bits.raised_exactly` (bit-safe on flush-to-zero backends, row 9089) and
    return `inv_scale = 2^-64`; otherwise return `(skip * dy, step, 1.0)` (row 8722, base reviews da734bf54
    F3, b65e3cab4 F8, b0601005b F6)."""
    if (generated := _generated_body("selective_dx_scale", (skip, dy, step))) is not None:
        return generated(skip, dy, step)
    base = skip * dy
    scaled = ((jnp.abs(step) < jnp.float32(_SELECTIVE_SUBNORMAL_STEP))
              & (jnp.abs(base) < jnp.float32(2.0 ** 60)))
    shift = jnp.where(scaled, jnp.int32(64), jnp.int32(0))
    return (raised_exactly(base, shift),
            raised_exactly(step, shift),
            jnp.where(scaled, jnp.float32(1.0 / _SELECTIVE_DX_SCALE), jnp.float32(1.0)))


def selective_adjoint(cotangent, before, decay, written, step, value, b, dy, rate, *, dx_step=None) -> tuple:
    """One state's gradients of the selective scan through a token, from the cotangent the token's state
    takes `cotangent` (`g_t = C_t dy_t + a_{t+1} g_{t+1}`, its read and what the tokens after pass back),
    the state before it `before` (`h_{t-1}`), its `decay` (`selective_decay`'s), `written` (`dt x B`), and the
    token's `step`, `value`, `b`, `dy` and the state's `rate` `A`: `(dx, ddt, dB, dC, drate)`, the rate's
    a token's term its schedule sums. `exp(dt A)`'s derivative in `dt` is `A` times it, in `A` `dt` times
    it. Where `dx_step` is given (`selective_dx_scale`), it scales only `dx`'s `step` factor so per-state
    terms and `skip * dy` stay normal across the state sum (row 8722, base review b0601005b F6)."""
    lifted = () if dx_step is None else (dx_step,)
    operands = (cotangent, before, decay, written, step, value, b, dy, rate, *lifted)
    name = "selective_lifted_adjoint" if lifted else "selective_adjoint"
    if (generated := _generated_body(name, operands)) is not None:
        return generated(*operands)
    through = cotangent * before * decay
    d_written = cotangent * b
    return (d_written * (step if dx_step is None else dx_step), d_written * value + through * rate,
            cotangent * step * value, dy * (decay * before + written), through * step)


def gated_values_back(u_grad, values, beta) -> tuple[jax.Array, jax.Array]:
    """The write gate's gradient on values gated, `u = v beta`, from `u_grad` `[..., V]`, fp32: the values'
    own, `du beta`, and beta's `[...]`, `sum(v du)` over the last axis's columns."""
    if (generated := _generated_body("gdn_value_pullback", (u_grad, values, beta))) is not None:
        return generated(u_grad, values, beta)
    return u_grad * beta[..., None], jnp.sum(u_grad * values, axis=-1)


def skipped_back(out_grad, values, skip) -> tuple[jax.Array, jax.Array]:
    """The skip `y += D v`'s gradient on values from the output's gradient `out_grad` `[..., V]`, fp32: the
    values' own, `D do` (`skip` broadcasting), and the skip's own rows `[...]`, `sum(v do)` over the last
    axis's columns."""
    if (generated := _generated_body("skipped_back", (out_grad, values, skip))) is not None:
        return generated(out_grad, values, skip)
    return skip * out_grad, jnp.sum(out_grad * values, axis=-1)


def values_gradient(lower, values: TileLoad, blocks: int, u_grads: TileLoad, beta, dtype, issue: Issue, *,
                    emit: Callable[[str, int, jax.Array], None]) -> jax.Array:
    """The WY representation's backward under a decay a head, its values' part: each value block's gradient
    `(X^T du) beta` ('inverse_t') to `emit('values', j, dv)` as it is computed. Returns beta's gradient
    through `u`, fp32."""
    d_betas = []
    for j in range(blocks):
        u_grad = u_grads(j).value().astype(jnp.float32)
        d_scaled = u_grad + issue("inverse_t", lower.T, u_grad.astype(dtype))
        d_values, d_beta = gated_values_back(d_scaled, values(j).value().astype(jnp.float32), beta)
        emit("values", j, d_values)
        d_betas.append(d_beta)
    return _summed(d_betas)


def keys_gradient(lower, keys: TileLoad, columns: int, w_grads: TileLoad, d_l, beta, gate, low, dtype,
                  issue: Issue, *, emit: Callable[[str, int, jax.Array], None],
                  reads: TileLoad | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The WY representation's backward under a decay a head, its keys' part, given the system's gradient
    `d_l` (`system_gradient`'s, a tile its products read as they are, in `dtype`): each key block's gradient,
    through `w` (`X^T dw` scaled, 'inverse_t') and through the system (`dL k` and `dL^T (k beta)`,
    'keys_dl', 'keys_dlt'), to `emit('keys', c, dk)` as it is final. Returns beta's gradient through `w` and
    the system, the decay's range terms through `w` (`summed_ranges`' `after`) and its pair terms
    `dL (k k^T) beta` (`summed_pairs`'), fp32. Under an erase along reads of their own (`reads(c)`, a key
    block of them: `recurrence.Hooks.erase` 'read', the system `beta b k^T` and `w = X (b beta exp2 g)`),
    the system's rows and `w` are the reads': their gradient through `w` and `dL k` to `emit('reads', c,
    db)`, the keys' through the system's columns `dL^T (b beta)` to `emit('keys', c, dk)`, beta's and the
    decay's terms through the reads, the pair terms `dL (b k^T) beta`."""
    decay = raised(gate, low)
    narrow = d_l if isinstance(d_l, jax.Array) else d_l.value()
    weighted_l = (narrow.astype(jnp.float32) * beta[:, None]).astype(dtype)
    rows = keys if reads is None else reads
    d_betas, d_gate = [], None
    for c in range(columns):
        key, read = keys(c), rows(c)
        w_grad = w_grads(c).value().astype(jnp.float32)
        # `X^T dw` decayed, then its sum with `dL k` (each term's own beta its row's): one tile of them live.
        through_w = (w_grad + issue("inverse_t", lower.T, w_grad.astype(dtype))) * decay[:, None]
        read_value = read.value().astype(jnp.float32)
        d_gate = _added(d_gate, beta * jnp.sum(through_w * read_value, axis=1))
        combined = through_w + issue("keys_dl", d_l, key)
        d_betas.append(jnp.sum(combined * read_value, axis=1))
        # `dL^T (b beta)` as `(beta dL)^T b`: one operand in registers, the reads a tile.
        columned = issue("keys_dlt", Transposed(weighted_l), read)
        if reads is None:
            emit("keys", c, combined * beta[:, None] + columned)
        else:
            emit("reads", c, combined * beta[:, None])
            emit("keys", c, columned)
    # The system's own pairs once every block's gradient is out: a tile not live beside theirs.
    products = None
    for c in range(columns):
        products = _added(products, issue("keys_dl", rows(c), keys(c).T))
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


def summed_ranges(after, leaving):
    """A chunk's per-token decays' gradient from the terms through ranges of its prefix, its ranges' part
    (beside `summed_pairs`; linear in its terms, so a sum of launches' terms sums the same as their sums),
    over the last axis, a chunk's tokens, from each token's terms summed across launches: `after` `[C]` each
    `s` a term through the chunk's start to `s`, `leaving` `[C]` each `t` a term through `(t, C - 1]`, a pair
    of the chunk's last row: `sum_{s >= t} after + sum_{j < t} leaving` each token's, `after` from it to the
    chunk's end and `leaving` from the chunk's start to before it, both cumulative sums. `leaving`'s sum
    before each token takes no token's own term added and taken away: a token's own `leaving` is order one
    where the terms before it are `exp(-30)`'s under steep decays, which the sum rounded away (an H100's decay
    gradient under decays near zero 140 times the port's distance from float64, whose `leaving` enters its
    strictly lower pairs, fla `chunk_bwd_dqkwg`; SE-numerics-h100-fd61e11eeb)."""
    before = jnp.concatenate((jnp.zeros_like(leaving[..., :1]), jnp.cumsum(leaving[..., :-1], -1)), -1)
    return jnp.flip(jnp.cumsum(jnp.flip(after, -1), -1), -1) + before


def decay_gradient(pairs, after, leaving, memory, ending):
    """A chunk's per-token decays' gradient `[..., C]` from a backward's terms: its pairs' part, summed per
    token already (`summed_pairs`'), its ranges' (`summed_ranges` of `after` and `leaving`), and the memory's
    decay across the chunk, `memory` `[...]` its gradient `sum(S dS)`, through `exp2(ending)` (the chunk's
    last base-2 prefix) from the chunk's start, every token's."""
    return pairs + summed_ranges(after, leaving) + (memory * jnp.exp2(ending))[..., None]


def summed_pairs(pairs, issue: Issue = portable, columns: Callable[[jax.Array], jax.Array] | None = None):
    """A chunk's per-token decays' gradient from the terms through ranges of its prefix, its pairs' part
    (beside `summed_ranges`): `pairs` `[C, C]` fp32, each `(i, j)` a term through `(j, i]` (strictly lower),
    each token summing only those whose range holds it, `sum_{j < t <= i} pairs`: each row's pairs before
    `t` a product with the strictly upper ones ('summed', an `Exact` triangle: no pair added and taken back),
    and each token's sum its column's (`columns`, a mainloop's own reduction across its rows' warps, where
    given)."""
    lag = _lags(pairs.shape)
    terms = jnp.where(lag >= 0, issue("summed", pairs, Exact(jnp.where(lag < 0, 1.0, 0.0))), 0.0)
    return jnp.sum(terms, axis=0) if columns is None else columns(terms)
