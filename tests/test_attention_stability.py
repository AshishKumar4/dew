"""The attention template's stability suite (bible §6): on inputs at fp32's edges, every attention form on
every mainloop (the Triton one interpreted, the plain-JAX one, the decode kernel) gives a finite output
and finite gradients wherever the float64 truth rounded to their dtype is, and never ties values the
reference keeps apart. Each
family fixes an adversarial structure (logits a ulp apart, near fp32's largest, a weight of exp(-87) on a
value of 2^126, rows that see nothing) and draws its values by seed, against the reference written apart
from Dew (`attention_oracle.exact_attention`), its float64 truth's derivative anchored
(`attention_oracle.anchored_softmax`); a row that sees no key is an exact claim, its output and
gradients zero. The accurate form (natural logits, `template.NATURAL`'s other form, and the backward's
accurate form, `template.ACCURATE`'s) holds check 1 (`reference_error.assert_as_exact`) on every family,
as exact as the reference. The default, which folds log2(e) into the scale where the scale allows and
takes FlashAttention's arithmetic for the backward's products and dQ (about the template's own anchored
D, not FlashAttention's rowsum(dO * O)), is finite wherever the rounded truth is and holds the exact-zero
claims (`assert_finite`, `_held_empty`; the owner's ruling, 2026-10-23) on every family but the fold's
documented divergences (`algebra.Forward.base2`; docs/design/kernel-templates.md, §3, the exponent's
forms), which the accurate form alone runs. Its error at these inputs is no claim, and it ties values the
reference keeps apart, its documented divergence (docs/design/testing.md, a template's stability suite):
groups of a leaf's entries it returns equal whose reference values spread past twice the reference's own
error and two ulps more, over 16 seeds of the plain-JAX mainloop, are the output's, by the fold, at
logits near 1e4 (1 group, 374 ulps), keys an ulp apart in bf16 (3, 8 ulps) and a sink and one key (1, 5
ulps); dQ's, by FlashAttention's one bf16 product at the first D, at keys an ulp apart in bf16 (278
groups, 379 ulps); dK's at a small weight on a large value in bf16 (12, 8 ulps); and dV's, by the fold's
recomputed weights, at keys an ulp apart in fp32 (100, 16 ulps)."""

import contextlib
import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from arm_arithmetic import Truth
from attention_oracle import anchored_softmax, exact_attention
from generation_support import here, interprets
from reference_error import (
    FEWEST_SEEDS,
    as_given,
    assert_as_exact,
    assert_merges_no_ties,
    widened,
    with_gradients,
)

from dew.nn.documents import CrossDocuments
from dew.nn.kernels import decode_attention, generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_decode,
    attention_mosaic,
    attention_portable,
    attention_route,
    attention_triton,
    mosaic_sm90,
)
from dew.nn.kernels.templates.hooks import Hook

PINS = {attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2),
        attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2)}
"""The Triton mainloop's tiles: a family is a block or less, so one tile walks it, but the equal-weight ones
over 64 and 256 keys, which walk two and eight."""

INTERPRETED = interprets()
"""Whether the Pallas-Triton mainloop runs interpreted, in a CPU process, or compiled, on a GPU: the A100's
default for every template call and the H100's for fp16 and the hooks Mosaic refuses, so the GPU tier
proves on the compiled kernel what the interpreter covers."""


@dataclasses.dataclass(frozen=True)
class Family:
    """A stability family: `draw(seed)` gives the queries `[B, Q, H, D]`, keys and values `[B, K, H, D]`,
    the sinks `[H]`, a resumed state `(m [B, H, Q], total, acc [B, H, Q, D])` or None, and the output
    cotangent, fp32, run in `dtype`, the softmax's scale
    1/sqrt(D); the keys each query keeps `[B, 1, Q, K]` and the same keys as the template's spans'
    arguments (`template.spans`); the rows `[B, Q]` that see no key, whose output and gradients are
    zero (`empty`); whether the default form holds it (`folded`), False only for the fold's documented
    divergences (`algebra.Forward.base2`), which run natural alone; whether its one divergence is its
    weighted values' sums passing fp32's largest (`overflows`), which a walk scaling its rows' sums
    (`algebra.SCALED_WALK`) keeps finite in either form; the slices the decode kernel cuts each row's keys
    into (`splits`), whose states it merges; whether it is a decode step's alone
    (`decode_only`), a shape the template's interpreted mainloops would take too long over, which its
    template twin covers in fewer programs; and whether its queries' products with the keys pass fp32's
    largest before the scale (`query_first`), where jax's attention, which scales its products, overflows,
    so its reference takes the query scaled first, as Dew's reference path and flax's do (`_reference`);
    and, where its values are a dtype's extremes, its draw in any dtype at that dtype's own (`drawn_in`,
    `retyped`). Its declared seeds are `SEEDS`'."""

    draw: Callable[[int], tuple]
    keep: np.ndarray
    spans: dict = dataclasses.field(default_factory=dict)
    empty: np.ndarray | None = None
    folded: bool = True
    overflows: bool = False
    dtype: str = "float32"
    splits: int = 1
    decode_only: bool = False
    query_first: bool = False
    drawn_in: Callable[[str], Callable[[int], tuple]] | None = None

    def retyped(self, dtype: str) -> "Family":
        """The family run in `dtype`: its draw rounded to it, or where its values are a dtype's extremes
        (`drawn_in`), drawn at `dtype`'s own, so the family means the same in each (fp32's largest is past
        fp16's, whose cast would make the family's large values infinite)."""
        draw = self.draw if self.drawn_in is None else self.drawn_in(dtype)
        return dataclasses.replace(self, draw=draw, dtype=dtype)


@functools.partial(jax.jit, static_argnums=1)
def _normal_drawn(seed, shapes: tuple) -> list:
    streams = jax.random.split(jax.random.key(seed), len(shapes))
    return [jax.random.normal(stream, shape) for stream, shape in zip(streams, shapes, strict=True)]


def _normal(seed: int, *shapes) -> list[np.ndarray]:
    """Standard-normal fp32 arrays of `shapes`, one stream each: one compiled program a set of shapes,
    where op by op the draws took a third of a family's test (ledger 3902)."""
    return [np.array(drawn, np.float32) for drawn in _normal_drawn(seed, tuple(map(tuple, shapes)))]


@functools.partial(jax.jit, static_argnums=3)
def _uniform_drawn(seed, low, high, shape: tuple):
    return jax.random.uniform(jax.random.key(seed), shape, jnp.float32, low, high)


def _uniform(seed: int, low: float, high: float, shape=()) -> np.ndarray:
    """fp32 draws uniform in `[low, high)`, compiled as `_normal`'s."""
    return np.asarray(_uniform_drawn(seed, low, high, (shape,) if isinstance(shape, int) else tuple(shape)))


ROWS = 32
"""The independent instances a draw holds along its batch, so a family's draw resolves check 1 over its
seeds (`reference_error.seeds_for`): each row its own drawn values under the family's structure."""
RESUMED = 128
"""The resumed family's rows, more, as each holds one query over two keys."""
SINK_ROWS = 2
"""The sink-and-one-key families' rows, fewer, their draws holding their sinks' gradients across heads: the
interpreted Triton mainloop's time a seed grows faster than its programs, 21.6 s at 32 rows of 16 heads,
0.92 at 4 and 0.18 at 2 (a 26-core CPU), and `seeds_for` sizes the families at 2 rows at 64, 32 and 8
seeds."""
SEEDS = {"ties at scale 1": 64, "ties at scale 1, cotangents near 2**125": 64, "ties at scale 1/2": 64,
         "ties at scale 1/2, cotangents near 2**125": 64, "ties at scale 1/4": 64,
         "ties at scale 1/4, cotangents near 2**125": 64, "a logit near fp32's largest": 8,
         "a logit near fp32's largest at scale 1/4": 8, "values near fp32's largest over 2 equal keys": 8,
         "values near fp32's largest over 64 equal keys": 8,
         "values near fp32's largest over 256 equal keys": 8,
         "values near fp32's largest over 2 equal keys, bfloat16": 8,
         "values at fp32's largest over 2 equal keys": 8, "values at fp32's largest over 256 equal keys": 8,
         "one key": 8, "a sink and one key": 64,
         "a sink near fp32's largest and one key": 8, "a sink and one key, its logit near fp32's largest": 8,
         "a sink near fp32's largest at scale 1/4": 8, "logits near 1e4": 32, "logits near 1e30": 8,
         "a small weight on a large value, float32": 128, "a small weight on a large value, bfloat16": 256,
         "a weight at the exponent's edge on a large value, float32": 64,
         "a weight at the exponent's edge on a large value, bfloat16": 64,
         "a decode step's weight at the exponent's edge on a large value, float32": 64,
         "a decode step's weight at the exponent's edge on a large value, bfloat16": 64,
         "a small column beside an overflowing one": 64,
         "a column near fp32's least normal beside an overflowing one": 64,
         "a cotangent on a small column beside an overflowing one": 64,
         "a tiny value beside an overflowing column at equal weights": 8,
         "an anchor-relative share beside an overflowing column": 8,
         "a column near fp32's least normal under a large cotangent": 8,
         "a subnormal share of an overflowing row, float32": 32,
         "a subnormal share of an overflowing row, bfloat16": 32,
         "a subnormal share beside 255 equal weights, bfloat16": 32, "a row of no keys": 8,
         "causal queries before every key": 8, "a row that overflows beside one that does not": 8,
         "two finite slices whose merge overflows": 8,
         "values at fp32's largest across a split decode's slices": 8,
         "a resumed state of a large denominator": 64,
         "a resumed state beside an overflowing column": 32,
         "a resumed state of a denominator near fp32's largest beside an overflowing column": 8,
         "fp16 values near their largest beside a state near fp32's largest": 32,
         "keys an ulp apart, float32": 32, "keys an ulp apart, bfloat16": 32,
         "keys near fp32's largest at equal logits": 32, "a far maximum in a later key tile": 8,
         "a decode step's sink and one key": 32,
         "a product past fp32's largest before a small scale, float32": 8,
         "a product past fp32's largest before a small scale, bfloat16": 8,
         "a decode step's sink near fp32's largest and one key": 8,
         "documents of one key": 8,
         "a share in bf16's subnormal band on a value near its largest": 32,
         "a share in bf16's subnormal band, the denominator's rest in a later tile": 32,
         "small values beside unseen ones near fp32's largest, float32": 16,
         "small values beside unseen ones near fp32's largest, bfloat16": 16,
         "a row of small values beside one that overflows in its tile, float32": 16,
         "a row of small values beside one that overflows in its tile, bfloat16": 16}
"""Each family's declared seeds, the fewest over which the reference compared with itself resolves check
1's parity (`reference_error.seeds_for`), sized for every family at the suite's own draws and reference."""


def _first(sizes: np.ndarray, width: int) -> np.ndarray:
    """`[B, n, 1, width]` vectors of `sizes` `[B, n]` in their first entry and zeros past it, so a product of
    two of them is its sizes' product exactly at every width."""
    sizes = np.asarray(sizes, np.float32)
    out = np.zeros((*sizes.shape, 1, width), np.float32)
    out[..., 0, 0] = sizes
    return out


def _causal(queries: int, keys: int, rows: int = 1) -> np.ndarray:
    return np.broadcast_to(np.tril(np.ones((queries, keys), bool)), (rows, 1, queries, keys))


def _ties(width: int, cotangent: float) -> Callable[[int], tuple]:
    """`ROWS` rows of one query of 1 over two keys whose logits are a drawn fp32 near 2^30 and the one a
    drawn 1 to 4 ulps below it (Sol's case: the reference weighs the first alone, which a base-2 fold
    rounding each logit at log2(e) times its size can tie), their keys scaled by sqrt(`width`) so the
    logits are those at every scale; values and cotangent in their first entry, drawn in [-1, 1), the
    cotangent times `cotangent`, so their products stay finite in the reference's backward."""
    def draw(seed: int):
        top = _uniform(seed, 2.0 ** 30, 2.0 ** 31, (ROWS,))
        below = top.copy()
        for row, ulps in enumerate(np.asarray(_uniform(seed + 1, 1, 5, (ROWS,)), np.int32)):
            for _ in range(ulps):
                below[row] = np.nextafter(below[row], np.float32(0))
        value = _first(_uniform(seed + 2, -1.0, 1.0, (ROWS, 2)), width)
        dout = _first(_uniform(seed + 3, -1.0, 1.0, (ROWS, 1)) * np.float32(cotangent), width)
        root = np.float32(np.sqrt(width))
        keys = _first(np.stack([top, below], axis=1) * root, width)
        return _first(np.ones((ROWS, 1), np.float32), width), keys, value, None, dout

    return draw


def _small_weight(dtype, *, queries: int, logits: tuple[float, float], sizes: tuple[float, float] | None,
                  rows: int = 128, cases: int = 1, decode_only: bool = False) -> Family:
    # `queries` queries a row over 32 keys of which two are seen: a key of logit 0 and one whose logit is a
    # quarter of each query's size (drawn in `sizes`, else 4) times its key's -l, l drawn in `logits`, on a
    # value of 2^e, e drawn in [120, 126]: its share of the output, which a weight scaled for the sums'
    # headroom before it is narrowed would lose. At logits of -80 to -87 the weight is a normal fp32 and
    # bf16 at their exponent's edge, which a weight scaled before it is narrowed would flush
    # (`algebra.at_scale`'s small weights). One key tile a row, so the family's many rows stay cheap to
    # interpret; `cases` independent cases a row, each its own document of `queries` queries over its own
    # two keys (`CrossDocuments`), so a draw holds rows times cases of them in few programs.
    count = queries * cases

    def draw(seed):
        size = (np.full((rows, count), 4.0, np.float32) if sizes is None
                else _uniform(seed + 4, *sizes, (rows, count)))
        key = np.zeros((rows, 32, 1, 16), np.float32)
        key[:, 1:2 * cases:2, 0, 0] = -_uniform(seed, *logits, (rows, cases))
        value = np.zeros((rows, 32, 1, 16), np.float32)
        value[:, 1:2 * cases:2, 0, 0] = 2.0 ** np.floor(_uniform(seed + 1, 120.0, 127.0, (rows, cases)))
        value[:, :2 * cases, 0, 1:] = _uniform(seed + 3, -1.0, 1.0, (rows, 2 * cases, 15))
        # The cotangent drawn times 2^-4, so its product with a value of 2^126 stays within bf16's and
        # fp32's largest in the reference's backward.
        (dout,) = _normal(seed + 2, (rows, count, 1, 16))
        return _first(size, 16), key, value, None, dout * np.float32(2.0 ** -4)
    query_ids = np.repeat(np.arange(1, cases + 1, dtype=np.int32), queries)
    key_ids = np.zeros(32, np.int32)
    key_ids[:2 * cases] = np.repeat(np.arange(1, cases + 1, dtype=np.int32), 2)
    seen = np.broadcast_to((query_ids[:, None] == key_ids[None, :]) & (key_ids[None, :] > 0),
                           (rows, 1, count, 32)).copy()
    spans = ({"lengths": np.full(rows, 2)} if cases == 1 else
             {"segment_ids": CrossDocuments(np.tile(query_ids, (rows, 1)), np.tile(key_ids, (rows, 1)))})
    return Family(draw, seen, spans, dtype=dtype, decode_only=decode_only)


def _mixed_columns(seed: int, rows: int = 128, small: tuple[float, float] = (5e29, 1e30), *,
                   small_cotangent: bool = False):
    # One query a row over four keys of logit 0 whose values near fp32's largest in the first column
    # overflow its sum, and a fifth of logit -85 whose value 1e30 sits alone in the second column: the
    # normalized walk holds the row's sums over its denominator, and the second column's share, about 1e-7,
    # must survive it.
    key = np.zeros((rows, 5, 1, 16), np.float32)
    key[:, 4, 0, 0] = -_uniform(seed, 80.0, 86.0, (rows,))
    value = np.zeros((rows, 5, 1, 16), np.float32)
    value[:, :4, 0, 0] = _uniform(seed + 1, 2.5e38, 3.4e38, (rows, 4))
    value[:, 4, 0, 1] = _uniform(seed + 2, *small, (rows,))
    (dout,) = _normal(seed + 3, (rows, 1, 1, 16))
    if small_cotangent:
        # The cotangent on the small column alone, so D is that column's share, which the backward must
        # read from the unshifted sums (`algebra.normalized_split`).
        dout[..., 0] = 0.0
        dout[..., 2:] = 0.0
    return _first(np.full((rows, 1), 4.0, np.float32), 16), key, value, None, dout * np.float32(2.0 ** -8)


def _subnormal_share(seed: int, rows: int = 64, *, equal: bool = False):
    # One query a row over 256 keys: 255 of logits drawn in (-1, 0] (`equal`, all 0, so each weighs
    # 1/255, which bf16 rounds by 2^-8; their other entries drawn, which the query reads nothing of)
    # whose values near fp32's largest in the first column overflow its sum, and one of logit -84 to -88
    # whose value near 1e38 sits alone in the second column, its weight over the denominator below fp32's
    # least normal; the cotangent reads the second column alone, so D is that key's share and dQ's other
    # entries are D times the other keys' weighted rows. The cotangent a quarter of a normal draw keeps its
    # product with that value within fp32's range, as every fp32 arm's backward needs: a draw past 3.4 times
    # 1e38 takes the reference's dP, a single term, past it too (row 2022, ET-SUBREF-h100).
    key = np.zeros((rows, 256, 1, 16), np.float32)
    if not equal:
        key[:, :255, 0, 0] = -_uniform(seed + 5, 0.0, 1.0, (rows, 255))
    key[:, :255, 0, 1:] = _normal(seed, (rows, 255, 15))[0]
    key[:, 255, 0, 0] = -_uniform(seed + 1, 84.0, 88.0, (rows,))
    value = np.zeros((rows, 256, 1, 16), np.float32)
    value[:, :255, 0, 0] = _uniform(seed + 2, 2.9e38, 3.3e38, (rows, 255))
    value[:, 255, 0, 1] = _uniform(seed + 3, 5e37, 1e38, (rows,))
    dout = np.zeros((rows, 1, 1, 16), np.float32)
    dout[..., 1] = _normal(seed + 4, (rows, 1, 1))[0] * np.float32(0.25)
    return _first(np.full((rows, 1), 4.0, np.float32), 16), key, value, None, dout


def _tiny_beside_large(seed: int, rows: int = 128):
    """One query a row over four keys of equal logits, every value's first column drawn near fp32's
    largest (their sum overflows) and the first key's second column a drawn 2^-123 to 2^-120: that
    column's output, a quarter of it, is normal, which a shifted product of ordinary weights would flush."""
    value = np.zeros((rows, 4, 1, 16), np.float32)
    value[:, :, 0, 0] = _uniform(seed, 2.5e38, 3.4e38, (rows, 4))
    value[:, 0, 0, 1] = 2.0 ** np.floor(_uniform(seed + 1, -123.0, -119.0, (rows,)))
    (dout,) = _normal(seed + 2, (rows, 1, 1, 16))
    return (np.zeros((rows, 1, 1, 16), np.float32), np.zeros((rows, 4, 1, 16), np.float32), value, None,
            dout * np.float32(2.0 ** -8))


def _subnormal_band(seed: int, rows: int = 64, *, later: bool = False):
    """One query a row over three keys seen (with `later`, of 64, the third in the second key tile): the
    first of logit 0 on a value near bf16's largest in its second column, the second whose weight over the
    denominator is a drawn 2^-133 to 2^-126, bf16's subnormal band, on a value near bf16's largest in its
    first column, and the third of weight a drawn 1/200 to 1/2, which takes the denominator past 1; every
    other value 0. The reference rounds the second key's share to its subnormal and keeps its product, 2^-6
    to 2^1, which a weight halved past the denominator's binade rounds to zero (Main's counterexample to a
    power-of-two scale at the denominator's binade). The first key's weighted value passes 2^127, so the
    scaled walk scales the row (`algebra.SCALED_WALK`) and takes the second's weight, below fp32's least
    normal at that scale, in its small weights' product, raised (`algebra.at_scale`), so it keeps every bit
    the reference does. Every other key, which its documents leave unseen (`_band_seen`), sits at a logit
    of -10^4, a weight of 0 in any form, so a decode step, which reads a row's keys whole, reads the same
    row."""
    keys, third = (64, 32) if later else (3, 2)
    other = _uniform(seed, 1.0 / 200, 0.5, (rows,))
    share = 2.0 ** _uniform(seed + 1, -133.0, -126.0, (rows,))
    key = np.zeros((rows, keys, 1, 16), np.float32)
    key[:, :, 0, 0] = -1e4
    key[:, 0, 0, 0] = 0.0
    key[:, 1, 0, 0] = np.log(share * (1.0 + other))
    key[:, third, 0, 0] = np.log(other)
    value = np.zeros((rows, keys, 1, 16), np.float32)
    value[:, 1, 0, 0] = _uniform(seed + 2, 2.0 ** 127, 3.3e38, (rows,))
    value[:, 0, 0, 1] = _uniform(seed + 4, 2.9e38, 3.3e38, (rows,))
    (dout,) = _normal(seed + 3, (rows, 1, 1, 16))
    return _first(np.full((rows, 1), 4.0, np.float32), 16), key, value, None, dout * np.float32(2.0 ** -4)


def _band_seen(*, later: bool) -> tuple[np.ndarray, dict]:
    """`_subnormal_band`'s keys seen and their spans: the first three, or with `later` the first two and the
    second key tile's first, each of the one document its query reads (`CrossDocuments`)."""
    keys, third = (64, 32) if later else (3, 2)
    ids = np.zeros(keys, np.int32)
    ids[[0, 1, third]] = 1
    seen = np.broadcast_to(ids[None, :] == 1, (64, 1, 1, keys)).copy()
    return seen, {"segment_ids": CrossDocuments(np.ones((64, 1), np.int32), np.tile(ids, (64, 1)))}


def _extremes(dtype: str) -> tuple[float, float]:
    """The factor fp32's near-largest values (2.5e38 to 3.3e38) take and the binades its near-least normal
    ones (2^-125 to 2^-120) shift by to be near `dtype`'s own: none where `dtype` holds them (fp32, bf16),
    else its largest over fp32's and its least normal's exponent over fp32's (fp16's, 65504 and 2^-14)."""
    held = ml_dtypes.finfo(dtype)
    largest = 1.0 if float(held.max) >= 3.3e38 else float(held.max) / float(np.finfo(np.float32).max)
    least = float(held.smallest_normal)
    return largest, 0.0 if least <= 2.0 ** -125 else float(np.log2(least)) + 126.0


def _small_and_large(seed: int, rows: int, dtype: str) -> np.ndarray:
    """Values `[rows, 8, 1, 16]` near `dtype`'s extremes (`_extremes`): the first four keys' a drawn 2^-125
    to 2^-120 in their first column and a draw in [-1, 1) times 2^-120 past it, the last four's first
    column near the largest, 2.5e38 to 3.3e38, each at `dtype`'s own."""
    largest, shift = _extremes(dtype)
    value = np.zeros((rows, 8, 1, 16), np.float32)
    value[:, :4, 0, 0] = 2.0 ** (_uniform(seed, -125.0, -120.0, (rows, 4)) + np.float32(shift))
    value[:, :4, 0, 1:] = _uniform(seed + 1, -1.0, 1.0, (rows, 4, 15)) * np.float32(2.0 ** (shift - 120))
    value[:, 4:, 0, 0] = _uniform(seed + 2, 2.5e38, 3.3e38, (rows, 4)) * np.float32(largest)
    return value


def _masked_outlier(seed: int, rows: int = 64, *, dtype: str = "float32", every_column: bool = False):
    """One query a row over eight keys of equal logits, the first four seen and the last four past its
    length: the seen keys' values a drawn 2^-125 to 2^-120 in their first column and a draw in [-1, 1) times
    2^-120 past it, the unseen ones' first column near fp32's largest, each at `dtype`'s own extremes
    (`_small_and_large`). A key the row does not see weighs nothing, so no value it holds scales the row's
    sums or flushes the small ones it reads. The cotangent reads none of the first column but `every_column`:
    jax's attention, the reference arm, forms every key's `dO · vᵀ`, an unseen one's too, which a cotangent
    on that column carries past fp32's largest and its zero weight to NaN (FlashAttention's backward alike);
    Dew's are finite there (`test_an_unseen_keys_value_past_fp32s_largest_takes_no_part_in_a_gradient`)."""
    value = _small_and_large(seed, rows, dtype)
    (dout,) = _normal(seed + 3, (rows, 1, 1, 16))
    if not every_column:
        dout[..., 0] = 0.0
    return (np.zeros((rows, 1, 1, 16), np.float32), np.zeros((rows, 8, 1, 16), np.float32), value, None,
            dout)


def _separated_rows(seed: int, rows: int = 64, *, dtype: str = "float32"):
    """Two queries a row of one query tile over eight keys of equal logits, each its own document: the first
    sees the first four, whose values hold a drawn 2^-125 to 2^-120 in their first column and a draw in
    [-1, 1) times 2^-120 past it; the second sees the last four, whose values near fp32's largest pass it
    in their sum; each at `dtype`'s own extremes (`_small_and_large`). Each row's sums are its own: the
    second's size moves nothing of the first's."""
    value = _small_and_large(seed, rows, dtype)
    (dout,) = _normal(seed + 3, (rows, 2, 1, 16))
    # The cotangent times 2^-8, so its products with the large values stay finite in the reference's dP.
    return (np.zeros((rows, 2, 1, 16), np.float32), np.zeros((rows, 8, 1, 16), np.float32), value, None,
            dout * np.float32(2.0 ** -8))


SEPARATED = CrossDocuments(np.tile(np.asarray([1, 2], np.int32), (64, 1)),
                           np.tile(np.asarray([1] * 4 + [2] * 4, np.int32), (64, 1)))
"""`_separated_rows`' documents: the first query reads the first four keys, the second the last four."""

SEPARATED_SEEN = (SEPARATED.query[:, None, :, None] == SEPARATED.key[:, None, None, :]).copy()
"""`SEPARATED`'s keys each query sees `[64, 1, 2, 8]`."""


def _anchor_relative(seed: int, rows: int = 64):
    """One query a row over 33 keys: 32 whose logit is a drawn -10 to -9 and whose values hold a draw near
    fp32's largest in the first column and a draw near 2 in the second, then one whose logit is 40 and whose
    value holds a draw near 1 in the second column, the cotangent on the second column alone. The first
    tile overflows the first column; D is the second column's anchor-relative share, which the backward
    must read from sums kept apart from the anchor's (`algebra.normalized_terms`)."""
    key = np.zeros((rows, 33, 1, 16), np.float32)
    key[:, :32, 0, 0] = _uniform(seed, -10.0, -9.0, (rows, 32))
    key[:, 32, 0, 0] = 40.0
    value = np.zeros((rows, 33, 1, 16), np.float32)
    value[:, :32, 0, 0] = _uniform(seed + 1, 2.5e38, 3.4e38, (rows, 32))
    value[:, :32, 0, 1] = _uniform(seed + 2, 1.5, 2.5, (rows, 32))
    value[:, 32, 0, 1] = _uniform(seed + 3, 0.5, 1.5, (rows,))
    dout = np.zeros((rows, 1, 1, 16), np.float32)
    dout[..., 1] = _uniform(seed + 4, 0.5, 1.5, (rows, 1, 1))
    return _first(np.full((rows, 1), 4.0, np.float32), 16), key, value, None, dout


def _least_normal_column(seed: int, rows: int = 64):
    """One query a row over three keys of equal logits whose values' first column is a draw near fp32's
    largest and whose middle key's second column is 3 times 2^-126 times a draw in [1, 2): that column's
    output is near fp32's least normal, and the cotangent on it a draw near 2^100, so D is resolved only if
    the backward keeps that column out of the subnormal range."""
    value = np.zeros((rows, 3, 1, 16), np.float32)
    value[:, :, 0, 0] = _uniform(seed, 2.5e38, 3.4e38, (rows, 3))
    value[:, 1, 0, 1] = 3.0 * 2.0 ** -126 * _uniform(seed + 1, 1.0, 2.0, (rows,))
    dout = np.zeros((rows, 1, 1, 16), np.float32)
    dout[..., 1] = 2.0 ** 100 * _uniform(seed + 2, 1.0, 2.0, (rows, 1, 1))
    return (np.zeros((rows, 1, 1, 16), np.float32), np.zeros((rows, 3, 1, 16), np.float32), value, None,
            dout)


def _kept(queries: int, keys: int, rows: int = ROWS) -> np.ndarray:
    return np.ones((rows, 1, queries, keys), bool)


def _near_max_logit(width: int):
    # One key a row whose product with the query is a drawn fp32 near its largest (its key's first
    # entry in [1/2, 1) times a query near fp32's largest): the output is the key's value.
    def draw(seed):
        size = _uniform(seed, 1e38, 3e38, (ROWS, 1))
        key = _first(_uniform(seed + 1, 0.5, 1.0, (ROWS, 1)), width)
        value, dout = _normal(seed + 2, (ROWS, 1, 1, width), (ROWS, 1, 1, width))
        return _first(size, width), key, value, None, dout
    return draw


def _large_values(keys: int, top: float = 3.4e38, bottom: float | None = None, *, at_limit: bool = False):
    # Equal logits over values drawn within a few percent of fp32's largest (from `bottom` to `top`, bf16's
    # below its largest), one sign: the sum of the weighted values passes fp32's largest where the
    # reference's normalized one does not. `at_limit` draws fp32's largest and the float below it into every
    # row's first two columns, their output a value there (row 6842: a walk forming its operands' parts or
    # its sums largest first passed it). The cotangent is drawn times 2^-8, so the cotangent's product with
    # the values stays finite in the reference's backward too.
    low = 3.0e38 * top / 3.4e38 if bottom is None else bottom

    def draw(seed):
        value = _uniform(seed, low, top, (ROWS, keys, 1, 16))
        if at_limit:
            value = value.copy()
            value[:, :, 0, 0] = np.finfo(np.float32).max
            value[:, :, 0, 1] = np.nextafter(np.finfo(np.float32).max, np.float32(0))
        (dout,) = _normal(seed + 1, (ROWS, 1, 1, 16))
        return (np.zeros((ROWS, 1, 1, 16), np.float32), np.zeros((ROWS, keys, 1, 16), np.float32), value,
                None, dout * np.float32(2.0 ** -8))
    return draw


def _at_the_limit(keys: int):
    # Values in fp32's top binade's last 0.08%, [3.4e38, its largest], and the limit itself (`_large_values`).
    return _large_values(keys, float(np.finfo(np.float32).max), 3.4e38, at_limit=True)


def _drawn(queries, keys, width=16, rows=ROWS, size=1.0, heads=2):
    def draw(seed):
        query, key, value, dout = _normal(seed, (rows, queries, heads, width), (rows, keys, heads, width),
                                          (rows, keys, heads, width), (rows, queries, heads, width))
        return query * np.float32(size), key, value, None, dout
    return draw


def _with_sinks(base, low, high):
    # Drawn sink logits for each of the base draw's heads.
    def draw(seed):
        query, key, value, _, dout = base(seed)
        return query, key, value, _uniform(seed + 7, low, high, (query.shape[2],)), dout
    return draw


def _near(size: float):
    # Logits of `size` a row spread over 60 below it: each query's first entry a drawn `4 size`, each
    # key's 1 there and a drawn -240 to 0 in its second, where the query holds 1, the rest standard
    # normal, so every row weighs many keys at magnitudes fp32 rounds at `size`'s ulp.
    def draw(seed):
        query, key, value, dout = _normal(seed, (4, 64, 2, 16), (4, 64, 2, 16), (4, 64, 2, 16),
                                          (4, 64, 2, 16))
        query[..., 0] = 4 * _uniform(seed + 5, size / 2, size, query.shape[:-1])
        query[..., 1] = 1.0
        key[..., 0] = 1.0
        key[..., 1] = _uniform(seed + 6, -240.0, 0.0, key.shape[:-1])
        return query, key, value, None, dout
    return draw


def _beside(seed):
    # Two documents a row: two queries over a key of logit 0 and value near 1e38 and one of logit -87
    # and value near 1e30 in another column, which no overflow touches; four queries over four equal
    # keys whose values near 1e38 overflow their sum. Only the latter rows may shift.
    query, key, value, dout = _normal(seed, (ROWS, 6, 1, 16), (ROWS, 6, 1, 16), (ROWS, 6, 1, 16),
                                      (ROWS, 6, 1, 16))
    query[:, :2], query[:, :2, 0, 0] = 0.0, 1.0
    query[:, 2:] = 0.0
    key[:, 0], key[:, 1] = 0.0, 0.0
    key[:, 1, 0, 0] = -87.0 * 4
    value[:, 0, 0, 0] = _uniform(seed + 5, 5e37, 1e38, (ROWS,))
    value[:, 1, 0, 1] = _uniform(seed + 6, 5e29, 1e30, (ROWS,))
    value[:, 2:, 0, 0] = _uniform(seed + 7, 9e37, 1e38, (ROWS, 4))
    return query, key, value, None, dout * np.float32(2.0 ** -8)


def _slices(seed):
    # One query a row over 64 equal keys whose values near 1e37 sum within fp32 over each half and
    # past it over both: two decode slices each finite, whose merge overflows.
    value = np.array(_uniform(seed, 6e36, 1e37, (ROWS, 64, 1, 16)))
    (dout,) = _normal(seed + 1, (ROWS, 1, 1, 16))
    return (np.zeros((ROWS, 1, 1, 16), np.float32), np.zeros((ROWS, 64, 1, 16), np.float32), value, None,
            dout * np.float32(2.0 ** -8))


SPLIT_KEYS = 1024
"""A split decode step's keys: enough key tiles for its plan to cut its row into slices, each twice its
ring's stages of them (`attention_decode.plan`)."""


def _split_at_the_limit(seed):
    # Two rows of one query over `SPLIT_KEYS` keys, a decode step whose plan cuts each row into slices: key 0
    # at logit 0 and the last key at a logit near -ln 29, their values in fp32's top binade's last 0.08%
    # ([3.4e38, its largest]), every other key's logit far below and its value 0. Each slice's output is
    # near fp32's largest at a total near 1, their shares near 29/30 and 1/30, whose sum a merge passed
    # fp32's largest with (row 6826).
    width = 64
    query = np.zeros((2, 1, 1, width), np.float32)
    query[..., 0] = 1.0
    logits = np.full((2, SPLIT_KEYS), -1000.0, np.float32)
    logits[:, 0] = 0.0
    logits[:, -1] = -np.log(29.0) * _uniform(seed, 0.9, 1.1, (2,))
    key = np.zeros((2, SPLIT_KEYS, 1, width), np.float32)
    key[:, :, 0, 0] = logits * np.float32(np.sqrt(width))
    value = np.zeros((2, SPLIT_KEYS, 1, width), np.float32)
    largest = float(np.finfo(np.float32).max)
    value[:, [0, SPLIT_KEYS - 1]] = _uniform(seed + 1, 3.4e38, largest, (2, 2, 1, width))
    (dout,) = _normal(seed + 2, (2, 1, 1, width))
    return query, key, value, None, dout * np.float32(2.0 ** -8)


def _resumed(seed):
    # One query a row resumed from a state of denominator 2^100 to 2^127 and weighted values that
    # times a draw in [-1, 1), over two keys of logit 0 whose values are the denominator times a draw:
    # each key's share of the output is about that draw, which a shift for the denominator's size
    # must not flush.
    total = 2.0 ** np.floor(_uniform(seed, 100.0, 128.0, (RESUMED, 1, 1)))
    acc = total[..., None] * _uniform(seed + 1, -1.0, 1.0, (RESUMED, 1, 1, 16))
    value = total[:, 0, 0, None, None, None] * _uniform(seed + 2, -1.0, 1.0, (RESUMED, 2, 1, 16))
    # The cotangent drawn times 2^-12, so its product with values near 2^127 stays finite in the
    # reference's backward.
    (dout,) = _normal(seed + 3, (RESUMED, 1, 1, 16))
    state = (np.zeros((RESUMED, 1, 1), np.float32), total.astype(np.float32), acc.astype(np.float32))
    return (np.zeros((RESUMED, 1, 1, 16), np.float32), np.zeros((RESUMED, 2, 1, 16), np.float32),
            value.astype(np.float32), state, dout * np.float32(2.0 ** -12))


def _resumed_beside_overflow(seed):
    # One query a row of zeros, so both of its keys weigh alike, resumed from a state of denominator 1 at a
    # drawn maximum in [-1, 1) whose weighted values hold a draw in the second column alone; the keys'
    # values near fp32's largest in the first column overflow its sum, and the cotangent reads the second
    # column alone, the state's: a normalized training walk's split sums carry the state's share.
    m = _uniform(seed, -1.0, 1.0, (RESUMED, 1, 1))
    acc = np.zeros((RESUMED, 1, 1, 16), np.float32)
    acc[..., 1] = _uniform(seed + 1, -1.0, 1.0, (RESUMED, 1, 1))
    value = np.zeros((RESUMED, 2, 1, 16), np.float32)
    value[..., 0] = _uniform(seed + 2, 2.9e38, 3.4e38, (RESUMED, 2, 1))
    value[..., 2:] = _uniform(seed + 3, -1.0, 1.0, (RESUMED, 2, 1, 14))
    key, cotangent = _normal(seed + 4, (RESUMED, 2, 1, 16), (RESUMED, 1, 1))
    dout = np.zeros((RESUMED, 1, 1, 16), np.float32)
    dout[..., 1] = cotangent
    state = (m.astype(np.float32), np.ones((RESUMED, 1, 1), np.float32), acc)
    return np.zeros((RESUMED, 1, 1, 16), np.float32), key, value, state, dout


def _resumed_huge_denominator(seed):
    # One query a row of zeros over three keys of logit 0 whose values near fp32's largest in the first
    # column overflow its sum, resumed from a state of no values and a denominator of 2^126 to 2^127 at
    # maximum 0: each key's weight over the denominator is below fp32's least normal while its own
    # exponential is 1, so its share's product must stay finite; the cotangent is
    # drawn small, its product with the values finite in the reference's backward (`algebra.SMALL_SHIFT`).
    total = 2.0 ** np.floor(_uniform(seed, 126.0, 128.0, (RESUMED, 1, 1)))
    value = np.zeros((RESUMED, 3, 1, 16), np.float32)
    value[..., 0] = _uniform(seed + 1, 2.9e38, 3.3e38, (RESUMED, 3, 1))
    value[..., 1:] = _uniform(seed + 2, -1.0, 1.0, (RESUMED, 3, 1, 15))
    key, dout = _normal(seed + 3, (RESUMED, 3, 1, 16), (RESUMED, 1, 1, 16))
    state = (np.zeros((RESUMED, 1, 1), np.float32), total.astype(np.float32),
             np.zeros((RESUMED, 1, 1, 16), np.float32))
    return np.zeros((RESUMED, 1, 1, 16), np.float32), key, value, state, dout * np.float32(2.0 ** -10)


def _fp16_beside_a_large_state(seed):
    # One query a row of zeros over two keys whose values near fp16's largest sit in the first column,
    # resumed at maximum 0 from a state of denominator 2^124 to 2^126 whose weighted values are it times a
    # draw in [2, 2.5) in that column: its sums near fp32's largest, which fp16 values never take past it
    # (`algebra.headroom`: an fp16 call's walk is not normalized), so the output is the state's mean,
    # finite.
    total = 2.0 ** np.floor(_uniform(seed, 124.0, 127.0, (RESUMED, 1, 1)))
    acc = np.zeros((RESUMED, 1, 1, 16), np.float32)
    acc[..., 0] = total * _uniform(seed + 1, 2.0, 2.5, (RESUMED, 1, 1))
    acc[..., 1:] = total[..., None] * _uniform(seed + 2, -1.0, 1.0, (RESUMED, 1, 1, 15))
    value = np.zeros((RESUMED, 2, 1, 16), np.float32)
    value[..., 0] = _uniform(seed + 3, 6.0e4, 6.5e4, (RESUMED, 2, 1))
    value[..., 1:] = _uniform(seed + 4, -1.0, 1.0, (RESUMED, 2, 1, 15))
    key, dout = _normal(seed + 5, (RESUMED, 2, 1, 16), (RESUMED, 1, 1, 16))
    state = (np.zeros((RESUMED, 1, 1), np.float32), total.astype(np.float32), acc)
    # The cotangent drawn times 2^-4, so its product with the values, which the reference's backward holds
    # in fp16 as it holds the probabilities (row 1362), stays within fp16's largest.
    return np.zeros((RESUMED, 1, 1, 16), np.float32), key, value, state, dout * np.float32(2.0 ** -4)


def _near_keys(dtype: str):
    # One query a row over two keys an ulp of the dtype apart in every entry, `k` and `k (1 + 2^-t)`, t its
    # mantissa's bits: their logits differ by about an ulp of their size, so the two weigh alike, and their
    # values are drawn apart and large, so each key's score cotangent is large and the two cancel, `dS_1 =
    # -dS_0`: the query's gradient `dS_0 k_0 + dS_1 k_1` is their difference's share alone, an ulp of either
    # term, which a cotangent rounded to the dtype before its product with the keys loses
    # (`cotangent_product`).
    bits = 7 if dtype == "bfloat16" else 23

    def draw(seed):
        query, key, value, dout = _normal(seed, (ROWS, 1, 1, 16), (ROWS, 1, 1, 16), (ROWS, 2, 1, 16),
                                          (ROWS, 1, 1, 16))
        key = np.concatenate([key, key * np.float32(1.0 + 2.0 ** -bits)], axis=1)
        return query * np.float32(4.0), key, value * np.float32(1e4), None, dout
    return draw


def _large_keys(seed):
    # One query a row of zeros over eight keys near fp32's largest, so every logit is 0 and the eight weigh
    # alike, their values drawn: the keys' weighted sum overflows fp32 even at the softmax's scale of 1/4,
    # where their weighted mean does not, and the query's gradient is the cotangents' sum against the keys
    # (`attention_gradient.DqSums.keyed`).
    key = np.zeros((ROWS, 8, 1, 16), np.float32)
    key[..., 0] = _uniform(seed, 1.7e38, 3.4e38, (ROWS, 8, 1))
    value, dout = _normal(seed + 1, (ROWS, 8, 1, 16), (ROWS, 1, 1, 16))
    return np.zeros((ROWS, 1, 1, 16), np.float32), key, value, None, dout


def _far_maximum_in_a_later_tile(seed: int, rows: int = 64):
    # One query a row over 64 keys in two tiles of 32: the first tile's keys 88 to 92 nats below the second
    # tile's, which hold the row's maximum, so the first tile's weights are subnormal beside the maximum
    # the backward walks at: a dQ pass's running sums over its first tile (`attention_gradient.DqSums`) take
    # only those, and must stay finite and exact when the maximum's tile joins them.
    query = np.zeros((rows, 1, 1, 16), np.float32)
    query[..., 0] = 4.0
    key = np.zeros((rows, 64, 1, 16), np.float32)
    key[:, :32, 0, 0] = -_uniform(seed, 88.0, 92.0, (rows, 32))
    key[:, 32:, 0, 1:] = _normal(seed + 1, (rows, 32, 15))[0]
    value, dout = _normal(seed + 2, (rows, 64, 1, 16), (rows, 1, 1, 16))
    return query, key, value, None, dout


def _product_past_largest(seed: int):
    # One query a row over two keys whose products with it pass fp32's largest before the scale (2^128 to
    # 3 * 2^128 at a width of 16, scale 1/4) where the reference, which scales the query first, keeps them
    # finite (a quarter of that): the kernels take the scale's power of two on the query before their
    # products (`attention.prescale`). The larger logit takes every weight.
    query = np.zeros((ROWS, 1, 1, 16), np.float32)
    query[..., 0] = np.float32(2.0 ** 127) * _uniform(seed, 0.5, 1.0, (ROWS, 1, 1))
    key = np.zeros((ROWS, 2, 1, 16), np.float32)
    key[:, :, 0, 0] = _uniform(seed + 1, 4.0, 6.0, (ROWS, 2))
    value, dout = _normal(seed + 2, (ROWS, 2, 1, 16), (ROWS, 1, 1, 16))
    return query, key, value, None, dout


def _families() -> dict[str, Family]:
    small = functools.partial(_small_weight, queries=8, logits=(16.0, 21.0), sizes=(3.0, 5.0))
    # 1024 cases a draw: 64 rows of 16 for the template, whose interpreted Triton mainloop's time grows with
    # its programs, and a decode step's 1024 rows of one query, the decode kernel's shape.
    edge = functools.partial(_small_weight, queries=1, logits=(80.0, 87.0), sizes=None, rows=64, cases=16)
    edge_step = functools.partial(_small_weight, queries=1, logits=(80.0, 87.0), sizes=None, rows=1024,
                                  decode_only=True)
    families = {
        "ties at scale 1": Family(_ties(1, 1.0), _kept(1, 2)),
        "ties at scale 1, cotangents near 2**125": Family(_ties(1, 2.0 ** 125), _kept(1, 2)),
        "ties at scale 1/2": Family(_ties(4, 1.0), _kept(1, 2), folded=False),
        "ties at scale 1/2, cotangents near 2**125": Family(_ties(4, 2.0 ** 125), _kept(1, 2), folded=False),
        "ties at scale 1/4": Family(_ties(16, 1.0), _kept(1, 2), folded=False),
        "ties at scale 1/4, cotangents near 2**125": Family(_ties(16, 2.0 ** 125), _kept(1, 2), folded=False),
        # The fold's divergences at extreme inputs, which run in the accurate form alone (docs/design/
        # kernel-templates.md, §3, the exponent's forms): a logit near fp32's largest, whose contracted
        # weight is 0 or infinite; values near fp32's largest, whose weighted sum overflows where a walk
        # does not normalize its sums (every mainloop's default does, `overflows`); a sink there.
        "a logit near fp32's largest": Family(_near_max_logit(1), _kept(1, 1), folded=False),
        # Folded at scale 1/4, the backward takes the base-2 maximum back to natural units, which near fp32's
        # largest rounds the key's own weight away from 1, as at logits near 1e30.
        "a logit near fp32's largest at scale 1/4": Family(_near_max_logit(16), _kept(1, 1), folded=False),
        # A key at its row's maximum weighs 1 exactly in the accurate form; the fold's contracted weight is
        # 1 within an ulp of its logit's exponent.
        "one key": Family(_drawn(4, 1), _kept(4, 1), folded=False),
        # Sixteen heads, so each draw holds sixteen sinks' gradients.
        "a sink and one key": Family(_with_sinks(_drawn(16, 1, rows=SINK_ROWS, heads=16), -3.0, 3.0),
                                     _kept(16, 1, SINK_ROWS)),
        "a sink near fp32's largest and one key": Family(_with_sinks(_drawn(16, 1), 1e38, 3e38),
                                                         _kept(16, 1), folded=False),
        "a sink and one key, its logit near fp32's largest": Family(
            _with_sinks(_near_max_logit(1), -3.0, 3.0), _kept(1, 1), folded=False),
        "a sink near fp32's largest at scale 1/4": Family(_with_sinks(_drawn(16, 1), 1e38, 3e38),
                                                          _kept(16, 1), folded=False),
        "logits near 1e4": Family(_near(1e4), _causal(64, 64, 4), {"causal": True}),
        # Folded, the backward takes the base-2 maximum back to natural units, which at 1e30 rounds the
        # anchor's own weight away from 1 by more than any bound states.
        "logits near 1e30": Family(_drawn(64, 64, rows=4, size=1e30), _causal(64, 64, 4), {"causal": True},
                                   folded=False),
        # Folded in fp32, one weight carries each row's output, so the fold's extra rounding of its logit is
        # the row's error: 2.04 times the reference's (an extreme-magnitude divergence the design names).
        "a small weight on a large value, float32": dataclasses.replace(small("float32"), folded=False),
        "a small weight on a large value, bfloat16": small("bfloat16"),
        "a weight at the exponent's edge on a large value, float32": edge("float32"),
        "a weight at the exponent's edge on a large value, bfloat16": edge("bfloat16"),
        "a decode step's weight at the exponent's edge on a large value, float32": edge_step("float32"),
        "a decode step's weight at the exponent's edge on a large value, bfloat16": edge_step("bfloat16"),
        # Their one divergence in the default form is their sums passing fp32's largest (`overflows`): the
        # fold's sum of those values overflows where a walk does not normalize its sums.
        "a small column beside an overflowing one": Family(_mixed_columns, np.ones((128, 1, 1, 5), bool),
                                                           folded=False, overflows=True),
        # Its values near 1, so the column's output is near fp32's least normal, a shifted product of
        # which would flush (`algebra.Partial.low`).
        "a column near fp32's least normal beside an overflowing one": Family(
            functools.partial(_mixed_columns, small=(0.5, 1.0)), np.ones((128, 1, 1, 5), bool), folded=False,
            overflows=True),
        "a cotangent on a small column beside an overflowing one": Family(
            functools.partial(_mixed_columns, small_cotangent=True), np.ones((128, 1, 1, 5), bool),
            folded=False, overflows=True),
        "a tiny value beside an overflowing column at equal weights": Family(
            _tiny_beside_large, np.ones((128, 1, 1, 4), bool), folded=False, overflows=True),
        "an anchor-relative share beside an overflowing column": Family(
            _anchor_relative, np.ones((64, 1, 1, 33), bool), folded=False, overflows=True),
        "a column near fp32's least normal under a large cotangent": Family(
            _least_normal_column, np.ones((64, 1, 1, 3), bool), folded=False, overflows=True),
        # A share whose normalized weight is subnormal, in either dtype (the normalized pass's small
        # weights, `algebra.normalized_terms`).
        "a subnormal share of an overflowing row, float32": Family(
            _subnormal_share, np.ones((64, 1, 1, 256), bool), folded=False, overflows=True),
        "a subnormal share of an overflowing row, bfloat16": Family(
            _subnormal_share, np.ones((64, 1, 1, 256), bool), folded=False, overflows=True, dtype="bfloat16"),
        # Every weight but one 1/255, bf16's worst rounding of a weight, beside the subnormal share.
        "keys near fp32's largest at equal logits": Family(_large_keys, _kept(1, 8)),
        "a far maximum in a later key tile": Family(_far_maximum_in_a_later_tile,
                                                    np.ones((64, 1, 1, 64), bool)),
        # Its logits near fp32's largest, so the default form's fold takes them past it (the fold's documented
        # divergence, `algebra.Forward.base2`): the accurate form alone holds check 1.
        "a product past fp32's largest before a small scale, float32": Family(
            _product_past_largest, _kept(1, 2), folded=False, query_first=True),
        "a product past fp32's largest before a small scale, bfloat16": Family(
            _product_past_largest, _kept(1, 2), folded=False, dtype="bfloat16", query_first=True),
        # A query's gradient that cancels across its keys, in either dtype (`algebra.cotangent_product`).
        "keys an ulp apart, float32": Family(_near_keys("float32"), _kept(1, 2)),
        "keys an ulp apart, bfloat16": Family(_near_keys("bfloat16"), _kept(1, 2), dtype="bfloat16"),
        "a subnormal share beside 255 equal weights, bfloat16": Family(
            functools.partial(_subnormal_share, equal=True), np.ones((64, 1, 1, 256), bool), folded=False,
            overflows=True, dtype="bfloat16"),
        # A share in bf16's subnormal band on a value near its largest, its denominator past 1 in its own key
        # tile or a later one (the running denominator then smaller than the final one).
        "a share in bf16's subnormal band on a value near its largest": Family(
            _subnormal_band, *_band_seen(later=False), folded=False, dtype="bfloat16"),
        "a share in bf16's subnormal band, the denominator's rest in a later tile": Family(
            functools.partial(_subnormal_band, later=True), *_band_seen(later=True), folded=False,
            dtype="bfloat16"),
        # A row of small values beside keys it does not see whose values near fp32's largest (its length's
        # padding), and rows of one query tile of either size, each its own.
        **{f"small values beside unseen ones near fp32's largest, {dtype}": Family(
            _masked_outlier, np.broadcast_to(np.arange(8) < 4, (64, 1, 1, 8)).copy(),
            {"lengths": np.full(64, 4)}, dtype=dtype,
            drawn_in=lambda dtype: functools.partial(_masked_outlier, dtype=dtype))
           for dtype in ("float32", "bfloat16")},
        **{f"a row of small values beside one that overflows in its tile, {dtype}": Family(
            _separated_rows, SEPARATED_SEEN, {"segment_ids": SEPARATED}, folded=False, overflows=True,
            dtype=dtype, drawn_in=lambda dtype: functools.partial(_separated_rows, dtype=dtype))
           for dtype in ("float32", "bfloat16")},
    }
    return {**families, **_large_value_families(), **_structured_families()}


def _large_value_families() -> dict[str, Family]:
    """The families of equal logits over values near fp32's largest (`_families`'), whose weighted sums pass
    it where the reference's normalized mean does not: to 3.4e38, and at fp32's limit itself."""
    return {
        "values near fp32's largest over 2 equal keys": Family(_large_values(2), _kept(1, 2), folded=False,
                                                               overflows=True),
        "values near fp32's largest over 64 equal keys": Family(_large_values(64), _kept(1, 64),
                                                                folded=False, overflows=True),
        # Eight key tiles of 32 (`PINS`, a decode step's `BLOCK`): a walk's chain of its tiles' sums errs past
        # the reference's, which sums a row whole, unless the walk keeps their additions' roundings
        # (`algebra.compensates`).
        "values near fp32's largest over 256 equal keys": Family(_large_values(256), _kept(1, 256),
                                                                 folded=False, overflows=True),
        "values near fp32's largest over 2 equal keys, bfloat16": Family(
            _large_values(2, top=3.3e38), _kept(1, 2), folded=False, overflows=True, dtype="bfloat16"),
        "values at fp32's largest over 2 equal keys": Family(_at_the_limit(2), _kept(1, 2), folded=False,
                                                             overflows=True),
        "values at fp32's largest over 256 equal keys": Family(_at_the_limit(256), _kept(1, 256),
                                                               folded=False, overflows=True),
    }


def _structured_families() -> dict[str, Family]:
    """The families of a call's structure beside its values (`_families`'): rows of no keys, documents, slices
    merged, resumed states and a decode step's sinks."""
    families = {}
    lengths = np.tile([4, 0], ROWS // 2)
    empty = np.broadcast_to((lengths > 0)[:, None, None, None], (ROWS, 1, 4, 4))
    families["a row of no keys"] = Family(_drawn(4, 4), empty, {"lengths": lengths},
                                          empty=np.broadcast_to((lengths == 0)[:, None], (ROWS, 4)))
    before = _causal(4, 4, ROWS) & (np.arange(4) >= 2)
    families["causal queries before every key"] = Family(
        _drawn(4, 4), before, {"causal": True, "starts": np.full(ROWS, 2)},
        empty=np.broadcast_to(np.asarray([True, True, False, False]), (ROWS, 4)))

    documents = np.tile(np.asarray([[1, 1, 2, 2, 2, 2]], np.int32), (ROWS, 1))
    families["a row that overflows beside one that does not"] = Family(
        _beside, documents[:, None, :, None] == documents[:, None, None, :], {"segment_ids": documents},
        folded=False, overflows=True)

    families["two finite slices whose merge overflows"] = Family(_slices, _kept(1, 64), splits=2,
                                                                 folded=False, overflows=True)
    families["values at fp32's largest across a split decode's slices"] = Family(
        _split_at_the_limit, _kept(1, SPLIT_KEYS, rows=2), splits=4, overflows=True, decode_only=True)

    families["a resumed state of a large denominator"] = Family(_resumed, np.ones((RESUMED, 1, 1, 2), bool),
                                                                folded=False)
    # Their one divergence in the default form is their sums passing fp32's largest (`overflows`).
    families["a resumed state beside an overflowing column"] = Family(
        _resumed_beside_overflow, np.ones((RESUMED, 1, 1, 2), bool), folded=False, overflows=True)
    families["a resumed state of a denominator near fp32's largest beside an overflowing column"] = Family(
        _resumed_huge_denominator, np.ones((RESUMED, 1, 1, 3), bool), folded=False, overflows=True)
    families["fp16 values near their largest beside a state near fp32's largest"] = Family(
        _fp16_beside_a_large_state, np.ones((RESUMED, 1, 1, 2), bool), dtype="float16")
    for low, high, label in ((-3.0, 3.0, ""), (1e38, 3e38, " near fp32's largest")):
        # One query a row, so the decode kernel takes the family too.
        name = f"a decode step's sink{label} and one key"
        # Thirty-two heads a row, so each draw holds 32 sinks' gradients over few rows.
        families[name] = Family(_with_sinks(_drawn(1, 1, rows=SINK_ROWS, heads=32), low, high),
                                _kept(1, 1, SINK_ROWS), folded=low < 0)
    ids = np.tile(np.asarray([[1, 2, 3, 3]], np.int32), (ROWS, 1))
    families["documents of one key"] = Family(_drawn(4, 4), _causal(4, 4, ROWS) & (ids[:, None, :, None] ==
                                                                                  ids[:, None, None, :]),
                                              {"causal": True, "segment_ids": ids})
    return families


FAMILIES = _families()


@functools.cache
def _drawn_first(family: str) -> tuple:
    """`FAMILIES[family]`'s draw at seed 0, drawn once a process: the shapes and state the family lists read
    (`DECODED`, `hookless_families`)."""
    return FAMILIES[family].draw(0)


FORMS = {"default": None, "natural": True}
"""The forms the decode kernel takes each family in: the default (`template.NATURAL`) and natural logits."""
FAMILY_RUNS = [(family, form) for family in FAMILIES for form in FORMS
               if form == "natural" or FAMILIES[family].folded]
TEMPLATE_FAMILIES = [family for family in FAMILIES if not FAMILIES[family].decode_only]
"""The template's families: every one but a decode step's alone (`Family.decode_only`)."""
ACCURATE = {"natural": True, "accurate": True}
"""The template's accurate form: natural logits and the backward's accurate form."""


def _freeze_value(value):
    """Hashable structural key for a `Family.spans` entry or `Family.keep` mask (`_spans_key`, row 7274)."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, CrossDocuments):
        return ("cross", _freeze_value(value.query), _freeze_value(value.key))
    if isinstance(value, (tuple, list)):
        return ("seq", tuple(_freeze_value(item) for item in value))
    return ("scalar", value)


def _thaw_value(frozen):
    """Reconstruct a `Family.spans` entry or `Family.keep` mask from `_freeze_value`."""
    tag = frozen[0]
    if tag == "ndarray":
        _, shape, dtype_str, raw = frozen
        return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    if tag == "cross":
        return CrossDocuments(_thaw_value(frozen[1]), _thaw_value(frozen[2]))
    if tag == "seq":
        return tuple(_thaw_value(item) for item in frozen[1])
    return frozen[1]


def _spans_key(spans: dict) -> tuple:
    return tuple((name, _freeze_value(value)) for name, value in sorted(spans.items()))


def _spans_from_key(spans_key: tuple, query, key) -> template.KeySpans:
    def placed(frozen):
        value = _thaw_value(frozen)
        if isinstance(value, CrossDocuments):
            return CrossDocuments(jnp.asarray(value.query), jnp.asarray(value.key))
        return jnp.asarray(value) if isinstance(value, np.ndarray) else value

    arguments = {name: placed(frozen) for name, frozen in spans_key}
    return template.spans(query.shape[1], key.shape[1], query.shape[0], **arguments)


def _spans(family: Family, query, key) -> template.KeySpans:
    return _spans_from_key(_spans_key(family.spans), query, key)


def _hooks(query, extra) -> template.Hooks | None:
    """The template's hooks for a draw's sinks `[H]` or resumed state `(m, total, acc)`."""
    if extra is None:
        return None
    if isinstance(extra, tuple):
        return template.Hooks(initial=template.Initial(*extra))
    batch, queries, _, width = query.shape
    return template.Hooks(initial=template.sinks(extra, batch, queries, width))


def _cast(family: Family, draw: Callable[[int], tuple]) -> Callable[[int], tuple]:
    """`draw` with its operands rounded to the family's dtype, the sinks fp32."""
    def cast(seed: int):
        query, key, value, extra, dout = draw(seed)
        operands = (jnp.asarray(x, jnp.dtype(family.dtype)) for x in (query, key, value, dout))
        query, key, value, dout = operands
        return query, key, value, jax.tree.map(jnp.asarray, extra), dout

    return cast


@functools.cache
def _cached_reference_attend(keep_key: tuple, dtype_name: str, query_first: bool, *, anchored: bool,
                             round_logits=None) -> Callable:
    """Reference forward `(query, key, value, extra) -> out` cached by structural mask/dtype/form so
    families with identical structure share one compiled XLA and float64 truth program (row 7274)."""
    keep = _thaw_value(keep_key)
    cpu_float16 = jnp.dtype(dtype_name) == jnp.float16 and jax.default_backend() == "cpu"
    jax_holds = not anchored and np.asarray(keep).any(-1).all() and not cpu_float16

    def attend(query, key, value, extra):
        if isinstance(extra, tuple):
            out = _resumed_attention(query, key, value, extra, keep, anchored=anchored,
                                     round_logits=round_logits)
        elif extra is None and jax_holds and query_first:
            scale = query.shape[-1] ** -0.5
            assert float(np.log2(scale)).is_integer(), "a query scaled first is exact at a power of two alone"
            out = jax.nn.dot_product_attention(query * jnp.asarray(scale, query.dtype), key, value, scale=1.0,
                                               mask=jnp.asarray(keep), implementation="xla")
        elif extra is None and jax_holds:
            out = jax.nn.dot_product_attention(query, key, value, mask=jnp.asarray(keep),
                                               implementation="xla")
        else:
            out = exact_attention(query, key, value, extra, keep=keep, anchored=anchored,
                                  round_logits=round_logits)
        return out.astype(query.dtype)

    return attend


@functools.cache
def _cached_reference_fn(keep_key: tuple, dtype_name: str, query_first: bool, *, anchored: bool,
                         round_logits=None) -> Callable:
    return with_gradients(_cached_reference_attend(
        keep_key, dtype_name, query_first, anchored=anchored, round_logits=round_logits))


@functools.cache
def _jitted_reference(keep_key: tuple, dtype_name: str, query_first: bool, *, anchored: bool,
                      round_logits=None) -> Callable:
    return as_given(jax.jit(_cached_reference_fn(
        keep_key, dtype_name, query_first, anchored=anchored, round_logits=round_logits)))


def _reference_key(family: Family) -> tuple[tuple, str, bool]:
    return _freeze_value(np.asarray(family.keep)), jnp.dtype(family.dtype).name, bool(family.query_first)


def _reference(family: Family, *, anchored: bool, round_logits=None) -> Callable:
    """The reference's output `(query, key, value, extra) -> out`, in the call's dtype as every attention
    call returns it (`_cached_reference_attend`, row 7274)."""
    return _cached_reference_attend(*_reference_key(family), anchored=anchored, round_logits=round_logits)


def _resumed_attention(query, key, value, state, keep, *, anchored: bool, round_logits=None):
    """Attention resumed from a softmax state `(m, total, acc)` (one query head a key head), written apart
    from Dew: the state one more column of the softmax, its logit `m + log total` and its value its values'
    mean `acc / total`, so a state whose sums are near fp32's largest stays finite; in fp32 or the inputs'
    wider dtype, as the state is, the probabilities rounded to the values' dtype for their product (row
    1362). Its derivative the float64 truth's where `anchored` (`anchored_softmax`): the state's logit's
    is `p_state sum_k p_k c (mean - v_k)`, no term cancelling, where a state of total 2^126 beside two keys
    had its `2 mean / total` share rounded off `total + 2` (row 6302: 6.34e-35 against 1.268e-34)."""
    wide = jnp.promote_types(query.dtype, jnp.float32)
    m, total, acc = (jnp.asarray(x, wide) for x in state)
    query, key = query.astype(wide), key.astype(wide)
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(jnp.asarray(query.shape[-1], wide))
    if round_logits is not None:
        logits = logits + jax.lax.stop_gradient(logits.astype(round_logits).astype(logits.dtype) - logits)
    logits = jnp.where(keep, logits, -jnp.inf)
    if anchored:
        joined = jnp.concatenate([(m + jnp.log(total))[..., None], logits], axis=-1)
        weights = anchored_softmax(joined)
        mean = acc / total[..., None]
        weighted = jnp.einsum("bhqk,bkhd->bhqd", weights[..., 1:].astype(value.dtype), value,
                              preferred_element_type=wide)
        out = mean * weights[..., :1] + weighted
    else:
        top = jnp.maximum(m, jnp.max(logits, axis=-1))
        prior = total * jnp.exp(m - top)
        exp_logits = jnp.exp(logits - top[..., None])
        denom = prior + jnp.sum(exp_logits, axis=-1)
        weighted = jnp.einsum("bhqk,bkhd->bhqd", (exp_logits / denom[..., None]).astype(value.dtype), value,
                              preferred_element_type=wide)
        out = (acc / total[..., None]) * (prior / denom)[..., None] + weighted
    return jnp.swapaxes(out, 1, 2)


def _differentiated(forward: Callable, backward: Callable) -> Callable:
    """`(query, key, value, initial) -> out`, a mainloop's `forward` whose derivative is its own
    `backward` (`template.Gradients`), the initial state's included."""
    @jax.custom_vjp
    def attend(query, key, value, initial):
        return forward(query, key, value, initial)

    def saved(query, key, value, initial):
        return forward(query, key, value, initial), (query, key, value, initial)

    def pulled(operands, dout):
        grads = backward(*operands, dout)
        return grads.query, grads.key, grads.value, grads.initial

    attend.defvjp(saved, pulled)
    return attend


def _differentiable_from_key(spans_key: tuple, forward: Callable, backward: Callable, *,
                             dropping: bool = False) -> Callable:
    """A mainloop's `forward` and `backward` as `(query, key, value, extra) -> out` over `spans_key`."""
    def run(query, key, value, extra):
        spans, hooks = _spans_from_key(spans_key, query, key), _hooks(query, extra)
        drop = _dropping(query).drop if dropping else None

        def ahead(query, key, value, initial):
            return forward(query, key, value, spans, template.Hooks(initial=initial, drop=drop))

        def back(query, key, value, initial, dout):
            return backward(query, key, value, spans, template.Hooks(initial=initial, drop=drop), dout)

        initial = None if hooks is None else hooks.initial
        return _differentiated(ahead, back)(query, key, value, initial)

    return run


def _differentiable(family: Family, forward: Callable, backward: Callable, *, dropping: bool = False
                    ) -> Callable:
    """A mainloop's `forward` and `backward` as `(query, key, value, extra) -> out` over `family`'s spans,
    its derivative the mainloop's own backward, the sinks' or resumed state's gradient through the softmax's
    initial state; under `_dropping`'s draw where `dropping`."""
    return _differentiable_from_key(_spans_key(family.spans), forward, backward, dropping=dropping)

DROP_RATE = 0.25
"""The dropout rate a dropping family's call takes (`_dropping`): a kept key's multiplier 4/3, inexact in
every format, so a walk that weighs it per key rounds where the reference's does not (row 8652)."""


def _dropping(query) -> template.Hooks:
    """A dropout draw at `DROP_RATE` over `query`'s rows and heads, its key fixed."""
    batch, _, heads, _ = query.shape
    return template.Hooks(drop=template.Dropout(jax.random.bits(jax.random.PRNGKey(7), (2,), jnp.uint32),
                                                jnp.arange(batch, dtype=jnp.int32),
                                                jnp.arange(heads, dtype=jnp.int32), DROP_RATE))


@functools.cache
def _jitted_mainloop(name: str, spans_key: tuple, options_key: tuple, dropping: bool) -> Callable:
    """One cached `jax.jit(with_gradients(...))` program per `(mainloop, spans, options, dropping)` shared
    across all stability families with identical structure, avoiding 30+ redundant host traces and
    certifications on GPU tiers (row 7274)."""
    options = dict(options_key)
    forward_options = {key: value for key, value in options.items() if key in ("natural", "scale")}
    tiles = attention_portable.Tiles(32, 32, attention_triton.BLOCK)
    mainloop = functools.partial(_differentiable_from_key, spans_key, dropping=dropping)

    def triton_forward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.forward(*operands, interpret=INTERPRETED, **forward_options)

    def triton_backward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.backward(*operands, interpret=INTERPRETED, **options)

    def mosaic_forward(*operands):
        return attention_mosaic.forward(*operands, **forward_options)

    def mosaic_backward(query, key, value, spans, hooks, dout):
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, residuals=template.Kept.SPLIT,
                                             **forward_options)[1:]
        return attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                         **options)

    def two_launch_backward(*operands):
        with tuning.pinned({attention_mosaic.MosaicBackward: TWO_LAUNCHES}):
            return mosaic_backward(*operands)

    if name == "portable":
        fn = mainloop(functools.partial(attention_portable.xla_mainloop, tiles, **forward_options),
                      functools.partial(attention_portable.backward, tiles, **options))
    elif name == "triton":
        fn = mainloop(triton_forward, triton_backward)
    elif name == "mosaic":
        fn = mainloop(mosaic_forward, mosaic_backward)
    elif name == "mosaic two launches":
        fn = mainloop(mosaic_forward, two_launch_backward)
    else:
        raise KeyError(name)
    return jax.jit(with_gradients(fn))


def _mainloop_names(family: Family, options: dict, *, dropping: bool = False) -> tuple[str, ...]:
    names = ["portable"] if dropping else ["triton", "portable"]
    if ON_HOPPER and _mosaic_takes(family, family.draw(0)):
        names.append("mosaic")
        if not options and family.draw(0)[3] is None:
            names.append("mosaic two launches")
    return tuple(names)


def _mainloops(family: Family, options: dict, *, dropping: bool = False) -> dict[str, Callable]:
    """Each template mainloop as a cached jitted `(query, key, value, extra, dout) -> (out, grads)` at the
    form `options` names (`ACCURATE`, or none for the default; row 7274)."""
    spans_key = _spans_key(family.spans)
    options_key = tuple(sorted(options.items()))
    return {name: _jitted_mainloop(name, spans_key, options_key, dropping)
            for name in _mainloop_names(family, options, dropping=dropping)}


@functools.cache
def _cached_truth(truth_fn: Callable, device) -> Truth:
    return Truth(truth_fn, device)


@functools.cache
def _cached_dropping_truth(spans_key: tuple) -> Callable:
    def truth(query, key, value, extra):
        del extra
        return template.reference(query, key, value, _spans_from_key(spans_key, query, key), _dropping(query))

    return with_gradients(truth)


TWO_LAUNCHES = attention_mosaic.BackwardTiles(64, 64, 2, 1, fused=False, family="wgmma")
"""The Mosaic backward's two-launch schedule at the derived tiles, the dQ launch then the dK/dV one at the
refined D: a hookless fast-form call's in deterministic mode, where its derived one is one launch."""


ON_HOPPER = here() == "sm90"
"""Whether the Mosaic mainloop runs here, compiled on an H100: interpreted, a family takes it 6 to 33 minutes
on a CPU, past a CI task's time, and the interpreter is coverage, not proof (`test_attention_mosaic`)."""


def _mosaic_takes(family: Family, drawn: tuple) -> bool:
    """Whether the Mosaic mainloop takes `family`'s calls, forward and backward, by its own legality rule
    (`attention_mosaic.refused`), read off a draw of the family's (`drawn`): its hooks' types alone
    (`jax.eval_shape`), computing nothing, as a GPU tier part's prepare builds the cells reading it, whose
    device computes nothing (rows 8025, 8071-8073)."""
    query, _, value, extra, _ = drawn
    hooks = jax.eval_shape(_hooks, query, extra)
    return attention_mosaic.refused(jnp.dtype(family.dtype), query.shape[-1], value.shape[-1], hooks,
                                    backward=True, family=mosaic_sm90.WGMMA) is None


def _held_empty(empty, found, label: str) -> None:
    """A row that sees no key reads nothing and passes nothing back: its output and query gradient
    (`found`'s first two leaves) are zero exactly at a family's `empty` rows, on each draw checked (row
    9041)."""
    rows = np.asarray(empty)
    for leaf, array in zip(("output", "dq"), found[:2], strict=True):
        np.testing.assert_array_equal(np.asarray(array, np.float32)[rows], 0.0, err_msg=f"{label} {leaf}")


def assert_finite(dew_fn, truth_fn, draw, label: str, *, seeds: int, empty=None) -> None:
    """The stability rule's finiteness over `seeds` draws: each leaf of `dew_fn`'s outputs finite wherever the
    float64 truth (`truth_fn` on the inputs widened) rounded to the leaf's dtype is. A reference computing
    in the call's own dtype is no measure of it: fp16's written-out attention at `logits near 1e4` gives a
    key gradient of -45376 where the truth's, -68612, rounds past fp16's largest (FS-F16-DK1). The truth runs
    on the host's CPU device, as the lane runs float64 references (tests/lane_environment.py): a family's
    draws are small, and XLA:GPU compiled each truth in 7.3 s where XLA:CPU takes 0.2 (MAINGPU-0243234e8's
    part 2 on an H100: 25 truths, 182 s). A family's `empty` rows, each draw's outputs held empty there
    (`_held_empty`)."""
    truth_of = _cached_truth(truth_fn, jax.local_devices(backend="cpu")[0])
    for seed in range(seeds):
        inputs = draw(seed)
        with jax.enable_x64(new_val=True):
            truth = jax.tree.leaves(truth_of(*widened(jax.device_get(inputs))))
        outputs = dew_fn(*inputs)
        if empty is not None:
            _held_empty(empty, outputs, f"{label} seed {seed}")
        found = jax.tree_util.tree_leaves_with_path(outputs)
        for (path, mine), wide in zip(found, truth, strict=True):
            mine = np.asarray(mine)
            rounded = np.asarray(wide, np.float64).astype(mine.dtype)
            lost = np.isfinite(rounded.astype(np.float32)) & ~np.isfinite(mine.astype(np.float32))
            assert not lost.any(), (f"{label}{jax.tree_util.keystr(path)} seed {seed}: {int(lost.sum())} "
                                    "values not finite where the truth's rounded to their dtype are")


GPU_ONLY = {
    "accurate": {("a small weight on a large value, bfloat16", "triton"),
                 ("a small weight on a large value, float32", "triton"),
                 ("a resumed state of a large denominator", "triton"),
                 ("a resumed state beside an overflowing column", "triton"),
                 ("a small column beside an overflowing one", "triton"),
                 ("a cotangent on a small column beside an overflowing one", "triton"),
                 ("a column near fp32's least normal beside an overflowing one", "triton")},
    "frozen rows": {"a small column beside an overflowing one",
                    "a resumed state beside an overflowing column",
                    "a column near fp32's least normal beside an overflowing one",
                    "a cotangent on a small column beside an overflowing one"},
    "decode": {("a decode step's weight at the exponent's edge on a large value, float32", "default")}}
"""The cases whose Pallas-Triton kernel interpreted on a CPU passes the suite's 120 s target (118 to 325 s on
armada, train 281's and 283's CI, rows 2117, 2193, 2204), by test: each runs compiled on a GPU in its flow
(tests/test_gpu_attention.py), which proves the kernel the route runs, in seconds, where the interpreter
proves an interpreter (as `GPU_TRITON_FAMILIES`')."""


ACCURATE_ARMS = [(family, name) for family in TEMPLATE_FAMILIES
                 for name in _mainloops(FAMILIES[family], ACCURATE)
                 if (family, name) not in GPU_ONLY["accurate"]]
"""Each template family beside each mainloop that takes it in the accurate form (`_mainloops`): a case of
its own, so none holds a CPU task past the suite's target (row 1849); those past it on a GPU (`GPU_ONLY`)."""


@pytest.mark.parametrize(("family", "name"), ACCURATE_ARMS)
def test_the_accurate_form_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family, name):
    """A template mainloop's output and query, key and value gradients in the accurate form (`ACCURATE`),
    check 1 over the family's draws, merging no ties the reference keeps apart (`assert_merges_no_ties`,
    row 6302); a row that sees nothing, zero."""
    seeds, spec = SEEDS[family], FAMILIES[family]
    draw = _cast(spec, spec.draw)
    ref_key = _reference_key(spec)
    reference = _jitted_reference(*ref_key, anchored=False)
    truth = _cached_reference_fn(*ref_key, anchored=True)
    rounded = _cached_reference_fn(*ref_key, anchored=True, round_logits=jnp.float32)
    # Each family a site of its own (row 13030: every float32 family one label, so a GPU part holding two
    # prepared one and read its programs for the other).
    label = f"{family}: {name} accurate {spec.dtype}"
    differentiated = _mainloops(spec, ACCURATE)[name]
    # Each result check 1 draws (its second stage's too) held empty at the family's empty rows (row 9041).
    held = (None if spec.empty is None
            else lambda outputs, inputs: _held_empty(spec.empty, outputs, label))
    evaluated = assert_as_exact(differentiated, reference, draw, label, seeds=seeds, truth_fn=truth,
                                held=held, returns=True)
    assert_merges_no_ties(differentiated, reference, truth, draw, label, seeds=seeds, rounded_fn=rounded,
                          evaluated=evaluated)


def default_is_finite(family_name: str, mainloop_name: str) -> None:
    """A template mainloop's output and query, key and value gradients in the default form, finite wherever
    the float64 truth's rounded to their dtype are (`assert_finite`) over the family's draws; a row that
    sees nothing, zero. Its error here is no claim, and its ties the documented divergence (the module's
    docstring)."""
    seeds, family = SEEDS[family_name], FAMILIES[family_name]
    draw = _cast(family, family.draw)
    truth = _cached_reference_fn(*_reference_key(family), anchored=True)
    label = f"{family_name}: {mainloop_name} default {family.dtype}"
    differentiated = _mainloops(family, {})[mainloop_name]
    assert_finite(differentiated, truth, draw, label, seeds=seeds, empty=family.empty)


@pytest.mark.parametrize("family", [name for name in TEMPLATE_FAMILIES if _drawn_first(name)[3] is None])
def test_the_default_form_under_dropout_is_finite_where_the_truth_is_at_fp32s_edges(family):
    """The plain-JAX mainloop, and on an H100 the Mosaic one in both schedules, under dropout (`_dropping`)
    over every family that starts from no state: output and gradients finite wherever the float64 truth's
    (the reference's own dropping call, its draw the same) rounded to their dtype are. A kept key's
    multiplier taken per weight and on the walk's sums before the division passed fp32's largest where the
    output is finite, and on D's sums before their product with dO overflowed dQ (row 8652)."""
    seeds, family_spec = min(SEEDS[family], 16), FAMILIES[family]
    draw = _cast(family_spec, family_spec.draw)
    truth = _cached_dropping_truth(_spans_key(family_spec.spans))
    for name, mainloop in _mainloops(family_spec, {}, dropping=True).items():
        assert_finite(mainloop, truth, draw, f"{family}: {name} dropping {family_spec.dtype}", seeds=seeds,
                      empty=family_spec.empty)


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_the_default_form_is_finite_where_the_truth_is_at_fp32s_edges(family):
    """The plain-JAX mainloop, its rows' sums scaled where they could pass fp32's largest
    (`algebra.SCALED_WALK`) and under the fold's guard (`algebra.walked`), finite in the default form over
    every family (`default_is_finite`)."""
    default_is_finite(family, "portable")


@pytest.mark.skipif(not (INTERPRETED or ON_HOPPER),
                    reason="runs interpreted as Hopper on CPU or compiled on Hopper (row 6881)")
@pytest.mark.parametrize("family", ["values at fp32's largest over 2 equal keys",
                                    "values at fp32's largest over 256 equal keys"])
def test_the_mosaic_forward_of_values_at_fp32s_largest_is_as_exact_as_the_reference_interpreted(family):
    """The Mosaic forward in fp32, routed through `attention_route.attend` (interpreted on a CPU as an H100
    runs it, or compiled natively on Hopper; rows 6842, 6869, 6881), over values at fp32's limit: check 1
    over the family's draws, where it summed its operands' bf16 parts and its products' pairs largest first
    and passed fp32's largest before the smaller terms took it back (inf over 2 keys already)."""
    seeds, family = SEEDS[family], FAMILIES[family]

    def forward(query, key, value, extra):
        spans = _spans(family, query, key)
        # An fp32 call's Mosaic parts are the default precision's alone (`MosaicForward.refusal`); the suite
        # runs at HIGHEST.
        if INTERPRETED:
            with generation.assumed("sm90"), jax.default_matmul_precision("default"):
                return attention_mosaic.forward(query, key, value, spans, None, interpret=True)
        with generation.chosen(attention_mainloop="mosaic"), jax.default_matmul_precision("default"):
            return attention_route.attend(query, key, value, spans, None)

    with generation.routes() as routes:
        assert_as_exact(jax.jit(forward), as_given(jax.jit(_reference(family, anchored=False))),
                        lambda seed: _cast(family, family.draw)(seed)[:4], "mosaic forward at fp32's limit",
                        seeds=seeds, truth_fn=_reference(family, anchored=True))
    if not INTERPRETED:
        assert any(found.choice == "attention_mainloop" and found.kernel == "mosaic"
                   for found in routes), routes


MAINLOOPS = ("portable", "triton", "decode_kernel", *(("mosaic", "decode") if INTERPRETED else ()))
"""The forwards a mean's test walks: the plain-JAX mainloop's, the Pallas-Triton one's and the Triton decode
kernel's, and in a CPU process the Mosaic forward's and its decode step's interpreted as an H100 runs them (a
GPU runs them in `_mainloops`' Mosaic arms and the decode cells)."""


@pytest.mark.parametrize("sink", [None, -20.0], ids=["no sink", "a sink"])
@pytest.mark.parametrize(("mainloop", "training"),
                         [(mainloop, training) for mainloop in MAINLOOPS for training in (False, True)
                          if mainloop not in ("decode", "decode_kernel") or not training],
                         ids=lambda value: {False: "inference", True: "training"}.get(value, value))
def test_an_fp16_mean_of_values_at_fp16s_largest_is_finite(mainloop, training, sink):
    """One fp16 query over a key at logit 0 and 1000 at a logit whose weight, 0.50035, narrows up to 0.5005
    for its product, every value fp16's largest, beside a `sink` or none: the output, their mean (and the
    sink's zero), is 65504, on par with the reference (check 1) on both walks, where the narrowed weights
    over their unrounded sum put it at 65522, inf in fp16 (row 9519; beside a sink of -20, the base review of
    545370f65, F2; through Triton and the decode step, of c55dff91f, F1: each dropped a sink's hold). The
    decode step's call is 17 such rows over 8 key heads, 136 programs past an H100's 132, so its work list
    takes one slice a row and its walk finishes each row from the packed state itself (of 12e76ca7e, F3).
    Each key's logit is two fp16 entries of it, their sum in fp32 log(0.50035) at the call's scale, 1/8."""
    keys, width = 1001, 64
    rows, heads = (17, 8) if mainloop == "decode" else (1, 1)
    logit = np.float32(8 * np.log(0.50035))
    high = np.float16(logit)
    query = jnp.zeros((rows, 1, heads, width), jnp.float16).at[..., :2].set(1)
    key = (jnp.zeros((rows, keys, heads, width), jnp.float16).at[:, 1:, :, 0].set(high)
           .at[:, 1:, :, 1].set(np.float16(logit - np.float32(high))))
    value = jnp.full((rows, keys, heads, width), jnp.finfo(jnp.float16).max, jnp.float16)
    spans = template.spans(1, keys, rows)
    hooks = template.Hooks(initial=None if sink is None
                           else template.sinks(jnp.full((heads,), sink, jnp.float32), rows, 1, width))

    def ours(query, key, value):
        kept = template.Kept.SPLIT if training else template.Kept.NONE
        if mainloop == "portable":
            out = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                                  hooks, residuals=training)
        elif mainloop == "triton":
            with tuning.pinned(PINS):
                out = attention_triton.forward(query, key, value, spans, hooks, residuals=kept,
                                               interpret=INTERPRETED)
        elif mainloop == "mosaic":
            with generation.assumed("sm90"):
                out = attention_mosaic.forward(query, key, value, spans, hooks, interpret=True,
                                               residuals=kept)
        elif mainloop == "decode_kernel":
            sunk = None if sink is None else jnp.full((heads,), sink, jnp.float32)
            out = _decode_attend(query, key, value, [keys], 1, sinks=sunk)
        else:
            with generation.assumed("sm90"):
                out = attention_decode.attend(query, key, value, spans, hooks, interpret=True)
        return out[0] if training else out

    assert_as_exact(ours, lambda query, key, value: template.reference(query, key, value, spans, hooks),
                    lambda seed: (query, key, value), "an fp16 mean at fp16's largest", seeds=FEWEST_SEEDS)


def test_a_mean_of_an_infinite_value_stays_infinite_through_the_hold():
    """One fp16 query over one key at logit 0 whose value is inf, and over two keys whose values are inf and
    -inf: the reference's mean held at fp16's largest from finite sums (`attention.held_mean`) leaves the
    infinite mean inf and the opposite infinities' NaN, as their mean is, where the held value less the
    unheld one took inf to NaN (the head review of 12e76ca7e, F2)."""
    query = jnp.zeros((1, 1, 1, 8), jnp.float16)
    one, two = template.spans(1, 1, 1), template.spans(1, 2, 1)
    inf = jnp.full((1, 1, 1, 8), jnp.inf, jnp.float16)
    alone = template.reference(query, jnp.zeros((1, 1, 1, 8), jnp.float16), inf, one)
    opposite = template.reference(query, jnp.zeros((1, 2, 1, 8), jnp.float16),
                                  jnp.concatenate([inf, -inf], axis=1), two)
    assert bool(jnp.all(alone == jnp.inf)) and bool(jnp.all(jnp.isnan(opposite)))


NARROWING_TIES = {
    "float16": [65520.0, 1.5 * 2.0 ** -14 + 2.0 ** -25, 3 * 2.0 ** -25, 2.0 ** -25, 1000.25, -65520.0],
    "bfloat16": [(2 - 2.0 ** -8) * 2.0 ** 127, 1 + 2.0 ** -8, -(1 + 3 * 2.0 ** -8) * 2.0 ** 100,
                 (1 + 2.0 ** -8) * 2.0 ** -126],
}
"""Points exactly between two neighbouring values of each 16-bit dtype, which a narrowing ties to even: fp16's
overflow midpoint, ties among its subnormals and the least (half its least subnormal), a normal one and a
negative one; bf16's overflow midpoint, normal ties of either sign and one at its least normal."""


@pytest.mark.parametrize("dtype", NARROWING_TIES, ids=NARROWING_TIES.keys())
def test_a_quotient_narrows_to_16_bits_as_its_exact_value_does(dtype):
    """Sums over a divisor whose fp32 quotient rounds onto a 16-bit tie (`NARROWING_TIES`), and the sums an
    fp32 ulp either side of the tie's product with the divisor and on it where exact: the settled quotient
    (`algebra.narrowing_settled`) narrows to `dtype` as the exact quotient rounds once, to the tie's nearer
    neighbour where it is not the tie and to the even one where it is, where the fp32 quotient's second
    rounding took a quotient below fp16's overflow midpoint to inf (row 9544), and so at the overflow
    midpoint does a register epilogue's settle of it alone (`algebra.midpoint_settled`); a held quotient,
    past the one it stands for, as it is. Each divisor a drawn fp32 value from 2^-6 to 2^6, the expectation
    the float64 quotient's side of the tie (two fp32 values' quotient lies on a tie in float64 only where it
    is one)."""
    narrow = jnp.dtype(dtype)
    rng = np.random.default_rng([9544, len(dtype)])
    ties = np.asarray(NARROWING_TIES[dtype], np.float64)
    # Drawn divisors and one just below 1, whose tie's product rounds so the quotient lies off it.
    divisors = np.append(np.float32(np.ldexp(1 + rng.random(64), rng.integers(-6, 7, 64))),
                         np.float32(1 - 2.0 ** -24))
    tie, divisor = (np.ravel(x) for x in np.meshgrid(ties, divisors, indexing="ij"))
    divisor = divisor.astype(np.float32)
    centre = np.float32(tie * divisor)
    acc = np.concatenate([np.nextafter(centre, -np.inf), centre, np.nextafter(centre, np.inf)])
    divisor, tie, centre = (np.tile(x, 3) for x in (divisor, tie, centre))
    # A tie's product with a divisor past fp32's largest is no sum a quotient of the tie has, and one below
    # fp32's least normal a sum XLA's CPU flushes (row 4023).
    finite = np.isfinite(centre) & np.isfinite(acc) & (np.abs(acc) >= np.finfo(np.float32).tiny)
    acc, divisor, tie = acc[finite], divisor[finite], tie[finite]
    rounded = np.asarray(jnp.asarray(acc) / jnp.asarray(divisor))
    settled = algebra.narrowing_settled(jnp.asarray(rounded), jnp.asarray(acc), jnp.asarray(divisor), narrow)
    found = np.asarray(settled.astype(narrow).astype(jnp.float32), np.float64)
    exact = acc.astype(np.float64) / divisor.astype(np.float64)

    def narrowed(x):
        return np.asarray(jnp.asarray(np.float32(x)).astype(narrow).astype(jnp.float32), np.float64)

    # The tie's neighbours, an fp32 value either side of it narrowed once; the even one, the tie narrowed.
    nearer, further = (narrowed(np.nextafter(np.float32(tie), np.float32(np.sign(tie) * side)))
                       for side in (0.0, np.inf))
    expected = np.where(exact == tie, narrowed(tie), np.where(np.abs(exact) < np.abs(tie), nearer, further))
    np.testing.assert_array_equal(found, expected)
    largest = jnp.finfo(narrow).max
    midpoint = np.abs(tie) > largest
    alone = algebra.midpoint_settled(jnp.asarray(rounded), jnp.asarray(acc), jnp.asarray(divisor), narrow)
    found = np.asarray(alone.astype(narrow).astype(jnp.float32), np.float64)
    np.testing.assert_array_equal(found[midpoint], expected[midpoint])
    held = algebra.narrowing_settled(jnp.float32(jnp.finfo(narrow).max), jnp.float32(65522.0),
                                     jnp.float32(1.0), narrow)
    assert held == jnp.finfo(narrow).max
    if narrow == jnp.bfloat16:
        settle_jit = jax.jit(lambda q, a, d: algebra.narrowing_settled(q, a, d, jnp.bfloat16))
        # Subnormal accumulator at exact BF16 tie (`divisor=2^-126`, `q=511/512`, `acc=511*2^-135`):
        # bit-based scaling keeps the remainder zero (even tie -> 1.0) rather than flushing `acc` to 0.
        sub_div = jnp.asarray([2.0 ** -126] * 3, jnp.float32)
        sub_q = jnp.asarray([511.0 / 512.0] * 3, jnp.float32)
        exact_sub_acc = np.float32(511.0 * 2.0 ** -135)
        sub_acc = jnp.asarray([np.nextafter(exact_sub_acc, np.float32(0.0)),
                               exact_sub_acc,
                               np.nextafter(exact_sub_acc, np.float32(np.inf))], jnp.float32)
        sub_out = np.asarray(settle_jit(sub_q, sub_acc, sub_div).astype(jnp.bfloat16), np.float32)
        np.testing.assert_array_equal(sub_out, np.asarray([255.0 / 256.0, 1.0, 1.0], np.float32))
        # Recovered subnormal BF16 half-least-subnormal tie (`q=2^-134`, `divisor=1.0`): exact tie rounds
        # even to 0, while 1 fp32 ulp above nudges up to `2^-133`.
        tiny_tie = np.float32(2.0 ** -134)
        tiny_q = jnp.asarray([tiny_tie] * 3, jnp.float32)
        tiny_div = jnp.ones((3,), jnp.float32)
        tiny_acc = jnp.asarray([np.nextafter(tiny_tie, np.float32(0.0)),
                                tiny_tie,
                                np.nextafter(tiny_tie, np.float32(np.inf))], jnp.float32)
        tiny_out = np.asarray(settle_jit(tiny_q, tiny_acc, tiny_div).astype(jnp.bfloat16), np.float32)
        np.testing.assert_array_equal(tiny_out, np.asarray([0.0, 0.0, 2.0 ** -133], np.float32))

@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_a_quantized_calls_codes_past_fp16s_largest_are_no_mean(training):
    """One fp16 key at logit 0 and value 1024 under a quantized model's weights (`Hooks.rounds` 127.5, the
    key's code 127) and an output hook taking the output by 1/128: the output is 127 x 1024 / 128, 1016, on
    par with the reference (check 1), the codes' product no mean of the values, so none held at fp16's
    largest (the base review of 545370f65, F1: held, 511.75)."""
    query = jnp.zeros((1, 1, 1, 64), jnp.float16)
    key = jnp.zeros((1, 1, 1, 64), jnp.float16)
    value = jnp.full((1, 1, 1, 64), 1024, jnp.float16)
    spans = template.spans(1, 1, 1)
    hooks = template.Hooks(output=Hook(lambda out: out / 128), rounds=127.5)

    def ours(query, key, value):
        out = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                              hooks, residuals=training)
        return out[0] if training else out

    assert_as_exact(ours, lambda query, key, value: template.reference(query, key, value, spans, hooks),
                    lambda seed: (query, key, value), "a quantized call's codes", seeds=FEWEST_SEEDS)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_an_unseen_keys_value_past_fp32s_largest_takes_no_part_in_a_gradient(dtype):
    """A row's unseen keys whose values pass fp32's largest in their first column under a cotangent on every
    column (`_masked_outlier`'s `every_column`): each template mainloop's gradients in both forms finite
    wherever the float64 truth's are (`gradient.where_seen`), where jax's attention, FlashAttention's
    backward and Dew's reference path, its formula as jax's, form `0 · inf` (row 5278)."""
    name = f"small values beside unseen ones near fp32's largest, {dtype}"
    family = dataclasses.replace(FAMILIES[name], draw=functools.partial(_masked_outlier, dtype=dtype,
                                                                        every_column=True))
    draw = _cast(family, family.draw)
    truth = _cached_reference_fn(*_reference_key(family), anchored=True)
    arms = {f"{mainloop} {form}": fn for form, options in (("default", {}), ("accurate", ACCURATE))
            for mainloop, fn in _mainloops(family, options).items()}
    for label, fn in arms.items():
        assert_finite(fn, truth, draw, f"{label} {dtype}", seeds=SEEDS[name])

BAND_FAMILIES = ("a share in bf16's subnormal band on a value near its largest",
                 "a share in bf16's subnormal band, the denominator's rest in a later tile")
"""The families of a share in bf16's subnormal band, which each reference's form weighs differently."""


def _normalized_first(family: Family) -> Callable:
    """A reference that rounds its logits to the call's dtype and normalizes its weights before narrowing
    them for their product with the values (torch's scaled dot-product attention's math path: its product
    of the query and keys in the inputs' dtype, the softmax over it, then the weights narrowed)."""
    def attend(query, key, value, extra):
        del extra
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
        logits = (logits.astype(query.dtype).astype(jnp.float32) * query.shape[-1] ** -0.5)
        weights = jax.nn.softmax(jnp.where(jnp.asarray(family.keep), logits, -jnp.inf), axis=-1)
        out = jnp.einsum("bhqk,bkhd->bqhd", weights.astype(value.dtype), value,
                         preferred_element_type=jnp.float32)
        return out.astype(query.dtype)

    return attend


def _unnormalized(family: Family) -> Callable:
    """A reference that narrows its weights at the row's maximum unnormalized for their product with the
    values and divides the sums by the fp32 weights' sum after it (FlashAttention's: its `P` in the inputs'
    dtype, its row sums in fp32), so a small share keeps the bits its denominator would take."""
    def attend(query, key, value, extra):
        del extra
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key,
                            preferred_element_type=jnp.float32) * query.shape[-1] ** -0.5
        logits = jnp.where(jnp.asarray(family.keep), logits, -jnp.inf)
        weights = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
        sums = jnp.einsum("bhqk,bkhd->bqhd", weights.astype(value.dtype), value,
                          preferred_element_type=jnp.float32)
        return (sums / jnp.swapaxes(weights.sum(axis=-1), 1, 2)[..., None]).astype(query.dtype)

    return attend


REFERENCE_FORMS = {"jax's attention": lambda family: _reference(family, anchored=False),
                   "normalized first": _normalized_first, "unnormalized": _unnormalized}
"""The references' forms a share in bf16's subnormal band is held to (`BAND_FAMILIES`)."""


@pytest.mark.parametrize("form", REFERENCE_FORMS)
@pytest.mark.parametrize("family", BAND_FAMILIES)
def test_a_share_in_bf16s_subnormal_band_is_on_par_with_every_references_form(family, form):
    """The plain-JAX mainloop's output and gradients over a share in bf16's subnormal band on a value near
    its largest (`_subnormal_band`), check 1 against each reference form a site of Dew's names
    (`REFERENCE_FORMS`): jax's attention, which normalizes its weights before narrowing them, torch's math
    path, which rounds its logits to the inputs' dtype first, and FlashAttention's, which narrows them
    unnormalized and keeps the bits a share's denominator would take. The scaled walk takes the share's
    weight, below fp32's least normal at its row's scale, in its small weights' product, raised
    (`algebra.at_scale`)."""
    seeds, cases = SEEDS[family], FAMILIES[family]
    draw = _cast(cases, cases.draw)
    # Every row of the family is scaled (its training residuals' shift past 0), so the share takes the small
    # weights' product.
    query, key, value, _, _ = draw(0)
    *_, split = attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                                query, key, value, _spans(cases, query, key), residuals=True)
    assert np.all(np.asarray(split.shift) > 0)
    reference = as_given(jax.jit(with_gradients(REFERENCE_FORMS[form](cases))))
    truth = _cached_reference_fn(*_reference_key(cases), anchored=True)
    portable = _mainloops(cases, {})["portable"]
    assert_as_exact(portable, reference, draw, f"portable against {form}", seeds=seeds, truth_fn=truth)

@functools.cache
def mosaic_families() -> tuple[str, ...]:
    """The template's families the Mosaic mainloop takes (`_mosaic_takes`), read where a GPU's run builds its
    cells (tests/test_gpu_attention.py), so no CPU collection draws them."""
    return tuple(family for family in TEMPLATE_FAMILIES
                 if _mosaic_takes(FAMILIES[family], _drawn_first(family)))


@functools.cache
def hookless_families() -> tuple[str, ...]:
    """The Mosaic families of no sinks or resumed state (`mosaic_families`), whose fast-form backward derives
    one launch (`MosaicBackward.derived`)."""
    return tuple(family for family in mosaic_families() if _drawn_first(family)[3] is None)


GPU_TRITON_FAMILIES = ("a small weight on a large value, bfloat16",
                       "a small weight on a large value, float32",
                       "a small column beside an overflowing one",
                       "a cotangent on a small column beside an overflowing one",
                       "a column near fp32's least normal beside an overflowing one",
                       "a resumed state beside an overflowing column",
                       "a resumed state of a large denominator")
"""The families whose Pallas-Triton default-form check runs on a GPU alone, compiled as its route runs it
(tests/test_gpu_attention.py): their draws interpreted take past the CPU suite's 120 s target (123 to 257
s, rows 1849, 2193, 4344, 5533; `GPU_ONLY`)."""


@pytest.mark.parametrize("family", [family for family in TEMPLATE_FAMILIES
                                    if family not in GPU_TRITON_FAMILIES])
def test_the_frozen_triton_mainloops_default_is_finite_where_the_truth_is_at_fp32s_edges(family):
    """The Triton mainloop's default form finite over every family (`default_is_finite`): its rows' sums
    scaled where they could pass fp32's largest (`algebra.SCALED_WALK`) and its launches under the fold's
    guard as the Mosaic mainloop's are (`algebra.walked_launches`), as the template's production route runs
    it on an sm80 or later GPU (ledger 1637)."""
    default_is_finite(family, "triton")


@pytest.mark.parametrize("family", [family for family in TEMPLATE_FAMILIES
                                    if FAMILIES[family].overflows and family not in GPU_ONLY["frozen rows"]])
def test_the_frozen_triton_backward_reads_a_scaled_forwards_rows_whatever_its_pass(family):
    """The Triton mainloop's default backward asked for its unsplit first pass (`split=False`) over sums that
    pass fp32's largest, finite wherever the rounded truth is: the forward's rows hold their sums at their
    scale (`algebra.SCALED_WALK`, `Split.shift`), and the backward takes D from them, the split pass,
    wherever the values can be scaled (`attention_triton.backward`), in either form."""
    seeds, family_of = SEEDS[family], FAMILIES[family]

    def forward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.forward(*operands, interpret=INTERPRETED)

    def backward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.backward(*operands, interpret=INTERPRETED, split=False)

    unsplit = jax.jit(with_gradients(_differentiable(family_of, forward, backward)))
    assert_finite(unsplit, _cached_reference_fn(*_reference_key(family_of), anchored=True),
                  _cast(family_of, family_of.draw), f"triton unsplit {family_of.dtype}", seeds=seeds)

def test_a_resumed_state_of_large_values_under_the_folds_guard_is_finite():
    """fp32, the default form: one query over one key of logit m0 = 24576.005859375 (folded below the fold's
    reach, `algebra.FOLD_REACH`, and naturalized back below m0), resumed from a state of maximum m0,
    denominator 1 and values 1023 * 2^118, the key's value the same: their weighted sum passes fp32's
    largest, which the walk's sums, at the row's scale (`algebra.SCALED_WALK`), never hold. The plain-JAX
    mainloop's output and gradients are finite, as the reference's are (`_resumed_attention`)."""
    m0, size = 24576.005859375, 1023.0 * 2.0 ** 118

    def first(fill, *shape):
        return jnp.zeros(shape, jnp.float32).at[..., 0].set(fill)

    query, key, value, dout = (first(fill, 1, 1, 1, 16) for fill in (4.0, m0, size, 2.0 ** -8))
    state = (jnp.full((1, 1, 1), m0, jnp.float32), jnp.ones((1, 1, 1), jnp.float32), first(size, 1, 1, 1, 16))
    family = Family(lambda seed: (query, key, value, state, dout), np.ones((1, 1, 1, 1), bool))
    reference = with_gradients(lambda *args: _resumed_attention(*args, family.keep, anchored=False))
    ours = _mainloops(family, {})["portable"]
    for name, found in (("reference", reference(query, key, value, state, dout)),
                        ("default", ours(query, key, value, state, dout))):
        assert all(np.isfinite(np.asarray(leaf, np.float32)).all() for leaf in jax.tree.leaves(found)), name

@pytest.mark.parametrize("options", [{"scale": 7.99}, {"scale": 7.99, **ACCURATE}],
                         ids=["default", "accurate"])
def test_a_score_cotangent_past_bf16s_largest_keeps_the_gradients_finite(options):
    """bf16, both forms: one zero query over two keys of first components +-1/128 at scale 7.99, values of
    first components +-2^126 and an output cotangent of 1 there, so each key weighs 1/2 and its score
    cotangent times the scale is +-7.99 * 2^125, finite in fp32 and past bf16's largest. Narrowed to bf16 for
    their products it would be infinite, dQ infinite and dK NaN against the zero query (`algebra._bf16`
    saturates it); every template mainloop's gradients are finite and dK exactly the reference's zero."""
    def shaped(seed):
        # The call's shapes, for the mainloops' legality rules (`_mosaic_takes`); the operands are below.
        del seed
        one, two = np.zeros((1, 1, 1, 16), np.float32), np.zeros((1, 2, 1, 16), np.float32)
        return one, two, two, None, one

    family = Family(shaped, np.ones((1, 1, 1, 2), bool), dtype="bfloat16")

    def first(*fills):
        return jnp.zeros((1, len(fills), 1, 16), jnp.bfloat16).at[:, :, 0, 0].set(jnp.asarray(fills))

    query, key, value, dout = first(0.0), first(1 / 128, -1 / 128), first(2.0 ** 126, -2.0 ** 126), first(1.0)
    spans = template.spans(1, 2, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, scale=options["scale"]), query, key,
                      value)
    assert all(np.isfinite(np.asarray(leaf, np.float32)).all() for leaf in pull(dout)), "the reference's"
    for name, mainloop in _mainloops(family, options).items():
        _, *grads = mainloop(query, key, value, None, dout)
        assert all(np.isfinite(np.asarray(leaf, np.float32)).all() for leaf in grads[:3]), name
        np.testing.assert_array_equal(np.asarray(grads[1], np.float32), 0.0, err_msg=f"{name}'s dK")

@pytest.mark.parametrize("scale", [-1.0, -2.0, -0.75])
def test_a_negative_scale_past_the_folds_reach_keeps_a_finite_logit_finite(scale):
    """fp32, the default form, one query over one key at a negative scale of magnitude at least 1/log2(e), the
    product such that the reference's logit is a finite -3e38 (-2.25e38 at -0.75), its one weight 1 and its
    output the value. Folding log2(e) into such a scale takes the logit past fp32's largest to -inf and its
    weight to NaN, so the fold bounds the factor's magnitude (`algebra.Forward.base2`) and such a call stays
    natural. Each template mainloop's output is the value and its gradients are finite, as the
    reference's are."""
    def first(fill, *shape):
        return jnp.zeros(shape, jnp.float32).at[..., 0].set(fill)

    product = 3e38 / abs(scale) if abs(scale) > 1 else 3e38
    query, key, value, dout = (first(fill, 1, 1, 1, 16) for fill in (product, 1.0, 2.0, 1.0))
    # Its one draw the call itself, which the mainloops' legality reads (`_mosaic_takes`).
    family = Family(lambda seed: (query, key, value, None, dout), np.ones((1, 1, 1, 1), bool))
    spans = template.spans(1, 1, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, scale=scale), query, key, value)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in pull(dout)), "the reference's gradients"
    for name, mainloop in _mainloops(family, {"scale": scale}).items():
        out, *grads = mainloop(query, key, value, None, dout)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(value), err_msg=name)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in grads[:3]), f"{name}'s gradients"

SMALL_SCALE = 2.0 ** -80
"""A scale far below one, whose power of two the queries take for their logits (`attention.prescale`)."""
SMALL_SCALE_FAMILY = Family(_drawn(16, 64, rows=8), _kept(16, 64, rows=8))
"""Drawn queries, keys and values at `SMALL_SCALE`, which `test_gradients_at_a_scale_far_below_one_are_on_par`
casts to each dtype."""


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("name", _mainloops(SMALL_SCALE_FAMILY, {}))
@pytest.mark.parametrize("options", [{}, ACCURATE], ids=["default", "accurate"])
def test_gradients_at_a_scale_far_below_one_are_on_par(options, name, dtype):
    """Both forms, fp32 and bf16, at a scale of 2^-80 over drawn calls: every logit is small and each key
    weighs about alike, and the reference's gradients are normal, dK the score cotangents at the whole scale
    times the queries. A template mainloop's output and gradients hold check 1: the queries take the scale's
    power of two for their logits alone, where taken for dK too its products fall 2^80 below the reference's
    and flush to zero (row 1957's review). The truth is the written-out attention in float64 (jax's softmaxes
    its logits in fp32), its queries times the scale over the oracle's 1/sqrt(D), a power of two, exactly."""
    family = dataclasses.replace(SMALL_SCALE_FAMILY, dtype=dtype)

    def attend(query, key, value, extra):
        del extra
        return jax.nn.dot_product_attention(query, key, value, scale=SMALL_SCALE,
                                            implementation="xla").astype(query.dtype)

    def exact(query, key, value, extra):
        del extra
        return exact_attention(query * (query.shape[-1] ** 0.5 * SMALL_SCALE), key, value, anchored=True)

    differentiated = _mainloops(family, {"scale": SMALL_SCALE, **options})[name]
    assert_as_exact(differentiated, as_given(jax.jit(with_gradients(attend))), _cast(family, family.draw),
                    f"{name} {options} {dtype} at scale 2^-80", seeds=8, truth_fn=with_gradients(exact))

DECODED = [(family, form) for family, form in FAMILY_RUNS
           if _drawn_first(family)[0].shape[1] == 1 and _drawn_first(family)[0].shape[3] >= 16
           and not isinstance(_drawn_first(family)[3], tuple)
           and not (form == "natural" and jnp.dtype(FAMILIES[family].dtype) == jnp.float32)]
"""The families a decode step is: one query a row, a head width the decode kernel takes (16 to 256),
sinks or none (a resumed state is the template's); fp32 operands in the default form alone, which resolves
to natural logits for them (`decode_attention._spec`, ledger 2151), so the resolution a caller gets is the
one checked."""


def _cache(array: jax.Array) -> jax.Array:
    """Keys or values `[B, K, H, D]` in a cache of whole decode blocks (`decode_attention.BLOCK`, the
    capacity its geometry takes), the slots past the keys zeros."""
    slots = -array.shape[1] % decode_attention.BLOCK
    return jnp.pad(array, ((0, 0), (0, slots), (0, 0), (0, 0)))


def _decode_attend(query: jax.Array, key: jax.Array, value: jax.Array, lengths, splits: int,
                   sinks: jax.Array | None = None, **form) -> jax.Array:
    """`decode_attention.attend` over `key` and `value` padded to whole decode blocks (`_cache`), with
    `splits` slices pinned (as an A100's launch on a CPU, where `INTERPRETED` holds)."""
    pin = {decode_attention.DecodeAttention: decode_attention.Split(splits)}
    target = generation.assumed("sm80") if INTERPRETED else contextlib.nullcontext()
    with target, tuning.pinned(pin):
        return decode_attention.attend(query, _cache(key), _cache(value), jnp.asarray(lengths, jnp.int32),
                                       sinks=sinks, interpret=INTERPRETED, **form)


def _decoded(family: Family, natural: bool | None) -> tuple[Callable, Callable]:
    """The decode kernel over `family`'s step, interpreted on a CPU (as an A100's launch) and compiled on a
    GPU, each row's keys the first `K` of its cache, in the form `natural` names (None, the default), the
    family's slice count pinned, and the family's draw without its cotangent (a decode step has no
    backward)."""
    lengths = family.spans.get("lengths")
    form = {} if natural is None else {"natural": natural}

    def decoded(query, key, value, sinks):
        count = np.full(query.shape[0], key.shape[1]) if lengths is None else lengths
        return _decode_attend(query, key, value, count, family.splits, sinks=sinks, **form)

    def forward(seed):
        return _cast(family, family.draw)(seed)[:4]

    return decoded, forward


DECODED_ON_CPU = [case for case in DECODED if case not in GPU_ONLY["decode"]]


@pytest.mark.parametrize("family, form", DECODED_ON_CPU,
                         ids=[f"{family}-{form}" for family, form in DECODED_ON_CPU])
def test_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family, form):
    """The decode kernel's output, interpreted on a CPU and compiled on a GPU (its launch and merge),
    each row's keys the first `K` of its cache, check 1 over the family's draws."""
    seeds, name, family = SEEDS[family], family, FAMILIES[family]
    decoded, forward = _decoded(family, FORMS[form])
    assert_as_exact(jax.jit(decoded), as_given(jax.jit(_reference(family, anchored=False))), forward,
                    f"{name}: decode {form} {family.dtype}", seeds=seeds,
                    truth_fn=_reference(family, anchored=True))


TEMPLATE_DECODED = list(dict.fromkeys(family for family, _ in DECODED
                                     if not FAMILIES[family].decode_only or FAMILIES[family].splits > 1))
"""The decode steps the template's decode takes (`attention_decode.attend`), in its one form, the raw fold
(no fold's reach, as the decode kernel's): 16-bit and fp32 operands (an fp32 step's operands as their bf16
parts, `MosaicForward.parts`); a decode step's alone (`Family.decode_only`) where it is a split one
(`Family.splits`), whose merge the template's plan reaches over its keys, else too long for its interpreted
mainloop."""


@pytest.mark.skipif(not INTERPRETED and here() != "sm90",
                    reason="the template's decode runs on warpgroup MMA's Mosaic mainloop")
@pytest.mark.parametrize("family", TEMPLATE_DECODED)
def test_the_template_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family):
    """The template's decode (`attention_decode.attend`: packed rows, sliced keys, merged), interpreted on a
    CPU as an H100 runs it and compiled on an H100, each row's keys its first `K`: check 1 over the family's
    draws, as the decode kernel it replaces is held (testing.md: decode holds the default's claims)."""
    seeds, name, family = SEEDS[family], family, FAMILIES[family]
    lengths = family.spans.get("lengths")

    def decoded(query, key, value, sinks):
        count = np.full(query.shape[0], key.shape[1]) if lengths is None else lengths
        spans = template.spans(1, key.shape[1], query.shape[0], lengths=jnp.asarray(count, jnp.int32))
        hooks = (None if sinks is None
                 else template.Hooks(initial=template.sinks(sinks, query.shape[0], 1, value.shape[-1])))
        target = generation.assumed("sm90") if INTERPRETED else contextlib.nullcontext()
        # An fp32 step's Mosaic parts are the default precision's alone (`MosaicForward.refusal`).
        with target, jax.default_matmul_precision("default"):
            return attention_decode.attend(query, key, value, spans, hooks, interpret=INTERPRETED)

    def forward(seed):
        return _cast(family, family.draw)(seed)[:4]

    assert_as_exact(jax.jit(decoded), as_given(jax.jit(_reference(family, anchored=False))), forward,
                    f"{name}: template decode {family.dtype}", seeds=seeds,
                    truth_fn=_reference(family, anchored=True))

def test_a_decode_steps_logits_past_the_folds_reach_weigh_as_the_references():
    """bf16, the default form: one query over two keys whose products are 2^30 and 2^30 + 128, an fp32 ulp
    apart (logits 2^28 and 2^28 + 32), their values the first two unit vectors: the reference weighs the
    first exp(-32). A fold that scales each logit before the difference (`algebra.Forward.base2`) rounds it
    at log2(e) times its size, 32 here, and weighed the first exp2(-32), 2.3e-10 against 1.27e-14; the decode
    kernel's raw fold (`algebra.Forward.subtracted`) scales the raw products' difference after it is taken,
    so its output is the float64 truth's in bf16, one slice or two, with no natural rewalk (row 1955)."""
    width = 16

    def first(rows, *fills):
        array = np.zeros((1, rows, 1, width), np.float32)
        for at, fill in fills:
            array[(0, *at)] = fill
        return jnp.asarray(array, jnp.bfloat16)

    query = first(1, ((0, 0, 0), 2.0 ** 15), ((0, 0, 1), 1.0))
    key = first(2, ((0, 0, 0), 2.0 ** 15), ((1, 0, 0), 2.0 ** 15), ((1, 0, 1), 128.0))
    value = first(2, ((0, 0, 0), 1.0), ((1, 0, 1), 1.0))
    logits = np.asarray([2.0 ** 28, 2.0 ** 28 + 32])
    weights = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
    truth = np.zeros(width, np.float64)
    truth[:2] = weights
    for splits in (1, 2):
        target = generation.assumed("sm80") if INTERPRETED else contextlib.nullcontext()
        with target, tuning.pinned({decode_attention.DecodeAttention: decode_attention.Split(splits)}):
            out = decode_attention.attend(query, _cache(key), _cache(value), jnp.asarray([2], jnp.int32),
                                          interpret=INTERPRETED)
        np.testing.assert_array_equal(np.asarray(out[0, 0, 0]), np.asarray(truth.astype(jnp.bfloat16)),
                                      err_msg=f"{splits} slices")


DECODED_OVERFLOWS = list(dict.fromkeys(family for family, _ in DECODED if FAMILIES[family].overflows))
"""The decode steps (`DECODED`) whose one divergence in the default form is their sums passing fp32's
largest."""


@pytest.mark.parametrize("family", DECODED_OVERFLOWS)
def test_decodes_default_is_finite_where_its_sums_pass_fp32s_largest(family):
    """The decode kernel's default form, interpreted on a CPU and compiled on a GPU, finite wherever the
    rounded truth is (`assert_finite`) over the families whose weighted values' sums pass fp32's largest: it
    scales its rows' sums in either form (`algebra.SCALED_WALK`)."""
    seeds, name, family = SEEDS[family], family, FAMILIES[family]
    decoded, forward = _decoded(family, None)
    assert_finite(jax.jit(decoded), _reference(family, anchored=True), forward,
                  f"{name}: decode default {family.dtype}", seeds=seeds)


SUBNORMAL_SHARE_SEEDS = {"float32": 8, "bfloat16": 16}
"""The subnormal share's declared check-1 seeds, sized by `seeds_for` paired on armada."""


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_scaled_rows_split_sums_keep_a_subnormal_share_element_wise(dtype):
    """The subnormal-share family's training forward in both mainloops: every row's sums would pass fp32's
    largest, so its split sums are at its scale (`algebra.SCALED_WALK`, its shift the scale's exponent, the
    anchor its largest logit's key). The share of the key whose normalized weight is below fp32's least
    normal, its weight at the row's scale below it too (taken in the small weights' product, raised,
    `algebra.at_scale`), alone in `B`'s second column, over the denominator at that scale, held element-wise
    to float64 as exactly as the reference's own fp32 arithmetic for it (its weight `e / T` and that times
    the value), each share a row of one value (check 1 over the family's draws): a check of its rows, which a
    leaf's RMS beside the first column's would not see."""
    draw = _cast(Family(_subnormal_share, np.ones((64, 1, 1, 256), bool), dtype=dtype), _subnormal_share)
    spans = template.spans(1, 256, 64)

    def logits_of(query, key, dtype):
        return np.einsum("bqhd,bkhd->bhqk", np.asarray(query, dtype), np.asarray(key, dtype)) / 4.0

    @as_given
    def exact(query, key, value, *_):
        logits = logits_of(query, key, np.float64)
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        share = weights[..., 255] / weights.sum(axis=-1)
        return (share * np.asarray(value, np.float64)[:, 255, 0, 1][:, None, None])[..., None]

    @as_given
    def fp32(query, key, value, *_):
        # The reference's own fp32 arithmetic for the share: its weight `e / T` and that times the value.
        narrow = np.asarray(jnp.einsum("bqhd,bkhd->bhqk", jnp.asarray(query, jnp.float32),
                                       jnp.asarray(key, jnp.float32)) / 4.0)
        exponentials = np.exp(narrow - narrow.max(axis=-1, keepdims=True))
        return (exponentials[..., 255] / exponentials.sum(axis=-1)
                * np.asarray(value, np.float32)[:, 255, 0, 1][:, None, None])[..., None].astype(np.float64)

    def triton(query, key, value, *_):
        with tuning.pinned(PINS):
            return attention_triton.forward(query, key, value, spans, residuals=True, natural=True,
                                            interpret=INTERPRETED)

    def portable(query, key, value, *_):
        return attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                               query, key, value, spans, residuals=True, natural=True)

    for name, attend in (("triton", triton), ("portable", portable)):
        query, key, _, _, _ = draw(0)
        _, _, _, anchors, _ = attend(*draw(0))
        np.testing.assert_array_equal(np.asarray(anchors), logits_of(query, key, np.float64).argmax(axis=-1),
                                      err_msg=name)
        @as_given
        def found(*drawn, attend=attend):
            # `B`'s second column over the denominator at the row's scale, in float64: no rounding of ours.
            _, _, total, _, split = attend(*drawn)
            shift = np.exp2(np.asarray(split.shift, np.float64))
            others = np.asarray(split.others[..., 1], np.float64) * shift
            return (others / np.asarray(total, np.float64))[..., None]

        assert_as_exact(found, fp32, draw, f"{name} {dtype} subnormal share",
                        seeds=SUBNORMAL_SHARE_SEEDS[dtype], truth_fn=exact)


def test_stability_families_with_identical_structure_share_jitted_programs():
    """Row 7274: stability families sharing spans, mask, and dtype reuse one cached `jax.jit` mainloop,
    reference, and CPU `Truth` program instead of re-tracing and re-certifying per family on GPU tiers."""
    buckets: dict[tuple, list[Family]] = {}
    for name in TEMPLATE_FAMILIES:
        fam = FAMILIES[name]
        buckets.setdefault((_spans_key(fam.spans), _reference_key(fam)), []).append(fam)
    shared = [group for group in buckets.values() if len(group) >= 2]
    assert shared and len(buckets) < len(TEMPLATE_FAMILIES), (len(buckets), len(TEMPLATE_FAMILIES))
    first_fam, second_fam = shared[0][0], shared[0][1]
    assert _mainloops(first_fam, {})["portable"] is _mainloops(second_fam, {})["portable"]
    assert _mainloops(first_fam, ACCURATE)["portable"] is _mainloops(second_fam, ACCURATE)["portable"]
    assert _jitted_reference(*_reference_key(first_fam), anchored=False) is _jitted_reference(
        *_reference_key(second_fam), anchored=False)
    truth_a = _cached_reference_fn(*_reference_key(first_fam), anchored=True)
    truth_b = _cached_reference_fn(*_reference_key(second_fam), anchored=True)
    assert truth_a is truth_b
    cpu = jax.local_devices(backend="cpu")[0]
    assert _cached_truth(truth_a, cpu) is _cached_truth(truth_b, cpu)
    # A family with distinct spans (causal/documents/empty) gets a distinct mainloop program.
    spanned = next(FAMILIES[name] for name in TEMPLATE_FAMILIES
                   if _spans_key(FAMILIES[name].spans) != _spans_key(first_fam.spans))
    assert _mainloops(first_fam, {})["portable"] is not _mainloops(spanned, {})["portable"]


@pytest.mark.gpu(reason="the ambient fp16 preset a GPU's product takes")
def test_a_bf16_call_of_wide_logits_scaled_after_keeps_its_bf16_product():
    """The reference path's bf16 call with wide logits scaled after (`wide_logits`, `scale_logits`) under an
    ambient F16_F16_F32 preset: its logits' product takes jax.nn's bf16 preset for its own bf16 operands
    (`attention.unscaled_scores`' `dtype`), so a query and key of 2^20 give finite logits and a finite
    output, where read off the widened operands it took the ambient fp16 product, whose operands overflow."""
    from dew.nn.attention import reference_attention

    query = jnp.zeros((1, 2, 1, 8), jnp.bfloat16).at[..., 0].set(2.0 ** 20)
    value = jnp.ones((1, 2, 1, 8), jnp.bfloat16)
    with jax.default_matmul_precision("F16_F16_F32"):
        out = reference_attention(query, query * 2.0 ** -40, value, dtype=jnp.bfloat16, wide_logits=True,
                                  scale_logits=True)
    assert np.isfinite(np.asarray(out, np.float32)).all()


def test_a_reference_call_weighs_a_hidden_key_nothing_beside_a_visible_one_at_fp32s_least():
    """The reference path's default fill (`mask_fill` None, `excluded`) over a causal row whose one visible
    key's logit is fp32's least (a query of 1 against a key of fp32's least, width 1): that key takes the
    whole weight and the hidden one none, its output the visible key's value, where a hidden key filled at
    fp32's least tied it and took half; a visible key whose score is NaN reads NaN, not an empty row's zero;
    a row whose one visible key's bias is -inf has no mass and reads exactly zero, as the template's guarded
    reduction weighs it; and a row that sees no key (an invalid first key) reads exactly zero, its gradients
    finite, alone and over two query heads sharing a key head in jax.nn's form (its grouped product) (train
    1368's F2)."""
    from dew.nn.attention import scaled_dot_product_attention

    least = float(np.finfo(np.float32).min)
    query = jnp.ones((1, 2, 1, 1), jnp.float32)
    key = jnp.asarray([[[[least]], [[0.0]]]], jnp.float32)
    value = jnp.asarray([[[[3.0]], [[5.0]]]], jnp.float32)
    out = scaled_dot_product_attention(query, key, value, causal=True, implementation="reference")
    np.testing.assert_array_equal(np.asarray(out)[0, 0], [[3.0]])

    unknown = key.at[0, 0].set(np.nan)
    out = scaled_dot_product_attention(query, unknown, value, causal=True, implementation="reference")
    assert np.isnan(np.asarray(out)[0, 0]).all()

    massless = jnp.asarray([[[[-np.inf, 0.0], [0.0, 0.0]]]], jnp.float32)
    out = scaled_dot_product_attention(query, key * 0, value, bias=massless, causal=True,
                                       implementation="reference")
    np.testing.assert_array_equal(np.asarray(out)[0, :, 0], [[0.0], [4.0]])

    valid = jnp.asarray([[False, True]])
    for heads, scale_logits in ((1, False), (2, True)):
        def attended(query, scale_logits=scale_logits):
            return scaled_dot_product_attention(query, key * 0, value, causal=True, key_valid=valid,
                                                implementation="reference", scale_logits=scale_logits)

        out, pull = jax.vjp(attended, jnp.ones((1, 2, heads, 1), jnp.float32))
        np.testing.assert_array_equal(np.asarray(out)[0, 0], [[0.0]] * heads)
        np.testing.assert_array_equal(np.asarray(out)[0, 1], [[5.0]] * heads)
        assert np.isfinite(np.asarray(pull(jnp.ones_like(out))[0])).all()


@pytest.mark.parametrize(("implementation", "wide"), [("reference", False), ("reference", True),
                                                     ("template", False)],
                         ids=["reference", "reference-wide", "template"])
def test_jax_nns_form_keeps_a_scaled_logit_whose_raw_product_overflows(implementation, wide):
    """jax.nn's form (`scale_logits`: the logits' products, then the scale) at width 16 on MSA's routes
    (`sparse_selection.indexed_route`'s preference): a first query of 2^127 against its one visible key of
    -4 has the finite logit -2^127, whose raw product, -2^129, passes fp32's largest; the reference path
    (the query taken by its scale's power of two before its products as the template takes it,
    `attention.range_safe`), its operands widened first too (`wide_logits`), and the template read that
    key's value whole, where the raw product's -inf read the row as empty or NaN; and MSA's teacher forms
    that logit (`block_sparse.msa_scores`, train 1373's F2)."""
    from dew.nn import block_sparse
    from dew.nn.attention import scaled_dot_product_attention

    width = 16
    query = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(-4.0)
    value = jnp.asarray([[[[3.0] * width], [[5.0] * width]]], jnp.float32)
    out = scaled_dot_product_attention(query, key, value, causal=True, implementation=implementation,
                                       scale_logits=True, wide_logits=wide)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], 3.0)
    logits = block_sparse.msa_scores(jnp.float32, None, wide_logits=wide)(
        query[:, :, 0], key[:, :, 0], jnp.ones((1, query.shape[1], key.shape[1]), bool))
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], -(2.0 ** 127))


@pytest.mark.parametrize("row", [None, 0, 1], ids=["alone", "beside-a-large-row", "in-a-large-row"])
@pytest.mark.parametrize("implementation", ["reference", "xla"])
def test_jax_nns_form_keeps_a_query_cotangent_the_scale_keeps_finite(implementation, row):
    """jax.nn's form's derivative at width 16: a second query of 0 over keys of 2^127 and 0 (their first
    entries) and values of 8 and 0, a unit cotangent on its output's first entry: its weights a half each,
    their logits' cotangents 2 and -2, its query's cotangent `2 x 2^127 / 4 = 2^126`, the scale taken on the
    cotangent before its product with the keys, finite, as jax.nn's own derivative takes it (the reference
    path's a declared tangent, `attention.range_safe`), its query alone or an 8 where the keys hold 0 in its
    own row or the first's, where the query's power taken after the product overflowed it."""
    from dew.nn.attention import scaled_dot_product_attention

    width = 16
    key = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 127)
    value = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(8.0)
    cotangent = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 1, 0, 0].set(1.0)
    query = jnp.zeros((1, 2, 1, width), jnp.float32)
    if row is not None:
        query = query.at[0, row, 0, 1].set(8.0)

    def attended(query):
        return scaled_dot_product_attention(query, key, value, causal=True, implementation=implementation,
                                            scale_logits=True)

    _, pull = jax.vjp(attended, query)
    (found,) = pull(cotangent)
    expected = np.zeros((1, 2, 1, width), np.float32)
    expected[0, 1, 0, 0] = 2.0 ** 126
    np.testing.assert_array_equal(np.asarray(found), expected)


def test_jax_nns_form_takes_each_head_alone_beside_a_large_one():
    """Two bf16 query heads over one key head at width 16 in jax.nn's form: head 0's fifteen entries of
    `(1 + 2^-7) 2^-126` against a key's 2^126 in those entries, beside head 1's 8 in the last entry: head 0's
    logits in the teacher's form (`attention.reference_scores`) and its attention's output (the reference
    path's) are the call of head 0 alone's to the bit, one policy a call's every head and route takes
    (`attention.range_safe`, the template's), whatever its neighbour holds."""
    from dew.nn.attention import reference_scores, scaled_dot_product_attention

    width = 16
    tiny = (1.0 + 2.0 ** -7) * 2.0 ** -126
    query = jnp.zeros((1, 1, 2, width), jnp.bfloat16).at[0, 0, 0, :15].set(tiny).at[0, 0, 1, 15].set(8.0)
    key = jnp.zeros((1, 2, 1, width), jnp.bfloat16).at[0, 0, 0, :15].set(2.0 ** 126)
    value = jnp.asarray([[[[3.0] * width], [[5.0] * width]]], jnp.bfloat16)

    def teacher(query):
        heads = query.shape[2]
        return reference_scores(query, jnp.repeat(key, heads, axis=2), '...qhd,...khd->...hqk',
                                scale_logits=True)

    np.testing.assert_array_equal(np.asarray(teacher(query))[:, :1], np.asarray(teacher(query[:, :, :1])))
    both = scaled_dot_product_attention(query, key, value, implementation="reference", scale_logits=True)
    alone = scaled_dot_product_attention(query[:, :, :1], key, value, implementation="reference",
                                         scale_logits=True)
    np.testing.assert_array_equal(np.asarray(both[:, :, :1], np.float32), np.asarray(alone, np.float32))


def test_jax_nns_form_keeps_a_query_tangent_the_scale_keeps_finite():
    """jax.nn's form's forward-mode derivative at width 16 (`attention.range_safe`): a query of 2^127 against
    a key of -4 in one entry, moving along itself with the key held, has the logit -2^127 and the tangent
    -2^127, its direction taken by the scale's power of two before its product, where a tangent of the raw
    products first, -2^129, read -inf."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-4.0)

    def scores(query):
        return reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)

    value, tangent = jax.jvp(scores, (query,), (query,))
    np.testing.assert_array_equal(np.asarray(value)[0, 0, 0], -(2.0 ** 127))
    np.testing.assert_array_equal(np.asarray(tangent)[0, 0, 0], -(2.0 ** 127))


def test_jax_nns_form_differentiates_its_scaled_pullback_at_every_order():
    """jax.nn's form's mixed derivative at width 16 (`attention.range_safe`): zero queries and keys, a score
    cotangent of 2, the query's cotangent `2 k / 4` moved along a key of `2^127` in one entry reads `2^126`,
    every order differentiating the pullback that takes the scale before its products (`traced`), where the
    primal's pullback formed `2^128` before the query's quarter."""
    from dew.nn.attention import reference_scores

    width = 16
    zeros = jnp.zeros((1, 1, width), jnp.float32)

    def query_cotangent(key):
        def scores(query):
            return reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)

        _, pull = jax.vjp(scores, zeros)
        return pull(jnp.full((1, 1, 1), 2.0, jnp.float32))[0]

    _, moved = jax.jvp(query_cotangent, (zeros,), (zeros.at[0, 0, 0].set(2.0 ** 127),))
    np.testing.assert_array_equal(np.asarray(moved)[0, 0, 0], 2.0 ** 126)


def test_jax_nns_form_keeps_a_small_querys_bits_against_a_large_key():
    """jax.nn's form at width 16 in bf16 on the reference path: a query entry of `(1 + 2^-7) 2^-126` against a
    key's `2^126` has the logit `(1 + 2^-7) / 4`, which no product of them can overflow, so its row is
    jax.nn's own arithmetic (`attention.range_safe`), its low bit kept, where its query taken by the scale's
    power of two first, `2^-128` in bf16, lost it."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set((1.0 + 2.0 ** -7) * 2.0 ** -126)
    key = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set(2.0 ** 126)
    logits = reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], np.float32((1.0 + 2.0 ** -7) / 4))


@pytest.mark.parametrize("hidden", ["nan", "infinite column"])
def test_a_hidden_non_finite_key_leaves_a_visible_logits_range_guard_on(hidden):
    """jax.nn's form at width 16 on the reference path, causal: a first query of 2^127 against its one
    visible key of -4 (the finite logit -2^127) beside a future key of NaNs, or of zeros and `+inf` in the
    last entry, where the query's 0 meets it, finite values: the row reads its visible key's value whole,
    each logit its raw product's or, where that is not finite, the prescaled one's (`attention.range_safe`),
    where a bound of the row's magnitudes read a NaN maximum or `0 · inf` and left the visible product
    overflowing; a visible NaN score is the product's own and propagates."""
    from dew.nn.attention import scaled_dot_product_attention

    width = 16
    future = (jnp.full((width,), np.nan, jnp.float32) if hidden == "nan"
              else jnp.zeros((width,), jnp.float32).at[-1].set(np.inf))
    query = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 2, 1, width), jnp.float32).at[0, 0, 0, 0].set(-4.0).at[0, 1, 0].set(future)
    value = jnp.asarray([[[[3.0] * width], [[5.0] * width]]], jnp.float32)
    out = scaled_dot_product_attention(query, key, value, causal=True, implementation="reference",
                                       scale_logits=True)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0], 3.0)


def test_jax_nns_range_guard_keeps_fp32_under_a_narrowing_ambient_preset():
    """jax.nn's form at width 16 in fp32 with `precision='highest'` on a CPU under the ambient preset
    `F16_F16_F32`: Q 2^127 against K -4 reads the logit -2^127, its range guard reading its products alone
    (`attention.range_safe`), where a bound's dot took the ambient float16 algorithm, which XLA's CPU
    compiler has no kernel for."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-4.0)
    with jax.default_matmul_precision("F16_F16_F32"):
        logits = reference_scores(query, key, "bqd,bkd->bqk", precision=jax.lax.Precision.HIGHEST,
                                  scale_logits=True)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], -(2.0 ** 127))


def test_a_small_querys_bits_survive_a_large_entry_meeting_a_small_key_column():
    """jax.nn's form at width 16 in bf16 on the reference path: a query of `(1 + 2^-7) 2^-126` in its first
    entry and 8 in its last, against a key of `2^126` in its first and 0 in its last: its logit
    `(1 + 2^-7) / 4`, its raw product finite, so jax.nn's own (`attention.range_safe`), where a bound of
    unrelated entries' maxima took its query by a quarter first and lost its low bit."""
    from dew.nn.attention import reference_scores

    width = 16
    tiny = (1.0 + 2.0 ** -7) * 2.0 ** -126
    query = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set(tiny).at[0, 0, -1].set(8.0)
    key = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set(2.0 ** 126)
    logits = reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], np.float32((1.0 + 2.0 ** -7) / 4))


def test_a_small_querys_bits_survive_another_key_heads_large_column():
    """jax.nn's form at width 16 in bf16 over two heads as the reference path pairs them
    (`'...qhd,...khd->...hqk'`): head 0's query `(1 + 2^-7) 2^-126` in its first entry and 8 in its last
    against its key's `2^126` and 0, head 1's key `2^126` in its last entry: head 0's logit
    `(1 + 2^-7) / 4`, its raw product finite (`attention.range_safe`), where a bound over every head's
    keys took head 1's last column and prescaled head 0, losing its low bit."""
    from dew.nn.attention import reference_scores

    width = 16
    tiny = (1.0 + 2.0 ** -7) * 2.0 ** -126
    query = jnp.zeros((1, 1, 2, width), jnp.bfloat16).at[0, 0, 0, 0].set(tiny).at[0, 0, 0, -1].set(8.0)
    key = jnp.zeros((1, 1, 2, width), jnp.bfloat16).at[0, 0, 0, 0].set(2.0 ** 126).at[0, 0, 1, -1].set(
        2.0 ** 126)
    logits = reference_scores(query, key, "...qhd,...khd->...hqk", scale_logits=True)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0, 0], np.float32((1.0 + 2.0 ** -7) / 4))


def test_a_small_logit_keeps_its_bits_beside_a_large_finite_one():
    """jax.nn's form at width 16 in bf16: a query of `(1 + 2^-7) 2^-126` and `2^127` in its first two
    entries against keys `[2^126, 0]` and `[0, 1.5]`: both raw products finite, so its logits
    `(1 + 2^-7) / 4` and `1.5 · 2^125` are jax.nn's own (`attention.range_safe`), where a row's bound of
    three quarters of fp32's largest prescaled the row and lost the first's low bit (train 1368's
    review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set((1.0 + 2.0 ** -7) * 2.0 ** -126).at[
        0, 0, 1].set(2.0 ** 127)
    key = jnp.zeros((1, 2, width), jnp.bfloat16).at[0, 0, 0].set(2.0 ** 126).at[0, 1, 1].set(1.5)
    logits = reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0], np.float32([(1.0 + 2.0 ** -7) / 4,
                                                                        1.5 * 2.0 ** 125]))


def _fp16_narrowed(query, key, subscripts: str = "bqd,bkd->bqk"):
    """`subscripts`' product of fp32 operands narrowed to fp16 first, as an `F16_F16_F32` preset's dot takes
    them on a GPU (XLA's CPU compiler has no float16 dot)."""
    return jnp.einsum(subscripts, query.astype(jnp.float16).astype(jnp.float32),
                      key.astype(jnp.float16).astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)


def test_a_logit_its_narrowed_operands_overflow_takes_the_prescaled_product():
    """jax.nn's form at width 16 in fp32 over a product whose dot narrows its operands to fp16 (as an
    `F16_F16_F32` preset's dot takes them on a GPU): a query of 65536 against a key of -1, the logit
    -16384, its raw product -inf (65536 past fp16's largest), so the query taken by its scale's quarter
    first (`attention.range_safe`), 16384 within fp16, where a bound of the fp32 products left it raw and
    the logit read -inf; and its cotangents pulled back through that prescaled query (train 1368's
    review)."""
    from dew.nn.attention import range_safe

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(65536.0)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-1.0)
    logits, pull = jax.vjp(lambda query, key: range_safe(_fp16_narrowed, query, key,
                                                         jnp.float32, None), query, key)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], -16384.0)
    # Its cotangents through the prescaled query too: the key's `1/4 · 65536`, where the raw query's
    # pullback read `1/4 · float16(65536) = inf`.
    dquery, dkey = pull(jnp.ones_like(logits))
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0], np.eye(1, width)[0] * 16384.0)
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0], np.eye(1, width)[0] * -0.25)


def test_a_raw_logits_key_cotangent_keeps_a_small_querys_bits_beside_a_recovered_one():
    """jax.nn's form at width 16 in fp32: a query of `(1 + 2^-23) 2^-126` and 8 against keys `[2^126, 0]`
    and `[0, -2^126]`, the first logit raw and the second recovered (-2^127, its raw product overflowing),
    under logit cotangents `[2^126, 0]`: the first key's cotangent `(1 + 2^-23) / 4` in its first entry,
    the raw product's pullback where it is finite (`attention.range_safe`), where pulling the whole row
    back through the prescaled query rounded its tiny entry and read 1/4 (train 1368's review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set((1.0 + 2.0 ** -23) * 2.0 ** -126).at[
        0, 0, 1].set(8.0)
    key = jnp.zeros((1, 2, width), jnp.float32).at[0, 0, 0].set(2.0 ** 126).at[0, 1, 1].set(-(2.0 ** 126))
    logits, pull = jax.vjp(lambda key: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True), key)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 1], -(2.0 ** 127))
    (dkey,) = pull(jnp.zeros_like(logits).at[0, 0, 0].set(2.0 ** 126))
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0, 0], np.float32((1.0 + 2.0 ** -23) / 4))


def test_a_logit_its_narrowed_key_overflows_takes_the_prescaled_key():
    """The same narrowing product with its operands' roles swapped: a query of -1 against a key of 65536,
    the logit -16384 and its cotangents the query's `1/4 · 65536` and the key's `-1/4`, the key taken by
    the scale's quarter first where the prescaled query's product still overflows (`attention.range_safe`),
    where only the query was prescaled and the logit and the query's cotangent read infinite (train 1461's
    review)."""
    from dew.nn.attention import range_safe

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-1.0)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(65536.0)
    logits, pull = jax.vjp(lambda query, key: range_safe(_fp16_narrowed, query, key, jnp.float32, None),
                           query, key)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], -16384.0)
    dquery, dkey = pull(jnp.ones_like(logits))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0], np.eye(1, width)[0] * 16384.0)
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0], np.eye(1, width)[0] * -0.25)


@pytest.mark.parametrize(("width", "magnitudes", "cotangents"),
                         [(16, (65536.0, 65536.0), (16384.0, 16384.0)),
                          (64, (65536.0, 131072.0), (16384.0, 8192.0))], ids=["even", "uneven"])
def test_a_logit_both_narrowed_operands_overflow_splits_the_prescale(width, magnitudes, cotangents):
    """The same narrowing product with a query and a key both past fp16's largest: the logit `2^30` and
    the operands' cotangents the scale times the other's, the prescale split between them by their headroom
    (`attention._prescale_split`: at width 64 and 1/8, a half on the query of 65536 and a quarter on the
    key of 131072, both 32768), where either one-sided prescale, or fixed splits of it, left an operand
    narrowed to infinity and read +inf (train 1461's reviews)."""
    from dew.nn.attention import range_safe

    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(magnitudes[0])
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(magnitudes[1])
    logits, pull = jax.vjp(lambda query, key: range_safe(_fp16_narrowed, query, key, jnp.float32, None),
                           query, key)
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], 2.0 ** 30)
    dquery, dkey = pull(jnp.ones_like(logits))
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0], np.eye(1, width)[0] * cotangents[0])
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0], np.eye(1, width)[0] * cotangents[1])


@pytest.mark.parametrize(("width", "queries", "keys", "logits", "cotangents"), [
    (16, (-1.0, -131072.0), (65536.0, 1.0), (-16384.0, -32768.0), ((16384.0, 0.25), (-0.25, -32768.0))),
    (64, (131072.0, 262144.0), (65536.0, 1.0), (2.0 ** 30, 32768.0), ((8192.0, 0.125), (16384.0, 32768.0))),
], ids=["one-sided", "uneven"])
def test_each_batchs_narrowed_logit_takes_its_own_prescale(width, queries, keys, logits, cotangents):
    """The same narrowing product over two batch items whose operands need different splits of the
    prescale: each item's logit and cotangents its own, a logit the whole call's headroom split leaves
    infinite taken by the split of the power's exponent that holds it (`attention._prescale_splits`: at
    width 64 and 1/8, a quarter on the first item's query of 131072 and a half on its key of 65536), where
    the call's one split, or a few fixed ones, left an item's operand narrowed to infinity (train 1461's
    reviews)."""
    from dew.nn.attention import range_safe

    query = jnp.zeros((2, 1, width), jnp.float32).at[:, 0, 0].set(jnp.asarray(queries))
    key = jnp.zeros((2, 1, width), jnp.float32).at[:, 0, 0].set(jnp.asarray(keys))
    found, pull = jax.vjp(lambda query, key: range_safe(_fp16_narrowed, query, key, jnp.float32, None),
                          query, key)
    np.testing.assert_array_equal(np.asarray(found)[:, 0, 0], logits)
    dquery, dkey = pull(jnp.ones_like(found))
    np.testing.assert_array_equal(np.asarray(dquery)[:, 0, 0], cotangents[0])
    np.testing.assert_array_equal(np.asarray(dkey)[:, 0, 0], cotangents[1])


def test_a_raw_rows_key_cotangent_keeps_its_bits_beside_a_narrowed_row():
    """jax.nn's form at width 16 in fp32 over a product whose dot narrows its operands to fp16: queries of
    65536 (narrowed to infinity, its logit recovered) and `(1 + 2^-10) 2^-14` (raw) against one key of -1,
    under logit cotangents `2^-16` and `-2^14`: the key's cotangent `1/4 - (1 + 2^-10) / 4 = -2^-12`, each
    query row pulled back through its own arm (`attention.range_safe`), where the prescaled pullback of
    both rows quartered the raw row's query below fp16's normals and read 0 (train 1368's review)."""
    from dew.nn.attention import range_safe

    width = 16
    query = jnp.zeros((1, 2, width), jnp.float32).at[0, 0, 0].set(65536.0).at[0, 1, 0].set(
        (1.0 + 2.0 ** -10) * 2.0 ** -14)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-1.0)
    _, pull = jax.vjp(lambda key: range_safe(_fp16_narrowed, query, key, jnp.float32, None),
                      key)
    (dkey,) = pull(jnp.asarray([[[2.0 ** -16], [-(2.0 ** 14)]]], jnp.float32))
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0, 0], -(2.0 ** -12))


def test_a_key_cotangent_past_fp32_in_its_split_sum_takes_the_common_power():
    """jax.nn's form at width 16 in fp32 under the explicit `BF16_BF16_F32` preset: queries of `2^127` (raw)
    and fp32's largest (narrowed to bf16's infinity, its logit recovered) against one key of -1, under
    logit cotangents 8 and -2: the key's cotangent `2^127`, its entry taken through the prescaled query at
    the common power (`attention.range_safe`), where the split's raw row alone formed `2 · 2^127` and read
    inf before the recovered row's `-2^127` met it (train 1368's review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 2, width), jnp.float32).at[0, 0, 0].set(2.0 ** 127).at[0, 1, 0].set(
        np.finfo(np.float32).max)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-1.0)
    _, pull = jax.vjp(lambda key: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True,
                                                   precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32), key)
    (dkey,) = pull(jnp.asarray([[[8.0], [-2.0]]], jnp.float32))
    np.testing.assert_array_equal(np.asarray(dkey)[0, 0, 0], 2.0 ** 127)


def test_a_broadcast_querys_recovered_cotangent_keeps_its_shape():
    """jax.nn's form at width 16 over `'...qd,...kd->...qk'` with a query `[1, 1, 16]` of 65536 (narrowed to
    fp16's infinity, its logits recovered) against keys `[2, 1, 16]` of -1: its cotangents under unit logit
    cotangents the query's shape, `2 · (-1/4)`, and each key's 16384 (`attention.range_safe`'s split
    pullback, its rows any of their broadcast copies), where the split broadcast the query to two rows and
    its pullback's branches disagreed in shape (train 1416's review)."""
    from dew.nn.attention import range_safe

    subscripts = "...qd,...kd->...qk"
    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(65536.0)
    key = jnp.zeros((2, 1, width), jnp.float32).at[:, 0, 0].set(-1.0)
    logits, pull = jax.vjp(lambda query, key: range_safe(
        lambda q, k: _fp16_narrowed(q, k, subscripts), query, key, jnp.float32, None), query, key)
    dquery, dkey = pull(jnp.ones_like(logits))
    assert dquery.shape == query.shape
    np.testing.assert_array_equal(np.asarray(dquery)[0, 0], np.eye(1, width)[0] * -0.5)
    np.testing.assert_array_equal(np.asarray(dkey)[:, 0, 0], [16384.0, 16384.0])
