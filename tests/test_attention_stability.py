"""The attention template's stability suite (bible §6): on inputs at fp32's edges, every attention form on
every mainloop (the Triton one interpreted, the plain-JAX one, the decode kernel) gives a finite output
and finite gradients wherever the reference does, and never ties values the reference keeps apart. Each
family fixes an adversarial structure (logits a ulp apart, near fp32's largest, a weight of exp(-87) on a
value of 2^126, rows that see nothing) and draws its values by seed, against the reference written apart
from Dew (`attention_oracle.exact_attention`), its float64 truth's derivative anchored
(`attention_oracle.anchored_softmax`); a row that sees no key is an exact claim, its output and
gradients zero. The accurate form (natural logits, `template.NATURAL`'s other form, and the backward's
accurate form, `template.ACCURATE`'s) holds check 1 (`reference_error.assert_as_exact`) on every family,
as exact as the reference. The default, which folds log2(e) into the scale where the scale allows and
takes FlashAttention's arithmetic for the backward's products and dQ (about the template's own anchored
D, not FlashAttention's rowsum(dO * O)), is finite wherever the reference is and holds the exact-zero
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
import numpy as np
import pytest
from attention_oracle import exact_attention
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference, with_gradients

from dew.nn.documents import CrossDocuments
from dew.nn.kernels import decode_attention, generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_portable,
    attention_triton,
    mosaic_sm90,
)

PINS = {attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2),
        attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2)}
"""The Triton mainloop's tiles: every family is a block or less, so one tile walks it."""

INTERPRETED = jax.default_backend() == "cpu"
"""Whether the frozen Triton mainloop runs interpreted, in a CPU process, or compiled, on a GPU: the A100's
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
    weighted values' sums passing fp32's largest (`overflows`), which a walk guarding its sums' headroom
    (`algebra.Forward.headroom`) keeps finite in either form; the slices the decode kernel cuts
    each row's keys into (`splits`), whose states it merges; and whether it is a decode step's alone
    (`decode_only`), a shape the template's interpreted mainloops would take too long over, which its
    template twin covers in fewer programs. Its declared seeds are `SEEDS`'."""

    draw: Callable[[int], tuple]
    keep: np.ndarray
    spans: dict = dataclasses.field(default_factory=dict)
    empty: np.ndarray | None = None
    folded: bool = True
    overflows: bool = False
    dtype: str = "float32"
    splits: int = 1
    decode_only: bool = False


def _normal(seed: int, *shapes) -> list[np.ndarray]:
    """Standard-normal fp32 arrays of `shapes`, one stream each."""
    streams = jax.random.split(jax.random.key(seed), len(shapes))
    return [np.array(jax.random.normal(stream, shape), np.float32)
            for stream, shape in zip(streams, shapes, strict=True)]


def _uniform(seed: int, low: float, high: float, shape=()) -> np.ndarray:
    """fp32 draws uniform in `[low, high)`."""
    return np.asarray(jax.random.uniform(jax.random.key(seed), shape, jnp.float32, low, high))


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
         "values near fp32's largest over 2 equal keys, bfloat16": 8, "one key": 8, "a sink and one key": 64,
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
         "two finite slices whose merge overflows": 8, "a resumed state of a large denominator": 64,
         "a resumed state beside an overflowing column": 32,
         "a resumed state of a denominator near fp32's largest beside an overflowing column": 8,
         "fp16 values near their largest beside a state near fp32's largest": 32,
         "keys an ulp apart, float32": 32, "keys an ulp apart, bfloat16": 32,
         "keys near fp32's largest at equal logits": 32, "a far maximum in a later key tile": 8,
         "a decode step's sink and one key": 32, "a decode step's sink near fp32's largest and one key": 8,
         "documents of one key": 8}
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
    # value of 2^e, e drawn in [120, 126]: its share of the output, which a weight shifted for the sums'
    # headroom before it is narrowed would lose. At logits of -80 to -87 the weight is a normal fp32 and
    # bf16 at their exponent's edge, which a weight shifted before it is narrowed would flush
    # (`Room.small`). One key tile a row, so the family's many rows stay cheap to interpret; `cases`
    # independent cases a row, each its own document of `queries` queries over its own two keys
    # (`CrossDocuments`), so a draw holds rows times cases of them in few programs.
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
    # guarded walk shifts the row, and the second column's share, about 1e-7, must survive it.
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
    # entries are D times the other keys' weighted rows.
    key = np.zeros((rows, 256, 1, 16), np.float32)
    if not equal:
        key[:, :255, 0, 0] = -_uniform(seed + 5, 0.0, 1.0, (rows, 255))
    key[:, :255, 0, 1:] = _normal(seed, (rows, 255, 15))[0]
    key[:, 255, 0, 0] = -_uniform(seed + 1, 84.0, 88.0, (rows,))
    value = np.zeros((rows, 256, 1, 16), np.float32)
    value[:, :255, 0, 0] = _uniform(seed + 2, 2.9e38, 3.3e38, (rows, 255))
    value[:, 255, 0, 1] = _uniform(seed + 3, 5e37, 1e38, (rows,))
    dout = np.zeros((rows, 1, 1, 16), np.float32)
    dout[..., 1] = _normal(seed + 4, (rows, 1, 1))[0]
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


def _large_values(keys: int, top: float = 3.4e38):
    # Equal logits over values drawn within a few percent of fp32's largest (to `top`, bf16's below its
    # largest), one sign: the sum of the weighted values passes fp32's largest where the reference's
    # normalized one does not. The cotangent is drawn times 2^-8, so the cotangent's product with the
    # values stays finite in the reference's backward too.
    def draw(seed):
        value = _uniform(seed, 3.0e38 * top / 3.4e38, top, (ROWS, keys, 1, 16))
        (dout,) = _normal(seed + 1, (ROWS, 1, 1, 16))
        return (np.zeros((ROWS, 1, 1, 16), np.float32), np.zeros((ROWS, keys, 1, 16), np.float32), value,
                None, dout * np.float32(2.0 ** -8))
    return draw


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
    # column alone, the state's: a guarded training walk's normalized split sums carry the state's share.
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
    # (`algebra.headroom`: an fp16 call compiles no guard), so the output is the state's mean, finite.
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
    # (`algebra.DqSums.keyed`).
    key = np.zeros((ROWS, 8, 1, 16), np.float32)
    key[..., 0] = _uniform(seed, 1.7e38, 3.4e38, (ROWS, 8, 1))
    value, dout = _normal(seed + 1, (ROWS, 8, 1, 16), (ROWS, 1, 1, 16))
    return np.zeros((ROWS, 1, 1, 16), np.float32), key, value, None, dout


def _far_maximum_in_a_later_tile(seed: int, rows: int = 64):
    # One query a row over 64 keys in two tiles of 32: the first tile's keys 88 to 92 nats below the second
    # tile's, which hold the row's maximum, so the first tile's weights are subnormal beside the maximum
    # the backward walks at: a dQ pass's running sums over its first tile (`algebra.DqSums`) take only
    # those, and must stay finite and exact when the maximum's tile joins them.
    query = np.zeros((rows, 1, 1, 16), np.float32)
    query[..., 0] = 4.0
    key = np.zeros((rows, 64, 1, 16), np.float32)
    key[:, :32, 0, 0] = -_uniform(seed, 88.0, 92.0, (rows, 32))
    key[:, 32:, 0, 1:] = _normal(seed + 1, (rows, 32, 15))[0]
    value, dout = _normal(seed + 2, (rows, 64, 1, 16), (rows, 1, 1, 16))
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
        # does not guard its sums (every mainloop's default does, `overflows`); a sink there.
        "a logit near fp32's largest": Family(_near_max_logit(1), _kept(1, 1), folded=False),
        # Folded at scale 1/4, the backward takes the base-2 maximum back to natural units, which near fp32's
        # largest rounds the key's own weight away from 1, as at logits near 1e30.
        "a logit near fp32's largest at scale 1/4": Family(_near_max_logit(16), _kept(1, 1), folded=False),
        "values near fp32's largest over 2 equal keys": Family(_large_values(2), _kept(1, 2), folded=False,
                                                               overflows=True),
        "values near fp32's largest over 64 equal keys": Family(_large_values(64), _kept(1, 64),
                                                                folded=False, overflows=True),
        "values near fp32's largest over 2 equal keys, bfloat16": Family(
            _large_values(2, top=3.3e38), _kept(1, 2), folded=False, overflows=True, dtype="bfloat16"),
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
        # fold's sum of those values overflows where a walk does not guard its sums.
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
        # A query's gradient that cancels across its keys, in either dtype (`algebra.cotangent_product`).
        "keys an ulp apart, float32": Family(_near_keys("float32"), _kept(1, 2)),
        "keys an ulp apart, bfloat16": Family(_near_keys("bfloat16"), _kept(1, 2), dtype="bfloat16"),
        "a subnormal share beside 255 equal weights, bfloat16": Family(
            functools.partial(_subnormal_share, equal=True), np.ones((64, 1, 1, 256), bool), folded=False,
            overflows=True, dtype="bfloat16"),
    }
    return {**families, **_structured_families()}


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
FORMS = {"default": None, "natural": True}
"""The forms the decode kernel takes each family in: the default (`template.NATURAL`) and natural logits."""
FAMILY_RUNS = [(family, form) for family in FAMILIES for form in FORMS
               if form == "natural" or FAMILIES[family].folded]
TEMPLATE_FAMILIES = [family for family in FAMILIES if not FAMILIES[family].decode_only]
"""The template's families: every one but a decode step's alone (`Family.decode_only`)."""
ACCURATE = {"natural": True, "accurate": True}
"""The template's accurate form: natural logits and the backward's accurate form."""


def _spans(family: Family, query, key) -> template.KeySpans:
    def placed(value):
        if isinstance(value, CrossDocuments):
            return CrossDocuments(jnp.asarray(value.query), jnp.asarray(value.key))
        return jnp.asarray(value) if isinstance(value, np.ndarray) else value
    arguments = {name: placed(value) for name, value in family.spans.items()}
    return template.spans(query.shape[1], key.shape[1], query.shape[0], **arguments)


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


def _reference(family: Family, *, anchored: bool) -> Callable:
    """The reference's output `(query, key, value, extra) -> out`, in the call's dtype as every attention
    call returns it, its gradients autodiff's (`reference_error.with_gradients`); the float64 truth's
    derivative `anchored`. The reference arm
    is jax's own attention (`jax.nn.dot_product_attention`'s xla path, differentiated by autodiff), the
    established computation Dew's is held to at its declared precision, which rounds the probabilities to
    the values' dtype for their products (the gate's ruling, row 1362), wherever its API holds the
    family; the written-out one where a family has sinks or a resumed state, which it has no input for,
    or a query of no keys, to which it gives the values' mean where Dew and the truth give zeros (every
    such family fp32, whose probabilities round to nothing)."""
    jax_holds = not anchored and np.asarray(family.keep).any(-1).all()

    def attend(query, key, value, extra):
        if isinstance(extra, tuple):
            out = _resumed_attention(query, key, value, extra, family.keep)
        elif extra is None and jax_holds:
            out = jax.nn.dot_product_attention(query, key, value, mask=jnp.asarray(family.keep),
                                               implementation="xla")
        else:
            out = exact_attention(query, key, value, extra, keep=family.keep, anchored=anchored)
        return out.astype(query.dtype)

    return attend


def _resumed_attention(query, key, value, state, keep):
    """Attention resumed from a softmax state `(m, total, acc)` (one query head a key head), written apart
    from Dew with normalized weights, so a state whose sums are near fp32's largest stays finite: the
    state's weight `total e^(m - M)` and each kept key's `e^(s - M)` over their sum, the state reading its
    values' mean `acc / total`; in fp32 or the inputs' wider dtype, as the state is, the probabilities
    rounded to the values' dtype for their product (row 1362)."""
    wide = jnp.promote_types(query.dtype, jnp.float32)
    m, total, acc = (jnp.asarray(x, wide) for x in state)
    query, key = query.astype(wide), key.astype(wide)
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) / jnp.sqrt(jnp.asarray(query.shape[-1], wide))
    logits = jnp.where(keep, logits, -jnp.inf)
    top = jnp.maximum(m, jnp.max(logits, axis=-1))
    own = total * jnp.exp(m - top)
    weights = jnp.exp(logits - top[..., None])
    denominator = own + weights.sum(axis=-1)
    mean = acc / total[..., None]
    shares = weights / denominator[..., None]
    weighted = jnp.einsum("bhqk,bkhd->bhqd", shares.astype(value.dtype), value, preferred_element_type=wide)
    out = mean * (own / denominator)[..., None] + weighted
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


def _differentiable(family: Family, forward: Callable, backward: Callable) -> Callable:
    """A mainloop's `forward` and `backward` as `(query, key, value, extra) -> out` over `family`'s spans,
    its derivative the mainloop's own backward, the sinks' or resumed state's gradient through the softmax's
    initial state."""
    def run(query, key, value, extra):
        spans, hooks = _spans(family, query, key), _hooks(query, extra)

        def ahead(query, key, value, initial):
            return forward(query, key, value, spans, template.Hooks(initial=initial))

        def back(query, key, value, initial, dout):
            return backward(query, key, value, spans, template.Hooks(initial=initial), dout)

        initial = None if hooks is None else hooks.initial
        return _differentiated(ahead, back)(query, key, value, initial)
    return run


def _mainloops(family: Family, options: dict) -> dict[str, Callable]:
    """Each template mainloop as `(query, key, value, extra) -> out` at the form `options` names (`ACCURATE`,
    or none for the default), its derivative the mainloop's own backward (`_differentiable`)."""
    forward_options = {name: value for name, value in options.items() if name in ("natural", "scale")}
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    mainloop = functools.partial(_differentiable, family)

    def triton_forward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.forward(*operands, interpret=INTERPRETED, **forward_options)

    def triton_backward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.backward(*operands, interpret=INTERPRETED, **options)

    def mosaic_forward(*operands):
        return attention_mosaic.forward(*operands, **forward_options)

    def mosaic_backward(query, key, value, spans, hooks, dout):
        # From the mainloop's own training forward's residuals, a guarded one's (its overflowed rows walk
        # again guarded, in the launch's form).
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, residuals=True,
                                             **forward_options)[1:]
        return attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                         **options)

    portable = mainloop(functools.partial(attention_portable.attend, **tiles, **forward_options),
                        functools.partial(attention_portable.backward, **tiles, **options))
    mainloops = {"triton": mainloop(triton_forward, triton_backward), "portable": portable}
    if ON_HOPPER and _mosaic_takes(family):
        mainloops["mosaic"] = mainloop(mosaic_forward, mosaic_backward)
    return mainloops


ON_HOPPER = generation.device_generation() == "sm90"
"""Whether the Mosaic mainloop runs here, compiled on an H100: interpreted, a family takes it 6 to 33 minutes
on a CPU, past a CI task's time, and the interpreter is coverage, not proof (`test_attention_mosaic`)."""


def _mosaic_takes(family: Family) -> bool:
    """Whether the Mosaic mainloop takes `family`'s calls, forward and backward, by its own legality rule
    (`attention_mosaic.refused`)."""
    query, _, value, extra, _ = family.draw(0)
    hooks = _hooks(jnp.asarray(query), jax.tree.map(jnp.asarray, extra))
    return attention_mosaic.refused(jnp.dtype(family.dtype), query.shape[-1], value.shape[-1], hooks,
                                    backward=True, family=mosaic_sm90.WGMMA) is None


def _held_empty(family: Family, found, label: str) -> None:
    """A row that sees no key reads nothing and passes nothing back: its output and query gradient are
    zero exactly."""
    if family.empty is None:
        return
    rows = np.asarray(family.empty)
    for leaf, array in zip(("output", "dq"), found[:2], strict=True):
        np.testing.assert_array_equal(np.asarray(array, np.float32)[rows], 0.0, err_msg=f"{label} {leaf}")


def assert_finite(dew_fn, reference_fn, draw, label: str, *, seeds: int) -> None:
    """The stability rule's finiteness over `seeds` draws: each leaf of `dew_fn`'s outputs finite wherever
    the reference's is."""
    for seed in range(seeds):
        inputs = draw(seed)
        found = jax.tree_util.tree_leaves_with_path(dew_fn(*inputs))
        for (path, mine), theirs in zip(found, jax.tree.leaves(reference_fn(*inputs)), strict=True):
            mine, theirs = (np.asarray(leaf, np.float32) for leaf in (mine, theirs))
            lost = np.isfinite(theirs) & ~np.isfinite(mine)
            assert not lost.any(), (f"{label}{jax.tree_util.keystr(path)} seed {seed}: {int(lost.sum())} "
                                    "values not finite where the reference's are")


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_the_accurate_form_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family):
    """Each template mainloop's output and query, key and value gradients in the accurate form
    (`ACCURATE`), check 1 over the family's draws; a row that sees nothing, zero."""
    seeds, family = SEEDS[family], FAMILIES[family]
    draw = _cast(family, family.draw)
    reference = as_given(jax.jit(with_gradients(_reference(family, anchored=False))))
    truth = with_gradients(_reference(family, anchored=True))
    for name, mainloop in _mainloops(family, ACCURATE).items():
        label = f"{name} accurate {family.dtype}"
        differentiated = jax.jit(with_gradients(mainloop))
        assert_as_exact(differentiated, reference, draw, label, seeds=seeds, truth_fn=truth)
        _held_empty(family, differentiated(*draw(0)), label)


def _default_is_finite(family_name: str, mainloop_name: str) -> None:
    """A template mainloop's output and query, key and value gradients in the default form, finite wherever
    the reference's are (`assert_finite`) over the family's draws; a row that sees nothing, zero. Its error
    here is no claim, and its ties the documented divergence (the module's docstring)."""
    seeds, family = SEEDS[family_name], FAMILIES[family_name]
    draw = _cast(family, family.draw)
    reference = as_given(jax.jit(with_gradients(_reference(family, anchored=False))))
    label = f"{mainloop_name} default {family.dtype}"
    differentiated = jax.jit(with_gradients(_mainloops(family, {})[mainloop_name]))
    assert_finite(differentiated, reference, draw, label, seeds=seeds)
    _held_empty(family, differentiated(*draw(0)), label)


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_the_default_form_is_finite_where_the_reference_is_at_fp32s_edges(family):
    """The plain-JAX mainloop, which walks under the default's guards (`algebra.walked`: its sums' headroom,
    the fold's reach), finite in the default form over every family (`_default_is_finite`)."""
    _default_is_finite(family, "portable")


MOSAIC_FAMILIES = [family for family in TEMPLATE_FAMILIES if _mosaic_takes(FAMILIES[family])]
"""The template's families the Mosaic mainloop takes (`_mosaic_takes`)."""


@pytest.mark.skipif(not ON_HOPPER, reason="runs the Mosaic mainloop compiled, on an H100")
@pytest.mark.gpu(reason="runs the Mosaic mainloop compiled, on an H100")
@pytest.mark.parametrize("family", MOSAIC_FAMILIES)
def test_the_mosaic_mainloops_default_is_finite_where_the_reference_is_at_fp32s_edges(family):
    """The Mosaic mainloop, which walks under the default's guards over whole launches
    (`algebra.walked_launches`), finite in the default form over every family it takes (its legality rule,
    `attention_mosaic.refused`: bf16 operands in its backward, hooks it lowers; `_default_is_finite`)."""
    _default_is_finite(family, "mosaic")


@pytest.mark.parametrize("family", [family for family in TEMPLATE_FAMILIES
                                    if FAMILIES[family].folded or FAMILIES[family].overflows])
def test_the_frozen_triton_mainloops_default_is_finite_where_the_reference_is_at_fp32s_edges(family):
    """The Triton mainloop's default form finite over the families the fold holds and those whose sums pass
    fp32's largest (`_default_is_finite`): it guards its sums' headroom in either form, as the template's
    production route runs it on an sm80 or later GPU. Frozen until its Mosaic mainloop replaces it, it
    walks without the fold's reach (`algebra.walked`), so the fold's other divergences remain an open
    defect (ledger 1637)."""
    _default_is_finite(family, "triton")


@pytest.mark.parametrize("family", [family for family in TEMPLATE_FAMILIES if FAMILIES[family].overflows])
def test_the_frozen_triton_backward_reads_a_guarded_forwards_rows_whatever_its_pass(family):
    """The Triton mainloop's default backward asked for its unsplit first pass (`split=False`) over sums that
    pass fp32's largest, finite wherever the reference is: a guarded forward's live rows carry normalized
    split sums, which only the split pass reads, so it takes that pass wherever the values can overflow
    (`attention_triton.backward`), in either form."""
    seeds, family_of = SEEDS[family], FAMILIES[family]

    def forward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.forward(*operands, interpret=INTERPRETED)

    def backward(*operands):
        with tuning.pinned(PINS):
            return attention_triton.backward(*operands, interpret=INTERPRETED, split=False)

    reference = as_given(jax.jit(with_gradients(_reference(family_of, anchored=False))))
    unsplit = jax.jit(with_gradients(_differentiable(family_of, forward, backward)))
    assert_finite(unsplit, reference, _cast(family_of, family_of.draw), f"triton unsplit {family_of.dtype}",
                  seeds=seeds)


def test_a_resumed_state_of_large_values_under_the_folds_guard_is_finite():
    """fp32, the default form: one query over one key of logit m0 = 24576.005859375 (folded below the fold's
    reach, `algebra.FOLD_REACH`, and naturalized back below m0), resumed from a state of maximum m0,
    denominator 1 and values 1023 * 2^118, the key's value the same: the plain walk's weighted values
    overflow, so the guarded walk's live row takes its split sums from the normalized pass, whose share of
    the state is taken at the walk's own maximum, at most 1 (`algebra.normalized_pass`); at the naturalized
    one it passed 1 and overflowed the share. The plain-JAX mainloop's output and gradients are finite, as
    the reference's are (`_resumed_attention`)."""
    m0, size = 24576.005859375, 1023.0 * 2.0 ** 118

    def first(fill, *shape):
        return jnp.zeros(shape, jnp.float32).at[..., 0].set(fill)

    query, key, value, dout = (first(fill, 1, 1, 1, 16) for fill in (4.0, m0, size, 2.0 ** -8))
    state = (jnp.full((1, 1, 1), m0, jnp.float32), jnp.ones((1, 1, 1), jnp.float32), first(size, 1, 1, 1, 16))
    family = Family(lambda seed: (query, key, value, state, dout), np.ones((1, 1, 1, 1), bool))
    reference = with_gradients(lambda *args: _resumed_attention(*args, family.keep))
    ours = jax.jit(with_gradients(_mainloops(family, {})["portable"]))
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
        _, *grads = jax.jit(with_gradients(mainloop))(query, key, value, None, dout)
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
        out, *grads = jax.jit(with_gradients(mainloop))(query, key, value, None, dout)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(value), err_msg=name)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in grads[:3]), f"{name}'s gradients"


DECODED = [(family, form) for family, form in FAMILY_RUNS
           if FAMILIES[family].draw(0)[0].shape[1] == 1 and FAMILIES[family].draw(0)[0].shape[3] >= 16
           and not isinstance(FAMILIES[family].draw(0)[3], tuple)]
"""The families a decode step is: one query a row, a head width the decode kernel takes (16 to 256),
sinks or none (a resumed state is the template's)."""


def _cache(array: jax.Array) -> jax.Array:
    """Keys or values `[B, K, H, D]` in a cache of whole decode blocks (`decode_attention.BLOCK`, the
    capacity its geometry takes), the slots past the keys zeros."""
    slots = -array.shape[1] % decode_attention.BLOCK
    return jnp.pad(array, ((0, 0), (0, slots), (0, 0), (0, 0)))


def _decoded(family: Family, natural: bool | None) -> tuple[Callable, Callable]:
    """The decode kernel over `family`'s step, interpreted on a CPU (as an A100's launch) and compiled on a
    GPU, each row's keys the first `K` of its cache, in the form `natural` names (None, the default), the
    family's slice count pinned, and the family's draw without its cotangent (a decode step has no
    backward)."""
    lengths = family.spans.get("lengths")
    form = {} if natural is None else {"natural": natural}

    def decoded(query, key, value, sinks):
        count = np.full(query.shape[0], key.shape[1]) if lengths is None else lengths
        # The family's slice count, which a launch on a CPU, of no multiprocessors, does not offer: there
        # as an A100's launch.
        pin = {decode_attention.DecodeAttention: decode_attention.Split(family.splits)}
        target = generation.assumed("sm80") if INTERPRETED else contextlib.nullcontext()
        with target, tuning.pinned(pin):
            return decode_attention.attend(query, _cache(key), _cache(value), jnp.asarray(count, jnp.int32),
                                           sinks=sinks, interpret=INTERPRETED, **form)

    def forward(seed):
        return _cast(family, family.draw)(seed)[:4]

    return decoded, forward


@pytest.mark.parametrize("family, form", DECODED, ids=[f"{family}-{form}" for family, form in DECODED])
def test_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family, form):
    """The decode kernel's output, interpreted on a CPU and compiled on a GPU (its guarded launch and merge),
    each row's keys the first `K` of its cache, check 1 over the family's draws."""
    seeds, family = SEEDS[family], FAMILIES[family]
    decoded, forward = _decoded(family, FORMS[form])
    assert_as_exact(jax.jit(decoded), as_given(jax.jit(_reference(family, anchored=False))), forward,
                    f"decode {form} {family.dtype}", seeds=seeds,
                    truth_fn=_reference(family, anchored=True))

DECODED_OVERFLOWS = list(dict.fromkeys(family for family, _ in DECODED if FAMILIES[family].overflows))
"""The decode steps (`DECODED`) whose one divergence in the default form is their sums passing fp32's
largest."""


@pytest.mark.parametrize("family", DECODED_OVERFLOWS)
def test_decodes_default_is_finite_where_its_sums_pass_fp32s_largest(family):
    """The decode kernel's default form, interpreted on a CPU and compiled on a GPU, finite wherever the
    reference is (`assert_finite`) over the families whose weighted values' sums pass fp32's largest: it
    guards its sums' headroom in either form (`algebra.Forward.headroom`)."""
    seeds, family = SEEDS[family], FAMILIES[family]
    decoded, forward = _decoded(family, None)
    assert_finite(jax.jit(decoded), as_given(jax.jit(_reference(family, anchored=False))), forward,
                  f"decode default {family.dtype}", seeds=seeds)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_guarded_rows_split_sums_keep_a_subnormal_share_element_wise(dtype):
    """The subnormal-share family's training forward in both mainloops: every row overflows, so its split
    sums are the normalized pass's (shift -1, the anchor its largest logit's key). The share of the key whose
    normalized weight is below fp32's least normal, alone in `B`'s second column, held element-wise to
    float64 as exactly as the reference's own fp32 arithmetic for it (its weight `e / T` and that times the
    value, `assert_as_exact_as_the_reference`): a check of its rows, which a leaf's RMS beside the first
    column's would not see."""
    query, key, value, _, _ = _cast(Family(_subnormal_share, np.ones((64, 1, 1, 256), bool), dtype=dtype),
                                    _subnormal_share)(0)
    spans = template.spans(1, 256, 64)
    wide = [np.asarray(x, np.float64) for x in (query, key, value)]
    logits = np.einsum("bqhd,bkhd->bhqk", wide[0], wide[1]) / 4.0
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    truth = weights[..., 255] / weights.sum(axis=-1) * wide[2][:, 255, 0, 1][:, None, None]
    narrow = np.asarray(jnp.einsum("bqhd,bkhd->bhqk", jnp.asarray(query, jnp.float32),
                                   jnp.asarray(key, jnp.float32)) / 4.0)
    exponentials = np.exp(narrow - narrow.max(axis=-1, keepdims=True))
    reference = (exponentials[..., 255] / exponentials.sum(axis=-1)
                 * np.asarray(value, np.float32)[:, 255, 0, 1][:, None, None])
    with tuning.pinned(PINS):
        triton = attention_triton.forward(query, key, value, spans, residuals=True, natural=True,
                                          interpret=INTERPRETED)
    portable = attention_portable.attend(query, key, value, spans, block_q=32, block_k=32,
                                         pad=attention_triton.BLOCK, residuals=True, natural=True)
    for name, (_, _, _, anchors, split) in (("triton", triton), ("portable", portable)):
        np.testing.assert_array_equal(np.asarray(split.shift), -1.0, err_msg=name)
        np.testing.assert_array_equal(np.asarray(anchors), logits.argmax(axis=-1), err_msg=name)
        found = np.asarray(split.others[..., 1], np.float64)[..., None]
        assert_as_exact_as_the_reference(found, reference[..., None], truth[..., None],
                                         f"{name} {dtype} subnormal share")

