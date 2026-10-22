"""The attention template's stability suite (bible §6): on inputs at fp32's edges, every attention form on
every mainloop (the Triton one interpreted, the plain-JAX one, the decode kernel) gives a finite output
and finite gradients wherever the reference does, and is as exact as it, which keeps logits it keeps
apart apart. Each family fixes an adversarial structure (logits a ulp apart, near fp32's largest, a
weight of exp(-87) on a value of 2^126, rows that see nothing) and draws its values by seed, held to
check 1 (`reference_error.assert_as_exact`) against the reference written apart from Dew
(`attention_oracle.exact_attention`), its float64 truth's derivative anchored
(`attention_oracle.anchored_softmax`); a row that sees no key is an exact claim, its output and
gradients zero. Each family runs with the logits natural and with the default
(`dew.nn.kernels.templates.attention.NATURAL`), which folds log2(e) into the scale where the scale
allows; the fold's documented divergences (`algebra.Forward.base2`; docs/design/kernel-templates.md, the
forms' divergences) run natural alone."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import exact_attention
from reference_error import assert_as_exact

from dew.nn.kernels import decode_attention, tuning
from dew.nn.kernels.templates import attention as template, attention_portable, attention_triton

PINS = {attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2),
        attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2)}
"""The Triton mainloop's tiles: every family is a block or less, so one tile walks it."""


@dataclasses.dataclass(frozen=True)
class Family:
    """A stability family: `draw(seed)` gives the queries `[B, Q, H, D]`, keys and values `[B, K, H, D]`,
    the sinks `[H]` or None and the output cotangent, fp32, run in `dtype`, the softmax's scale
    1/sqrt(D); the keys each query keeps `[B, 1, Q, K]` and the same keys as the template's spans'
    arguments (`template.spans`); the rows `[B, Q]` that see no key, whose output and gradients are
    zero (`empty`); whether the default form holds it (`folded`), False only for the fold's documented
    divergences (`algebra.Forward.base2`), which run natural alone; and its declared seeds, sized by
    `reference_error.seeds_for`."""

    draw: Callable[[int], tuple]
    keep: np.ndarray
    spans: dict = dataclasses.field(default_factory=dict)
    empty: np.ndarray | None = None
    folded: bool = True
    dtype: str = "float32"
    seeds: int = 8


def _normal(seed: int, *shapes) -> list[np.ndarray]:
    """Standard-normal fp32 arrays of `shapes`, one stream each."""
    streams = jax.random.split(jax.random.key(seed), len(shapes))
    return [np.asarray(jax.random.normal(stream, shape), np.float32)
            for stream, shape in zip(streams, shapes, strict=True)]


def _uniform(seed: int, low: float, high: float, shape=()) -> np.ndarray:
    """fp32 draws uniform in `[low, high)`."""
    return np.asarray(jax.random.uniform(jax.random.key(seed), shape, jnp.float32, low, high))


def _first(sizes: np.ndarray, width: int) -> np.ndarray:
    """`[1, len(sizes), 1, width]` vectors of `sizes` in their first entry and zeros past it, so a
    product of two of them is its sizes' product exactly at every width."""
    out = np.zeros((1, len(sizes), 1, width), np.float32)
    out[0, :, 0, 0] = sizes
    return out


def _causal(queries: int, keys: int) -> np.ndarray:
    return np.tril(np.ones((queries, keys), bool))[None, None]


def _ties(width: int, cotangent: float) -> Callable[[int], tuple]:
    """One query of 1 over two keys whose logits are a drawn fp32 near 2^30 and the one a drawn 1 to 4
    ulps below it (Sol's case: the reference weighs the first alone, which a base-2 fold rounding each
    logit at log2(e) times its size can tie), their keys scaled by sqrt(`width`) so the logits are those
    at every scale; drawn values, the cotangent drawn times `cotangent`."""
    def draw(seed: int):
        top = np.float32(_uniform(seed, 2.0 ** 30, 2.0 ** 31))
        below = top
        for _ in range(int(_uniform(seed + 1, 1, 5))):
            below = np.nextafter(below, np.float32(0))
        value, dout = _normal(seed + 2, (1, 2, 1, width), (1, 1, 1, width))
        root = np.float32(np.sqrt(width))
        return (_first(np.ones(1, np.float32), width), _first(np.asarray([top, below]) * root, width), value,
                None, dout * np.float32(cotangent))

    return draw


def _families() -> dict[str, Family]:
    every = np.ones((1, 1, 1, 1), bool)

    def near_max_logit(width: int):
        # One key whose logit is a drawn fp32 near its largest: the output is the key's value.
        def draw(seed):
            size = _uniform(seed, 1e38, 3e38) * np.sqrt(width)
            key, value, dout = _normal(seed + 1, (1, 1, 1, width), (1, 1, 1, width), (1, 1, 1, width))
            return _first(np.asarray([size], np.float32), width), np.abs(key) + 0.5, value, None, dout
        return draw

    def large_values(keys: int):
        # Equal logits over values drawn within a few percent of fp32's largest, one sign: the sum of the
        # weighted values passes fp32's largest where the reference's normalized one does not.
        def draw(seed):
            value = _uniform(seed, 3.0e38, 3.4e38, (1, keys, 1, 16))
            (dout,) = _normal(seed + 1, (1, 1, 1, 16))
            return (np.zeros((1, 1, 1, 16), np.float32), np.zeros((1, keys, 1, 16), np.float32), value, None,
                    dout)
        return draw

    def drawn(queries, keys, width=16, rows=1, size=1.0):
        def draw(seed):
            query, key, value, dout = _normal(seed, (rows, queries, 2, width), (rows, keys, 2, width),
                                              (rows, keys, 2, width), (rows, queries, 2, width))
            return query * np.float32(size), key, value, None, dout
        return draw

    def with_sinks(base, low, high):
        # Drawn sink logits for each of the base draw's heads.
        def draw(seed):
            query, key, value, _, dout = base(seed)
            return query, key, value, _uniform(seed + 7, low, high, (query.shape[2],)), dout
        return draw

    def small_weight(dtype):
        # A weight of exp(-l), l drawn in [80, 87), normal in fp32 and bf16, on a value of 2^e, e drawn in
        # [120, 126], among 256 keys of which two are seen: the output is that product alone, which a
        # weight shifted for the keys' count before it is narrowed would lose.
        def draw(seed):
            key = np.zeros((1, 256, 1, 16), np.float32)
            key[0, 1, 0, 0] = -_uniform(seed, 80.0, 87.0)
            value = np.zeros((1, 256, 1, 16), np.float32)
            value[0, 1, 0, 0] = 2.0 ** np.floor(_uniform(seed + 1, 120.0, 127.0))
            (dout,) = _normal(seed + 2, (1, 1, 1, 16))
            return _first(np.asarray([4.0], np.float32), 16), key, value, None, dout
        seen = np.zeros((1, 1, 1, 256), bool)
        seen[..., :2] = True
        return Family(draw, seen, {"lengths": np.asarray([2])}, dtype=dtype)

    families = {
        "ties at scale 1": Family(_ties(1, 1.0), np.ones((1, 1, 1, 2), bool)),
        "ties at scale 1, cotangents near 2**125": Family(_ties(1, 2.0 ** 125), np.ones((1, 1, 1, 2), bool)),
        "ties at scale 1/2": Family(_ties(4, 1.0), np.ones((1, 1, 1, 2), bool), folded=False),
        "ties at scale 1/2, cotangents near 2**125": Family(_ties(4, 2.0 ** 125), np.ones((1, 1, 1, 2), bool),
                                                            folded=False),
        "ties at scale 1/4": Family(_ties(16, 1.0), np.ones((1, 1, 1, 2), bool), folded=False),
        "ties at scale 1/4, cotangents near 2**125": Family(_ties(16, 2.0 ** 125),
                                                            np.ones((1, 1, 1, 2), bool), folded=False),
        "a logit near fp32's largest": Family(near_max_logit(1), every),
        "a logit near fp32's largest at scale 1/4": Family(near_max_logit(16), every),
        "values near fp32's largest over 2 equal keys": Family(large_values(2), np.ones((1, 1, 1, 2), bool)),
        "values near fp32's largest over 64 equal keys": Family(large_values(64),
                                                                np.ones((1, 1, 1, 64), bool)),
        # Folded, the key's logit times log2(e) and its maximum round apart (the product contracts into the
        # subtraction), so its weight is 1 within an ulp and the output its value within one ulp: a bound
        # check 2 holds where the gate's exp2 oracle states it (`algebra.Forward.base2`).
        "one key": Family(drawn(4, 1), np.ones((1, 1, 4, 1), bool), folded=False),
        "a sink and one key": Family(with_sinks(drawn(1, 1), -3.0, 3.0), every),
        "a sink near fp32's largest and one key": Family(with_sinks(drawn(1, 1), 1e38, 3e38), every),
        "a sink and one key, its logit near fp32's largest": Family(
            with_sinks(near_max_logit(1), -3.0, 3.0), every),
        "a sink near fp32's largest at scale 1/4": Family(with_sinks(drawn(1, 1), 1e38, 3e38), every,
                                                          folded=False),
        "logits near 1e4": Family(drawn(64, 64, size=1e4), _causal(64, 64), {"causal": True}),
        # Folded, the backward takes the base-2 maximum back to natural units, which at 1e30 rounds the
        # anchor's own weight away from 1 by more than any bound states.
        "logits near 1e30": Family(drawn(64, 64, size=1e30), _causal(64, 64), {"causal": True}, folded=False),
        "a small weight on a large value, float32": small_weight("float32"),
        "a small weight on a large value, bfloat16": small_weight("bfloat16"),
    }
    empty = np.ones((2, 1, 4, 4), bool)
    empty[1] = False
    families["a row of no keys"] = Family(drawn(4, 4, rows=2), empty, {"lengths": np.asarray([4, 0])},
                                          empty=np.asarray([[False] * 4, [True] * 4]))
    before = _causal(4, 4) & (np.arange(4) >= 2)
    families["causal queries before every key"] = Family(
        drawn(4, 4), before, {"causal": True, "starts": np.asarray([2])},
        empty=np.asarray([[True, True, False, False]]))
    ids = np.asarray([[1, 2, 3, 3]], np.int32)
    families["documents of one key"] = Family(drawn(4, 4), _causal(4, 4) & (ids[:, None, :, None] ==
                                                                            ids[:, None, None, :]),
                                              {"causal": True, "segment_ids": ids})
    return families


FAMILIES = _families()
FORMS = {"default": None, "natural": True}
"""The forms each family runs in: the default (`template.NATURAL`) and natural logits."""
RUNS = [(family, form) for family in FAMILIES for form in FORMS
        if form == "natural" or FAMILIES[family].folded]


def _spans(family: Family, query, key) -> template.KeySpans:
    arguments = {name: jax.tree.map(jnp.asarray, value) for name, value in family.spans.items()}
    return template.spans(query.shape[1], key.shape[1], query.shape[0], **arguments)


def _hooks(query, sinks) -> template.Hooks | None:
    if sinks is None:
        return None
    batch, queries, _, width = query.shape
    return template.Hooks(initial=template.sinks(sinks, batch, queries, width))


def _cast(family: Family, draw: Callable[[int], tuple]) -> Callable[[int], tuple]:
    """`draw` with its operands rounded to the family's dtype, the sinks fp32."""
    def cast(seed: int):
        query, key, value, sinks, dout = draw(seed)
        operands = (jnp.asarray(x, jnp.dtype(family.dtype)) for x in (query, key, value, dout))
        query, key, value, dout = operands
        return query, key, value, None if sinks is None else jnp.asarray(sinks), dout

    return cast


def _reference(family: Family, *, anchored: bool) -> Callable:
    """The reference's output, in the call's dtype as every attention call returns it, and its query, key
    and value gradients under the cotangent; the float64 truth's derivative `anchored`."""
    def run(query, key, value, sinks, dout):
        def attend(q, k, v):
            return exact_attention(q, k, v, sinks, keep=family.keep, anchored=anchored)
        out, pull = jax.vjp(attend, query, key, value)
        return (out.astype(query.dtype), *pull(dout.astype(out.dtype)))

    return run


def _mainloops(family: Family, form) -> dict[str, Callable]:
    """Each template mainloop as `(query, key, value, sinks, dout) -> (out, dq, dk, dv)`."""
    natural = {} if form is None else {"natural": form}
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}

    def triton(query, key, value, sinks, dout):
        spans, hooks = _spans(family, query, key), _hooks(query, sinks)
        with tuning.pinned(PINS):
            out = attention_triton.forward(query, key, value, spans, hooks, interpret=True, **natural)
            grads = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True,
                                              **natural)
        return out, grads.query, grads.key, grads.value

    def portable(query, key, value, sinks, dout):
        spans, hooks = _spans(family, query, key), _hooks(query, sinks)
        out = attention_portable.attend(query, key, value, spans, hooks, **tiles, **natural)
        grads = attention_portable.backward(query, key, value, spans, hooks, dout, **tiles, **natural)
        return out, grads.query, grads.key, grads.value

    return {"triton": triton, "portable": portable}


def _held_empty(family: Family, found, label: str) -> None:
    """A row that sees no key reads nothing and passes nothing back: its output and query gradient are
    zero exactly."""
    if family.empty is None:
        return
    rows = np.asarray(family.empty)
    for leaf, array in zip(("output", "dq"), found[:2], strict=True):
        np.testing.assert_array_equal(np.asarray(array, np.float32)[rows], 0.0, err_msg=f"{label} {leaf}")


@pytest.mark.parametrize("family, form", RUNS, ids=[f"{family}-{form}" for family, form in RUNS])
def test_attention_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family, form):
    """Each template mainloop's output and query, key and value gradients, check 1 over the family's
    draws; a row that sees nothing, zero."""
    family = FAMILIES[family]
    draw = _cast(family, family.draw)
    reference = jax.jit(_reference(family, anchored=False))
    truth = _reference(family, anchored=True)
    for name, mainloop in _mainloops(family, FORMS[form]).items():
        label = f"{name} {form} {family.dtype}"
        assert_as_exact(jax.jit(mainloop), reference, draw, label, seeds=family.seeds, truth_fn=truth)
        _held_empty(family, jax.jit(mainloop)(*draw(0)), label)


DECODED = [(family, form) for family, form in RUNS if FAMILIES[family].draw(0)[0].shape[1] == 1
           and FAMILIES[family].draw(0)[0].shape[3] >= 16]
"""The families a decode step is: one query a row, a head width the decode kernel takes (16 to 256)."""


def _cache(array: jax.Array) -> jax.Array:
    """Keys or values `[B, K, H, D]` in a cache of whole decode blocks (`decode_attention.BLOCK`, the
    capacity its geometry takes), the slots past the keys zeros."""
    slots = -array.shape[1] % decode_attention.BLOCK
    return jnp.pad(array, ((0, 0), (0, slots), (0, 0), (0, 0)))


@pytest.mark.parametrize("family, form", DECODED, ids=[f"{family}-{form}" for family, form in DECODED])
def test_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(family, form):
    """The decode kernel's output, interpreted, each row's keys the first `K` of its cache, check 1 over
    the family's draws."""
    family = FAMILIES[family]
    natural = {} if FORMS[form] is None else {"natural": FORMS[form]}
    lengths = family.spans.get("lengths")

    def decoded(query, key, value, sinks, dout):
        del dout
        count = np.full(query.shape[0], key.shape[1]) if lengths is None else lengths
        return decode_attention.attend(query, _cache(key), _cache(value), jnp.asarray(count, jnp.int32),
                                       sinks=sinks, interpret=True, **natural)

    def output(fn):
        return lambda *operands: fn(*operands)[0]

    assert_as_exact(jax.jit(decoded), jax.jit(output(_reference(family, anchored=False))),
                    _cast(family, family.draw), f"decode {form} {family.dtype}", seeds=family.seeds,
                    truth_fn=output(_reference(family, anchored=True)))
