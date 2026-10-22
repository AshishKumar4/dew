"""The attention template's stability suite (bible §6, 2026-10-19): on inputs at fp32's edges, every
attention form on every mainloop (the Triton one interpreted, the plain-JAX one, the decode kernel)
gives a finite output and finite gradients wherever the reference does, and is as exact as it, which
keeps logits it keeps apart apart. The reference is attention written apart from Dew
(`attention_oracle.exact_attention`): the softmax, then its weighted values, in fp32, and in float64 the
truth both are measured from (`reference_error.assert_as_exact_as_the_reference`, its exact limit
where the reference is exact). Each case runs with the logits natural and with the default
(`dew.nn.kernels.templates.attention.NATURAL`), which folds log2(e) into the scale where the scale
allows; the fold's documented divergences (`algebra.Forward.base2`; docs/design/kernel-templates.md, the
forms' divergences) run natural alone."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import exact_attention
from reference_error import assert_as_exact_as_the_reference

from dew.nn.kernels import decode_attention, tuning
from dew.nn.kernels.templates import attention as template, attention_portable, attention_triton

PINS = {attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2),
        attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2)}
"""The Triton mainloop's tiles: every case is a block or less, so one tile walks it."""

TIES = (1610613120.0, 1610612992.0)
"""Two logits one fp32 ulp apart at 1.6e9 (Sol's case): the reference weighs the first alone."""


@dataclasses.dataclass(frozen=True)
class Case:
    """A stability case: queries `[B, Q, H, D]`, keys and values `[B, K, H, D]` and the output cotangent,
    fp32, the softmax's scale 1/sqrt(D); the keys each query keeps `[B, 1, Q, K]` and the same keys as
    the template's spans' arguments (`template.spans`); sinks `[H]` or None; and whether the default
    form holds it (`folded`), False only for the fold's documented divergences."""

    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    dout: np.ndarray
    keep: np.ndarray
    spans: dict = dataclasses.field(default_factory=dict)
    sinks: np.ndarray | None = None
    folded: bool = True


def _along(sizes, width: int) -> np.ndarray:
    """`[1, len(sizes), 1, width]` vectors of `sizes` in their first entry and zeros past it, so a
    product of two of them is its sizes' product exactly at every width."""
    out = np.zeros((1, len(sizes), 1, width), np.float32)
    out[0, :, 0, 0] = sizes
    return out


def _tie(width: int, cotangent: float, folded: bool) -> Case:
    """One query of 1 over `TIES` scaled by sqrt(`width`), values 1 and 0: the logits are `TIES` at every
    scale."""
    root = np.sqrt(width)
    return Case(_along([1.0], width), _along([size * root for size in TIES], width),
                _along([1.0, 0.0], width), np.full((1, 1, 1, width), cotangent, np.float32),
                np.ones((1, 1, 1, 2), bool), folded=folded)


def _drawn(queries: int, keys: int, width: int, seed: int, size: float = 1.0, rows: int = 1):
    """Standard-normal queries (times `size`), keys, values and cotangent, fp32, `rows` rows, two heads."""
    streams = jax.random.split(jax.random.key(seed), 4)
    shapes = [(rows, queries, 2, width), (rows, keys, 2, width), (rows, keys, 2, width),
              (rows, queries, 2, width)]
    query, key, value, dout = (np.asarray(jax.random.normal(stream, shape), np.float32)
                               for stream, shape in zip(streams, shapes, strict=True))
    return query * np.float32(size), key, value, dout


def _causal(queries: int, keys: int) -> np.ndarray:
    return np.tril(np.ones((queries, keys), bool))[None, None]


def _cases() -> dict[str, Case]:
    ones, scalar, kept = _along([1.0], 16), _along([1.0], 1), np.ones((1, 1, 1, 1), bool)
    cases = {
        "ties at scale 1": _tie(1, 1.0, folded=True),
        "ties at scale 1, a cotangent of 2**125": _tie(1, 2.0 ** 125, folded=True),
        "ties at scale 1/2": _tie(4, 1.0, folded=False),
        "ties at scale 1/2, a cotangent of 2**125": _tie(4, 2.0 ** 125, folded=False),
        "ties at scale 1/4": _tie(16, 1.0, folded=False),
        "ties at scale 1/4, a cotangent of 2**125": _tie(16, 2.0 ** 125, folded=False),
        "a logit near fp32's largest": Case(_along([3e38], 1), scalar, scalar, scalar, kept),
        "a logit near fp32's largest at scale 1/4": Case(_along([3e38], 16), ones, ones, ones, kept),
        "values near fp32's largest over 2 equal keys": Case(
            np.zeros((1, 1, 1, 16), np.float32), np.zeros((1, 2, 1, 16), np.float32),
            np.full((1, 2, 1, 16), 3e38, np.float32), ones, np.ones((1, 1, 1, 2), bool)),
        "values near fp32's largest over 64 equal keys": Case(
            np.zeros((1, 1, 1, 16), np.float32), np.zeros((1, 64, 1, 16), np.float32),
            np.full((1, 64, 1, 16), 3e38, np.float32), ones, np.ones((1, 1, 1, 64), bool)),
        "one key": Case(*_drawn(4, 1, 16, 0), np.ones((1, 1, 4, 1), bool)),
        "a sink and one key": Case(*_drawn(1, 1, 16, 1), kept, sinks=np.asarray([2.0, -1.0], np.float32)),
        "a sink near fp32's largest and one key": Case(scalar, scalar, scalar, scalar, kept,
                                                       sinks=np.asarray([3e38], np.float32)),
        "a sink and one key, its logit near fp32's largest": Case(
            _along([3e38], 1), scalar, scalar, scalar, kept, sinks=np.asarray([2.0], np.float32)),
        "a sink near fp32's largest at scale 1/4": Case(ones, ones, ones, ones, kept,
                                                        sinks=np.asarray([3e38], np.float32), folded=False),
        "logits near 1e4": Case(*_drawn(64, 64, 16, 2, size=1e4), _causal(64, 64), {"causal": True}),
        "logits near 1e30": Case(*_drawn(64, 64, 16, 3, size=1e30), _causal(64, 64), {"causal": True}),
    }
    drawn = _drawn(4, 4, 16, 4, rows=2)
    empty = np.ones((2, 1, 4, 4), bool)
    empty[1] = False
    cases["a row of no keys"] = Case(*drawn, empty, {"lengths": np.asarray([4, 0])})
    places = np.asarray([[0, 1, 5, 6]]), np.asarray([[2, 3, 4, 5]])
    before = places[0][0][:, None] >= places[1][0][None, :]
    cases["causal queries placed before every key"] = Case(
        *_drawn(4, 4, 16, 5), before[None, None], {"causal": True, "positions": places})
    ids = np.asarray([[1, 2, 3, 3]], np.int32)
    documents = _causal(4, 4) & (ids[:, None, :, None] == ids[:, None, None, :])
    cases["documents of one key"] = Case(*_drawn(4, 4, 16, 6), documents,
                                         {"causal": True, "segment_ids": ids})
    return cases


CASES = _cases()
FORMS = {"default": None, "natural": True}
"""The forms each case runs in: the default (`template.NATURAL`) and natural logits."""
RUNS = [(case, form) for case in CASES for form in FORMS if form == "natural" or CASES[case].folded]


def _reference(case: Case, dtype):
    """The reference's output and its three input gradients under the case's cotangent, in `dtype`."""
    def run(query, key, value, dout, sinks):
        def attend(q, k, v):
            return exact_attention(q, k, v, sinks, keep=case.keep)
        out, pull = jax.vjp(attend, query, key, value)
        return [np.asarray(x) for x in (out, *pull(dout))]

    arrays = [None if x is None else jnp.asarray(np.asarray(x, dtype))
              for x in (case.query, case.key, case.value, case.dout, case.sinks)]
    return run(*arrays)


def _truth(case: Case):
    with jax.enable_x64(new_val=True):
        return _reference(case, np.float64)


def _hooks(case: Case) -> template.Hooks | None:
    if case.sinks is None:
        return None
    batch, queries, _, width = case.query.shape
    return template.Hooks(initial=template.sinks(jnp.asarray(case.sinks), batch, queries, width))


def _spans(case: Case) -> template.KeySpans:
    batch, queries = case.query.shape[:2]
    arguments = {name: jax.tree.map(jnp.asarray, value) for name, value in case.spans.items()}
    return template.spans(queries, case.key.shape[1], batch, **arguments)


def _template(case: Case, form) -> dict[str, list[np.ndarray]]:
    """Each template mainloop's output and gradients."""
    natural = {} if form is None else {"natural": form}
    query, key, value, dout = (jnp.asarray(x) for x in (case.query, case.key, case.value, case.dout))
    spans, hooks = _spans(case), _hooks(case)
    with tuning.pinned(PINS):
        out = attention_triton.forward(query, key, value, spans, hooks, interpret=True, **natural)
        grads = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True, **natural)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    portable_out = attention_portable.attend(query, key, value, spans, hooks, **tiles, **natural)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, **tiles, **natural)
    return {name: [np.asarray(x) for x in (out, found.query, found.key, found.value)]
            for name, out, found in (("triton", out, grads), ("portable", portable_out, portable))}


def _held(found, reference, truth, label: str) -> None:
    """`found` finite wherever the reference is and as exact as it: equal to float64 where the reference
    is (the rule's exact limit, `rows_over`'s for a row of float64 zero), otherwise within the rule."""
    found, reference, truth = (np.asarray(x, np.float64) for x in (found, reference, truth))
    finite = np.isfinite(reference)
    assert np.isfinite(found[finite]).all(), f"{label}: not finite where the reference is"
    if np.array_equal(reference, truth):
        np.testing.assert_array_equal(found, truth, err_msg=label)
    else:
        assert_as_exact_as_the_reference(found, reference, truth, label)


@pytest.mark.parametrize("case, form", RUNS, ids=[f"{case}-{form}" for case, form in RUNS])
def test_attention_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(case, form):
    """Each template mainloop's output and query, key and value gradients."""
    case = CASES[case]
    reference, truth = _reference(case, np.float32), _truth(case)
    for mainloop, found in _template(case, FORMS[form]).items():
        for leaf, *arrays in zip(("output", "dq", "dk", "dv"), found, reference, truth, strict=True):
            _held(*arrays, f"{mainloop} {leaf}")


DECODED = [(case, form) for case, form in RUNS
           if CASES[case].query.shape[1] == 1 and CASES[case].query.shape[3] >= 16]
"""The cases a decode step is: one query a row, a head width the decode kernel takes (16 to 256)."""


@pytest.mark.parametrize("case, form", DECODED, ids=[f"{case}-{form}" for case, form in DECODED])
def test_decode_is_finite_and_as_exact_as_the_reference_at_fp32s_edges(case, form):
    """The decode kernel's output, interpreted, each row's keys its first `K`."""
    case = CASES[case]
    natural = {} if FORMS[form] is None else {"natural": FORMS[form]}
    keys = case.key.shape[1]
    lengths = jnp.full((case.query.shape[0],), keys, jnp.int32)
    found = decode_attention.attend(jnp.asarray(case.query), jnp.asarray(case.key), jnp.asarray(case.value),
                                    lengths, sinks=None if case.sinks is None else jnp.asarray(case.sinks),
                                    interpret=True, **natural)
    _held(found, _reference(case, np.float32)[0], _truth(case)[0], "decode output")
