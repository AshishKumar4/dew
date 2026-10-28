"""The attention template's Mosaic GPU mainloop on the wgmma family
(dew.nn.kernels.templates.attention_mosaic): its forward and its backward in both forms, check 1
(`reference_error.assert_as_exact`) against the template's reference over ordinary draws, under Mosaic GPU's
interpreter on the CPU (as an H100's launches, `generation.assumed`) and compiled on an H100. The interpreter
is coverage, not proof (it treats wgmma_wait as a no-op and takes the register operand's product as a dot);
the GPU gate runs the compiled kernels. Its backward reads its own training forward's residuals (its split
sums), and the Triton mainloop's, as the route feeds them. Its configs are its Template implementations'
(`MosaicForward`, `MosaicBackward`): generated from the family's facts and the call's extents and fitted to
the device's, a site pinning one where it names a schedule (`tuning.pinned`)."""

import contextlib
import dataclasses
import functools
import os

import lane_environment
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, lowers
from jax.sharding import Mesh
from reference_error import as_given, assert_as_exact
from test_attention_template import CASES, ROWS, RUN_CASES, arguments, geometry as core_geometry
from test_attention_triton import BACKWARD_SEEDS, HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_mosaic_program,
    attention_triton,
)

pytestmark = pytest.mark.kernel

ON_HOPPER = pytest.mark.skipif(generation.device_generation() != "sm90",
                               reason="runs the sm90 mainloop compiled, on an H100")
CALLS = {"full": {}, "causal": {"causal": True}}
"""The call forms the mainloop takes."""
SHAPES = {"interpreted": (1, 160, 2, 1), "compiled": (2, 400, 4, 2)}
"""Rows, tokens (not whole tiles, so the call pads, over more than one key tile past the keys where a
program's two consumers pad it to 128), query and key heads (grouped), interpreted on the CPU and compiled
on an H100."""
SEEDS = 8
"""Draws a case takes, check 1's fewest (`reference_error.seeds_for`): it bounds the worst row over them
against the reference's own."""
ALL_HOOKS_SEEDS = 32
"""The draws of a site with every hook at once (`test_attention_triton.ALL_HOOKS_SEEDS`)."""
BACKWARD_HOOKS = list(SCORE_HOOKS)
"""The backward's score hooks: every one the backward differentiates (`test_attention_triton.SCORE_HOOKS`),
the hook indexed off the query's coordinates' column, which forms a unit axis, among them (ledger 1428)."""
FORWARD_HOOKS = {**{name: (SCORE_HOOKS[name], False) for name in BACKWARD_HOOKS},
                 "gate": ((), True),
                 "all": (("softcap", "custom", "sinks", "alibi", "temperature", "key", "tile"), True)}
"""The forward's hooks: each score hook the backward differentiates (`test_attention_triton.SCORE_HOOKS`:
every kind of parameter, the initial state as sinks or a resumed partial, a hook through a custom VJP), an
output gate, and all at once; each `(score hooks, gate)`."""
HOOK_FORMS = {"bfloat16": (jnp.bfloat16, False), "float16": (jnp.float16, False),
              "bfloat16-natural": (jnp.bfloat16, True), "float16-natural": (jnp.float16, True)}
"""The hooked forward's dtypes and forms: the fold and natural logits over both 16-bit dtypes."""
BACKWARD_FORMS = {"default": (False, attention_mosaic.BackwardTiles(64, 64, 2, 2)),
                  "fused": (False, attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True)),
                  "accurate": (True, attention_mosaic.BackwardTiles(64, 64, 2, 1))}
"""The backward's forms and schedules, `(accurate, config)`: the fast form in two launches of two consumers
and in one (the dK/dV program adding dQ), and the accurate form, one consumer's (its keyed sums beside dQ's
hold a consumer's registers, `MosaicBackward.register_budget`)."""
FORM_TILES = attention_mosaic.Tiles(64, 64, 2, 2)
"""The call forms' tiles: the template core's 100 queries over 96 keys (`test_attention_template.CASES`) in a
pair of query blocks over two key tiles, so a block's spans begin and end within the keys it streams."""
HEAD_WIDTHS = ((64, 128), (128, 64), (80, 48))
"""The (query, value) head widths the forward is checked at: a value wider than its query, a narrower one, and
both off a power of two (80 and 48 pad to 128 and 64, `attention.padded_call`)."""


def _drawn(shape: tuple[int, int, int, int], dtype, width: int = 128, keys: int | None = None,
           value_width: int | None = None):
    """A draw `(query, key, value, dout)` of unit Gaussians at `shape` (its keys `keys`, else its tokens) in
    `dtype`, by seed: the query and key `width` wide, the value and its cotangent `value_width` (`width` where
    None)."""
    rows, tokens, heads, kv_heads = shape
    keys = tokens if keys is None else keys
    value_width = width if value_width is None else value_width

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        dims = ((heads, tokens, width), (kv_heads, keys, width), (kv_heads, keys, value_width),
                (heads, tokens, value_width))
        return tuple(jax.random.normal(stream, (rows, count, heads_, depth), jnp.float32).astype(dtype)
                     for stream, (heads_, count, depth) in zip(streams, dims, strict=True))

    return draw


def _device(interpret: bool, target: generation.GpuTarget | None = None):
    """The device a site's launches are sized and run for: interpreted on the CPU, which has no instruction
    family of its own, `target`'s facts (an H100's where None, `generation.assumed`); the process's own
    compiled."""
    return generation.assumed(target or "sm90") if interpret else contextlib.nullcontext()


def _pinned(forward: attention_mosaic.Tiles | None = None,
            backward: attention_mosaic.BackwardTiles | None = None):
    """The launches pinned to `forward`'s and `backward`'s configs where given (`tuning.pinned`), each
    else at the cost model's pick."""
    pins = {attention_mosaic.MosaicForward: forward, attention_mosaic.MosaicBackward: backward}
    return tuning.pinned({op: config for op, config in pins.items() if config is not None})


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool, width: int = 128,
                   value_width: int = 128, tiles: attention_mosaic.Tiles | None = None,
                   training: bool = False, target: generation.GpuTarget | None = None):
    """The forward, or with `training` the training forward's output, over the call form `call` of `shape`,
    its query and key heads `width` wide and its value heads `value_width`, at `tiles` (the default when
    None)."""
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    draw = _drawn(shape, dtype, width=width, value_width=value_width)

    def ours(query, key, value, dout):
        del dout
        with _device(interpret, target), _pinned(tiles):
            out = attention_mosaic.forward(query, key, value, spans, natural=natural, residuals=training,
                                           interpret=interpret)
        return out[0] if training else out

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic {'training ' if training else ''}forward {call} {jnp.dtype(dtype).name} "
                    f"natural {natural} widths {width} {value_width}", seeds=SEEDS)


def _check_spans(label: str, spans: template.KeySpans, shape, *, interpret: bool, backward: bool = False,
                 form: str = "default", score: tuple = (), gate: bool = False, dtype=jnp.bfloat16,
                 natural: bool = False, seeds: int | None = None):
    """The forward, or the backward in the form `form` names (`BACKWARD_FORMS`), over `spans` at `FORM_TILES`
    under the score hooks `score` (`test_attention_triton.score_hooks`, their parameters and
    initial state drawn with the operands) and with `gate` an output gate (`hooks_for`), draws of `shape`
    (rows, queries, query and key heads, four and two under a hook) over the spans' keys in `dtype`, the
    forward's logits `natural` or folded. The backward differentiates the queries, keys, values, the hook's
    parameters and the initial state, over `seeds` draws where given (the site's own, else the template
    site's)."""
    rows, queries, heads, _ = shape
    score_mod, kinds, draw_hooks = score_hooks(score, queries, spans.keys, 64, rows=rows, heads=heads)
    output = hooks_for(HOOKS["gate" if gate else "none"], queries, 64, gate_of(queries, 64))
    draw_operands = _drawn(shape, dtype, width=64, keys=spans.keys)

    def draw(seed):
        return *draw_operands(seed), *draw_hooks(seed)

    def hooked(params, initial):
        return template.Hooks(score_mod, tuple(template.Param(value, kind)
                                               for value, kind in zip(params, kinds, strict=True)), initial,
                              output.output_mod, output.output_params)

    accurate, config = BACKWARD_FORMS[form]

    def ours(query, key, value, dout, params, initial):
        hooks = hooked(params, initial)
        with _device(interpret), _pinned(FORM_TILES, config):
            if not backward:
                return attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                interpret=interpret)
            # The backward from this mainloop's own training forward's residuals.
            residuals = attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                                 residuals=True, interpret=interpret)[1:]
            grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                              accurate=accurate, interpret=interpret)
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def reference(query, key, value, dout, params, initial):
        if not backward:
            return template.reference(query, key, value, spans, hooked(params, initial))

        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, hooked(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, initial)
        return pull(dout)

    name = (f"mosaic {'backward' if backward else 'forward'} {label} {score}{' gate' if gate else ''} "
            f"{jnp.dtype(dtype).name}{f' {form}' if backward else ''}{' natural' if natural else ''}")
    if seeds is None:
        seeds = (BACKWARD_SEEDS.get((score, jnp.dtype(dtype).name), SEEDS) if backward
                 else ALL_HOOKS_SEEDS if len(score) + gate > 1 else SEEDS)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, name, seeds=seeds)


def _check_hooks(hooks: str, dtype, natural: bool, *, interpret: bool):
    """The forward under the hooks `FORWARD_HOOKS[hooks]` over the template core's causal packed documents
    under a window (`test_attention_triton`'s hook site)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    score, gate = FORWARD_HOOKS[hooks]
    _check_spans("causal-documents-window", spans, (ROWS, queries, 4, 2), interpret=interpret, score=score,
                 gate=gate, dtype=dtype, natural=natural)


INTERPRETED_HOOK_SITES = {"none": ((1, 2, 1), 8), "alibi": ((1, 2, 1), 128), "key-bias": ((1, 2, 1), 16),
                          "custom-vjp": ((1, 2, 1), 8), "softcap": ((1, 2, 1), 8), "sinks": ((1, 4, 2), 16),
                          "temperature": ((1, 2, 1), 8), "tile-bias": ((1, 2, 1), 8),
                          "resumed": ((1, 2, 1), 128), "softcap-sinks-alibi": ((2, 2, 1), 128),
                          "row-indexed": ((1, 2, 1), 8)}
"""The interpreted hooked backward's site under each hook: its rows, query heads and key heads, and its
seeds, sized by `seeds_for` at that shape (the reference against itself), never the compiled site's. The
claim interpreted is the hooked backward lowering and computing the template's gradients, so each site
keeps its paths: the first row's three documents and its padding, their boundaries inside key blocks, two
key blocks a query, more query heads than key heads, and the hook's every mode. Of the shapes that keep
them, each takes the fewest seeds times the interpreter's seconds a seed (a row of two heads 1.9 s, of four
3.8 s, two rows of four 7.5 s): a head's ALiBi slope or sink is one value a draw, so those hooks' seeds
grow as heads shrink. The H100 site (`_check_hooked_backward`, compiled) keeps the template site's shape
and seeds, at full resolution."""


def _check_hooked_backward(hooks: str, *, form: str, interpret: bool):
    """The backward in the form `form` names (`BACKWARD_FORMS`) under the score hook `SCORE_HOOKS[hooks]` over
    the template core's causal packed documents under a window (`_check_hooks`' site): the gradients of the
    queries, keys, values, of the hook's parameters and of the initial state. Interpreted, at the hook's own
    site (`INTERPRETED_HOOK_SITES`)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    (rows, heads, kv_heads), seeds = INTERPRETED_HOOK_SITES[hooks] if interpret else ((ROWS, 4, 2), None)
    # The call's documents of its first `rows` rows.
    called = {name: value[:rows] if hasattr(value, "shape") else value
              for name, value in arguments(case).items()}
    spans = template.spans(queries, keys, rows, **called)
    _check_spans("causal-documents-window", spans, (rows, queries, heads, kv_heads), interpret=interpret,
                 backward=True, form=form, score=SCORE_HOOKS[hooks], seeds=seeds)


def _check_call_form(case: str, *, interpret: bool, backward: bool = False, form: str = "default"):
    """`_check_spans` over the template core's call form `case` (`test_attention_template.CASES`, or its
    calls over runs, `RUN_CASES`), two query heads a key head, the backward in the form `form` names."""
    call = CASES[case] if case in CASES else RUN_CASES[case]
    queries, keys = core_geometry(call)
    spans = template.spans(queries, keys, ROWS, **arguments(call))
    _check_spans(case, spans, (ROWS, queries, 4, 2), interpret=interpret, backward=backward, form=form)


def _check_backward(call: str, shape, *, form: str | None, interpret: bool, keys: int | None = None,
                    residuals_from=None, target: generation.GpuTarget | None = None,
                    config: attention_mosaic.BackwardTiles | None = None):
    """The backward in the form `form` names (`BACKWARD_FORMS`; None, the default form at `config`, else its
    derived one) over the call form `call` of `shape` (its keys `keys`) on `target` (`_device`), from the
    training forward's residuals at its pick: this mainloop's, or `residuals_from`'s (another mainloop's
    `forward`, as the route feeds the Triton mainloop's)."""
    keys = shape[1] if keys is None else keys
    spans = template.spans(shape[1], keys, shape[0], **CALLS[call])
    draw = _drawn(shape, jnp.bfloat16, keys=keys)
    accurate, config = (False, config) if form is None else BACKWARD_FORMS[form]
    trained = attention_mosaic.forward if residuals_from is None else residuals_from

    def ours(query, key, value, dout):
        with _device(interpret, target):
            residuals = trained(query, key, value, spans, residuals=True, interpret=interpret)[1:]
            with _pinned(backward=config):
                grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                                  accurate=accurate, interpret=interpret)
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"mosaic backward {call} {form}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("call", CALLS)
def test_the_training_forward_is_on_par_interpreted(call, natural):
    """The training forward's output, bf16, under Mosaic GPU's interpreter: the weighted values it
    normalizes are its split sums joined with the anchor's term (`algebra.split_output`), the anchor's value
    row the one it kept; its residuals are the backward sites' (`_check_backward`)."""
    _check_forward(call, jnp.bfloat16, SHAPES["interpreted"], natural=natural, interpret=True, training=True)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("call", CALLS)
def test_the_training_forward_is_on_par_on_an_h100(call):
    """The training forward's output, bf16, compiled on an H100."""
    _check_forward(call, jnp.bfloat16, SHAPES["compiled"], natural=False, interpret=False, training=True)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_interpreted(call, dtype):
    """The forward with natural logits under the interpreter, bf16's under its headroom guard
    (`algebra.walked_launches`)."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=True, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("form", CASES)
def test_the_forward_is_on_par_over_every_call_form_interpreted(form):
    """The forward over each of the template core's call forms under the interpreter."""
    _check_call_form(form, interpret=True)


@INTERPRETS
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_forward_is_on_par_at_head_widths_interpreted(width, value_width):
    """The forward of query heads `width` wide and value heads `value_width` over a causal call at
    `FORM_TILES`, under the interpreter."""
    _check_forward("causal", jnp.bfloat16, SHAPES["interpreted"], natural=False, interpret=True, width=width,
                   value_width=value_width, tiles=FORM_TILES)


@INTERPRETS
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def test_the_forward_is_on_par_under_every_hook_interpreted(hooks, dtype, natural):
    """The forward under each hook (`FORWARD_HOOKS`) over causal packed documents under a window, under the
    interpreter, in each of `HOOK_FORMS`."""
    _check_hooks(hooks, dtype, natural, interpret=True)


PADDED_CALLS = {"one-query-default": (1, "default"), "one-query-accurate": (1, "accurate"),
                "whole-padding-tile": (64, "default")}
"""The padded calls' query counts and backward forms (`BACKWARD_FORMS`): one query, its tile mostly padding,
in both forms; and a tile's worth of queries under two consumers, which pads to a pair of query tiles, its
second wholly padding (the accurate form's one consumer pads 64 queries to no more)."""


@INTERPRETS
@pytest.mark.parametrize(("queries", "form"), PADDED_CALLS.values(), ids=PADDED_CALLS)
def test_padding_rows_reach_no_hook_gradient_interpreted(queries, form):
    """A call of `queries` queries over as many keys, padded to whole programs, under a hook dividing by a
    query's parameter and adding a head's: the padding rows' parameter is 0, so the hook is 0/0 there, and a
    padding row that saw a key would carry its NaN into the values' gradient and the head's, a sum over the
    rows. Padding rows see no key, a tile's last ones or a whole tile of them (`algebra.Forward.queries`),
    and every gradient is the reference's, the head's 0."""
    spans = template.spans(queries, queries, 1)
    query, key, value = (jnp.zeros((1, queries, 1, 64), jnp.bfloat16) for _ in range(3))
    dout = jnp.ones((1, queries, 1, 64), jnp.bfloat16)
    accurate, config = BACKWARD_FORMS[form]

    def score_mod(s, q_pos, k_pos, temperature, bias):
        del q_pos, k_pos
        return s / temperature + bias

    def hooked(temperature, bias):
        return template.Hooks(score_mod, (template.Param(temperature, template.Kind.QUERY),
                                          template.Param(bias, template.Kind.HEAD)))

    params = (jnp.ones((1, 1, queries), jnp.float32), jnp.zeros((1,), jnp.float32))
    hooks = hooked(*params)
    with _device(interpret=True), _pinned(backward=config):
        residuals = attention_mosaic.forward(query, key, value, spans, hooks, residuals=True,
                                             interpret=True)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                          accurate=accurate, interpret=True)
    _, pull = jax.vjp(lambda q, k, v, t, b: template.reference(q, k, v, spans, hooked(t, b)), query, key,
                      value, *params)
    for found, expected in zip((grads.query, grads.key, grads.value, *grads.score_params), pull(dout),
                               strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


@INTERPRETS
def test_the_training_forwards_overflowed_rows_keep_their_form_and_every_other_row_interpreted():
    """A call as many queries as heads, one consumer's tile of them, whose head 0 sees two keys of equal
    logits past the fold's reach over values near bf16's largest: the call walks again unfolded, and head
    0's rows, whose weighted values overflow, take the guarded walk in that form (`algebra.unfolded`), its
    residuals in its units. Head 0's output is its values' mean exactly, its one value row (where the
    reference's fp32 sums overflow); every other head's rows are the plain launch's bits, as where head 0's
    values do not overflow; and the backward from these residuals gives head 0's values a gradient wherever
    the reference does, finite where it is."""
    queries, keys, width = 64, 2, 64
    spans = template.spans(queries, keys, 1)
    streams = jax.random.split(jax.random.key(0), 4)
    query, key, value, dout = (jax.random.normal(stream, (1, count, queries, width), jnp.float32)
                               for stream, count in zip(streams, (queries, keys, keys, queries), strict=True))
    # Head 0's logits, 128 * 128 * 64 / 8, past the fold's reach; its values overflow a sum of two or not.
    query, key = (array.at[:, :, 0].set(128.0) for array in (query, key))
    query, key, dout = (array.astype(jnp.bfloat16) for array in (query, key, dout))
    tiles = attention_mosaic.Tiles(64, 64, 2, 1)

    def trained(value):
        with _device(interpret=True), _pinned(tiles):
            return attention_mosaic.forward(query, key, value, spans, residuals=True, interpret=True)

    largest = value.at[:, :, 0].set(3e38).astype(jnp.bfloat16)
    out, *residuals = trained(largest)
    ordinary_out, *ordinary = trained(value.at[:, :, 0].set(1.0).astype(jnp.bfloat16))
    head, row = np.asarray(out[:, :, 0], np.float32), np.asarray(largest[:, :1, 0], np.float32)
    np.testing.assert_array_equal(head, np.broadcast_to(row, head.shape))
    np.testing.assert_array_equal(np.asarray(out[:, :, 1:]), np.asarray(ordinary_out[:, :, 1:]))
    for found, plain in zip(residuals[:3], ordinary[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(found[:, 1:]), np.asarray(plain[:, 1:]))
    with _device(interpret=True):
        grads = attention_mosaic.backward(query, key, largest, spans, None, dout, residuals=tuple(residuals),
                                          interpret=True)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, largest)
    reference_value = np.asarray(pull(dout)[2], np.float32)
    found_value = np.asarray(grads.value, np.float32)
    finite = np.isfinite(reference_value)
    assert np.isfinite(found_value[finite]).all()
    assert (found_value[finite & (reference_value != 0)] != 0).all()


UNSEEN = {
    "hopper-class": generation.GpuTarget("sm90", "Unmeasured Hopper-class GPU", shared_limit=180 * 1024,
                                         multiprocessors=78, clock_hz=1.6e9, memory_bandwidth=2.4e12),
}
"""A GPU no row measured of this mainloop's family (`test_grouped_gemm.UNSEEN`'s Hopper-class one): fewer
multiprocessors, a smaller block and another L2 bandwidth than an H100's."""


def _bound(op, *arrays, **options):
    """`op`'s arguments for a call of `arrays` (`forward`'s, or `backward`'s with its residuals), as its
    launch binds them compiled, as a GPU process runs it (`MosaicForward.eligible` takes an interpreted call
    in a CPU process alone)."""
    return op.bind(*arrays, interpret=False, **options)


def _forward_call(shape, width: int, value_width: int, residuals: bool):
    """A causal forward's op and bound arguments over `shape`'s operands, its heads `width` and its values
    `value_width` wide."""
    rows, tokens, heads, kv_heads = shape
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    value = jax.ShapeDtypeStruct((rows, tokens, kv_heads, value_width), jnp.bfloat16)
    op = attention_mosaic.MosaicForward(form="causal", keys=tokens, scale=width ** -0.5)
    return op, _bound(op, query, key, value, spans.start, spans.end, None, None, (), (), None,
                      residuals=residuals)


@pytest.mark.parametrize("residuals", [False, True], ids=["forward", "training"])
@pytest.mark.parametrize(("width", "value_width"), [(128, 64), (128, 128), (128, 256), (256, 256)])
def test_the_derived_tiles_stage_within_a_hoppers_block(width, value_width, residuals):
    """On an H100 the forward's tiles derived from the facts (`MosaicForward.derived`) stage within its
    block (`generation.shared_budget`) and a consumer's registers (`register_budget`), whatever the widths,
    the training forward's too (its anchored walk keeps no value row): no table's tile, which at heads 256
    wide staged 320 KiB against its 227 (ledger 1582)."""
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _forward_call((2, 4096, 32, 8), width, value_width, residuals)
        derived = op.derived(ba, target)
        assert derived is not None
        assert op.legal(ba, derived, target)
        assert op.staged_bytes(ba, derived, target) <= generation.shared_budget(target)
        assert op.registers(ba, derived) <= op.register_budget(derived)


def _backward_call(shape, width: int, accurate: bool):
    """A causal backward's op and bound arguments over `shape`'s operands, its heads `width` wide, with a
    training forward's residuals."""
    rows, tokens, heads, kv_heads = shape
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    stats = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    residuals = (stats, stats, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32),
                 template.Split(jax.ShapeDtypeStruct((rows, heads, tokens, width), jnp.float32), stats, stats,
                                stats))
    op = attention_mosaic.MosaicBackward(form="causal", keys=tokens, scale=width ** -0.5, accurate=accurate)
    return op, _bound(op, query, key, key, query, spans.start, spans.end, None, None, (), None, residuals)


@pytest.mark.parametrize("call, fused", [("fast", True), ("accurate", False), ("deterministic", False)])
def test_the_fast_forms_backward_is_one_launch_and_deterministic_mode_two(call, fused, monkeypatch):
    """The backward derived on an H100 is one launch, the dK/dV program adding each tile's dQ atomically, for
    a hookless fast-form call (FUSEDFORM-h100's speed, FUSEDSTAB-h100's check 1 and finiteness); the accurate
    form and deterministic mode take two, the dQ launch then the dK/dV one, in a fixed order."""
    flags = lane_environment.without_deterministic_ops(os.environ.get("XLA_FLAGS", ""))
    if call == "deterministic":
        flags = f"{flags} --xla_gpu_deterministic_ops=true"
    monkeypatch.setenv("XLA_FLAGS", flags)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _backward_call((2, 4096, 32, 8), 128, accurate=call == "accurate")
        derived = op.derived(ba, target)
        assert derived is not None and derived.fused == fused
        assert op.legal(ba, derived, target)


def _softcapped(s, q_pos, k_pos):
    """A score hook a tuning run keys by what it computes (`AttentionLaunch.HOOKS`)."""
    del q_pos, k_pos
    return 30.0 * jnp.tanh(s / 30.0)


def _gated(o):
    """An output hook a tuning run keys by what it computes."""
    return o * 0.5


@pytest.mark.parametrize("launch", ["forward", "backward"])
def test_a_hooked_launchs_keyed_configs_are_its_derived_ones(launch):
    """A hooked launch as its picks key it (`tuning.KernelOp.keyed`: each hook its fingerprint, which a
    stored pick and a tuning run look its configs up by) gives the configs its live call does: its
    legality reads the fingerprint's lowering as the live call established it, never traces the string."""
    rows, tokens, heads, kv_heads = SHAPES["interpreted"]
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    with generation.assumed("sm90"):
        if launch == "forward":
            op = attention_mosaic.MosaicForward(score_mod=_softcapped, output_mod=_gated, form="causal",
                                                keys=tokens, scale=128 ** -0.5)
            ba = _bound(op, query, key, key, spans.start, spans.end, None, None, (), (), None,
                        residuals=False)
        else:
            stats = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
            residuals = (stats, stats, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32),
                         template.Split(jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32), stats,
                                        stats, stats))
            op = attention_mosaic.MosaicBackward(score_mod=_softcapped, form="causal", keys=tokens,
                                                 scale=128 ** -0.5)
            ba = _bound(op, query, key, key, query, spans.start, spans.end, None, None, (), None, residuals)
        keyed = op.keyed(ba)
        assert isinstance(keyed.op.score_mod, str)
        assert set(keyed.autotuning_configs) == set(ba.autotuning_configs)
        assert keyed.heuristics_config == ba.heuristics_config


def test_a_pin_the_mosaic_program_cannot_take_is_illegal():
    """A config a pin or a stored pick names is held to the tiles a program takes (`whole_tiles`) as a
    derived one is: a 32-query tile under wgmma's 64-row roles, or three consumers where the family's walks
    take two, is not legal, so a pin of it is refused before any launch."""
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        op, ba = _forward_call(SHAPES["interpreted"], 128, 128, residuals=False)
        derived = op.derived(ba, target)
        assert derived is not None and op.legal(ba, derived, target)
        rows, tokens, _, _ = SHAPES["interpreted"]
        arrays = [jnp.zeros(ba.arguments[name].shape, ba.arguments[name].dtype)
                  for name in ("query", "key", "value")]
        for config in (dataclasses.replace(derived, block_q=32), dataclasses.replace(derived, consumers=3)):
            assert not op.legal(ba, config, target)
            with pytest.raises(tuning.Unpinnable), _pinned(config):
                attention_mosaic.forward(*arrays, template.spans(tokens, tokens, rows, causal=True),
                                         interpret=True)


@INTERPRETS
@pytest.mark.parametrize("launch", ["forward", "training", "backward"])
def test_every_tuning_variant_is_on_par_interpreted(launch):
    """Each config an opt-in tuning run may time on an H100 (`Template.variants`: the derived one and its
    legal neighbours, `_Mainloop.neighbours`) holds check 1 on the interpreted causal call: a tuner picks
    among kernels the gate already holds."""
    shape = SHAPES["interpreted"]
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        if launch == "backward":
            op, ba = _backward_call(shape, 128, accurate=False)
        else:
            op, ba = _forward_call(shape, 128, 128, residuals=launch == "training")
        variants = op.variants(ba)
    assert len(variants) > 1
    for config in variants:
        if launch == "backward":
            _check_backward("causal", shape, form=None, interpret=True, config=config, target=target)
        else:
            _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, tiles=config,
                           training=launch == "training", target=target)


@INTERPRETS
@pytest.mark.parametrize("tiles", [attention_mosaic.Tiles(64, 128, 2, 2),
                                   attention_mosaic.Tiles(64, 192, 2, 1)], ids=["128-keys", "192-keys"])
def test_a_guarded_launch_walks_its_own_key_tiles(tiles):
    """A spans call (four documents over 256 keys, their ids from 1: 0 is padding) whose weighted values
    pass fp32's largest, at forward tiles whose consumers' registers hold the guarded launch's live set
    over a narrower key tile (`guarded_tiles`), 128 keys halved, 192 keys cut to a whole tile of the
    family's rows that divides it: the guarded launch runs on tiles its program takes, reads each block's
    keys from its own block map, not the plain launch's, and every row's output is on par with jax's own
    attention, which normalizes its weights before their product (bf16 values near their largest, equal
    logits)."""
    rows, queries, heads, kv_heads = 1, 256, 2, 1
    documents = jnp.repeat(jnp.arange(1, 5, dtype=jnp.int32), queries // 4)[None]
    spans = template.spans(queries, queries, rows, segment_ids=documents)
    keep = (documents[:, :, None] == documents[:, None, :])[:, None]

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jnp.zeros((rows, queries, heads, 128), jnp.bfloat16)
        key = jnp.zeros((rows, queries, kv_heads, 128), jnp.bfloat16)
        value = jax.random.uniform(streams[2], (rows, queries, kv_heads, 128), jnp.float32, 2.9e38,
                                   3.3e38).astype(jnp.bfloat16)
        return query, key, value

    def ours(query, key, value):
        with _device(interpret=True), _pinned(tiles):
            return attention_mosaic.forward(query, key, value, spans, interpret=True)

    def reference(query, key, value):
        return jax.nn.dot_product_attention(query, key, value, mask=keep, implementation="xla")

    assert_as_exact(jax.jit(ours), as_given(jax.jit(reference)), draw, "mosaic guarded spans forward",
                    seeds=SEEDS)


@pytest.mark.parametrize("unseen", UNSEEN)
def test_a_gpu_no_one_measured_runs_the_tiles_its_facts_give(unseen):
    """On a GPU no row measured the forward's and the backward's tiles come from the family's facts and the
    call's extents and fit the GPU's block, and the forward, the training forward and the backward at the
    tiles derived from its facts alone are on par interpreted."""
    target = UNSEEN[unseen]
    shape = (1, 160, 2, 1)
    with generation.assumed(target):
        op, ba = _forward_call(shape, 128, 128, residuals=True)
        derived = op.derived(ba, target)
        assert derived is not None and derived.family == "wgmma"
        assert op.staged_bytes(ba, derived, target) <= 180 * 1024
    _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, target=target)
    _check_forward("causal", jnp.bfloat16, shape, natural=False, interpret=True, training=True, target=target)
    _check_backward("causal", shape, form=None, interpret=True, target=target)


@INTERPRETS
@pytest.mark.parametrize("form", ["default", "accurate"])
def test_the_backward_is_on_par_from_the_triton_mainloops_residuals_interpreted(form):
    """The backward from the Triton mainloop's training forward's residuals, as the route feeds them on a GPU
    (`attention_route`), under the interpreter."""
    _check_backward("causal", SHAPES["interpreted"], form=form, interpret=True,
                    residuals_from=attention_triton.forward)


@INTERPRETS
def test_a_hook_shaped_to_the_launchs_tile_is_taken():
    """An output hook that reshapes its gate to the value width, on the launch's `[64, 128]` output tile as
    on the reference's rows: the forward is on par under it."""
    queries = 128
    spans = template.spans(queries, queries, 1, causal=True)
    draw_operands = _drawn((1, queries, 2, 1), jnp.bfloat16, width=128, keys=queries)

    def draw(seed):
        query, key, value, _ = draw_operands(seed)
        gate = jax.random.normal(jax.random.PRNGKey(seed), (1, queries, 2, 128), jnp.float32)
        return query, key, value, gate

    def hooked(gate):
        return template.Hooks(output_mod=lambda out, g: out * jax.nn.sigmoid(g.reshape((-1, 128))),
                              output_params=(template.Param(gate, template.Kind.OUTPUT),))

    def ours(query, key, value, gate):
        with _device(interpret=True):
            return attention_mosaic.forward(query, key, value, spans, hooked(gate), interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, hooked(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward, a gate shaped to the tile",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("case", CASES)
def test_the_backward_is_on_par_over_every_call_form_interpreted(case, form):
    """The backward in both forms over each of the template core's call forms under the interpreter."""
    _check_call_form(case, interpret=True, backward=True, form=form)


@INTERPRETS
@pytest.mark.parametrize("form", ["forward", *BACKWARD_FORMS])
@pytest.mark.parametrize("case", RUN_CASES)
def test_a_call_over_runs_is_on_par_interpreted(case, form):
    """A decision row's runs (`test_attention_template.RUN_CASES`), several key intervals a query, under the
    interpreter: each block walks its runs' envelope (`attention.block_map`), each tile masked by the union
    of its queries' runs (`algebra.seen`), the forward and the backward in each form."""
    backward = form != "forward"
    _check_call_form(case, interpret=True, backward=backward, form=form if backward else "default")


@INTERPRETS
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize("call", CALLS)
def test_a_call_of_whole_key_tiles_reads_no_padding_interpreted(call, backward):
    """64 keys under 128 queries at 64-key tiles: the call pads its keys to a pair's 128, a tile wholly past
    them, which a causal query past them reaches too; the keys are whole tiles, so no tile the walk reads
    takes the keys' bound (`algebra.seen`): the walk reads none wholly past them."""
    spans = template.spans(128, 64, 1, **CALLS[call])
    _check_spans(f"{call} of whole key tiles", spans, (1, 128, 2, 1), interpret=True, backward=backward)


@INTERPRETS
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, form):
    """The backward in both forms under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], form=form, interpret=True)


@INTERPRETS
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_interpreted():
    """A causal call of 64 queries over 256 keys, whose last key tiles no query sees: the fast form's two
    consumers each take a key block, one wholly past the queries' diagonal, and walk no query tile, the
    producer loading none for them."""
    for form in ("default", "fused"):
        _check_backward("causal", (1, 64, 2, 1), form=form, interpret=True, keys=256)


HOPPER_HOOK_SITES = ("alibi", "resumed", "softcap-sinks-alibi")
"""The hooked backward's sites the H100 alone runs (`…_under_every_score_hook_on_an_h100`, compiled, at the
template site's shape and seeds): a head's ALiBi slope or a resumed state is one value a draw, so check 1
takes 128 draws at any head count, past the CPU suite's 240 s target interpreted (1.9 s a draw a row of two
heads: 270 to 650 s, row 1849); the CPU keeps every other hook, sinks and softcap among them."""


@INTERPRETS
@pytest.mark.parametrize("form", ["default", "accurate"])
@pytest.mark.parametrize("hooks", [hooks for hooks in BACKWARD_HOOKS if hooks not in HOPPER_HOOK_SITES])
def test_the_backward_is_on_par_under_every_score_hook_interpreted(hooks, form):
    """The backward in both forms under each score hook over causal packed documents under a window, under
    the interpreter: the gradients of the queries, keys and values, of each hook parameter and of the initial
    state."""
    _check_hooked_backward(hooks, form=form, interpret=True)


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", ["causal", "causal-documents-window", "runs-documents-valid"])
def test_the_launches_lower_for_an_h100(call, accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them: a
    causal call, a spans one of documents, whose programs read its block map and ids, and one over a
    decision row's runs, whose programs read each run's bounds; the forward under every hook too, whose
    programs read each kind of parameter and the initial state."""
    if call == "causal":
        rows, tokens, heads, kv_heads = SHAPES["compiled"]
        spans = template.spans(tokens, tokens, rows, causal=True)
    else:
        case = CASES[call] if call in CASES else RUN_CASES[call]
        (tokens, _), rows, heads, kv_heads = core_geometry(case), ROWS, 4, 2
        spans = template.spans(tokens, tokens, rows, **arguments(case))
    q = jax.ShapeDtypeStruct((rows, tokens, heads, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, 128), jnp.bfloat16)
    rows_shape = jax.ShapeDtypeStruct((rows, heads, tokens), jnp.float32)
    others = jax.ShapeDtypeStruct((rows, heads, tokens, 128), jnp.float32)
    split = template.Split(others, rows_shape, rows_shape, rows_shape)
    residuals = (rows_shape, rows_shape, jax.ShapeDtypeStruct((rows, heads, tokens), jnp.int32), split)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    score_mod, kinds, draw_hooks = score_hooks(FORWARD_HOOKS["all"][0], tokens, tokens, 128)
    params, initial = draw_hooks(0)
    gate = hooks_for(HOOKS["gate"], tokens, 128, gate_of(tokens, 128))
    hooks = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                            for value, kind in zip(params, kinds, strict=True)), initial,
                           gate.output_mod, gate.output_params)
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        for forward_hooks in (None, hooks):
            for training in (False, True):
                entry = functools.partial(attention_mosaic.forward, spans=spans, hooks=forward_hooks,
                                          residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
            q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize(("hooks", "approximated"), [("none", True), ("alibi", True), ("sinks", True),
                                                     ("temperature", True), ("softcap", False),
                                                     ("custom-vjp", False), ("softcap-sinks-alibi", False)])
def test_a_launch_takes_approximate_math_only_where_no_hook_computes_a_transcendental(hooks, approximated):
    """The hardware's approximate transcendentals (`approx_math`) hold the mainloop's own exponentials on par;
    a hook's tanh at tanh.approx's 2^-11 does not (a softcap beside an initial state put the state's
    gradient at 7.04 times the reference's error on an H100, and every hook on par with exact math), so a
    launch whose score hook or its pullback computes one Mosaic GPU approximates compiles exact math, and
    every other launch the approximate."""
    queries = keys = 64
    score_mod, kinds, draw = score_hooks(SCORE_HOOKS[hooks], queries, keys, 64)
    params, initial = draw(0)
    hooked = template.Hooks(score_mod, tuple(map(template.Param, params, kinds)), initial)
    for differentiated in (False, True):
        assert attention_mosaic_program.approximate(hooked, (64, 64), 64,
                                                    differentiated=differentiated) == approximated


APPROXIMATED_BY_MOSAIC = ("exp", "exp2", "log", "log2", "log1p", "tanh", "logistic", "sin", "cos", "sqrt",
                          "rsqrt")
"""The unary operations whose Mosaic GPU lowering takes `approx_math` (jax's Pallas Mosaic GPU lowering rules
for exp, exp2, log, log2, tanh, sin, cos, sqrt and rsqrt, which pass it to the lane implementation and set
the fast-math flag; logistic and log1p lower through them), read from the lowering, not from the
mainloop's own list."""


@pytest.mark.parametrize("operation", APPROXIMATED_BY_MOSAIC)
def test_every_approximated_operation_in_a_hook_takes_exact_math(operation):
    """A score hook and an output hook each computing one operation Mosaic GPU approximates under
    `approx_math` (`APPROXIMATED_BY_MOSAIC`) take exact math, in the forward and differentiated: no such
    operation of a hook passes unseen."""
    def applied(x):
        return getattr(jax.lax, operation)(jnp.abs(x) + 1.0)

    def score_mod(s, q_pos, k_pos):
        del q_pos, k_pos
        return s + applied(s)

    def output_mod(out):
        return applied(out)

    for hooks in (template.Hooks(score_mod), template.Hooks(output_mod=output_mod)):
        for differentiated in (False, True):
            assert not attention_mosaic_program.approximate(hooks, (64, 64), 64,
                                                            differentiated=differentiated)


def _fullest(op, ba, target):
    """The fitting config of `op`'s launch of `ba` on `target` that stages the most shared memory."""
    return max(op.fitting(ba, target), key=lambda config: op.staged_bytes(ba, config, target))


@lowers("sm90")
@pytest.mark.parametrize("launch", ["forward", "training", "default", "accurate"])
def test_the_fullest_fitting_tiles_lower_within_a_hoppers_block(launch):
    """Mosaic GPU's own lowering, which refuses a kernel staging past a block's shared memory, takes each
    launch at the fitting config that stages the most (`staged_bytes`), the one nearest the H100's block: the
    model counts every array and barrier the launch allocates (`_staged`), an independent witness of it."""
    rows, tokens, heads, kv_heads = SHAPES["compiled"]
    width = 128
    spans = template.spans(tokens, tokens, rows, causal=True)
    q = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    stats = jnp.zeros((rows, heads, tokens), jnp.float32)
    residuals = (stats, stats, jnp.zeros((rows, heads, tokens), jnp.int32),
                 template.Split(jnp.zeros((rows, heads, tokens, width), jnp.float32), stats, stats, stats))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        target = generation.gpu_target()
        if launch in ("forward", "training"):
            training = launch == "training"
            op, ba = _forward_call((rows, tokens, heads, kv_heads), width, width, training)
            fullest = _fullest(op, ba, target)
            assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
            with _pinned(fullest):
                entry = functools.partial(attention_mosaic.forward, spans=spans, residuals=training)
                jax.jit(entry).trace(q, kv, kv).lower(lowering_platforms=("cuda",))
            return
        accurate = launch == "accurate"
        op = attention_mosaic.MosaicBackward(form="causal", keys=tokens, scale=width ** -0.5,
                                             accurate=accurate)
        ba = _bound(op, q, kv, kv, q, spans.start, spans.end, None, None, (), None, residuals)
        fullest = _fullest(op, ba, target)
        assert op.staged_bytes(ba, fullest, target) <= generation.shared_budget(target)
        with _pinned(backward=fullest):
            jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
                q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
                q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@lowers("sm90")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_under_every_score_hook_lowers_for_an_h100(hooks, accurate):
    """The backward under each score hook lowers for an H100 on any host, as the GPU gate runs it: each hook
    parameter and the initial state read per tile by the dQ and dK/dV programs."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    q = jax.ShapeDtypeStruct((ROWS, queries, 4, 128), jnp.bfloat16)
    kv = jax.ShapeDtypeStruct((ROWS, keys, 2, 128), jnp.bfloat16)
    rows_shape = jnp.zeros((ROWS, 4, queries), jnp.float32)
    split = template.Split(jnp.zeros((ROWS, 4, queries, 128), jnp.float32), rows_shape, rows_shape,
                           rows_shape)
    residuals = (rows_shape, rows_shape, jnp.zeros((ROWS, 4, queries), jnp.int32), split)
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, 128)
    params, initial = draw_hooks(0)

    def backward_of(q, k, v, dout, residuals, params, initial):
        hooked = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                                 for value, kind in zip(params, kinds, strict=True)),
                                initial)
        return attention_mosaic.backward(q, k, v, spans, hooked, dout, residuals=residuals, accurate=accurate)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(backward_of).trace(q, kv, kv, q, residuals, params, initial).lower(
            lowering_platforms=("cuda",))


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_on_an_h100(call, dtype):
    """The forward, the base-2 fold, compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=False, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_on_an_h100(call, dtype):
    """The forward with natural logits compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=True, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("form", CASES)
def test_the_forward_is_on_par_over_every_call_form_on_an_h100(form):
    """The forward over each of the template core's call forms compiled on an H100."""
    _check_call_form(form, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize(("width", "value_width"), HEAD_WIDTHS)
def test_the_forward_is_on_par_at_head_widths_on_an_h100(width, value_width):
    """The forward of query heads `width` wide and value heads `value_width` over a causal call at
    `FORM_TILES`, compiled on an H100."""
    _check_forward("causal", jnp.bfloat16, SHAPES["compiled"], natural=False, interpret=False, width=width,
                   value_width=value_width, tiles=FORM_TILES)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def test_the_forward_is_on_par_under_every_hook_on_an_h100(hooks, dtype, natural):
    """The forward under each hook compiled on an H100."""
    _check_hooks(hooks, dtype, natural, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("case", CASES)
def test_the_backward_is_on_par_over_every_call_form_on_an_h100(case, form):
    """The backward in both forms over each of the template core's call forms compiled on an H100."""
    _check_call_form(case, interpret=False, backward=True, form=form)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("form", ["forward", *BACKWARD_FORMS])
@pytest.mark.parametrize("case", RUN_CASES)
def test_a_call_over_runs_is_on_par_on_an_h100(case, form):
    """A decision row's runs compiled on an H100, the forward and the backward in each form."""
    backward = form != "forward"
    _check_call_form(case, interpret=False, backward=backward, form=form if backward else "default")


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100():
    """A causal call of 64 queries over 256 keys compiled on an H100, where a consumer waiting for a load
    the producer never issues hangs (the interpreted test's case)."""
    for form in ("default", "fused"):
        _check_backward("causal", (2, 64, 4, 2), form=form, interpret=False, keys=256)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("form", BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_on_an_h100(call, form):
    """The backward in both forms compiled on an H100."""
    _check_backward(call, SHAPES["compiled"], form=form, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("form", ["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook_on_an_h100(hooks, form):
    """The backward in both forms under each score hook compiled on an H100."""
    _check_hooked_backward(hooks, form=form, interpret=False)


@pytest.mark.parametrize("causal, queries, keys, pairs", [
    (True, 256, 256, 256 * 257 // 2), (True, 128, 256, 128 * 129 // 2),
    (True, 256, 128, 128 * 129 // 2 + 128 * 128), (False, 256, 256, 256 * 256)],
    ids=["causal", "causal-fewer-queries", "causal-more-queries", "full"])
def test_the_mosaic_launches_state_the_models_attention_flops(causal, queries, keys, pairs):
    """A bf16 call's plain forward, training forward and backward through the Mosaic mainloop, each lowered
    for an H100: every launch states its cost (`attention_mosaic_program.stated`), which Dew's FLOP count
    reads to size it (`instrumentation._stated_flops`: no launch unsized), and together they state the
    library's fused attention's own count, a multiply-add a reached pair, head and product column (a causal
    call's `pairs` each query's keys up to its own place, written out: the first key aligned to the first
    query, as the mainloop masks them): the forward's two products, the training forward's two and the
    backward's four
    (`instrumentation._fused_attention_flops`), a rerun (the guarded launch, the fold's natural walk) adding
    none. The launches' own statements are read: the training forward's overflowed rows rerun the plain-JAX
    walk in a conditional, whose products the count bounds apart."""
    import re

    from dew.telemetry.instrumentation import _stated_flops

    rows, heads, kv_heads, width = 1, 4, 2, 64
    query = jax.ShapeDtypeStruct((rows, queries, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, keys, kv_heads, width), jnp.bfloat16)
    spans = template.spans(queries, keys, rows, causal=causal)
    forward = 2 * rows * heads * pairs * (width + width)

    # Lowered deviceless for an H100 whatever the host, so a GPU host of another generation lowers no TMA
    # of its own.
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")

    def counted(fn, *arrays):
        with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
            text = jax.jit(fn).trace(*arrays).lower(lowering_platforms=("cuda",)).as_text(dialect="hlo")
        launches = [line for line in text.splitlines() if re.search(r'custom_call_target="mosaic_gpu', line)]
        assert launches and all(_stated_flops(line) is not None for line in launches)
        return sum(_stated_flops(line) or 0.0 for line in launches)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, residuals=True)

    with generation.assumed("sm90"):
        residuals = jax.eval_shape(trained, query, key, key)[1:]

    def backward(query, key, value, dout, residuals):
        return attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals)

    assert counted(lambda q, k, v: attention_mosaic.forward(q, k, v, spans), query, key, key) == forward
    assert counted(trained, query, key, key) == forward
    assert counted(backward, query, key, key, query, residuals) == 2 * forward


@pytest.mark.parametrize("axis", [0, 1])
def test_a_hook_that_forms_a_unit_axis_compiles_for_an_h100(axis):
    """A score hook that forms a unit column (`q_pos[:, :1]`, `axis` 0) or a unit row (`k_pos[:1, :]`, 1,
    which Mosaic GPU selects by a reduction across warps, its scratch the launch's warpgroups reserve,
    ledger 1428), the training forward and the backward compiled by ptxas for an H100 on this CPU
    (tests/kernel_compile.py), within its shared memory and registers."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    queries = 256
    spans = template.spans(queries, queries, 1, causal=True)
    def score_mod(s, q_pos, k_pos):
        return s + 0.01 * (q_pos[:, :1] if axis == 0 else k_pos[:1, :]).astype(s.dtype)

    hooks = template.Hooks(score_mod)
    query = jax.ShapeDtypeStruct((1, queries, 4, 128), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((1, queries, 2, 128), jnp.bfloat16)

    def trained(query, key, value):
        return attention_mosaic.forward(query, key, value, spans, hooks, residuals=True)

    def backward(query, key, value, dout):
        residuals = trained(query, key, value)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals)
        return grads.query, grads.key, grads.value

    found = compile_mosaic_kernels([KernelCase("forward", lambda: (trained, (query, key, key))),
                                    KernelCase("backward", lambda: (backward, (query, key, key, query)))],
                                   "sm90")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
