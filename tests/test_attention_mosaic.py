"""The attention template's Mosaic GPU mainloop on the wgmma family
(dew.nn.kernels.templates.attention_mosaic): its forward and its backward in both forms, check 1
(`reference_error.assert_as_exact`) against the template's reference over ordinary draws, under Mosaic GPU's
interpreter on the CPU and compiled on an H100. The interpreter is coverage, not proof (it treats wgmma_wait
as a no-op and takes the register operand's product as a dot); the GPU gate runs the compiled kernels. Its
backward reads the Triton mainloop's training forward's residuals (its split sums), interpreted on the
CPU."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.sharding import Mesh
from reference_error import assert_as_exact
from test_attention_template import CASES, ROWS, arguments, geometry as core_geometry
from test_attention_triton import HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_mosaic,
    attention_triton,
    mosaic,
    mosaic_sm90,
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
FORWARD_HOOKS = {**{name: (names, False) for name, names in SCORE_HOOKS.items()}, "gate": ((), True),
                 "all": (("softcap", "sinks", "alibi", "temperature", "key", "tile", "indexed"), True)}
"""The forward's hooks: each score hook the backward differentiates (`test_attention_triton.SCORE_HOOKS`:
every kind of parameter, the initial state as sinks or a resumed partial, a hook through a custom VJP and
one indexed off the query's coordinates), an output gate, and all at once; each `(score hooks, gate)`."""
HOOK_FORMS = {"bfloat16": (jnp.bfloat16, False), "float16": (jnp.float16, False),
              "float16-natural": (jnp.float16, True)}
"""The hooked forward's dtypes and forms: the fold over both 16-bit dtypes, and fp16's natural form (bf16's
takes a headroom guard the mainloop has not yet)."""
FORM_TILES = attention_mosaic.Tiles(64, 64, 2)
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


def _family(interpret: bool) -> mosaic.Family | None:
    """The bundle a test runs: the wgmma family's interpreted on the CPU, which has no instruction family of
    its own; the device's compiled (`attention_mosaic.family_of`)."""
    return mosaic_sm90.WGMMA if interpret else None


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool, width: int = 128,
                   value_width: int = 128, tiles: attention_mosaic.Tiles | None = None):
    """The forward over the call form `call` of `shape`, its query and key heads `width` wide and its value
    heads `value_width`, at `tiles` (the forward's default when None)."""
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    draw = _drawn(shape, dtype, width=width, value_width=value_width)

    def ours(query, key, value, dout):
        del dout
        return attention_mosaic.forward(query, key, value, spans, natural=natural, tiles=tiles,
                                        interpret=interpret, family=_family(interpret))

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic forward {call} {jnp.dtype(dtype).name} natural {natural} widths {width} "
                    f"{value_width}", seeds=SEEDS)


def _check_spans(label: str, spans: template.KeySpans, shape, *, interpret: bool, backward: bool = False,
                 accurate: bool = False, score: tuple = (), gate: bool = False, dtype=jnp.bfloat16,
                 natural: bool = False):
    """The forward, or the backward in the form `accurate` names, over `spans` at `FORM_TILES` under the
    score hooks `score` (`test_attention_triton.score_hooks`, their parameters and initial state drawn with
    the operands) and with `gate` an output gate (`hooks_for`), draws of `shape` (rows, queries, query and
    key heads, four and two under a hook) over the spans' keys in `dtype`, the forward's logits `natural` or
    folded."""
    queries = shape[1]
    score_mod, kinds, draw_hooks = score_hooks(score, queries, spans.keys, 64)
    output = hooks_for(HOOKS["gate" if gate else "none"], queries, 64, gate_of(queries, 64))
    draw_operands = _drawn(shape, dtype, width=64, keys=spans.keys)

    def draw(seed):
        return *draw_operands(seed), *draw_hooks(seed)

    def hooked(params, initial):
        return template.Hooks(score_mod, tuple(template.Param(value, kind)
                                               for value, kind in zip(params, kinds, strict=True)), initial,
                              output.output_mod, output.output_params)

    def ours(query, key, value, dout, params, initial):
        hooks = hooked(params, initial)
        if not backward:
            return attention_mosaic.forward(query, key, value, spans, hooks, natural=natural,
                                            tiles=FORM_TILES, interpret=interpret, family=_family(interpret))
        residuals = attention_triton.forward(query, key, value, spans, hooks, residuals=True,
                                             interpret=interpret)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                          accurate=accurate, tiles=FORM_TILES, interpret=interpret,
                                          family=_family(interpret))
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout, params, initial):
        hooks = hooked(params, initial)
        if not backward:
            return template.reference(query, key, value, spans, hooks)
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
        return pull(dout)

    name = (f"mosaic {'backward' if backward else 'forward'} {label} {score}{' gate' if gate else ''} "
            f"{jnp.dtype(dtype).name}{' accurate' if accurate else ''}{' natural' if natural else ''}")
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, name,
                    seeds=ALL_HOOKS_SEEDS if len(score) + gate > 1 else SEEDS)


def _check_hooks(hooks: str, dtype, natural: bool, *, interpret: bool):
    """The forward under the hooks `FORWARD_HOOKS[hooks]` over the template core's causal packed documents
    under a window (`test_attention_triton`'s hook site)."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    score, gate = FORWARD_HOOKS[hooks]
    _check_spans("causal-documents-window", spans, (ROWS, queries, 4, 2), interpret=interpret, score=score,
                 gate=gate, dtype=dtype, natural=natural)


def _check_call_form(form: str, *, interpret: bool, backward: bool = False, accurate: bool = False):
    """`_check_spans` over the template core's call form `form` (`test_attention_template.CASES`), two
    query heads a key head."""
    queries, keys = core_geometry(CASES[form])
    spans = template.spans(queries, keys, ROWS, **arguments(CASES[form]))
    _check_spans(form, spans, (ROWS, queries, 4, 2), interpret=interpret, backward=backward,
                 accurate=accurate)


def _check_backward(call: str, shape, *, accurate: bool, interpret: bool, keys: int | None = None):
    keys = shape[1] if keys is None else keys
    spans = template.spans(shape[1], keys, shape[0], **CALLS[call])
    draw = _drawn(shape, jnp.bfloat16, keys=keys)

    def ours(query, key, value, dout):
        residuals = attention_triton.forward(query, key, value, spans, residuals=True,
                                             interpret=interpret)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                          accurate=accurate, interpret=interpret, family=_family(interpret))
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic backward {call} accurate {accurate}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_interpreted(call):
    """The forward with natural logits, fp16's (bf16's takes a headroom guard the mainloop has not yet), under
    the interpreter."""
    _check_forward(call, jnp.float16, SHAPES["interpreted"], natural=True, interpret=True)


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
    interpreter: the fold over both 16-bit dtypes, and fp16's natural form (bf16's takes a headroom guard the
    mainloop has not yet)."""
    _check_hooks(hooks, dtype, natural, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("form", CASES)
def test_the_backward_is_on_par_over_every_call_form_interpreted(form, accurate):
    """The backward in both forms over each of the template core's call forms under the interpreter."""
    _check_call_form(form, interpret=True, backward=True, accurate=accurate)


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
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, accurate):
    """The backward in both forms under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], accurate=accurate, interpret=True)


@INTERPRETS
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_interpreted():
    """A causal call of 64 queries over 256 keys, whose last key tiles no query sees: the fast form's two
    consumers each take a key block, one wholly past the queries' diagonal, and walk no query tile, the
    producer loading none for them."""
    _check_backward("causal", (1, 64, 2, 1), accurate=False, interpret=True, keys=256)


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", ["causal", "causal-documents-window"])
def test_the_launches_lower_for_an_h100(call, accurate):
    """The forward and the backward's launches lower for an H100 on any host, as the GPU gate runs them: a
    causal call, and a spans one of documents, whose programs read its block map and ids; the forward under
    every hook too, whose programs read each kind of parameter and the initial state."""
    if call == "causal":
        rows, tokens, heads, kv_heads = SHAPES["compiled"]
        spans = template.spans(tokens, tokens, rows, causal=True)
    else:
        (tokens, _), rows, heads, kv_heads = core_geometry(CASES[call]), ROWS, 4, 2
        spans = template.spans(tokens, tokens, rows, **arguments(CASES[call]))
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
            jax.jit(functools.partial(attention_mosaic.forward, spans=spans, hooks=forward_hooks)).trace(
                q, kv, kv).lower(lowering_platforms=("cuda",))
        jax.jit(lambda q, k, v, dout, residuals: attention_mosaic.backward(
            q, k, v, spans, None, dout, residuals=residuals, accurate=accurate)).trace(
            q, kv, kv, q, residuals).lower(lowering_platforms=("cuda",))


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_on_an_h100(call, dtype):
    """The forward, the base-2 fold, compiled on an H100."""
    _check_forward(call, dtype, SHAPES["compiled"], natural=False, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("call", CALLS)
def test_the_natural_forward_is_on_par_on_an_h100(call):
    """The forward with natural logits, fp16's, compiled on an H100."""
    _check_forward(call, jnp.float16, SHAPES["compiled"], natural=True, interpret=False)


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
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("form", CASES)
def test_the_backward_is_on_par_over_every_call_form_on_an_h100(form, accurate):
    """The backward in both forms over each of the template core's call forms compiled on an H100."""
    _check_call_form(form, interpret=False, backward=True, accurate=accurate)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100():
    """A causal call of 64 queries over 256 keys compiled on an H100, where a consumer waiting for a load
    the producer never issues hangs (the interpreted test's case)."""
    _check_backward("causal", (2, 64, 4, 2), accurate=False, interpret=False, keys=256)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_on_an_h100(call, accurate):
    """The backward in both forms compiled on an H100."""
    _check_backward(call, SHAPES["compiled"], accurate=accurate, interpret=False)
