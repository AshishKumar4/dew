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
from test_attention_triton import BACKWARD_SEEDS, HOOKS, SCORE_HOOKS, gate_of, hooks_for, score_hooks

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
BACKWARD_HOOKS = [name for name in SCORE_HOOKS if name != "row-indexed"]
"""The backward's score hooks: every one the backward differentiates (`test_attention_triton.SCORE_HOOKS`) but
the hook indexed off the query's coordinates' column, which forms a unit axis the mainloop refuses
(`attention_mosaic.unlowered`, ledger row 1428): `test_a_hook_that_forms_a_unit_axis_is_refused`."""
FORWARD_HOOKS = {**{name: (SCORE_HOOKS[name], False) for name in BACKWARD_HOOKS},
                 "gate": ((), True),
                 "all": (("softcap", "custom", "sinks", "alibi", "temperature", "key", "tile"), True)}
"""The forward's hooks: each score hook the backward differentiates (`test_attention_triton.SCORE_HOOKS`:
every kind of parameter, the initial state as sinks or a resumed partial, a hook through a custom VJP), an
output gate, and all at once; each `(score hooks, gate)`. The hook indexed off the query's coordinates'
column forms a unit axis, which the mainloop refuses (`attention_mosaic.unlowered`, ledger row 1428):
`test_a_hook_that_forms_a_unit_axis_is_refused`."""
HOOK_FORMS = {"bfloat16": (jnp.bfloat16, False), "float16": (jnp.float16, False),
              "bfloat16-natural": (jnp.bfloat16, True), "float16-natural": (jnp.float16, True)}
"""The hooked forward's dtypes and forms: the fold and natural logits over both 16-bit dtypes."""
BACKWARD_FORMS = {"default": (False, False), "fused": (False, True), "accurate": (True, False)}
"""The backward's forms and schedules, `(accurate, fused)`: the fast form in two launches and in one (the
dK/dV program adding dQ), and the accurate form."""
FORM_TILES = attention_mosaic.Tiles(64, 64, 2)
"""The call forms' tiles: the template core's 100 queries over 96 keys (`test_attention_template.CASES`) in a
pair of query blocks over two key tiles, so a block's spans begin and end within the keys it streams."""


def _drawn(shape: tuple[int, int, int, int], dtype, width: int = 128, keys: int | None = None):
    """A draw `(query, key, value, dout)` of unit Gaussians at `shape` (its keys `keys`, else its tokens) in
    `dtype`, by seed."""
    rows, tokens, heads, kv_heads = shape
    keys = tokens if keys is None else keys

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        dims = ((heads, tokens), (kv_heads, keys), (kv_heads, keys), (heads, tokens))
        return tuple(jax.random.normal(stream, (rows, count, heads_, width), jnp.float32).astype(dtype)
                     for stream, (heads_, count) in zip(streams, dims, strict=True))

    return draw


def _family(interpret: bool) -> mosaic.Family | None:
    """The bundle a test runs: the wgmma family's interpreted on the CPU, which has no instruction family of
    its own; the device's compiled (`attention_mosaic.family_of`)."""
    return mosaic_sm90.WGMMA if interpret else None


def _check_forward(call: str, dtype, shape, *, natural: bool, interpret: bool):
    spans = template.spans(shape[1], shape[1], shape[0], **CALLS[call])
    draw = _drawn(shape, dtype)

    def ours(query, key, value, dout):
        del dout
        return attention_mosaic.forward(query, key, value, spans, natural=natural, interpret=interpret,
                                        family=_family(interpret))

    def reference(query, key, value, dout):
        del dout
        return template.reference(query, key, value, spans)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic forward {call} {jnp.dtype(dtype).name} natural {natural}", seeds=SEEDS)


def _check_spans(label: str, spans: template.KeySpans, shape, *, interpret: bool, backward: bool = False,
                 accurate: bool = False, fused: bool = False, score: tuple = (), gate: bool = False,
                 dtype=jnp.bfloat16, natural: bool = False):
    """The forward, or the backward in the form `accurate` names (`fused`, its one launch), over `spans` at
    `FORM_TILES` under the score hooks `score` (`test_attention_triton.score_hooks`, their parameters and
    initial state drawn with the operands) and with `gate` an output gate (`hooks_for`), draws of `shape`
    (rows, queries, query and key heads, four and two under a hook) over the spans' keys in `dtype`, the
    forward's logits `natural` or folded. The backward differentiates the queries, keys, values, the hook's
    parameters and the initial state."""
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
                                          accurate=accurate, fused=fused, tiles=FORM_TILES,
                                          interpret=interpret, family=_family(interpret))
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def reference(query, key, value, dout, params, initial):
        if not backward:
            return template.reference(query, key, value, spans, hooked(params, initial))

        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, hooked(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, initial)
        return pull(dout)

    name = (f"mosaic {'backward' if backward else 'forward'} {label} {score}{' gate' if gate else ''} "
            f"{jnp.dtype(dtype).name}{' accurate' if accurate else ''}{' fused' if fused else ''}"
            f"{' natural' if natural else ''}")
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


def _check_hooked_backward(hooks: str, *, accurate: bool, interpret: bool):
    """The backward in the form `accurate` names under the score hook `SCORE_HOOKS[hooks]` over the template
    core's causal packed documents under a window (`_check_hooks`' site): the gradients of the queries, keys,
    values, of the hook's parameters and of the initial state."""
    case = CASES["causal-documents-window"]
    queries, keys = core_geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    _check_spans("causal-documents-window", spans, (ROWS, queries, 4, 2), interpret=interpret, backward=True,
                 accurate=accurate, score=SCORE_HOOKS[hooks])


def _check_call_form(form: str, *, interpret: bool, backward: bool = False, accurate: bool = False,
                     fused: bool = False):
    """`_check_spans` over the template core's call form `form` (`test_attention_template.CASES`), two
    query heads a key head."""
    queries, keys = core_geometry(CASES[form])
    spans = template.spans(queries, keys, ROWS, **arguments(CASES[form]))
    _check_spans(form, spans, (ROWS, queries, 4, 2), interpret=interpret, backward=backward,
                 accurate=accurate, fused=fused)


def _check_backward(call: str, shape, *, accurate: bool, interpret: bool, keys: int | None = None,
                    fused: bool = False):
    keys = shape[1] if keys is None else keys
    spans = template.spans(shape[1], keys, shape[0], **CALLS[call])
    draw = _drawn(shape, jnp.bfloat16, keys=keys)

    def ours(query, key, value, dout):
        residuals = attention_triton.forward(query, key, value, spans, residuals=True,
                                             interpret=interpret)[1:]
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=residuals,
                                          accurate=accurate, fused=fused, interpret=interpret,
                                          family=_family(interpret))
        return grads.query, grads.key, grads.value

    def reference(query, key, value, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans), query, key, value)
        return pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"mosaic backward {call} accurate {accurate} fused {fused}", seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("call", CALLS)
def test_the_forward_is_on_par_interpreted(call, dtype):
    """The forward, the base-2 fold, under Mosaic GPU's interpreter."""
    _check_forward(call, dtype, SHAPES["interpreted"], natural=False, interpret=True)


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
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def test_the_forward_is_on_par_under_every_hook_interpreted(hooks, dtype, natural):
    """The forward under each hook (`FORWARD_HOOKS`) over causal packed documents under a window, under the
    interpreter, in each of `HOOK_FORMS`."""
    _check_hooks(hooks, dtype, natural, interpret=True)


@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_padding_rows_reach_no_hook_gradient_interpreted(accurate):
    """A call of one query and one key, padded to whole tiles, under a hook dividing by a query's
    parameter and adding a head's: the padding rows' parameter is 0, so the hook is 0/0 there, and a
    padding row that saw a key would carry its NaN into the head's gradient, a sum over the rows. Padding
    rows see no key (`algebra.Forward.queries`), and every gradient is the reference's, the head's 0."""
    spans = template.spans(1, 1, 1)
    query, key, value = (jnp.zeros((1, 1, 1, 64), jnp.bfloat16) for _ in range(3))
    dout = jnp.ones((1, 1, 1, 64), jnp.bfloat16)

    def score_mod(s, q_pos, k_pos, temperature, bias):
        del q_pos, k_pos
        return s / temperature + bias

    def hooked(temperature, bias):
        return template.Hooks(score_mod, (template.Param(temperature, template.Kind.QUERY),
                                          template.Param(bias, template.Kind.HEAD)))

    params = (jnp.ones((1, 1, 1), jnp.float32), jnp.zeros((1,), jnp.float32))
    hooks = hooked(*params)
    residuals = attention_triton.forward(query, key, value, spans, hooks, residuals=True, interpret=True)[1:]
    grads = attention_mosaic.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                      accurate=accurate, interpret=True, family=mosaic_sm90.WGMMA)
    _, pull = jax.vjp(lambda q, k, v, t, b: template.reference(q, k, v, spans, hooked(t, b)), query, key,
                      value, *params)
    for found, expected in zip((grads.query, grads.key, grads.value, *grads.score_params), pull(dout),
                               strict=True):
        np.testing.assert_array_equal(np.asarray(found, np.float32), np.asarray(expected, np.float32))


def test_a_hook_that_forms_a_unit_axis_is_refused():
    """The hook indexed off the query's coordinates' column (`q_pos[:, :1]`) forms a `[bq, 1]` array, which
    Mosaic GPU's layout inference takes no operation on: the mainloop refuses it before any launch, the
    legality rule the route's chooser reads (ledger row 1428), where a pointwise hook lowers."""
    queries, keys = core_geometry(CASES["causal"])
    spans = template.spans(queries, keys, ROWS, causal=True)
    query, key, value, _ = _drawn((ROWS, queries, 4, 2), jnp.bfloat16, width=64, keys=keys)(0)
    for names, refused in ((("indexed",), True), (("softcap", "alibi"), False)):
        score_mod, kinds, draw_hooks = score_hooks(names, queries, keys, 64)
        params, initial = draw_hooks(0)
        hooks = template.Hooks(score_mod, tuple(template.Param(param, kind)
                                                for param, kind in zip(params, kinds, strict=True)), initial)
        assert (attention_mosaic.unlowered(hooks, (64, 64), 64) is not None) == refused, names
    with pytest.raises(NotImplementedError):
        attention_mosaic.forward(query, key, value, spans,
                                 template.Hooks(score_hooks(("indexed",), queries, keys, 64)[0]),
                                 interpret=True, family=mosaic_sm90.WGMMA)


@INTERPRETS
def test_a_hook_shaped_to_the_launchs_tile_is_taken():
    """An output hook that reshapes its gate to the value width, legal on the launch's `[64, 128]` output
    tile as on the reference's rows: the legality rule traces it at the tile and value width the launch
    runs (`attention_mosaic.refused`), and the forward is on par under it."""
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

    tiles = attention_mosaic.DEFAULTS["wgmma"]
    assert attention_mosaic.refused(jnp.bfloat16, 128, 128, hooked(draw(0)[3]), backward=False,
                                    tile=(tiles.block_q, tiles.block_k)) is None

    def ours(query, key, value, gate):
        return attention_mosaic.forward(query, key, value, spans, hooked(gate), interpret=True,
                                        family=mosaic_sm90.WGMMA)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, hooked(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward, a gate shaped to the tile",
                    seeds=SEEDS)


@INTERPRETS
@pytest.mark.parametrize(("accurate", "fused"), BACKWARD_FORMS.values(), ids=BACKWARD_FORMS)
@pytest.mark.parametrize("form", CASES)
def test_the_backward_is_on_par_over_every_call_form_interpreted(form, accurate, fused):
    """The backward in both forms over each of the template core's call forms under the interpreter."""
    _check_call_form(form, interpret=True, backward=True, accurate=accurate, fused=fused)


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
@pytest.mark.parametrize(("accurate", "fused"), BACKWARD_FORMS.values(), ids=BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_interpreted(call, accurate, fused):
    """The backward in both forms under the interpreter."""
    _check_backward(call, SHAPES["interpreted"], accurate=accurate, interpret=True, fused=fused)


@INTERPRETS
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_interpreted():
    """A causal call of 64 queries over 256 keys, whose last key tiles no query sees: the fast form's two
    consumers each take a key block, one wholly past the queries' diagonal, and walk no query tile, the
    producer loading none for them."""
    for fused in (False, True):
        _check_backward("causal", (1, 64, 2, 1), accurate=False, interpret=True, keys=256, fused=fused)


@INTERPRETS
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook_interpreted(hooks, accurate):
    """The backward in both forms under each score hook over causal packed documents under a window, under
    the interpreter: the gradients of the queries, keys and values, of each hook parameter and of the initial
    state."""
    _check_hooked_backward(hooks, accurate=accurate, interpret=True)


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
@pytest.mark.parametrize(("dtype", "natural"), HOOK_FORMS.values(), ids=HOOK_FORMS)
@pytest.mark.parametrize("hooks", FORWARD_HOOKS)
def test_the_forward_is_on_par_under_every_hook_on_an_h100(hooks, dtype, natural):
    """The forward under each hook compiled on an H100."""
    _check_hooks(hooks, dtype, natural, interpret=False)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize(("accurate", "fused"), BACKWARD_FORMS.values(), ids=BACKWARD_FORMS)
@pytest.mark.parametrize("form", CASES)
def test_the_backward_is_on_par_over_every_call_form_on_an_h100(form, accurate, fused):
    """The backward in both forms over each of the template core's call forms compiled on an H100."""
    _check_call_form(form, interpret=False, backward=True, accurate=accurate, fused=fused)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
def test_a_causal_backward_of_more_keys_than_queries_is_on_par_on_an_h100():
    """A causal call of 64 queries over 256 keys compiled on an H100, where a consumer waiting for a load
    the producer never issues hangs (the interpreted test's case)."""
    for fused in (False, True):
        _check_backward("causal", (2, 64, 4, 2), accurate=False, interpret=False, keys=256, fused=fused)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize(("accurate", "fused"), BACKWARD_FORMS.values(), ids=BACKWARD_FORMS)
@pytest.mark.parametrize("call", CALLS)
def test_the_backward_is_on_par_on_an_h100(call, accurate, fused):
    """The backward in both forms compiled on an H100."""
    _check_backward(call, SHAPES["compiled"], accurate=accurate, interpret=False, fused=fused)


@ON_HOPPER
@pytest.mark.gpu(reason="runs the sm90 mainloop compiled, on an H100")
@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("hooks", BACKWARD_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook_on_an_h100(hooks, accurate):
    """The backward in both forms under each score hook compiled on an H100."""
    _check_hooked_backward(hooks, accurate=accurate, interpret=False)
