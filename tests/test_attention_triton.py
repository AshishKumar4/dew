"""dew.nn.kernels.templates.attention_triton: the template's Pallas Triton forward, at each config it is
proven over, on par with the template's XLA arithmetic from float64 (check 1) over every call form and
hook, its residuals too; interpreted on the CPU, compiled on a GPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_attention_template import CASES, ROWS, SINKS, SOFTCAP, arguments, geometry

from dew.nn.attention import alibi_bias
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template, attention_triton

ON_GPU = jax.default_backend() == "gpu"
SEEDS = 8
"""The forward's declared draws (check 1), sized by `seeds_for`."""

ALL_HOOKS_SEEDS = 32
"""The draws of a site with every hook at once, which `seeds_for` sizes at 32: the reference compared with
itself resolves PARITY only there."""

PINNED = (attention_triton.Tiles(32, 32, 4, 2), attention_triton.Tiles(64, 32, 4, 2),
          attention_triton.Tiles(32, 64, 8, 3), attention_triton.Tiles(128, 128, 8, 2))
"""The configs each case runs at on the CPU, where interpreting all of `SPACE` would take hours; a GPU
runs every config (`test_every_config_on_a_gpu`)."""


def drawing(queries: int, keys: int, dtype, width: int = 32, value_width: int = 32):
    """A draw of queries `[2, queries, 4, width]` over keys `[2, keys, 2, width]` and values of
    `value_width`, two query heads a key head."""
    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        query = jax.random.normal(streams[0], (ROWS, queries, 4, width), jnp.float32).astype(dtype)
        key = jax.random.normal(streams[1], (ROWS, keys, 2, width), jnp.float32).astype(dtype)
        value = jax.random.normal(streams[2], (ROWS, keys, 2, value_width), jnp.float32)
        return query, key, value.astype(dtype)

    return draw


def hooks_for(names: frozenset, queries: int, value_width: int, gate=None) -> template.Hooks:
    """The hooks `names` asks for: a softcap and ALiBi as one score_mod, sinks as the initial state, an
    output gate."""
    slopes = alibi_bias(jnp.ones((1, 1)), 4, dtype=jnp.float32).reshape(4)

    def score_mod(s, q_pos, k_pos, *params):
        if "softcap" in names:
            s = SOFTCAP * jnp.tanh(s / SOFTCAP)
        if "alibi" in names:
            s = s + params[0].astype(s.dtype) * k_pos.astype(s.dtype)
        return s

    def output_mod(o, gate):
        return o * jax.nn.sigmoid(gate.astype(o.dtype))

    scoring = "softcap" in names or "alibi" in names
    return template.Hooks(
        score_mod if scoring else None,
        (template.Param(slopes, template.Kind.HEAD),) if "alibi" in names else (),
        template.sinks(SINKS, ROWS, queries, value_width) if "sinks" in names else None,
        output_mod if "gate" in names else None,
        (template.Param(gate, template.Kind.OUTPUT),) if "gate" in names else ())


HOOKS = {"none": frozenset(), "softcap": frozenset({"softcap"}), "sinks": frozenset({"sinks"}),
         "alibi": frozenset({"alibi"}), "gate": frozenset({"gate"}),
         "all": frozenset({"softcap", "sinks", "alibi", "gate"})}

FORMS = ("full", "causal", "causal-window", "two-sided-window", "causal-chunk", "documents",
         "causal-documents-window", "cross-causal", "cross-causal-lengths-starts", "lengths-starts")


def gate_of(queries: int, value_width: int) -> jax.Array:
    return jax.random.normal(jax.random.key(99), (ROWS, queries, 4, value_width), jnp.float32)


def check(case: dict, names: frozenset, dtype, config: attention_triton.Tiles, *, residuals: bool = False,
          width: int = 32, value_width: int = 32):
    """`attention_triton.forward` pinned at `config` on par with `template.reference` over `case`'s keys and
    the hooks `names`, at the site's declared seeds."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    hooks = hooks_for(names, queries, value_width, gate_of(queries, value_width))
    pin = tuning.pinned({attention_triton.TritonForward: config})

    def ours(query, key, value):
        with pin:
            return attention_triton.forward(query, key, value, spans, hooks, residuals=residuals,
                                            interpret=not ON_GPU)

    def xla(query, key, value):
        return template.reference(query, key, value, spans, hooks, residuals=residuals)

    assert_as_exact(jax.jit(ours), jax.jit(xla), drawing(queries, keys, dtype, width, value_width),
                    f"triton forward {case} {sorted(names)} {jnp.dtype(dtype).name} {config}",
                    seeds=ALL_HOOKS_SEEDS if names == HOOKS["all"] else SEEDS)


@pytest.mark.parametrize("config", PINNED, ids=str)
@pytest.mark.parametrize("form", FORMS)
def test_the_forward_is_on_par_over_every_call_form(form, config):
    """Each call form's keys (`template.spans`), fp32, no hooks, at each pinned config."""
    check(CASES[form], HOOKS["none"], jnp.float32, config)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("hooks", HOOKS.values(), ids=HOOKS)
def test_the_forward_is_on_par_under_every_hook(hooks, dtype):
    """Each hook over causal packed documents under a window, the residuals too, at one config."""
    check(CASES["causal-documents-window"], hooks, dtype, PINNED[1], residuals=True)


@pytest.mark.parametrize("width, value_width", [(64, 64), (80, 48), (128, 128)])
def test_the_forward_is_on_par_at_head_widths_padded_to_powers_of_two(width, value_width):
    """Widths of a power of two and narrower ones that ride in zero-padded, values narrower than keys."""
    check(CASES["causal"], HOOKS["none"], jnp.bfloat16, PINNED[0], width=width, value_width=value_width)


@pytest.mark.gpu(reason="every config compiles for the GPU")
@pytest.mark.parametrize("config", attention_triton.SPACE, ids=str)
def test_every_config_on_a_gpu(config):
    """Every config of the space, bf16, causal documents under a window, every hook."""
    check(CASES["causal-documents-window"], HOOKS["all"], jnp.bfloat16, config, residuals=True)


def test_a_query_that_sees_no_key_reads_zero_and_a_sink_keeps_its_share():
    """Padding queries (id 0) see nothing: their output and denominator are zero without sinks; with
    sinks their denominator is the sink's one and their output still zero."""
    case = CASES["documents"]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    query, key, value = drawing(queries, keys, jnp.float32)(0)
    with tuning.pinned({attention_triton.TritonForward: PINNED[0]}):
        out, _, total = attention_triton.forward(query, key, value, spans, residuals=True,
                                                 interpret=not ON_GPU)
        sinks = hooks_for(HOOKS["sinks"], queries, 32)
        _, _, sunk = attention_triton.forward(query, key, value, spans, sinks, residuals=True,
                                              interpret=not ON_GPU)
    padding = np.asarray(spans.query_ids) == 0
    assert padding.any()
    assert np.all(np.asarray(out)[padding] == 0)
    assert np.all(np.asarray(total).transpose(0, 2, 1)[padding] == 0)
    assert np.all(np.asarray(sunk).transpose(0, 2, 1)[padding] == 1)


SCORE_HOOKS = {"none": (), "softcap": ("softcap",), "sinks": ("sinks",), "alibi": ("alibi",),
               "key-bias": ("key",), "tile-bias": ("tile",),
               "softcap-sinks-alibi": ("softcap", "sinks", "alibi")}
"""The score hooks the backward differentiates, each kind of parameter among them: ALiBi's slopes a
head's, a per-key bias, a dense bias."""

BACKWARD_SEEDS = {(("alibi",), "float32"): 64, (("alibi",), "bfloat16"): 128,
                  (("sinks",), "bfloat16"): 16, (("softcap", "sinks", "alibi"), "float32"): 64,
                  (("softcap", "sinks", "alibi"), "bfloat16"): 256}
"""The backward sites' declared draws where `seeds_for` sizes them above `SEEDS`: a head's ALiBi slope
is a gradient of four values a draw, and the sinks' bf16 state's of a few."""


def score_hooks(names: tuple, queries: int, keys: int) -> template.Hooks:
    """The score hooks `names` asks for, their parameters fixed: a softcap, ALiBi, a per-key bias, a dense
    bias as one score_mod, sinks as the softmax's initial state."""
    slopes = alibi_bias(jnp.ones((1, 1)), 4, dtype=jnp.float32).reshape(4)
    streams = jax.random.split(jax.random.key(77), 2)
    params = []
    if "alibi" in names:
        params.append(template.Param(slopes, template.Kind.HEAD))
    if "key" in names:
        params.append(template.Param(jax.random.normal(streams[0], (ROWS, 4, keys)), template.Kind.KEY))
    if "tile" in names:
        dense = jax.random.normal(streams[1], (ROWS, 4, queries, keys))
        params.append(template.Param(dense, template.Kind.TILE))

    def score_mod(s, q_pos, k_pos, *values):
        values = list(values)
        if "softcap" in names:
            s = SOFTCAP * jnp.tanh(s / SOFTCAP)
        if "alibi" in names:
            s = s + values.pop(0).astype(s.dtype) * k_pos.astype(s.dtype)
        for _ in range(("key" in names) + ("tile" in names)):
            s = s + values.pop(0).astype(s.dtype)
        return s

    scoring = bool({"softcap", "alibi", "key", "tile"} & set(names))
    return template.Hooks(score_mod if scoring else None, tuple(params),
                          template.sinks(SINKS, ROWS, queries, 32) if "sinks" in names else None)


ROW_TERMS = ("exact", "online", "output", "compensated", "lean")
"""The backward's forms of each query's `D = sum_j p dP`: summed from its tiles in a first pass
(`backward`'s default), in the dQ launch's one pass beside dQ (`online=`), and `rowsum(dO * O)` from the
forward's fp32 output (`residuals=`), plainly or compensated (`compensated=`)."""


def check_backward(case: dict, names: tuple, dtype, forward_config, backward_config, row_term: str = "exact"):
    """`attention_triton.backward`, its every gradient (queries, keys, values, each score parameter, the
    initial state's) on par with `jax.vjp` of the template's XLA arithmetic, from float64, its `D` in the
    form `row_term` names (`ROW_TERMS`), the second from `forward`'s residuals."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    hooks = score_hooks(names, queries, keys)
    draw_qkv = drawing(queries, keys, dtype)

    def draw(seed):
        query, key, value = draw_qkv(seed)
        dout = jax.random.normal(jax.random.key(seed + 1000), (ROWS, queries, 4, 32), jnp.float32)
        return query, key, value, dout.astype(dtype)

    def with_hooks(params, initial):
        return template.Hooks(hooks.score_mod, tuple(template.Param(value, param.kind)
                                                     for value, param in zip(params, hooks.score_params,
                                                                             strict=True)),
                              initial)

    pins = tuning.pinned({attention_triton.TritonForward: forward_config,
                          attention_triton.TritonDq: backward_config,
                          attention_triton.TritonDkv: backward_config})

    def ours(query, key, value, dout):
        with pins:
            residuals = None
            if row_term in ("output", "compensated", "lean"):
                _, *residuals = attention_triton.forward(query, key, value, spans, hooks, residuals=True,
                                                         wide=True, interpret=not ON_GPU)
            grads = attention_triton.backward(query, key, value, spans, hooks, dout, residuals=residuals,
                                              online=row_term == "online",
                                              compensated=row_term == "compensated", lean=row_term == "lean",
                                              interpret=not ON_GPU)
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def xla(query, key, value, dout):
        params = tuple(param.value for param in hooks.score_params)

        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, with_hooks(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, hooks.initial)
        dq, dk, dv, dparams, dinitial = pull(dout.astype(query.dtype))
        return dq, dk, dv, dparams, dinitial

    assert_as_exact(jax.jit(ours), jax.jit(xla), draw,
                    f"triton backward {case} {names} {jnp.dtype(dtype).name} {backward_config} {row_term}",
                    seeds=BACKWARD_SEEDS.get((names, jnp.dtype(dtype).name), SEEDS))


BACKWARD_FORMS = ("full", "causal-window", "documents", "cross-causal-lengths-starts", "two-sided-window")


@pytest.mark.parametrize("row_term", ROW_TERMS)
@pytest.mark.parametrize("config", PINNED[:3], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_the_backward_is_on_par_over_every_call_form(form, config, row_term):
    """Each call form's keys, fp32, no hooks, at pinned dQ and dK/dV configs, each form of `D`."""
    check_backward(CASES[form], (), jnp.float32, PINNED[1], config, row_term)


@pytest.mark.parametrize("row_term", ROW_TERMS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("hooks", SCORE_HOOKS.values(), ids=SCORE_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook(hooks, dtype, row_term):
    """Each score hook over causal packed documents under a window: the gradients of its parameters and
    of the sinks' initial state too, each form of `D`."""
    check_backward(CASES["causal-documents-window"], hooks, dtype, PINNED[1], PINNED[0], row_term)


@pytest.mark.gpu(reason="every config compiles for the GPU")
@pytest.mark.parametrize("row_term", ROW_TERMS)
@pytest.mark.parametrize("config", attention_triton.SPACE, ids=str)
def test_every_backward_config_on_a_gpu(config, row_term):
    """Every config of the backward's space, bf16, causal documents under a window, a softcap, sinks and
    ALiBi."""
    check_backward(CASES["causal-documents-window"], SCORE_HOOKS["softcap-sinks-alibi"], jnp.bfloat16,
                   PINNED[1], config, row_term)
