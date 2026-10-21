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


DTYPE_IDS = [jnp.dtype(dtype).name for dtype in attention_triton.DTYPES]


@pytest.mark.parametrize("config", PINNED, ids=str)
@pytest.mark.parametrize("form", FORMS)
def test_the_forward_is_on_par_over_every_call_form(form, config):
    """Each call form's keys (`template.spans`), fp32, no hooks, at each pinned config."""
    check(CASES[form], HOOKS["none"], jnp.float32, config)


@pytest.mark.parametrize("form", [form for form in CASES if form not in FORMS])
def test_the_forward_is_on_par_over_the_core_s_other_call_forms(form):
    """The rest of the template core's call forms (`test_attention_template.CASES`), at one config."""
    check(CASES[form], HOOKS["none"], jnp.float32, PINNED[1])


@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
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
               "temperature": ("temperature",), "key-bias": ("key",), "tile-bias": ("tile",),
               "resumed": ("resumed",), "custom-vjp": ("custom",),
               "softcap-sinks-alibi": ("softcap", "sinks", "alibi")}
"""The score hooks the backward differentiates, each kind of parameter among them: ALiBi's slopes a
head's, a query's temperature (divided by, so padding's zeros would be infinite slopes), a per-key bias,
a dense bias, a softcap through a `jax.custom_vjp` (which forward-mode differentiation refuses); and the
softmax's initial state as sinks or as a resumed partial."""


@jax.custom_vjp
def custom_softcap(s):
    """A softcap whose derivative is its own rule, as a hook may call one."""
    return SOFTCAP * jnp.tanh(s / SOFTCAP)


def _custom_softcap_forward(s):
    capped = custom_softcap(s)
    return capped, capped


def _custom_softcap_backward(capped, cotangent):
    return (cotangent * (1 - (capped / SOFTCAP) ** 2),)


custom_softcap.defvjp(_custom_softcap_forward, _custom_softcap_backward)

BACKWARD_SEEDS = {(("alibi",), "float32"): 64, (("alibi",), "bfloat16"): 128,
                  (("alibi",), "float16"): 128, (("sinks",), "bfloat16"): 16,
                  (("resumed",), "float32"): 32, (("resumed",), "bfloat16"): 128,
                  (("resumed",), "float16"): 32, (("softcap", "sinks", "alibi"), "float32"): 64,
                  (("softcap", "sinks", "alibi"), "bfloat16"): 256,
                  (("softcap", "sinks", "alibi"), "float16"): 256}
"""The backward sites' declared draws where `seeds_for` sizes them above `SEEDS`: a head's ALiBi slope
is a gradient of four values a draw, and an initial state's of a few a query."""


def score_hooks(names: tuple, queries: int, keys: int, value_width: int = 32):
    """The score hooks `names` asks for, as a score_mod and a draw of its parameters and the softmax's
    initial state for a seed (each a `[B, H, ...]` array or `Initial`, fp32): a softcap, ALiBi, a query's
    temperature, a per-key bias, a dense bias as one score_mod; sinks, or a resumed partial (its maximum,
    a positive denominator and a numerator of any sign), as the initial state."""
    kinds = [kind for name, kind in (("alibi", template.Kind.HEAD), ("temperature", template.Kind.QUERY),
                                     ("key", template.Kind.KEY), ("tile", template.Kind.TILE))
             if name in names]

    def score_mod(s, q_pos, k_pos, *values):
        values = list(values)
        if "softcap" in names:
            s = SOFTCAP * jnp.tanh(s / SOFTCAP)
        if "custom" in names:
            s = custom_softcap(s)
        if "alibi" in names:
            s = s + values.pop(0).astype(s.dtype) * k_pos.astype(s.dtype)
        if "temperature" in names:
            s = s / values.pop(0).astype(s.dtype)
        for _ in range(("key" in names) + ("tile" in names)):
            s = s + values.pop(0).astype(s.dtype)
        return s

    def draw(seed: int):
        streams = jax.random.split(jax.random.fold_in(jax.random.key(77), seed), 6)
        params = []
        if "alibi" in names:
            params.append(alibi_bias(jnp.ones((1, 1)), 4, dtype=jnp.float32).reshape(4))
        if "temperature" in names:
            params.append(jax.random.uniform(streams[0], (ROWS, 4, queries), jnp.float32, 0.5, 1.5))
        if "key" in names:
            params.append(jax.random.normal(streams[1], (ROWS, 4, keys)))
        if "tile" in names:
            params.append(jax.random.normal(streams[2], (ROWS, 4, queries, keys)))
        initial = None
        if "sinks" in names:
            initial = template.sinks(SINKS, ROWS, queries, value_width)
        if "resumed" in names:
            total = jnp.exp(0.5 * jax.random.normal(streams[3], (ROWS, 4, queries)))
            acc = jax.random.normal(streams[4], (ROWS, 4, queries, value_width)) * total[..., None]
            initial = template.Initial(jax.random.normal(streams[5], (ROWS, 4, queries)), total, acc)
        return tuple(params), initial

    scoring = bool({"softcap", "custom", "alibi", "temperature", "key", "tile"} & set(names))
    return (score_mod if scoring else None), tuple(kinds), draw


def check_backward(case: dict, names: tuple, dtype, config, dkv_config=None, *, width: int = 32,
                   value_width: int = 32):
    """`attention_triton.backward` with its dQ launch at `config` and its dK/dV launch at `dkv_config`
    (`config` too by default; its tiles are always the dQ launch's), its every gradient (queries, keys,
    values, each score parameter, the initial state's) on par with `jax.vjp` of the template's XLA
    arithmetic, from float64: the hooks' parameters and initial state are drawn with the operands, so
    their gradients' truth is float64 too."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    score_mod, kinds, draw_hooks = score_hooks(names, queries, keys, value_width)
    draw_qkv = drawing(queries, keys, dtype, width, value_width)

    def draw(seed):
        query, key, value = draw_qkv(seed)
        dout = jax.random.normal(jax.random.key(seed + 1000), (ROWS, queries, 4, value_width), jnp.float32)
        return query, key, value, dout.astype(dtype), *draw_hooks(seed)

    def hooked(params, initial):
        return template.Hooks(score_mod, tuple(template.Param(value, kind)
                                               for value, kind in zip(params, kinds, strict=True)), initial)

    pins = tuning.pinned({attention_triton.TritonDq: config,
                          attention_triton.TritonDkv: config if dkv_config is None else dkv_config})

    def ours(query, key, value, dout, params, initial):
        with pins:
            grads = attention_triton.backward(query, key, value, spans, hooked(params, initial), dout,
                                              interpret=not ON_GPU)
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def xla(query, key, value, dout, params, initial):
        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, hooked(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, initial)
        return pull(dout.astype(query.dtype))

    assert_as_exact(jax.jit(ours), jax.jit(xla), draw,
                    f"triton backward {case} {names} {jnp.dtype(dtype).name} {config} {dkv_config} "
                    f"{width}/{value_width}",
                    seeds=BACKWARD_SEEDS.get((names, jnp.dtype(dtype).name), SEEDS))


BACKWARD_FORMS = ("full", "causal-window", "documents", "cross-causal-lengths-starts", "two-sided-window")


@pytest.mark.parametrize("config", PINNED[:3], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_the_backward_is_on_par_over_every_call_form(form, config):
    """Each call form's keys, fp32, no hooks, at pinned dQ and dK/dV configs."""
    check_backward(CASES[form], (), jnp.float32, config)


@pytest.mark.parametrize("form", [form for form in CASES if form not in BACKWARD_FORMS])
def test_the_backward_is_on_par_over_the_core_s_other_call_forms(form):
    """The rest of the template core's call forms, the dK/dV launch at warps and stages of its own."""
    check_backward(CASES[form], (), jnp.float32, PINNED[0], attention_triton.Tiles(32, 32, 8, 3))


@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("hooks", SCORE_HOOKS.values(), ids=SCORE_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook(hooks, dtype):
    """Each score hook over causal packed documents under a window: the gradients of its parameters and
    of the initial state too."""
    check_backward(CASES["causal-documents-window"], hooks, dtype, PINNED[0])


def test_a_wide_parameter_computes_as_its_fp32_value_and_its_gradient_returns_wide():
    """Under x64, a float64 ALiBi slope reaches the kernels as its fp32 value, which they compute in, and
    its gradient returns in float64, equal to the fp32 slope's."""
    case = CASES["causal-window"]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    score_mod, kinds, draw_hooks = score_hooks(("alibi",), queries, keys)
    query, key, value = drawing(queries, keys, jnp.float32)(0)
    dout = jax.random.normal(jax.random.key(5), query.shape, jnp.float32)
    (slopes,), _ = draw_hooks(0)
    with jax.enable_x64(new_val=True), tuning.pinned({attention_triton.TritonDq: PINNED[0],
                                                       attention_triton.TritonDkv: PINNED[0]}):
        grads = [attention_triton.backward(query, key, value, spans,
                                           template.Hooks(score_mod, (template.Param(wide, kinds[0]),)), dout,
                                           interpret=not ON_GPU)
                 for wide in (slopes.astype(jnp.float64), slopes)]
    assert grads[0].score_params[0].dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(grads[0].score_params[0], np.float32),
                                  np.asarray(grads[1].score_params[0]))
    np.testing.assert_array_equal(np.asarray(grads[0].query), np.asarray(grads[1].query))


def test_the_backward_resumes_queries_that_read_no_key():
    """A resumed partial over causal keys that begin at row starts (the second row's first queries see
    no key and read only their state): every gradient on par, the initial maximum's 0 for those."""
    check_backward({"causal": True, "starts": True}, SCORE_HOOKS["resumed"], jnp.float32, PINNED[0])


@pytest.mark.parametrize("width, value_width", [(48, 24), (80, 48)])
def test_the_backward_is_on_par_at_head_widths_padded_to_powers_of_two(width, value_width):
    """Head widths that ride in zero-padded, values narrower than keys, bf16."""
    check_backward(CASES["causal-window"], (), jnp.bfloat16, PINNED[0], width=width, value_width=value_width)


@pytest.mark.gpu(reason="every config compiles for the GPU")
@pytest.mark.parametrize("config", attention_triton.SPACE, ids=str)
def test_every_backward_config_on_a_gpu(config):
    """Every config of the backward's space, bf16, causal documents under a window, a softcap, sinks and
    ALiBi."""
    other = attention_triton.Tiles(config.block_q, config.block_k, 12 - config.warps, 5 - config.stages)
    check_backward(CASES["causal-documents-window"], SCORE_HOOKS["softcap-sinks-alibi"], jnp.bfloat16, config,
                   other)
