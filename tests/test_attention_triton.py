"""dew.nn.kernels.templates.attention_triton: the template's Pallas Triton forward, at each config it is
proven over, on par with the template's XLA arithmetic from float64 (check 1) over every call form and
hook, its residuals too; interpreted on the CPU, compiled on a GPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, assert_fp32_reduction_bound
from test_attention_template import (
    CASES,
    ONE_RUN,
    ROWS,
    RUN_CASES,
    SINKS,
    SOFTCAP,
    arguments,
    geometry as core_geometry,
)

from dew.nn.attention import alibi_bias
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_portable,
    attention_triton,
)

ON_GPU = jax.default_backend() == "gpu"
SEEDS = 8
"""The forward's declared draws (check 1), sized by `seeds_for`."""

ALL_HOOKS_SEEDS = 32
"""The draws of a site with every hook at once, which `seeds_for` sizes at 32: the reference compared with
itself resolves PARITY only there."""

IRREGULAR = {"full-irregular": {"shape": (257, 385)},
             "causal-irregular": {"causal": True, "shape": (257, 385)}}
"""Calls of more tiles than the core's cases, of no whole multiple of any tile: every config walks
several key tiles, a query's maximum moving across them, and the full call holds whole tiles every
query of a block sees. The CPU walks them at 32-key tiles (13 of them); the GPU sweeps walk them at every
config, 128-key tiles among them, with the real dot. (Interpreted, XLA:CPU's fp32 product over a
128-by-128 tile rounds the same entries 1.554 times as far from float64 as the reference's whole
257-by-385 product does, over 10 draws; recorded 2026-10-21, ledger 882.)"""


@pytest.fixture(autouse=True)
def _interpreted_as_an_a100():
    """An interpreted call in a CPU process states the GPU it emulates (`generation.assumed`): the frozen
    Triton mainloop's tiles and their costs are an A100's, its target's facts, where a CPU has none."""
    if jax.default_backend() != "cpu":
        yield
        return
    with generation.assumed("sm80"):
        yield


def geometry(case: dict) -> tuple[int, int]:
    """`case`'s query and key counts: its own `shape`, or the core's (`test_attention_template`)."""
    return case["shape"] if "shape" in case else core_geometry(case)


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
          width: int = 32, value_width: int = 32, natural: bool = template.NATURAL, mainloop: str = "triton"):
    """`attention_triton.forward` pinned at `config` (or with `mainloop` "portable" the plain-JAX mainloop's
    `attend` at its tiles) on par with `template.reference` over `case`'s keys and the hooks `names`, at the
    site's declared seeds, its exponent `natural` or the fold."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    hooks = hooks_for(names, queries, value_width, gate_of(queries, value_width))
    pin = tuning.pinned({attention_triton.TritonForward: config})

    # The maxima are in the logits' own units, raw products in the raw form (`algebra.Forward.raw`), the
    # query's taken by its scale's power of two (`template.prescale`), taken to the reference's natural ones
    # as the algebra takes them (`algebra.Forward.naturalized`).
    power, scale = template.prescale(dtype, width, width ** -0.5)
    spec = algebra.Forward(spans.form, keys, ids=spans.query_ids is not None, scale=scale,
                           score_mod=hooks.score_mod, natural=natural, initial=hooks.initial is not None,
                           prescale=power)

    def ours(query, key, value):
        if mainloop == "portable":
            out = attention_portable.attend(query, key, value, spans, hooks, block_q=config.block_q,
                                            block_k=config.block_k, pad=attention_triton.BLOCK,
                                            residuals=residuals, natural=natural)
        else:
            with pin:
                out = attention_triton.forward(query, key, value, spans, hooks, residuals=residuals,
                                               natural=natural, interpret=not ON_GPU)
        # The maxima and denominators the reference keeps too; the anchor keys only the backward reads,
        # held to its gradients (`check_backward`).
        if not residuals:
            return out
        return out[0], spec.naturalized(out[1]) if spec.raw else out[1], out[2]

    def xla(query, key, value):
        return template.reference(query, key, value, spans, hooks, residuals=residuals)

    assert_as_exact(jax.jit(ours), jax.jit(xla), drawing(queries, keys, dtype, width, value_width),
                    f"{mainloop} forward {case} {sorted(names)} {jnp.dtype(dtype).name} {config} "
                    f"natural {natural}",
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


@pytest.mark.parametrize("config", PINNED[:3], ids=str)
@pytest.mark.parametrize("form", RUN_CASES)
def test_the_plain_forward_is_on_par_over_runs(form, config):
    """Each query's runs (`test_attention_template.RUN_CASES`), a decision row's: the plain-JAX mainloop
    walks each query block's ranges in turn, at tiles that part its fragments."""
    check(RUN_CASES[form], HOOKS["none"], jnp.float32, config, mainloop="portable")


def test_the_frozen_mainloop_refuses_runs():
    """The frozen Triton mainloop walks one key interval a query: called on a call's runs, its forward and
    backward refuse them before any launch rather than compute them as one interval (ledger 956)."""
    queries, keys = core_geometry(RUN_CASES["runs"])
    spans = template.spans(queries, keys, ROWS, **arguments(RUN_CASES["runs"]))
    query = jnp.zeros((ROWS, queries, 2, 32), jnp.float32)
    key = jnp.zeros((ROWS, keys, 1, 32), jnp.float32)
    with pytest.raises(NotImplementedError):
        attention_triton.forward(query, key, key, spans, interpret=True)
    with pytest.raises(NotImplementedError):
        attention_triton.backward(query, key, key, spans, None, query, interpret=True)


@pytest.mark.parametrize("config", PINNED[:2], ids=str)
def test_one_run_a_query_is_the_interval_the_frozen_mainloop_walks(config):
    """One run a query (`test_attention_template.ONE_RUN`, a chunked or banded local-attention layer's runs)
    is one key interval, which the frozen Triton mainloop walks forward and backward (ledger 1991)."""
    check(ONE_RUN, HOOKS["none"], jnp.float32, config)
    check_backward(ONE_RUN, (), jnp.float32, config)


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("hooks", HOOKS.values(), ids=HOOKS)
def test_the_forward_is_on_par_under_every_hook(hooks, dtype, natural):
    """Each hook over causal packed documents under a window, the residuals too, at one config, in both
    forms of the exponent."""
    check(CASES["causal-documents-window"], hooks, dtype, PINNED[1], residuals=True, natural=natural)


@pytest.mark.parametrize("width, value_width", [(64, 64), (80, 48), (128, 128)])
def test_the_forward_is_on_par_at_head_widths_padded_to_powers_of_two(width, value_width):
    """Widths of a power of two and narrower ones that ride in zero-padded, values narrower than keys."""
    check(CASES["causal"], HOOKS["none"], jnp.bfloat16, PINNED[0], width=width, value_width=value_width)


@pytest.mark.gpu(reason="every config compiles for the GPU")
@pytest.mark.parametrize("case", ["causal-documents-window", *IRREGULAR])
@pytest.mark.parametrize("config", attention_triton.SPACE, ids=str)
def test_every_config_on_a_gpu(config, case):
    """Every config of the space, bf16, every hook: over causal documents under a window, and over the
    irregular full and causal calls (`IRREGULAR`)."""
    check({**CASES, **IRREGULAR}[case], HOOKS["all"], jnp.bfloat16, config, residuals=True)


@pytest.mark.parametrize("case", IRREGULAR)
def test_the_forward_walks_many_key_tiles(case):
    """The irregular calls (`IRREGULAR`) at 32-key tiles, every hook, fp32: the maximum moves across
    key tiles and whole tiles take the unmasked path."""
    check(IRREGULAR[case], HOOKS["all"], jnp.float32, PINNED[0], residuals=True)


def test_a_query_that_sees_no_key_reads_zero_and_a_sink_keeps_its_share():
    """Padding queries (id 0) see nothing: their output and denominator are zero without sinks; with
    sinks their denominator is the sink's one and their output still zero."""
    case = CASES["documents"]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    query, key, value = drawing(queries, keys, jnp.float32)(0)
    with tuning.pinned({attention_triton.TritonForward: PINNED[0]}):
        out, _, total, *_ = attention_triton.forward(query, key, value, spans, residuals=True,
                                                     interpret=not ON_GPU)
        sinks = hooks_for(HOOKS["sinks"], queries, 32)
        _, _, sunk, *_ = attention_triton.forward(query, key, value, spans, sinks, residuals=True,
                                                  interpret=not ON_GPU)
    padding = np.asarray(spans.query_ids) == 0
    assert padding.any()
    assert np.all(np.asarray(out)[padding] == 0)
    assert np.all(np.asarray(total).transpose(0, 2, 1)[padding] == 0)
    assert np.all(np.asarray(sunk).transpose(0, 2, 1)[padding] == 1)


SCORE_HOOKS = {"none": (), "softcap": ("softcap",), "sinks": ("sinks",), "alibi": ("alibi",),
               "temperature": ("temperature",), "key-bias": ("key",), "tile-bias": ("tile",),
               "resumed": ("resumed",), "custom-vjp": ("custom",), "row-indexed": ("indexed",),
               "column-indexed": ("key-indexed",), "softcap-sinks-alibi": ("softcap", "sinks", "alibi")}
"""The score hooks the backward differentiates, each kind of parameter among them: ALiBi's slopes a
head's, a query's temperature (divided by, so padding's zeros would be infinite slopes), a per-key bias,
a dense bias, a softcap through a `jax.custom_vjp` (which forward-mode differentiation refuses), a bias
indexed off the query's coordinates' column and one off the keys' coordinates' row (each reads them at its
tile rank, a unit axis on the Mosaic mainloop, whose unit row reduces across warps, ledger 1428); and the
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


def score_hooks(names: tuple, queries: int, keys: int, value_width: int = 32, *, rows: int = ROWS,
                heads: int = 4):
    """The score hooks `names` asks for, as a score_mod and a draw of its parameters and the softmax's
    initial state for a seed (each a `[B, H, ...]` array or `Initial`, fp32, `rows` by `heads` query heads):
    a softcap, ALiBi, a query's temperature, a per-key bias, a dense bias as one score_mod; sinks, or a
    resumed partial (its maximum, a positive denominator and a numerator of any sign), as the initial
    state."""
    kinds = [kind for name, kind in (("alibi", template.Kind.HEAD), ("temperature", template.Kind.QUERY),
                                     ("key", template.Kind.KEY), ("tile", template.Kind.TILE))
             if name in names]

    def score_mod(s, q_pos, k_pos, *values):
        values = list(values)
        if "softcap" in names:
            s = SOFTCAP * jnp.tanh(s / SOFTCAP)
        if "custom" in names:
            s = custom_softcap(s)
        if "indexed" in names:
            s = s + 0.01 * q_pos[:, :1].astype(s.dtype)
        if "key-indexed" in names:
            s = s + 0.01 * k_pos[:1, :].astype(s.dtype)
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
            params.append(alibi_bias(jnp.ones((1, 1)), heads, dtype=jnp.float32).reshape(heads))
        if "temperature" in names:
            params.append(jax.random.uniform(streams[0], (rows, heads, queries), jnp.float32, 0.5, 1.5))
        if "key" in names:
            params.append(jax.random.normal(streams[1], (rows, heads, keys)))
        if "tile" in names:
            params.append(jax.random.normal(streams[2], (rows, heads, queries, keys)))
        initial = None
        if "sinks" in names:
            initial = template.sinks(SINKS[:heads], rows, queries, value_width)
        if "resumed" in names:
            total = jnp.exp(0.5 * jax.random.normal(streams[3], (rows, heads, queries)))
            acc = jax.random.normal(streams[4], (rows, heads, queries, value_width)) * total[..., None]
            initial = template.Initial(jax.random.normal(streams[5], (rows, heads, queries)), total, acc)
        return tuple(params), initial

    scoring = bool({"softcap", "custom", "indexed", "key-indexed", "alibi", "temperature", "key", "tile"}
                   & set(names))
    return (score_mod if scoring else None), tuple(kinds), draw


def joint(config: attention_triton.Tiles, dkv_config: attention_triton.Tiles | None = None):
    """The backward's config of the dQ launch's `config` and the dK/dV launch's warps and stages, from
    `dkv_config` (`config` by default)."""
    other = config if dkv_config is None else dkv_config
    return attention_triton.BackwardTiles(config.block_q, config.block_k, config.warps, config.stages,
                                          other.warps, other.stages)


def check_backward(case: dict, names: tuple, dtype, config, dkv_config=None, *, width: int = 32,
                   value_width: int = 32, natural: bool = template.NATURAL,
                   accurate: bool = template.ACCURATE, mainloop: str = "triton"):
    """`attention_triton.backward` with its dQ launch at `config` and its dK/dV launch at `dkv_config`
    (`config` too by default; its tiles are always the dQ launch's), or with `mainloop` "portable" the
    plain-JAX mainloop's `backward` at `config`'s tiles, its every gradient (queries, keys,
    values, each score parameter, the initial state's) on par with `jax.vjp` of the template's XLA
    arithmetic, from float64: the hooks' parameters and initial state are drawn with the operands, so
    their gradients' truth is float64 too; its exponent `natural` or the fold, and its backward `accurate`
    or the fast form."""
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

    pins = tuning.pinned({attention_triton.TritonBackward: joint(config, dkv_config)})

    def ours(query, key, value, dout, params, initial):
        if mainloop == "portable":
            grads = attention_portable.backward(query, key, value, spans, hooked(params, initial), dout,
                                                block_q=config.block_q, block_k=config.block_k,
                                                pad=attention_triton.BLOCK, natural=natural,
                                                accurate=accurate)
        else:
            with pins:
                grads = attention_triton.backward(query, key, value, spans, hooked(params, initial), dout,
                                                  natural=natural, accurate=accurate, interpret=not ON_GPU)
        return grads.query, grads.key, grads.value, grads.score_params, grads.initial

    def xla(query, key, value, dout, params, initial):
        def attend(query, key, value, params, initial):
            return template.reference(query, key, value, spans, hooked(params, initial))

        _, pull = jax.vjp(attend, query, key, value, params, initial)
        return pull(dout.astype(query.dtype))

    assert_as_exact(jax.jit(ours), jax.jit(xla), draw,
                    f"{mainloop} backward {case} {names} {jnp.dtype(dtype).name} {config} {dkv_config} "
                    f"{width}/{value_width} natural {natural} accurate {accurate}",
                    seeds=BACKWARD_SEEDS.get((names, jnp.dtype(dtype).name), SEEDS))


BACKWARD_FORMS = ("full", "causal-window", "documents", "cross-causal-lengths-starts", "two-sided-window")


TEMPLATE_FORMS = {"default": {}, "accurate": {"natural": True, "accurate": True}}
"""The template's two forms an ordinary draw holds to check 1: the default (the base-2 fold and the
backward's fast form) and the accurate one (natural logits and the backward's accurate form)."""


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("config", PINNED[:3], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_the_backward_is_on_par_over_every_call_form(form, config, options):
    """Each call form's keys, fp32, no hooks, at pinned dQ and dK/dV configs, in both forms."""
    check_backward(CASES[form], (), jnp.float32, config, **options)


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("config", PINNED[:2], ids=str)
@pytest.mark.parametrize("form", RUN_CASES)
def test_the_plain_backward_is_on_par_over_runs(form, config, options):
    """Each query's runs on the plain-JAX mainloop: its query walk takes its ranges as the forward does, and
    its key walk reads the query blocks some range of which holds its keys, in both forms."""
    check_backward(RUN_CASES[form], (), jnp.float32, config, mainloop="portable", **options)


@pytest.mark.parametrize("form", [form for form in CASES if form not in BACKWARD_FORMS])
def test_the_backward_is_on_par_over_the_core_s_other_call_forms(form):
    """The rest of the template core's call forms, the dK/dV launch at warps and stages of its own."""
    check_backward(CASES[form], (), jnp.float32, PINNED[0], attention_triton.Tiles(32, 32, 8, 3))


@pytest.mark.parametrize("options", TEMPLATE_FORMS.values(), ids=TEMPLATE_FORMS)
@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("hooks", SCORE_HOOKS.values(), ids=SCORE_HOOKS)
def test_the_backward_is_on_par_under_every_score_hook(hooks, dtype, options):
    """Each score hook over causal packed documents under a window: the gradients of its parameters and
    of the initial state too, in both forms."""
    check_backward(CASES["causal-documents-window"], hooks, dtype, PINNED[0], **options)


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
    with jax.enable_x64(new_val=True), tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        grads = [attention_triton.backward(query, key, value, spans,
                                           template.Hooks(score_mod, (template.Param(wide, kinds[0]),)), dout,
                                           interpret=not ON_GPU)
                 for wide in (slopes.astype(jnp.float64), slopes)]
    assert grads[0].score_params[0].dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(grads[0].score_params[0], np.float32),
                                  np.asarray(grads[1].score_params[0]))
    np.testing.assert_array_equal(np.asarray(grads[0].query), np.asarray(grads[1].query))


def test_an_integer_parameter_keys_and_runs():
    """A hook reading an int32 key parameter (a bucket's low bit) keys its launches at that dtype and
    runs, forward and backward, on par with the reference arithmetic from float64."""
    case = CASES["causal-window"]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    buckets = jnp.broadcast_to(jnp.arange(keys, dtype=jnp.int32), (ROWS, 4, keys))

    def score_mod(s, q_pos, k_pos, bucket):
        return s + (bucket & 1).astype(s.dtype)

    hooks = template.Hooks(score_mod, (template.Param(buckets, template.Kind.KEY),))
    draw_qkv = drawing(queries, keys, jnp.float32)

    def draw(seed):
        query, key, value = draw_qkv(seed)
        return query, key, value, jax.random.normal(jax.random.key(seed + 1000), query.shape, jnp.float32)

    pins = tuning.pinned({attention_triton.TritonForward: PINNED[1],
                          attention_triton.TritonBackward: joint(PINNED[0])})

    def ours(query, key, value, dout):
        with pins:
            out = attention_triton.forward(query, key, value, spans, hooks, interpret=not ON_GPU)
            grads = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=not ON_GPU)
        return out, grads.query, grads.key, grads.value

    def xla(query, key, value, dout):
        out, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
        return out, *pull(dout)

    assert_as_exact(jax.jit(ours), jax.jit(xla), draw, "triton integer parameter", seeds=SEEDS)


def test_an_output_hook_of_the_call_s_own_width_keys_and_runs():
    """An output hook scaling each of the call's 32 value features by its own factor keys its launch at
    that width and runs on par with the reference arithmetic from float64."""
    case = CASES["causal-window"]
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))

    def output_mod(o):
        return o * jnp.arange(32, dtype=o.dtype)

    hooks = template.Hooks(output_mod=output_mod)
    pins = tuning.pinned({attention_triton.TritonForward: PINNED[1]})

    def ours(query, key, value):
        with pins:
            return attention_triton.forward(query, key, value, spans, hooks, interpret=not ON_GPU)

    def xla(query, key, value):
        return template.reference(query, key, value, spans, hooks)

    assert_as_exact(jax.jit(ours), jax.jit(xla), drawing(queries, keys, jnp.float32), "triton output width",
                    seeds=SEEDS)


def test_many_fp16_cotangents_too_small_once_normalized_still_sum_into_dv():
    """fp16, 32 queries over 32 equal keys, every output cotangent 2^-24: each query's share 2^-29 of
    every value is below fp16's least subnormal, yet their 32 sum to dV = 2^-24, as the reference's
    derivative gives."""
    spans = template.spans(32, 32, 1)
    zeros = jnp.zeros((1, 32, 1, 1), jnp.float16)
    dout = jnp.full((1, 32, 1, 1), 2.0 ** -24, jnp.float16)
    with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        ours = attention_triton.backward(zeros, zeros, zeros, spans, None, dout, scale=1.0,
                                         interpret=not ON_GPU).value
    _, pull = jax.vjp(lambda value: template.reference(zeros, zeros, value, spans, scale=1.0), zeros)
    [theirs] = pull(dout)
    assert float(theirs[0, 0, 0, 0]) == 2.0 ** -24
    np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(theirs, np.float32))


@pytest.mark.parametrize("dtype", attention_triton.DTYPES, ids=DTYPE_IDS)
def test_cotangents_that_cancel_across_a_key_cancel_within_the_sums_bound(dtype):
    """Three queries over three equal keys, cotangents 1, 2 and -3: each key's gradient is the queries'
    equal shares of them, which cancel in float64. A cancellation's zero is no structure: each
    implementation's gradient is its sums' rounding, so Dew's and the reference's are each within the
    fp32 reduction bound of the shares' magnitudes (`assert_fp32_reduction_bound`), and cotangents drawn to
    cancel the same way hold check 1. The bound's derivation: every weight is the one fp32 value 1/3
    rounded, a common factor of a sum that cancels exactly, so its rounding moves nothing; what remains is
    a length-3 dot product's rounding, within gamma_3 of its products' magnitudes with each product's own
    rounding included (Higham, Accuracy and Stability, 3.1), which the bound takes twice."""
    spans = template.spans(3, 3, 1)

    def ours(value, dout):
        zeros = jnp.zeros((1, 3, 1, 1), value.dtype)
        with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
            return attention_triton.backward(zeros, zeros, value, spans, None, dout, scale=1.0,
                                             interpret=not ON_GPU).value

    def theirs(value, dout):
        def attend(value):
            zeros = jnp.zeros((1, 3, 1, 1), value.dtype)
            return template.reference(zeros, zeros, value, spans, scale=1.0)

        return jax.vjp(attend, value)[1](dout)[0]

    zeros = jnp.zeros((1, 3, 1, 1), dtype)
    dout = jnp.asarray([1.0, 2.0, -3.0], dtype).reshape(1, 3, 1, 1)
    shares = np.full((1, 3, 1, 1), np.abs(np.asarray(dout, np.float64)).sum() / 3)
    for found in (ours(zeros, dout), theirs(zeros, dout)):
        assert_fp32_reduction_bound(np.zeros((1, 3, 1, 1)), np.asarray(found, np.float64), shares, terms=3)

    def draw(seed: int):
        # Two cotangents drawn a channel, the third their negated sum: equal keys share them alike.
        drawn = jax.random.normal(jax.random.key(seed), (1, 2, 1, 8), jnp.float32)
        cancelling = jnp.concatenate([drawn, -drawn.sum(axis=1, keepdims=True)], axis=1).astype(dtype)
        return jnp.zeros((1, 3, 1, 8), dtype), cancelling

    assert_as_exact(jax.jit(ours), jax.jit(theirs), draw, f"cancelling cotangents {jnp.dtype(dtype).name}",
                    seeds=8)


def test_an_fp32_weight_that_normalizing_would_lose_still_reaches_dv():
    """fp32, one query over one key beside a resumed state of denominator 2^25 and maximum 87: the key's
    weight exp(-87) / 2^25 is below fp32's least subnormal, yet against a cotangent of 2^100 its gradient
    is a normal number, as the reference's derivative's is: kept apart from zero where the reference keeps
    it (the stability rule)."""
    spans = template.spans(1, 1, 1)
    zeros = jnp.zeros((1, 1, 1, 1), jnp.float32)
    state = template.Initial(jnp.full((1, 1, 1), 87.0), jnp.full((1, 1, 1), 2.0 ** 25),
                             jnp.zeros((1, 1, 1, 1)))
    hooks = template.Hooks(initial=state)
    dout = jnp.full((1, 1, 1, 1), 2.0 ** 100, jnp.float32)
    with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        ours = attention_triton.backward(zeros, zeros, zeros, spans, hooks, dout, scale=1.0,
                                         interpret=not ON_GPU).value
    _, pull = jax.vjp(lambda value: template.reference(zeros, zeros, value, spans, hooks, scale=1.0), zeros)
    [theirs] = pull(dout)
    assert float(theirs[0, 0, 0, 0]) > 0
    assert np.isfinite(np.asarray(ours)).all() and float(ours[0, 0, 0, 0]) >= np.finfo(np.float32).tiny


def test_a_small_fp16_cotangent_channel_beside_a_large_one_still_sums_into_dv():
    """fp16, 32 queries over 32 equal keys, cotangents 2 in one value channel and 2^-24 in the other:
    the small channel's 32 shares of 2^-29 still sum to 2^-24, as the reference's derivative gives."""
    spans = template.spans(32, 32, 1)
    zeros = jnp.zeros((1, 32, 1, 2), jnp.float16)
    dout = jnp.broadcast_to(jnp.asarray([2.0, 2.0 ** -24], jnp.float16), (1, 32, 1, 2))
    with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        ours = attention_triton.backward(zeros, zeros, zeros, spans, None, dout, scale=1.0,
                                         interpret=not ON_GPU).value
    _, pull = jax.vjp(lambda value: template.reference(zeros, zeros, value, spans, scale=1.0), zeros)
    [theirs] = pull(dout)
    np.testing.assert_array_equal(np.asarray(theirs, np.float32)[0, :, 0], [[2.0, 2.0 ** -24]] * 32)
    np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(theirs, np.float32))


def test_a_small_fp16_cotangent_beside_a_large_one_still_reaches_its_key():
    """fp16, two queries each seeing only its own key, cotangents 65504 and 2^-24: each key's gradient
    is its own query's, exactly, as the reference's derivative gives, the small beside the large."""
    spans = template.spans(2, 2, 1, chunk=1)
    zeros = jnp.zeros((1, 2, 1, 1), jnp.float16)
    dout = jnp.asarray([65504.0, 2.0 ** -24], jnp.float16).reshape(1, 2, 1, 1)
    with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        ours = attention_triton.backward(zeros, zeros, zeros, spans, None, dout, scale=1.0,
                                         interpret=not ON_GPU).value
    _, pull = jax.vjp(lambda value: template.reference(zeros, zeros, value, spans, scale=1.0), zeros)
    [theirs] = pull(dout)
    np.testing.assert_array_equal(np.asarray(theirs, np.float32).ravel(), [65504.0, 2.0 ** -24])
    np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(theirs, np.float32))


def test_an_fp16_weight_too_small_once_normalized_still_reaches_dv():
    """fp16, one query over keys whose third weight is about 5.3e-8 before its denominator of 2 and
    2.7e-8 after it, which fp16's least subnormal (6e-8) holds only before: dV's third row takes it, as
    the reference's derivative does, kept apart from zero where the reference keeps it (the stability
    rule)."""
    spans = template.spans(1, 3, 1)
    query = jnp.ones((1, 1, 1, 1), jnp.float16)
    key = jnp.asarray([0.0, 0.0, -16.75], jnp.float16).reshape(1, 3, 1, 1)
    value = jnp.zeros((1, 3, 1, 1), jnp.float16)
    dout = jnp.full((1, 1, 1, 1), 2.0, jnp.float16)
    with tuning.pinned({attention_triton.TritonBackward: joint(PINNED[0])}):
        ours = attention_triton.backward(query, key, value, spans, None, dout, scale=1.0,
                                         interpret=not ON_GPU).value
    _, pull = jax.vjp(lambda value: template.reference(query, key, value, spans, scale=1.0), value)
    [theirs] = pull(dout)
    assert float(theirs[0, 2, 0, 0]) > 0
    assert np.isfinite(np.asarray(ours, np.float32)).all() and float(ours[0, 2, 0, 0]) > 0


def test_the_backward_resumes_queries_that_read_no_key():
    """A resumed partial over causal keys that begin at row starts (the second row's first queries see
    no key and read only their state): every gradient on par, the initial maximum's 0 for those."""
    check_backward({"causal": True, "starts": True}, SCORE_HOOKS["resumed"], jnp.float32, PINNED[0])


@pytest.mark.parametrize("width, value_width", [(48, 24), (80, 48)])
def test_the_backward_is_on_par_at_head_widths_padded_to_powers_of_two(width, value_width):
    """Head widths that ride in zero-padded, values narrower than keys, bf16."""
    check_backward(CASES["causal-window"], (), jnp.bfloat16, PINNED[0], width=width, value_width=value_width)


@pytest.mark.parametrize("case", IRREGULAR)
def test_the_backward_walks_many_key_tiles(case):
    """The irregular calls (`IRREGULAR`) at 32-key tiles, a softcap, sinks and ALiBi, fp32: the first
    pass's maximum and anchor move across key tiles."""
    check_backward(IRREGULAR[case], SCORE_HOOKS["softcap-sinks-alibi"], jnp.float32, PINNED[0])


@pytest.mark.gpu(reason="every config compiles for the GPU")
@pytest.mark.parametrize("case", ["causal-documents-window", *IRREGULAR])
@pytest.mark.parametrize("config", attention_triton.BACKWARD_SPACE, ids=str)
def test_every_backward_config_on_a_gpu(config, case):
    """Every config of the backward's space, bf16, a softcap, sinks and ALiBi, over causal documents
    under a window and over the irregular full and causal calls (`IRREGULAR`): each config the launch takes
    at the call (`TritonBackward.fitting`: its threads within their registers, its programs within a block's
    shared memory) on par with the reference, and each other refused where it is pinned
    (`tuning.Unpinnable`), so no call runs it."""
    import tokamax

    from dew.nn.kernels import generation

    call = {**CASES, **IRREGULAR}[case]
    # The launch's operands as the backward pads them: whole blocks of queries and keys, 32 wide.
    queries, keys = (-(-count // attention_triton.BLOCK) * attention_triton.BLOCK for count in geometry(call))
    arrays = {"query": jax.ShapeDtypeStruct((ROWS, queries, 4, 32), jnp.bfloat16),
              "key": jax.ShapeDtypeStruct((ROWS, keys, 2, 32), jnp.bfloat16),
              "value": jax.ShapeDtypeStruct((ROWS, keys, 2, 32), jnp.bfloat16)}
    op = attention_triton.TritonBackward()
    takes = config in op.fitting(tokamax.BoundArguments(op, arrays), generation.gpu_target())
    tiles = attention_triton.Tiles

    def run():
        check_backward(call, SCORE_HOOKS["softcap-sinks-alibi"], jnp.bfloat16,
                       tiles(config.block_q, config.block_k, config.dq_warps, config.dq_stages),
                       tiles(config.block_q, config.block_k, config.dkv_warps, config.dkv_stages))

    if takes:
        run()
    else:
        with pytest.raises(tuning.Unpinnable):
            run()


SHARED_BYTES = (
    # (launch, width, config, the bytes Triton asked for where it refused, or None where it compiled, GPU)
    ("forward", 128, attention_triton.Tiles(128, 128, 8, 3), None, "sm80"),
    ("forward", 128, attention_triton.Tiles(128, 128, 8, 3), 163840, "sm86"),
    ("forward", 128, attention_triton.Tiles(32, 128, 4, 3), 147456, "sm86"),
    ("forward", 128, attention_triton.Tiles(64, 128, 4, 3), 163840, "sm86"),
    ("forward", 128, attention_triton.Tiles(128, 64, 4, 3), None, "sm86"),
    ("forward", 256, attention_triton.Tiles(64, 32, 4, 3), 102400, "sm86"),
    ("forward", 256, attention_triton.Tiles(128, 128, 4, 2), 229376, "sm86"),
    ("backward", 256, attention_triton.Tiles(32, 32, 4, 2), None, "sm86"),
    ("backward", 128, attention_triton.Tiles(128, 128, 4, 2), 167424, "sm80"),
    ("backward", 64, attention_triton.Tiles(128, 128, 4, 2), 101888, "sm86"),
)
"""Configs Triton compiled or refused on an A100 (sm80) and an A10 (sm86), bf16, from
bench_attention_smem.py's runs SMEM-a100 and SMEM-a10 (2026-10-21)."""


@pytest.mark.parametrize("launch, width, tiles, requested, generation_name", SHARED_BYTES)
def test_the_shared_memory_bound_keeps_what_triton_compiles_and_prunes_what_it_refuses(
        launch, width, tiles, requested, generation_name):
    """Each launch's shared-memory bound against what Triton did on a GPU: within the block's budget
    where it compiled (an A100's 128 by 128 forward tile in three stages, jax's mha's fastest there),
    and at or above the bytes it asked for, past the budget, where it refused."""
    import tokamax

    from dew.nn.kernels import generation

    arrays = {"query": jax.ShapeDtypeStruct((1, 1024, 4, width), jnp.bfloat16),
              "value": jax.ShapeDtypeStruct((1, 1024, 2, width), jnp.bfloat16)}
    if launch == "forward":
        op, config = attention_triton.TritonForward(), tiles
    else:
        op = attention_triton.TritonBackward()
        config = attention_triton.BackwardTiles(tiles.block_q, tiles.block_k, tiles.warps, tiles.stages,
                                                tiles.warps, tiles.stages)
    target = generation.TARGETS[generation_name]
    staged = op.staged_bytes(tokamax.BoundArguments(op, arrays), config, target)
    budget = generation.shared_budget(target)
    if requested is None:
        assert staged <= budget
    else:
        assert staged >= requested > budget


@pytest.mark.parametrize("target", ["sm80", "sm90"])
def test_a_hook_through_a_custom_vjp_lowers_and_compiles_on_tritons_compiler(target):
    """The softcap through a `jax.custom_vjp`, whose forward rule calls the hook itself as JAX's
    documented pattern does, as the score hook of the forward (the output hook too) and the backward,
    lowered as on the GPU and compiled by XLA's CUDA backend for a compile-only device on this CPU
    (tests/kernel_compile.py): no Pallas Triton lowering takes a custom-derivative call, which the
    interpreted suite never lowers, so a hook's call must reach the kernels as its primal."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_kernels, failures

    spans = template.spans(256, 256, 1, causal=True)
    score = template.Hooks(score_mod=lambda s, q_pos, k_pos: custom_softcap(s))
    hooks = template.Hooks(score_mod=score.score_mod, output_mod=custom_softcap)
    shape = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.float32)
    pins = {attention_triton.TritonForward: PINNED[0], attention_triton.TritonBackward: joint(PINNED[0])}

    def forward(query, key, value):
        with tuning.pinned(pins):
            return attention_triton.forward(query, key, value, spans, hooks, residuals=True)

    def backward(query, key, value, dout):
        with tuning.pinned(pins):
            grads = attention_triton.backward(query, key, value, spans, score, dout)
        return grads.query, grads.key, grads.value

    found = compile_kernels([KernelCase("forward", lambda: (forward, (shape,) * 3)),
                             KernelCase("backward", lambda: (backward, (shape,) * 4))], target)
    assert len(found) == 2
    assert not failures(found), "\n".join(failures(found))
    assert all(record["shared"] > 0 for record in found), found


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("sinks", [False, True], ids=["raw", "sinks"])
@pytest.mark.parametrize("target", ["sm80", "sm90"])
def test_the_accurate_form_lowers_and_compiles_on_tritons_compiler(target, sinks, dtype):
    """The accurate form (`natural`), raw (no initial state, `algebra.Forward.raw`) and from sinks: its
    forward with residuals (the plain launch that flags overflowing rows, the one that walks them again,
    the normalized pass) and its backward, lowered as on the GPU and compiled by XLA's CUDA backend for a
    compile-only device on this CPU (tests/kernel_compile.py): the interpreted suite lowers none of them,
    and the raw form's constant scale the algebra splits must reach no bitcast Pallas Triton cannot lower
    (`algebra.carried`, `precision.two_product`). In both 16-bit dtypes, whose cotangent products take
    their own scaled forms (`algebra.cotangent_product`)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_kernels, failures

    spans = template.spans(256, 256, 1, causal=True)
    initial = template.sinks(jnp.asarray([0.5, -1.0], jnp.float32), 1, 256, 64) if sinks else None
    hooks = template.Hooks(initial=initial)
    shape = jax.ShapeDtypeStruct((1, 256, 2, 64), dtype)
    pins = {attention_triton.TritonForward: PINNED[0], attention_triton.TritonBackward: joint(PINNED[0])}

    def forward(query, key, value):
        with tuning.pinned(pins):
            return attention_triton.forward(query, key, value, spans, hooks, residuals=True, natural=True)

    def backward(query, key, value, dout):
        with tuning.pinned(pins):
            grads = attention_triton.backward(query, key, value, spans, hooks, dout, natural=True)
        return grads.query, grads.key, grads.value

    found = compile_kernels([KernelCase("forward", lambda: (forward, (shape,) * 3)),
                             KernelCase("backward", lambda: (backward, (shape,) * 4))], target)
    assert len(found) >= 2
    assert not failures(found), "\n".join(failures(found))
