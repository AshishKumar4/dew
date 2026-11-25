"""Compile-point region recognition (`dew.algebra.recognize`, docs/design/algebra.md §2), split out of
`tests/test_algebra_fusion.py` so each file prepares well under the GPU tier's train bound (row 12643):
plain-program attention, norms, SwiGLU recognition, barrier refusals, and unwritten-region reporting."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_algebra_fusion import ran

from dew.nn.kernels import generation


@pytest.mark.mesh(devices=2)
def test_a_norm_over_rows_split_on_an_explicit_axis_is_reported_where_unwritten():
    """A plain program's RMSNorm over rows split on a mesh whose one axis is Explicit (a value's sharding its
    type), through `dew.jit` on an H100, traces: its normal form types each value it erases or lifts by that
    value's own sharding (`normal_form._erased`, `_lifted`: the whole result's spec on a value of another rank
    raised). Its region's call, traced over its inputs' shapes and dtypes, makes `bfloat16[256,1024]` beside
    the region's own `bfloat16[256@data,1024]`, so the region is not written and its route is reported
    refused (`fusion.UNWRITTEN`), where it was dropped with no report (row 10691). Typed whole there, a
    product region's replay contracted over the sharded axis with no output sharding, which JAX refuses."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    import dew
    from dew.algebra import fusion

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    program, _, (rows, weight) = NORM_FORMS["plain RMSNorm"]
    split, whole = NamedSharding(mesh, PartitionSpec("data")), NamedSharding(mesh, PartitionSpec())
    with jax.set_mesh(mesh), generation.assumed("sm90"), generation.routes() as found:
        dew.jit(program).trace(jax.ShapeDtypeStruct(rows, jnp.bfloat16, sharding=split),
                               jax.ShapeDtypeStruct(weight, jnp.bfloat16, sharding=whole))
    assert any(route.choice == "rms_norm" and route.refusal is not None
               and route.refusal.code == fusion.UNWRITTEN for route in found)


def test_a_region_whose_call_makes_other_types_is_reported_refused():
    """A region the compile point writes (a norm Dew's kernel takes on an H100, `recognize.recognized`) whose
    written call makes other types than the region's own (`jax_core.typematch`: here its outputs widened to
    float32) is not written, its equations kept as traced, and each route its call took is reported XLA's,
    refused for it (`fusion.UNWRITTEN`), where it was dropped with no report (row 10719), at the site its
    call took it from (row 10856: reported anew, it took the reporting frame's)."""
    from dew.algebra import fusion, recognize

    closed = jax.make_jaxpr(_norm_of)(jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                                      jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    with generation.assumed("sm90"):
        (region,) = recognize.recognized(closed, fusion.Index(closed), recognize.NORMS)
        widened = region._replace(call=lambda *values: jax.tree.map(
            lambda value: value.astype(jnp.float32), region.call(*values)), key=(region.key, "widened"))
        with generation.routes() as found:
            spliced = fusion._spliced(closed.jaxpr, [widened])
    refused = [route for route in found
               if route.refusal is not None and route.refusal.code == fusion.UNWRITTEN]
    assert spliced is closed.jaxpr and refused
    # Each reported as its call took it, its site the kernel's own dispatch's, not the report's (row 10856).
    assert all(route.site.split(":")[0] != fusion.__name__ for route in refused)


def test_an_unwritten_regions_route_to_xla_is_reported_as_it_ran():
    """A region its compile point does not write (its call making other types than its own) reports every
    route its call took: one to a kernel of Dew's as refused (`fusion.UNWRITTEN`), and one its call took to
    XLA already as it ran, which reaches no other report, where it was dropped (row 10879)."""
    from dew.algebra import fusion, recognize

    closed = jax.make_jaxpr(_norm_of)(jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                                      jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    with generation.assumed("sm90"):
        (region,) = recognize.recognized(closed, fusion.Index(closed), recognize.NORMS)

        def on_xla(*values):
            with generation.chosen(rms_norm="xla"):
                return jax.tree.map(lambda value: value.astype(jnp.float32), region.call(*values))

        widened = region._replace(call=on_xla, key=(region.key, "on xla, widened"))
        with generation.routes() as found:
            fusion._spliced(closed.jaxpr, [widened])
    assert any(route.choice == "rms_norm" and route.kernel == "xla" for route in found)



def jax_attention(*, causal: bool):
    """jax.nn's attention as a plain program writes it (its XLA form)."""
    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=causal, implementation="xla")
    return attend


def flax_attention(*, causal: bool, wide: bool = True):
    """Flax's attention as a Flax model writes it, its causal mask built from the rows, its softmax in
    float32 where `wide` (`force_fp32_for_softmax`, the kernels' arithmetic) and Flax's default, the
    operands' dtype, else."""
    from flax import linen as nn

    def attend(q, k, v):
        mask = nn.make_causal_mask(jnp.ones(q.shape[:2])) if causal else None
        return nn.dot_product_attention(q, k, v, mask=mask, force_fp32_for_softmax=wide)
    return attend


SPELLINGS = {"jax.nn causal, grouped": (jax_attention(causal=True), 2, True),
             "jax.nn, grouped": (jax_attention(causal=False), 2, False)}
"""Each plain program's attention, its key heads over 4 query heads, and whether it is causal."""

FLAX_SPELLINGS = {"softmax in bfloat16": False, "probabilities in float32": True}
"""Flax's attention, its softmax in its operands' bfloat16 (Flax's default) or in float32
(`force_fp32_for_softmax`), each different arithmetic from Dew's call: the one a softmax no kernel of Dew's
takes, the other the float32 probabilities carried into the value product unrounded (row 4860)."""


@pytest.mark.parametrize("spelling", list(FLAX_SPELLINGS))
def test_flaxs_attention_computes_as_written(spelling):
    """Flax's attention in either form (`FLAX_SPELLINGS`), compiled through `dew.jit` on an H100, runs as
    written, causal or not: no region is written as a kernel of Dew's (row 4783)."""
    import dew

    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),) * 3
    with generation.assumed("sm90"):
        for causal in (False, True):
            attend = flax_attention(causal=causal, wide=FLAX_SPELLINGS[spelling])
            assert not _routes("attention", lambda attend=attend: dew.jit(attend).trace(*shapes))


def _routes(choice: str, trace) -> list[str]:
    """The kernels the routes of `choice` that `trace()` reports name."""
    with generation.routes() as found:
        trace()
    return [route.kernel for route in found if route.choice == choice]


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_a_plain_programs_attention_runs_dews_kernel_through_dew_jit(spelling):
    """A plain JAX or Flax program's softmax attention (`dew.algebra.recognize`) compiled through
    `dew.jit` on an H100 runs the kernels Dew's own attention call routes to (`scaled_dot_product_attention`,
    or the template a call names), where the program as written runs XLA's; on this host, its output and
    gradients under the template (its portable mainloop) on par with the program as written (check 1)."""
    from attention_oracle import attention_on_par, kept
    from contract_support import normal_drawn

    import dew
    from dew.algebra.recognize import KEPT
    from dew.nn.attention import scaled_dot_product_attention

    attend, key_heads, causal = SPELLINGS[spelling]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),
              *(jax.ShapeDtypeStruct((2, 128, key_heads, 64), jnp.bfloat16),) * 2)

    def own(q, k, v):
        return scaled_dot_product_attention(q, k, v, scale_logits=True, causal=causal)

    with generation.assumed("sm90"):
        assert not _routes("attention", lambda: jax.jit(attend).trace(*shapes))
        expected = {kernel for kernel in _routes("attention", lambda: jax.jit(own).trace(*shapes))
                    if kernel not in KEPT}
        assert expected and set(_routes("attention", lambda: dew.jit(attend).trace(*shapes))) == expected
        with generation.chosen(attention="template"):
            assert "template" in _routes("attention", lambda: dew.jit(attend).trace(*shapes))
    draw = normal_drawn(37, (2, 64, 4, 32), (2, 64, key_heads, 32), (2, 64, key_heads, 32), (2, 64, 4, 32),
                        dtype=jnp.bfloat16)
    with generation.chosen(attention="template"):
        attention_on_par(dew.jit(attend), attend, draw, f"dew.jit {spelling}",
                         keep=kept(64, 64, causal=causal))


def input_mask(q, k):
    """A mask drawn from the queries, `[B, H, Q, K]`: each query row keeps every key where its first entry is
    positive, none else."""
    rows = (q[..., :1] > 0).transpose(0, 2, 1, 3)
    return jnp.broadcast_to(rows, (q.shape[0], q.shape[2], q.shape[1], k.shape[1]))


def input_bias(q, k):
    """A bias drawn from the queries, `[B, H, Q, K]`: a query row's first entry over every key."""
    rows = (q[..., 0] * 0.25).transpose(0, 2, 1)[..., None]
    return jnp.broadcast_to(rows, (q.shape[0], q.shape[2], q.shape[1], k.shape[1])).astype(q.dtype)


def _wide(x):
    """The dtype a plain program computes `x`'s logits or statistics in: float32 over bfloat16 or float32
    operands, as its author writes it, and the operands' own where they are wider, so the program run on
    float64 operands is its float64 truth and no second fp32 run."""
    return jnp.promote_types(x.dtype, jnp.float32)


SOFTCAP = 50.0


def softcapped_attention(q, k, v):
    """Gemma 2's attention as a plain program writes it: scaled logits softcapped at 50, causal."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
    logits = SOFTCAP * jnp.tanh(logits / SOFTCAP)
    causal = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), bool))
    weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


JAX_FILL = -0.7 * float(jnp.finfo(jnp.float32).max)
"""jax.nn's fill for masked logits (its `_get_large_negative` at float32)."""


def keyed_mask(q, k):
    """`input_mask` with every query row keeping its first key: no row wholly masked."""
    return input_mask(q, k).at[..., 0].set(True)


def masked_attention(mask):
    """Attention written out over `mask(q, k)`, its masked logits -inf: a row the mask leaves no key is
    NaN."""
    def attend(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
        weights = jax.nn.softmax(jnp.where(mask(q, k), logits, -jnp.inf), axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return attend


def jax_nn_written(*, mask=None, bias=None, scale=1, causal=False, window=None):
    """`jax.nn.dot_product_attention` under those arguments, written out at its operands' width: its logits
    scaled (`scale` times the default), biased, then filled with jax.nn's large negative (`JAX_FILL`)
    wherever the mask of the inputs, causality or a local window of `window` keys back drops a key. jax.nn's
    own runs its softmax in float32 whatever its operands' (`_dot_product_attention_core`), so on float64
    operands it is a second fp32 run and no truth of itself; this is the function it computes."""
    def attend(q, k, v):
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=_wide(q))
        logits = logits * q.shape[-1] ** -0.5 * scale
        if bias is not None:
            logits = logits + bias(q, k)
        kept = jnp.ones(logits.shape, bool) if mask is None else mask(q, k)
        rows, keys = jnp.arange(q.shape[1])[:, None], jnp.arange(k.shape[1])[None, :]
        if causal:
            kept &= keys <= rows
        if window is not None:
            kept &= rows <= keys + window
        weights = jax.nn.softmax(jnp.where(kept, logits, JAX_FILL), axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return attend


WINDOW = 16
"""The causal window's keys a query sees, itself among them (jax.nn's lookback of WINDOW - 1): one width its
program, Dew's call and the oracle's keep set read."""

FORMS = {
    "a mask of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, mask=input_mask(q, k), implementation="xla"),
        lambda q, k, v: {"mask": input_mask(q, k), "mask_fill": JAX_FILL}, jax_nn_written(mask=input_mask)),
    "a mask of its inputs filled with -inf": (
        masked_attention(keyed_mask), lambda q, k, v: {"mask": keyed_mask(q, k), "mask_fill": -jnp.inf},
        masked_attention(keyed_mask)),
    "a bias of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, bias=input_bias(q, k), implementation="xla"),
        lambda q, k, v: {"bias": input_bias(q, k)}, jax_nn_written(bias=input_bias)),
    "twice the default scale": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, scale=2 * q.shape[-1] ** -0.5,
                                                     implementation="xla"),
        None, jax_nn_written(scale=2)),
    "a causal window": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                                     local_window_size=(WINDOW - 1, 0), implementation="xla"),
        lambda q, k, v: {"causal": True, "sliding_window": WINDOW, "mask_fill": JAX_FILL},
        jax_nn_written(causal=True, window=WINDOW - 1)),
    "a softcap": (softcapped_attention,
                  lambda q, k, v: {"causal": True, "softcap": SOFTCAP, "mask_fill": -jnp.inf},
                  softcapped_attention),
}
"""Each plain program's attention under a form beside the default scale and a causal mask, the arguments
Dew's own attention takes for it, its fill the program's (None where no call of Dew's computes it: a scale
other than the default is another constant, which the compile point keeps as written), and the program
whose run on float64 operands is its truth: itself where it is written at its operands' width, jax.nn's
written out (`jax_nn_written`) where it is jax.nn's. The masks leave rows partly masked (a window,
causality, -inf's mask) and wholly masked (jax.nn's mask, uniform rows under its finite fill); a row -inf
masks wholly is NaN, as written, which check 1 does not compare."""


@pytest.mark.parametrize("form", list(FORMS))
def test_a_plain_programs_attention_under_each_form_runs_the_kernel_dews_own_call_takes(form):
    """A plain program's attention under a mask (filled as jax.nn fills it, or with -inf) or a bias it draws
    from its inputs, a causal window or a softcap, compiled through `dew.jit` on an H100, runs the kernel
    Dew's own attention call under those arguments routes to (none where that is XLA's), and under a scale
    no call of Dew's takes runs as written, XLA's; on this host, under the template, its output and
    gradients on par with the program as written (check 1)."""
    from attention_oracle import attention_on_par, kept
    from contract_support import normal_drawn

    import dew
    from dew.algebra.recognize import KEPT
    from dew.nn.attention import scaled_dot_product_attention

    attend, arguments, exact = FORMS[form]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),) * 3

    def own(q, k, v):
        return scaled_dot_product_attention(q, k, v, scale_logits=True, **arguments(q, k, v))

    with generation.assumed("sm90"):
        assert not _routes("attention", lambda: jax.jit(attend).trace(*shapes))
        expected = [] if arguments is None else [kernel for kernel in _routes(
            "attention", lambda: jax.jit(own).trace(*shapes)) if kernel not in KEPT]
        assert set(_routes("attention", lambda: dew.jit(attend).trace(*shapes))) == set(expected)
    drawn = (2, 64, 4, 32)
    draw = normal_drawn(41, *(drawn,) * 4, dtype=jnp.bfloat16)
    # The draw's rows by its declared shape, no draw computed: a GPU tier part's prepare computes nothing on
    # its compile-only device (bible §8).
    queries = keys = drawn[1]
    # Size the structured masks' singleton rows by the terms the fast backward subtracts (dP and D),
    # not by the written softmax's already-cancelled gradient. The causal window's finite fill contributes
    # no weight on these draws.
    if form == "a softcap":
        oracle = {"keep": kept(queries, keys, causal=True), "softcap": SOFTCAP}
    elif form == "a causal window":
        oracle = {"keep": kept(queries, keys, causal=True, sliding_window=WINDOW)}
    else:
        oracle = {"truth": lambda *x: exact(*(a.astype(jnp.float64) for a in x))}
    with generation.chosen(attention="template"):
        attention_on_par(dew.jit(attend), attend, draw, f"dew.jit {form}", **oracle)

def written_out(*, fill=-jnp.inf, logits_cast=None, initial=-jnp.inf, other_rows=False, sum_cast=None,
                root=None):
    """Causal softmax attention as a program writes it out (its products as einsums, a causal where, the
    safe softmax), each keyword a way it computes something other than attention: a finite fill keeping the
    future keys' weight, logits truncated or narrowed through `logits_cast`, a maximum's initial value above
    the logits, each row divided by another row's sum (heads and rows swapped in its broadcast), the sum
    narrowed through `sum_cast` (to float16, where 65536 keys' sum overflows), the queries divided by a
    `root` near sqrt(D) but not it (Flax's division, its divisor another scale). Under a `root` it is
    Flax's attention as Flax writes it, its masked logits its dtype's least value before they widen."""
    def attend(q, k, v):
        if root is not None:
            # Flax's (`dot_product_attention_weights`): the queries divided before the product, no scale after
            # it, the logits in the operands' dtype masked there with its least value, then widened.
            s = jnp.einsum("bthd,bshd->bhts", q / jnp.float32(root).astype(q.dtype), k)
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, jnp.finfo(s.dtype).min)
            s = s.astype(_wide(s))
        else:
            s = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
            if logits_cast is not None:
                s = s.astype(logits_cast).astype(_wide(q))
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, fill)
        e = jnp.exp(s - jax.lax.stop_gradient(jnp.max(s, -1, keepdims=True, initial=initial)))
        total = jnp.sum(e, -1)
        if sum_cast is not None:
            total = total.astype(sum_cast).astype(total.dtype)
        if other_rows:
            total = jax.lax.broadcast_in_dim(total, (*e.shape[:-1], 1), (0, 2, 1))
        else:
            total = total[..., None]
        return jnp.einsum("bhts,bshd->bthd", (e / total).astype(v.dtype), v)
    return attend


def test_a_programs_own_choice_by_the_platform_stays_as_written():
    """A plain program's attention whose weights pass through a choice by the platform it lowers for, which
    doubles them on CUDA alone, is no call of Dew's on an H100: its choice computes another function there,
    so the compile point keeps the program as written, XLA's, where the same attention without it runs Dew's
    kernel."""
    import dew

    def attend(doubled):
        def run(q, k, v):
            s = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=_wide(q)) * q.shape[-1] ** -0.5
            s = jnp.where(jnp.tril(jnp.ones(s.shape[-2:], bool)), s, -jnp.inf)
            weights = jax.nn.softmax(s, axis=-1)
            if doubled:
                weights = jax.lax.platform_dependent(weights, cuda=lambda w: 2 * w, default=lambda w: w)
            return jnp.einsum("bhts,bshd->bthd", weights.astype(v.dtype), v)
        return run

    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), jnp.bfloat16),) * 3
    with generation.assumed("sm90"):
        assert _routes("attention", lambda: dew.jit(attend(doubled=False)).trace(*shapes))
        assert not _routes("attention", lambda: dew.jit(attend(doubled=True)).trace(*shapes))


@pytest.mark.mesh(devices=2)
def test_a_plain_programs_masked_attention_compiles_under_a_sequence_axis():
    """A plain program's causal attention, its masked logits filled, compiled through `dew.jit` on a mesh
    that splits the sequence, where Dew's call takes no fill (its exchanges weigh a masked key nothing):
    no candidate the call refuses is tried, and the program computes as written."""
    import dew
    from dew.training import MeshSpec

    attend = written_out()
    mesh = MeshSpec(sequence=2).build(jax.devices()[:2])
    q, k, v = (jax.random.normal(key, (1, 16, 4, 32), jnp.float32)
               for key in jax.random.split(jax.random.key(7), 3))
    with jax.set_mesh(mesh):
        ours, written = dew.jit(attend)(q, k, v), jax.jit(attend)(q, k, v)
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(written))


@pytest.mark.parametrize(("variant", "options"), [("attention", {}), ("finite fill", {"fill": 0.0}),
                                                  ("a fill of NaN", {"fill": jnp.nan}),
                                                  ("truncated logits", {"logits_cast": jnp.int32}),
                                                  ("initial above", {"initial": 1e30}),
                                                  ("another row's sum", {"other_rows": True}),
                                                  ("logits narrowed", {"logits_cast": jnp.float16}),
                                                  ("sum narrowed", {"sum_cast": jnp.float16}),
                                                  ("attention by a root", {"root": 32 ** 0.5}),
                                                  ("attention by a root", {"root": 32 ** 0.5,
                                                                           "dtype": jnp.float32}),
                                                  ("a root near sqrt(D)", {"root": 32 ** 0.5 * 1.0005,
                                                                           "dtype": jnp.float32})])
def test_only_attention_written_out_is_rewritten_as_dews(variant, options):
    """Causal softmax attention written out runs Dew's kernel through `dew.jit` on an H100, scaled after its
    product or its queries divided by sqrt(D) before it (Flax's), its masked logits filled with -inf or a
    finite constant (Dew's call's `mask_fill`, the future keys weighed as that fill weighs them); written out
    with any step computing otherwise (a NaN fill, which no call of Dew's takes, truncated or narrowed logits,
    an initial maximum above the logits, a row normalized by another's sum, a sum narrowed past its range, a
    divisor near sqrt(D) but not it, in float32, where bfloat16 holds no value that near), it stays as
    written, XLA's."""
    import dew

    options = dict(options)
    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), options.pop("dtype", jnp.bfloat16)),) * 3
    with generation.assumed("sm90"):
        routes = _routes("attention", lambda: dew.jit(written_out(**options)).trace(*shapes))
    assert bool(routes) == (variant in ("attention", "finite fill", "attention by a root")), routes


@pytest.mark.parametrize("order", [("xla", "template"), ("template", "xla")],
                         ids=["xla first", "template first"])
def test_a_bodys_attention_takes_each_scopes_kernel_choice(order):
    """Written-out attention inside a scanned body of a program held as a jaxpr, which every trace of it
    binds again as the same body, traced through `dew.jit` under one kernel choice and then another, in
    either order: each trace takes its own scope's choice, the template's kernel under
    `attention="template"` and none under `attention="xla"`, not the body the earlier scope rewrote."""
    from jax.extend.core import jaxpr_as_fun

    import dew

    attend = written_out()

    def layers(q, k, v):
        def one(carried, _):
            return attend(carried, k, v), None
        return jax.lax.scan(one, q, None, length=2)[0]

    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), jnp.bfloat16),) * 3
    held = jaxpr_as_fun(jax.make_jaxpr(layers)(*shapes))
    with generation.assumed("sm90"):
        for choice in order:
            with dew.kernels.use(attention=choice):
                routes = _routes("attention", lambda: dew.jit(held).trace(*shapes))
            assert bool(routes) == (choice == "template"), (choice, routes)


def flax_rms(x, w):
    """Flax's RMSNorm as a Flax model applies it."""
    from flax import linen as nn

    return nn.RMSNorm(epsilon=1e-6, dtype=x.dtype).apply({"params": {"scale": w}}, x)


def widened_rms(x, w):
    """An RMSNorm as a plain program writes it: float32 statistics, the weight before the cast."""
    widened = x.astype(_wide(x))
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w).astype(x.dtype)


def offset_rms(x, w):
    """Gemma's RMSNorm: the normalized row cast first, then times one more than its weight."""
    widened = x.astype(_wide(x))
    normed = widened * jax.lax.rsqrt(jnp.mean(jnp.square(widened), axis=-1, keepdims=True) + 1e-6)
    return normed.astype(x.dtype) * (1 + w.astype(x.dtype))


def gated_product_of(a, b):
    """A gated MLP's product as a plain program writes Dew's (torch's) math: the activation in float32,
    rounded once to the gate's dtype, then times the up projection."""
    return jax.nn.silu(a.astype(_wide(a))).astype(a.dtype) * b


def dews_gated_product(a, b):
    """A gated MLP's product written with Dew's own silu (`activations.silu`: its sigmoid Dew's one logistic,
    a declared derivative over a span stating it), as Dew's layers write theirs (TSSM's, Mamba-3's gate)."""
    from dew.nn.activations import silu

    return silu(a) * b


def _norms_rewritten(fn, *operands):
    """`fn` with its norms written as the compile point writes them on an H100 (`_norms_written`), its first
    output, made where it is traced at `operands`' shapes: a program of it is stated by `fn`
    (`truths_ahead.arm_role`), where the written jaxpr's function holds JAX's token, which no statement
    holds."""
    shapes = [jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for x in operands]
    written, _ = _norms_written(fn, *shapes)
    return written(*operands)[0]


def _norms_written(fn, *shapes, families=None):
    """`fn` with each region the compile point takes on an H100 under `families` (`recognize.NORMS` by
    default) written as Dew's call (`recognize.recognized`), decided under the H100 and written on this
    host, whose calls run their portable arithmetic: what the compile point's rewrite computes, run here."""
    from jax.extend.core import jaxpr_as_fun

    from dew.algebra import fusion, recognize

    closed = jax.make_jaxpr(fn)(*shapes)
    with generation.assumed("sm90"):
        written = recognize.recognized(closed, fusion.Index(closed),
                                       recognize.NORMS if families is None else families)
    return jaxpr_as_fun(closed.replace(jaxpr=fusion._spliced(closed.jaxpr, written))), len(written)


def residual_rms(x, branch, w):
    """A pre-norm block's seam as a plain program writes it: the stream `x + branch`, its RMSNorm, and the
    stream read on after it."""
    stream = x + branch
    return widened_rms(stream, w) + stream


def gated_rms(x, gate, w):
    """Qwen3-Next's gated RMSNorm as a plain program writes it: fp32 statistics, the weight, then the gate's
    SiLU, cast once."""
    widened = x.astype(_wide(x))
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w.astype(_wide(w)) * jax.nn.silu(gate.astype(_wide(gate)))).astype(x.dtype)


def packed_swiglu(gate_up):
    """A packed gate|up projection's SwiGLU as a plain program writes Dew's math (`gated_product_of`): its
    two halves sliced out."""
    gate = gate_up[..., 0, :]
    return jax.nn.silu(gate.astype(_wide(gate))).astype(gate.dtype) * gate_up[..., 1, :]


def _own_norm(*, offset: bool, after_cast: bool):
    from dew.nn.norms import rms_normalized

    return lambda x, w: rms_normalized(x, w, 1e-6, x.dtype, scale_offset=offset, scale_after_cast=after_cast,
                                       fp32_statistics=True)


def _own_residual(x, branch, w):
    from dew.nn.norms import stream_rms_normalized

    stream, normed = stream_rms_normalized(x, branch, w, 1e-6, x.dtype, scale_offset=False,
                                           scale_after_cast=False, fp32_statistics=True)
    return normed + stream


def _own_gated(x, gate, w):
    from dew.nn.norms import gated_rms_normalized

    return gated_rms_normalized(x, gate, w, 1e-6, x.dtype, "silu")


def _own_swiglu(gate, up):
    from dew.nn.activations import gated_product

    return gated_product("swiglu")(gate, up)


def _own_packed(gate_up):
    from dew.nn.activations import packed_gated_product

    return packed_gated_product("swiglu")(gate_up)


ROWS, WEIGHT, PACKED = (256, 1024), (1024,), (256, 2, 1024)
NORM_FORMS = {
    "flax RMSNorm": (flax_rms, _own_norm(offset=False, after_cast=False), (ROWS, WEIGHT)),
    "plain RMSNorm": (widened_rms, _own_norm(offset=False, after_cast=False), (ROWS, WEIGHT)),
    "Gemma RMSNorm": (offset_rms, _own_norm(offset=True, after_cast=True), (ROWS, WEIGHT)),
    "residual RMSNorm": (residual_rms, _own_residual, (ROWS, ROWS, WEIGHT)),
    "gated RMSNorm": (gated_rms, _own_gated, (ROWS, ROWS, WEIGHT)),
    "SwiGLU": (gated_product_of, _own_swiglu, (ROWS, ROWS)),
    "SwiGLU of Dew's silu": (dews_gated_product, _own_swiglu, (ROWS, ROWS)),
    "packed SwiGLU": (packed_swiglu, _own_packed, (PACKED,)),
}
"""Each plain program's fused norm, Dew's own call of it, and its operands' shapes."""


@pytest.mark.parametrize("form", list(NORM_FORMS))
def test_a_plain_programs_norms_run_the_kernel_dews_own_call_takes(form):
    """A plain program's RMSNorm (Flax's, a plain one, Gemma's offset weight after the cast, a residual
    stream's read on after it, Qwen3-Next's gated one) or SwiGLU (two projections, its sigmoid Dew's one
    logistic too, which train 1026's review found no anchor read, or one packed one's halves), compiled
    through `dew.jit` on an H100, runs the kernel Dew's own call routes it to (none where
    that is XLA's arithmetic), where the program as written runs XLA's; on this host, its output and
    gradients on par with the program as written (check 1)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients

    import dew

    program, own, sizes = NORM_FORMS[form]
    shapes = [jax.ShapeDtypeStruct(size, jnp.bfloat16) for size in sizes]

    def fused(trace) -> list[str]:
        with generation.routes() as found:
            trace()
        return [route.kernel for route in found if route.choice in ("rms_norm", "swiglu")
                and route.kernel != "xla"]

    with generation.assumed("sm90"):
        assert not fused(lambda: jax.jit(program).trace(*shapes))
        expected = set(fused(lambda: jax.jit(own).trace(*shapes)))
        assert set(fused(lambda: dew.jit(program).trace(*shapes))) == expected
    output = jax.eval_shape(program, *shapes).shape
    draw = normal_drawn(43, *sizes, output, dtype=jnp.bfloat16)
    _, found = _norms_written(program, *shapes)
    assert found == 1
    assert_as_exact(jax.jit(with_gradients(functools.partial(_norms_rewritten, program))),
                    jax.jit(with_gradients(program)),
                    draw, f"dew.jit {form}", seeds=8, truth_fn=with_gradients(
                        lambda *x: program(*(a.astype(jnp.float64) for a in x))))


def checkpointed_swiglu(a, b):
    """A SwiGLU whose sigmoid of the fp32-widened gate alone is checkpointed (`jax.checkpoint`): what the
    derivative saves is the user's choice at that boundary."""
    widened = a.astype(_wide(a))
    return (widened * jax.checkpoint(jax.nn.sigmoid)(widened)).astype(a.dtype) * b


def _rows_inverse(x):
    """Each row's inverse root mean square, fp32 `[rows]`, the feature axis reduced."""
    widened = x.astype(_wide(x))
    return jax.lax.rsqrt(jnp.sum(widened * widened, axis=-1) / x.shape[-1] + 1e-6)


def _named_silu(x):
    return jnp.tanh(x)


_named_silu.__name__ = "silu"


NEAR_NORMS = {
    "statistics in bfloat16, the widening dropped": lambda x, w: (
        x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w.astype(x.dtype)),
    "the weight applied before the statistic": lambda x, w: (
        _rows_inverse(x * w.astype(x.dtype))[:, None] * x.astype(_wide(x))).astype(x.dtype),
    "a SwiGLU whose sigmoid rounds to bfloat16": lambda x, w: jax.nn.silu(x) * (x * w.astype(x.dtype)),
    "columns scaled by the rows' statistics":
        lambda x, w: (x.astype(_wide(x)) * _rows_inverse(x)[None, :]).astype(x.dtype),
    "a weight along the rows":
        lambda x, w: (x.astype(_wide(x)) * _rows_inverse(x)[:, None] * w[:, None]).astype(x.dtype),
    "a jit named silu that is tanh": lambda x, w: jax.jit(_named_silu)(x) * w.astype(x.dtype),
    "a zero mean multiplier": lambda x, w: (x.astype(_wide(x)) * jax.lax.rsqrt(
        jnp.sum(jnp.square(x.astype(_wide(x))), axis=-1, keepdims=True) * 0.0 + 1e-6)).astype(x.dtype),
    "its own sum as the weight": lambda x, w: (
        x.astype(_wide(x)) * _rows_inverse(x)[:, None]
        * jnp.sum(jnp.square(x.astype(_wide(x))), axis=-1)[None, :]).astype(x.dtype),
    "a weight rounded to bf16 before it weighs": lambda x, w: (
        x.astype(_wide(x)) * _rows_inverse(x)[:, None]
        * (w.astype(_wide(w)) * 1.001).astype(jnp.bfloat16).astype(_wide(w))).astype(x.dtype),
}
"""Programs a norm's recogniser reads almost as an RMSNorm or SwiGLU and computes otherwise (a square row,
so every axis has the feature axis's extent)."""


@pytest.mark.parametrize("width", [256, 256.001])
def test_a_norm_is_its_mean_over_the_rows_width_alone(width):
    """A row's sum of squares over its width is its mean, which the compile point writes as Dew's norm; over
    a divisor near the width, closer than any output of the row's dtype shows, it is no norm, and the
    compile point writes none."""
    def program(x, w):
        widened = x.astype(_wide(x))
        inverse = jax.lax.rsqrt(jnp.sum(jnp.square(widened), axis=-1, keepdims=True) / width + 1e-6)
        return (widened * inverse * w.astype(_wide(w))).astype(x.dtype)

    _, found = _norms_written(program, jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                              jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    assert found == (width == 256)


def _two_results(x):
    """Two rows of one shape from one nested program: one equation making both."""
    return jax.jit(lambda x: (x * 2, x * 3))(x)


def _norm_of(rows, w):
    widened = rows.astype(_wide(rows))
    return (widened * jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
            * w.astype(_wide(w))).astype(rows.dtype)


def _first_normed(x, w):
    a, b = _two_results(x)
    return _norm_of(a, w), a, b


def _second_normed(x, w):
    a, b = _two_results(x)
    return _norm_of(b, w), a, b


def _divided_by(divisor):
    """A norm-shaped program whose statistic divides by `divisor`, a constant array it closes over: the host's
    (NumPy's), which a statement of the program states by its values alone, where JAX's is stated by its
    device too, the prepare's CPU and the run's GPU two programs (rows 10008, 10009: the run found no
    reference its prepare compiled)."""
    held = np.asarray([divisor], np.float32)

    def program(x, w):
        widened = x.astype(_wide(x))
        mean = jnp.sum(widened * widened, axis=-1, keepdims=True) / held
        return (widened * jax.lax.rsqrt(mean + 1e-6) * w.astype(_wide(w))).astype(x.dtype)
    return program


def _permuted_statistic(x, w):
    """A row normalized by the statistic of its elements read in another order."""
    swapped = jax.lax.reshape(x, x.shape, dimensions=(1, 0))
    widened, other = x.astype(_wide(x)), swapped.astype(_wide(swapped))
    inverse = jax.lax.rsqrt(jnp.mean(other * other, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w.astype(_wide(w))).astype(x.dtype)


def _stream_read_early(x, branch, w):
    """A residual stream another value reads before its norm does."""
    stream = x + branch
    early = jnp.sin(stream)
    return early, _norm_of(stream, w)


WRITTEN_AS_PROGRAMS = {
    "the first of two results normed": (_first_normed, 1),
    "the second of two results normed": (_second_normed, 1),
    "a statistic over the width": (_divided_by(256.0), 1),
    "a statistic over twice the width": (_divided_by(512.0), 0),
    "a statistic of the elements in another order": (_permuted_statistic, 0),
}
"""Programs the compile point writes, each with the norms it writes in it: the same structure apart only
in which result is normed or in a constant it closes over, each weighed for its own (row 4783's review)."""


@pytest.mark.parametrize("name", list(WRITTEN_AS_PROGRAMS))
def test_the_compile_point_writes_a_program_by_its_own_values(name):
    """Each program (`WRITTEN_AS_PROGRAMS`), written after the others in either order, writes the norms it
    holds and computes what it does as written, on par from float64 (check 1: the norm's call, a GPU's kernel
    summing in its own order, and the program as written are two programs XLA promises no common bits): a
    region is the call's only where the call computes the region's own arithmetic, a statistic over twice the
    width or of permuted elements written as the norm off by its whole factor."""
    from contract_support import normal_draw
    from reference_error import as_given, assert_as_exact

    from dew.parity import FEWEST_SEEDS

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    names = list(WRITTEN_AS_PROGRAMS)
    for other in (*names[names.index(name) + 1:], *names[:names.index(name)]):
        _norms_written(WRITTEN_AS_PROGRAMS[other][0], *shapes)
    program, norms = WRITTEN_AS_PROGRAMS[name]
    written, found = _norms_written(program, *shapes)
    assert found == norms
    assert_as_exact(as_given(lambda x, w: tuple(written(x, w))),
                    lambda x, w: tuple(jax.tree.leaves(program(x, w))),
                    lambda seed: normal_draw(seed, (256, 256), (256,), dtype=jnp.bfloat16),
                    f"the compile point's writing of {name}", seeds=FEWEST_SEEDS)


def test_a_stream_read_before_its_norm_computes_as_written():
    """A residual stream another value reads before its norm, compiled through the rewrite, computes as
    written, on par from float64 (check 1, two programs: the norm's call and the stream's sum as written):
    the stream's sum stays where its first reader needs it."""
    from contract_support import normal_draw
    from reference_error import as_given, assert_as_exact

    from dew.parity import FEWEST_SEEDS

    rows = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    shapes = (rows, rows, jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    written, _ = _norms_written(_stream_read_early, *shapes)
    assert_as_exact(as_given(lambda x, branch, w: tuple(written(x, branch, w))), _stream_read_early,
                    lambda seed: normal_draw(seed, (256, 256), (256, 256), (256,), dtype=jnp.bfloat16),
                    "a stream read before its norm, written", seeds=FEWEST_SEEDS)


def test_a_norm_of_a_stopped_row_passes_no_gradient_to_it():
    """A norm of a row whose gradient the program stops, rewritten, passes the row no gradient, as written."""
    def program(x, w):
        return _norm_of(jax.lax.stop_gradient(x), w).astype(jnp.float32).sum()

    from contract_support import normal_draw

    x, w = normal_draw(4785, (256, 256), (256,), dtype=np.float32)
    grad = jax.grad(functools.partial(_norms_rewritten, program))
    assert not np.any(np.asarray(ran(grad, "a stopped row's gradient, its norm written", x, w)))


def test_a_gate_less_a_negative_zero_keeps_its_sign_as_written():
    """A SwiGLU whose gate is the program's `g - (-0.0)`, at a gate of -0.0, computes as written bit for
    bit: subtracting a negative zero makes a negative zero positive, which no rewrite drops."""
    def program(g, u):
        gate = g - jnp.asarray(-0.0, g.dtype)
        return gated_product_of(gate, u)

    g = np.full((256, 256), -0.0, jnp.bfloat16)
    u = np.ones((256, 256), jnp.bfloat16)
    np.testing.assert_array_equal(
        np.signbit(np.asarray(ran(functools.partial(_norms_rewritten, program), "the gate, its norm written",
                                  g, u))),
        np.signbit(np.asarray(ran(program, "the gate as written", g, u))))


@jax.custom_jvp
def _doubled_identity(x):
    return x


@_doubled_identity.defjvp
def _doubled_identity_tangent(primals, tangents):
    return primals[0], 2 * tangents[0]


def test_a_norm_of_a_row_under_its_own_derivative_rule_keeps_the_rule():
    """A norm of a row a nested program's own derivative rule makes (a jitted custom JVP doubling its
    tangent), rewritten, keeps that rule: the row's gradient twice the norm's own, as written."""
    def program(x, w):
        return _norm_of(jax.jit(_doubled_identity)(x), w).astype(jnp.float32)[0, 0]

    from contract_support import normal_draw

    x, w = normal_draw(4786, (256, 256), (256,), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(ran(jax.grad(functools.partial(_norms_rewritten, program)),
                       "the row's gradient, its norm written", x, w)),
        np.asarray(ran(jax.grad(program), "the row's gradient as written", x, w)))


def test_a_norm_of_a_row_an_effect_reads_keeps_the_effect():
    """A norm of a row a nested program passes through a host callback, rewritten, keeps the callback: the
    program's effects are the program's as written, the callback's, which XLA runs as written. Nothing
    computes: a program holding a callback of this process's compiles for no compile-only GPU (a GPU tier
    part's prepare) and runs from no executable another process compiled."""
    from jax.experimental import io_callback

    def recorded(x):
        io_callback(lambda _: None, None, x)
        return x

    def program(x, w):
        return _norm_of(jax.jit(recorded)(x), w)

    x, w = np.ones((256, 256), jnp.bfloat16), np.ones((256,), jnp.bfloat16)
    written = jax.make_jaxpr(program)(x, w).effects
    assert written
    assert jax.make_jaxpr(functools.partial(_norms_rewritten, program))(x, w).effects == written


@pytest.mark.parametrize("program", list(NEAR_NORMS))
def test_a_plain_program_near_a_norm_computes_as_written(program):
    """A program whose arithmetic resembles an RMSNorm or SwiGLU without being one (the statistics or the
    weight laid along another axis, a nested program only named silu, a mean scaled by zero, the region's own
    sum read as its weight, a weight rounded by a cast the norm would drop, its statistics left in bfloat16,
    the weight applied to the row before its statistic, a SwiGLU rounding its sigmoid and each product to
    bfloat16 where Dew's rounds once after a float32 activation), compiled through `dew.jit` on an H100,
    computes what it does as written: its output and gradients on par with the program's (check 1),
    whatever of it the compile point writes as Dew's norm (an unweighted RMSNorm inside it, its product with
    another axis's weight left as written)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients


    fn = NEAR_NORMS[program]
    draw = normal_drawn(47, (256, 256), (256,), (256, 256), dtype=jnp.bfloat16)
    assert_as_exact(jax.jit(with_gradients(functools.partial(_norms_rewritten, fn))),
                    jax.jit(with_gradients(fn)), draw,
                    f"dew.jit {program}", seeds=8,
                    truth_fn=with_gradients(lambda *x: fn(*(a.astype(jnp.float64) for a in x))))


@pytest.mark.parametrize("enclosing", ["jit", "grad"])
def test_a_norm_whose_epsilon_an_enclosing_trace_closes_over_compiles_through_dew_jit(enclosing):
    """A plain RMSNorm in `dew.jit` whose epsilon an enclosing `jax.jit` or `jax.grad` traces (a value with no
    number until it runs, which no fused call takes as its own) compiles on an H100, and on this host
    computes what the program as written computes, bitwise."""
    import dew

    def norm(x, w, epsilon):
        widened = x.astype(jnp.float32)
        inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + epsilon)
        return (widened * inverse * w).astype(x.dtype)

    x = np.random.default_rng(5).standard_normal(ROWS).astype(jnp.bfloat16)
    w = np.random.default_rng(6).standard_normal(WEIGHT).astype(jnp.bfloat16)

    def enclosed(compiled):
        """The enclosing program, its norm compiled by `compiled` over the epsilon it traces."""
        def normed(epsilon):
            return compiled(lambda x, w: norm(x, w, epsilon))(x, w)

        if enclosing == "jit":
            return jax.jit(normed)
        return jax.jit(jax.grad(lambda epsilon: normed(epsilon).astype(jnp.float32).sum()))

    through, written = enclosed(dew.jit), enclosed(jax.jit)
    epsilon = np.float32(1e-6)
    with generation.assumed("sm90"):
        through.trace(epsilon)
    np.testing.assert_array_equal(np.asarray(ran(through, "the norm through dew.jit", epsilon)),
                                  np.asarray(ran(written, "the norm as written", epsilon)))


def _held_logistic(x):
    """Dew's one logistic's value (`precision.logistic_value`), rounded once to `x`'s dtype."""
    from dew.nn.precision import logistic_value

    given = jnp.asarray(x)
    return logistic_value(given).astype(given.dtype)


def _no_tangent(primals, tangents):
    """A tangent of the logistic's own: none."""
    value = _held_logistic(primals[0])
    return value, jnp.zeros_like(value)


def held_logistics_swiglu(a, b):
    """A SwiGLU whose sigmoid states Dew's one logistic's operation (`Protected.computes`) but is declared
    with a rule of its own, no tangent: one a fused call written in its place would not keep (train 1026's
    F6 review: the primal alone read as Dew's logistic would have fused it)."""
    from dew.nn.derivatives import Protected, declared_tangent
    from dew.nn.precision import logistic_value

    held = declared_tangent(Protected(computes=logistic_value), _held_logistic, _no_tangent)
    work = a.astype(jnp.float32)
    return (work * held(work)).astype(a.dtype) * b


def batched_by_its_rule(a, b):
    """A SwiGLU whose SiLU carries a custom batching rule of its own (`jax.custom_batching.custom_vmap`),
    one a fused call written in its place would not keep."""
    from jax.custom_batching import custom_vmap

    silu = custom_vmap(jax.nn.silu)
    silu.def_vmap(lambda size, batched, x: (jax.nn.silu(x) + 1, batched[0]))
    return silu(a.astype(jnp.float32)).astype(a.dtype) * b


def swiglu_traced(program):
    """`program` over two bf16 operands traced through `dew.jit` on an assumed H100: its trace, and whether
    a SwiGLU in it runs Dew's fused kernel (`generation.routes`)."""
    import dew

    shapes = [jax.ShapeDtypeStruct(ROWS, jnp.bfloat16)] * 2
    with generation.assumed("sm90"), generation.routes() as found:
        made = dew.jit(program).trace(*shapes)
    return made, any(route.choice == "swiglu" and route.kernel != "xla" for route in found)


@pytest.mark.parametrize("barred", [checkpointed_swiglu, batched_by_its_rule, held_logistics_swiglu],
                         ids=["a checkpointed sigmoid", "an activation's own batching rule",
                              "a logistic declared with a rule of its own"])
def test_a_gated_product_across_a_barrier_is_never_fused(barred):
    """A SwiGLU across a barrier runs as written through `dew.jit` on an H100, where the same product
    without it runs Dew's fused kernel: a checkpoint (`checkpointed_swiglu`), whose boundary decides what
    its derivative saves (§6.2), its program keeping the checkpoint; an activation's own batching rule
    (`batched_by_its_rule`), and a logistic stating Dew's operation declared with a derivative of its own
    (`held_logistics_swiglu`), which a fused call written in its place would not keep."""
    from jax.extend.core import primitives

    from dew.nn.programs import equations

    _, fused = swiglu_traced(gated_product_of)
    made, crossed = swiglu_traced(barred)
    assert fused and not crossed
    if barred is checkpointed_swiglu:
        assert any(equation.primitive is primitives.remat_p for equation in equations(made.jaxpr.jaxpr))


def test_swiglu_recognition_stays_local_on_deep_residual_chains_and_barriers():
    """On a deep multi-layer residual chain with checkpoint barriers and non-SwiGLU `logistic` anchors,
    `recognize._swiglus` / `_extras` / `_walked_forward` stop at barriers and stay within `REACH` / `SPAN`
    without traversing downstream layers while still recognizing a genuine SwiGLU (row 12465)."""
    from dew.algebra import fusion, recognize

    @jax.checkpoint
    def block(x, scale):
        return x + jax.nn.sigmoid(x) * scale

    def deep_program(x, up, scale):
        h = x
        for _ in range(16):
            h = block(h, scale)
            # Residual non-SwiGLU logistic followed by a deep elementwise tail.
            g = jax.nn.sigmoid(h)
            h = h + g * (h + 1.0)
        return gated_product_of(h, up)

    shapes = [jax.ShapeDtypeStruct(ROWS, jnp.bfloat16)] * 3
    closed = jax.make_jaxpr(deep_program)(*shapes)
    index = fusion.Index(closed.jaxpr)
    barriers = recognize._barriers(index)
    assert barriers
    # Forward walks from inputs before a barrier never cross past it.
    first_var = closed.jaxpr.invars[0]
    reached = recognize._forward(index, first_var, 2 * recognize.REACH)
    assert closed.jaxpr.outvars[0] not in reached
    # The genuine SwiGLU at the end is still recognized cleanly on an assumed H100.
    with generation.assumed("sm90"):
        found = recognize.recognized(closed.jaxpr, index, recognize.NORMS)
    assert len(found) == 1
    assert found[0].outputs == tuple(closed.jaxpr.outvars)


def test_outer_containers_never_anchor_recognition_and_swiglu_stays_inside_elementwise_ops():
    """Outer `scan`, `pjit[layer]`, and `remat` equations binding `logistic`/`rsqrt`/`exp` inside nested
    bodies emit no outer candidates in `_swiglus`/`_norms`/`_attentions`, and `_swiglus` walks never cross
    `dot_general`, while inner recognition still finds both SwiGLU and RMSNorm (row 12465)."""
    from dew.algebra import fusion, recognize
    from dew.nn.programs import bodies

    @jax.jit
    def layer(x, w_norm, w_gate, w_up, w_down):
        normed = widened_rms(x, w_norm)
        gate = normed @ w_gate
        up = normed @ w_up
        return x + gated_product_of(gate, up) @ w_down

    def stacked(x, w_norm, w_gate, w_up, w_down):
        def step(carry, _):
            return layer(carry, w_norm, w_gate, w_up, w_down), None
        out, _ = jax.lax.scan(step, x, None, length=2)
        return out

    shapes = (
        jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((256,), jnp.bfloat16),
        jax.ShapeDtypeStruct((256, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((256, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((512, 256), jnp.bfloat16),
    )
    closed = jax.make_jaxpr(stacked)(*shapes)
    outer_index = fusion.Index(closed.jaxpr)
    for place, eq in enumerate(closed.jaxpr.eqns):
        assert not recognize._holds(eq, jax.lax.logistic_p)
        assert not recognize._holds(eq, jax.lax.rsqrt_p)
        assert list(recognize._swiglus(outer_index, place)) == []
        assert list(recognize._norms(outer_index, place)) == []
        assert list(recognize._attentions(outer_index, place)) == []

    # Even directly around `pjit[layer]`, outer `_swiglus`/`_norms` emit 0 candidates (container holds dot).
    closed_call = jax.make_jaxpr(layer)(*shapes)
    call_index = fusion.Index(closed_call.jaxpr)
    for place in range(len(closed_call.jaxpr.eqns)):
        assert list(recognize._swiglus(call_index, place)) == []
        assert list(recognize._norms(call_index, place)) == []

    # Inside `layer`'s body: elementwise walks never cross `dot_general`, and both norm & SwiGLU match.
    (inner_jaxpr,) = bodies(closed_call.jaxpr.eqns[0])
    inner_index = fusion.Index(inner_jaxpr)
    (anchor_place,) = [p for p, eq in enumerate(inner_jaxpr.eqns) if recognize._holds(eq, jax.lax.logistic_p)]
    anchor_var = inner_jaxpr.eqns[anchor_place].outvars[0]
    back = recognize._walked(inner_index, "entrywise_backward", anchor_var, 3)
    fwd = recognize._walked(inner_index, "entrywise_forward", anchor_var, 4)
    assert inner_jaxpr.invars[0] not in back
    assert inner_jaxpr.outvars[0] not in fwd
    with generation.assumed("sm90"):
        found = recognize.recognized(inner_jaxpr, inner_index, recognize.NORMS)
    assert len(found) == 2


def test_swiglu_recognition_keeps_unit_axis_reshapes_around_activation():
    """A bf16 SwiGLU whose widened gate is reshaped to `[..., H, 1]` before `jax.nn.silu` and reshaped back
    before rounding and multiplying by `up` is still recognized (`normal_form._erased` removes the unit axes
    and `_entrywise_eqn` admits unit-axis-only layouts)."""
    from dew.algebra import fusion, recognize

    def reshaped_swiglu(gate, up):
        widened = gate.astype(jnp.float32).reshape(*gate.shape, 1)
        activated = jax.nn.silu(widened).reshape(gate.shape)
        return activated.astype(gate.dtype) * up

    shapes = [jax.ShapeDtypeStruct(ROWS, jnp.bfloat16)] * 2
    closed = jax.make_jaxpr(reshaped_swiglu)(*shapes)
    index = fusion.Index(closed.jaxpr)
    with generation.assumed("sm90"):
        found = recognize.recognized(closed.jaxpr, index, recognize.NORMS)
    assert len(found) == 1
    assert found[0].outputs == tuple(closed.jaxpr.outvars)


def _l2norm(x):
    """Plain-JAX L2 normalization along the last axis in at least fp32 (no Dew imports)."""
    wide = x.astype(jnp.float64 if x.dtype == jnp.float64 else jnp.float32)
    denom = jnp.sum(jnp.square(wide), axis=-1, keepdims=True) + jnp.asarray(1e-6, wide.dtype)
    return (wide * jax.lax.rsqrt(denom)).astype(x.dtype)


def _gdn_step_math(state, qf, kf, vf, gt, bt, *, subtract: bool = True, flipped: bool = False):
    """Core token arithmetic of a plain-JAX gated delta rule step (no Dew imports or `swapaxes`)."""
    decay = jnp.exp(gt)[..., None]
    write = bt[..., None]
    scaled_k = (kf * decay)[..., :, None]
    retrieved = jnp.sum(state * scaled_k if flipped else scaled_k * state, axis=-2)
    corrected = vf - retrieved if subtract else vf + retrieved
    outer = (kf * write)[..., :, None] * corrected[..., None, :]
    out = (jnp.sum((qf * decay)[..., :, None] * state, axis=-2)
           + jnp.sum(qf[..., :, None] * outer, axis=-2))
    return state * decay[..., :, None] + outer, out


def _bounded_box(x, limit: float = 1.0):
    """Clamp `x` to `[-limit, limit]` in its own dtype so `terms.proven` proves a static finite bound."""
    return jax.lax.clamp(jnp.asarray(-limit, x.dtype), x, jnp.asarray(limit, x.dtype))


def _gdn_scan_natural(state, q, k, v, g, beta):
    """Plain-JAX `jax.lax.scan` gated delta rule written with natural `[K, V]` outer product, flipped
    `state * (k * decay)` order, L2-normalized Q/K, sigmoid `beta`, bf16-lossless bounded state, bounded V,
    and `minimum(g, 0.0)`."""
    s_in, v_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32), _bounded_box(v)
    q_n, k_n, neg_g, b_n = _l2norm(q), _l2norm(k), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)

    def step(s, xs):
        qt, kt, vt, gt, bt = xs
        next_s, out = _gdn_step_math(s, qt.astype(jnp.float32), kt.astype(jnp.float32),
                                     vt.astype(jnp.float32), gt, bt.astype(jnp.float32), flipped=True)
        return next_s, out.astype(vt.dtype)

    return jax.lax.scan(step, s_in, (q_n, k_n, v_in, neg_g, b_n))


def _gdn_scan_grouped_scaled(state, g, v, q, beta, k):
    """Plain-JAX `jax.lax.scan` gated delta rule with permuted operand order, grouped key/value heads
    (`Hv = 2 * Hk`), explicit query scaling `Dk ** -0.5`, L2-normalized Q/K, sigmoid `beta`, bf16-lossless
    bounded state, bounded V, and `-jax.nn.softplus(g)` nonpositive decay."""
    scale = 64.0 ** -0.5
    s_in, v_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32), _bounded_box(v)
    q_n, k_n, neg_g, b_n = _l2norm(q), _l2norm(k), -jax.nn.softplus(g), jax.nn.sigmoid(beta)

    def step(s, xs):
        gt, vt, qt, bt, kt = xs
        group = s.shape[1] // qt.shape[1]
        qf = jnp.repeat(qt.astype(jnp.float32) * scale, group, axis=1)
        kf = jnp.repeat(kt.astype(jnp.float32), group, axis=1)
        next_s, out = _gdn_step_math(s, qf, kf, vt.astype(jnp.float32), gt, bt.astype(jnp.float32))
        return next_s, out.astype(vt.dtype)

    return jax.lax.scan(step, s_in, (neg_g, v_in, q_n, b_n, k_n))


def _gdn_scan_f64_truth(state, q, k, v, g, beta, *, scale: float | None = None, softplus: bool = False):
    """Float64 token-by-token truth for the gated delta rule scan."""
    s_in, v_in = _bounded_box(state, 0.125), _bounded_box(v)
    q_n, k_n = _l2norm(q), _l2norm(k)
    neg_g = -jax.nn.softplus(g) if softplus else jnp.minimum(g, 0.0)
    b_n = jax.nn.sigmoid(beta)

    def step(s, xs):
        qt, kt, vt, gt, bt = xs
        group = s.shape[1] // qt.shape[1]
        qf = jnp.repeat(qt if scale is None else qt * scale, group, axis=1)
        kf = jnp.repeat(kt, group, axis=1)
        return _gdn_step_math(s, qf, kf, vt, gt, bt)

    return jax.lax.scan(step, s_in, (q_n, k_n, v_in, neg_g, b_n))


def _recurrence_rewritten(fn, *operands):
    """`fn` with its recognized recurrence scans lowered as on an H100 (`recognize.RECURRENCES`) and run on
    this host's portable math (same pattern as `_norms_rewritten`)."""
    from dew.algebra import recognize

    shapes = [jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x)) for x in operands]
    built, found = _norms_written(fn, *shapes, families=recognize.RECURRENCES)
    assert found == 1
    return tuple(built(*operands))


@pytest.mark.parametrize("case", ["natural", "grouped_scaled"])
def test_a_plain_gated_delta_net_scan_runs_dews_recurrence_kernel_through_dew_jit(case):
    """A plain-JAX `jax.lax.scan` gated delta rule compiled through `dew.jit` on an assumed H100 lowers to
    Dew's chunked recurrence kernel (`RECURRENCES`, row 3450) across operand orders, grouped heads, query
    scales, and natural non-`swapaxes` outer products; under the interpreted template it matches Dew's own
    chunked call bit-for-bit, and its rewritten portable math holds check 1 against the plain scan."""
    from reference_error import assert_as_exact

    import dew
    from dew.algebra.recognize import KEPT
    from dew.nn.kernels import recurrence, recurrence_chunk

    seq, batch, hk, hv, dk, dv = 64, 1, (2 if case == "natural" else 1), 2, 64, 64
    scale = None if case == "natural" else (64.0 ** -0.5)
    hooks = recurrence.Hooks(decay="scalar", delta=True, scale=scale)
    state_s = jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32)
    q_s = jax.ShapeDtypeStruct((seq, batch, hk, dk), jnp.bfloat16)
    k_s = jax.ShapeDtypeStruct((seq, batch, hk, dk), jnp.bfloat16)
    v_s = jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16)
    g_s = jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32)
    b_s = jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16)
    if case == "natural":
        fn, shapes = _gdn_scan_natural, (state_s, q_s, k_s, v_s, g_s, b_s)
    else:
        fn, shapes = _gdn_scan_grouped_scaled, (state_s, g_s, v_s, q_s, b_s, k_s)

    def own(state, q, k, v, g, beta):
        s_in, v_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32), _bounded_box(v)
        q_n, k_n, b_n = _l2norm(q), _l2norm(k), jax.nn.sigmoid(beta)
        neg_g = jnp.minimum(g, 0.0) if case == "natural" else -jax.nn.softplus(g)
        out, final = recurrence_chunk.chunk(
            hooks, recurrence_chunk.Chunk(
                state=s_in, query=jnp.swapaxes(q_n, 0, 1), key=jnp.swapaxes(k_n, 0, 1),
                value=jnp.swapaxes(v_in, 0, 1), log_decay=jnp.swapaxes(neg_g, 0, 1),
                beta=jnp.swapaxes(b_n, 0, 1)), dtype=jnp.bfloat16)
        return final, jnp.swapaxes(out, 0, 1)
    with generation.assumed("sm90"):
        assert not _routes("gated_delta_rule", lambda: jax.jit(fn).trace(*shapes))
        expected = {kernel for kernel in _routes(
            "gated_delta_rule", lambda: jax.jit(own).trace(state_s, q_s, k_s, v_s, g_s, b_s))
            if kernel not in KEPT}
        assert expected and set(_routes("gated_delta_rule", lambda: dew.jit(fn).trace(*shapes))) == expected

    def draw(seed: int):
        rng = np.random.default_rng(101 + seed)
        state = jnp.asarray(rng.standard_normal((batch, hv, dk, dv)) * 0.02, jnp.bfloat16).astype(jnp.float32)
        q = jnp.asarray(rng.standard_normal((seq, batch, hk, dk)), jnp.bfloat16)
        k = jnp.asarray(rng.standard_normal((seq, batch, hk, dk)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((seq, batch, hv, dv)) * 0.2, jnp.bfloat16)
        g = jnp.asarray((rng.uniform(-1.0, -0.05, (seq, batch, hv)) if case == "natural"
                         else rng.uniform(-1.0, 1.0, (seq, batch, hv))).astype(np.float32))
        beta = jnp.asarray(rng.uniform(-2.5, 0.0, (seq, batch, hv)), jnp.bfloat16)
        return (state, q, k, v, g, beta) if case == "natural" else (state, g, v, q, beta, k)

    sample = draw(0)
    own_sample = sample if case == "natural" else (sample[0], sample[3], sample[5], sample[2], sample[1],
                                                   sample[4])
    with generation.chosen(gated_delta_rule="mosaic_gpu"):
        got_final, got_out = dew.jit(fn)(*sample)
        ref_final, ref_out = jax.jit(own)(*own_sample)
    np.testing.assert_array_equal(np.asarray(got_final), np.asarray(ref_final))
    np.testing.assert_array_equal(np.asarray(got_out), np.asarray(ref_out))

    def truth(*args):
        if case == "natural":
            return _gdn_scan_f64_truth(*args, scale=scale, softplus=False)
        state, g, v, q, beta, k = args
        return _gdn_scan_f64_truth(state, q, k, v, g, beta, scale=scale, softplus=True)

    assert_as_exact(jax.jit(functools.partial(_recurrence_rewritten, fn)), jax.jit(fn), draw,
                    f"dew.jit gated delta scan ({case})", seeds=8, truth_fn=truth)


def test_recognized_recurrence_preserves_neginf_hard_reset():
    """A recognized GDN scan whose nonpositive decay hits `-inf` mid-sequence stays finite, resets the state
    cleanly on both the portable math and interpreted `mosaic_gpu` routes (scaling the entering state and
    pre-reset values changes neither the final state nor any post-reset output), and passes check 1 against
    the plain scan."""
    from reference_error import assert_as_exact

    import dew

    seq, batch, hv, dk, dv = 64, 1, 2, 64, 64

    def draw(seed: int):
        rng = np.random.default_rng(707 + seed)
        state = jnp.asarray(
            rng.standard_normal((batch, hv, dk, dv)) * 0.0002, jnp.bfloat16).astype(jnp.float32)
        q = jnp.asarray(rng.standard_normal((seq, batch, hv, dk)), jnp.bfloat16)
        k = jnp.asarray(rng.standard_normal((seq, batch, hv, dk)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((seq, batch, hv, dv)) * 0.002, jnp.bfloat16)
        g = jnp.asarray(rng.uniform(-1.0, -0.05, (seq, batch, hv)).astype(np.float32))
        g = g.at[16, :, :].set(-jnp.inf)
        beta = jnp.asarray(rng.uniform(-2.0, 0.0, (seq, batch, hv)), jnp.bfloat16)
        return state, q, k, v, g, beta

    state, q, k, v, g, beta = draw(0)
    big_state = state * 100.0
    big_v = v.at[:16].set((v[:16].astype(jnp.float32) * 100.0).astype(jnp.bfloat16))
    xla_fn = jax.jit(functools.partial(_recurrence_rewritten, _gdn_scan_natural))
    xla_final, xla_out = xla_fn(state, q, k, v, g, beta)
    xla_big_final, xla_big_out = xla_fn(big_state, q, k, big_v, g, beta)
    with generation.chosen(gated_delta_rule="mosaic_gpu"):
        mos_fn = dew.jit(_gdn_scan_natural)
        mos_final, mos_out = mos_fn(state, q, k, v, g, beta)
        mos_big_final, mos_big_out = mos_fn(big_state, q, k, big_v, g, beta)
    for final_val, out_val in ((xla_final, xla_out), (mos_final, mos_out)):
        assert np.isfinite(np.asarray(final_val, np.float32)).all()
        assert np.isfinite(np.asarray(out_val, np.float32)).all()
    np.testing.assert_array_equal(np.asarray(xla_final), np.asarray(xla_big_final))
    np.testing.assert_array_equal(np.asarray(xla_out[16:]), np.asarray(xla_big_out[16:]))
    np.testing.assert_array_equal(np.asarray(mos_final), np.asarray(mos_big_final))
    np.testing.assert_array_equal(np.asarray(mos_out[16:]), np.asarray(mos_big_out[16:]))
    assert_as_exact(xla_fn, jax.jit(_gdn_scan_natural), draw, "dew.jit gated delta scan (-inf reset)",
                    seeds=8, truth_fn=lambda *a: _gdn_scan_f64_truth(*a, scale=None, softplus=False))


def _gdn_make_step(batch: int, hv: int, *, subtract: bool = True, narrow_state: bool = False,
                   perturb_iota: bool = False):
    """Helper step body for barred/non-matching recurrence scan tests."""
    def step(s, xs):
        qt, kt, vt, gt, bt = xs
        if perturb_iota:
            iota = jax.lax.broadcasted_iota(jnp.float32, (batch, hv), 0)
            # Broadcast to a distinct rank so JAX's same-shape forwarding rule does not drop it.
            gt = gt - (iota - jax.lax.broadcast_in_dim(iota, (1, batch, hv), (2, 1))[0])
        next_s, out = _gdn_step_math(
            s.astype(jnp.float32) if narrow_state else s,
            qt.astype(jnp.float32), kt.astype(jnp.float32), vt.astype(jnp.float32),
            gt, bt.astype(jnp.float32), subtract=subtract)
        return (next_s.astype(s.dtype) if narrow_state else next_s), out.astype(vt.dtype)
    return step


def _gdn_make_scan(step_fn, *, clamp_g: bool = True, reverse: bool = False,
                   bound_inputs: bool = True, bound_state_v: bool = True, mixed_concat_g: bool = False):
    """Helper scan wrapper configuring decay/range/reverse variations for recurrence tests."""
    def run(state, q, k, v, g, beta):
        s_in, v_in = (
            (_bounded_box(state, 0.125).astype(q.dtype).astype(state.dtype), _bounded_box(v))
            if bound_state_v else (state, v))
        q_in, k_in, b_in = (
            (_l2norm(q), _l2norm(k), jax.nn.sigmoid(beta)) if bound_inputs else (q, k, beta))
        if mixed_concat_g:
            g_in = jnp.concatenate([jnp.minimum(g[:32], 0.0), g[32:]], axis=0)
        else:
            g_in = jnp.minimum(g, 0.0) if clamp_g else g
        return jax.lax.scan(step_fn, s_in, (q_in, k_in, v_in, g_in, b_in), reverse=reverse)

    return run


def test_non_matching_or_barred_recurrence_scans_stay_as_written():
    """Reverse scans, narrow bf16 state carries, checkpointed step bodies, mismatched step arithmetic,
    unconstrained or partially concatenated decays, unbounded Q/K/beta ranges, fp32 write gates beside bf16
    operands, and transposed-iota perturbations never lower through `RECURRENCES`; deep diamond DAGs before
    transpose canonicalize in linear time."""
    import dew
    from dew.algebra import normal_form

    seq, batch, hv, dk, dv = 64, 2, 2, 64, 64
    state_f32 = jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32)
    state_bf16 = jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.bfloat16)
    q_s = jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16)
    k_s = jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16)
    v_s = jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16)
    g_s = jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32)
    b_bf16 = jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16)
    b_f32 = jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32)
    std_shapes = (state_f32, q_s, k_s, v_s, g_s, b_bf16)
    cases = [
        (_gdn_make_scan(_gdn_make_step(batch, hv), reverse=True), std_shapes),
        (_gdn_make_scan(_gdn_make_step(batch, hv, narrow_state=True)),
         (state_bf16, q_s, k_s, v_s, g_s, b_bf16)),
        (_gdn_make_scan(jax.checkpoint(_gdn_make_step(batch, hv))), std_shapes),
        (_gdn_make_scan(_gdn_make_step(batch, hv, subtract=False)), std_shapes),
        (_gdn_make_scan(_gdn_make_step(batch, hv), clamp_g=False), std_shapes),
        (_gdn_make_scan(_gdn_make_step(batch, hv), mixed_concat_g=True), std_shapes),
        (_gdn_make_scan(_gdn_make_step(batch, hv), bound_inputs=False), std_shapes),
        (_gdn_scan_natural, (state_f32, q_s, k_s, v_s, g_s, b_f32)),
        (_gdn_make_scan(_gdn_make_step(batch, hv, perturb_iota=True)), std_shapes),
    ]
    with generation.assumed("sm90"):
        for fn, shapes in cases:
            assert not _routes("gated_delta_rule", lambda f=fn, s=shapes: dew.jit(f).trace(*s))

    # Deep shared diamond DAG before transpose stays linear in graph size under `normal_form.canonical`.
    def diamond_transpose(a, b):
        outer = a[:, None] * b[None, :]
        for _ in range(30):
            outer = jnp.sin(outer) + jnp.cos(outer)
        return jnp.swapaxes(outer, 0, 1)

    closed = jax.make_jaxpr(diamond_transpose)(jax.ShapeDtypeStruct((4,), jnp.float32),
                                               jax.ShapeDtypeStruct((4,), jnp.float32))
    canon = normal_form.canonical(closed)
    assert not any(eq_.primitive is jax.lax.transpose_p for eq_ in canon.jaxpr.eqns)


def test_recurrence_recognition_guards_range_tracers_tokens_and_training_vjp():
    """`RECURRENCES` preserves unbounded bf16 `Q=K=2^80, beta=2^-80` token scans (avoiding FLA `Q @ K^T`
    overflow), handles closed-over outer tracers without `TracerArrayConversionError`, skips effect-free
    token-carry scans, and leaves forward-only forced Mosaic training routes intact so `jax.vjp` succeeds."""
    import dew
    from dew.nn.precision import Numerics

    seq, batch, hv, dk, dv = 64, 1, 2, 64, 64
    std_shapes = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16),
    )

    # 1. Unbounded bf16 one-hot Q=K=2^80, beta=2^-80 stays as token scan and produces exact finite 2^80.
    unbounded_scan = _gdn_make_scan(_gdn_make_step(batch, hv), bound_inputs=False)
    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
        assert not _routes("gated_delta_rule", lambda: dew.jit(unbounded_scan).trace(*std_shapes))
    q_big = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16).at[0, :, :, 0].set(jnp.bfloat16(2.0 ** 80))
    k_big = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16).at[0, :, :, 0].set(jnp.bfloat16(2.0 ** 80))
    v_one = jnp.ones((seq, batch, hv, dv), jnp.bfloat16)
    g_zero = jnp.zeros((seq, batch, hv), jnp.float32)
    b_tiny = jnp.full((seq, batch, hv), jnp.bfloat16(2.0 ** -80))
    s_zero = jnp.zeros((batch, hv, dk, dv), jnp.float32)
    _, out_big = dew.jit(unbounded_scan)(s_zero, q_big, k_big, v_one, g_zero, b_tiny)
    assert np.isfinite(np.asarray(out_big, np.float32)).all()
    assert float(out_big[0, 0, 0, 0]) == float(jnp.bfloat16(2.0 ** 80))

    # 2. Nested `dew.jit` closing over an outer tracer never raises `TracerArrayConversionError`.
    def outer_with_tracer(extra, state, q, k, v, g, beta):
        @dew.jit
        def inner(s, q_, k_, v_, g_, b_):
            fin, out = _gdn_scan_natural(s, q_, k_, v_, g_, b_)
            return fin, out + extra

        return inner(state, q, k, v, g, beta)

    with generation.assumed("sm90"):
        assert jax.jit(outer_with_tracer).trace(
            jax.ShapeDtypeStruct((), jnp.bfloat16), *std_shapes) is not None

    # 3. Effect-free token-carry scans with 3 scanned arrays compile cleanly without touching `AbstractToken`.
    def token_scan(a, b, c):
        _, ys = jax.lax.scan(lambda tok, xs: (jax.lax.after_all(tok), xs[0] + xs[1] + xs[2]),
                             jax.lax.create_token(), (a, b, c))
        return ys

    xs_tok = tuple(jnp.ones((4, 2), jnp.float32) for _ in range(3))
    with generation.assumed("sm90"):
        assert dew.jit(token_scan).trace(*xs_tok) is not None
    tok_ys = dew.jit(token_scan)(*xs_tok)
    np.testing.assert_array_equal(np.asarray(tok_ys), np.full((4, 2), 3.0, np.float32))

    # 4a. Under `Numerics(forms="accurate")`, an otherwise-eligible bf16 GDN scan stays as a plain scan.
    with generation.assumed("sm90"), generation.run_precision(Numerics(forms="accurate")):
        assert not _routes("gated_delta_rule", lambda: dew.jit(_gdn_scan_natural).trace(*std_shapes))

    # 4b. Under default numerics, bf16 per-key GLA (delta=False, write_gate=True) takes `mosaic_gpu` in
    # forward (`inter_dtype=fp32`) but raises `KernelRefused` in training VJP (`fast_backward_form` rejects
    # per-key without delta): verify forward eligibility + direct VJP refusal, then confirm `dew.jit` leaves
    # the scan intact so `jax.vjp` traces cleanly without hitting the accurate-mode skip.
    from dew.nn.kernels import recurrence, recurrence_chunk
    from dew.nn.kernels.selection import KernelRefused

    def bf16_per_key_gla_scan(state, q, k, v, g, beta):
        s_in, v_in = _bounded_box(state, 0.125), _bounded_box(v)
        q_n, k_n, neg_g, b_n = _l2norm(q), _l2norm(k), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)

        def step(s, xs):
            qt, kt, vt, gt, bt = xs
            qf, kf, vf, bf = (x.astype(jnp.float32) for x in (qt, kt, vt, bt))
            decay = jnp.exp(gt)
            outer = (kf * bf[..., None])[..., :, None] * vf[..., None, :]
            out = (jnp.sum((qf * decay)[..., :, None] * s, axis=-2)
                   + jnp.sum(qf[..., :, None] * outer, axis=-2))
            return s * decay[..., :, None] + outer, out.astype(vt.dtype)

        return jax.lax.scan(step, s_in, (q_n, k_n, v_in, neg_g, b_n))

    gla_hooks = recurrence.Hooks(decay="vector", delta=False, write_gate=True)

    def direct_gla(state, q, k, v, g, beta):
        out, final = recurrence_chunk.chunk(
            gla_hooks, recurrence_chunk.Chunk(
                state=state, query=jnp.swapaxes(q, 0, 1), key=jnp.swapaxes(k, 0, 1),
                value=jnp.swapaxes(v, 0, 1), log_decay=jnp.swapaxes(g, 0, 1),
                beta=jnp.swapaxes(beta, 0, 1)), dtype=jnp.bfloat16)
        return final, jnp.swapaxes(out, 0, 1)

    gla_shapes = (*std_shapes[:4], jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.float32), std_shapes[5])
    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="mosaic_gpu"):
        assert set(_routes("gated_delta_rule", lambda: jax.eval_shape(direct_gla, *gla_shapes))) == {
            "mosaic_gpu"}
        with pytest.raises(KernelRefused):
            jax.eval_shape(
                lambda *a: (lambda p, pull: (p, pull(p)))(*jax.vjp(direct_gla, *a)), *gla_shapes)
        assert not _routes("gated_delta_rule", lambda: dew.jit(bf16_per_key_gla_scan).trace(*gla_shapes))
        vjp_out, vjp_grads = jax.eval_shape(
            lambda *a: (lambda p, pull: (p, pull(p)))(*jax.vjp(dew.jit(bf16_per_key_gla_scan), *a)),
            *gla_shapes)
        assert vjp_out[1].shape == (seq, batch, hv, dv) and len(vjp_grads) == len(gla_shapes)


def test_recurrence_recognition_candidate_cache_wy_state_bounds_literal_guard_and_f64_scale():
    """`RECURRENCES` keys region cache hits on eligible candidate identities (never reusing a stale position
    when outer proofs drop reversed K/V candidates), refuses non-contractive `256*tanh(k)` WY solves and
    unconstrained fp32 entering state, handles direct-literal NaN guards without hashing `Literal`, and
    discards overflowed float64 reciprocal scales."""
    import dew

    seq, batch, hv, dk, dv = 64, 1, 2, 64, 64
    std_shapes = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16),
    )

    # 1. Equal K/V shapes with `(q, v, k)` order: warm cache with both V and K unit-normed (reversed
    # candidate offered first), then trace with only K unit-normed (reversed dropped) without IndexError.
    def step_qvk(s, xs):
        qt, vt, kt, gt, bt = xs
        next_s, out = _gdn_step_math(s, qt.astype(jnp.float32), kt.astype(jnp.float32),
                                     vt.astype(jnp.float32), gt, bt.astype(jnp.float32))
        return next_s, out.astype(vt.dtype)

    def scan_both_unit(state, q, v, k, g, beta):
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(step_qvk, s_in,
                            (_l2norm(q), _l2norm(v), _l2norm(k), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    def scan_only_k_unit(state, q, v, k, g, beta):
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(step_qvk, s_in,
                            (_l2norm(q), _bounded_box(v), _l2norm(k), jnp.minimum(g, 0.0),
                             jax.nn.sigmoid(beta)))
    with generation.assumed("sm90"):
        assert _routes("gated_delta_rule", lambda: dew.jit(scan_both_unit).trace(*std_shapes))
        assert _routes("gated_delta_rule", lambda: dew.jit(scan_only_k_unit).trace(*std_shapes))

    # 2. Non-contractive `256*tanh(k)` WY solve and unconstrained fp32 entering state stay as token scans.
    def scan_big_tanh_k(state, q, k, v, g, beta):
        k_big = jnp.asarray(256.0, k.dtype) * jnp.tanh(k)
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(_gdn_make_step(batch, hv), s_in,
                            (_l2norm(q), k_big, _bounded_box(v), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))
    unbounded_state_scan = _gdn_make_scan(_gdn_make_step(batch, hv), bound_state_v=False)
    qkv_shapes = (std_shapes[0], std_shapes[1], std_shapes[3], std_shapes[2], std_shapes[4], std_shapes[5])
    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
        assert not _routes("gated_delta_rule", lambda: dew.jit(scan_big_tanh_k).trace(*qkv_shapes))
        assert not _routes("gated_delta_rule", lambda: dew.jit(unbounded_state_scan).trace(*qkv_shapes))
    z_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
    z_q = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16)
    k_hot = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16).at[:, :, :, 0].set(jnp.bfloat16(100.0))
    z_v = jnp.zeros((seq, batch, hv, dv), jnp.bfloat16)
    z_g = jnp.zeros((seq, batch, hv), jnp.float32)
    b_hi = jnp.full((seq, batch, hv), jnp.bfloat16(100.0))
    _, wy_out = dew.jit(scan_big_tanh_k)(z_s, z_q, k_hot, z_v, z_g, b_hi)
    np.testing.assert_array_equal(np.asarray(wy_out, np.float32), np.zeros((seq, batch, hv, dv), np.float32))
    huge_s = jnp.full((batch, hv, dk, dv), np.float32(3.4e38), jnp.float32)
    fin_s, out_s = dew.jit(unbounded_state_scan)(
        huge_s, z_q, z_q, z_v, jnp.full_like(z_g, -1.0), z_g.astype(jnp.bfloat16))
    assert np.isfinite(np.asarray(fin_s)).all() and np.isfinite(np.asarray(out_s, np.float32)).all()

    # 3. Direct-literal NaN guard in decay producer never hashes JAX 0.12 `Literal`.
    def scan_literal_nan_guard(state, q, k, v, g, beta):
        diff = jax.lax.sub(g, np.float32(1.0))
        neg_g = jax.lax.select(diff != diff, jax.lax.add(g, np.float32(1.0)), jnp.minimum(g, 0.0))
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(_gdn_make_step(batch, hv), s_in,
                            (_l2norm(q), _l2norm(k), _bounded_box(v), neg_g, jax.nn.sigmoid(beta)))
    with generation.assumed("sm90"):
        assert _routes("gated_delta_rule", lambda: dew.jit(scan_literal_nan_guard).trace(*qkv_shapes))

    # 4. Float64 scan dividing Q by `5e-324` (overflowing `1.0 / factor` to `inf`) compiles cleanly.
    def scan_f64_subnormal_div(state, q, k, v, g, beta):
        def step(s, xs):
            qt, kt, vt, gt, bt = xs
            return _gdn_step_math(s, qt / np.float64(5e-324), kt, vt, gt, bt)

        return jax.lax.scan(
            step, _bounded_box(state, 0.125),
            (_l2norm(q), _l2norm(k), _bounded_box(v), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))
    with jax.enable_x64(new_val=True), generation.assumed("sm90"):
        f64_shapes = tuple(jax.ShapeDtypeStruct(s.shape, jnp.float64) for s in qkv_shapes)
        assert dew.jit(scan_f64_subnormal_div).trace(*f64_shapes) is not None


def test_recurrence_recognition_guards_fla_zero_gate_gram_overflow_score_underflow_and_expanded_keys():
    """`RECURRENCES` preserves token scans when ungated `K @ K^T` would overflow under zero Q/beta/state,
    when a singleton key is broadcast-expanded by its reciprocal norm (non-contractive WY), when FLA score
    narrowing would flush `Q @ K^T = 2^-136` to zero ahead of `V = 2^44`, and when large entering state
    would lift flushed scores via `corrected = V - K^T h_0`."""
    import dew

    seq, batch, hv, dk, dv = 64, 1, 2, 64, 64
    qkv_shapes = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16),
    )
    step = _gdn_make_step(batch, hv)

    # 1. Zero Q, zero beta, zero state, huge bounded K = 2^80 * tanh(k): ungated K @ K^T would overflow FLA.
    def scan_zero_gate_huge_k(state, q, k, v, g, beta):
        del state, q, beta
        zero_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
        zero_q = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16)
        zero_b = jnp.zeros((seq, batch, hv), jnp.bfloat16)
        k_huge = jnp.asarray(float(2 ** 80), jnp.bfloat16) * jnp.tanh(k)
        return jax.lax.scan(step, zero_s, (zero_q, k_huge, _bounded_box(v), jnp.minimum(g, 0.0), zero_b))

    # 2. Singleton [S, B, H, 1] key normalized then broadcast-expanded to dk=64 via its reciprocal norm.
    def scan_expanded_singleton_k(state, q, k, v, g, beta):
        k1 = k[..., :1]
        inv = jax.lax.rsqrt(jnp.sum(k1.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
        inv_exp = jnp.broadcast_to(inv.astype(k.dtype), k.shape)
        k_exp = k1 * inv_exp
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(step, s_in,
                            (_l2norm(q), k_exp, _bounded_box(v), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    # 3. Finite tiny Q=K=2^-68 * tanh(x) with V=2^44: token scan returns normal 2^-92, FLA flushes Q·K to 0.
    def scan_tiny_score_large_v(state, q, k, v, g, beta):
        del state
        zero_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
        tiny = jnp.asarray(2.0 ** -68, jnp.bfloat16)
        q_t = jnp.tanh(q) * tiny
        k_t = jnp.tanh(k) * tiny
        v_big = _bounded_box(v, float(2 ** 44))
        return jax.lax.scan(step, zero_s, (q_t, k_t, v_big, jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    # 4. Large entering state `h_0 = [0, 2^50]` with unit K and tiny Q: `-K^T h_0` lifts flushed score.
    def scan_large_state_tiny_q(state, q, k, v, g, beta):
        s_big = _bounded_box(state, float(2 ** 50)).astype(q.dtype).astype(jnp.float32)
        q_t = jnp.tanh(q) * jnp.asarray(2.0 ** -105, jnp.bfloat16)
        return jax.lax.scan(step, s_big, (q_t, _l2norm(k), jnp.zeros_like(v), jnp.minimum(g, 0.0),
                                          jax.nn.sigmoid(beta)))

    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
        for fn in (scan_zero_gate_huge_k, scan_expanded_singleton_k,
                   scan_tiny_score_large_v, scan_large_state_tiny_q):
            assert not _routes("gated_delta_rule", lambda fn=fn: dew.jit(fn).trace(*qkv_shapes))

    z_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
    z_g = jnp.zeros((seq, batch, hv), jnp.float32)
    b_hi = jnp.full((seq, batch, hv), jnp.bfloat16(100.0))
    hot = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16).at[:, :, :, 0].set(jnp.bfloat16(100.0))
    v_small = jnp.zeros((seq, batch, hv, dv), jnp.bfloat16)
    with generation.chosen(gated_delta_rule="fla"):
        fin1, out1 = dew.jit(scan_zero_gate_huge_k)(z_s, hot, hot, v_small, z_g, b_hi)
        np.testing.assert_array_equal(np.asarray(fin1), np.zeros((batch, hv, dk, dv), np.float32))
        np.testing.assert_array_equal(
            np.asarray(out1, np.float32), np.zeros((seq, batch, hv, dv), np.float32))
        v_44 = jnp.zeros((seq, batch, hv, dv), jnp.bfloat16).at[0, :, :, 0].set(jnp.bfloat16(2.0 ** 44))
        ref_fin3, ref_out3 = jax.jit(scan_tiny_score_large_v)(z_s, hot, hot, v_44, z_g, b_hi)
        got_fin3, got_out3 = dew.jit(scan_tiny_score_large_v)(z_s, hot, hot, v_44, z_g, b_hi)
        np.testing.assert_array_equal(np.asarray(got_fin3), np.asarray(ref_fin3))
        np.testing.assert_array_equal(np.asarray(got_out3, np.float32), np.asarray(ref_out3, np.float32))
        assert float(np.asarray(got_out3, np.float32)[0, 0, 0, 0]) == float(jnp.bfloat16(2.0 ** -92))


def _scan_bf16_state_init(step):
    """Return a bounded GDN scan whose fp32 state initializer is losslessly widened from bf16."""
    def scan_fn(state, q, k, v, g, beta):
        s_in = _bounded_box(state, 0.125).astype(q.dtype).astype(jnp.float32)
        return jax.lax.scan(step, s_in, (_bounded_box(q, 0.125), _bounded_box(k, 0.125), _bounded_box(v),
                                         jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))
    return scan_fn


def _assert_fla_cancel_preserved(scan_fn, state, q, k, v, g, beta, *, token_idx: int, expected: float):
    """Assert direct forced FLA flushes `token_idx` to 0 while `dew.jit(scan_fn)` preserves exact cancellation
    matching `jax.jit(scan_fn)` bit-for-bit."""
    import dew
    from dew.nn.kernels import recurrence, recurrence_chunk

    with generation.chosen(gated_delta_rule="fla"):
        fla_out, _ = recurrence_chunk.chunk(
            recurrence.Hooks(decay="scalar", delta=True),
            recurrence_chunk.Chunk(
                state=state, query=jnp.swapaxes(q, 0, 1), key=jnp.swapaxes(k, 0, 1),
                value=jnp.swapaxes(v, 0, 1), log_decay=jnp.swapaxes(g, 0, 1),
                beta=jnp.swapaxes(jax.nn.sigmoid(beta), 0, 1)), dtype=jnp.bfloat16)
        assert float(np.asarray(fla_out, np.float32)[0, token_idx, 0, 0]) == 0.0
        ref_fin, ref_out = jax.jit(scan_fn)(state, q, k, v, g, beta)
        got_fin, got_out = dew.jit(scan_fn)(state, q, k, v, g, beta)
        np.testing.assert_array_equal(np.asarray(got_fin), np.asarray(ref_fin))
        np.testing.assert_array_equal(np.asarray(got_out, np.float32), np.asarray(ref_out, np.float32))
        assert float(np.asarray(got_out, np.float32)[token_idx, 0, 0, 0]) == expected


def test_recurrence_recognition_preserves_fp32_entering_state_cancellation():
    """`RECURRENCES` preserves token scans when the selected route (`fla`/`mosaic_gpu`) narrows a live fp32
    entering state to bf16 before the readout dot, isolating region cache keys by entering-state source dtype
    and preserving exact `-2^-19` cancellation where forced FLA flushes to zero."""
    import dew

    seq, batch, hv, dk, dv = 64, 1, 2, 64, 64
    qkv_shapes = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq, batch, hv), jnp.bfloat16),
    )
    step = _gdn_make_step(batch, hv)
    scan_bf16_state_warm = _scan_bf16_state_init(step)

    def scan_fp32_state_cancel(state, q, k, v, g, beta):
        return jax.lax.scan(step, _bounded_box(state, 0.125),
                            (_bounded_box(q, 0.125), _bounded_box(k, 0.125), _bounded_box(v),
                             jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
        assert _routes("gated_delta_rule", lambda: dew.jit(scan_bf16_state_warm).trace(*qkv_shapes))
        assert not _routes("gated_delta_rule", lambda: dew.jit(scan_fp32_state_cancel).trace(*qkv_shapes))
    with generation.assumed("sm90"):
        assert not _routes("gated_delta_rule", lambda: dew.jit(scan_fp32_state_cancel).trace(*qkv_shapes))

    s_cancel = jnp.zeros((batch, hv, dk, dv), jnp.float32).at[:, :, 0, 0].set(np.float32(0.0625)).at[
        :, :, 1, 0].set(np.float32(-(0.0625 + 2.0 ** -16)))
    q_cancel = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16).at[0, :, :, :2].set(jnp.bfloat16(0.125))
    z_k = jnp.zeros((seq, batch, hv, dk), jnp.bfloat16)
    z_v = jnp.zeros((seq, batch, hv, dv), jnp.bfloat16)
    z_g = jnp.zeros((seq, batch, hv), jnp.float32)
    z_b = jnp.zeros((seq, batch, hv), jnp.bfloat16)
    _assert_fla_cancel_preserved(
        scan_fp32_state_cancel, s_cancel, q_cancel, z_k, z_v, z_g, z_b,
        token_idx=0, expected=float(jnp.bfloat16(-(2.0 ** -19))))


def test_recurrence_recognition_guards_later_chunk_carry_and_unsound_norms():
    """`RECURRENCES` preserves multi-chunk scans whose carried fp32 state would narrow at chunk 1+, and
    rejects keys normalized through lossy `bf16->fp16` casts or underflowing `fp16` squares."""
    import dew

    batch, hv, dk, dv = 1, 2, 64, 64
    step = _gdn_make_step(batch, hv)

    # 1. 128-token bf16-initializer carry cancellation across chunk boundary (token 0 writes -> token 64).
    seq128 = 128
    shapes128 = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq128, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq128, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq128, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq128, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq128, batch, hv), jnp.bfloat16),
    )
    scan_bf16_init_128 = _scan_bf16_state_init(step)
    with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
        assert not _routes("gated_delta_rule", lambda: dew.jit(scan_bf16_init_128).trace(*shapes128))
    with generation.assumed("sm90"):
        assert not _routes("gated_delta_rule", lambda: dew.jit(scan_bf16_init_128).trace(*shapes128))

    s0 = jnp.zeros((batch, hv, dk, dv), jnp.float32).at[:, :, 0, 0].set(np.float32(0.0625)).at[
        :, :, 1, 0].set(np.float32(-0.0625))
    q128 = jnp.zeros((seq128, batch, hv, dk), jnp.bfloat16).at[64, :, :, :2].set(jnp.bfloat16(0.125))
    k128 = jnp.zeros((seq128, batch, hv, dk), jnp.bfloat16).at[0, :, :, 1].set(jnp.bfloat16(0.125))
    v128 = jnp.zeros((seq128, batch, hv, dv), jnp.bfloat16).at[0, :, :, 0].set(jnp.bfloat16(-33.0 / 4096.0))
    g128 = jnp.zeros((seq128, batch, hv), jnp.float32)
    b128 = jnp.zeros((seq128, batch, hv), jnp.bfloat16)
    _assert_fla_cancel_preserved(
        scan_bf16_init_128, s0, q128, k128, v128, g128, b128,
        token_idx=64, expected=float(jnp.bfloat16(-(2.0 ** -19))))

    # 2. Lossy bf16 -> fp16 -> fp32 norm denominator must not certify unit_norm or admit FLA WY overflow.
    seq64 = 64
    shapes64 = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.bfloat16),
    )
    def scan_lossy_norm_key(state, q, k, v, g, beta):
        k_f16 = k.astype(jnp.float16).astype(jnp.float32)
        inv = jax.lax.rsqrt(jnp.sum(jnp.square(k_f16), axis=-1, keepdims=True) + jnp.float32(2.0 ** -100))
        k_bad = (k.astype(jnp.float32) * inv).astype(jnp.bfloat16)
        return jax.lax.scan(step, jnp.zeros_like(state), (
            jnp.zeros_like(q), k_bad, jnp.zeros_like(v), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    def scan_fp16_sq_underflow_key(state, q, k, v, g, beta):
        inv = jax.lax.rsqrt(
            jnp.sum(jnp.square(k), axis=-1, keepdims=True).astype(jnp.float32) + jnp.float32(2.0 ** -100))
        k_bad = (k.astype(jnp.float32) * inv).astype(jnp.bfloat16)
        return jax.lax.scan(step, jnp.zeros_like(state), (
            jnp.zeros_like(q), k_bad, jnp.zeros_like(v), jnp.minimum(g, 0.0), jax.nn.sigmoid(beta)))

    shapes64_f16k = (*shapes64[:2], jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.float16), *shapes64[3:])
    z_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
    z_q = jnp.zeros((seq64, batch, hv, dk), jnp.bfloat16)
    z_v = jnp.zeros((seq64, batch, hv, dv), jnp.bfloat16)
    z_g = jnp.zeros((seq64, batch, hv), jnp.float32)
    z_b = jnp.zeros((seq64, batch, hv), jnp.bfloat16)
    for bad_scan, bad_shapes, k_val in (
        (scan_lossy_norm_key, shapes64,
         jnp.zeros((seq64, batch, hv, dk), jnp.bfloat16).at[:, :, :, 0].set(jnp.bfloat16(2.0 ** -26))),
        (scan_fp16_sq_underflow_key, shapes64_f16k,
         jnp.zeros((seq64, batch, hv, dk), jnp.float16).at[:, :, :, 0].set(jnp.float16(2.0 ** -14))),
    ):
        with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
            assert not _routes("gated_delta_rule", lambda f=bad_scan, s=bad_shapes: dew.jit(f).trace(*s))
        with generation.assumed("sm90"):
            assert not _routes("gated_delta_rule", lambda f=bad_scan, s=bad_shapes: dew.jit(f).trace(*s))
        with generation.chosen(gated_delta_rule="fla"):
            ref_f, ref_o = jax.jit(bad_scan)(z_s, z_q, k_val, z_v, z_g, z_b)
            got_f, got_o = dew.jit(bad_scan)(z_s, z_q, k_val, z_v, z_g, z_b)
            np.testing.assert_array_equal(np.asarray(got_f), np.asarray(ref_f))
            np.testing.assert_array_equal(np.asarray(got_o, np.float32), np.asarray(ref_o, np.float32))


def test_recurrence_recognition_accurate_intra_chunk_and_observed_layout():
    """Intra-chunk 2-token witness preserves exact `-2^-23` under `forms="accurate"` while holding fast-form
    finiteness/zeros (`docs/design/algebra.md §3`), and `chunk_forward.observed()` + `_recurrence_kernels`
    capture Mosaic `Call.held` & `Call.chunk` directly from forward/training dispatch."""
    import dew
    from dew.algebra import recognize
    from dew.nn.kernels import recurrence
    from dew.nn.kernels.templates.recurrence import chunk_forward
    from dew.nn.precision import Numerics

    seq64, batch, hv, dk, dv = 64, 1, 2, 64, 64
    shapes64 = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.bfloat16),
    )
    z_s = jnp.zeros((batch, hv, dk, dv), jnp.float32)
    z_q = jnp.zeros((seq64, batch, hv, dk), jnp.bfloat16)
    z_v = jnp.zeros((seq64, batch, hv, dv), jnp.bfloat16)
    z_g = jnp.zeros((seq64, batch, hv), jnp.float32)
    z_b = jnp.zeros((seq64, batch, hv), jnp.bfloat16)
    scan_64 = _scan_bf16_state_init(_gdn_make_step(batch, hv))
    q_intra = jnp.zeros((seq64, batch, hv, dk), jnp.bfloat16).at[1, :, :, 0].set(jnp.bfloat16(0.125))
    k_intra = jnp.zeros((seq64, batch, hv, dk), jnp.bfloat16).at[:2, :, :, 0].set(jnp.bfloat16(1.0 / 64.0))
    v_intra = jnp.zeros((seq64, batch, hv, dv), jnp.bfloat16).at[0, :, :, 0].set(
        jnp.bfloat16(1.0)).at[1, :, :, 0].set(jnp.bfloat16(-1.0))
    with generation.run_precision(Numerics(forms="accurate")):
        with generation.assumed("sm90"):
            assert not _routes("gated_delta_rule", lambda: dew.jit(scan_64).trace(*shapes64))
        with generation.assumed("sm90"), generation.chosen(gated_delta_rule="fla"):
            assert not _routes("gated_delta_rule", lambda: dew.jit(scan_64).trace(*shapes64))
        _assert_fla_cancel_preserved(
            scan_64, z_s, q_intra, k_intra, v_intra, z_g, z_b,
            token_idx=1, expected=float(jnp.bfloat16(-(2.0 ** -23))))
    fast_f, fast_o = dew.jit(scan_64)(z_s, q_intra, k_intra, v_intra, z_g, z_b)
    assert np.isfinite(np.asarray(fast_f)).all() and np.isfinite(np.asarray(fast_o, np.float32)).all()
    zero_f, zero_o = dew.jit(scan_64)(z_s, z_q, k_intra, z_v, z_g, z_b)
    assert not np.asarray(zero_f).any() and not np.asarray(zero_o, np.float32).any()

    # `chunk_forward.observed()` and `_recurrence_kernels` capture Mosaic `Call.held` & `Call.chunk` live
    # (`sm80` TF32 10b, `sm90` 2-part bf16 15b, `sm90` unrounded 3-part 23b, and nothing outside `observed`).
    f32_shapes = (
        jax.ShapeDtypeStruct((batch, hv, dk, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv, dk), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv, dv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.float32),
        jax.ShapeDtypeStruct((seq64, batch, hv), jnp.float32),
    )
    vars_f32 = jax.make_jaxpr(lambda *xs: xs)(*f32_shapes).jaxpr.invars
    vars_bf16 = jax.make_jaxpr(lambda *xs: xs)(*shapes64).jaxpr.invars
    roles = (("query", 1), ("key", 2), ("value", 3), ("log_decay", 4), ("beta", 5))
    delta_hooks = recurrence.Hooks(decay="scalar", delta=True)
    wg_hooks = recurrence.Hooks(decay="scalar", delta=False, write_gate=True)
    call_f32 = functools.partial(
        recognize._recurrence_chunk_call, hooks=delta_hooks, roles=roles, out_dtype=jnp.float32)
    call_wg = functools.partial(
        recognize._recurrence_chunk_call, hooks=wg_hooks, roles=roles, out_dtype=jnp.bfloat16)
    assert chunk_forward._OBSERVED.get() is None
    with jax.default_matmul_precision("default"), generation.chosen(gated_delta_rule="mosaic_gpu"):
        with generation.assumed("sm80"):
            _, k80, lay80 = recognize._recurrence_kernels(
                call_f32, vars_f32, (delta_hooks, roles, jnp.float32), vjp=False)
        with generation.assumed("sm90"):
            _, k90, lay90 = recognize._recurrence_kernels(
                call_f32, vars_f32, (delta_hooks, roles, jnp.float32), vjp=False)
            _, k90_bwd, lay90_bwd = recognize._recurrence_kernels(
                call_f32, vars_f32, (delta_hooks, roles, jnp.float32), vjp=True)
            _, kwg, laywg = recognize._recurrence_kernels(
                call_wg, vars_bf16, (wg_hooks, roles, jnp.bfloat16), vjp=False)
            with chunk_forward.observed() as direct_calls:
                jax.eval_shape(call_wg, *shapes64)
            jax.eval_shape(call_wg, *shapes64)
    assert chunk_forward._OBSERVED.get() is None
    assert k80 == {"mosaic_gpu"} and lay80 is not None and lay80[:2] == (jnp.dtype(jnp.float32), 10)
    assert k90 == {"mosaic_gpu"} and lay90 is not None and lay90[:2] == (jnp.dtype(jnp.float32), 15)
    assert k90_bwd == {"mosaic_gpu"} and lay90_bwd is not None
    assert lay90_bwd[:2] == (jnp.dtype(jnp.float32), 15)
    assert kwg == {"mosaic_gpu"} and laywg is not None and laywg[:2] == (jnp.dtype(jnp.float32), 23)
    assert len(direct_calls) == 1 and int(direct_calls[0].chunk) == laywg[3]
    assert recognize._operand_precision(direct_calls[0].held) == laywg[:3]
