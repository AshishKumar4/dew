"""The fusion search (docs/design/algebra.md §2 "Fusion", `dew.algebra.fusion`): a program's products and
their elementwise consumers fused into the GEMM template's epilogue by the compile-time pass, where the
template takes the product and its price says fused wins. Every program computes what it does as written,
its gradients too, whatever the pass fuses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.nn.kernels import generation
from dew.nn.precision import Numerics


def draw(seed: int = 7):
    """Integer operands whose sums pass bf16's 256, so a dot's own rounding to bf16 matters, every product
    and fp32 sum exact."""
    keys = jax.random.split(jax.random.key(seed), 4)
    return (jax.random.randint(keys[0], (256, 512), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[1], (512, 256), -7, 8).astype(jnp.bfloat16),
            jax.random.randint(keys[2], (256,), -3, 4).astype(jnp.bfloat16),
            jax.random.randint(keys[3], (256, 256), -3, 4).astype(jnp.bfloat16))


def biased(x, w, bias, other):
    """A projection, its bias and a scale after it."""
    del other
    return (x @ w + bias) * jnp.bfloat16(2)


def escaping(x, w, bias, other):
    """A projection whose product is read after its consumers too."""
    del other
    product = x @ w
    return product + bias, product


def widened(x, w, bias, other):
    """A projection rounded to bf16, then widened and added to in float32."""
    del other
    return (x @ w).astype(jnp.float32) + bias.astype(jnp.float32)


def chained(x, w, bias, other):
    """A projection whose consumer feeds a second projection, its own consumer after it."""
    hidden = jnp.maximum(x @ w + bias, jnp.bfloat16(0))
    return hidden @ other - bias


def bare(x, w, bias, other):
    """A projection alone, nothing after it."""
    del bias, other
    return x @ w


def kernel_gradient(x, w, bias, other):
    """A kernel gradient's product, xᵀ @ dy: the rows contracted."""
    del w, bias
    return x.T @ other


SIZES = (96, 160)
"""Two groups' rows of `draw`'s 256."""


def grouped(x, w, bias, other):
    """Each group's rows times its matrix (`lax.ragged_dot`, an MoE's experts), a bias after it."""
    del other
    return jax.lax.ragged_dot(x, jnp.stack([w, -w]), jnp.asarray(SIZES, jnp.int32)) + bias


def grouped_kernel_gradient(x, w, bias, other):
    """Each group's rows of x contracted with its rows of dy, `[groups, depth, columns]`: the grouped
    kernel gradient's form (`lax.ragged_dot_general`, the rows ragged within the contraction)."""
    del w, bias
    numbers = jax.lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(((0,), (0,)), ((), ())),
                                               lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
    return jax.lax.ragged_dot_general(x, other, jnp.asarray(SIZES, jnp.int32), numbers)


LAYERS = [biased, escaping, widened, chained, bare, kernel_gradient, grouped, grouped_kernel_gradient]


def as_written(layer, *operands):
    return jax.jit(layer)(*operands)


def rewritten(layer, *operands):
    return jax.jit(algebra.rewritten(layer, kernels=True))(*operands)


def assert_equal(got, expected):
    for value, reference in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(value, np.float32), np.asarray(reference, np.float32))


ARMS = [("sm90", {"grouped_matmul": "mosaic_gpu"}), ("sm80", {"grouped_matmul": "mosaic_gpu"}), ("sm90", {}),
        ("sm80", {})]
"""Each GPU and the kernels a call names: an H100's template named for its projections (an A/B's arm, where
the pass fuses their consumers into its epilogue), an A100's (whose template takes no side operand, so the
pass keeps the program as traced), and an H100's and an A100's selectors, which give a dense projection
XLA's dot."""


@pytest.mark.parametrize("layer", LAYERS, ids=lambda layer: layer.__name__)
@pytest.mark.parametrize(("generation_name", "named"), ARMS, ids=["sm90-named", "sm80-named", "sm90", "sm80"])
def test_a_layer_under_the_pass_computes_what_it_does_as_written(layer, generation_name, named):
    """Each layer under the pass computes what it does as written, bitwise, in each arm (`ARMS`): an
    epilogue reads its product as the dot rounds it, a product read after its consumers is still its own,
    a widened product widens the dot's rounding."""
    operands = draw()
    with generation.assumed(generation_name), generation.chosen(**named):
        assert_equal(rewritten(layer, *operands), as_written(layer, *operands))


@pytest.mark.parametrize("layer", [biased, chained], ids=lambda layer: layer.__name__)
def test_a_layers_gradient_under_the_pass_is_its_gradient_as_written(layer):
    """The gradient of each layer's sum against a fixed cotangent, under the pass on a GPU whose template a
    call names (its backward's products candidates of their own), is the gradient as written, bitwise."""
    operands = draw()
    cotangent = jax.random.randint(jax.random.key(11), (256, 256), -2, 3).astype(jnp.bfloat16)

    def loss(f):
        return lambda *xs: jnp.sum((f(*xs) * cotangent).astype(jnp.float32))

    grad = jax.grad(loss(layer), argnums=(0, 1, 2))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
        assert_equal(jax.jit(algebra.rewritten(grad))(*operands), jax.jit(grad)(*operands))


@pytest.mark.parametrize("forms", ["fast", "accurate"])
@pytest.mark.parametrize("layer", [bare, kernel_gradient, grouped, grouped_kernel_gradient],
                         ids=lambda layer: layer.__name__)
def test_a_plain_programs_product_runs_the_template_its_call_names(layer, forms):
    """A plain JAX program's product of each form the template takes (`fusion.Form`: alone, a kernel
    gradient's, each group's rows times its matrix, a grouped kernel gradient's), under the pass on an H100
    whose template a call names, runs the template's launch (its route reported as the selector's), where
    the program as written runs XLA's dot; under the accurate forms too, which limit the pass's
    reassociation alone (row 4730)."""
    operands = draw()
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"),
          generation.run_precision(Numerics(forms=forms))):
        with generation.routes() as written:
            jax.jit(layer).trace(*operands)
        with generation.routes() as passed:
            jax.jit(algebra.rewritten(layer, kernels=True)).trace(*operands)
    assert not any(route.choice == "grouped_matmul" for route in written)
    assert any(route.choice == "grouped_matmul" and route.kernel == "mosaic_gpu" for route in passed)


def _template_routes(trace) -> list:
    """The grouped matmul's routes `trace()` reports that run the template."""
    with generation.routes() as found:
        trace()
    return [route for route in found if route.choice == "grouped_matmul" and route.kernel == "mosaic_gpu"]


def test_a_plain_programs_gradient_runs_the_template_in_either_order_through_dew_jit():
    """`dew.jit` over a plain JAX loss on an H100 whose template a call names: its gradient's products run
    the template whether `jax.grad` is taken of the compiled program (inside a plain jit, the pass reads the
    program as it traces) or the program compiles the gradient (the pass reads the joint program), each
    the gradient as written, bitwise. The loss contracts its projection whole with a fixed cotangent, so
    the projection is a product alone: a fused epilogue's derivative recomputes its sums in IEEE fp32,
    which the template does not take, and a call naming it refuses."""
    import dew

    operands = draw()
    cotangent = jax.random.randint(jax.random.key(11), (256, 256), -2, 3).astype(jnp.bfloat16)

    def loss(x, w, bias, other):
        del bias, other
        return jnp.tensordot(x @ w, cotangent, axes=2).astype(jnp.float32)

    plain = jax.grad(loss, argnums=(0, 1))
    outside = jax.jit(jax.grad(dew.jit(loss), argnums=(0, 1)))
    inside = dew.jit(jax.grad(loss, argnums=(0, 1)))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"):
        assert not _template_routes(lambda: jax.jit(plain).trace(*operands))
        assert _template_routes(lambda: outside.trace(*operands))
        assert _template_routes(lambda: inside.trace(*operands))
        expected = jax.jit(plain)(*operands)
        assert_equal(outside(*operands), expected)
        assert_equal(inside(*operands), expected)


def jax_attention(*, causal: bool):
    """jax.nn's attention as a plain program writes it (its XLA form)."""
    def attend(q, k, v):
        return jax.nn.dot_product_attention(q, k, v, is_causal=causal, implementation="xla")
    return attend


def flax_attention(*, causal: bool):
    """Flax's attention as a Flax model writes it, its causal mask built from the rows."""
    from flax import linen as nn

    def attend(q, k, v):
        mask = nn.make_causal_mask(jnp.ones(q.shape[:2])) if causal else None
        return nn.dot_product_attention(q, k, v, mask=mask)
    return attend


SPELLINGS = {"jax.nn causal, grouped": (jax_attention(causal=True), 2, True),
             "jax.nn, grouped": (jax_attention(causal=False), 2, False),
             "flax causal": (flax_attention(causal=True), 4, True),
             "flax": (flax_attention(causal=False), 4, False)}
"""Each plain program's attention, its key heads over 4 query heads, and whether it is causal."""


def _attention_routes(trace) -> list[str]:
    """The kernels the attention routes `trace()` reports name."""
    with generation.routes() as found:
        trace()
    return [route.kernel for route in found if route.choice == "attention"]


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_a_plain_programs_attention_runs_dews_kernel_through_dew_jit(spelling):
    """A plain JAX or Flax program's softmax attention (`dew.algebra.attention_regions`) compiled through
    `dew.jit` on an H100 runs the kernel Dew's selector gives the call (FlashAttention, or the template a call
    names), where the program as written runs XLA's; on this host, its output and gradients under the
    template (its portable mainloop) on par with the program as written (check 1)."""
    from attention_oracle import attention_on_par, kept
    from contract_support import normal_drawn

    import dew

    attend, key_heads, causal = SPELLINGS[spelling]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),
              *(jax.ShapeDtypeStruct((2, 128, key_heads, 64), jnp.bfloat16),) * 2)
    with generation.assumed("sm90"):
        assert not _attention_routes(lambda: jax.jit(attend).trace(*shapes))
        assert "flash" in _attention_routes(lambda: dew.jit(attend).trace(*shapes))
        with generation.chosen(attention="template"):
            assert "template" in _attention_routes(lambda: dew.jit(attend).trace(*shapes))
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


def softcapped_attention(q, k, v):
    """Gemma 2's attention as a plain program writes it: scaled logits softcapped at 50, causal."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * q.shape[-1] ** -0.5
    logits = 50.0 * jnp.tanh(logits / 50.0)
    causal = jnp.tril(jnp.ones((q.shape[1], k.shape[1]), bool))
    weights = jax.nn.softmax(jnp.where(causal, logits, -jnp.inf), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


FORMS = {
    "a mask of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, mask=input_mask(q, k), implementation="xla"),
        lambda q, k, v: {"mask": input_mask(q, k)}),
    "a bias of its inputs": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, bias=input_bias(q, k), implementation="xla"),
        lambda q, k, v: {"bias": input_bias(q, k)}),
    "twice the default scale": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, scale=2 * q.shape[-1] ** -0.5,
                                                     implementation="xla"),
        lambda q, k, v: {}),
    "a causal window": (
        lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True, local_window_size=(15, 0),
                                                     implementation="xla"),
        lambda q, k, v: {"causal": True, "sliding_window": 16}),
    "a softcap": (softcapped_attention, lambda q, k, v: {"causal": True, "softcap": 50.0}),
}
"""Each plain program's attention under a form beside the default scale and a causal mask, and the arguments
Dew's own attention takes for it."""


@pytest.mark.parametrize("form", list(FORMS))
def test_a_plain_programs_attention_under_each_form_runs_the_kernel_dews_own_call_takes(form):
    """A plain program's attention under a mask or bias it draws from its inputs, a scale a power of two past
    the default, a causal window or a softcap, compiled through `dew.jit` on an H100, runs the kernel Dew's
    own attention call under those arguments routes to (none where that is XLA's), where the program as
    written runs XLA's; on this host, under the template, its output and gradients on par with the program
    as written (check 1)."""
    from attention_oracle import attention_on_par
    from contract_support import normal_drawn

    import dew
    from dew.algebra.attention_regions import KEPT
    from dew.nn.attention import scaled_dot_product_attention

    attend, arguments = FORMS[form]
    shapes = (jax.ShapeDtypeStruct((2, 128, 4, 64), jnp.bfloat16),) * 3

    def own(q, k, v):
        return scaled_dot_product_attention(q, k, v, **arguments(q, k, v))

    with generation.assumed("sm90"):
        assert not _attention_routes(lambda: jax.jit(attend).trace(*shapes))
        expected = [kernel for kernel in _attention_routes(lambda: jax.jit(own).trace(*shapes))
                    if kernel not in KEPT]
        assert set(_attention_routes(lambda: dew.jit(attend).trace(*shapes))) == set(expected)
    draw = normal_drawn(41, *((2, 64, 4, 32),) * 4, dtype=jnp.bfloat16)
    with generation.chosen(attention="template"):
        attention_on_par(dew.jit(attend), attend, draw, f"dew.jit {form}",
                         truth=lambda *x: attend(*(a.astype(jnp.float64) for a in x)))

def written_out(*, fill=-jnp.inf, logits_cast=None, initial=-jnp.inf, other_rows=False, sum_cast=None):
    """Causal softmax attention as a program writes it out (its products as einsums, a causal where, the
    safe softmax), each keyword a way it computes something other than attention: a finite fill keeping the
    future keys' weight, logits truncated or narrowed through `logits_cast`, a maximum's initial value above
    the logits, each row divided by another row's sum (heads and rows swapped in its broadcast), the sum
    narrowed through `sum_cast` (to float16, where 65536 keys' sum overflows)."""
    def attend(q, k, v):
        s = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=jnp.float32) * q.shape[-1] ** -0.5
        if logits_cast is not None:
            s = s.astype(logits_cast).astype(jnp.float32)
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


@pytest.mark.parametrize(("variant", "options"), [("attention", {}), ("finite fill", {"fill": 0.0}),
                                                  ("truncated logits", {"logits_cast": jnp.int32}),
                                                  ("initial above", {"initial": 1e30}),
                                                  ("another row's sum", {"other_rows": True}),
                                                  ("logits narrowed", {"logits_cast": jnp.float16}),
                                                  ("sum narrowed", {"sum_cast": jnp.float16})])
def test_only_attention_written_out_is_rewritten_as_dews(variant, options):
    """Causal softmax attention written out runs Dew's kernel through `dew.jit` on an H100; written out with
    any step computing otherwise (a finite fill, truncated or narrowed logits, an initial maximum above the
    logits, a row normalized by another's sum, a sum narrowed past its range), it stays as written, XLA's."""
    import dew

    shapes = (jax.ShapeDtypeStruct((1, 16, 16, 32), jnp.bfloat16),) * 3
    with generation.assumed("sm90"):
        routes = _attention_routes(lambda: dew.jit(written_out(**options)).trace(*shapes))
    assert bool(routes) == (variant == "attention"), routes


def flax_rms(x, w):
    """Flax's RMSNorm as a Flax model applies it."""
    from flax import linen as nn

    return nn.RMSNorm(epsilon=1e-6, dtype=x.dtype).apply({"params": {"scale": w}}, x)


def widened_rms(x, w):
    """An RMSNorm as a plain program writes it: float32 statistics, the weight before the cast."""
    widened = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w).astype(x.dtype)


def offset_rms(x, w):
    """Gemma's RMSNorm: the normalized row cast first, then times one more than its weight."""
    widened = x.astype(jnp.float32)
    normed = widened * jax.lax.rsqrt(jnp.mean(jnp.square(widened), axis=-1, keepdims=True) + 1e-6)
    return normed.astype(x.dtype) * (1 + w.astype(x.dtype))


def gated_product_of(a, b):
    """A gated MLP's product as a plain program writes it."""
    return jax.nn.silu(a) * b


def _norms_written(fn, *shapes):
    """`fn` with each RMSNorm and SwiGLU the compile point takes on an H100 written as Dew's norm
    (`norm_regions`), decided under the H100 and written on this host, whose norms run their portable
    arithmetic: what the compile point's rewrite computes, run here."""
    from jax.extend.core import jaxpr_as_fun

    from dew.algebra import fusion, norm_regions

    closed = jax.make_jaxpr(fn)(*shapes)
    with generation.assumed("sm90"):
        found = [region for region in norm_regions.regions(fusion.Index(closed.jaxpr))
                 if norm_regions.taken(region)]
    written = [fusion.Written(region.places, region.operands, region.outputs, region.places[-1],
                              norm_regions.written(region)) for region in found]
    return jaxpr_as_fun(closed.replace(jaxpr=fusion._spliced(closed.jaxpr, written))), len(found)


def residual_rms(x, branch, w):
    """A pre-norm block's seam as a plain program writes it: the stream `x + branch`, its RMSNorm, and the
    stream read on after it."""
    stream = x + branch
    return widened_rms(stream, w) + stream


def gated_rms(x, gate, w):
    """Qwen3-Next's gated RMSNorm as a plain program writes it: fp32 statistics, the weight, then the gate's
    SiLU, cast once."""
    widened = x.astype(jnp.float32)
    inverse = jax.lax.rsqrt(jnp.mean(widened * widened, axis=-1, keepdims=True) + 1e-6)
    return (widened * inverse * w.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))).astype(x.dtype)


def packed_swiglu(gate_up):
    """A packed gate|up projection's SwiGLU as a plain program writes it: its two halves sliced out."""
    return jax.nn.silu(gate_up[..., 0, :]) * gate_up[..., 1, :]


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
    "packed SwiGLU": (packed_swiglu, _own_packed, (PACKED,)),
}
"""Each plain program's fused norm, Dew's own call of it, and its operands' shapes."""


@pytest.mark.parametrize("form", list(NORM_FORMS))
def test_a_plain_programs_norms_run_the_kernel_dews_own_call_takes(form):
    """A plain program's RMSNorm (Flax's, a plain one, Gemma's offset weight after the cast, a residual
    stream's read on after it, Qwen3-Next's gated one) or SwiGLU (two projections, or one packed one's
    halves), compiled through `dew.jit` on an H100, runs the kernel Dew's own call routes it to (none where
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
    written, found = _norms_written(program, *shapes)
    assert found == 1
    assert_as_exact(jax.jit(with_gradients(lambda *x: written(*x)[0])), jax.jit(with_gradients(program)),
                    draw, f"dew.jit {form}", seeds=8, truth_fn=with_gradients(
                        lambda *x: program(*(a.astype(jnp.float64) for a in x))))


def _rows_inverse(x):
    """Each row's inverse root mean square, fp32 `[rows]`, the feature axis reduced."""
    widened = x.astype(jnp.float32)
    return jax.lax.rsqrt(jnp.sum(widened * widened, axis=-1) / x.shape[-1] + 1e-6)


def _named_silu(x):
    return jnp.tanh(x)


_named_silu.__name__ = "silu"


NEAR_NORMS = {
    "columns scaled by the rows' statistics":
        lambda x, w: (x.astype(jnp.float32) * _rows_inverse(x)[None, :]).astype(x.dtype),
    "a weight along the rows":
        lambda x, w: (x.astype(jnp.float32) * _rows_inverse(x)[:, None] * w[:, None]).astype(x.dtype),
    "a jit named silu that is tanh": lambda x, w: jax.jit(_named_silu)(x) * w.astype(x.dtype),
    "a zero mean multiplier": lambda x, w: (x.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) * 0.0 + 1e-6)).astype(x.dtype),
    "its own sum as the weight": lambda x, w: (
        x.astype(jnp.float32) * _rows_inverse(x)[:, None]
        * jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1)[None, :]).astype(x.dtype),
}
"""Programs a norm's recogniser reads almost as an RMSNorm or SwiGLU and computes otherwise (a square row,
so every axis has the feature axis's extent)."""


@pytest.mark.parametrize("program", list(NEAR_NORMS))
def test_a_plain_program_near_a_norm_computes_as_written(program):
    """A program whose arithmetic resembles an RMSNorm or SwiGLU without being one (the statistics or the
    weight laid along another axis, a nested program only named silu, a mean scaled by zero, the region's own
    sum read as its weight), compiled through `dew.jit` on an H100, computes what it does as written: its
    output and gradients on par with the program's (check 1), whatever of it the compile point writes as
    Dew's norm (an unweighted RMSNorm inside it, its product with another axis's weight left as written)."""
    from contract_support import normal_drawn
    from reference_error import assert_as_exact, with_gradients


    fn = NEAR_NORMS[program]
    draw = normal_drawn(47, (256, 256), (256,), (256, 256), dtype=jnp.bfloat16)
    written, _ = _norms_written(fn, jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                                jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    assert_as_exact(jax.jit(with_gradients(lambda *x: written(*x)[0])), jax.jit(with_gradients(fn)), draw,
                    f"dew.jit {program}", seeds=8,
                    truth_fn=with_gradients(lambda *x: fn(*(a.astype(jnp.float64) for a in x))))
