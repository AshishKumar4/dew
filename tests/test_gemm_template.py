"""The grouped GEMM template (`dew.nn.kernels.templates.gemm.template`): its prologue and epilogue through
the Triton (sm80) and Mosaic GPU (sm90) launches, interpreted on the CPU, held to check 1 against XLA's
portable arithmetic from float64 truth, forward and reverse; its forward mode; its CUDA lowering on an
H100 with both hooks."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from reference_error import assert_as_exact

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates.gemm import template

pytestmark = pytest.mark.kernel

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = 307, 128, 128
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of XLA's arithmetic on them)."""


def _wide(x):
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def normed(x, mean_square, weight):
    """An RMS norm's apply as a prologue: rows scaled by their inverse root, depth by the weight."""
    return (_wide(x) * jax.lax.rsqrt(mean_square)[:, None] * weight[None, :]).astype(x.dtype)


def gated(gate, up):
    """SwiGLU's product as the down projection's prologue, rounded where the reference rounds it."""
    activated = jax.nn.silu(_wide(gate)).astype(gate.dtype)
    return (_wide(activated) * _wide(up)).astype(gate.dtype)


def biased(acc, bias):
    return acc + bias[None, :]


def scaled(acc, scale):
    return acc * scale[None, :]


def doubled(acc):
    return acc, acc * 2


def _draw(seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH), jnp.float32),
            jnp.asarray(rng.uniform(0.5, 2.0, size=(ROWS,)), jnp.float32),
            jnp.asarray(rng.normal(size=(DEPTH,)), jnp.float32),
            jnp.asarray(rng.normal(size=(WIDTH,)), jnp.float32),
            jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16))


CASES = {
    "plain": lambda x, w, ms, nw, b, u, **kw: template.gemm(x, w, jnp.asarray(SIZES), out_dtype=jnp.bfloat16,
                                                           **kw),
    "norm and bias": lambda x, w, ms, nw, b, u, **kw: template.gemm(
        x, w, jnp.asarray(SIZES), prologue=normed, prologue_operands=((ms, ("rows",)), (nw, ("depth",))),
        epilogue=biased, epilogue_operands=((b, ("columns",)),), out_dtype=jnp.bfloat16, **kw),
    "gated prologue": lambda x, w, ms, nw, b, u, **kw: template.gemm(
        x, w, jnp.asarray(SIZES), prologue=gated, prologue_operands=((u, ("rows", "depth")),),
        out_dtype=jnp.bfloat16, **kw),
    "two outputs": lambda x, w, ms, nw, b, u, **kw: template.gemm(
        x, w, jnp.asarray(SIZES), epilogue=doubled, out_dtype=jnp.bfloat16,
        out_dtypes=(jnp.bfloat16, jnp.float32), **kw),
    "dense": lambda x, w, ms, nw, b, u, **kw: template.gemm(x, w[1], None, out_dtype=jnp.float32, **kw),
}


def _truth(case):
    """`case`'s arithmetic in float64: each group's rows of the prologue's output times its matrix, then
    the epilogue, nothing rounded."""
    group = np.repeat(np.arange(len(SIZES)), SIZES)

    def product(x, w):
        out = np.zeros((ROWS, WIDTH))
        out[:len(group)] = np.einsum("rk,rkn->rn", np.asarray(x)[:len(group)], np.asarray(w)[group])
        return out

    def truth(x, w, ms, nw, b, u):
        if case == "dense":
            return x @ w[1]
        if case == "norm and bias":
            x = normed(x, ms, nw)
        if case == "gated prologue":
            x = gated(x, u)
        acc = product(x, w)
        routed = (np.arange(ROWS) < len(group))[:, None]
        if case == "norm and bias":
            return np.where(routed, acc + np.asarray(b)[None, :], 0)
        if case == "two outputs":
            return acc, acc * 2
        return acc

    return truth


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_template_holds_check_1_with_its_hooks(generation_name, case):
    """Each hook through the launch the generation picks, interpreted on the CPU: as exact as XLA's
    portable arithmetic of the same hooks, from float64 truth, the rows past the routed ones zero."""
    with generation.assumed(generation_name):
        dew = jax.jit(functools.partial(CASES[case], interpret_on_cpu=True))
        reference = jax.jit(CASES[case])
        assert_as_exact(dew, reference, _draw, f"{case} on {generation_name}", seeds=SEEDS,
                        truth_fn=_truth(case))


GRADIENT_CASES = ("norm and bias", "scaled")


def _loss(case, **kw):
    def loss(x, w, ms, nw, b, u):
        if case == "scaled":
            out = template.gemm(x, w, jnp.asarray(SIZES), epilogue=scaled,
                                epilogue_operands=((b, ("columns",)),), out_dtype=jnp.float32, **kw)
        else:
            out = CASES[case](x, w, ms, nw, b, u, **kw)
        return jnp.sum(out.astype(jnp.float32) * u[:, :1].astype(jnp.float32))
    return loss


def _gradient_truth(case):
    group = np.repeat(np.arange(len(SIZES)), SIZES)

    def loss(x, w, ms, nw, b, u):
        if case == "norm and bias":
            x = normed(x, ms, nw)
        acc = jax.lax.ragged_dot(x, w, jnp.asarray(SIZES), preferred_element_type=jnp.float64)
        routed = (jnp.arange(ROWS) < len(group))[:, None]
        out = jnp.where(routed, acc + b[None, :] if case == "norm and bias" else acc * b[None, :], 0)
        return jnp.sum(out * u[:, :1])

    return jax.grad(loss, argnums=tuple(range(6)))


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_the_templates_gradients_hold_check_1(generation_name, case):
    """Every operand's cotangent through the template's own launches (a bias's, a scale's that reads the
    accumulator, the norm's statistics and weight among them): as exact as XLA's gradients of the same
    arithmetic, from float64 truth. Mosaic's interpreter runs ordered callbacks, which a jitted
    `linear_call` cannot carry, so its gradients run eagerly."""
    with generation.assumed(generation_name):
        dew = jax.grad(_loss(case, interpret_on_cpu=True), argnums=tuple(range(6)))
        dew = dew if generation_name == "sm90" else jax.jit(dew)
        reference = jax.jit(jax.grad(_loss(case), argnums=tuple(range(6))))
        assert_as_exact(dew, reference, _draw, f"{case} gradients on {generation_name}", seeds=SEEDS,
                        truth_fn=_gradient_truth(case))


def test_forward_mode_takes_the_portable_tangent():
    """`jax.jvp` of the template through its kernels gives the portable arithmetic's tangent, the value
    from the kernels (`generation.linear_tangent`)."""
    x, w, ms, nw, b, u = _draw(0)
    with generation.assumed("sm80"):
        def through(*, interpret):
            return lambda x, w: CASES["norm and bias"](x, w, ms, nw, b, u, interpret_on_cpu=interpret)

        value, tangent = jax.jit(lambda x, w: jax.jvp(through(interpret=True), (x, w), (x, w)))(x, w)
        portable_value, portable_tangent = jax.jit(
            lambda x, w: jax.jvp(through(interpret=False), (x, w), (x, w)))(x, w)
    np.testing.assert_array_equal(np.asarray(tangent, np.float32), np.asarray(portable_tangent, np.float32))
    assert value.dtype == portable_value.dtype == jnp.bfloat16


def test_the_backward_recomputes_the_accumulator_only_where_the_epilogue_reads_it():
    """A bias's VJP reads only the cotangent; a scale's reads the accumulator, which the backward then
    recomputes."""
    bias = jax.ShapeDtypeStruct((WIDTH,), jnp.float32)
    assert not template._reads_accumulator(biased, (ROWS, WIDTH), (bias,))
    assert template._reads_accumulator(scaled, (ROWS, WIDTH), (bias,))


@pytest.mark.parametrize("launch", [grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()],
                         ids=lambda launch: type(launch).__name__)
def test_the_hooked_mosaic_launches_lower_for_an_h100(launch):
    """Each Mosaic GPU launch with a prologue (and the forward's epilogue) lowers for an H100 on the CPU
    at a fitting tile of every block shape, rows no multiple of 8."""
    rows, depth, width, groups = 2044, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    prologue = (normed, (f32((rows,)), f32((depth,))), (("rows",), ("depth",)))
    if isinstance(launch, grouped_matmul.MosaicTgmm):
        args = (bf16((rows, depth)), bf16((rows, width)), sizes, prologue[1])
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2])
        options = {"out_dtype": "float32", "hooks": hooks, "interpret": False}
    else:
        args = (bf16((rows, depth)), bf16((groups, depth, width)), sizes, prologue[1], (f32((width,)),))
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2], biased, (("columns",),))
        options = {"out_dtype": "bfloat16", "trans_rhs": False, "hooks": hooks, "interpret": False}
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        fits = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        covering = set({(c.block_m, c.block_n): c for c in reversed(fits)}.values())
        assert covering
        for config in sorted(covering, key=repr):
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


def test_an_epilogue_leaves_the_unrouted_rows_zero():
    """A bias the epilogue adds to every accumulator, the empty ones too, reaches no row past the routed
    ones, through either generation's launch."""
    x, w, ms, nw, b, u = _draw(1)
    for generation_name in ("sm80", "sm90"):
        with generation.assumed(generation_name):
            hooked = functools.partial(CASES["norm and bias"], interpret_on_cpu=True)
            out = jax.jit(hooked)(x, w, ms, nw, b, u)
        assert not np.any(np.asarray(out, np.float32)[sum(SIZES):]), generation_name
