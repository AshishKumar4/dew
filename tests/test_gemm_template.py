"""The grouped GEMM template (`dew.nn.kernels.templates.gemm`): its prologue and epilogue through the
Triton (sm80) and Mosaic GPU (sm90) launches, interpreted on the CPU, held to check 1 against the plain
program they stand for (`lax.ragged_dot` between the same hooks and casts, differentiated by JAX) from
float64 truth, forward and reverse; its forward mode; the fuser's value; its CUDA lowering on an H100."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import fuser
from jax.sharding import Mesh
from reference_error import assert_as_exact

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates.gemm import fused, template

pytestmark = pytest.mark.kernel

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = 307, 128, 128
ROUTED = sum(SIZES)
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them)."""


def _wide(x):
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def normed(x, mean_square, weight):
    """An RMS norm's apply as a prologue: rows scaled by their inverse root, depth by the weight."""
    return (_wide(x) * jax.lax.rsqrt(mean_square)[:, None] * weight[None, :]).astype(x.dtype)


def normed_broadcast(x, mean_square, weight):
    """`normed` with the mean square a `[rows, 1]` column, broadcast by its dimension of one."""
    return (_wide(x) * jax.lax.rsqrt(mean_square) * weight[None, :]).astype(x.dtype)


def gated(gate, up):
    """SwiGLU's product as the down projection's prologue, rounded where the reference rounds it."""
    activated = jax.nn.silu(_wide(gate)).astype(gate.dtype)
    return (_wide(activated) * _wide(up)).astype(gate.dtype)


def casted(x, scale):
    """A prologue returning fp32 from 16-bit rows: the template rounds it to the rows' dtype."""
    return _wide(x) * scale[None, :]


def shifted(x, side):
    """A prologue adding a `[depth, rows]` side operand's transpose."""
    return (_wide(x) + side.T).astype(x.dtype)


def reciprocal(x):
    return 1 / x


def biased(acc, bias):
    return acc + bias[None, :]


def biased_row(acc, bias):
    """`biased` with the bias a `[1, width]` row."""
    return acc + bias


def scaled(acc, scale):
    return acc * scale[None, :]


def doubled(acc):
    return acc, acc * 2


@dataclasses.dataclass(frozen=True)
class Case:
    """One call of the template: its hooks, their side operands from the drawn inputs, its outputs."""
    prologue: object = None
    prologue_operands: object = lambda x, w, ms, nw, b, u, s: ()
    epilogue: object = None
    epilogue_operands: object = lambda x, w, ms, nw, b, u, s: ()
    out_dtypes: tuple = (jnp.bfloat16,)
    dense: bool = False


CASES = {
    "plain": Case(),
    "norm and bias": Case(normed, lambda x, w, ms, nw, b, u, s: ((ms, ("rows",)), (nw, ("depth",))),
                          biased, lambda x, w, ms, nw, b, u, s: ((b, ("columns",)),)),
    "gated prologue": Case(gated, lambda x, w, ms, nw, b, u, s: ((u, ("rows", "depth")),)),
    "cast prologue": Case(casted, lambda x, w, ms, nw, b, u, s: ((s, ("depth",)),)),
    "broadcast axes": Case(normed_broadcast,
                           lambda x, w, ms, nw, b, u, s: ((ms[:, None], ("rows", None)), (nw, ("depth",))),
                           biased_row, lambda x, w, ms, nw, b, u, s: ((b[None, :], (None, "columns")),)),
    "scaled": Case(epilogue=scaled, epilogue_operands=lambda x, w, ms, nw, b, u, s: ((s, ("columns",)),),
                   out_dtypes=(jnp.float32,)),
    "two outputs": Case(epilogue=doubled, out_dtypes=(jnp.bfloat16, jnp.float32)),
    "wide output": Case(out_dtypes=(jnp.float32,)),
    "dense": Case(out_dtypes=(jnp.float32,), dense=True),
}
GRADIENT_CASES = ("norm and bias", "gated prologue", "cast prologue", "broadcast axes", "scaled",
                  "two outputs", "wide output")


def _draw(seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(len(SIZES), DEPTH, WIDTH)) / np.sqrt(DEPTH), jnp.float32),
            jnp.asarray(rng.uniform(0.5, 2.0, size=(ROWS,)), jnp.float32),
            jnp.asarray(rng.normal(size=(DEPTH,)), jnp.float32),
            jnp.asarray(rng.normal(size=(WIDTH,)), jnp.float32),
            jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.uniform(0.5, 1.5, size=(DEPTH,)), jnp.float32),
            jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.float32))


def _dew(case, **kw):
    def run(x, w, ms, nw, b, u, s):
        drawn = (x, w, ms, nw, b, u, s)
        outs = template.gemm(x, w[1] if case.dense else w, None if case.dense else jnp.asarray(SIZES),
                             prologue=case.prologue, prologue_operands=case.prologue_operands(*drawn),
                             epilogue=case.epilogue, epilogue_operands=case.epilogue_operands(*drawn),
                             out_dtype=case.out_dtypes[0],
                             out_dtypes=case.out_dtypes if len(case.out_dtypes) > 1 else None, **kw)
        return outs if isinstance(outs, tuple) else (outs,)
    return run


def _plain(case, out_dtypes=None):
    """The program the template stands for: the prologue rounded to the rows' dtype, each group's rows
    times its matrix rounded alike, summed wide, the epilogue, each output rounded once."""
    def run(x, w, ms, nw, b, u, s):
        drawn, h = (x, w, ms, nw, b, u, s), x
        if case.prologue is not None:
            h = case.prologue(x, *(a for a, _ in case.prologue_operands(*drawn))).astype(x.dtype)
        wide = jnp.promote_types(x.dtype, jnp.float32)
        if case.dense:
            acc = jnp.dot(h, w[1].astype(x.dtype), preferred_element_type=wide)
        else:
            acc = jax.lax.ragged_dot(h, w.astype(x.dtype), jnp.asarray(SIZES), preferred_element_type=wide)
        if case.epilogue is not None:
            acc = case.epilogue(acc, *(a for a, _ in case.epilogue_operands(*drawn)))
        outs = acc
        outs = outs if isinstance(outs, tuple) else (outs,)
        dtypes = out_dtypes or case.out_dtypes
        return tuple(out.astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return run


def _drop(f):
    """`f` of the drawn inputs but the cotangent weights."""
    return lambda x, w, ms, nw, b, u, s, ct: f(x, w, ms, nw, b, u, s)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_template_holds_check_1_with_its_hooks(generation_name, case):
    """Each hook through the launch the generation picks, interpreted on the CPU: as exact as the plain
    program from float64 truth, the rows past the routed ones included."""
    spec = CASES[case]
    with generation.assumed(generation_name):
        dew = jax.jit(_drop(_dew(spec, interpret_on_cpu=True)))
        assert_as_exact(dew, jax.jit(_drop(_plain(spec))), _draw, f"{case} on {generation_name}",
                        seeds=SEEDS, truth_fn=_drop(_plain(spec, (jnp.float64,) * len(spec.out_dtypes))))


def _loss(run):
    def loss(x, w, ms, nw, b, u, s, ct):
        outs = run(x, w, ms, nw, b, u, s)
        return sum(jnp.sum(_wide(out) * ct.astype(_wide(out).dtype)) for out in outs)
    return jax.grad(loss, argnums=tuple(range(7)))


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_the_templates_gradients_hold_check_1(generation_name, case):
    """Every operand's cotangent through the template's own reverse mode, against JAX's derivative of
    the plain program, from float64 truth. Mosaic's interpreter runs ordered callbacks, which a jitted
    `linear_call` cannot carry, so its gradients run eagerly."""
    spec = CASES[case]
    with generation.assumed(generation_name):
        dew = _loss(_dew(spec, interpret_on_cpu=True))
        dew = dew if generation_name == "sm90" else jax.jit(dew)
        assert_as_exact(dew, jax.jit(_loss(_plain(spec))), _draw, f"{case} gradients on {generation_name}",
                        seeds=SEEDS, truth_fn=_loss(_plain(spec, (jnp.float64,) * len(spec.out_dtypes))))


def test_an_empty_rows_undefined_derivative_reaches_no_shared_gradient():
    """A row past the routed ones whose mean square is zero has no defined derivative of `rsqrt`; it
    multiplied nothing, so the norm weight's and the matrix's gradients are the routed row's alone, and
    the empty row's own are zero."""
    x = jnp.zeros((2, DEPTH), jnp.bfloat16).at[0].set(1)
    ms, nw = jnp.asarray([1.0, 0.0]), jnp.ones((DEPTH,))
    w = jnp.ones((1, DEPTH, WIDTH)) / DEPTH

    def through(x, ms, nw, w, *, sizes, interpret):
        out = template.gemm(x, w, jnp.asarray(sizes), prologue=normed,
                            prologue_operands=((ms, ("rows",)), (nw, ("depth",))), out_dtype=jnp.float32,
                            interpret_on_cpu=interpret)
        return jnp.sum(out)

    with generation.assumed("sm80"):
        dx, dms, dnw, dw = jax.jit(jax.grad(functools.partial(through, sizes=[1], interpret=True),
                                            (0, 1, 2, 3)))(x, ms, nw, w)
        alone = jax.jit(jax.grad(functools.partial(through, sizes=[1], interpret=False), (0, 1, 2, 3)))(
            x[:1], ms[:1], nw, w)
    np.testing.assert_array_equal(np.asarray(dnw), np.asarray(alone[2]))
    np.testing.assert_array_equal(np.asarray(dw), np.asarray(alone[3]))
    assert not np.any(np.asarray(dx[1], np.float32)) and float(dms[1]) == 0


def test_a_wide_cotangent_meets_the_rounded_matrix_once():
    """An fp32 output's cotangent 1.00390625 times a bf16 matrix entry 1.0078125 rounds once to bf16's
    1.015625, as JAX's derivative of the plain program has it; rounding the cotangent first would give
    1.0078125."""
    x = jnp.zeros((1, DEPTH), jnp.bfloat16)
    w = jnp.zeros((1, DEPTH, WIDTH)).at[0, 0, 0].set(1.0078125)
    ct = jnp.zeros((1, WIDTH)).at[0, 0].set(1.00390625)

    def loss(x, w, *, interpret):
        return jnp.sum(template.gemm(x, w, jnp.asarray([1]), out_dtype=jnp.float32,
                                     interpret_on_cpu=interpret) * ct)

    with generation.assumed("sm80"):
        dx = jax.jit(jax.grad(functools.partial(loss, interpret=True)))(x, w)
    assert float(dx[0, 0]) == 1.015625


def test_the_prologues_rounding_is_in_its_derivative():
    """A prologue returning fp32 is rounded to the rows' dtype before its product, so its cotangent is
    rounded there too: the template's scale gradient is JAX's of the plain program, bit for bit."""
    x = jnp.ones((1, DEPTH), jnp.bfloat16)
    w = jnp.zeros((1, DEPTH, WIDTH)).at[0, :, :2].set(0.503)
    scale = jnp.ones((DEPTH,))
    ct = jnp.zeros((1, WIDTH)).at[0, :2].set(jnp.asarray([1.0, 0.125]))

    def loss(scale, *, interpret):
        out = template.gemm(x, w, jnp.asarray([1]), prologue=casted,
                            prologue_operands=((scale, ("depth",)),), out_dtype=jnp.float32,
                            interpret_on_cpu=interpret)
        return jnp.sum(out * ct)

    def plain(scale):
        h = casted(x, scale).astype(x.dtype)
        return jnp.sum(jax.lax.ragged_dot(h, w.astype(x.dtype), jnp.asarray([1]),
                                          preferred_element_type=jnp.float32) * ct)

    with generation.assumed("sm80"):
        got = jax.jit(jax.grad(functools.partial(loss, interpret=True)))(scale)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jax.jit(jax.grad(plain))(scale)))


@pytest.mark.parametrize("generation_name", ["sm80"])
def test_a_prologue_never_sees_the_depths_padding(generation_name):
    """A depth no tile divides is padded with zeros, which `reciprocal` turns to infinities: the kernel
    zeroes them again before the product, so every output is the routed rows' sum of 17 ones."""
    x, w = jnp.ones((3, 17), jnp.bfloat16), jnp.ones((1, 17, 16))
    with generation.assumed(generation_name):
        out = jax.jit(lambda x, w: template.gemm(x, w, jnp.asarray([3]), prologue=reciprocal,
                                                 out_dtype=jnp.float32, interpret_on_cpu=True))(x, w)
    np.testing.assert_array_equal(np.asarray(out), np.full((3, 16), 17.0))


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_side_operands_rows_may_be_any_of_its_dimensions(generation_name):
    """A side operand's rows as its second dimension load each group's rows as its first would: the
    forward product and the kernel gradient are the same bit for bit, whatever rows are a multiple of."""
    rows, sizes = 136, jnp.asarray([70, 66])
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(rows, 64)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(2, 64, 64)) / 8, jnp.float32)
    dy = jnp.asarray(rng.normal(size=(rows, 64)), jnp.bfloat16)
    side = jnp.asarray(rng.normal(size=(rows, 64)), jnp.float32)

    def row_major(x, side):
        return (_wide(x) + side).astype(x.dtype)

    layouts = ((row_major, ("rows", "depth"), side), (shifted, ("depth", "rows"), side.T))

    def call(entry, hook, layout):
        return jax.jit(lambda a, b, operand: entry(a, b, sizes, prologue=hook,
                                                   prologue_operands=((operand, layout),),
                                                   out_dtype=jnp.float32, interpret_on_cpu=True))

    with generation.assumed(generation_name):
        products = [call(template.gemm, hook, layout)(x, w, operand) for hook, layout, operand in layouts]
        gradients = [call(template.gemm_t, hook, layout)(x, dy, operand) for hook, layout, operand in layouts]
    np.testing.assert_array_equal(np.asarray(products[0]), np.asarray(products[1]))
    np.testing.assert_array_equal(np.asarray(gradients[0]), np.asarray(gradients[1]))


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_broadcast_dimension_is_the_squeezed_operands(generation_name):
    """A `[rows, 1]` mean square and a `[1, width]` bias with a None axis give the products of their
    vectors, bit for bit."""
    drawn = _draw(2)
    with generation.assumed(generation_name):
        column = jax.jit(_drop(_dew(CASES["broadcast axes"], interpret_on_cpu=True)))(*drawn)
        vector = jax.jit(_drop(_dew(CASES["norm and bias"], interpret_on_cpu=True)))(*drawn)
    np.testing.assert_array_equal(np.asarray(column[0], np.float32), np.asarray(vector[0], np.float32))


def test_a_side_operand_names_one_launch_axis_per_dimension():
    """Axes the hook cannot take are refused before any launch, naming the operand and its axes."""
    x, w = jnp.ones((8, DEPTH), jnp.bfloat16), jnp.ones((1, DEPTH, WIDTH))
    for operand, axes in ((jnp.ones((8,)), ("columns",)), (jnp.ones((8, 2)), ("rows", None)),
                          (jnp.ones((1,)), (None,))):
        with pytest.raises(ValueError, match="prologue side operand of shape"):
            template.gemm(x, w, jnp.asarray([8]), prologue=normed, prologue_operands=((operand, axes),),
                          out_dtype=jnp.float32)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_the_epilogue_applies_to_every_row(generation_name):
    """A row past the routed ones sums nothing: its output is the epilogue of zero, the bias."""
    x, w, _, _, b, *_ = _draw(1)
    with generation.assumed(generation_name):
        out = jax.jit(lambda x, w, b: template.gemm(
            x, w, jnp.asarray(SIZES), epilogue=biased, epilogue_operands=((b, ("columns",)),),
            out_dtype=jnp.float32, interpret_on_cpu=True))(x, w, b)
    expected = np.broadcast_to(np.asarray(b), (ROWS - ROUTED, WIDTH))
    np.testing.assert_array_equal(np.asarray(out)[ROUTED:], expected)


def test_the_backward_reads_what_the_epilogues_derivative_reads():
    """A bias's VJP passes its cotangent through and reads no accumulator; a scale's reads it, and its
    cotangent is no output's unchanged; two outputs' sum theirs."""
    acc, side = (ROWS, WIDTH), (jax.ShapeDtypeStruct((WIDTH,), jnp.float32),)
    bf16 = (jnp.dtype(jnp.bfloat16),)
    assert not template._reads_accumulator(biased, acc, side)
    assert template._passes_cotangent(biased, acc, side, bf16, jnp.bfloat16)
    assert template._reads_accumulator(scaled, acc, side)
    assert not template._passes_cotangent(scaled, acc, side, bf16, jnp.bfloat16)
    assert not template._passes_cotangent(biased, acc, side, (jnp.dtype(jnp.float32),), jnp.bfloat16)
    assert not template._passes_cotangent(doubled, acc, (), bf16 * 2, jnp.bfloat16)


def test_forward_mode_takes_the_plain_programs_tangent():
    """`jax.jvp` of the template through its kernels gives the portable arithmetic's tangent, the value
    from the kernels (`generation.linear_tangent`)."""
    x, w, ms, nw, b, u, s, _ = _draw(0)
    spec = CASES["norm and bias"]
    with generation.assumed("sm80"):
        def through(*, interpret):
            return lambda x, w: _dew(spec, interpret_on_cpu=interpret)(x, w, ms, nw, b, u, s)[0]

        value, tangent = jax.jit(lambda x, w: jax.jvp(through(interpret=True), (x, w), (x, w)))(x, w)
        portable_value, portable_tangent = jax.jit(
            lambda x, w: jax.jvp(through(interpret=False), (x, w), (x, w)))(x, w)
    np.testing.assert_array_equal(np.asarray(tangent, np.float32), np.asarray(portable_tangent, np.float32))
    assert value.dtype == portable_value.dtype == jnp.bfloat16


def bias_cast(prod, b, r):
    return (prod + b[None, :]).astype(jnp.bfloat16)


def silu_residual(prod, b, r):
    return (jax.nn.silu(prod * 2) + r.astype(jnp.float32)).astype(jnp.bfloat16)


def two_consumers(prod, b, r):
    return prod.astype(jnp.bfloat16), prod + b[None, :]


def positional(prod, b, r):
    return (prod + jax.lax.broadcasted_iota(jnp.float32, prod.shape, 1)).astype(jnp.bfloat16)


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("consumer", [bias_cast, silu_residual, two_consumers, positional],
                         ids=lambda consumer: consumer.__name__)
def test_the_fused_epilogue_keeps_the_programs_value(generation_name, consumer):
    """Under `fuser.fuse`, a product's consumers run in the epilogue (or after it in XLA, where they read
    the tile's position) with the unfused program's value on every row: as exact as it from float64."""
    x, w, _, _, b, u, *_ = _draw(3)
    r = u[:, :WIDTH]
    sizes = jnp.asarray(SIZES)

    def unfused(x, w, b, r):
        return consumer(template.gemm(x, w, sizes, out_dtype=jnp.float32), b, r)

    with generation.assumed(generation_name):
        def fusing(x, w, b, r):
            return consumer(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), b, r)

        got = jax.jit(fuser.fuse(fusing))(x, w, b, r)
        want = jax.jit(unfused)(x, w, b, r)
    for g, v in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        assert g.dtype == v.dtype
        np.testing.assert_allclose(np.asarray(g, np.float32), np.asarray(v, np.float32),
                                   rtol=0, atol=float(jnp.finfo(g.dtype).eps) * 4 * float(np.max(np.abs(v))))


@pytest.mark.parametrize("launch", [grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()],
                         ids=lambda launch: type(launch).__name__)
def test_the_hooked_mosaic_launches_lower_for_an_h100(launch):
    """Each Mosaic GPU launch with a prologue (and the forward's epilogue) lowers for an H100 on the CPU
    at a fitting tile of every block shape, rows no multiple of 8, a side operand's rows its second
    dimension."""
    rows, depth, width, groups = 2044, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    prologue = (normed, (f32((rows,)), f32((depth,))), (("rows",), ("depth",)))
    if isinstance(launch, grouped_matmul.MosaicTgmm):
        args = (bf16((rows, depth)), bf16((rows, width)), sizes, (f32((depth, rows)),))
        hooks = grouped_matmul.Hooks(shifted, (("depth", "rows"),))
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


def test_a_keyed_launch_buckets_its_side_operands_rows_with_its_tokens():
    """The launch a key measures covers its side operands' rows as it covers the tokens': both at the
    rows' power of two, wherever the rows are among a side operand's dimensions."""
    rows = 307
    hooks = grouped_matmul.Hooks(shifted, (("depth", "rows"),), biased, (("columns",),))
    ba = grouped_matmul.Gmm().bind(jax.ShapeDtypeStruct((rows, DEPTH), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2, DEPTH, WIDTH), jnp.bfloat16),
                                   jax.ShapeDtypeStruct((2,), jnp.int32),
                                   (jax.ShapeDtypeStruct((DEPTH, rows), jnp.float32),),
                                   (jax.ShapeDtypeStruct((WIDTH,), jnp.float32),),
                                   out_dtype="bfloat16", trans_rhs=False, hooks=hooks, interpret=False)
    keyed = grouped_matmul.Gmm().keyed(ba)
    assert keyed.arguments["tokens"].shape == (512, DEPTH)
    assert keyed.arguments["prologue_operands"][0].shape == (DEPTH, 512)
    assert keyed.arguments["epilogue_operands"][0].shape == (WIDTH,)
