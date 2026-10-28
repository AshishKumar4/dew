"""The grouped GEMM template (`dew.nn.kernels.templates.gemm`): its prologue and epilogue through the
Triton (sm80), Mosaic GPU (sm90) and TPU (v5e) launches, interpreted on the CPU, held to check 1 against
the plain program they stand for (`lax.ragged_dot` between the same hooks and casts, differentiated by
JAX) from float64 truth, forward and reverse; its forward mode; the fuser's value; its CUDA lowering on
an H100 and its TPU lowering."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS, SUMS_IN_IEEE, generations, lowers, runs
from jax.experimental.pallas import fuser
from jax.sharding import AbstractDevice, AbstractMesh, AxisType, Mesh
from reference_error import as_given, assert_as_exact, with_gradients
from tuned import each_config

from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.templates import product, resources
from dew.nn.kernels.templates.gemm import fused, mosaic_gpu, mosaic_tpu, template, triton_dense

pytestmark = pytest.mark.kernel

SIZES = (0, 70, 1, 129, 100, 0)
ROWS, DEPTH, WIDTH = 307, 128, 128
ROUTED = sum(SIZES)
SEEDS = 8
"""Each site's declared draws (`reference_error.seeds_for` of the plain program on them; the sites that held
one program bitwise to another before bible §6, owner 10-27, sized paired with the kernel's arm on armada,
8 at every case)."""


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


def sided(x, side):
    """A prologue adding a `[rows, depth]` side operand."""
    return (_wide(x) + side).astype(x.dtype)


SIDE_ROWS, SIDE_SIZES = 136, (70, 66)
SIDE_LAYOUTS = ((sided, ("rows", "depth"), False), (shifted, ("depth", "rows"), True))
"""A side operand's rows as its first dimension and as its second: each prologue, its operand's axes, and
whether its operand is the side transposed."""


def _side_draw(matrix_dtype):
    """A side-operand case's draw: rows, two groups' matrices in `matrix_dtype`, a cotangent and a
    `[rows, depth]` side operand."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(SIDE_ROWS, 64)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(2, 64, 64)) / 8, matrix_dtype),
                jnp.asarray(rng.normal(size=(SIDE_ROWS, 64)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(SIDE_ROWS, 64)), jnp.float32))
    return draw


def _kernel_gradient(h, dy, sizes):
    """Each group's rows transposed times its cotangent rows, summed wide (a dense call's whole rows, its one
    group, where `sizes` is None): the kernel gradient the template stands for."""
    wide = jnp.promote_types(h.dtype, jnp.float32)
    h, dy = h.astype(wide), dy.astype(wide)
    if sizes is None:
        return jnp.dot(h.T, dy, preferred_element_type=wide)[None]
    matrices = jnp.zeros((len(sizes), h.shape[1], dy.shape[1]), wide)
    return jax.vjp(lambda m: jax.lax.ragged_dot(h, m, jnp.asarray(sizes), preferred_element_type=wide),
                   matrices)[1](dy)[0]


def _side_plain(*, truth: bool = False):
    """The program a side-operand case's product and kernel gradient stand for: the rows plus the side
    rounded to bf16, times each group's matrix rounded alike, and transposed times the cotangent, summed
    wide; as `truth`, on float64 inputs with those roundings alone."""
    def run(x, w, dy, side):
        wide = jnp.promote_types(x.dtype, jnp.float32)
        h = (x.astype(wide) + side.astype(wide)).astype(jnp.bfloat16).astype(wide)
        matrix = w.astype(jnp.bfloat16).astype(wide)
        product = jax.lax.ragged_dot(h, matrix, jnp.asarray(SIDE_SIZES), preferred_element_type=wide)
        return product, _kernel_gradient(h, dy, SIDE_SIZES)
    return run


def reciprocal(x):
    return 1 / x


def biased(acc, bias):
    return acc + bias[None, :]


def biased_row(acc, bias):
    """`biased` with the bias a `[1, width]` row."""
    return acc + bias


def scaled(acc, scale):
    return acc * scale[None, :]


def scaled_rows(acc, scale):
    return acc * scale[:, None]


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
    # Each row its own scale (routing weights'): a cooperative consumer reads its half's rows of it.
    "row scaled": Case(epilogue=scaled_rows,
                       epilogue_operands=lambda x, w, ms, nw, b, u, s: ((ms, ("rows",)),)),
    "two outputs": Case(epilogue=doubled, out_dtypes=(jnp.bfloat16, jnp.float32)),
    "wide output": Case(out_dtypes=(jnp.float32,)),
    "dense": Case(out_dtypes=(jnp.float32,), dense=True),
    "dense norm and bias": Case(normed, lambda x, w, ms, nw, b, u, s: ((ms, ("rows",)), (nw, ("depth",))),
                                biased, lambda x, w, ms, nw, b, u, s: ((b, ("columns",)),), dense=True),
    "dense scaled": Case(epilogue=scaled,
                         epilogue_operands=lambda x, w, ms, nw, b, u, s: ((s, ("columns",)),),
                         out_dtypes=(jnp.float32,), dense=True),
    "dense two outputs": Case(epilogue=doubled, out_dtypes=(jnp.bfloat16, jnp.float32), dense=True),
}
GRADIENT_CASES = ("norm and bias", "gated prologue", "cast prologue", "broadcast axes", "scaled",
                  "two outputs", "wide output", "dense norm and bias", "dense scaled", "dense two outputs")


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


def _plain(case, *, truth: bool = False):
    """The program the template stands for: the prologue rounded to the rows' dtype, each group's rows
    times its matrix rounded alike, summed wide, the epilogue, each output rounded once. As `truth`, on
    float64 inputs: the operands rounded as the program rounds them (forward only), nothing else."""
    def operand(array, dtype):
        rounded = array.astype(dtype).astype(array.dtype)
        return array + jax.lax.stop_gradient(rounded - array) if truth else rounded

    def run(x, w, ms, nw, b, u, s):
        drawn, h = (x, w, ms, nw, b, u, s), x
        if case.prologue is not None:
            h = operand(case.prologue(x, *(a for a, _ in case.prologue_operands(*drawn))), jnp.bfloat16)
        wide = jnp.promote_types(x.dtype, jnp.float32)
        matrix = operand(w, jnp.bfloat16).astype(h.dtype)
        if case.dense:
            acc = jnp.dot(h, matrix[1], preferred_element_type=wide)
        else:
            acc = jax.lax.ragged_dot(h, matrix, jnp.asarray(SIZES), preferred_element_type=wide)
        if case.epilogue is not None:
            acc = case.epilogue(acc, *(a for a, _ in case.epilogue_operands(*drawn)))
        outs = acc if isinstance(acc, tuple) else (acc,)
        dtypes = (jnp.float64,) * len(outs) if truth else case.out_dtypes
        return tuple(out.astype(dtype) for out, dtype in zip(outs, dtypes, strict=True))
    return run


def _drop(f):
    """`f` of the drawn inputs but the cotangent weights."""
    return lambda x, w, ms, nw, b, u, s, ct: f(x, w, ms, nw, b, u, s)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90", "v5e"))
@pytest.mark.parametrize("case", list(CASES))
def test_the_template_holds_check_1_with_its_hooks(generation_name, case):
    """Each hook through the launch the generation picks, interpreted on the CPU: as exact as the plain
    program from float64 truth, the rows past the routed ones included."""
    spec = CASES[case]
    with generation.assumed(generation_name):
        dew = jax.jit(_drop(_dew(spec, interpret_on_cpu=True)))
        assert_as_exact(dew, jax.jit(_drop(_plain(spec))), _draw, f"{case} on {generation_name}",
                        seeds=SEEDS, truth_fn=_drop(_plain(spec, truth=True)))


COOPERATIVE = mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2)
"""A tile two consumer warpgroups compute, on one multiprocessor."""

PAIRED = dataclasses.replace(COOPERATIVE, cluster=2)
"""`COOPERATIVE` on a pair of multiprocessors a cluster, the forward's two row tiles of a group sharing its
matrix's blocks (the interpreter runs the pair's two programs, each loading the shared block whole)."""


@runs("sm90")
@pytest.mark.parametrize("forward", [COOPERATIVE, PAIRED], ids=["one multiprocessor", "a pair"])
@pytest.mark.parametrize("case", ["norm and bias", "gated prologue", "cast prologue", "row scaled"])
def test_a_cooperative_tile_holds_check_1_with_its_hooks(case, forward):
    """The Hopper launches pinned to a cooperative tile, the forward's alone or a pair's (whose second row
    tile of a pair past its group's rows stores none of them), interpreted: each consumer rewrites and
    multiplies its half of the stage's rows and computes their epilogue, as exact as the plain program
    from float64 truth."""
    spec = CASES[case]
    pinned = {grouped_matmul.MosaicGmm: forward, grouped_matmul.MosaicTgmm: COOPERATIVE}
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned):
        dew = jax.jit(_drop(_dew(spec, interpret_on_cpu=True)))
        assert_as_exact(dew, jax.jit(_drop(_plain(spec))), _draw, f"{case} at {forward}",
                        seeds=SEEDS, truth_fn=_drop(_plain(spec, truth=True)))


def _gradient_draw(case: Case, base=None):
    """`base`'s draw (`_draw` by default) of a case's inputs with a cotangent for each of its outputs, in the
    output's dtype, so each arm and the float64 truth pull back the same values."""
    def draw(seed):
        rng = np.random.default_rng(seed + 1000)
        cotangents = tuple(jnp.asarray(rng.normal(size=(ROWS, WIDTH)), dtype) for dtype in case.out_dtypes)
        return (*(base or _draw)(seed)[:7], cotangents)
    return draw


def _gradients(run):
    """`run`'s outputs and every input's gradient under the drawn cotangents (`with_gradients`)."""
    return with_gradients(lambda *operands: tuple(run(*operands)))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90", "v5e"))
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_the_templates_gradients_hold_check_1(generation_name, case):
    """Every operand's cotangent through the template's own reverse mode, against JAX's derivative of
    the plain program, from float64 truth. The Mosaic interpreters run ordered callbacks, which a jitted
    `linear_call` cannot carry, so their gradients run eagerly."""
    spec = CASES[case]
    with generation.assumed(generation_name):
        dew = _gradients(_dew(spec, interpret_on_cpu=True))
        # The sm90 interpreter runs eagerly; the gate compiles the other generations' arm itself.
        dew = as_given(dew) if generation_name == "sm90" else dew
        assert_as_exact(dew, _gradients(_plain(spec)), _gradient_draw(spec),
                        f"{case} gradients on {generation_name}", seeds=SEEDS,
                        truth_fn=_gradients(_plain(spec, truth=True)))



@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_every_variant_of_the_mosaic_launches_holds_check_1(generation_name):
    """Each config an opt-in tuning run may time (`KernelOp.variants`: the derived tile and its legal
    neighbours) of the Mosaic GPU forward, and on warpgroup MMA of its kernel gradient, pinned in turn
    and interpreted: as exact as the plain program from float64 truth, so the tuner times only configs
    proven here."""
    rows, depth, width = 136, 128, 256
    sizes, split = jnp.asarray([70, 66], jnp.int32), 70

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(2, depth, width)) / 8, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16))

    def forward(x, w, dy):
        del dy
        return template.gemm(x, w, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def plain_forward(x, w, dy, wide=jnp.float32, out=jnp.bfloat16):
        del dy
        w = w.astype(wide)
        return jnp.concatenate([x[:split].astype(wide) @ w[0], x[split:].astype(wide) @ w[1]]).astype(out)

    def gradient(x, w, dy):
        del w
        return template.gemm_t(x, dy, sizes, out_dtype=jnp.float32, interpret_on_cpu=True)

    def plain_gradient(x, w, dy, wide=jnp.float32, out=jnp.float32):
        del w
        x, dy = x.astype(wide), dy.astype(wide)
        return jnp.stack([x[:split].T @ dy[:split], x[split:].T @ dy[split:]]).astype(out)

    calls = [(grouped_matmul.MosaicGmm, forward, plain_forward)]
    if generation.gpu_target(generation_name).wgmma:
        calls.append((grouped_matmul.MosaicTgmm, gradient, plain_gradient))
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic_gpu"):
        for kernel, dew, plain in calls:
            configs = each_config(kernel, dew, *draw(0))
            assert len(configs) > 1, kernel.__name__
            for config in configs:
                with tuning.pinned({kernel: config}):
                    assert_as_exact(dew, jax.jit(plain), draw, f"{kernel.__name__} at {config}", seeds=SEEDS,
                                    truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_stacked_matrix_is_read_in_place_at_a_layer(generation_name):
    """A layer of a stacked `[layers, groups, depth, width]` matrix picked by an index (`gemm`'s `layer`):
    as exact as the plain program's product with that layer's matrices from float64 truth, the stack's
    gradient included (zero at every other layer). The gradients run eagerly, as the Mosaic interpreters'
    ordered callbacks need."""
    layers, groups, rows, depth, width = 3, 2, 136, 64, 128
    sizes, layer, split = jnp.asarray([70, 66], jnp.int32), jnp.asarray(1, jnp.int32), 70

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(layers, groups, depth, width)) / 8, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16))

    def dew(x, stack):
        return template.gemm(x, stack, sizes, layer=layer, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def plain(x, stack, wide=jnp.float32, out=jnp.bfloat16):
        matrices = stack[1].astype(wide)
        return jnp.concatenate([x[:split].astype(wide) @ matrices[0],
                                x[split:].astype(wide) @ matrices[1]]).astype(out)

    with generation.assumed(generation_name):
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), draw,
                        f"stacked layer on {generation_name}", seeds=SEEDS,
                        truth_fn=with_gradients(functools.partial(plain, wide=jnp.float64, out=jnp.float64)))


@runs("sm80")
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


@runs("sm80")
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


@runs("sm80")
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


@pytest.mark.parametrize("generation_name", generations("sm80"))
@pytest.mark.parametrize("sizes", [[3], None], ids=["ragged", "dense"])
def test_a_prologue_never_sees_the_depths_padding(generation_name, sizes):
    """A depth no tile divides is padded with zeros, which `reciprocal` turns to infinities: the kernel,
    ragged or dense, zeroes them again before the product, so every output is the sum of 17 ones."""
    x, w = jnp.ones((3, 17), jnp.bfloat16), jnp.ones((1, 17, 16))
    w = w if sizes else w[0]
    with generation.assumed(generation_name):
        out = jax.jit(lambda x, w: template.gemm(x, w, None if sizes is None else jnp.asarray(sizes),
                                                 prologue=reciprocal, out_dtype=jnp.float32,
                                                 interpret_on_cpu=True))(x, w)
    np.testing.assert_array_equal(np.asarray(out), np.full((3, 16), 17.0))


@runs("sm80")
@pytest.mark.parametrize("transpose_rhs", [False, True])
def test_a_dense_product_whose_extents_no_tile_divides_holds_check_1(transpose_rhs):
    """A dense product of 65 rows, 40 deep and 100 wide, which no tile divides, through the plain kernel's
    padded rows, depth and width, with a norm prologue and a bias epilogue: the output and its gradients
    hold check 1 against JAX's derivative of the plain program, the matrix stored either way."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(65, 40)), jnp.bfloat16)
        w = jnp.asarray(rng.normal(size=(100, 40) if transpose_rhs else (40, 100)) / 7, jnp.bfloat16)
        ms = jnp.asarray(rng.uniform(0.5, 2, size=(65,)), jnp.float32)
        nw = jnp.asarray(rng.normal(size=(40,)), jnp.float32)
        b = jnp.asarray(rng.normal(size=(100,)), jnp.float32)
        return x, w, ms, nw, b, jnp.asarray(rng.normal(size=(65, 100)), jnp.bfloat16)

    def dew(x, w, ms, nw, b):
        return template.gemm(x, w, None, prologue=normed,
                             prologue_operands=((ms, ("rows",)), (nw, ("depth",))), epilogue=biased,
                             epilogue_operands=((b, ("columns",)),), out_dtype=jnp.bfloat16,
                             transpose_rhs=transpose_rhs, interpret_on_cpu=True)

    def plain(x, w, ms, nw, b, wide=jnp.float32, out=jnp.bfloat16):
        # A 16-bit output: an fp32 one's cotangent makes the kernel gradient an fp32 product, XLA's.
        w = w.T if transpose_rhs else w
        return (jnp.dot(normed(x, ms, nw).astype(wide), w.astype(wide)) + b[None, :].astype(wide)).astype(out)

    def forward(run):
        return lambda x, w, ms, nw, b, cotangent: run(x, w, ms, nw, b)

    # Named: a dense cell runs XLA's dot unnamed (`grouped_matmul.DECLARED`).
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="triton"):
        truth = functools.partial(plain, wide=jnp.float64, out=jnp.float64)
        assert_as_exact(jax.jit(forward(dew)), jax.jit(forward(plain)), draw, "dense 65x40x100", seeds=SEEDS,
                        truth_fn=forward(truth))
        assert_as_exact(jax.jit(with_gradients(dew)), jax.jit(with_gradients(plain)), draw,
                        "dense 65x40x100 gradients", seeds=SEEDS, truth_fn=with_gradients(truth))


def test_a_named_kernel_runs_the_calls_derivative_too():
    """A grouped product traced on an A100 within `generation.chosen(grouped_matmul='xla')`, its pullback
    run after the block closes, as MoE's implementation='xla' scopes only its own call: every grouped
    dispatch its gradient reports runs XLA's program, the kernel the call named (train 243's review), not
    the launches an unnamed call would select."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(ROUTED, DEPTH)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(len(SIZES), DEPTH, WIDTH)), jnp.bfloat16)
    with generation.assumed("sm80"), generation.routes() as found:
        with generation.chosen(grouped_matmul="xla"):
            y, pull = jax.vjp(lambda x, w: template.gemm(x, w, jnp.asarray(SIZES), out_dtype=jnp.bfloat16), x, w)
        pull(jnp.ones_like(y))
    assert {route.kernel for route in found if route.choice == "grouped_matmul"} == {"xla"}


def _dense_draw(rows: int, depth: int, width: int, *, cotangent: bool = False):
    """A draw of a dense product's bf16 operands, `[rows, depth]` and `[depth, width]`, and with
    `cotangent` a bf16 cotangent of its bf16 output."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        drawn = (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                 jnp.asarray(rng.normal(size=(depth, width)), jnp.bfloat16))
        return (*drawn, jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16)) if cotangent else drawn
    return draw


@runs("sm90")
def test_a_dense_gradient_takes_a_padded_tile_where_no_dividing_one_fits():
    """A dense kernel gradient at depth 96 on Hopper through Triton's launches (`generation.chosen`): the
    tiles dividing 96 have 32 output rows, which Hopper's warpgroup MMA refuses, so a 64-row tile runs
    over the padded depth, and the gradients hold check 1 against JAX's derivative of the plain product."""
    draw = _dense_draw(256, 96, 128, cotangent=True)

    def dew(x, w):
        return template.gemm(x, w, None, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def plain(x, w, wide=jnp.float32, out=jnp.bfloat16):
        # A 16-bit output: an fp32 one's cotangent makes the kernel gradient an fp32 product, XLA's.
        return jnp.dot(x.astype(wide), w.astype(wide)).astype(out)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="triton"):
        assert_as_exact(jax.jit(with_gradients(dew)), jax.jit(with_gradients(plain)), draw, "dense depth 96",
                        seeds=SEEDS,
                        truth_fn=with_gradients(functools.partial(plain, wide=jnp.float64, out=jnp.float64)))


@pytest.mark.gpu(reason="checks Triton's MMA accumulation on a GPU")
@pytest.mark.skipif(not generation.triton_runs(), reason="needs a GPU the Pallas kernels compile for")
def test_a_dense_products_exact_sums_survive_one_row_tile():
    """At the highest precision, a dense fp32 product of one row tile and many depth steps (its first grid
    axis one program, where a program id is a constant) through Triton's launch, named (a dense cell runs
    XLA's dot unnamed, `grouped_matmul.DECLARED`), sums each step's product by an IEEE add: its error
    holds check 1 against XLA's IEEE sums, which a tensor core's truncating accumulation does not."""
    draw = _dense_draw(64, 4096, 128)

    def dew(x, w):
        return template.gemm(x, w, None, out_dtype=jnp.float32, precision="highest")

    def xla(x, w):
        return jnp.dot(x, w, precision="highest", preferred_element_type=jnp.float32)

    def truth(x, w):
        return jnp.dot(x.astype(jnp.float64), w.astype(jnp.float64))

    with generation.chosen(grouped_matmul="triton"):
        assert_as_exact(jax.jit(dew), jax.jit(xla), draw, "dense exact sums, one row tile", seeds=SEEDS,
                        truth_fn=truth)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_side_operands_rows_may_be_any_of_its_dimensions(generation_name):
    """A side operand's rows as its first dimension or its second, whatever rows are a multiple of, through
    Triton's launches (`generation.chosen`), the one implementation that takes both layouts (Mosaic's
    lowering has no transpose): the forward product and the kernel gradient on par with the plain
    program's over the draws (check 1), truth the plain program's in float64."""
    sizes = jnp.asarray(SIDE_SIZES)
    for hook, layout, transposed in SIDE_LAYOUTS:
        def dew(x, w, dy, side, hook=hook, layout=layout, transposed=transposed):
            options = {"prologue": hook, "prologue_operands": ((side.T if transposed else side, layout),),
                       "out_dtype": jnp.float32, "interpret_on_cpu": True}
            return template.gemm(x, w, sizes, **options), template.gemm_t(x, dy, sizes, **options)

        with generation.assumed(generation_name), generation.chosen(grouped_matmul="triton"):
            assert_as_exact(jax.jit(dew), jax.jit(_side_plain()), _side_draw(jnp.float32),
                            f"side {layout} on {generation_name}", seeds=SEEDS,
                            truth_fn=_side_plain(truth=True))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_broadcast_dimension_is_the_squeezed_operands(generation_name):
    """A `[rows, 1]` mean square and a `[1, width]` bias with a None axis, and their vectors, through the one
    implementation that takes both (Triton's: Mosaic's refuses a side operand given back with a dimension
    of one): each on par with its plain program over the draws (check 1), from float64 truth."""
    for case in ("broadcast axes", "norm and bias"):
        spec = CASES[case]
        with generation.assumed(generation_name), generation.chosen(grouped_matmul="triton"):
            assert_as_exact(jax.jit(_drop(_dew(spec, interpret_on_cpu=True))), jax.jit(_drop(_plain(spec))),
                            _draw, f"{case} through Triton on {generation_name}",
                            seeds=SEEDS, truth_fn=_drop(_plain(spec, truth=True)))


def test_a_side_operand_names_one_launch_axis_per_dimension():
    """Axes the hook cannot take are refused before any launch, naming the operand and its axes."""
    x, w = jnp.ones((8, DEPTH), jnp.bfloat16), jnp.ones((1, DEPTH, WIDTH))
    for operand, axes in ((jnp.ones((8,)), ("columns",)), (jnp.ones((8, 2)), ("rows", None)),
                          (jnp.ones((1,)), (None,))):
        with pytest.raises(ValueError):
            template.gemm(x, w, jnp.asarray([8]), prologue=normed, prologue_operands=((operand, axes),),
                          out_dtype=jnp.float32)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
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


@runs("sm80")
def test_forward_mode_through_the_kernels_gives_the_plain_programs_tangent():
    """`jax.jvp` of the template through its kernels (`generation.linear_tangent`) along drawn tangents of
    the rows and the matrices: its value and tangent in the output's dtype, and on par with the plain
    program's over the draws (check 1), truth the plain program's in float64."""
    spec = CASES["norm and bias"]

    def draw(seed):
        x, w, ms, nw, b, u, s, _ = _draw(seed)
        rng = np.random.default_rng(seed + 2000)
        along = (jnp.asarray(rng.normal(size=x.shape), x.dtype),
                 jnp.asarray(rng.normal(size=w.shape) / np.sqrt(DEPTH), w.dtype))
        return x, w, *along, ms, nw, b, u, s

    def forward(run):
        def jvp(x, w, dx, dw, *rest):
            return jax.jvp(lambda x, w: run(x, w, *rest)[0], (x, w), (dx, dw))
        return jvp

    with generation.assumed("sm80"):
        value, tangent = jax.jit(forward(_dew(spec, interpret_on_cpu=True)))(*draw(0))
        assert value.dtype == tangent.dtype == jnp.bfloat16
        assert_as_exact(jax.jit(forward(_dew(spec, interpret_on_cpu=True))), jax.jit(forward(_plain(spec))),
                        draw, "forward mode through the kernels", seeds=SEEDS,
                        truth_fn=forward(_plain(spec, truth=True)))


def bias_cast(prod, b, r, dtype=jnp.bfloat16):
    return (prod + b[None, :]).astype(dtype)


def silu_residual(prod, b, r, dtype=jnp.bfloat16):
    return (jax.nn.silu(prod * 2) + _wide(r)).astype(dtype)


def two_consumers(prod, b, r, dtype=jnp.bfloat16):
    return prod.astype(dtype), prod + b[None, :]


def positional(prod, b, r, dtype=jnp.bfloat16):
    return (prod + jax.lax.broadcasted_iota(prod.dtype, prod.shape, 1)).astype(dtype)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
@pytest.mark.parametrize("consumer", [bias_cast, silu_residual, two_consumers, positional],
                         ids=lambda consumer: consumer.__name__)
def test_the_fused_epilogue_keeps_the_programs_value(generation_name, consumer):
    """Under `fuser.fuse`, a product's consumers run in the epilogue (or after it in XLA, where they read
    the tile's position) with the unfused program's value on every row: as exact as it, from float64."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, _, b, u, *_ = _draw(seed)
        return x, w, b, u[:, :WIDTH]

    def fusing(x, w, b, r):
        return consumer(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), b, r)

    def unfused(x, w, b, r):
        return consumer(template.gemm(x, w, sizes, out_dtype=jnp.float32), b, r)

    def truth(x, w, b, r):
        matrix = w.astype(jnp.bfloat16).astype(x.dtype)
        product = jax.lax.ragged_dot(x, matrix, sizes, preferred_element_type=x.dtype)
        return consumer(product, b, r, dtype=x.dtype)

    with generation.assumed(generation_name):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, f"fused {consumer.__name__}",
                        seeds=SEEDS, truth_fn=truth)


def rms_norm(x, weight, epsilon=1e-6):
    """An RMS norm of the residual stream as a model writes it: statistics at least fp32 (float64 for the
    float64 truth), one rounding."""
    wide = _wide(x)
    mean_square = jnp.mean(wide * wide, axis=-1, keepdims=True)
    return (wide * jax.lax.rsqrt(mean_square + epsilon) * weight).astype(x.dtype)


def _normed_truth(sizes, x, w, weight):
    """The float64 truth of a normed operand's grouped product."""
    matrix = w.astype(jnp.bfloat16).astype(x.dtype)
    return jax.lax.ragged_dot(rms_norm(x, weight), matrix, sizes, preferred_element_type=x.dtype)


def dew_rms_norm(x, weight):
    """Dew's own RMS norm (its XLA body, checkpointed), as a model's block calls it."""
    from dew.nn.attention import rms_normalized
    return rms_normalized(x, weight, 1e-6, x.dtype, scale_offset=False, scale_after_cast=False,
                          fp32_statistics=True, implementation="xla")


def relu(x, weight):
    """A producer with a custom derivative (ReLU's, zero at zero) and a depth weight."""
    return (jax.nn.relu(_wide(x)) * weight).astype(x.dtype)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_normed_operand_runs_in_the_prologue_with_the_programs_value(generation_name):
    """Under `fuser.fuse`, an RMS norm producing the product's left operand runs in the kernels' prologue
    on each loaded tile, its row statistic in XLA before the product, with the unfused program's value: as
    exact as it, from float64."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, nw, *_ = _draw(seed)
        return x, w, nw

    def fusing(x, w, weight):
        return fused.fusible_gemm(rms_norm(x, weight), w, sizes, interpret_on_cpu=True).astype(jnp.bfloat16)

    def unfused(x, w, weight):
        return template.gemm(rms_norm(x, weight), w, sizes, out_dtype=jnp.bfloat16)


    with generation.assumed(generation_name):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "normed operand", seeds=SEEDS,
                        truth_fn=functools.partial(_normed_truth, sizes))


@lowers("sm90")
@pytest.mark.parametrize("producer", [rms_norm, dew_rms_norm], ids=lambda producer: producer.__name__)
def test_a_fused_norm_lowers_for_an_h100(producer):
    """An RMS norm fused into the Mosaic GPU forward's prologue lowers for an H100 on the CPU: its row
    statistic, a column the template squeezed, comes back to its tile by a broadcast, which Mosaic GPU's
    layout inference lays out (a reshape it does not; FUSEDNORM-h100)."""
    # A Qwen3 MoE projection's extents, where the cost model ranks Mosaic GPU's forward first (at the draws'
    # few rows a Triton tile may rank first); lowered, never run.
    rows, depth, width = 8192, 2048, 1536
    sizes = jnp.asarray((0, rows // 2, 1, rows // 2 - 101, 100, 0))
    x, w, nw = (jax.ShapeDtypeStruct((rows, depth), jnp.bfloat16),
                jax.ShapeDtypeStruct((len(sizes), depth, width), jnp.float32),
                jax.ShapeDtypeStruct((depth,), jnp.float32))

    def fusing(x, w, weight):
        return fused.fusible_gemm(producer(x, weight), w, sizes).astype(jnp.bfloat16)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(fuser.fuse(fusing)).trace(x, w, nw).lower(lowering_platforms=("cuda",))


def _three_gemms(sizes):
    """A fused producer's product three ways: under `fuser.fuse`, unfused, and its float64 truth."""
    def fused_gemm(operand, w):
        return fused.fusible_gemm(operand, w, sizes, interpret_on_cpu=True)

    def plain_gemm(operand, w):
        return template.gemm(operand, w, sizes, out_dtype=jnp.float32)

    def truth_gemm(operand, w):
        # The matrix's value rounded as the program rounds it; its gradient not (straight through).
        matrix = (w + jax.lax.stop_gradient(w.astype(jnp.bfloat16).astype(w.dtype) - w)).astype(operand.dtype)
        return jax.lax.ragged_dot(operand, matrix, sizes, preferred_element_type=operand.dtype)

    return fused_gemm, plain_gemm, truth_gemm


@runs("sm80")
@pytest.mark.parametrize("producer", [rms_norm, dew_rms_norm, relu], ids=lambda producer: producer.__name__)
def test_a_fused_producers_gradients_are_its_own(producer):
    """Every input's gradient through a producer fused into the prologue is the unfused program's: the
    operand's cotangent goes back through the producer's own VJP, so a norm's two paths from its input
    join where the producer joins them and a custom derivative (ReLU's at zero) stays its own; Dew's
    checkpointed norm fuses too (the route says so)."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, nw, *_ = _draw(seed)
        x = x.at[:4].set(0)  # a ReLU's zero, where its custom derivative differs from max(x, 0)'s
        cotangent = np.random.default_rng(seed + 1000).normal(size=(ROWS, WIDTH))
        return x, w, nw, jnp.asarray(cotangent, jnp.float32)

    def loss(gemm, transform=lambda value: value):
        def value(x, w, weight):
            out = gemm(producer(x, weight), w)
            return _wide(out)  # fp32, or the truth's float64, as the drawn cotangent is widened
        return with_gradients(transform(value))

    fused_gemm, plain_gemm, truth_gemm = _three_gemms(sizes)

    with generation.assumed("sm80"):
        assert_as_exact(jax.jit(loss(fused_gemm, fuser.fuse)), jax.jit(loss(plain_gemm)), draw,
                        f"{producer.__name__} gradients", seeds=SEEDS, truth_fn=loss(truth_gemm))


@runs("sm80")
@pytest.mark.parametrize("composition", ["jvp", "radial_jvp", "jvp_of_grad"])
def test_a_fused_producers_forward_and_higher_derivatives_are_its_own(composition):
    """Forward mode and a higher order through a norm fused into the prologue differentiate the whole
    producer, its row statistic included, as exact as the unfused program's from float64: along the
    radial tangent (the operand itself), where an RMS norm is constant, the product's tangent is the
    epsilon's alone, not the normalized operand's projection a held statistic would give."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, nw, *_ = _draw(seed)
        rng = np.random.default_rng(seed + 2000)
        direction = x if composition == "radial_jvp" else jnp.asarray(rng.normal(size=x.shape), jnp.float32)
        cotangent = jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.float32)
        return x, w, nw, direction, cotangent

    def derivative(gemm, transform=lambda value: value):
        program = transform(lambda x, w, weight: _wide(gemm(rms_norm(x, weight), w)))

        def run(x, w, weight, direction, cotangent):
            def scored(x):
                out = program(x, w, weight)
                return jnp.sum(out * cotangent.astype(out.dtype))

            along = (direction.astype(x.dtype),)
            if composition == "jvp_of_grad":
                return jax.jvp(jax.grad(scored), (x,), along)[1]
            return jax.jvp(lambda x: program(x, w, weight), (x,), along)[1]
        return run

    fused_gemm, plain_gemm, truth_gemm = _three_gemms(sizes)

    with generation.assumed("sm80"):
        assert_as_exact(jax.jit(derivative(fused_gemm, fuser.fuse)), jax.jit(derivative(plain_gemm)), draw,
                        f"normed operand {composition}", seeds=SEEDS, truth_fn=derivative(truth_gemm))


@runs("sm80")
def test_both_fused_stages_chosen_through_xla_are_as_exact_as_the_unfused_program():
    """`generation.chosen(gemm_epilogue='xla', gemm_prologue='xla')`: the fused program's value, as exact
    as the unfused program from float64."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, nw, b, *_ = _draw(seed)
        return x, w, nw, b

    def fusing(x, w, weight, b):
        return bias_cast(fused.fusible_gemm(rms_norm(x, weight), w, sizes, interpret_on_cpu=True), b, None)

    def unfused(x, w, weight, b):
        return bias_cast(template.gemm(rms_norm(x, weight), w, sizes, out_dtype=jnp.float32), b, None)

    def truth(x, w, weight, b):
        return _normed_truth(sizes, x, w, weight) + b[None, :]

    with generation.assumed("sm80"), generation.chosen(gemm_epilogue="xla", gemm_prologue="xla"):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "both stages in XLA",
                        seeds=SEEDS, truth_fn=truth)


@runs("sm80")
def test_a_one_row_normed_operand_fuses():
    """A one-row operand's row statistic keeps its rows axis: the product of one normed row, as exact as
    the unfused program from float64."""
    sizes = jnp.asarray([1] + [0] * (len(SIZES) - 1))

    def draw(seed):
        x, w, _, nw, *_ = _draw(seed)
        return x[:1], w, nw

    def fusing(x, w, weight):
        return fused.fusible_gemm(rms_norm(x, weight), w, sizes, interpret_on_cpu=True)

    def unfused(x, w, weight):
        return template.gemm(rms_norm(x, weight), w, sizes, out_dtype=jnp.float32)


    with generation.assumed("sm80"):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "one normed row", seeds=SEEDS,
                        truth_fn=functools.partial(_normed_truth, sizes))


@lowers("sm90")
@pytest.mark.parametrize("launch", [grouped_matmul.MosaicGmm(), grouped_matmul.MosaicTgmm()],
                         ids=lambda launch: type(launch).__name__)
def test_the_hooked_mosaic_launches_lower_for_an_h100(launch):
    """Each Mosaic GPU launch with a prologue (and the forward's epilogue) lowers for an H100 on the CPU
    at the tile it derives, rows no multiple of 8."""
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
        [config] = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
            lowering_platforms=("cuda",))



def _on_a_v5e():
    """An assumed v5e and a mesh of one of its devices, abstract, which Pallas's TPU lowering reads its
    target from on any host."""
    device = AbstractDevice(device_kind=generation.gpu_target("v5e").device_kind, num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("data",), axis_types=(AxisType.Auto,), abstract_device=device)
    return generation.assumed("v5e"), jax.sharding.use_abstract_mesh(mesh)


@pytest.mark.parametrize("side_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("launch", [grouped_matmul.MosaicTpuGmm(), grouped_matmul.MosaicTpuTgmm()],
                         ids=lambda launch: type(launch).__name__)
def test_the_hooked_tpu_launches_lower_for_a_v5e(launch, side_dtype):
    """Each TPU launch with a prologue (and the forward's epilogue) lowers for a TPU on the CPU at the
    block it derives and at the least block a pin may name, rows no multiple of a block: a 16-bit side
    vector's blocks hold at least the 256 elements Mosaic TPU lays out, so its least block is no legal
    pin."""
    rows, depth, width, groups = 2044, 1536, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sides = functools.partial(jax.ShapeDtypeStruct, dtype=side_dtype)
    sizes = jax.ShapeDtypeStruct((groups,), jnp.int32)
    prologue = (normed, (sides((rows,)), sides((depth,))), (("rows",), ("depth",)))
    if isinstance(launch, grouped_matmul.MosaicTpuTgmm):
        args = (bf16((rows, depth)), bf16((rows, width)), sizes, prologue[1])
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2])
        options = {"out_dtype": "float32", "hooks": hooks, "interpret": False}
    else:
        args = (bf16((rows, depth)), bf16((groups, depth, width)), sizes, prologue[1], (sides((width,)),))
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2], biased, (("columns",),))
        options = {"out_dtype": "bfloat16", "trans_rhs": False, "hooks": hooks, "interpret": False}
    assumed, mesh = _on_a_v5e()
    with assumed, mesh:
        ba, target = launch.bind(*args, **options), generation.gpu_target()
        least = mosaic_tpu.Blocks(mosaic_tpu.LANES, mosaic_tpu.LANES, mosaic_tpu.LANES)
        assert launch.legal(ba, least, target) == (jnp.dtype(side_dtype).itemsize == 4)
        for config in {launch.derived(ba, target), *([least] if launch.legal(ba, least, target) else [])}:
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("tpu",))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_cuda_call_naming_xla_runs_its_program(generation_name):
    """A bf16 product the GPU's launches take, with `generation.chosen` naming XLA, lowers for the GPU
    (the audit's forced-xla route, ledger 1702 (a))."""
    x = jax.ShapeDtypeStruct((256, 128), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((2, 128, 256), jnp.bfloat16)
    sizes = jnp.asarray([128, 128], jnp.int32)
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="xla"):
        jax.jit(lambda x, w: template.gemm(x, w, sizes, out_dtype=jnp.bfloat16)).trace(x, w).lower(
            lowering_platforms=("cuda",))


@runs("v5e")
@pytest.mark.parametrize("product", ["gemm", "gemm_t"])
def test_a_tpu_lowers_xlas_program_for_a_call_no_launch_takes(product):
    """A product of widths no TPU block divides lowers for a TPU, its kernel branches traced on the way and
    none raising. A CPU process's: a GPU's interprets its CPU branch with Triton's launch whatever the
    target (`grouped_matmul.launched`)."""
    x = jax.ShapeDtypeStruct((32, 64), jnp.bfloat16)
    other = jax.ShapeDtypeStruct((1, 64, 64) if product == "gemm" else (32, 64), jnp.bfloat16)
    sizes = jnp.asarray([32], jnp.int32)
    if product == "gemm":
        def run(x, other):
            return template.gemm(x, other, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True)
    else:
        def run(x, other):
            return template.gemm_t(x, other, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True)
    assumed, mesh = _on_a_v5e()
    with assumed, mesh:
        jax.jit(run).trace(x, other).lower(lowering_platforms=("tpu",))


NARROW_GRADIENT_CASES = ("norm and bias", "gated prologue", "cast prologue", "broadcast axes",
                         "dense norm and bias")
"""The gradient cases whose cotangents are 16-bit, so each product of their reverse mode multiplies
16-bit operands."""


@runs("v5e")
@pytest.mark.parametrize("case", NARROW_GRADIENT_CASES)
def test_the_tpu_launches_gradients_hold_check_1(case):
    """Every operand's cotangent of a bf16 matrix at the default precision, where no product adds step by
    step, so the forward and both gradients run the TPU's launches, named and interpreted: as exact as
    JAX's derivative of the plain program from float64 truth."""
    spec = CASES[case]

    def drawn(seed):
        x, w, *rest = _draw(seed)
        return (x, w.astype(jnp.bfloat16), *rest)

    # Named, as the TPU's launches are tested here.
    with (generation.assumed("v5e"), jax.default_matmul_precision("default"),
          generation.chosen(grouped_matmul="mosaic_tpu")):
        assert_as_exact(_gradients(_dew(spec, interpret_on_cpu=True)), jax.jit(_gradients(_plain(spec))),
                        _gradient_draw(spec, drawn), f"{case} gradients on the TPU's launches", seeds=SEEDS,
                        truth_fn=_gradients(_plain(spec, truth=True)))

@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("launch,trans_rhs", [(grouped_matmul.Gmm(), False), (grouped_matmul.Gmm(), True),
                                              (grouped_matmul.Tgmm(), None)],
                         ids=["Gmm", "Gmm-transposed", "Tgmm"])
def test_the_hooked_dense_launches_lower_for_cuda(launch, trans_rhs, generation_name):
    """Each Triton launch of a dense call, with a prologue (and the forward's epilogue) and at the highest
    precision's exact sums, lowers for CUDA on the CPU at a fitting tile of every block shape, with rows,
    depth and width no tile divides."""
    rows, depth, width = 1021, 200, 300
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    prologue = (normed, (f32((rows,)), f32((depth,))), (("rows",), ("depth",)))
    if isinstance(launch, grouped_matmul.Tgmm):
        args = (bf16((rows, depth)), bf16((rows, width)), sizes, prologue[1])
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2])
        options = {"out_dtype": "float32", "hooks": hooks, "interpret": False}
    else:
        matrix = bf16((1, width, depth) if trans_rhs else (1, depth, width))
        args = (bf16((rows, depth)), matrix, sizes, prologue[1], (f32((width,)),))
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2], biased, (("columns",),),
                                     out_dtypes=("float32",))
        options = {"out_dtype": "float32", "trans_rhs": trans_rhs, "hooks": hooks, "interpret": False}
    options |= {"dense": True, "exact_sums": True}
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        fits = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        covering = set({(c.block_m, c.block_n, c.block_k): c for c in reversed(fits)}.values())
        assert covering
        for config in sorted(covering, key=repr):
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


@INTERPRETS
def test_the_mosaic_kernels_load_a_side_operands_rows_from_any_dimension():
    """Interpreted, Mosaic GPU's kernels pad and slice a side operand along its rows wherever they sit: with
    a `[rows, depth]` side and a `[depth, rows]` one, the forward product and the kernel gradient on par
    with the plain program's over the draws (check 1), truth the plain program's in float64 (eagerly: the
    interpreter's ordered callbacks)."""
    sizes = jnp.asarray(SIDE_SIZES)
    common = {"tiles": mosaic_gpu.Tiles(64, 64, 64, 2, 1), "out_dtype": jnp.float32, "multiprocessors": 1,
              "interpret": grouped_matmul.INTERPRETED}
    for hook, layout, transposed in SIDE_LAYOUTS:
        def dew(x, w, dy, side, hook=hook, layout=layout, transposed=transposed):
            options = {"prologue": hook, "prologue_operands": (side.T if transposed else side,),
                       "prologue_axes": (layout,), **common}
            return (mosaic_gpu.gmm(x, w, sizes, transpose_rhs=False, **options),
                    mosaic_gpu.tgmm(x, dy, sizes, **options))

        assert_as_exact(dew, jax.jit(_side_plain()), _side_draw(jnp.bfloat16), f"Mosaic side {layout}",
                        seeds=SEEDS, truth_fn=_side_plain(truth=True))


def _gradient_rows_draw(seed):
    """Rows and a cotangent for a kernel gradient, both bf16."""
    rng = np.random.default_rng(seed + 4)
    return (jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16),
            jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.bfloat16))


@SUMS_IN_IEEE
@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_exact_sums_add_each_steps_product_as_the_plain_sum_does(generation_name):
    """At the highest precision an fp32 kernel gradient adds each step's product apart (`exact_sums`):
    through each launch at one tile, grouped and dense, it and the default path are each on par with the
    plain kernel gradient over the draws (check 1), truth the plain kernel gradient in float64."""
    # One tile both ways, a step of EXACT_STEP rows.
    tiles = {grouped_matmul.Tgmm: grouped_matmul.Tile(64, 128, 64, 4, 2),
             grouped_matmul.MosaicTgmm: mosaic_gpu.Tiles(64, 128, 64, 2, 1)}
    launches = ("triton", "mosaic_gpu") if generation_name == "sm90" else ("triton",)
    for launch, group_sizes, precision in itertools.product(launches, (SIZES, None),
                                                            ("highest", "default")):
        def dew(x, dy, group_sizes=group_sizes, precision=precision):
            return template.gemm_t(x, dy, None if group_sizes is None else jnp.asarray(group_sizes),
                                   out_dtype=jnp.float32, precision=precision, interpret_on_cpu=True)

        def plain(x, dy, group_sizes=group_sizes):
            return _kernel_gradient(x, dy, group_sizes)

        with (generation.assumed(generation_name), tuning.pinned(tiles),
              generation.chosen(grouped_matmul=launch)):
            assert_as_exact(jax.jit(dew), jax.jit(plain), _gradient_rows_draw,
                            f"{precision} sums through {launch}, {'grouped' if group_sizes else 'dense'}",
                            seeds=SEEDS, truth_fn=plain)


@runs("sm90")
@pytest.mark.parametrize("case", ["norm and bias"])
def test_a_cooperative_tiles_gradients_hold_check_1(case):
    """`test_a_cooperative_tile_holds_check_1_with_its_hooks`'s reverse mode: the input gradient through the
    cooperative transposed launch, the kernel gradient through the cooperative kernel gradient with the
    prologue each consumer applies to its depth half, against JAX's derivative of the plain program
    (eagerly: `test_the_templates_gradients_hold_check_1`)."""
    spec = CASES[case]
    pinned = {grouped_matmul.MosaicGmm: COOPERATIVE, grouped_matmul.MosaicTgmm: COOPERATIVE}
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned):
        assert_as_exact(as_given(_gradients(_dew(spec, interpret_on_cpu=True))), _gradients(_plain(spec)),
                        _gradient_draw(spec), f"{case} cooperative gradients", seeds=SEEDS,
                        truth_fn=_gradients(_plain(spec, truth=True)))


@SUMS_IN_IEEE
def test_a_cooperative_tiles_exact_sums_add_as_its_plain_sum_does():
    """`test_exact_sums_add_each_steps_product_as_the_plain_sum_does` at a cooperative tile: each consumer's
    running fp32 sum, the pipeline's carry, at the highest precision and the default, each on par with the
    plain kernel gradient over the draws (check 1), from float64 truth."""
    def plain(x, dy):
        return _kernel_gradient(x, dy, SIZES)

    for precision in ("highest", "default"):
        def dew(x, dy, precision=precision):
            return template.gemm_t(x, dy, jnp.asarray(SIZES), out_dtype=jnp.float32, precision=precision,
                                   interpret_on_cpu=True)

        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"),
              tuning.pinned({grouped_matmul.MosaicTgmm: COOPERATIVE})):
            assert_as_exact(jax.jit(dew), jax.jit(plain), _gradient_rows_draw,
                            f"cooperative {precision} sums", seeds=SEEDS, truth_fn=plain)


def test_a_tile_states_a_schedule_the_kernels_run():
    """A pair of multiprocessors needs two warpgroups, which stream at least two stages; a warpgroup count
    or cluster outside one or two is refused where the tile is made; the kernel gradient's schedule is one
    multiprocessor's, so the tile it derives is no pair and a direct kernel gradient refuses one."""
    for fields in ({"warpgroups": 1, "cluster": 2}, {"warpgroups": 3}, {"warpgroups": 2, "cluster": 4},
                   {"warpgroups": 2, "stages": 1}):
        with pytest.raises(ValueError):
            mosaic_gpu.Tiles(**{"block_m": 128, "block_n": 128, "block_k": 64, "stages": 3, "grid_block_n": 1}
                             | fields)
    launch = grouped_matmul.MosaicTgmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    with generation.assumed("sm90"):
        space = launch.space(launch.bind(bf16((ROWS, 512)), bf16((ROWS, WIDTH)),
                                         jax.ShapeDtypeStruct((3,), jnp.int32), out_dtype="float32",
                                         interpret=False))
    assert space and all(tile.cluster == 1 for tile in space)
    x = jnp.zeros((ROWS, 512), jnp.bfloat16)
    with pytest.raises(ValueError):
        mosaic_gpu.tgmm(x, jnp.zeros((ROWS, WIDTH), jnp.bfloat16), jnp.asarray(SIZES),
                        tiles=mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, cluster=2),
                        out_dtype=jnp.float32, multiprocessors=2)


def test_a_pin_names_only_a_config_the_launchs_kernel_takes():
    """A pinned config is checked by the launch's legality as it traces: Mosaic's ring of one stage (its
    walk keeps a step's products in flight while the next step loads, so it needs two slots) and Triton's
    exact-sums kernel gradient over more rows a step than its sums stay exact for (`EXACT_STEP`, which its
    fitting, not its resources, states) are refused; a legal pin of either runs."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jnp.asarray([128, 128], jnp.int32)
    x, w, dy = bf16((256, 128)), bf16((2, 128, 64)), bf16((256, 64))

    def forward(x, w):
        return template.gemm(x, w, sizes, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def gradient(x, dy):
        return template.gemm_t(x, dy, sizes, out_dtype=jnp.float32, precision="highest",
                               interpret_on_cpu=True)

    calls = ((forward, (x, w), "mosaic_gpu", grouped_matmul.MosaicGmm, mosaic_gpu.Tiles(64, 64, 64, 1, 1),
              mosaic_gpu.Tiles(64, 64, 64, 2, 1)),
             (gradient, (x, dy), "triton", grouped_matmul.Tgmm, grouped_matmul.Tile(128, 128, 64, 4, 3),
              grouped_matmul.Tile(64, 128, 64, 4, 3)))
    for run, args, named, launch, refused, legal in calls:
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=named):
            with tuning.pinned({launch: refused}), pytest.raises(tuning.Unpinnable):
                jax.jit(run).trace(*args)
            with tuning.pinned({launch: legal}), tuning.seen() as launches:
                jax.jit(run).trace(*args)
        assert legal in {config for (op, _), (config, _) in zip(launches, launches.taken, strict=True)
                         if isinstance(op, launch)}


@pytest.mark.parametrize("trans_rhs", [False, True], ids=["forward", "input gradient"])
def test_the_forward_pairs_a_groups_row_tiles_where_its_rows_span_two(trans_rhs):
    """On an H100 the forward and the input gradient derive a pair of multiprocessors a cluster
    (`mosaic.Family.cluster`) where a group averages a row tile's rows, each multiprocessor's tile and
    stages as alone, and one multiprocessor where it averages fewer; an A100's mma.sync bundle takes no
    cluster at any rows. Each derived tile is legal."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    groups, depth, width = 8, 2048, 1536
    launch = grouped_matmul.MosaicGmm()
    derived = {}
    for name, rows in (("sm90", 8 * 512), ("sm90", 8 * 100), ("sm80", 8 * 512)):
        with generation.assumed(name):
            target = generation.gpu_target()
            ba = launch.bind(bf16((rows, width if trans_rhs else depth)), bf16((groups, depth, width)),
                             jax.ShapeDtypeStruct((groups,), jnp.int32), out_dtype="bfloat16",
                             trans_rhs=trans_rhs, interpret=False)
            derived[name, rows] = launch.derived(ba, target)
            assert launch.legal(ba, derived[name, rows], target)
    paired, alone = derived["sm90", 8 * 512], derived["sm90", 8 * 100]
    assert (paired.cluster, alone.cluster, derived["sm80", 8 * 512].cluster) == (2, 1, 1)
    assert (paired.block_m, paired.block_n, paired.stages) == (alone.block_m, alone.block_n, alone.stages)


def test_exact_sums_go_to_fp32_outputs_at_a_precision_above_the_default():
    """The declared product promotes (`exact_sums`) for an fp32 output at the highest precision, never
    for a 16-bit output or the default precision; Mosaic's kernel gradient keeps only tiles whose two
    accumulators stay in registers, and Triton's only steps of at most `EXACT_STEP` rows."""
    assert product.promoted(above=True, outputs=("float32",))
    assert not product.promoted(above=True, outputs=("bfloat16",))
    assert not product.promoted(above=False, outputs=("float32",))
    rows = 2048
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        launch = grouped_matmul.MosaicTgmm()
        exact = launch.bind(bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32",
                            interpret=False, exact_sums=True)
        plain = launch.bind(bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32",
                            interpret=False)
        # Two accumulators a consumer warpgroup: its 128 threads hold its share of the tile's columns, in
        # the registers a cooperative consumer has (232) or a lone warpgroup's (255). The generated tile
        # holds them; a tile that does not is no legal pin.
        [tile] = launch.fitting(exact, target)
        assert 2 * tile.block_m * tile.block_n // (128 * tile.warpgroups) < resources.CONSUMER_REGISTERS
        narrow, wide = (mosaic_gpu.Tiles(128, width, 64, 2, 1, warpgroups=2) for width in (128, 256))
        assert launch.legal(exact, narrow, target) and launch.legal(plain, narrow, target)  # 128 a thread
        assert not launch.legal(exact, wide, target) and launch.legal(plain, wide, target)  # 256
        # Triton's kernel gradient with exact sums takes at most EXACT_STEP rows a step.
        triton = grouped_matmul.Tgmm()
        steps = {tile.block_m for tile in triton.fitting(triton.bind(
            bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32", interpret=False,
            exact_sums=True), target)}
        assert steps and max(steps) <= grouped_matmul.EXACT_STEP


@lowers("sm90")
def test_the_exact_mosaic_kernel_gradient_lowers_for_an_h100():
    """Mosaic GPU's kernel gradient with `exact_sums`, its running sum the pipeline's carry, lowers for
    an H100 on the CPU at the tile it derives."""
    rows, depth, width, groups = 2044, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((rows, depth)), bf16((rows, width)), jax.ShapeDtypeStruct((groups,), jnp.int32), ())
    options = {"out_dtype": "float32", "interpret": False, "exact_sums": True}
    launch = grouped_matmul.MosaicTgmm()
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        [config] = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
            lowering_platforms=("cuda",))


def selected(x, mask):
    """A prologue keeping the rows an integer mask selects."""
    return jnp.where(mask[:, None] > 0, x, jnp.zeros_like(x))


@runs("sm80")
def test_an_integer_side_operand_takes_no_gradient():
    """A prologue selecting rows by an integer mask differentiates through the kernels, the mask's
    gradient float0: the rows' gradient on par with JAX's of the plain program over the draws (check 1),
    truth each selected row's matrix summed over its columns in float64, and every row the mask drops
    exactly zero. The rows past the groups' meet no matrix."""
    mask = jnp.asarray(np.arange(ROWS) % 3 == 0, jnp.int32)
    routed = sum(SIZES)
    groups = np.concatenate([np.repeat(np.arange(len(SIZES)), SIZES), np.zeros(ROWS - routed, np.int64)])
    kept = (np.asarray(mask) > 0) & (np.arange(ROWS) < routed)

    def draw(seed):
        x, w, *_ = _draw(seed)
        return x, w

    def gradient(*, interpret):
        def loss(x, w):
            out = template.gemm(x, w, jnp.asarray(SIZES), prologue=selected,
                                prologue_operands=((mask, ("rows",)),), out_dtype=jnp.float32,
                                interpret_on_cpu=interpret)
            return jnp.sum(out)
        return jax.jit(jax.grad(loss))

    def truth(x, w):
        # d/dx of the sum of x[r] @ w[g(r)] over the selected rows r.
        return jnp.where(kept[:, None], jnp.sum(w, axis=-1)[groups], 0.0)

    with generation.assumed("sm80"):
        got = gradient(interpret=True)(*draw(0))
        assert not np.any(np.asarray(got, np.float32)[~kept])
        assert_as_exact(gradient(interpret=True), gradient(interpret=False), draw, "integer mask gradient",
                        seeds=SEEDS, truth_fn=truth)


@runs("sm80")
@pytest.mark.parametrize("consumer, exact", [(lambda p, s: p * s, lambda p, s: p * s),
                                             (lambda p, s: p.T.astype(jnp.bfloat16), lambda p, s: p.T)],
                         ids=["scalar", "transpose"])
def test_a_consumer_the_kernels_cannot_tile_runs_after_the_product(consumer, exact):
    """A consumer of a captured scalar (no launch axis) or of the product transposed (its block not the
    tile's) runs after the template's product: on par with the plain product's consumer over the draws
    (check 1), truth the product and the consumer in float64."""
    scale = jnp.asarray(1.5, jnp.float32)
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, *_ = _draw(seed)
        return x, w

    def plain(consume):
        def run(x, w):
            wide = jnp.promote_types(x.dtype, jnp.float32)
            matrix = w.astype(jnp.bfloat16).astype(wide)
            return consume(jax.lax.ragged_dot(x.astype(wide), matrix, sizes, preferred_element_type=wide),
                           scale.astype(wide))
        return run

    def fusing(x, w):
        return consumer(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), scale)

    with generation.assumed("sm80"):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(plain(consumer)), draw,
                        "a consumer after the product", seeds=SEEDS, truth_fn=plain(exact))


HIGHEST, DEFAULT = jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT


@pytest.mark.parametrize(("compute", "precision", "outputs", "promoted", "refused"), [
    (jnp.bfloat16, HIGHEST, (jnp.float32,), True, None),
    (jnp.bfloat16, HIGHEST, (jnp.bfloat16,), False, None),
    (jnp.bfloat16, HIGHEST, (jnp.bfloat16, jnp.float32), True, None),
    (jnp.float16, DEFAULT, (jnp.float32,), False, None),
    (jnp.float32, DEFAULT, (jnp.float32,), False, None),
    (jnp.float32, HIGHEST, (jnp.float32,), True, "TF32"),
    (jnp.int8, DEFAULT, (jnp.float32,), False, "int8"),
])
def test_a_product_declares_its_promotion_and_the_kernels_refusal(compute, precision, outputs, promoted,
                                                                   refused):
    """A precision above the default promotes each depth step's partial where an output is fp32; the
    kernels refuse fp32 products above the default (they multiply in TF32) and a dtype they do not
    multiply in."""
    declared = product.declared(compute, precision, outputs=outputs)
    assert declared.promoted is promoted
    assert (declared.refusal is None) is (refused is None)
    assert refused is None or refused in declared.refusal


def swiglu(gate, up):
    """A SwiGLU epilogue on a paired product's two fp32 accumulators, each rounded to bf16 first and the
    activated gate again, as `activations.gated_product` rounds them."""
    gate, up = gate.astype(jnp.bfloat16), up.astype(jnp.bfloat16)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up).astype(jnp.float32)


PAIRED_TILE = grouped_matmul.Tile(64, 128, 64, 4, 2)


def test_a_grouped_call_no_cooperative_tile_holds_derives_one_consumer_alone():
    """Where no two-consumer tile holds a grouped call, its derivation falls back to one consumer, which
    runs on one multiprocessor: a pair is two consumers' (`mosaic_gpu.Tiles`), so the fallback derives a
    tile rather than refusing one it built."""
    from dew.nn.kernels.templates import mosaic_sm90

    tile = mosaic_gpu.derived(mosaic_sm90.WGMMA, rows=512, depth=2048, width=1536, dtype=jnp.bfloat16,
                              multiprocessors=132, holds=lambda tile: tile.warpgroups == 1, grouped=True)
    assert tile is not None and (tile.warpgroups, tile.cluster) == (1, 1)


def test_a_paired_product_holds_check_1_against_its_two_products_combined():
    """The two-source dense kernel (one `lhs` tile, two matrices, two accumulators) at a tile, with rows,
    depth and width no tile divides: its SwiGLU of the two products on par with the plain two products'
    combined by the same epilogue over the draws (check 1), truth the two products' SwiGLU in float64."""
    rows, depth, width = 200, 300, 140
    shapes = ((rows, depth), (depth, width), (depth, width))

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(7), seed), 3)
        return tuple(jax.random.normal(key, shape).astype(jnp.bfloat16)
                     for key, shape in zip(keys, shapes, strict=True))

    def run(lhs, gate, up):
        [paired] = triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",),
                                    tile=PAIRED_TILE, transpose_rhs=False, interpret=True)
        return paired

    def plain(lhs, gate, up):
        def product(matrix):
            return jnp.dot(lhs, matrix, preferred_element_type=jnp.float32)
        return swiglu(product(gate), product(up)).astype(jnp.bfloat16)

    def truth(lhs, gate, up):
        return jax.nn.silu(lhs @ gate) * (lhs @ up)

    paired = run(*draw(0))
    assert paired.shape == (rows, width) and paired.dtype == jnp.bfloat16
    assert_as_exact(run, jax.jit(plain), draw, "paired product", seeds=SEEDS, truth_fn=truth)


@pytest.mark.parametrize("generation_name", ["sm80", "sm86", "sm90"])
def test_a_paired_dense_product_lowers_for_cuda(generation_name):
    """The two-source dense kernel lowers for CUDA on the CPU."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((1021, 200)), bf16((200, 300)), bf16((200, 300)))

    def paired(lhs, gate, up):
        return triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",),
                                tile=PAIRED_TILE, transpose_rhs=False, interpret=False)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(paired).trace(*args).lower(lowering_platforms=("cuda",))


def test_a_paired_product_without_an_epilogue_is_refused():
    lhs = jnp.zeros((64, 64), jnp.bfloat16)
    with pytest.raises(ValueError):
        triton_dense.gmm(lhs, lhs, paired=lhs, out_dtypes=("bfloat16",), tile=PAIRED_TILE,
                         transpose_rhs=False, interpret=True)


@lowers("sm90")
def test_a_hook_that_broadcasts_a_side_operand_lowers_for_hopper():
    """A side operand given back with its dimension of one, for which Mosaic's layout inference finds no
    layout: a Hopper call's forward and kernel gradient with such a prologue lower (GMMGATE-h100-8)."""
    drawn = [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in _draw(0)]
    dew = _drop(_dew(CASES["broadcast axes"]))

    def loss(*drawn):
        return sum(jnp.sum(out.astype(jnp.float32)) for out in dew(*drawn))

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(jax.grad(loss, argnums=range(len(drawn)))).trace(*drawn).lower(lowering_platforms=("cuda",))



@pytest.mark.parametrize("columns", [128, 300, 384], ids=lambda n: f"{n}-columns")
def test_an_interpreted_fp32_product_at_highest_rounds_its_exact_sum_once(columns):
    """A promoted fp32 product issued off a GPU (`product.interpreted`) accumulates in float64 and rounds
    once, at every shape, where XLA's CPU dot takes a less exact fp32 accumulation at a 128-aligned shape
    than at another. The operands are 12-bit integers at scales 2^-4 to 2^4, so every product and partial
    sum is a multiple of 2^-8 below 2^37, exact in float64 in any order and past fp32's 24 bits: the
    product is the exact sum rounded once to fp32, to the bit. That is the contract under test, the float64
    accumulation: an fp32 one would round the partial sums, so equality is no claim that fp32 accumulation
    is exact here."""
    generator = np.random.default_rng(0)

    def drawn(shape):
        scales = 2.0 ** generator.integers(-4, 5, shape)
        return (generator.integers(-2048, 2048, shape) * scales).astype(np.float32)

    lhs, rhs = jnp.asarray(drawn((128, 128))), jnp.asarray(drawn((128, columns)))
    declared = product.declared(jnp.float32, jax.lax.Precision.HIGHEST, outputs=(jnp.float32,))
    found = jax.jit(lambda a, b: product.interpreted(declared, a, b, (((1,), (0,)), ((), ()))))(lhs, rhs)
    exact = np.asarray(lhs, np.float64) @ np.asarray(rhs, np.float64)
    np.testing.assert_array_equal(np.asarray(found), exact.astype(np.float32))


def test_an_interpreted_fp32_sum_that_cancels_at_mixed_magnitudes_is_its_float64_sum():
    """2^24 * 2^24, 1 * 1 and -(2^24 * 2^24): each product is exact in fp32, and their float64 sum is 1 in
    every order (2^48 + 1 is a float64), where an fp32 accumulation that adds a large one first rounds the
    1 away, 0. The interpreted product accumulates in float64: 1 exactly."""
    lhs = jnp.asarray([[2.0 ** 24, 1.0, -(2.0 ** 24)]], jnp.float32)
    rhs = jnp.asarray([[2.0 ** 24], [1.0], [2.0 ** 24]], jnp.float32)
    declared = product.declared(jnp.float32, jax.lax.Precision.HIGHEST, outputs=(jnp.float32,))
    assert float(product.interpreted(declared, lhs, rhs, (((1,), (0,)), ((), ())))[0, 0]) == 1.0


def test_a_declared_cell_compiles_nothing_for_the_template_it_prefers(monkeypatch, tmp_path):
    """A declared cell's call (an A100's grouped kernel gradient, `grouped_matmul.DECLARED`), traced, keeps
    no kernel report: no launch rests on the template it prefers, so the compiler's word on its tile is never
    asked (`selection.preferring`; the word cost 1.5 s of an A100 trace, CONFIRMCOST1)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    x, dy = jax.ShapeDtypeStruct((4096, 1024), jnp.bfloat16), jax.ShapeDtypeStruct((4096, 2048), jnp.bfloat16)
    sizes = jnp.full((8,), 512, jnp.int32)
    with generation.assumed("sm80"):
        jax.jit(lambda x, dy: template.gemm_t(x, dy, sizes, out_dtype=jnp.bfloat16)).trace(x, dy)
    assert not (tmp_path / "dew" / "kernel-reports").exists()
