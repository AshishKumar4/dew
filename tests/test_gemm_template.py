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
from generation_support import INTERPRETS, SUMS_IN_IEEE, generations, lowers, runs
from jax.experimental.pallas import fuser
from jax.sharding import Mesh
from reference_error import as_given, assert_as_exact

from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.templates import product
from dew.nn.kernels.templates.gemm import fused, mosaic_gpu, template, triton_dense

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


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
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
"""A tile two consumer warpgroups compute (a cluster the interpreter does not model)."""


MOSAIC_RAN = {'mosaic_gpu'} if jax.default_backend() == 'cpu' else {'mosaic_gpu', 'triton'}
"""The grouped matmul's kernels a call pinned to Mosaic reports: a GPU process also traces the CPU branch
of its platform choice, which interprets Triton's and never runs there (`grouped_matmul._launch`)."""


def _kernels(taken) -> set:
    return {route.kernel for route in taken if route.choice == 'grouped_matmul'}


@runs("sm90")
@pytest.mark.parametrize("case", ["norm and bias", "gated prologue", "cast prologue"])
def test_a_cooperative_tile_holds_check_1_with_its_hooks(case):
    """The Hopper launches pinned to a cooperative tile, interpreted: each consumer rewrites half the stage
    both read and computes half the columns' epilogue, as exact as the plain program from float64 truth;
    the Mosaic kernels ran."""
    spec = CASES[case]
    pinned = {grouped_matmul.MosaicGmm: COOPERATIVE, grouped_matmul.MosaicGmmTransposed: COOPERATIVE,
              grouped_matmul.MosaicTgmm: COOPERATIVE}
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned),
          generation.routes() as taken):
        dew = jax.jit(_drop(_dew(spec, interpret_on_cpu=True)))
        assert_as_exact(dew, jax.jit(_drop(_plain(spec))), _draw, f"{case} cooperative",
                        seeds=SEEDS, truth_fn=_drop(_plain(spec, truth=True)))
    assert _kernels(taken) == MOSAIC_RAN


def _loss(run):
    def loss(x, w, ms, nw, b, u, s, ct):
        outs = run(x, w, ms, nw, b, u, s)
        return sum(jnp.sum(_wide(out) * ct.astype(_wide(out).dtype)) for out in outs)
    return jax.grad(loss, argnums=tuple(range(7)))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_the_templates_gradients_hold_check_1(generation_name, case):
    """Every operand's cotangent through the template's own reverse mode, against JAX's derivative of
    the plain program, from float64 truth. Mosaic's interpreter runs ordered callbacks, which a jitted
    `linear_call` cannot carry, so its gradients run eagerly."""
    spec = CASES[case]
    with generation.assumed(generation_name):
        dew = _loss(_dew(spec, interpret_on_cpu=True))
        dew = dew if generation_name == "sm90" else jax.jit(dew)
        assert_as_exact(as_given(dew), _loss(_plain(spec)), _draw, f"{case} gradients on {generation_name}",
                        seeds=SEEDS, truth_fn=_loss(_plain(spec, truth=True)))


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
        return x, w, ms, nw, b

    def dew(x, w, ms, nw, b):
        return template.gemm(x, w, None, prologue=normed,
                             prologue_operands=((ms, ("rows",)), (nw, ("depth",))), epilogue=biased,
                             epilogue_operands=((b, ("columns",)),), out_dtype=jnp.bfloat16,
                             transpose_rhs=transpose_rhs, interpret_on_cpu=True)

    def plain(x, w, ms, nw, b, wide=jnp.float32, out=jnp.bfloat16):
        # A 16-bit output: an fp32 one's cotangent makes the kernel gradient an fp32 product, XLA's.
        w = w.T if transpose_rhs else w
        return (jnp.dot(normed(x, ms, nw).astype(wide), w.astype(wide)) + b[None, :].astype(wide)).astype(out)

    def loss(run):
        return jax.grad(lambda *a: jnp.sum(run(*a) * jnp.linspace(-1, 1, 100)), argnums=(0, 1, 2, 3, 4))

    with generation.assumed("sm80"), tuning.seen() as launches:
        truth = functools.partial(plain, wide=jnp.float64, out=jnp.float64)
        assert_as_exact(jax.jit(dew), jax.jit(plain), draw, "dense 65x40x100", seeds=SEEDS, truth_fn=truth)
        assert_as_exact(jax.jit(loss(dew)), jax.jit(loss(plain)), draw, "dense 65x40x100 gradients",
                        seeds=SEEDS, truth_fn=loss(truth))
    ran = {(type(launch).__name__, ba.arguments.get('dense')) for launch, ba in launches}
    assert {('GmmTransposed' if transpose_rhs else 'Gmm', True), ('Tgmm', True)} <= ran


def _dense_draw(rows: int, depth: int, width: int):
    """A draw of a dense product's bf16 operands, `[rows, depth]` and `[depth, width]`."""
    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=(depth, width)), jnp.bfloat16))
    return draw


def test_a_dense_gradient_takes_a_padded_tile_where_no_dividing_one_fits():
    """A dense kernel gradient at depth 96 on Hopper: the tiles dividing 96 have 32 output rows, which
    Hopper's warpgroup MMA refuses, so a 64-row tile runs over the padded depth, and the gradients hold
    check 1 against JAX's derivative of the plain product."""
    draw = _dense_draw(256, 96, 128)

    def loss(run):
        return jax.grad(lambda x, w: jnp.sum(run(x, w) * jnp.linspace(-1, 1, 128)), argnums=(0, 1))

    def dew(x, w):
        return template.gemm(x, w, None, out_dtype=jnp.bfloat16, interpret_on_cpu=True)

    def plain(x, w, wide=jnp.float32, out=jnp.bfloat16):
        # A 16-bit output: an fp32 one's cotangent makes the kernel gradient an fp32 product, XLA's.
        return jnp.dot(x.astype(wide), w.astype(wide)).astype(out)

    with generation.assumed("sm90"), tuning.seen() as launches:
        assert_as_exact(jax.jit(loss(dew)), jax.jit(loss(plain)), draw, "dense depth 96", seeds=SEEDS,
                        truth_fn=loss(functools.partial(plain, wide=jnp.float64, out=jnp.float64)))
    assert ('Tgmm', True) in {(type(launch).__name__, ba.arguments.get('dense')) for launch, ba in launches}


@pytest.mark.gpu(reason="checks Triton's MMA accumulation on a GPU")
@pytest.mark.skipif(not generation.triton_runs(), reason="needs a GPU the Pallas kernels compile for")
def test_a_dense_products_exact_sums_survive_one_row_tile():
    """At the highest precision, a dense fp32 product of one row tile and many depth steps (its first grid
    axis one program, where a program id is a constant) sums each step's product by an IEEE add: its
    error holds check 1 against XLA's IEEE sums, which a tensor core's truncating accumulation does not."""
    draw = _dense_draw(64, 4096, 128)

    def dew(x, w):
        return template.gemm(x, w, None, out_dtype=jnp.float32, precision="highest")

    def xla(x, w):
        return jnp.dot(x, w, precision="highest", preferred_element_type=jnp.float32)

    def truth(x, w):
        return jnp.dot(x.astype(jnp.float64), w.astype(jnp.float64))

    with generation.routes() as taken:
        assert_as_exact(jax.jit(dew), jax.jit(xla), draw, "dense exact sums, one row tile", seeds=SEEDS,
                        truth_fn=truth)
    assert {route.kernel for route in taken if route.choice == 'grouped_matmul'} == {'triton'}


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
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


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_broadcast_dimension_is_the_squeezed_operands(generation_name):
    """A `[rows, 1]` mean square and a `[1, width]` bias with a None axis give the products of their
    vectors, bit for bit, through the one implementation that takes both (Triton's: Mosaic's refuses a
    side operand given back with a dimension of one)."""
    drawn = _draw(2)
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="triton"):
        column = jax.jit(_drop(_dew(CASES["broadcast axes"], interpret_on_cpu=True)))(*drawn)
        vector = jax.jit(_drop(_dew(CASES["norm and bias"], interpret_on_cpu=True)))(*drawn)
    np.testing.assert_array_equal(np.asarray(column[0], np.float32), np.asarray(vector[0], np.float32))


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


@lowers("sm90")
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


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("launch",
                         [grouped_matmul.Gmm(), grouped_matmul.GmmTransposed(), grouped_matmul.Tgmm()],
                         ids=lambda launch: type(launch).__name__)
def test_the_hooked_dense_launches_lower_for_cuda(launch, generation_name):
    """Each Triton launch of a dense call, with a prologue (and the forward's epilogue) and at the highest
    precision's exact sums, lowers for CUDA on the CPU at a fitting tile of every block shape, through the
    plain tiled kernel (its `pallas_call` named), with rows, depth and width no tile divides."""
    rows, depth, width = 1021, 200, 300
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((1,), jnp.int32)
    prologue = (normed, (f32((rows,)), f32((depth,))), (("rows",), ("depth",)))
    if isinstance(launch, grouped_matmul.Tgmm):
        args = (bf16((rows, depth)), bf16((rows, width)), sizes, prologue[1])
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2])
        options = {"out_dtype": "float32", "hooks": hooks, "interpret": False}
        name = "pallas_triton_dense_tdot"
    else:
        trans_rhs = isinstance(launch, grouped_matmul.GmmTransposed)
        matrix = bf16((1, width, depth) if trans_rhs else (1, depth, width))
        args = (bf16((rows, depth)), matrix, sizes, prologue[1], (f32((width,)),))
        hooks = grouped_matmul.Hooks(prologue[0], prologue[2], biased, (("columns",),),
                                     out_dtypes=("float32",))
        options = {"out_dtype": "float32", "trans_rhs": trans_rhs, "hooks": hooks, "interpret": False}
        name = "pallas_triton_dense_dot"
    options |= {"dense": True, "exact_sums": True}
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        fits = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        covering = set({(c.block_m, c.block_n, c.block_k): c for c in reversed(fits)}.values())
        assert covering
        for config in sorted(covering, key=repr):
            lowered = jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))
            assert name in lowered.as_text()


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


def test_mosaic_takes_only_the_hooks_its_lowering_has_rules_for():
    """Mosaic GPU's warpgroup lowering has no transpose: a hook that transposes a side tile leaves the
    Hopper launch to Triton's, which takes it; a hook of elementwise arithmetic stays Mosaic's."""
    rows = 256
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    launch = grouped_matmul.MosaicTgmm()
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        operands = (bf16((rows, DEPTH)), bf16((rows, WIDTH)), sizes)
        transposing = launch.bind(*operands, (f32((DEPTH, rows)),), out_dtype="float32",
                                  hooks=grouped_matmul.Hooks(shifted, (("depth", "rows"),)), interpret=False)
        elementwise = launch.bind(*operands, (f32((rows,)), f32((DEPTH,))), out_dtype="float32",
                                  hooks=grouped_matmul.Hooks(normed, (("rows",), ("depth",))),
                                  interpret=False)
        assert not launch.eligible(transposing, target)
        assert launch.eligible(elementwise, target)


@INTERPRETS
def test_the_mosaic_kernels_load_a_side_operands_rows_from_any_dimension():
    """Interpreted, Mosaic GPU's kernels pad and slice a side operand along its rows wherever they sit:
    a `[depth, rows]` side gives the `[rows, depth]` one's products bit for bit."""
    rows, sizes = 136, jnp.asarray([70, 66])
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(rows, 64)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(2, 64, 64)) / 8, jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(rows, 64)), jnp.bfloat16)
    side = jnp.asarray(rng.normal(size=(rows, 64)), jnp.float32)

    def row_major(x, side):
        return (_wide(x) + side).astype(x.dtype)

    tiles = mosaic_gpu.Tiles(64, 64, 64, 2, 1)
    layouts = ((row_major, ("rows", "depth"), side), (shifted, ("depth", "rows"), side.T))
    common = {"tiles": tiles, "out_dtype": jnp.float32, "multiprocessors": 1,
              "interpret": grouped_matmul.INTERPRETED}
    products = [mosaic_gpu.gmm(x, w, sizes, transpose_rhs=False, prologue=hook, prologue_operands=(operand,),
                               prologue_axes=(layout,), **common) for hook, layout, operand in layouts]
    gradients = [mosaic_gpu.tgmm(x, dy, sizes, prologue=hook, prologue_operands=(operand,),
                                 prologue_axes=(layout,), **common) for hook, layout, operand in layouts]
    np.testing.assert_array_equal(np.asarray(products[0]), np.asarray(products[1]))
    np.testing.assert_array_equal(np.asarray(gradients[0]), np.asarray(gradients[1]))


@SUMS_IN_IEEE
@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_exact_sums_add_each_steps_product_as_the_plain_sum_does(generation_name):
    """At the highest precision an fp32 kernel gradient adds each step's product apart (`exact_sums`):
    interpreted on the CPU, whose dots sum in IEEE fp32 either way, at one tile it is the default path's
    bit for bit."""
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.bfloat16)
    sizes = jnp.asarray(SIZES)
    # One tile both ways, a step of EXACT_STEP rows, so the order of the sums is the same.
    tiles = {grouped_matmul.Tgmm: grouped_matmul.Tile(64, 128, 64, 4, 2),
             grouped_matmul.MosaicTgmm: mosaic_gpu.Tiles(64, 128, 64, 2, 1)}
    with generation.assumed(generation_name), tuning.pinned(tiles):
        for group_sizes in (sizes, None):
            exact, default = (jax.jit(functools.partial(template.gemm_t, out_dtype=jnp.float32,
                                                        precision=precision, interpret_on_cpu=True))(
                x, dy, group_sizes) for precision in ("highest", "default"))
            np.testing.assert_array_equal(np.asarray(exact), np.asarray(default))


@runs("sm90")
@pytest.mark.parametrize("case", ["norm and bias"])
def test_a_cooperative_tiles_gradients_hold_check_1(case):
    """`test_a_cooperative_tile_holds_check_1_with_its_hooks`'s reverse mode: the input gradient through the
    cooperative transposed launch, the kernel gradient through the cooperative kernel gradient with the
    prologue each consumer applies to its depth half, against JAX's derivative of the plain program
    (eagerly: `test_the_templates_gradients_hold_check_1`)."""
    spec = CASES[case]
    pinned = {grouped_matmul.MosaicGmm: COOPERATIVE, grouped_matmul.MosaicGmmTransposed: COOPERATIVE,
              grouped_matmul.MosaicTgmm: COOPERATIVE}
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned),
          generation.routes() as taken):
        assert_as_exact(as_given(_loss(_dew(spec, interpret_on_cpu=True))), _loss(_plain(spec)), _draw,
                        f"{case} cooperative gradients", seeds=SEEDS,
                        truth_fn=_loss(_plain(spec, truth=True)))
    assert _kernels(taken) == MOSAIC_RAN


@SUMS_IN_IEEE
def test_a_cooperative_tiles_exact_sums_add_as_its_plain_sum_does():
    """`test_exact_sums_add_each_steps_product_as_the_plain_sum_does` at a cooperative tile: each consumer's
    running fp32 sum, the pipeline's carry, matches its accumulator's bit for bit on the CPU."""
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.bfloat16)
    with generation.assumed("sm90"), tuning.pinned({grouped_matmul.MosaicTgmm: COOPERATIVE}):
        exact, default = (jax.jit(functools.partial(template.gemm_t, out_dtype=jnp.float32,
                                                    precision=precision, interpret_on_cpu=True))(
            x, dy, jnp.asarray(SIZES)) for precision in ("highest", "default"))
    np.testing.assert_array_equal(np.asarray(exact), np.asarray(default))


def test_a_tile_states_a_schedule_the_kernels_run():
    """A pair of multiprocessors needs two warpgroups; a warpgroup count or cluster outside one or two is
    refused where the tile is made, and a direct kernel gradient refuses a depth its pairs do not divide."""
    for fields in ({"warpgroups": 1, "cluster": 2}, {"warpgroups": 3}, {"warpgroups": 2, "cluster": 4}):
        with pytest.raises(ValueError):
            mosaic_gpu.Tiles(128, 128, 64, 3, 1, **fields)
    x = jnp.zeros((ROWS, 384), jnp.bfloat16)
    with pytest.raises(ValueError):
        mosaic_gpu.tgmm(x, jnp.zeros((ROWS, WIDTH), jnp.bfloat16), jnp.asarray(SIZES),
                        tiles=mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, cluster=2),
                        out_dtype=jnp.float32, multiprocessors=2)


def test_an_interpreted_launch_takes_no_pair():
    """The interpreter runs one block, which no cluster fits: an interpreted launch's space has no pair,
    and a pair pinned on one multiprocessor is refused rather than launched over no blocks."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((ROWS, DEPTH)), bf16((3, DEPTH, WIDTH)), jax.ShapeDtypeStruct((3,), jnp.int32))
    launch = grouped_matmul.MosaicGmm()
    with generation.assumed("sm90"):
        interpreted = launch.space(launch.bind(*args, out_dtype="bfloat16", trans_rhs=False, interpret=True))
        compiled = launch.space(launch.bind(*args, out_dtype="bfloat16", trans_rhs=False, interpret=False))
    assert interpreted and not any(tile.cluster == 2 for tile in interpreted)
    assert any(tile.cluster == 2 for tile in compiled)
    with pytest.raises(ValueError):
        mosaic_gpu.gmm(jnp.zeros((ROWS, DEPTH), jnp.bfloat16), jnp.zeros((3, DEPTH, WIDTH), jnp.bfloat16),
                       jnp.asarray([100, 100, 107]), tiles=mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2,
                                                                             cluster=2),
                       transpose_rhs=False, out_dtype=jnp.bfloat16, multiprocessors=1,
                       interpret=grouped_matmul.INTERPRETED)


def test_exact_sums_go_to_fp32_outputs_at_a_precision_above_the_default():
    """The declared product promotes (`exact_sums`) for an fp32 output at the highest precision, never
    for a 16-bit output or the default precision; Mosaic's forward launches, which do not add apart,
    refuse it, Mosaic's kernel gradient keeps only tiles whose two accumulators stay in registers, and
    Triton's only steps of at most `EXACT_STEP` rows."""
    assert product.promoted(above=True, outputs=("float32",))
    assert not product.promoted(above=True, outputs=("bfloat16",))
    assert not product.promoted(above=False, outputs=("float32",))
    rows = 2048
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((8,), jnp.int32)
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        forward = grouped_matmul.MosaicGmm().bind(bf16((rows, 768)), bf16((8, 768, 1536)), sizes, (), (),
                                                  out_dtype="float32", trans_rhs=False, interpret=False,
                                                  exact_sums=True)
        assert not grouped_matmul.MosaicGmm().eligible(forward, target)
        launch = grouped_matmul.MosaicTgmm()
        exact = launch.bind(bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32",
                            interpret=False, exact_sums=True)
        plain = launch.bind(bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32",
                            interpret=False)
        assert launch.eligible(exact, target)
        kept = launch.fitting(exact, target)
        # Two accumulators a consumer warpgroup: its 128 threads hold its share of the tile's columns, in
        # the registers a cooperative consumer has (232) or a lone warpgroup's (255).
        assert mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2) in kept  # 128 registers a thread
        assert mosaic_gpu.Tiles(128, 256, 64, 3, 1, warpgroups=2) not in kept  # 256
        assert kept and all(2 * tile.block_m * tile.block_n // (128 * tile.warpgroups)
                            < (grouped_matmul.CONSUMER_REGISTERS if tile.warpgroups == 2 else 255)
                            for tile in kept)
        assert set(kept) < set(launch.fitting(plain, target))
        # Triton's kernel gradient with exact sums takes at most EXACT_STEP rows a step.
        triton = grouped_matmul.Tgmm()
        steps = {tile.block_m for tile in triton.fitting(triton.bind(
            bf16((rows, 768)), bf16((rows, 1536)), sizes, (), out_dtype="float32", interpret=False,
            exact_sums=True), target)}
        assert steps and max(steps) <= grouped_matmul.EXACT_STEP


@lowers("sm90")
def test_the_exact_mosaic_kernel_gradient_lowers_for_an_h100():
    """Mosaic GPU's kernel gradient with `exact_sums`, its running sum the pipeline's carry, lowers for
    an H100 on the CPU at a fitting tile of every block shape."""
    rows, depth, width, groups = 2044, 768, 1536, 8
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((rows, depth)), bf16((rows, width)), jax.ShapeDtypeStruct((groups,), jnp.int32), ())
    options = {"out_dtype": "float32", "interpret": False, "exact_sums": True}
    launch = grouped_matmul.MosaicTgmm()
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        fits = launch.fitting(launch.bind(*args, **options), generation.gpu_target())
        covering = set({(c.block_m, c.block_n): c for c in reversed(fits)}.values())
        assert covering
        for config in sorted(covering, key=repr):
            jax.jit(functools.partial(launch.replace(config=config), **options)).trace(*args).lower(
                lowering_platforms=("cuda",))


def selected(x, mask):
    """A prologue keeping the rows an integer mask selects."""
    return jnp.where(mask[:, None] > 0, x, jnp.zeros_like(x))


def transposed_twice_nested(x, side):
    """`shifted`, its transpose two jitted calls deep."""
    return (_wide(x) + jax.jit(jax.jit(lambda s: s.T))(side)).astype(x.dtype)


def test_an_fp32_product_above_the_default_precision_runs_xla():
    """fp32 products are TF32 in the kernels, which only the default precision asks for: an explicit
    'highest' under a configured default runs XLA's portable program, reported as the route with the
    reason, its products of low-mantissa fp32 values XLA's at 'highest' bit for bit; a kernel chosen for
    it raises."""
    rng = np.random.default_rng(8)
    x = jnp.asarray(1 + rng.integers(1, 2 ** 10, size=(64, DEPTH)) * 2.0 ** -20, jnp.float32)
    w = jnp.asarray(1 + rng.integers(1, 2 ** 10, size=(1, DEPTH, WIDTH)) * 2.0 ** -20, jnp.float32)

    def product(precision):
        return jax.jit(lambda x, w: template.gemm(x, w, jnp.asarray([64]), out_dtype=jnp.float32,
                                                  precision=precision, interpret_on_cpu=True))

    with generation.assumed("sm80"), jax.default_matmul_precision("default"):
        with generation.routes() as highest:
            got = product("highest")(x, w)
        with generation.routes() as default:
            product("default")(x, w)
        with generation.chosen(grouped_matmul="triton"), pytest.raises(ValueError):
            product("highest").lower(x, w)
    taken = [(route.kernel, route.refusal is not None) for route in highest
             if route.choice == "grouped_matmul"]
    assert taken == [("xla", True)]
    assert {route.kernel for route in default if route.choice == "grouped_matmul"} == {"triton"}
    xla = jax.jit(lambda x, w: jax.lax.ragged_dot(x, w, jnp.asarray([64]), precision="highest",
                                                  preferred_element_type=jnp.float32))(x, w)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(xla))


def test_an_integer_side_operand_takes_no_gradient():
    """A prologue selecting rows by an integer mask differentiates as JAX's plain program does: the
    rows' gradient through the kernels, the mask's float0."""
    x, w, *_ = _draw(5)
    mask = jnp.asarray(np.arange(ROWS) % 3 == 0, jnp.int32)

    def loss(x, *, interpret):
        out = template.gemm(x, w, jnp.asarray(SIZES), prologue=selected,
                            prologue_operands=((mask, ("rows",)),), out_dtype=jnp.float32,
                            interpret_on_cpu=interpret)
        return jnp.sum(out)

    with generation.assumed("sm80"):
        got = jax.jit(jax.grad(functools.partial(loss, interpret=True)))(x)
        portable = jax.jit(jax.grad(functools.partial(loss, interpret=False)))(x)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(portable, np.float32))


@pytest.mark.parametrize("consumer", [lambda p, s: p * s, lambda p, s: p.T.astype(jnp.bfloat16)],
                         ids=["scalar", "transpose"])
def test_a_consumer_the_kernels_cannot_tile_runs_after_the_product(consumer):
    """A consumer of a captured scalar (no launch axis) or of the product transposed (its block not the
    tile's) runs in XLA after the template's product, with the unfused program's value."""
    x, w, *_ = _draw(6)
    scale = jnp.asarray(1.5, jnp.float32)
    sizes = jnp.asarray(SIZES)
    with generation.assumed("sm80"):
        def fusing(x, w, s):
            return consumer(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), s)

        got = jax.jit(fuser.fuse(fusing))(x, w, scale)
        want = jax.jit(lambda x, w, s: consumer(template.gemm(x, w, sizes, out_dtype=jnp.float32,
                                                               interpret_on_cpu=True), s))(x, w, scale)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


def test_mosaic_reads_a_hooks_primitives_at_any_depth():
    """A transpose two jitted calls deep is a primitive Mosaic GPU's lowering has no rule for: the
    Hopper launch leaves the hook to Triton's."""
    rows = 256
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((2,), jnp.int32)
    launch = grouped_matmul.MosaicTgmm()
    with generation.assumed("sm90"):
        ba = launch.bind(bf16((rows, DEPTH)), bf16((rows, WIDTH)), sizes,
                         (jax.ShapeDtypeStruct((DEPTH, rows), jnp.float32),), out_dtype="float32",
                         hooks=grouped_matmul.Hooks(transposed_twice_nested, (("depth", "rows"),)),
                         interpret=False)
        assert not launch.eligible(ba, generation.gpu_target())


def test_a_broadcast_hooks_launch_stores_and_reads_back():
    """A launch whose side operand has a dimension of one keeps its module-level hook and the dimension's
    place as data: the launches a trace records store as JSON and read back."""
    x, w, ms, nw, b, u, s, ct = _draw(7)
    with generation.assumed("sm80"), tuning.seen() as launches:
        jax.jit(_drop(_dew(CASES["broadcast axes"], interpret_on_cpu=True))).lower(x, w, ms, nw, b, u, s, ct)
    hooked = [ba for _, ba in launches
              if ba.arguments.get("hooks", grouped_matmul.NO_HOOKS).prologue is not None]
    assert hooked
    assert tuning.loads_launches(tuning.dumps_launches(hooked))


def test_a_measured_launch_draws_its_integer_side_operands_as_operands():
    """Measuring a launch draws group sizes for its group sizes alone: an integer side operand keeps its
    own shape."""
    bf16, int32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16), functools.partial(
        jax.ShapeDtypeStruct, dtype=jnp.int32)
    specs = [bf16((128, 64)), bf16((2, 64, 64)), int32((2,)), int32((128,))]
    arrays = grouped_matmul.Gmm().operands(specs, seed=0)
    assert [a.shape for a in arrays] == [spec.shape for spec in specs]
    assert int(jnp.sum(arrays[2])) == 128


def test_a_dense_product_runs_the_plain_tiled_kernel():
    """On Triton a dense product (no group sizes) runs `triton_dense`'s kernels, forward and both gradients,
    keyed apart from a one-group ragged call's: their launches are told `dense`."""
    x, w, ms, nw, b, u, s, ct = _draw(9)
    spec = CASES["dense norm and bias"]
    with generation.assumed("sm80"), tuning.seen() as launches:
        jax.jit(_loss(_dew(spec, interpret_on_cpu=True))).trace(x, w, ms, nw, b, u, s, ct).lower()
    told = {(type(launch).__name__, ba.arguments.get('dense')) for launch, ba in launches}
    assert {('Gmm', True), ('GmmTransposed', True), ('Tgmm', True)} <= told


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
    multiply in, and `ragged_dot_refusal` is the declaration's refusal."""
    declared = product.declared(compute, precision, outputs=outputs)
    assert declared.promoted is promoted
    assert (declared.refusal is None) is (refused is None)
    assert refused is None or refused in declared.refusal
    assert grouped_matmul.ragged_dot_refusal(compute, (), precision) == declared.refusal


def test_a_product_under_x64_or_from_a_complex_master_is_refused():
    """x64 widens the kernels' int32 group offsets; a complex master's gradient is one the kernels,
    summing in fp32, cannot write, at a real compute dtype as at any."""
    with jax.enable_x64(new_val=True):
        assert "x64" in product.declared(jnp.bfloat16, DEFAULT).refusal
    refusal = product.declared(jnp.bfloat16, DEFAULT, operands=(jnp.complex64,)).refusal
    assert refusal is not None and "fp32 does not hold" in refusal
    assert grouped_matmul.ragged_dot_refusal(jnp.bfloat16, (jnp.complex64,), DEFAULT) == refusal


def swiglu(gate, up):
    """A SwiGLU epilogue on a paired product's two fp32 accumulators, each rounded to bf16 first and the
    activated gate again, as `activations.gated_product` rounds them."""
    gate, up = gate.astype(jnp.bfloat16), up.astype(jnp.bfloat16)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up).astype(jnp.float32)


PAIRED_TILE = grouped_matmul.Tile(64, 128, 64, 4, 2)


def test_a_paired_product_is_its_two_products_combined_bitwise():
    """The two-source dense kernel (one `lhs` tile, two matrices, two accumulators) at a tile is the same
    kernel's two plain products at that tile, their fp32 accumulators combined by the same epilogue in
    XLA, bit for bit, with rows, depth and width no tile divides."""
    rows, depth, width = 200, 300, 140
    keys = jax.random.split(jax.random.key(7), 3)
    shapes = ((rows, depth), (depth, width), (depth, width))
    lhs, gate, up = (jax.random.normal(key, shape).astype(jnp.bfloat16)
                     for key, shape in zip(keys, shapes, strict=True))
    options = {"tile": PAIRED_TILE, "transpose_rhs": False, "interpret": True}
    [paired] = triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",), **options)
    [gate_acc] = triton_dense.gmm(lhs, gate, out_dtypes=("float32",), **options)
    [up_acc] = triton_dense.gmm(lhs, up, out_dtypes=("float32",), **options)
    expected = swiglu(gate_acc, up_acc).astype(jnp.bfloat16)
    assert paired.shape == (rows, width) and paired.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(paired)))
    np.testing.assert_array_equal(np.asarray(jax.lax.bitcast_convert_type(paired, jnp.uint16)),
                                  np.asarray(jax.lax.bitcast_convert_type(expected, jnp.uint16)))


@pytest.mark.parametrize("generation_name", ["sm80", "sm86", "sm90"])
def test_a_paired_dense_product_lowers_for_cuda(generation_name):
    """The two-source dense kernel lowers for CUDA on the CPU, its pallas_call named."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((1021, 200)), bf16((200, 300)), bf16((200, 300)))

    def paired(lhs, gate, up):
        return triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",),
                                tile=PAIRED_TILE, transpose_rhs=False, interpret=False)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(paired).trace(*args).lower(lowering_platforms=("cuda",))
    assert "pallas_triton_dense_dot" in lowered.as_text()


def test_a_paired_product_without_an_epilogue_is_refused():
    lhs = jnp.zeros((64, 64), jnp.bfloat16)
    with pytest.raises(ValueError):
        triton_dense.gmm(lhs, lhs, paired=lhs, out_dtypes=("bfloat16",), tile=PAIRED_TILE,
                         transpose_rhs=False, interpret=True)


def test_a_hook_that_broadcasts_a_side_operand_lowers_for_hopper_through_triton():
    """Mosaic's layout inference finds no layout for a side operand given back with its dimension of
    one, so the Mosaic launches refuse such a hook: a Hopper call's every launch with one, the forward's
    and the kernel gradient's with its prologue, lowers through Triton's, beside the gradients' plain
    launches, which may be Mosaic's (GMMGATE-h100-8)."""
    drawn = [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in _draw(0)]
    dew = _drop(_dew(CASES["broadcast axes"]))

    def loss(*drawn):
        return sum(jnp.sum(out.astype(jnp.float32)) for out in dew(*drawn))

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh), tuning.seen() as launches:
        jax.jit(jax.grad(loss, argnums=range(len(drawn)))).trace(*drawn).lower(lowering_platforms=("cuda",))
    broadcast = [(launch, ba) for launch, ba in launches
                 if ba.arguments['hooks'].prologue_broadcast or ba.arguments['hooks'].epilogue_broadcast]
    assert {type(launch) for launch, _ in broadcast} >= {grouped_matmul.Gmm, grouped_matmul.Tgmm}
    assert not any(isinstance(launch, grouped_matmul._Mosaic) for launch, _ in broadcast)

