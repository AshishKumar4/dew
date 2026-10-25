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
from reference_error import assert_as_exact, with_gradients
from tuned import each_config

from dew.nn.kernels import generation, grouped_matmul, tuning
from dew.nn.kernels.templates import mosaic, product
from dew.nn.kernels.templates.gemm import fused, mosaic_gpu, mosaic_tpu, template, triton_dense

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
"""A tile two consumer warpgroups compute, on one multiprocessor (the interpreter models no cluster)."""


MOSAIC_RAN = {'mosaic_gpu'} if jax.default_backend() == 'cpu' else {'mosaic_gpu', 'triton'}
"""The grouped matmul's kernels a call pinned to Mosaic reports: a GPU process also traces the CPU branch
of its platform choice, which interprets Triton's and never runs there (`grouped_matmul.launched`)."""


def _kernels(taken) -> set:
    return {route.kernel for route in taken if route.choice == 'grouped_matmul'}


@runs("sm90")
@pytest.mark.parametrize("case", ["norm and bias", "gated prologue", "cast prologue", "row scaled"])
def test_a_cooperative_tile_holds_check_1_with_its_hooks(case):
    """The Hopper launches pinned to a cooperative tile, interpreted: each consumer rewrites and multiplies
    its half of the stage's rows and computes their epilogue, as exact as the plain program from float64
    truth; the Mosaic kernels ran."""
    spec = CASES[case]
    pinned = {grouped_matmul.MosaicGmm: COOPERATIVE, grouped_matmul.MosaicTgmm: COOPERATIVE}
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned),
          generation.routes() as taken):
        dew = jax.jit(_drop(_dew(spec, interpret_on_cpu=True)))
        assert_as_exact(dew, jax.jit(_drop(_plain(spec))), _draw, f"{case} cooperative",
                        seeds=SEEDS, truth_fn=_drop(_plain(spec, truth=True)))
    assert _kernels(taken) == MOSAIC_RAN


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
        assert_as_exact(dew, jax.jit(_gradients(_plain(spec))), _gradient_draw(spec),
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
                    assert_as_exact(dew, jax.jit(plain), draw, f"{kernel.__name__} at {config}", seeds=2,
                                    truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_stacked_matrix_is_read_in_place_at_a_layer(generation_name):
    """A layer of a stacked `[layers, groups, depth, width]` matrix picked by an index (`gemm`'s `layer`):
    as exact as the plain program's product with that layer's matrices from float64 truth, the stack's
    gradient included (zero at every other layer), and the launch reads the whole stack in place, its
    matrix operand the stack's every group, no slice of it made. The gradients run eagerly, as the
    Mosaic interpreters' ordered callbacks need."""
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
        with tuning.seen() as launches:
            jax.jit(dew).trace(*draw(0)[:2])
    read = {ba.arguments['kernel'].shape for op, ba in launches if isinstance(op, grouped_matmul.MosaicGmm)}
    assert read == {(layers * groups, depth, width)}


@runs("v5e")
def test_a_tpu_runs_an_exactly_summed_product_in_xla():
    """A kernel gradient into fp32 above the default precision adds each step's products apart, which
    the TPU's launches do not promise (their MXUs add within a step): XLA runs it, the refusal reported."""
    x, _, _, _, _, _, _, dy = _draw(0)
    with generation.assumed("v5e"), generation.routes() as taken:
        template.gemm_t(x, dy.astype(jnp.bfloat16), jnp.asarray(SIZES), out_dtype=jnp.float32,
                        precision="highest", interpret_on_cpu=True)
    [found] = [route for route in taken if route.choice == 'grouped_matmul']
    assert found.kernel == 'xla' and "MXUs" in found.refusal.detail

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

    with generation.assumed("sm80"), tuning.seen() as launches:
        truth = functools.partial(plain, wide=jnp.float64, out=jnp.float64)
        assert_as_exact(jax.jit(forward(dew)), jax.jit(forward(plain)), draw, "dense 65x40x100", seeds=SEEDS,
                        truth_fn=forward(truth))
        assert_as_exact(jax.jit(with_gradients(dew)), jax.jit(with_gradients(plain)), draw,
                        "dense 65x40x100 gradients", seeds=SEEDS, truth_fn=with_gradients(truth))
    ran = {(type(launch).__name__, ba.arguments.get('trans_rhs'), ba.arguments.get('dense'))
           for launch, ba in launches}
    assert {('Gmm', transpose_rhs, True), ('Tgmm', None, True)} <= ran


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

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="triton"), tuning.seen() as launches:
        assert_as_exact(jax.jit(with_gradients(dew)), jax.jit(with_gradients(plain)), draw, "dense depth 96",
                        seeds=SEEDS,
                        truth_fn=with_gradients(functools.partial(plain, wide=jnp.float64, out=jnp.float64)))
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
    forward product and the kernel gradient are the same bit for bit, whatever rows are a multiple of,
    through Triton's launches (`generation.chosen`), the one implementation that takes both layouts
    (Mosaic's lowering has no transpose)."""
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

    with generation.assumed(generation_name), generation.chosen(grouped_matmul="triton"):
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


@runs("sm80")
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
    on each loaded tile, its row statistic in XLA before the product (the route says so), with the
    unfused program's value: as exact as it, from float64."""
    sizes = jnp.asarray(SIZES)

    def draw(seed):
        x, w, _, nw, *_ = _draw(seed)
        return x, w, nw

    def fusing(x, w, weight):
        return fused.fusible_gemm(rms_norm(x, weight), w, sizes, interpret_on_cpu=True).astype(jnp.bfloat16)

    def unfused(x, w, weight):
        return template.gemm(rms_norm(x, weight), w, sizes, out_dtype=jnp.bfloat16)


    with generation.assumed(generation_name), generation.routes() as taken:
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "normed operand", seeds=SEEDS,
                        truth_fn=functools.partial(_normed_truth, sizes))
    assert {route.kernel for route in taken if route.choice == 'gemm_prologue'} == {'fused'}


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
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh), generation.routes() as taken:
        jax.jit(fuser.fuse(fusing)).trace(x, w, nw).lower(lowering_platforms=("cuda",))
    assert 'mosaic_gpu' in {route.kernel for route in taken if route.choice == 'grouped_matmul'}
    assert {route.kernel for route in taken if route.choice == 'gemm_prologue'} == {'fused'}


def _three_gemms(sizes):
    """A fused producer's product three ways: under `fuser.fuse`, unfused, and its float64 truth."""
    def fused_gemm(operand, w):
        return fused.fusible_gemm(operand, w, sizes, interpret_on_cpu=True)

    def plain_gemm(operand, w):
        return template.gemm(operand, w, sizes, out_dtype=jnp.float32)

    def truth_gemm(operand, w):
        matrix = w.astype(jnp.bfloat16).astype(operand.dtype)
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

    with generation.assumed("sm80"), generation.routes() as taken:
        assert_as_exact(jax.jit(loss(fused_gemm, fuser.fuse)), jax.jit(loss(plain_gemm)), draw,
                        f"{producer.__name__} gradients", seeds=SEEDS, truth_fn=loss(truth_gemm))
    assert {route.kernel for route in taken if route.choice == 'gemm_prologue'} == {'fused'}


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

    with generation.assumed("sm80"), generation.routes() as taken:
        assert_as_exact(jax.jit(derivative(fused_gemm, fuser.fuse)), jax.jit(derivative(plain_gemm)), draw,
                        f"normed operand {composition}", seeds=SEEDS, truth_fn=derivative(truth_gemm))
    assert {route.kernel for route in taken if route.choice == 'gemm_prologue'} == {'fused'}


@runs("sm80")
def test_chosen_routes_each_fused_stage_through_xla_or_insists_on_the_kernels():
    """`generation.chosen(gemm_epilogue='xla', gemm_prologue='xla')` runs both stages in XLA beside the
    product (each route says so) with the fused program's value, as exact as the unfused program from
    float64; `gemm_epilogue='fused'` raises for a consumer the kernels cannot take rather than run XLA."""
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

    with (generation.assumed("sm80"), generation.chosen(gemm_epilogue="xla", gemm_prologue="xla"),
          generation.routes() as taken):
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "both stages in XLA",
                        seeds=SEEDS, truth_fn=truth)
    assert {(route.choice, route.kernel) for route in taken if route.choice.startswith('gemm_')} == {
        ('gemm_epilogue', 'xla'), ('gemm_prologue', 'xla')}
    x, w, _, _, b, u, *_ = _draw(0)

    def positioned(x, w, b, r):
        return positional(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), b, r)

    with generation.assumed("sm80"), generation.chosen(gemm_epilogue="fused"), pytest.raises(ValueError):
        jax.jit(fuser.fuse(positioned)).trace(x, w, b, u[:, :WIDTH])


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


    with generation.assumed("sm80"), generation.routes() as taken:
        assert_as_exact(jax.jit(fuser.fuse(fusing)), jax.jit(unfused), draw, "one normed row", seeds=SEEDS,
                        truth_fn=functools.partial(_normed_truth, sizes))
    assert {route.kernel for route in taken if route.choice == 'gemm_prologue'} == {'fused'}


@runs("sm80")
@pytest.mark.parametrize(("consumer", "kernel", "code"), [(bias_cast, 'fused', None),
                                                          (positional, 'xla', 'reads_position')],
                         ids=("tiled", "positional"))
def test_a_consumer_the_kernels_cannot_take_reports_its_route(consumer, kernel, code):
    """A consumer fused into the epilogue reports so; one left to XLA reports why, so a model that loses
    its fusion shows where (`generation.route`'s 'gemm_epilogue')."""
    sizes = jnp.asarray(SIZES)
    x, w, _, _, b, u, *_ = _draw(0)

    def fusing(x, w, b, r):
        return consumer(fused.fusible_gemm(x, w, sizes, interpret_on_cpu=True), b, r)

    with generation.assumed("sm80"), generation.routes() as taken:
        jax.jit(fuser.fuse(fusing)).trace(x, w, b, u[:, :WIDTH])
    [found] = [route for route in taken if route.choice == 'gemm_epilogue']
    assert found.kernel == kernel and (found.refusal is None if code is None else found.refusal.code == code)


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


@runs("v5e")
@pytest.mark.parametrize("product", ["gemm", "gemm_t"])
def test_a_tpu_lowers_xlas_program_for_a_call_no_launch_takes(product):
    """A product of widths no TPU block divides lowers for a TPU through XLA's program, the refusal
    reported, its kernel branches traced on the way and none raising. A CPU process's: a GPU's interprets
    its CPU branch with Triton's launch whatever the target (`grouped_matmul.launched`)."""
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
    with assumed, mesh, generation.routes() as taken:
        jax.jit(run).trace(x, other).lower(lowering_platforms=("tpu",))
    # The TPU branch's and the CPU branch's each.
    found = [route for route in taken if route.choice == 'grouped_matmul']
    assert found and all(route.kernel == 'xla' and "widths" in route.refusal.detail for route in found)


NARROW_GRADIENT_CASES = ("norm and bias", "gated prologue", "cast prologue", "broadcast axes",
                         "dense norm and bias")
"""The gradient cases whose cotangents are 16-bit, so each product of their reverse mode multiplies
16-bit operands."""


@runs("v5e")
@pytest.mark.parametrize("case", NARROW_GRADIENT_CASES)
def test_the_tpu_launches_gradients_hold_check_1(case):
    """Every operand's cotangent of a bf16 matrix at the default precision, where no product adds step by
    step, so the forward and both gradients run the TPU's launches, interpreted: as exact as JAX's
    derivative of the plain program from float64 truth."""
    spec = CASES[case]

    def drawn(seed):
        x, w, *rest = _draw(seed)
        return (x, w.astype(jnp.bfloat16), *rest)

    with (generation.assumed("v5e"), jax.default_matmul_precision("default"),
          generation.routes() as taken):
        assert_as_exact(_gradients(_dew(spec, interpret_on_cpu=True)), jax.jit(_gradients(_plain(spec))),
                        _gradient_draw(spec, drawn), f"{case} gradients on the TPU's launches", seeds=SEEDS,
                        truth_fn=_gradients(_plain(spec, truth=True)))
    assert _kernels(taken) == {'mosaic_tpu'}

@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
@pytest.mark.parametrize("launch,trans_rhs", [(grouped_matmul.Gmm(), False), (grouped_matmul.Gmm(), True),
                                              (grouped_matmul.Tgmm(), None)],
                         ids=["Gmm", "Gmm-transposed", "Tgmm"])
def test_the_hooked_dense_launches_lower_for_cuda(launch, trans_rhs, generation_name):
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
    interpreted on the CPU, whose dots sum in IEEE fp32 either way, through one launch at one tile it is
    the default path's bit for bit."""
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.bfloat16)
    sizes = jnp.asarray(SIZES)
    # One tile both ways, a step of EXACT_STEP rows, so the order of the sums is the same.
    tiles = {grouped_matmul.Tgmm: grouped_matmul.Tile(64, 128, 64, 4, 2),
             grouped_matmul.MosaicTgmm: mosaic_gpu.Tiles(64, 128, 64, 2, 1)}
    launches = ("triton", "mosaic_gpu") if generation_name == "sm90" else ("triton",)
    for launch, group_sizes in itertools.product(launches, (sizes, None)):
        with (generation.assumed(generation_name), tuning.pinned(tiles),
              generation.chosen(grouped_matmul=launch)):
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
    pinned = {grouped_matmul.MosaicGmm: COOPERATIVE, grouped_matmul.MosaicTgmm: COOPERATIVE}
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"), tuning.pinned(pinned),
          generation.routes() as taken):
        assert_as_exact(_gradients(_dew(spec, interpret_on_cpu=True)), jax.jit(_gradients(_plain(spec))),
                        _gradient_draw(spec), f"{case} cooperative gradients", seeds=SEEDS,
                        truth_fn=_gradients(_plain(spec, truth=True)))
    assert _kernels(taken) == MOSAIC_RAN


@SUMS_IN_IEEE
def test_a_cooperative_tiles_exact_sums_add_as_its_plain_sum_does():
    """`test_exact_sums_add_each_steps_product_as_the_plain_sum_does` at a cooperative tile: each consumer's
    running fp32 sum, the pipeline's carry, matches its accumulator's bit for bit on the CPU."""
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(ROWS, DEPTH)), jnp.bfloat16)
    dy = jnp.asarray(rng.normal(size=(ROWS, WIDTH)), jnp.bfloat16)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic_gpu"),
          tuning.pinned({grouped_matmul.MosaicTgmm: COOPERATIVE})):
        exact, default = (jax.jit(functools.partial(template.gemm_t, out_dtype=jnp.float32,
                                                    precision=precision, interpret_on_cpu=True))(
            x, dy, jnp.asarray(SIZES)) for precision in ("highest", "default"))
    np.testing.assert_array_equal(np.asarray(exact), np.asarray(default))


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


def test_a_launch_takes_no_pair():
    """The bundle's rings stream to one multiprocessor: no launch derives a pair, and a pair pinned on the
    forward is refused rather than launched."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((ROWS, DEPTH)), bf16((3, DEPTH, WIDTH)), jax.ShapeDtypeStruct((3,), jnp.int32))
    launch = grouped_matmul.MosaicGmm()
    with generation.assumed("sm90"):
        compiled = launch.space(launch.bind(*args, out_dtype="bfloat16", trans_rhs=False, interpret=False))
    assert compiled and not any(tile.cluster == 2 for tile in compiled)
    with pytest.raises(ValueError):
        mosaic_gpu.gmm(jnp.zeros((ROWS, DEPTH), jnp.bfloat16), jnp.zeros((3, DEPTH, WIDTH), jnp.bfloat16),
                       jnp.asarray([100, 100, 107]), tiles=mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2,
                                                                             cluster=2),
                       transpose_rhs=False, out_dtype=jnp.bfloat16, multiprocessors=2,
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
        # Two accumulators a consumer warpgroup: its 128 threads hold its share of the tile's columns, in
        # the registers a cooperative consumer has (232) or a lone warpgroup's (255). The generated tile
        # holds them; a tile that does not is no legal pin.
        [tile] = launch.fitting(exact, target)
        assert 2 * tile.block_m * tile.block_n // (128 * tile.warpgroups) < mosaic.CONSUMER_REGISTERS
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


def transposed_twice_nested(x, side):
    """`shifted`, its transpose two jitted calls deep."""
    return (_wide(x) + jax.jit(jax.jit(lambda s: s.T))(side)).astype(x.dtype)


@runs("sm80")
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
    taken = {(route.kernel, route.refusal is not None) for route in highest
             if route.choice == "grouped_matmul"}
    assert taken == {("xla", True)}
    assert {route.kernel for route in default if route.choice == "grouped_matmul"} == {"triton"}
    xla = jax.jit(lambda x, w: jax.lax.ragged_dot(x, w, jnp.asarray([64]), precision="highest",
                                                  preferred_element_type=jnp.float32))(x, w)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(xla))


@runs("sm80")
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


@runs("sm80")
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


@runs("sm80")
def test_a_dense_product_runs_the_plain_tiled_kernel():
    """On Triton (`generation.chosen`) a dense product (no group sizes) runs `triton_dense`'s kernels,
    forward and both gradients, apart from a one-group ragged call's: their launches are told `dense`."""
    spec = CASES["dense norm and bias"]
    with generation.assumed("sm80"), generation.chosen(grouped_matmul="triton"), tuning.seen() as launches:
        jax.jit(_gradients(_dew(spec, interpret_on_cpu=True))).trace(*_gradient_draw(spec)(9)).lower()
    told = {(type(launch).__name__, ba.arguments.get('trans_rhs'), ba.arguments.get('dense'))
            for launch, ba in launches}
    assert {('Gmm', False, True), ('Gmm', True, True), ('Tgmm', None, True)} <= told


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


def test_a_product_under_x64_or_from_a_complex_master_is_refused():
    """x64 widens the kernels' int32 group offsets; a complex master's gradient is one the kernels,
    summing in fp32, cannot write, at a real compute dtype as at any."""
    with jax.enable_x64(new_val=True):
        assert "x64" in product.declared(jnp.bfloat16, DEFAULT).refusal
    refusal = product.declared(jnp.bfloat16, DEFAULT, operands=(jnp.complex64,)).refusal
    assert refusal is not None and "fp32 does not hold" in refusal


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



@pytest.mark.parametrize("columns", [128, 300, 384], ids=lambda n: f"{n}-columns")
def test_an_interpreted_fp32_product_at_highest_rounds_its_exact_sum_once(columns):
    """A promoted fp32 product issued off a GPU (`product.interpreted`) accumulates in float64 and rounds
    once, at every shape, where XLA's CPU dot takes a less exact fp32 accumulation at a 128-aligned shape
    than at another. The operands are 12-bit integers at scales 2^-4 to 2^4, so every product and partial
    sum is a multiple of 2^-8 below 2^37, exact in float64 in any order and past fp32's 24 bits: the
    product is the exact sum rounded once to fp32, to the bit. That is the contract under test, the float64
    accumulation: an fp32 one would round the partial sums, so equality is no claim that fp32 accumulation
    is exact here. A 16-bit product keeps an fp32 sum."""
    generator = np.random.default_rng(0)

    def drawn(shape):
        scales = 2.0 ** generator.integers(-4, 5, shape)
        return (generator.integers(-2048, 2048, shape) * scales).astype(np.float32)

    lhs, rhs = jnp.asarray(drawn((128, 128))), jnp.asarray(drawn((128, columns)))
    declared = product.declared(jnp.float32, jax.lax.Precision.HIGHEST, outputs=(jnp.float32,))
    found = jax.jit(lambda a, b: product.interpreted(declared, a, b, (((1,), (0,)), ((), ()))))(lhs, rhs)
    exact = np.asarray(lhs, np.float64) @ np.asarray(rhs, np.float64)
    np.testing.assert_array_equal(np.asarray(found), exact.astype(np.float32))
    half = product.declared(jnp.bfloat16, None, outputs=(jnp.float32,))
    narrow = (lhs.astype(jnp.bfloat16), rhs.astype(jnp.bfloat16))
    out = product.interpreted(half, *narrow, (((1,), (0,)), ((), ())))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(jax.lax.dot_general(
        *narrow, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)))


def test_an_interpreted_fp32_sum_that_cancels_at_mixed_magnitudes_is_its_float64_sum():
    """2^24 * 2^24, 1 * 1 and -(2^24 * 2^24): each product is exact in fp32, and their float64 sum is 1 in
    every order (2^48 + 1 is a float64), where an fp32 accumulation that adds a large one first rounds the
    1 away, 0. The interpreted product accumulates in float64: 1 exactly."""
    lhs = jnp.asarray([[2.0 ** 24, 1.0, -(2.0 ** 24)]], jnp.float32)
    rhs = jnp.asarray([[2.0 ** 24], [1.0], [2.0 ** 24]], jnp.float32)
    declared = product.declared(jnp.float32, jax.lax.Precision.HIGHEST, outputs=(jnp.float32,))
    assert float(product.interpreted(declared, lhs, rhs, (((1,), (0,)), ((), ())))[0, 0]) == 1.0


HOOKED_RECEIPTS = {
    # (shape, case): (Mosaic ms, Triton ms), GMMSB-h100-main at eb378b58d (an H100's tools/bench_scoreboard).
    ("llama-3-8b gate_up", "plain"): (5.925, 10.191),
    ("llama-3-8b gate_up", "norm"): (8.171, 12.815),
    ("llama-3-8b gate_up", "bias"): (5.970, 9.322),
    ("qwen3-8b gate_up", "plain"): (4.919, 8.818),
    ("qwen3-8b gate_up", "norm"): (6.950, 11.036),
    ("qwen3-8b gate_up", "bias"): (4.996, 7.912),
}
"""The two Hopper launches' measured times for a dense projection, plain and with each hook the chooser
ranks right (the gated prologue joins once the model's facts rank it, row 1336)."""

@pytest.mark.parametrize(("shape", "case"), list(HOOKED_RECEIPTS), ids=lambda key: str(key))
def test_hoppers_chooser_against_the_hooked_receipts(shape, case):
    """The chooser's launch for a hooked dense projection on an H100 is the one measured faster."""
    width = {"llama-3-8b gate_up": 28672, "qwen3-8b gate_up": 24576}[shape]
    bf16, f32 = (functools.partial(jax.ShapeDtypeStruct, dtype=dtype)
                 for dtype in (jnp.bfloat16, jnp.float32))
    x, w = bf16((16384, 4096)), bf16((4096, width))
    hooks = {"plain": {},
             "norm": {"prologue": normed, "prologue_operands": ((f32((16384,)), ("rows",)),
                                                                 (f32((4096,)), ("depth",)))},
             "bias": {"epilogue": biased, "epilogue_operands": ((f32((width,)), ("columns",)),)}}[case]

    def product(x, w):
        return template.gemm(x, w, None, out_dtype=jnp.bfloat16, **hooks)

    with generation.assumed("sm90"), generation.routes() as taken:
        jax.jit(product).trace(x, w)
    mosaic, triton = HOOKED_RECEIPTS[(shape, case)]
    faster = "mosaic_gpu" if mosaic < triton else "triton"
    assert {route.kernel for route in taken if route.choice == 'grouped_matmul'} == {faster}
