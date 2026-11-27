"""A frozen bfloat16 matrix's lossless 13-bit window (`formats.Bf16Window`, row 5387): each weight its bits
again decoded, a group past its escapes refused by its site's name, and a Dense served its window as its
kernel (`quantization.bf16_windows`), its variables' kernels back bitwise (`decoded_bf16_windows`)."""

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from kernel_compile import compile_mosaic_kernels

from dew import quantization
from dew.nn import programs
from dew.nn.dense import Dense
from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.formats import WINDOW_ESCAPES, Bf16Window


def _bits(values) -> np.ndarray:
    return np.asarray(values, jnp.bfloat16).view(np.uint16)


def _drawn(shape, seed=5387) -> np.ndarray:
    """A normal bfloat16 weight draw (a trained projection's scale) beside the edges its window codes:
    signed zeros, subnormals and a tiny weight among its escapes, a group's floor `2^b` exactly, and a group
    anchored at the largest finite and holding infinities and NaNs (each its bits)."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.02, shape).astype(jnp.bfloat16)
    bits = weights.view(np.uint16)
    bits[0, 0], bits[1, 0], bits[2, 0], bits[3, 0] = 0x8000, 0x0000, 0x0001, 0x8005  # -0, +0, subnormals
    top = int(((bits[:128, 1] >> 7) & 0xFF).max())
    bits[5, 1] = (top - 14) << 7  # exactly 2^b, the window's floor
    bits[128:256, 2] = 0x7F00 | (bits[128:256, 2] & 0x807F)  # near the largest finite
    bits[128, 2], bits[129, 2], bits[130, 2] = 0x7F80, 0xFF80, 0x7FC1  # +inf, -inf, a NaN's payload
    return weights


def test_a_bf16_window_decodes_every_weight_to_its_bits():
    """Every weight of a window, decoded, is the weight it encoded, bit for bit, at 13 bits a weight."""
    weights = _drawn((512, 96))
    window = Bf16Window.of(weights).checked()
    assert window.shape == weights.shape
    np.testing.assert_array_equal(np.asarray(window.decoded()).view(np.uint16), weights.view(np.uint16))
    stored = sum(np.asarray(plane).size * 32 for plane in (window.words, window.meta))
    assert stored / weights.size == 13


def test_a_group_past_its_escapes_is_refused():
    """A group holding as many nonzeros below its window as it stores escapes (`WINDOW_ESCAPES`: tiny
    weights, 2^-20, beside 0.25) is held, each weight its bits again; one more is refused, where no window
    holds it."""
    weights = np.full((128, 4), 0.25, jnp.bfloat16)
    weights[:WINDOW_ESCAPES, 3] = np.asarray(2.0 ** -20, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(Bf16Window.of(weights).decoded()).view(np.uint16),
                                  weights.view(np.uint16))
    weights[WINDOW_ESCAPES, 3] = np.asarray(2.0 ** -20, jnp.bfloat16)
    with pytest.raises(ValueError):
        Bf16Window.of(weights, site="layers/q_proj")


def test_a_dense_served_its_window_is_its_kernels_product_and_its_kernel_comes_back():
    """A bfloat16 Dense whose kernel `bf16_windows` stores as its window computes what it computes over its
    kernel, bit for bit (its product of exact dyadic values, its bfloat16 bias added past it as Dense adds
    one), and `decoded_bf16_windows` gives its kernel back bitwise, the collection gone."""
    rng = np.random.default_rng(5387)
    kernel = (rng.integers(-64, 64, (256, 64)) / 16.0).astype(jnp.bfloat16)
    bias = (rng.integers(-8, 8, (64,)) / 4.0).astype(jnp.float32)
    x = jnp.asarray(rng.integers(-4, 4, (8, 256)), jnp.bfloat16)
    layer = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    variables = {"params": {"kernel": jnp.asarray(kernel), "bias": jnp.asarray(bias, jnp.bfloat16)}}
    windowed = quantization.bf16_windows(variables, sites=[""])
    assert "kernel" not in windowed["params"]
    np.testing.assert_array_equal(np.asarray(jax.jit(layer.apply)(windowed, x)).view(np.uint16),
                                  np.asarray(jax.jit(layer.apply)(variables, x)).view(np.uint16))
    back = quantization.decoded_bf16_windows(windowed)
    assert set(back) == {"params"}
    np.testing.assert_array_equal(np.asarray(back["params"]["kernel"]).view(np.uint16),
                                  kernel.view(np.uint16))


def _two_steps(seed: int = 5387) -> np.ndarray:
    """`_drawn`'s edges in a first 256-deep step and, in a second, a draw 2**-20 smaller (other bases) with
    three weights of one group below its window (its escapes): a launch reading step zero's words or meta
    again decodes the second wrong."""
    second = _drawn((256, 64), seed + 1)
    finite = np.isfinite(second.astype(np.float32))
    second = np.where(finite, (second.astype(np.float32) * np.float32(2.0 ** -20)), 0.0).astype(jnp.bfloat16)
    second[0:3, 5] = np.asarray(2.0 ** -50, jnp.bfloat16)
    return np.concatenate([_drawn((256, 64), seed), second])


@pytest.mark.parametrize("grouped", [False, True], ids=["dense", "grouped"])
def test_a_window_product_decoded_in_its_stages_is_its_weights_product(grouped):
    """A window over two 256-deep steps (`_two_steps`: signed zeros, subnormals, escapes, a group's floor, a
    group near the largest finite, infinities and NaNs, and a second step of other bases and escapes) times
    the rows of the identity, each output entry one weight times one: its product on the template's route,
    each stage decoded in place (interpreted), and on XLA's, the decoded matrix's, each XLA's plain product
    of the weights themselves, bit for bit (a NaN where it is NaN), dense and grouped (two groups of
    finite weights, the second's reversed in its columns, each group's rows the even or the odd depths, so
    each reads both steps); the route each took reported."""
    weights = _two_steps()
    depth = weights.shape[0]
    rows = np.eye(depth, dtype=jnp.bfloat16)
    sizes = np.asarray([depth // 2, depth // 2], np.int32)
    if grouped:
        # Finite: XLA's ragged dot reads every group's matrix for every row, so one group's infinity is NaN
        # in the others' rows (row 13562); the dense case holds the non-finite edges.
        weights = np.where(np.isfinite(weights.astype(np.float32)), weights, np.asarray(0.0, jnp.bfloat16))
        stacked = np.stack([weights, weights[:, ::-1]])
        rows = np.concatenate([rows[0::2], rows[1::2]])
        window = Bf16Window.of(stacked)
        half = depth // 2
        want = np.concatenate([
            jax.lax.dot(jnp.asarray(rows[at * half:(at + 1) * half]), jnp.asarray(stacked[at]),
                        preferred_element_type=jnp.float32) for at in range(2)])
    else:
        window = Bf16Window.of(weights)
        want = jax.lax.dot(jnp.asarray(rows), jnp.asarray(weights), preferred_element_type=jnp.float32)
    for kernel in ("mosaic", "xla"):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel),
              generation.routes() as seen):
            got = jax.jit(lambda rows: gemm.window_product(
                rows, window, jnp.asarray(sizes) if grouped else None, out_dtype=jnp.float32,
                interpret_on_cpu=True))(jnp.asarray(rows))
        assert any(route.kernel == kernel for route in seen), (kernel, seen)
        got, held = np.asarray(got), np.asarray(want)
        # A column holding an infinity is NaN in every row (its zeros times it); every other entry its bits.
        np.testing.assert_array_equal(np.isnan(got), np.isnan(held), err_msg=kernel)
        finite = ~np.isnan(held)
        np.testing.assert_array_equal(got[finite].view(np.uint32), held[finite].view(np.uint32),
                                      err_msg=kernel)


def test_a_window_products_rows_cotangent_keeps_an_infinite_weight():
    """A window holding one infinite weight, its product's cotangent of ones pulled back on the template's
    route (interpreted) and XLA's: the rows' cotangent is XLA's plain product's of the weights (the
    infinity's depth infinite, no NaN a split cotangent's zero parts would make), bit for bit."""
    weights = np.full((256, 64), 0.5, jnp.bfloat16)
    # Its group's window anchored at the largest finite binade, which an infinity's group holds.
    weights[:128, 3] = np.asarray(2.0 ** 126, jnp.bfloat16)
    weights[7, 3] = np.asarray(np.inf, jnp.bfloat16)
    window = Bf16Window.of(weights)
    rows = jnp.ones((8, 256), jnp.bfloat16)
    cotangent = jnp.ones((8, 64), jnp.float32)
    [want] = jax.vjp(lambda rows: jax.lax.dot(rows, jnp.asarray(weights), preferred_element_type=jnp.float32),
                     rows)[1](cotangent)
    for kernel in ("mosaic", "xla"):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel):
            [got] = jax.vjp(lambda rows: gemm.window_product(rows, window, out_dtype=jnp.float32,
                                                             interpret_on_cpu=True), rows)[1](cotangent)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32),
                                      err_msg=kernel)


def test_a_window_holds_a_group_of_the_least_normal_binades():
    """A group of the least normal bfloat16 binades (`0x0080`, `0x0100`, `0x0180`), fewer than its window's
    fifteen from the subnormals, takes its window from the least base, its weights decoded to their bits."""
    bits = np.tile(np.asarray([0x0080, 0x0100, 0x0180, 0x8080], np.uint16), (32, 2))
    weights = bits.view(jnp.bfloat16).reshape(128, 2)
    window = Bf16Window.of(weights)
    np.testing.assert_array_equal(np.asarray(window.decoded()).view(np.uint16), weights.view(np.uint16))


def test_a_window_launch_takes_its_staged_matrix_and_refuses_its_rows_swapped():
    """A window's launch reads its columns' meta halves beside its words (`gmm`'s `scales`): its staged
    matrix of a 256-deep step is taken, and a swapped call (its matrix the products' rows, which stream no
    columns' meta) is refused where the same call of a ternary matrix is taken (an H100's warpgroup MMA)."""
    from dew.nn.kernels.templates.gemm import packed, pricing, tiling

    tiles = tiling.Tiles(128, 64, 256, 2, 1, warpgroups=2, family="wgmma")
    with generation.assumed("sm90"):
        family = pricing.family_of(tiles, jnp.bfloat16)
    words = jax.ShapeDtypeStruct((1, 64, 256 // 128 * 48), jnp.uint32)
    packed.word_taken(words, "window", tiles, transpose_rhs=True, prologue=None, paired=None,
                      transpose_lhs=False, depth=256, width=64)
    for fmt, per_group, group in (("ternary", 16, 256), ("window", 48, 128)):
        rows = jax.ShapeDtypeStruct((128, 256 // group * per_group), jnp.uint32)
        swapped = functools.partial(packed.word_taken, words, fmt, tiles, transpose_rhs=True, prologue=None,
                                    paired=None, transpose_lhs=False, depth=256, width=64, lhs=rows,
                                    family=family, dense=True, packed_rows=True)
        if fmt == "ternary":
            swapped()
        else:
            with pytest.raises(ValueError):
                swapped()


@pytest.mark.parametrize("gpu", [pytest.param("sm80", marks=lowers("sm80")),
                                 pytest.param("sm90", marks=lowers("sm90"))])
def test_a_window_product_compiles_spill_free(gpu: str):
    """A window's Mosaic launch, each stage decoded in place, compiles spill-free on a compile-only sm80
    and sm90 at a Qwen3-4B projection's shape (row 5387)."""
    if generation.compile_only_gpus(gpu):
        rows = jax.ShapeDtypeStruct((4096, 2560), jnp.bfloat16)
        sample = Bf16Window.of(np.ones((2560, 9728), jnp.bfloat16))
        spec = jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), sample)

        def compiled(a, w):
            return gemm.window_product(a, w, out_dtype=jnp.bfloat16)

        with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"):
            records = compile_mosaic_kernels(
                [KernelCase(f"window_{gpu}", lambda fn=compiled, x=rows, w=spec: (fn, (x, w)))], gpu)
        for record in records:
            assert not record.get("error"), record
            assert record["kernels"], record
            assert all(kernel.get("spilled") == 0 for kernel in record["kernels"]), record


def test_a_float32_dense_served_its_window_computes_as_its_kernel():
    """A Dense computing in float32 over a bfloat16 kernel (its dtype float32, its inputs `1 + 2**-9` that
    bfloat16 rounds away) served its window computes what it computes over its kernel, bit for bit: a
    window multiplies bfloat16 rows alone (`gemm.window_product`), so the layer multiplies its weights as
    its kernel there."""
    rng = np.random.default_rng(5388)
    kernel = (rng.integers(-64, 64, (128, 64)) / 16.0).astype(jnp.bfloat16)
    x = jnp.full((4, 128), 1 + 2.0 ** -9, jnp.float32)
    layer = Dense(64, dtype=jnp.float32, param_dtype=jnp.bfloat16, use_bias=False)
    variables = {"params": {"kernel": jnp.asarray(kernel)}}
    plain = jax.jit(layer.apply)(variables, x)
    served = jax.jit(layer.apply)(quantization.bf16_windows(variables, sites=[""]), x)
    np.testing.assert_array_equal(np.asarray(served).view(np.uint32), np.asarray(plain).view(np.uint32))


def test_a_dense_of_its_own_promotion_served_its_window_computes_as_its_kernel():
    """A Dense of no dtype over bfloat16 inputs and kernel and a float32 bias (`1 + 2**-9`, which bfloat16
    rounds away; flax's promotion computes in float32), and one boxing its kernel with its logical axes
    (`kernel_axes`, `nn.LogicallyPartitioned`), each served its window: each computes what it computes over
    its kernel, bit for bit."""
    rng = np.random.default_rng(5389)
    kernel = (rng.integers(-64, 64, (128, 64)) / 16.0).astype(jnp.bfloat16)
    x = jnp.asarray(rng.integers(-4, 4, (4, 128)), jnp.bfloat16)
    promoted = Dense(64, param_dtype=jnp.bfloat16)
    bias = jnp.full((64,), 1 + 2.0 ** -9, jnp.float32)
    variables = {"params": {"kernel": jnp.asarray(kernel), "bias": bias}}
    boxed = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, kernel_axes=("embed", "mlp"))
    drawn = boxed.init(jax.random.key(0), x)
    for layer, held in ((promoted, variables), (boxed, drawn)):
        plain = jax.jit(layer.apply)(held, x)
        served = jax.jit(layer.apply)(quantization.bf16_windows(held, sites=[""]), x)
        assert served.dtype == plain.dtype
        np.testing.assert_array_equal(np.asarray(served, np.float32).view(np.uint32),
                                      np.asarray(plain, np.float32).view(np.uint32))


def test_a_window_products_rows_cotangent_keeps_a_cancellation():
    """A window of `+1` and `-1` columns, its product's cotangent `[1 + 2**-12, 1]` pulled back on the
    template's route, XLA's and the one the selector takes at the highest precision: the rows' cotangent is
    `2**-12` at each depth, the
    cotangent's difference read whole (a product rounding it to TF32's 10 bits cancels it to zero)."""
    weights = np.zeros((256, 64), jnp.bfloat16)
    weights[:, 0], weights[:, 1] = 1.0, -1.0
    window = Bf16Window.of(weights)
    rows = jnp.ones((2, 256), jnp.bfloat16)
    cotangent = jnp.zeros((2, 64), jnp.float32).at[:, 0].set(1 + 2.0 ** -12).at[:, 1].set(1.0)
    for kernel in ("mosaic", "xla", None):
        chosen = contextlib.nullcontext() if kernel is None else generation.chosen(grouped_matmul=kernel)
        with jax.default_matmul_precision("highest"), generation.assumed("sm90"), chosen:
            [got] = jax.vjp(lambda rows: gemm.window_product(rows, window, out_dtype=jnp.float32,
                                                             interpret_on_cpu=True), rows)[1](cotangent)
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.full((2, 256), 2.0 ** -12, np.float32),
                                      err_msg=kernel)


def test_a_bf16_dense_with_a_narrow_bias_served_its_window_rounds_as_its_kernel():
    """A bfloat16 Dense whose bias is stored bfloat16 (`2**-8`) over a kernel column `[1, 2**-8, 0, ...]`
    and inputs of ones: its product rounds to bfloat16 before the bias adds, as over its kernel (two ties to
    even, `1`), not once with it (`1 + 2**-7`), served its window too, bit for bit."""
    kernel = np.zeros((128, 64), jnp.bfloat16)
    kernel[0, 0], kernel[1, 0] = 1.0, 2.0 ** -8
    layer = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    variables = {"params": {"kernel": jnp.asarray(kernel), "bias": jnp.full((64,), 2.0 ** -8, jnp.bfloat16)}}
    x = jnp.ones((2, 128), jnp.bfloat16)
    plain = jax.jit(layer.apply)(variables, x)
    served = jax.jit(layer.apply)(quantization.bf16_windows(variables, sites=[""]), x)
    np.testing.assert_array_equal(np.asarray(served).view(np.uint16), np.asarray(plain).view(np.uint16))


def test_a_bf16_dense_stating_a_narrowing_algorithm_served_its_window_computes_as_its_kernel():
    """A bfloat16 Dense stating a dot algorithm that narrows its operands (`F16_F16_F32`: a bfloat16 input of
    `2**-30` is zero in fp16) over a unit kernel, and one stating `F32_F32_F32` under a run's narrowing one
    (the window's dot reading the run's), compute what they compute over their kernel, served their window
    too, bit for bit, or refuse alike where the backend runs no such algorithm (a CPU): the window's product
    multiplies its operands whole, so each layer's own product runs."""
    kernel = np.zeros((128, 64), jnp.bfloat16)
    kernel[0, 0] = 1.0
    variables = {"params": {"kernel": jnp.asarray(kernel)}}
    x = jnp.full((2, 128), 2.0 ** -30, jnp.bfloat16)
    narrowing = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False,
                      precision=jax.lax.DotAlgorithmPreset.F16_F16_F32)
    whole = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False,
                  precision=jax.lax.DotAlgorithmPreset.F32_F32_F32)
    for layer, run in ((narrowing, None), (whole, "F16_F16_F32")):
        with jax.default_matmul_precision(run):
            outs = []
            for held in (variables, quantization.bf16_windows(variables, sites=[""])):
                try:
                    outs.append(np.asarray(jax.jit(layer.apply)(held, x)).view(np.uint16))
                except ValueError as refused:
                    outs.append(type(refused))
        if isinstance(outs[0], np.ndarray):
            np.testing.assert_array_equal(outs[1], outs[0])
        else:
            assert outs[1] is outs[0]


@pytest.mark.parametrize(("compute", "rounded"), [(jnp.float32, False), (jnp.bfloat16, True)],
                         ids=["fp32 compute", "bf16 compute"])
def test_gemm_over_a_window_computes_in_its_calls_dtype(compute, rounded):
    """`gemm.gemm` over a bfloat16 window computes in its call's `compute` (row 13686): fp32 rows of
    1 + 2^-9 under fp32 compute keep their low bit (the decoded kernel's ordinary product), under bf16
    compute read as 1 (the window's product), each output its exact oracle: its kernel sixteenths of small
    integers, every sum exact in float32 (testing.md's oracle rule)."""
    rng = np.random.default_rng(13686)
    kernel = (rng.integers(-64, 64, (128, 64)) / 16.0).astype(np.float64)
    x = np.full((8, 128), 1 + 2.0 ** -9)
    window = Bf16Window.of(jnp.asarray(kernel, jnp.bfloat16)).checked()
    served = gemm.gemm(jnp.asarray(x, jnp.float32), window, None, out_dtype=jnp.float32, compute=compute)
    oracle = (np.ones_like(x) if rounded else x) @ kernel
    np.testing.assert_array_equal(np.asarray(served, np.float64), oracle)


def test_gemm_over_a_window_sums_exactly_where_its_call_asks():
    """`gemm.gemm` over a bfloat16 window asked above the default precision into fp32 (its partial sums
    promoted, `templates.product.declared`) takes the ordinary product's launch summing exactly, on a
    Hopper's template (interpreted): the window's own launch promotes none (row 13686)."""
    kernel = jnp.asarray(_drawn((256, 64)), jnp.bfloat16)
    window = Bf16Window.of(jnp.where(jnp.isfinite(kernel), kernel, 0).astype(jnp.bfloat16)).checked()
    rows = jax.ShapeDtypeStruct((128, 256), jnp.bfloat16)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          jax.default_matmul_precision("highest"), tuning.seen() as seen):
        jax.eval_shape(lambda x: gemm.gemm(x, window, None, out_dtype=jnp.float32, interpret_on_cpu=True),
                       rows)
    assert [call.arguments["exact_sums"] for _, call in seen.taken
            if isinstance(call.op, grouped_matmul.MosaicGmm) and call.arguments["interpret"]] == [True]


def _dot_precisions(fn, *args) -> list:
    """The precision every dot of `fn`'s program states, its nested programs' too (`programs.equations`)."""
    return [eqn.params["precision"] for eqn in programs.equations(jax.make_jaxpr(fn)(*args).jaxpr)
            if eqn.primitive is jax.lax.dot_general_p]


@pytest.mark.parametrize(("run", "stated", "algorithm"),
                         [("F16_F16_F32", None, jax.lax.DotAlgorithmPreset.F16_F16_F32),
                          ("F16_F16_F32", jax.lax.Precision.DEFAULT, None),
                          ("F32_F32_F32", jax.lax.Precision.DEFAULT, None)],
                         ids=["omitted: the run's narrowing algorithm", "the default over a narrowing run",
                              "the default over a whole run"])
def test_gemm_over_a_window_states_the_algorithm_its_call_and_run_declare(run, stated, algorithm):
    """`gemm.gemm` over a bfloat16 window of bf16 rows under a run's algorithm states to XLA the algorithm
    the call declares (row 13686's review): the run's where the call states no precision, none where it
    states the default (its level overriding the run's, a narrowing one's or a whole one a TPU does not
    run), on its decoded fallback (the window's own dot reads the run's)."""
    kernel = jnp.zeros((128, 64), jnp.bfloat16).at[0, 0].set(1.0)
    x = jnp.full((2, 128), 2.0 ** -30, jnp.bfloat16)
    with jax.default_matmul_precision(run):
        found = _dot_precisions(lambda x: gemm.gemm(x, Bf16Window.of(kernel).checked(), None,
                                                    out_dtype=jnp.float32, precision=stated), x)
    algorithms = {each for each in found
                  if isinstance(each, jax.lax.DotAlgorithmPreset | jax.lax.DotAlgorithm)}
    assert found and algorithms == ({algorithm} if algorithm else set()), found


def test_a_window_of_another_encoding_is_refused_however_it_is_made():
    """A window whose form is another encoder's is refused where it is made, before any trace reads it
    (row 13686's review): replaced into a window, or unflattened from a checkpoint's leaves, its words never
    reach a product, jitted or not, on the window's own product or its decoded fallback."""
    kernel = jnp.asarray((np.arange(128 * 64).reshape(128, 64) % 7 - 3) / 4.0, jnp.bfloat16)
    window = Bf16Window.of(kernel)
    form = np.asarray(window.form).copy()
    form.reshape(-1)[0] ^= 1
    with pytest.raises(ValueError):
        dataclasses.replace(window, form=form)
    leaves, structure = jax.tree.flatten(window)
    with pytest.raises(ValueError):
        jax.tree.unflatten(structure, [form if leaf is window.form else leaf for leaf in leaves])
