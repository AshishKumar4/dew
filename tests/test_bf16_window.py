"""A frozen bfloat16 matrix's lossless 13-bit window (`formats.Bf16Window`, row 5387): each weight its bits
again decoded, a group past its escapes refused by its site's name, and a Dense served its window as its
kernel (`quantization.bf16_windows`), its variables' kernels back bitwise (`decoded_packed`)."""

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
from dew.nn.kernels import formats, gemm, generation, grouped_matmul, grouped_quantized, tuning
from dew.nn.kernels.formats import Bf16Window
from dew.nn.kernels.templates.gemm import packed
from dew.nn.kernels.templates.gemm.packed import WINDOW


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


def test_a_window_codec_at_another_meta_layout_decodes_in_its_stages_and_reads_as_anothers():
    """The window codec at a layout of eight meta words a group, its base and escapes moved among them
    (`packed.WindowLayout`, the one descriptor its encoder, its decode, its stage reader and its stream
    read, row 13707; this Dew serves its own): its meta the default encoding's words relocated alone
    (`formats.window_planes`); its decode (`formats.window_weights`) and every lane of each step the shared
    stage walk decodes from its blocks and its step's meta halves (`packed.packed_lanes`, a step's 32 halves,
    `PackedWeightFormat.meta_columns`), its weights' bits; and a window holding its planes and its form is
    refused by this Dew's check (its digest another layout's)."""
    drawn = _two_steps()
    depth, width = drawn.shape
    moved = packed.WindowLayout(meta_words=8, base_word=5, escape_words=(6, 2))
    _, default = formats.window_planes(drawn)
    words, meta = formats.window_planes(drawn, moved)
    relocated = np.zeros((width, depth // moved.group, 8), np.uint32)
    relocated[..., [5, 6, 2]] = default.reshape(width, -1, WINDOW.meta_words)[
        ..., [WINDOW.base_word, *WINDOW.escape_words]]
    np.testing.assert_array_equal(meta.reshape(relocated.shape), relocated)
    decoded = np.asarray(formats.window_weights(words, meta, layout=moved))
    np.testing.assert_array_equal(decoded.view(np.uint16), drawn.view(np.uint16))
    desc = packed.weight_format("window", layout=moved)
    per_step = desc.meta_columns(moved.step)
    assert per_step == 32
    halves = np.stack([meta & 0xFFFF, meta >> 16], axis=-1).astype(np.uint16).reshape(width, -1)
    groups_a_step = moved.step // moved.group
    blocks_a_step = groups_a_step * desc.blocks
    for step in range(depth // moved.step):
        blocks = [jnp.asarray(words[:, 16 * block:16 * (block + 1)])
                  for block in range(step * blocks_a_step, (step + 1) * blocks_a_step)]
        step_meta = jnp.asarray(halves[:, step * per_step:(step + 1) * per_step])
        for group, code, lane in packed.packed_lanes(blocks, desc, interpret=True, meta=step_meta):
            at = step * moved.step + group * moved.group + 16 * code
            held = drawn[at:at + 16].T
            np.testing.assert_array_equal(np.asarray(lane).view(np.uint16), held.view(np.uint16))
    # Its digest folds its meta's placement in, not its width alone: another eight-word layout's differs too.
    other = packed.WindowLayout(meta_words=8, base_word=0, escape_words=(1, 2))
    for digest in (formats.window_form_version(), formats.window_form_version(other)):
        assert formats.window_form_version(moved) != digest
    with pytest.raises(ValueError):
        Bf16Window(words, meta, formats.window_form_version(moved)).checked()


def test_a_window_saved_before_its_layout_was_a_descriptor_restores():
    """A window whose form is the digest every saved `Bf16Window` has stored (`bf16-window-v2` and both
    planes' staged orders: 05 34 8c 5a 97 fb 62 4d, as main 057e56cc804 encoded it) passes this Dew's check
    and decodes to its weights, its words and meta this Dew's encoding's (train 1366's review: the layout's
    fingerprint folded into the served layout's digest refused every saved window)."""
    drawn = _two_steps()
    encoded = Bf16Window.of(drawn)
    stored = bytes([5, 52, 140, 90, 151, 251, 98, 77])
    saved = Bf16Window(encoded.words, encoded.meta, stored).checked()
    np.testing.assert_array_equal(np.asarray(saved.decoded()).view(np.uint16), drawn.view(np.uint16))


def test_a_group_past_its_escapes_is_refused():
    """A group holding as many nonzeros below its window as it stores escapes (`packed.WindowLayout.escapes`:
    tiny weights, 2^-20, beside 0.25) is held, each weight its bits again; one more is refused, where no
    window holds it."""
    weights = np.full((128, 4), 0.25, jnp.bfloat16)
    weights[:WINDOW.escapes, 3] = np.asarray(2.0 ** -20, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(Bf16Window.of(weights).decoded()).view(np.uint16),
                                  weights.view(np.uint16))
    weights[WINDOW.escapes, 3] = np.asarray(2.0 ** -20, jnp.bfloat16)
    with pytest.raises(ValueError):
        Bf16Window.of(weights, site="layers/q_proj")


def test_a_dense_served_its_window_is_its_kernels_product_and_its_kernel_comes_back():
    """A bfloat16 Dense whose kernel `bf16_windows` stores as its window computes what it computes over its
    kernel, bit for bit (its product of exact dyadic values, its bfloat16 bias added past it as Dense adds
    one), and `decoded_packed` gives its kernel back bitwise, the collection gone."""
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
    back = quantization.decoded_packed(windowed)
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


def _flipped(form: bytes) -> bytes:
    """`form` with its first bit flipped: another encoder's digest."""
    return bytes([form[0] ^ 1, *form[1:]])


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


def test_a_windowed_source_saves_as_its_kernels_would(tmp_path):
    """A loaded source whose MLP's down projection `bf16_windows` stores as its window saves
    (`Pretrained.save(variables=windowed)`) the release its own kernels save, every tensor bit for bit:
    the export reads every packed leaf back through the one traversal (`decoded_packed`, train 1349's
    review F4)."""
    from pathlib import Path

    from safetensors.numpy import load_file

    from dew.interop.pretrained import Pretrained

    loaded = Pretrained.load(str(Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"), dtype="bfloat16",
                             param_dtype="bfloat16")
    path = next(layout.paths[0] for layout in loaded.weight_layouts if ".mlp.down_proj." in layout.name)
    windowed = quantization.bf16_windows(loaded.variables, sites=["/".join(path[1:-1])])
    assert quantization.QUANTIZATION in windowed
    for name, variables in (("plain", loaded.variables), ("windowed", windowed)):
        loaded.save(tmp_path / name, variables=variables)
    plain, saved = (load_file(tmp_path / name / "model.safetensors") for name in ("plain", "windowed"))
    assert sorted(saved) == sorted(plain)
    for name, tensor in plain.items():
        assert (saved[name].dtype, saved[name].tobytes()) == (tensor.dtype, tensor.tobytes()), name


def test_a_windowed_dense_the_recipe_takes_runs_the_recipes_product():
    """A bfloat16 Dense whose kernel `bf16_windows` stores as its window, in a role the run's FP8 recipe
    takes (whole 128-wide blocks, the feed-forward's), computes the recipe's product of its kernel bit for
    bit, as the layer over its kernel does: a row holding 1 and 2^-20 in one block, whose small coordinate a
    unit weight alone reads, rounds that coordinate to zero in E4M3 where the window's exact product keeps
    it (row 13699)."""
    from dew.nn.kernels import generation
    from dew.nn.precision import FP8, projection_role

    kernel = np.zeros((128, 128), jnp.bfloat16)
    kernel[1, 0] = 1.0
    x = np.zeros((4, 128), jnp.bfloat16)
    x[:, 0], x[:, 1] = 1.0, 2.0 ** -20
    layer = Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False)
    variables = {"params": {"kernel": jnp.asarray(kernel)}}
    windowed = quantization.bf16_windows(variables, sites=[""])
    with generation.run_precision(FP8), projection_role("mlp"):
        plain, served = (np.asarray(jax.jit(layer.apply)(held, jnp.asarray(x))).view(np.uint16)
                         for held in (variables, windowed))
    np.testing.assert_array_equal(served, plain)


def test_a_windowed_dense_with_an_fp32_bias_keeps_its_declared_tangent():
    """A bfloat16 Dense with its bias stored fp32, its kernel a window: its tangent is Dense's own, each
    direction's product rounded to bfloat16 before the bias's direction joins it (`dense._widened`), bit
    for bit the layer's over its kernel: a column `[1, 2^-8, 0, ...]`, an input direction `[1, 1, 0, ...]`
    and a bias direction -1 give 0, where an fp32 sum of the products gives 2^-8 (row 13701)."""
    kernel = np.zeros((128, 64), jnp.bfloat16)
    kernel[0, 0], kernel[1, 0] = 1.0, 2.0 ** -8
    layer = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    params = {"kernel": jnp.asarray(kernel), "bias": jnp.zeros((64,), jnp.float32)}
    windowed = quantization.bf16_windows({"params": params}, sites=[""])
    x = jnp.ones((2, 128), jnp.bfloat16)
    direction = jnp.zeros((2, 128), jnp.bfloat16).at[:, :2].set(1.0)
    bias_direction = jnp.full((64,), -1.0, jnp.float32)

    def tangent(variables):
        def run(inputs, bias):
            held = {**variables, "params": {**variables["params"], "bias": bias}}
            return layer.apply(held, inputs)

        _, out = jax.jvp(run, (x, params["bias"]), (direction, bias_direction))
        return np.asarray(out, np.float32)

    plain, served = tangent({"params": params}), tangent(windowed)
    np.testing.assert_array_equal(served.view(np.uint32), plain.view(np.uint32))
    assert plain[0, 0] == 0.0
    # Another encoder's window, its digest not this Dew's, is refused on this path as on the window's own.
    held = windowed[quantization.QUANTIZATION]["kernel"]
    foreign = dataclasses.replace(held, form=_flipped(held.form))
    with pytest.raises(ValueError):
        tangent({**windowed, quantization.QUANTIZATION: {"kernel": foreign}})


@pytest.mark.distributed
def test_a_kernel_no_process_holds_whole_packs_where_its_columns_live(tmp_path):
    """A `[256, 256]` kernel split by its columns across two processes' devices, which no process holds
    whole: `bf16_windows` encodes each column's window where it lives, each process's planes its own
    columns' window, decoded its own columns' bits again (row 13702: it read the whole kernel on each
    process, which JAX refuses for a value lying on other processes' devices)."""
    from pathlib import Path

    from process_support import run_pool

    worker = Path(__file__).with_name("bf16_window_worker.py")
    for report in run_pool("width_split", tmp_path / "pool", 2, devices=2, script=worker):
        assert report["planes"] and all(report["planes"]), report
        assert report["decoded"] and all(report["decoded"]), report
    # On process 0's devices alone: every process windows it alike, process 1 holding none of its planes.
    reports = run_pool("one_rank", tmp_path / "one", 2, devices=2, script=worker)
    assert reports[0]["planes"] and all(reports[0]["planes"]) and reports[1]["planes"] == [], reports
    assert reports[0]["placed"] == reports[1]["placed"], reports
    assert all(report["shape"] == [256, 256] for report in reports), reports
    # Each process's own kernel on its own device, an ordinary array of no mesh: windowed whole where it is.
    for report in run_pool("own_device", tmp_path / "own", 2, devices=2, script=worker):
        assert report["decoded"], report
    # A mesh of process 0's own devices, windowed there alone: no peer's agreement waited on.
    reports = run_pool("own_mesh", tmp_path / "own_mesh", 2, devices=2, script=worker, timeout=300)
    assert [report["decoded"] for report in reports] == [True, None], reports
    # Saved and read back on every process (row 13700): each process's own columns' bits; another encoder's
    # form refused on every process, by the form's check where it reads the digest.
    reports = run_pool("checkpointed", tmp_path / "checkpointed", 2, devices=2, script=worker)
    for report in reports:
        assert report["own"] and all(report["own"]), report
        assert report["foreign"] in ("ValueError", "PeerFailure"), report
    assert any(report["foreign"] == "ValueError" for report in reports), reports
    # Split by its depth, a column's groups apart, which its window's decode reads whole: refused.
    for report in run_pool("depth_split", tmp_path / "depths", 2, devices=2, script=worker):
        assert report["refused"] == "ValueError", report


def test_a_window_of_another_shape_than_its_dense_is_refused_on_every_route():
    """A `[128, 256]` window in a `Dense(128)` is refused by its shape whichever route would read it, the
    run's FP8 recipe's decode among them, which would otherwise emit 256 features (train 1360's review)."""
    from dew.nn.kernels import generation
    from dew.nn.precision import FP8, projection_role

    kernel = jnp.zeros((128, 256), jnp.bfloat16)
    windowed = quantization.bf16_windows({"params": {"kernel": kernel}}, sites=[""])
    layer = Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False)
    x = jnp.ones((4, 128), jnp.bfloat16)
    with pytest.raises(ValueError):
        layer.apply(windowed, x)
    with generation.run_precision(FP8), projection_role("mlp"), pytest.raises(ValueError):
        layer.apply(windowed, x)


def test_a_windowed_dense_under_delayed_nvfp4_records_its_rows_largest_magnitude():
    """A bfloat16 Dense whose kernel is its window, under a delayed-scaling NVFP4 recipe that takes it,
    reads its amax history and sows its rows' largest magnitude as the layer over its kernel does: under
    two histories that quantize its rows apart, its output and its sown amax bit for bit the unpacked
    layer's (row 13699; train 1360's review)."""
    from dew.nn.dense import NVFP4_AMAX, NVFP4_HISTORY
    from dew.nn.kernels import generation
    from dew.nn.precision import DTYPE_PRESETS, LowPrecision, Numerics, projection_role

    delayed = Numerics(DTYPE_PRESETS["bf16-mixed"], low_precision=LowPrecision(recipe="nvfp4",
                                                                              tensor_scaling="delayed"))
    rng = np.random.default_rng(13699)
    x = jnp.asarray(rng.standard_normal((4, 128)), jnp.bfloat16)
    layer = Dense(128, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, use_bias=False)
    outputs = []
    with generation.run_precision(delayed), projection_role("mlp"):
        variables = layer.init(jax.random.key(0), x)
        # Two histories, each a scale the rows quantize at: the window's layer reads each as its kernel's.
        for largest in (1.0, 64.0):
            held = {**variables, NVFP4_HISTORY: jax.tree.map(lambda history, largest=largest:
                                                             jnp.full_like(history, largest),
                                                             variables[NVFP4_HISTORY])}
            windowed = quantization.bf16_windows(held, sites=[""])
            (plain, plain_amax), (served, served_amax) = (layer.apply(tree, x, mutable=[NVFP4_AMAX])
                                                          for tree in (held, windowed))
            assert jax.tree.leaves(plain_amax), "the unpacked layer sows its rows' largest magnitude"
            np.testing.assert_array_equal(np.asarray(served).view(np.uint16),
                                          np.asarray(plain).view(np.uint16))
            for sown, own in zip(jax.tree.leaves(served_amax), jax.tree.leaves(plain_amax), strict=True):
                np.testing.assert_array_equal(np.asarray(sown), np.asarray(own))
            outputs.append(np.asarray(plain))
    assert not np.array_equal(*outputs), "the two histories quantize the rows apart"


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


def test_gemm_over_a_window_into_bf16_at_the_highest_precision_takes_the_windows_launch():
    """`gemm.gemm` over a bfloat16 window into bf16 asked the highest precision (its partial sums promoted
    nowhere: a bf16 output's, `templates.product.declared`) keeps the window: its own launch decoding each
    stage (`MosaicWordPackedGmm`, interpreted on a Hopper's template), whose identity rows give each weight
    back exactly, bit for bit (row 13686's review F3: a call routed to the decoded matrix's product took no
    window launch)."""
    # Finite dyadics, sixteenths of small integers, their zeros positive: the identity's sums are each one
    # weight exactly (a signed zero's survival is the codec's own test).
    weights = (np.random.default_rng(13686).integers(-64, 64, (256, 64)) / 16.0).astype(jnp.bfloat16)
    window = Bf16Window.of(weights).checked()
    rows = jnp.asarray(np.eye(256, dtype=jnp.bfloat16))
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          jax.default_matmul_precision("highest"), tuning.seen() as seen):
        got = gemm.gemm(rows, window, None, out_dtype=jnp.bfloat16, interpret_on_cpu=True)
    assert any(isinstance(call.op, grouped_quantized.MosaicWordPackedGmm) for _, call in seen.taken), seen
    np.testing.assert_array_equal(np.asarray(got).view(np.uint16), weights.view(np.uint16))


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


def _dot_arithmetics(lowered: jax.stages.Lowered) -> list:
    """Each `stablehlo.dot_general` of `lowered`'s module, in order, as the arithmetic its algorithm states
    (`programs.algorithm_arithmetic`), None where it states none."""
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import stablehlo

    found = []

    def visit(op):
        if op.name == "stablehlo.dot_general":
            attributes = op.attributes
            found.append(programs.algorithm_arithmetic(stablehlo.DotAlgorithm(attributes["algorithm"]))
                         if "algorithm" in attributes else None)
        return ir.WalkResult.ADVANCE

    module = lowered.compiler_ir("stablehlo")
    with module.context:
        module.operation.walk(visit)
    return found


F16_PRODUCTS = programs.dot_arithmetic(jax.lax.DotAlgorithmPreset.F16_F16_F32, jnp.bfloat16, jnp.bfloat16)
"""The arithmetic `F16_F16_F32` states a bf16 product in, which a dot declared with it carries."""


def test_a_grouped_window_products_value_and_tangent_state_one_algorithm():
    """A grouped window product (`window_product`, its XLA route) under a run's `F16_F16_F32`, its precision
    omitted, lowered for a GH200 (its platform's branch alone), states that algorithm in every ragged dot of
    its value and its tangent, the contractions it lowers to (row 13704): the value's no longer forced to the
    default where the tangent's took the run's (bf16 rows of 2^-30 would read 2^-30 and a zero tangent on a
    CUDA device)."""
    from kernel_compile import lowered_for

    kernel = jnp.zeros((2, 128, 64), jnp.bfloat16).at[:, 0, 0].set(1.0)
    window = Bf16Window.of(kernel).checked()
    sizes = jnp.asarray([2, 2], jnp.int32)

    def product(x):
        return gemm.window_product(x, window, sizes, out_dtype=jnp.float32)

    def value_and_tangent(x):
        # Its value and tangent, and its pullback of the value as cotangent: every contraction of the three.
        value, tangent = jax.jvp(product, (x,), (x,))
        return value, tangent, jax.vjp(product, x)[1](value)

    with jax.default_matmul_precision("F16_F16_F32"):
        rows = (jax.ShapeDtypeStruct((4, 128), jnp.bfloat16),)
        found = _dot_arithmetics(lowered_for(value_and_tangent, rows, "sm90"))
    # A named algorithm's ragged dot lowers through dots that carry it, the value's and the tangent's alike.
    assert len(found) >= 2 and all(each == F16_PRODUCTS for each in found), found


@pytest.mark.parametrize(("run", "stated", "algorithm"),
                         [("F16_F16_F32", None, F16_PRODUCTS),
                          ("F16_F16_F32", jax.lax.Precision.DEFAULT, None),
                          ("F32_F32_F32", jax.lax.Precision.DEFAULT, None)],
                         ids=["omitted: the run's narrowing algorithm", "the default over a narrowing run",
                              "the default over a whole run"])
def test_gemm_over_a_window_states_the_algorithm_its_call_and_run_declare(run, stated, algorithm):
    """`gemm.gemm` over a bfloat16 window of bf16 rows under a run's algorithm, lowered for a GH200 (its
    platform's branch alone), states to XLA the algorithm the call declares (row 13686's review): the run's
    where the call states no precision, none where it states the default (its level overriding the run's,
    a narrowing one's or a whole one a TPU does not run), on its decoded fallback (the window's own dot
    reads the run's)."""
    from kernel_compile import lowered_for

    window = Bf16Window.of(jnp.zeros((128, 64), jnp.bfloat16).at[0, 0].set(1.0)).checked()
    with jax.default_matmul_precision(run):
        found = _dot_arithmetics(lowered_for(
            lambda x: gemm.gemm(x, window, None, out_dtype=jnp.float32, precision=stated),
            (jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),), "sm90"))
    assert found and all(each == algorithm for each in found), found


@pytest.mark.parametrize("compute", [jnp.bfloat16, jnp.float32], ids=["window", "decoded"])
def test_gemm_over_a_window_of_another_encoding_is_refused(compute):
    """A window of another encoder's form is refused by `gemm.gemm` at its product, on the window's own
    route (bf16 compute at the default precision, `window_product`) and on its decoded fallback (fp32
    compute, `Bf16Window.decoded`) alike, its words never read through this form's decode (row 13686's
    review): called as it is, passed into a jitted program, and forged inside one by a replace of its form
    (row 13700: a form traced there was checked nowhere); this Dew's window runs in the same program."""
    kernel = jnp.asarray((np.arange(128 * 64).reshape(128, 64) % 7 - 3) / 4.0, jnp.bfloat16)
    window = Bf16Window.of(kernel)
    foreign = dataclasses.replace(window, form=_flipped(window.form))
    rows = jnp.ones((4, 128), jnp.float32)

    def product(window):
        return gemm.gemm(rows, window, None, out_dtype=jnp.float32, compute=compute)

    def forged(window):
        return product(dataclasses.replace(window, form=_flipped(window.form)))

    with jax.default_matmul_precision("default"):
        for refused in (lambda: product(foreign), lambda: jax.jit(product)(foreign),
                        lambda: jax.jit(forged)(window)):
            with pytest.raises(ValueError):
                refused()
        np.testing.assert_array_equal(np.asarray(jax.jit(product)(window)), np.asarray(product(window)))

