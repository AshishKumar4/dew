"""A frozen bfloat16 matrix's lossless 13-bit window (`formats.Bf16Window`, row 5387): each weight its bits
again decoded, a group past its escapes refused by its site's name, and a Dense served its window as its
kernel (`quantization.bf16_windows`), its variables' kernels back bitwise (`decoded_bf16_windows`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_gdn_support import KernelCase
from generation_support import lowers
from kernel_compile import compile_mosaic_kernels

from dew import quantization
from dew.nn.dense import Dense
from dew.nn.kernels import gemm, generation
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


def test_a_group_past_its_escapes_is_refused_by_its_sites_name():
    """A group holding more nonzeros below its window than it stores escapes (`WINDOW_ESCAPES`: one weight
    near 2^0 beside five tiny ones, 2^-20 and less) is refused, naming its site, where no window holds it."""
    weights = np.full((128, 4), 0.25, jnp.bfloat16)
    weights[: WINDOW_ESCAPES + 1, 3] = np.asarray([2.0 ** -20] * (WINDOW_ESCAPES + 1), jnp.bfloat16)
    with pytest.raises(ValueError, match=r"layers/q_proj's column 3"):
        Bf16Window.of(weights, site="layers/q_proj")


def test_a_dense_served_its_window_is_its_kernels_product_and_its_kernel_comes_back():
    """A bfloat16 Dense whose kernel `bf16_windows` stores as its window computes its kernel's product (each
    value dyadic, so every order sums it exactly in fp32, its bias added and rounded once), its kernel
    decoded back bitwise by
    `decoded_bf16_windows`, the collection gone."""
    rng = np.random.default_rng(5387)
    kernel = (rng.integers(-64, 64, (256, 64)) / 16.0).astype(jnp.bfloat16)
    bias = (rng.integers(-8, 8, (64,)) / 4.0).astype(jnp.float32)
    x = jnp.asarray(rng.integers(-4, 4, (8, 256)), jnp.bfloat16)
    layer = Dense(64, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    variables = {"params": {"kernel": jnp.asarray(kernel), "bias": jnp.asarray(bias, jnp.bfloat16)}}
    windowed = quantization.bf16_windows(variables)
    assert "kernel" not in windowed["params"]
    rounded = np.asarray(bias.astype(jnp.bfloat16), np.float64)
    want = np.asarray(x, np.float64) @ np.asarray(kernel, np.float64) + rounded
    # Its fp32 sum exact, plus its bias, rounded once to the layer's bfloat16.
    np.testing.assert_array_equal(np.asarray(jax.jit(layer.apply)(windowed, x), np.float32),
                                  np.asarray(want.astype(jnp.bfloat16), np.float32))
    back = quantization.decoded_bf16_windows(windowed)
    assert set(back) == {"params"}
    np.testing.assert_array_equal(np.asarray(back["params"]["kernel"]).view(np.uint16),
                                  kernel.view(np.uint16))


@pytest.mark.parametrize("grouped", [False, True], ids=["dense", "grouped"])
def test_a_window_product_decoded_in_its_stages_is_its_weights_product(grouped):
    """A window of the edges `_drawn` codes (signed zeros, subnormals, escapes, a group's floor, a group near
    the largest finite, infinities and NaNs) times the rows of the identity, each output entry one weight
    times one: its product on the template's route, each stage decoded in place (interpreted), and on XLA's,
    the decoded matrix's, each XLA's plain product of the weights themselves, bit for bit (a NaN where it
    is NaN), dense and grouped
    (two groups of 128 rows, the second's weights reversed in its columns); the route each took reported."""
    weights = _drawn((256, 64))
    rows = np.eye(256, dtype=jnp.bfloat16)
    sizes = np.asarray([128, 128], np.int32)
    if grouped:
        stacked = np.stack([weights, weights[:, ::-1]])
        window = Bf16Window.of(stacked)
        want = np.concatenate([
            jax.lax.dot(jnp.asarray(rows[at * 128:(at + 1) * 128]), jnp.asarray(stacked[at]),
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


def test_a_window_launch_takes_its_staged_matrix_and_refuses_its_rows_swapped():
    """A window's launch reads its columns' meta halves beside its words (`gmm`'s `scales`): a staged
    matrix of a 256-deep step is taken, and a call swapping its matrix into the products' rows, which stream
    no columns' meta, is refused by name."""
    from dew.nn.kernels.templates.gemm import packed, tiling

    tiles = tiling.Tiles(64, 64, 256, 2, 1, family="wgmma")
    words = jax.ShapeDtypeStruct((1, 64, 256 // 128 * 48), jnp.uint32)
    packed.word_taken(words, "window", tiles, transpose_rhs=True, prologue=None, paired=None,
                      transpose_lhs=False, depth=256, width=64)
    with pytest.raises(ValueError, match="staged matrix"):
        packed.word_taken(words, "window", tiles, transpose_rhs=True, prologue=None, paired=None,
                          transpose_lhs=False, depth=256, width=64, packed_rows=True)


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
