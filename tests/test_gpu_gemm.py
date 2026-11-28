"""The GEMM template and the Mosaic toolkit on the GPU (G14, the owner's rule: no unit tests on GPUs), run in
one process (`gpu_flow`), each cell its own file's property:

- A dense product's exact sums survive one row tile through Triton's MMA accumulation (test_gemm_template).
- A pair of multiprocessors' cooperative tile, a cluster multicasting its shared blocks, holds check 1 with
  its hooks, and a dense kernel gradient's balanced tail, its partials fixed up across programs, holds
  check 1, on an H100 (test_gemm_template).
- CUTLASS runs the grouped projection as exactly as XLA wherever it is chosen, over every grouping that
  routes a row (test_grouped_gemm).
- The blockwise FP8 kernel gradient (rows contiguous, step scales) holds check 1 on an H100, and the
  products a receipt timed (a routed gate's forward and its input gradient's launch) hold check 1 unnamed
  at their own extents (test_gemm_blockwise).
- The template takes gpt-oss's 2880-wide projection, its ragged stores' one-row rungs, as exactly as XLA's
  (test_grouped_gemm; row 4474).
- A ring streams tiles of their own trip counts, consumers feeding their own ring split a step, and every
  read path of a warpgroup accumulator lands its products under a sign op on them (row 4346), on an H100
  (test_mosaic_toolkit).
- The shared GEMM mainloop runs on Blackwell's tcgen05, exactly, on a B200 or a B300, sm100 and sm103 alike,
  and its epilogues (a column bias, a per-row scale, two outputs of two dtypes) hold check 1 at each
  accumulator width and on a two-CTA pair, the pair also on extents no tile divides (test_gemm_tcgen05).
- The NVFP4 product's packed kernel (its mainloop decoding each stage in place) holds every order on an
  H100 and an A100 (test_contracts_gemm).
- The quantize epilogue's E4M3 and E5M2 encodes, `bits.encode_form`'s PTX inline in the compiled epilogue,
  are `bits.encoded` bit for bit on an H100 and an A100 (test_gemm_nvfp4; row 14105).
- A delayed NVFP4 MLP's chain on a Blackwell, its up projection's block-scaled epilogue storing the down
  projection's rows as codes two a byte and transposed block scales, is bitwise its projections apart,
  value and every gradient, gated and ungated (test_gemm_nvfp4_chain; rows 12721, 12743).
- A grouped product whose producer gathers its rows from the tokens by a table (cp.async, row 1376) holds
  check 1 on an H100, plain and paired, the paired one's rows stored where a permutation says; a kernel
  gradient reading its tokens and its cotangent through tables is bitwise its gathered rows' on an H100;
  and on a B200 tcgen05's load warp gathers by TMA's four-row gathers, bitwise the rows gathered first
  (test_gemm_gather).
- The template's grouped product of 16-bit operands sums into fp32 as exactly as XLA's at depths 128 and
  512, its sums the tensor cores' own at the default precision (test_grouped_gemm; M-EM-3).
- The toolkit's warpgroup scan walks a linear recurrence forward and reversed, each position's state or
  the one before it, as float64 does (test_warpgroup_scan).
- GEMM's check-1 matrix (tests/kernel_matrix.py): the template route's training step on par with the reference
  path's, dense and grouped, each matrix layout, hook and dtype its served pairs name (gemm_matrix).
- The fused GEMM's check-1 matrix (tests/kernel_matrix.py): a region's training step through the fuser's
  prologue and epilogue on par with the same region's stages in XLA (fused_gemm_matrix).
- The NVFP4, blockwise FP8 and adapted products' check-1 matrices (tests/kernel_matrix.py), each at the cases
  a red found on a GPU (`kernel_matrix.seeded_cells`), and the MX products' over a covering of their block
  formats and output dtypes on a Blackwell, whose block-scaled MMA no interpreter runs; every other pair is
  the CPU's, interpreted.
- The joined projections' check-1 matrix (tests/join_matrix.py), each cell on a GPU of its generation: an
  attention layer's query, key and value side by side on a Hopper's warpgroup MMA among them.
- The fused linear cross-entropy's check-1 matrix (tests/linear_ce_matrix.py), each cell on a GPU of its
  generation: an LM head's loss and gradients through the template's fused kernel on par with the chunked
  XLA program's.
"""

import functools

import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
import pytest
import syrk_matrix
import test_contracts_gemm as contracts_gemm
import test_declared_cells_memory as declared_cells_memory
import test_fed_walks as fed_walks
import test_gemm_blockwise as gemm_blockwise
import test_gemm_converted as gemm_converted
import test_gemm_gather as gemm_gather
import test_gemm_nvfp4 as gemm_nvfp4
import test_gemm_nvfp4_chain as gemm_nvfp4_chain
import test_gemm_paired as gemm_paired
import test_gemm_tcgen05 as gemm_tcgen05
import test_gemm_template as gemm_template
import test_gemm_widths as gemm_widths
import test_grouped_gemm as grouped_gemm
import test_mosaic_toolkit as mosaic_toolkit
import test_warpgroup_scan as warpgroup_scan
from gpu_flow import Cell, cells_of, covering, in_parts, parted, run_cells, when
from jax.experimental.pallas import fuser
from kernel_compile import gate_generations
from reference_error import ahead_arm, assert_as_exact
from test_gemm_template import rms_norm as template_rms_norm

from dew.algebra import bits
from dew.nn.kernels import fused_gemm, generation
from dew.nn.kernels.templates.gemm import tiling
from dew.nn.norms import rms_normalized

pytestmark = pytest.mark.gpu(reason="the GEMM template's and the Mosaic toolkit's kernels, compiled")


def _fused_projection_training():
    """One routed projection update, including an empty expert and rows beyond the routed ones. The
    input cotangent exercises both paths through RMS normalization, and the updated parameters run
    another forward. The reference is the plain program main's GEMM tests take: the RMS norm as a model
    writes it and XLA's ragged product (`test_gemm_template`), outside the template."""
    rows, depth, width = 256, 256, 256
    sizes = (0, 100, 150)
    groups = np.asarray(sizes, np.int32)

    def native(x, params):
        normalized = rms_normalized(x, params["norm"], 1e-6, x.dtype, scale_offset=False,
                                    scale_after_cast=False, fp32_statistics=True, implementation="xla")
        projected = fused_gemm.fusible_gemm(normalized, params["weight"].astype(x.dtype), groups)
        return jax.nn.silu(projected + params["bias"]) + x[:, :width].astype(jnp.float32)

    def reference(x, params):
        work = jnp.promote_types(x.dtype, jnp.float32)
        normalized = template_rms_norm(x, params["norm"])
        matrices = params["weight"].astype(x.dtype).astype(work)
        # Rows past the routed ones take no expert: ragged_dot leaves them zero.
        projected = jax.lax.ragged_dot(normalized.astype(work), matrices, groups, preferred_element_type=work)
        return jax.nn.silu(projected + params["bias"]) + x.astype(work)[:, :width]

    def step(forward):
        def update(x, params, targets):
            def loss(x, params):
                values = forward(x, params)
                return jnp.mean(jnp.square(values - targets)), values

            (value, outputs), (dx, gradients) = jax.value_and_grad(
                loss, argnums=(0, 1), has_aux=True)(x, params)
            updated = jax.tree.map(lambda p, g: p - 0.25 * g, params, gradients)
            after = forward(x, updated)
            return value, outputs, dx, gradients, updated, jnp.mean(jnp.square(after - targets))
        return update

    def draw(seed):
        # NumPy's, as a site's draws are: a GPU tier's prepare computes on no device (rows 8505, 8506).
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(rows, depth)).astype(jnp.bfloat16)
        x[:2] = 0
        # Master weights begin at representable operand values; the update itself stays in fp32.
        weight = (rng.normal(size=(len(sizes), depth, width)) / np.sqrt(depth)).astype(jnp.bfloat16)
        params = {"weight": weight.astype(np.float32),
                  "norm": rng.uniform(0.8, 1.2, depth).astype(np.float32),
                  "bias": (rng.normal(size=width) * 0.1).astype(np.float32)}
        return x, params, rng.normal(size=(rows, width)).astype(np.float32)

    return step(fuser.fuse(native)), step(reference), draw


def a_fused_projection_trains_its_producer_and_epilogue():
    native, reference, draw = _fused_projection_training()
    with generation.chosen(gemm_prologue="fused", gemm_epilogue="fused"):
        assert_as_exact(jax.jit(native), jax.jit(reference), draw,
                        "fused projection producer and epilogue update", seeds=8, truth_fn=reference)


def native_hopper_cluster2_balanced_tail_holds_check_1():
    """Native Hopper `cluster=2` dense forward with a verified nonempty balanced `Tail` holds check 1
    against XLA dot_general from float64 truth at Qwen3.5-4B q dX outlier shape (row 10151), its tail's
    first partials a slot a tail tile (`tail.slots`, NG29)."""
    from dew.nn.kernels import gemm, grouped_matmul, tuning
    tile = tiling.Tiles(128, 256, 64, 4, 8, warpgroups=2, cluster=2, family="wgmma")
    target = generation.gpu_target()
    launch = grouped_matmul.MosaicGmm()
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    ba = launch.bind(bf16((2048, 8192)), bf16((1, 8192, 2560)), jax.ShapeDtypeStruct((1,), jnp.int32),
                     out_dtype="bfloat16", trans_rhs=False, interpret=False, dense=True)
    assert launch.legal(ba, tile, target) and launch._tail(ba, tile, target)[1] is not None
    def draw(seed):
        rng = np.random.default_rng(10151 + seed)
        return ((rng.normal(size=(2048, 8192)) / np.sqrt(8192)).astype(jnp.bfloat16),
                (rng.normal(size=(8192, 2560)) / np.sqrt(8192)).astype(jnp.bfloat16))

    def dew_fn(x, w):
        with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicGmm: tile}):
            return gemm.gemm(x, w, out_dtype=jnp.bfloat16)

    def plain(x, w, *, wide=jnp.float32, out=jnp.bfloat16):
        return jax.lax.dot(x.astype(wide), w.astype(wide), preferred_element_type=wide).astype(out)

    assert_as_exact(jax.jit(dew_fn), jax.jit(plain), draw, "hopper cluster=2 balanced tail 2048x8192x2560",
                    seeds=8, truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))

def native_hopper_cluster2_balanced_tail_kernel_gradient_holds_check_1():
    """The Qwen3.5-4B q outlier's kernel gradient (8192 rows, 2560 by 2048: 80 tiles over a GH200's 66
    pairs, 14 of them a balanced tail) into bf16, as a bf16 model's weights take it, at a pinned 128 x 256
    cluster tile, its tail tiles' first partials a slot a tile (`tail.slots`, NG29): check 1 against XLA's
    from float64 truth. An fp32 output is summed exactly under the suite's precision, whose cluster tile's
    tail the launch refuses (`BALANCED_TAILS`)."""
    from dew.nn.kernels import gemm, grouped_matmul, tuning
    tile = tiling.Tiles(128, 256, 64, 4, 8, warpgroups=2, cluster=2, family="wgmma")
    launch, bf16 = grouped_matmul.MosaicTgmm(), functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    ba = launch.bind(bf16((8192, 2560)), bf16((8192, 2048)), jax.ShapeDtypeStruct((1,), jnp.int32),
                     out_dtype="bfloat16", interpret=False, dense=True, exact_sums=False)
    target = generation.gpu_target()
    assert launch.legal(ba, tile, target) and launch.balanced(ba, tile, target)

    def draw(seed):
        rng = np.random.default_rng(29 + seed)
        return ((rng.normal(size=(8192, 2560)) / np.sqrt(8192)).astype(jnp.bfloat16),
                rng.normal(size=(8192, 2048)).astype(jnp.bfloat16))

    def dew_fn(x, dy):
        with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicTgmm: tile}):
            return gemm.gemm_t(x, dy, None, out_dtype=jnp.bfloat16)[0]

    def plain(x, dy, *, wide=jnp.float32, out=jnp.bfloat16):
        return jnp.dot(x.T.astype(wide), dy.astype(wide)).astype(out)

    assert_as_exact(jax.jit(dew_fn), jax.jit(plain), draw, "hopper cluster=2 balanced tail dW 2560x8192x2048",
                    seeds=8, truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


def an_xla_fallbacks_dense_bf16_product_holds_check_1_against_xlas_own_dot():
    """A dense bf16 product the template leaves to XLA, no derivative taken, stores bf16 directly
    (`gemm_xla.direct_dtype`, row 12898): check 1 against XLA's own dense dot (Dense's, the reference's bf16
    product) from float64 truth, over draws whose rows cancel to below a bf16 ulp of their terms."""
    from dew.nn.kernels import gemm

    def draw(seed):
        rng = np.random.default_rng(12898 + seed)
        x = rng.normal(size=(512, 1024))
        w = rng.normal(size=(1024, 768)) / np.sqrt(1024)
        x[:, 512:] = -x[:, :512] * (1 + 2.0 ** -9)  # each row's second half nearly cancels its first
        w[512:] = w[:512]
        return x.astype(jnp.bfloat16), w.astype(jnp.bfloat16)

    def dew_fn(x, w):
        with generation.chosen(grouped_matmul="xla"):
            return gemm.gemm(x, w, None, out_dtype=jnp.bfloat16)

    def plain(x, w, *, wide=None, out=jnp.bfloat16):
        return jax.lax.dot(x.astype(wide or x.dtype), w.astype(wide or w.dtype),
                           preferred_element_type=wide).astype(out)

    assert_as_exact(jax.jit(dew_fn), jax.jit(plain), draw, "xla fallback dense bf16 512x1024x768", seeds=8,
                    truth_fn=functools.partial(plain, wide=jnp.float64, out=jnp.float64))


def _converted_cells() -> tuple[Cell, ...]:
    """The converting FP8 codes product's check 1 on an A100, each format's (row 8548): its first native
    run qualifies the accuracy its receipt's timing rests on."""
    on_par = gemm_converted.an_fp8_codes_product_through_the_converting_template_is_on_par
    return tuple(Cell(f"an fp8 codes product through the converting template is on par ({name}, {form})",
                      functools.partial(on_par, name, orientation=form), "sm80", mosaic=True,
                      native={"launch": "converted fp8 gmm", "format": name, "orientation": form})
                 for name in gemm_converted.FORMATS for form in gemm_converted.ORIENTATIONS)


def _scale_edge_cells() -> tuple[Cell, ...]:
    """The blockwise FP8 template's exact witnesses at its scales' ends (rows 13257, 13274, 13291), each an
    exact value its run reads through the template's route."""
    return (
        Cell("a blockwise FP8 product at far-apart scales is exact",
             gemm_blockwise.a_blockwise_fp8_product_at_far_apart_scales_is_exact, "sm90",
             at_run=EXACT_OVERFLOW, native={"launch": "blockwise fp8 gmm", "pass": "subnormal scale"}),
        Cell("a blockwise FP8 product at large finite scales is finite",
             gemm_blockwise.a_blockwise_fp8_product_at_large_finite_scales_is_finite, "sm90",
             at_run=EXACT_OVERFLOW, native={"launch": "blockwise fp8 gmm", "pass": "step scaling"}),
        *(Cell(f"a blockwise FP8 receipt cell holds check 1 unnamed ({name})",
               functools.partial(gemm_blockwise.a_receipt_cells_blockwise_product_holds_check_1_unnamed,
                                 name),
               "sm90", mosaic=True, native={"launch": "blockwise fp8 gmm", "pass": f"receipt cell, {name}"})
          for name in gemm_blockwise.RECEIPT_CELLS),
        *(Cell(f"an E5M2 blockwise {name} at fp32's largest is finite ({edge})",
               functools.partial(witness, edge), "sm90", at_run=EXACT_OVERFLOW,
               native={"launch": launch, "pass": f"e5m2 step scaling, {edge}"})
          for name, witness, launch in (
              ("product", gemm_blockwise.an_e5m2_blockwise_product_at_fp32s_largest_is_finite,
               "blockwise fp8 gmm"),
              ("kernel gradient", gemm_blockwise.an_e5m2_blockwise_kernel_gradient_at_fp32s_largest_is_finite,
               "blockwise fp8 tgmm"))
          for edge in gemm_blockwise.E5M2_EDGES),
        Cell("fake_quant's E4M3 codes convert as their rounding does",
             gemm_blockwise.fp8_codes_convert_as_the_rounding_does, "sm90", at_run=EXACT_CODES,
             native={"launch": "fp8 codes", "pass": "convert"}),
        Cell("an owed product's native fold rounds once below the normals",
             gemm_nvfp4.an_owed_products_native_fold_rounds_once_below_the_normals, "sm90",
             at_run=EXACT_OVERFLOW, native={"launch": "owed fold", "pass": "subnormal product"}),
        Cell("a packed NVFP4 call placed on its host's CPU keeps a subnormal column scale",
             gemm_nvfp4.a_packed_nvfp4_call_placed_on_its_hosts_cpu_keeps_a_subnormal_scale, "sm90",
             at_run=EXACT_OVERFLOW, native={"launch": "nvfp4 packed", "pass": "host-placed finish"}),
    )


def _encode_cells() -> tuple[Cell, ...]:
    """The quantize epilogue's FP8 encodes compiled, each `bits.encode_form`'s PTX inline in the epilogue
    (`inline_forms.registers`), which the interpreter runs as the form's evaluation instead (algebra.md §7.3,
    row 14105): on an H100's warpgroup MMA and an A100's mma.sync, each `bits.encoded` bit for bit."""
    return tuple(
        Cell(f"the quantize epilogue's {name} encode is bits.encoded's on {gpu}",
             functools.partial(gemm_nvfp4.the_quantize_epilogue_encodes_as_bits_encoded, minifloat), gpu,
             mosaic=gpu == "sm80", fault=INLINE_ENCODE, at_run=EXACT_CODES)
        for gpu in ("sm90", "sm80") for name, minifloat in (("E4M3", bits.E4M3), ("E5M2", bits.E5M2)))


def _syrk_and_tail_cells():
    return [
        # The family's cells as its own grid states them, each pinned to its GPU (`gpu_flow.cells_of`).
        *cells_of(syrk_matrix.a_symmetric_product_is_on_par,
                  native=lambda case: {"launch": "syrk", "form": case[1]},
                  fault=lambda case: ("native SYRK upper-triangle mirror store (row 11941)"
                                      if case[1] == "gram" else None)),
        Cell("an XLA fallback's dense bf16 product holds check 1 against XLA's own dot",
             an_xla_fallbacks_dense_bf16_product_holds_check_1_against_xlas_own_dot, "sm90",
             fault=CUBLAS_BF16, native={"launch": "xla fallback", "pass": "forward"}),
        *(Cell(f"an XLA fallback's {product} tangent sums its terms before rounding",
               functools.partial(TANGENT_SUMS, product),
               "sm90", fault=CUBLAS_BF16, at_run=EXACT_TANGENT,
               native={"launch": "xla fallback", "pass": f"{product} tangent"})
          for product in ("gemm", "gemm_t")),
        Cell("native Hopper cluster=2 balanced tail holds check 1",
             native_hopper_cluster2_balanced_tail_holds_check_1, "sm90", fault=BALANCED_TAIL,
             native={"launch": "cluster2 tail", "pass": "forward"}),
        Cell("native Hopper cluster=2 balanced tail's kernel gradient holds check 1",
             native_hopper_cluster2_balanced_tail_kernel_gradient_holds_check_1, "sm90", fault=BALANCED_TAIL,
             native={"launch": "cluster2 tail", "pass": "kernel gradient"}),
    ]


PARTIAL_TILE = "a TMA copy of a partial tile, which reads zeros and clips stores where the interpreter pads"
"""Why a width no tile divides stays on the GPU (the GEMM lane, 11-11)."""

RING = "a ring's asynchronous slots, which the interpreter walks in one order"
"""Why the toolkit's rings stay on the GPU (the GEMM lane, 11-11)."""

SHARED = "a collective cp.async pipeline's meets, which the interpreter takes as no-ops (row 5272)"

MMA_SYNC = "mma.sync's products, which the interpreter sums by a dot (`mosaic_sm80.MmaIssuer.products`)"

CONVERTED = ("a quantized stage's conversion in registers (its cvt, its packed int8 assembly), which the "
             "interpreter computes in fp32 (row 8549)")
"""Why two mma.sync consumers' shared walk stays on an A100 (row 12652)."""

CLUSTER = "a cluster's multicast of a shared block, which the interpreter loads whole in each program"
"""Why a pair of multiprocessors' cooperative tile stays on the GPU (the GEMM lane, 11-12)."""

TANGENT_SUMS = declared_cells_memory.a_dense_bf16_products_tangent_sums_its_terms_before_rounding
"""The tangent's cancellation check (tests/test_declared_cells_memory.py), run on CUDA's fallback."""

EXACT_TANGENT = "an exact value its run compiles and reads, no check-1 site for a prepare to compile ahead"
"""Why the fallback's tangent cells run their programs at run (`Cell.at_run`): each asserts one exact
tangent, bitwise, where a compile-only prepare's program has no device to run on (main f37f47cdf's sm90
prepare: 'Executable' object has no attribute 'local_devices')."""

EXACT_OVERFLOW = "an exact finite value its run compiles and reads, no check-1 site for a prepare to compile"
"""Why the blockwise FP8 overflow witness runs at its run (`Cell.at_run`): it asserts one exact value of a
native launch, which a compile-only prepare's device cannot run (row 13257)."""

INLINE_ENCODE = "an encode's PTX inline in a compiled epilogue, which the interpreter runs as its evaluation"
"""Why the quantize epilogue's encode cells stay on the GPU (`_encode_cells`, row 14105)."""

EXACT_CODES = "exact codes its run compiles and reads, no check-1 site for a prepare to compile ahead"
"""Why the encode cells run their programs at run (`Cell.at_run`): each asserts every code bit for bit, a
native launch's, which a compile-only prepare's device cannot run."""

CUBLAS_BF16 = "cuBLASLt's dense bf16 product stored in bf16, which no CPU dot computes"
"""Why the XLA fallback's direct bf16 product stays on the GPU (the GEMM lane, 11-25)."""

BALANCED_TAIL = "a balanced tail's partials fixed up across programs, which the interpreter runs in one order"
"""Why a dense kernel gradient's balanced tail stays on the GPU (the GEMM lane, 11-12)."""

A100_BALANCED_TAILS = ((320, 192, 1408, (64, 128, 64, 3, 2)), (320, 384, 704, (128, 64, 64, 3, 4)))
"""Dense bf16 kernel gradients' rows, depth and width, each at a one-consumer mma.sync tile (its rows,
columns, step, stages and band; two programs a multiprocessor) whose tail an A100 of 8 multiprocessors
balances: a 128 x 128 tile holds no merge (280 registers a thread, `MosaicTgmm.balanced`)."""


def _toolkit_cells() -> list[Cell]:
    """The toolkit's rings and walks on the GPUs whose waits the interpreter takes as no-ops: an H100's rings
    and reads, an A100's shared cp.async pipeline; and the products and conversions the interpreter computes
    by a dot and in fp32: mma.sync's chained and row-bounded products, each generation's quantized stage
    (each kept by name for its fault)."""
    return [
        *(Cell(f"a ring streams tiles of their own trip counts: {consumers} consumers",
               functools.partial(mosaic_toolkit.a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100,
                                 consumers), "sm90", fault=RING)
          for consumers in (1, 2)),
        *(Cell(f"every warpgroup read lands its products: {path}",
               functools.partial(mosaic_toolkit.every_warpgroup_read_lands_its_products_on_an_h100, path),
               "sm90", fault="ptxas's fold of a read path's sign op (row 4346)")
          for path in mosaic_toolkit.READS),
        *(Cell(f"consumers feeding their own ring split a step: {consumers} consumers from {lo}",
               functools.partial(fed_walks.consumers_feeding_their_own_ring_split_a_step_on_an_h100,
                                 consumers, lo), "sm90", fault=RING)
          for consumers, lo in ((1, 0), (2, 0), (2, 3))),
        *(Cell(f"consumers sharing one cp.async pipeline: {consumers} consumers{', split' * split}",
               functools.partial(fed_walks.consumers_sharing_one_pipeline_on_an_a100, consumers, split),
               # Its cp.async copies, which Mosaic GPU lowers for the process's own device and refuses on a
               # GPU copying by TMA (`lowers_as`; train 1378's sm90 prepare).
               "sm80", mosaic=True, fault=SHARED)
          for consumers, split in ((1, False), (2, False), (2, True))),
        *(Cell(f"an accumulator feeds the next product: {rows} rows, {m_warps} warps",
               functools.partial(mosaic_toolkit.an_accumulator_feeds_the_next_product_on_a_gpu, rows,
                                 m_warps),
               "sm80", fault=MMA_SYNC)
          for rows, m_warps in mosaic_toolkit.CHAINED),
        *(Cell(f"mma.sync products within a row bound: {m_warps} warps",
               functools.partial(mosaic_toolkit.mma_sync_products_within_a_row_bound_on_a_gpu, m_warps),
               "sm80", fault=MMA_SYNC)
          for m_warps in mosaic_toolkit.BOUND_WARPS),
        *(Cell(f"a quantized stage converts natively: {name}, {jnp.dtype(dtype).name}, scaled {scaled}",
               functools.partial(mosaic_toolkit.a_quantized_stage_converts_natively, name, dtype, scaled),
               name, fault=CONVERTED)
          for name in gate_generations() for dtype in mosaic_toolkit.QUANTIZED for scaled in (True, False)),
    ]


KERNEL_PARTS = 6
"""The kernels flow's parts (`in_parts`): its prepare alone 164 s warm on a GH200's host (MAINGPU-63d4814e1),
and with the join matrix's 665 s cold in a train's prepare group (row 12759), past the train's bound; six
since its native tail and fallback cells, whose sm90 prepare group took 647 s against a 606 s bound
(row 13037), so a group packs smaller parts under it."""

JOIN_PARTS = 2
"""The join matrix's parts (`in_parts`): its prepare 153 s warm on a GH200's host (MAINGPU-63d4814e1)."""


@in_parts(KERNEL_PARTS)
def test_the_gemm_kernels_run_on_this_gpu(part):
    """The GEMM and toolkit cells above a GPU runs (`gpu_flow.covering`: each kept for a GPU-only fault by
    name, the fewest others covering its native axes, every other the CPU's, interpreted and through the
    compile gate), each one's failure kept and all raised together, a part's share of them (`in_parts`); the
    toolkit's apart (`_toolkit_cells`)."""
    run_cells(parted(covering([
        Cell("a fused projection trains its producer and epilogue",
             a_fused_projection_trains_its_producer_and_epilogue, "sm90",
             native={"launch": "fused prologue and epilogue", "pass": "training"}),
        Cell("a dense product's exact sums survive one row tile",
             when(generation.triton_runs(), "needs a GPU the Pallas kernels compile for",
                  gemm_template.a_dense_products_exact_sums_survive_one_row_tile),
             native={"launch": "dense", "pass": "forward"}),
        *_cutlass_cells(),
        *(Cell(f"the template sums 16-bit products into fp32 as exactly as XLA: depth {depth}",
               functools.partial(grouped_gemm.the_template_sums_16_bit_products_into_fp32_as_exactly_as_xla,
                                 depth),
               # Its depth native: the tensor cores' own fp32 sums over the contraction, the claim.
               native={"launch": "grouped", "pass": "forward", "dtype": "16-bit", "depth": str(depth)})
          for depth in (128, 512)),
        Cell("the template takes a gpt-oss width as exactly as XLA",
             grouped_gemm.the_template_takes_a_gpt_oss_width_as_exactly_as_xla, fault=PARTIAL_TILE),
        Cell("the template runs widths no tile divides",
             functools.partial(gemm_widths.unaligned_widths_run_on_the_template, gemm_widths.WIDTHS,
                               interpret=False), fault=PARTIAL_TILE),
        *(Cell(f"a pair's cooperative tile holds check 1 with its hooks: {case}",
               functools.partial(gemm_template.a_cooperative_tile_holds_check_1_with_its_hooks, case,
                                 gemm_template.PAIRED), "sm90", fault=CLUSTER)
          for case in ("scaled", "row scaled")),
        *(Cell(f"a dense kernel gradient's balanced tail holds check 1: {rows}x{depth}x{width} into "
               f"{jnp.dtype(dtype).name}",
               functools.partial(gemm_template.a_dense_kernel_gradients_balanced_tail_holds_check_1, rows,
                                 depth, width, dtype), "sm90", fault=BALANCED_TAIL)
          for rows, depth, width, dtype in gemm_template.BALANCED_TAILS),
        *(Cell(f"an A100's dense kernel gradient's balanced tail holds check 1: {rows}x{depth}x{width}",
               functools.partial(gemm_template.a_dense_kernel_gradients_balanced_tail_holds_check_1, rows,
                                 depth, width, jnp.bfloat16, "sm80",
                                 tiling.Tiles(*tile, warpgroups=1, cluster=1, family="mma.sync", programs=2)),
               # An A100's cp.async pipelines, which a GPU copying by TMA lowers as another kernel and refuses
               # (`lowers_as`; main's GH200 tier, rows 13127, 13129, 13130).
               "sm80", mosaic=True, fault=BALANCED_TAIL)
          for rows, depth, width, tile in A100_BALANCED_TAILS),
        *(Cell(f"a chained input gradient holds check 1: {case}",
               functools.partial(gemm_paired.a_chained_input_gradient_holds_check_1, case), name,
               native={"launch": "chained", "generation": name, "case": case})
          for name in ("sm90", "sm100") for case in gemm_paired.CHAINED_CASES),
        *(Cell("a chained input gradient's overflowing sum is walked again",
               gemm_paired.a_chained_input_gradient_is_as_finite_as_its_products_sum, name,
               native={"launch": "chained", "generation": name, "case": "overflow"})
          for name in ("sm90", "sm100")),
        *_scale_edge_cells(), *_encode_cells(),
        *_converted_cells(),
        Cell("the blockwise FP8 kernel gradient is on par",
             gemm_blockwise.a_blockwise_kernel_gradient_is_on_par, "sm90", fault="red at 475913063"),
        Cell("the blockwise FP8 kernel gradient's portable arm sums exactly",
             gemm_blockwise.a_blockwise_kernel_gradients_portable_arm_sums_exactly,
             native={"launch": "blockwise portable", "pass": "gradient"}),
        *(Cell(f"a gathered grouped product is on par{': paired and scattered' if paired else ''}",
               functools.partial(gemm_gather.a_gathered_product_is_on_par, paired=paired, scattered=paired),
               "sm90", fault="red at 475913063")
          for paired in (False, True)),
        *(Cell(f"a gathered kernel gradient is its gathered rows' gradient: {splits} splits",
               functools.partial(gemm_gather.a_gathered_kernel_gradient_is_its_gathered_rows_gradient,
                                 splits=splits), "sm90",
               fault="a cross-CTA partial combine, whose order the interpreter fixes" if splits > 1 else None,
               native={"launch": "gathered", "pass": "gradient"})
          for splits in (1, 2)),
        Cell("a Blackwell's gathered product is its gathered rows' product",
             gemm_gather.a_blackwells_gathered_product_is_its_gathered_rows_product, "sm100",
             native={"launch": "gathered", "pass": "forward"}),
        Cell("a Blackwell's gathered ring reads a strided table's entries",
             gemm_gather.a_gathered_ring_reads_a_strided_tables_entries, "sm100",
             native={"launch": "gathered", "pass": "strided table"}),
        *(Cell(f"CUTLASS runs a mapped projection as exactly as XLA where chosen: "
               f"{'shared' if shared else 'mapped'} matrices",
               functools.partial(grouped_gemm.cutlass_runs_a_mapped_projection_as_exactly_as_xla_where_chosen,
                                 shared), native={"launch": "cutlass", "groups": "mapped",
                                                  "matrices": "shared" if shared else "mapped"})
          for shared in (False, True)),
        *_toolkit_cells(),
        Cell("the shared GEMM mainloop runs on Blackwell's tcgen05",
             gemm_tcgen05.the_shared_mainloop_runs_exactly_on_blackwell, "sm100",
             native={"launch": "tcgen05 mainloop"}),
        *(Cell(f"tcgen05's epilogues hold check 1: {case} at {tiles.family} {tiles.block_n}x{tiles.stages}",
               functools.partial(gemm_tcgen05.tcgen05s_epilogues_hold_check_1_on_blackwell, case, tiles),
               "sm100", native={"launch": f"tcgen05 {tiles.family}", "epilogue": case})
          for tiles in (*gemm_tcgen05.TILES, gemm_tcgen05.PAIRED) for case in gemm_tcgen05.EPILOGUES),
        *(Cell(f"a pair's dense kernel gradient holds check 1 into {name}, k_major={k_major}",
               functools.partial(gemm_tcgen05.pairs_dense_kernel_gradient_holds_check_1_on_blackwell, dtype,
                                 k_major),
               "sm100", native={"launch": "tcgen05 tcgen05.pair", "kernel gradient": name,
                                "k_major": k_major})
          for name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16))
          for k_major in (False, True)),
        *(Cell(f"a delayed NVFP4 chain is its projections apart: {'gated' if gated else 'ungated'}",
               functools.partial(gemm_nvfp4_chain.a_delayed_nvfp4_chain_is_its_projections_apart, gated,
                                 interpret=False),
               "sm100", native={"launch": "nvfp4 chain", "epilogue": "packed codes and scales"})
          for gated in (False, True)),
        *(Cell(f"a pair's partial extents hold check 1: {rows}x{depth}x{width}{' transposed' * transposed}",
               functools.partial(gemm_tcgen05.a_pairs_partial_extents_hold_check_1_on_blackwell, rows, depth,
                                 width, transposed), "sm100", native={"launch": "tcgen05 pair partial"})
          for rows, depth, width, transposed in gemm_tcgen05.PARTIALS),
        *(Cell(f"the NVFP4 product's packed kernel holds every order: {name} {edge}",
               functools.partial(contracts_gemm.test_an_nvfp4_products_kernels_hold_every_order, edge,
                                 "packed"), name,
               fault=("the NVFP4 packed decode compiled (its traced form, row 3733)"
                      if edge == next(iter(contracts_gemm.EDGES))
                      else None),
               native={"launch": "nvfp4 packed"})
          for name in ("sm90", "sm80") for edge in contracts_gemm.EDGES),
        *_syrk_and_tail_cells(),
        *(Cell(f"the warpgroup scan walks the recurrence: {positions} positions"
               f"{' reversed' if reverse else ''}{' exclusive' if exclusive else ''}",
               functools.partial(warpgroup_scan.the_scan_walks_the_recurrence_on_this_gpu, positions, reverse,
                                 exclusive),
               native={"launch": "warpgroup scan", "reverse": str(reverse), "exclusive": str(exclusive)})
          for positions in warpgroup_scan.POSITIONS for reverse in (False, True)
          for exclusive in (False, True)),
    ]), part, KERNEL_PARTS))


def _cutlass_cells() -> tuple:
    """The GPU flow's CUTLASS cells: the projection through CUTLASS's launches against XLA's, at each
    group sizing, plain, a stack's layer and its rows scaled by a prologue."""
    return tuple(
        Cell(f"CUTLASS runs the projection as exactly as XLA where chosen: {sizes}"
             + (", a stack's layer" if stacked else "") + (", its rows scaled" if prologued else ""),
             functools.partial(grouped_gemm.cutlass_runs_the_projection_as_exactly_as_xla_where_chosen,
                               sizes, stacked, prologued),
             fault=("CUTLASS's launches, which no CPU runs"
                    if sizes == "ragged" and not (stacked or prologued) else None),
             native={"launch": "cutlass", "groups": sizes, "stacked": stacked, "prologued": prologued})
        for sizes in grouped_gemm.SIZES if any(grouped_gemm.SIZES[sizes])
        for stacked, prologued in ((False, False), (True, False), (False, True)))


@pytest.mark.users_process
def test_the_gemm_kernels_run_on_this_gpu_as_a_users_process_runs_them():
    """The GEMM cells whose claim is what a user's process computes, run in one (`users_process`: no matmul
    precision set, XLA's own schedules and autotuning as `lane_environment.USERS_GPU_FLAGS` state them), not
    in the suite's, whose JAX_DEFAULT_MATMUL_PRECISION=highest has gemm.gemm promote each step's sum: the
    gathered product at the default precision, its default form (the tensor cores' running sum) against the
    selector's XLA arm as a user's process compiles it and against cuBLAS on the same inputs (the library
    arm, row 5491: the owner's "Ofcourse Cublas parity!"), at the cooperative tile a user can pin and at the
    tile it derives. Its running sum read 1.658x XLA's default product's error, a Triton fusion on mma.sync,
    in a user's process on a GH200 (M-EM-19c), where cuBLAS's is the template's."""
    run_cells([
        Cell(f"a gathered grouped product is on par in a user's process: "
             f"{'pinned' if pinned else 'derived'}{', paired and scattered' if paired else ''}",
             functools.partial(gemm_gather.a_gathered_product_is_on_par, paired=paired, scattered=paired,
                               pinned=pinned, cublas=True), "sm90")
        for pinned in (True, False) for paired in (False, True)])


def test_the_gemm_matrix_holds_on_this_gpu():
    """GEMM's check-1 matrix (tests/kernel_matrix.py): the template route's training step on par with the
    reference path's (XLA's ragged dot), at each case a GPU-only red seeded
    (`kernel_matrix.seeded_cells`), each cell on its own GPU; its other pairs are the CPU's."""
    run_cells(kernel_matrix.seeded_cells("gemm"))


def test_the_fused_gemm_matrix_holds_on_this_gpu():
    """The fused GEMM's check-1 matrix (tests/kernel_matrix.py): a region's training step with its producer
    and consumers fused into the template's stages on par with the same region's stages and product in XLA,
    at each case a GPU-only red seeded (`kernel_matrix.seeded_cells`), each cell on its own GPU; its other
    pairs are the CPU's."""
    run_cells(kernel_matrix.seeded_cells("fused_gemm"))


def test_the_nvfp4_matrix_holds_on_this_gpu():
    """The NVFP4 product's check-1 matrix (tests/kernel_matrix.py): its training step with the matrix decoded
    in the mainloop on par with it decoded once a call, at each case a GPU-only red seeded
    (`kernel_matrix.seeded_cells`), each cell on its own GPU; its other pairs are the CPU's."""
    run_cells(kernel_matrix.seeded_cells("nvfp4"))


def test_the_mx_matrix_holds_on_this_gpu():
    """The block-scaled product's check-1 matrix (tests/kernel_matrix.py): its forward through the template's
    block-scaled products, MXFP8's and NVFP4's, on par with the decoded operands' product in XLA, over a
    covering of its cells by block format and output dtype (`gpu_flow.covering`), each on its own GPU: a
    Blackwell's block-scaled MMA and its scale loads, which no interpreter runs, though no red seeded them."""
    run_cells(covering(cells_of(
        kernel_matrix.check("mxfp8"),
        native=lambda case: dict(zip(("format", "out_dtype"), (case[1], case[4]), strict=True)))))


@in_parts(JOIN_PARTS)
def test_the_join_matrix_holds_on_this_gpu(part):
    """The joined projections' check-1 matrix (tests/join_matrix.py): a layer's training step with two of its
    projections joined, or an attention layer's query, key and value side by side, on par with them apart,
    each cell on a GPU of its generation (its pin): Hopper's side-by-side forward and chained input gradient
    on the warpgroup MMA, which the CPU only interprets (head review of 0099a4701 F6); a part's share of them
    (`in_parts`)."""
    run_cells(parted(cells_of(kernel_matrix.check("join")), part, JOIN_PARTS))


def test_the_linear_ce_matrix_holds_on_this_gpu():
    """The fused linear cross-entropy's check-1 matrix (tests/kernel_matrix.py): an LM head's loss and its
    gradients through the template's fused kernel on par with the chunked XLA program's, each cell on a GPU
    of its generation (its pin): the fused kernel's native launch, which the CPU only interprets
    (tests/test_gemm_linear_ce.py), on the GPU tier alone."""
    run_cells(cells_of(kernel_matrix.check("linear_ce")))


def test_the_blockwise_matrix_holds_on_this_gpu():
    """The blockwise-scaled FP8 product's check-1 matrix (tests/kernel_matrix.py): its forward through the
    template's step-scaled walk on par with the scaled blocks summed in XLA, at each case a GPU-only red
    seeded (`kernel_matrix.seeded_cells`), each cell on its own GPU; its other pairs are the CPU's."""
    run_cells(kernel_matrix.seeded_cells("blockwise_fp8"))


def test_the_low_rank_matrix_holds_on_this_gpu():
    """The adapted product's check-1 matrix (tests/kernel_matrix.py): a LoRA-adapted projection's training
    step with its base product and update fused on par with them apart, at each case a GPU-only red seeded
    (`kernel_matrix.seeded_cells`), each cell on its own GPU; its other pairs are the CPU's."""
    run_cells(kernel_matrix.seeded_cells("low_rank"))


def test_an_adapted_products_factor_gradients_hold_fp32_under_a_bf16_preset():
    """An adapted product of fp32 rows under the BF16_BF16_F32 preset, which a GPU's dot takes: its factor
    `a`'s gradient, `rowsᵀ · (scale · dy · bᵀ)`, keeps fp32's digits (`low_rank._kept`), so a row entry of
    `1 + 2**-20`, which bf16 rounds to 1, reaches it: exactly float64's (row 12394). Its operands are
    NumPy's and its gradient an arm (`ahead_arm`), so a GPU tier part's prepare compiles it (row 12663)."""
    from dew.nn.kernels.low_rank import low_rank

    rows = np.zeros((8, 16), np.float32)
    rows[0, 0] = 1.0 + 2.0 ** -20
    kernel, a = np.zeros((16, 8), np.float32), np.zeros((16, 4), np.float32)
    b, dy = np.zeros((4, 8), np.float32), np.zeros((8, 8), np.float32)
    b[0, 0] = dy[0, 0] = 1.0

    def factor_gradient(rows, kernel, a, b, dy):
        _, pull = jax.vjp(lambda a_: low_rank(rows, kernel, a_, b, 1.0, out_dtype=jnp.float32,
                                              precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32), a)
        return pull(dy)[0]

    d_a = ahead_arm(jax.jit(factor_gradient), "an adapted product's factor gradient under a bf16 preset")(
        rows, kernel, a, b, dy)
    want = np.zeros((16, 4), np.float32)
    want[0, 0] = 1.0 + 2.0 ** -20
    np.testing.assert_array_equal(np.asarray(d_a), want)


def test_an_adapted_products_backward_keeps_a_narrow_presets_fp32_products_wide():
    """An adapted product of fp16 rows under the F16_F16_F32 dot preset, which a GPU's dot takes: an fp32
    cotangent of 2**16 against a 2**-10 kernel, and an fp16 one of 1 through a scale of 2**16 and a factor of
    2**-10, each give the rows 64; the backward's products of fp32 operands keep them at fp32 (`_kept`), not
    narrowed back to fp16's inf (row 1304). Its operands are NumPy's and each gradient an arm of its own
    (`ahead_arm`), so a GPU tier part's prepare compiles them (row 12664)."""
    from dew.nn.kernels.low_rank import low_rank

    preset = jax.lax.DotAlgorithmPreset.F16_F16_F32
    x = np.zeros((8, 16), np.float16)
    # Every arm computed before any is checked: a GPU tier's prepare computes nothing, so a check between
    # them stops it before the next arm (row 12871).
    gradients = []
    for kernel_entry, a_entry, scale, out_dtype, cotangent in (
            (2.0 ** -10, 0.0, 1.0, np.float32, 2.0 ** 16), (0.0, 2.0 ** -10, 2.0 ** 16, np.float16, 1.0)):
        kernel = np.full((16, 8), kernel_entry, np.float16)
        a = np.full((16, 4), a_entry, np.float16)
        b = np.zeros((4, 8), np.float16)
        b[:, 0] = 1.0 if scale > 1 else 0.0
        dy = np.zeros((8, 8), out_dtype)
        dy[0, 0] = cotangent

        def rows_gradient(rows, kernel, a, b, dy, scale=scale, out_dtype=out_dtype):
            _, pull = jax.vjp(lambda rows: low_rank(rows, kernel, a, b, scale, out_dtype=out_dtype,
                                                    precision=preset), rows)
            return pull(dy)[0]

        label = f"an adapted product's rows gradient into {np.dtype(out_dtype)}"
        gradients.append((scale, out_dtype, ahead_arm(jax.jit(rows_gradient), label)(x, kernel, a, b, dy)))
    for scale, out_dtype, d_x in gradients:
        # Every feature of the cotangent's row 64 (or 256 through the scale), every other row zero: exact.
        want = np.zeros((8, 16), np.float32)
        want[0] = 64.0 * (4.0 if scale > 1 else 1.0)
        np.testing.assert_array_equal(np.asarray(d_x, np.float32), want, err_msg=str(out_dtype))
