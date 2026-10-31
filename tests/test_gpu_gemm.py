"""The GEMM template and the Mosaic toolkit on the GPU (G14, the owner's rule: no unit tests on GPUs), run in
one process (`gpu_flow`), each cell its own file's property:

- A dense product's exact sums survive one row tile through Triton's MMA accumulation (test_gemm_template).
- CUTLASS runs the grouped projection as exactly as XLA wherever it is chosen, over every grouping that
  routes a row (test_grouped_gemm).
- A ring streams tiles of their own trip counts, and consumers feeding their own ring split a step, on an
  H100 (test_mosaic_toolkit).
- The Blackwell grouped matmul holds check 1 on a B200 or a B300, sm100 and sm103 alike
  (test_gemm_blackwell).
- The toolkit's warpgroup scan walks a linear recurrence forward and reversed, each position's state or
  the one before it, as float64 does (test_warpgroup_scan).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_gemm_blackwell as gemm_blackwell
import test_gemm_template as gemm_template
import test_grouped_gemm as grouped_gemm
import test_mosaic_toolkit as mosaic_toolkit
import test_warpgroup_scan as warpgroup_scan
from gpu_flow import Cell, cells_of, run_cells
from jax.experimental.pallas import fuser
from reference_error import assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates.gemm import fused
from dew.nn.norms import rms_normalized

pytestmark = pytest.mark.gpu(reason="the GEMM template's and the Mosaic toolkit's kernels, compiled")


def _fused_projection_training():
    """One routed projection update, including an empty expert and rows beyond the routed ones. The
    input cotangent exercises both paths through RMS normalization, and the updated parameters run
    another forward. The reference is the plain program main's GEMM tests take: the RMS norm as a model
    writes it and XLA's ragged product (`test_gemm_template`), outside the template."""
    rows, depth, width = 256, 256, 256
    sizes = (0, 100, 150)
    groups = jnp.asarray(sizes, jnp.int32)

    def native(x, params):
        normalized = rms_normalized(x, params["norm"], 1e-6, x.dtype, scale_offset=False,
                                    scale_after_cast=False, fp32_statistics=True, implementation="xla")
        projected = fused.fusible_gemm(normalized, params["weight"].astype(x.dtype), groups)
        return jax.nn.silu(projected + params["bias"]) + x[:, :width].astype(jnp.float32)

    def reference(x, params):
        work = jnp.promote_types(x.dtype, jnp.float32)
        normalized = gemm_template.rms_norm(x, params["norm"])
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
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16).at[:2].set(0)
        # Master weights begin at representable operand values; the update itself stays in fp32.
        weight = jnp.asarray(rng.normal(size=(len(sizes), depth, width)) / np.sqrt(depth),
                             jnp.bfloat16).astype(jnp.float32)
        params = {"weight": weight, "norm": jnp.asarray(rng.uniform(0.8, 1.2, depth), jnp.float32),
                  "bias": jnp.asarray(rng.normal(size=width) * 0.1, jnp.float32)}
        return x, params, jnp.asarray(rng.normal(size=(rows, width)), jnp.float32)

    return step(fuser.fuse(native)), step(reference), draw


def a_fused_projection_trains_its_producer_and_epilogue():
    native, reference, draw = _fused_projection_training()
    with generation.chosen(gemm_prologue="fused", gemm_epilogue="fused"):
        assert_as_exact(jax.jit(native), jax.jit(reference), draw,
                        "fused projection producer and epilogue update", seeds=8, truth_fn=reference)


def test_the_gemm_kernels_run_on_this_gpu():
    """Every GEMM and toolkit cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all
    raised together; the toolkit's on an H100 alone."""
    def on_triton():
        if not generation.triton_runs():
            pytest.skip("needs a GPU the Pallas kernels compile for")
        gemm_template.a_dense_products_exact_sums_survive_one_row_tile()

    run_cells([
        Cell("a fused projection trains its producer and epilogue",
             a_fused_projection_trains_its_producer_and_epilogue, "sm90"),
        Cell("a dense product's exact sums survive one row tile", on_triton),
        *(Cell(f"CUTLASS runs the projection as exactly as XLA where chosen: {sizes}",
               functools.partial(grouped_gemm.cutlass_runs_the_projection_as_exactly_as_xla_where_chosen,
                                 sizes))
          for sizes in grouped_gemm.SIZES if any(grouped_gemm.SIZES[sizes])),
        *(Cell(f"a ring streams tiles of their own trip counts: {consumers} consumers",
               functools.partial(mosaic_toolkit.a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100,
                                 consumers), "sm90")
          for consumers in (1, 2)),
        *(Cell(f"consumers feeding their own ring split a step: {consumers} consumers from {lo}",
               functools.partial(mosaic_toolkit.consumers_feeding_their_own_ring_split_a_step_on_an_h100,
                                 consumers, lo), "sm90")
          for consumers, lo in ((1, 0), (2, 0), (2, 3))),
        *cells_of(gemm_blackwell.the_blackwell_gmm_holds_check_1_on_blackwell),
        *(Cell(f"the warpgroup scan walks the recurrence: {positions} positions"
               f"{' reversed' if reverse else ''}{' exclusive' if exclusive else ''}",
               functools.partial(warpgroup_scan.the_scan_walks_the_recurrence_on_this_gpu, positions, reverse,
                                 exclusive))
          for positions in warpgroup_scan.POSITIONS for reverse in (False, True)
          for exclusive in (False, True)),
    ])
