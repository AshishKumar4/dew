"""The GEMM template's shared mainloop (`mosaic_gpu.gmm`) on Blackwell's tcgen05 (`mosaic_sm100.RINGS`, the
fold of the old Blackwell GEMM mainloop into it, template push piece 7): interpreted as an sm100 on a CPU, and
run on a B200 or a B300 itself (tests/test_gpu_gemm.py's cell)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

from dew.nn.kernels import gemm, generation
from dew.nn.kernels.templates.gemm import mosaic_gpu


def _on_tcgen05(*, interpret: bool) -> None:
    """The shared GEMM mainloop (`mosaic_gpu.gmm`) on tcgen05's bundle (`mosaic_sm100.RINGS`), over integers
    from -1 to 1 at a depth of 256 (every sum at most 256, which a bf16 output holds exactly; an fp32 output
    of 16-bit products is exactly summed, `templates.product.promoted`, whose step tcgen05's launch does not
    take, its products issued by one warp): a grouped forward, a grouped
    input gradient (its matrix read transposed) and a dense forward, each exactly numpy's, routed to the
    template on a 128-row tile of tcgen05's; a kernel gradient, whose step prepares its operands, takes
    another route. Interpreted as an sm100 on a CPU, or run on the Blackwell GPU itself."""
    sizes = jnp.asarray([128, 200, 56], jnp.int32)
    starts = np.cumsum([0, *np.asarray(sizes)])

    def ints(key, shape):
        return jax.random.randint(jax.random.key(key), shape, -1, 2).astype(jnp.bfloat16)

    x, w, dy = ints(0, (384, 256)), ints(1, (3, 256, 256)), ints(2, (384, 256))

    def grouped(lhs, matrices):
        return np.concatenate([np.asarray(lhs[starts[g]:starts[g + 1]], np.float32)
                               @ np.asarray(matrices[g], np.float32) for g in range(3)])

    def summed(lhs, rhs, sizes, **options):
        return gemm.gemm(lhs, rhs, sizes, out_dtype=jnp.bfloat16, precision="default",
                         interpret_on_cpu=interpret, **options)

    calls = {
        "grouped forward": (lambda: summed(x, w, sizes), grouped(x, w)),
        "grouped input gradient": (lambda: summed(dy, w, sizes, transpose_rhs=True),
                                   grouped(dy, jnp.swapaxes(w, 1, 2))),
        "dense forward": (lambda: summed(x, w[0], None),
                          np.asarray(x, np.float32) @ np.asarray(w[0], np.float32)),
    }
    derived = []
    original = mosaic_gpu.gmm

    def recorded(*args, tiles, **options):
        derived.append(tiles)
        return original(*args, tiles=tiles, **options)

    for name, (call, expected) in calls.items():
        derived.clear()
        with generation.routes() as routes, pytest.MonkeyPatch.context() as patched:
            patched.setattr(mosaic_gpu, "gmm", recorded)
            got = np.asarray(jax.jit(call)(), np.float32)
        assert {route.kernel for route in routes} == {"mosaic_gpu"}, name
        assert {(tile.family, tile.block_m // tile.warpgroups) for tile in derived} == {("tcgen05", 128)}
        np.testing.assert_array_equal(got, expected, err_msg=name)
    with generation.routes() as routes:
        jax.jit(lambda: gemm.gemm_t(x, dy, sizes, out_dtype=jnp.float32, precision="default",
                                    interpret_on_cpu=interpret))()
    assert "mosaic_gpu" not in {route.kernel for route in routes}


@INTERPRETS
def test_blackwells_products_run_the_shared_mainloop_on_tcgen05():
    """On an sm100 GPU the template's products run the one GEMM mainloop on tcgen05's bundle, its
    accumulators in tensor memory, read out a chunk of columns at a time (`_on_tcgen05`, interpreted)."""
    with generation.assumed("sm100"):
        _on_tcgen05(interpret=True)


def the_shared_mainloop_runs_exactly_on_blackwell():
    """`_on_tcgen05` on a B200 or a B300 itself (tests/test_gpu_gemm.py's cell)."""
    _on_tcgen05(interpret=False)
