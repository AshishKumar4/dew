"""The GEMM template on widths and depths no tile divides (`grouped_matmul._Mosaic._aligned`): the call
run padded with zeros to whole tiles of its family and its outputs sliced back, as Triton's masked tiles
ran it."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS

from dew.nn.kernels import gemm, generation


@INTERPRETS
@pytest.mark.parametrize("generation_name", ["sm80", "sm90", "v5e"])
@pytest.mark.parametrize(("depth", "width"), [(96, 100), (200, 72)], ids=["96x100", "200x72"])
def test_a_product_of_unaligned_widths_runs_on_the_template(generation_name, depth, width):
    """A grouped forward, its input gradient and its kernel gradient, and a dense forward, at a depth and a
    width no tile of the family divides, the template's launch named (`'mosaic'`): over integers from -1 to
    1, every product and sum exact, bitwise numpy's, groups of rows no tile divides and one empty."""
    sizes = [70, 0, 129, 57]
    starts = np.cumsum([0, *sizes])
    keys = jax.random.split(jax.random.key(4), 3)

    def ints(key, shape):
        return jax.random.randint(key, shape, -1, 2).astype(jnp.bfloat16)

    x, w, dy = ints(keys[0], (sum(sizes), depth)), ints(keys[1], (len(sizes), depth, width)), ints(
        keys[2], (sum(sizes), width))
    group_sizes = jnp.asarray(sizes, jnp.int32)
    options = {"out_dtype": jnp.float32, "precision": "default", "interpret_on_cpu": True}

    def grouped(lhs, matrices):
        return np.concatenate([np.asarray(lhs[a:b], np.float32) @ np.asarray(matrices[g], np.float32)
                               for g, (a, b) in enumerate(itertools.pairwise(starts))])

    calls = {
        "grouped forward": (lambda: gemm.gemm(x, w, group_sizes, **options), grouped(x, w)),
        "grouped input gradient": (lambda: gemm.gemm(dy, w, group_sizes, transpose_rhs=True, **options),
                                   grouped(dy, jnp.swapaxes(w, 1, 2))),
        "grouped kernel gradient": (lambda: gemm.gemm_t(x, dy, group_sizes, **options), np.stack(
            [np.asarray(x[a:b], np.float32).T @ np.asarray(dy[a:b], np.float32)
             for a, b in itertools.pairwise(starts)])),
        "dense forward": (lambda: gemm.gemm(x, w[0], None, **options),
                          np.asarray(x, np.float32) @ np.asarray(w[0], np.float32)),
    }
    with generation.assumed(generation_name), generation.chosen(grouped_matmul="mosaic"):
        for name, (call, expected) in calls.items():
            with generation.routes() as routes:
                got = np.asarray(jax.jit(call)(), np.float32)
            assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}, name
            np.testing.assert_array_equal(got, expected, err_msg=name)
