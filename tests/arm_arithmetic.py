"""The arithmetic a check's arms and truths run in (tests/reference_error.py): the scope each arm runs in
(`summed_as_run`) and the float64 truth's inputs (`widened`)."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.precision import FP32_SUMS


def widened(tree):
    """`tree` in float64, its integer leaves as they are: the inputs of a float64 truth. A leaf an
    accelerator holds is widened where it lies, so its truth runs on that device's FP64 units with no host
    round trip (a decode step's keys and values, a hundred megabytes a draw to the host and back); any other
    is a host array."""
    def wide(leaf):
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jnp.floating) and all(
                device.platform != "cpu" for device in leaf.devices()):
            with jax.enable_x64(new_val=True):
                return leaf.astype(jnp.float64)
        return np.asarray(leaf, np.float64 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf.dtype)

    return jax.tree.map(wide, tree)


def summed_as_run():
    """The scope a check runs each arm in (`_drawn`), the truth and its terms outside it: on the CPU lane,
    the process's default backend (`jax.default_backend()`), fp32 products summed in fp32
    (`dew.nn.precision.FP32_SUMS`), as an accelerator sums them and as a CPU does in any process but
    the suite's, whose wide arithmetic (`WIDE_CPU_ARITHMETIC`) sums Dew's products in float64, so the
    gate measures each arm as its users run it; elsewhere the process's own precision. A Dew arm that
    sums wide only under the suite's setting is no nearer float64 here than its users' runs are, and
    no route is kept off a CPU for summing narrower than that setting. The preset is JAX's default dot
    algorithm in it: every dot of no stated precision takes float32 operands, a bfloat16's exactly,
    and sums them in float32, as a GPU's fp32-accumulated product does, and a loop's running sum of
    them is the plain one (`compensates_loops`); the setting's widened reductions, which are not
    product sums, stay."""
    if jax.default_backend() == "cpu":
        return jax.default_matmul_precision(FP32_SUMS)
    return contextlib.nullcontext()
