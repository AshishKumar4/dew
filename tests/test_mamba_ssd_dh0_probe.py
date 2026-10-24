"""The standalone torch-free stage probe keeps the recorded draw and the existing float64 truth."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import widened
from ssd_support import mamba_ssd_draw, mamba_ssd_xla
from tool_support import load


@pytest.mark.parametrize("seed", [0, 3, 17])
def test_the_stage_probe_keeps_the_native_draw_and_float64_h0_truth(seed):
    probe = load("mamba_ssd_dh0_probe")
    arrays = probe.draw(seed)
    expected = mamba_ssd_draw(seed, probe.SHAPE, jnp.float32, True)
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, expected[name], err_msg=name)
    _, _, got, _ = probe.exact_stages(arrays)
    with jax.enable_x64(new_val=True):
        want = jax.jit(lambda inputs: mamba_ssd_xla(inputs, probe.CHUNK))(widened(arrays))["h0"]
    np.testing.assert_allclose(got, want, atol=1e-13, rtol=1e-13)
