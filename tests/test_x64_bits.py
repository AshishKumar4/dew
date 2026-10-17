"""x64, one flag for the whole process, leaves what a model below float64 computes as it was: every
registered architecture at float32 and bfloat16 on the host CPU, where every lane runs it."""

import jax
import numpy as np
import pytest
from tiny_models import DECODERS, forward

from dew.registry import models


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("architecture", sorted(models) + sorted(DECODERS))
def test_a_model_below_float64_computes_the_same_bits_whether_x64_is_on(architecture, dtype, rng):
    """x64 is one flag for the whole process, which a float64 reference
    turns on beside the model it checks. A float32 or bfloat16 model
    computes the same bits either way: nothing it computes widens past
    float32, whatever a Python float or an index widens to under x64.

    Both run on the CPU backend, which every lane keeps beside its
    accelerator. The claim is about the arithmetic the model asks for, and
    a TPU compiles the same arithmetic apart once x64 widens its indices to
    int64, which it emulates."""
    with jax.default_device(jax.devices("cpu")[0]):
        run, operands = forward(architecture, dtype, rng)
        outputs = []
        for x64 in (False, True):
            with jax.enable_x64(x64):
                outputs.append(jax.tree.map(np.asarray, jax.jit(run)(*operands)))
    for without, under in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(outputs[1]), strict=True):
        assert under.dtype == without.dtype
        np.testing.assert_array_equal(under, without)
