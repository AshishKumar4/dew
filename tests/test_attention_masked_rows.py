"""A query row that sees no key reads nothing, on every attention implementation.

Its output is exactly zero and so is its query's gradient, and it adds
nothing to any key's or value's gradient. A kernel left to such a row
softmaxes its masked logits into the mean of the values, and Splash's
backward recomputes that uniform weight into its query's and its keys'
gradients. Two ways a row sees no key: an explicit mask with an all-false
row, and a packed row's padding (segment id 0), which `document_mask` keeps
from every key. Each implementation is held to the float64 attention of
tests/test_attention_splash.py, which defines such a row as zero: in fp32 by
tests/reference_error.py's rule, as the fp32 reference attention is, and in
bf16 to the tolerance the Splash tests hold bf16 to.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
from test_attention_splash import (
    TOLERANCE,
    exact_value_and_grads,
    packed_ids,
    qkv,
    reference,
    value_and_grads,
)


def masked():
    """Rows 10-19 and 200-209 keep no key; rows 130 on keep only the first 128 keys."""
    mask = np.ones((1, 1, 256, 256), bool)
    mask[..., 130:, 128:] = False
    mask[..., 10:20, :] = False
    mask[..., 200:210, :] = False
    return {"mask": mask}, np.r_[10:20, 200:210]


def padded():
    """Two packed rows of 256 that end in padding."""
    segment_ids = packed_ids((100, 84, 50), (30, 190, 10))
    return {"causal": True, "segment_ids": segment_ids}, None


CASES = {"mask": masked, "padding": padded}
CPU = [("reference", "mask"), ("xla", "mask"), ("tpu", "mask"),
       ("reference", "padding"), ("xla", "padding"), ("tpu", "padding")]
GPU = [("cudnn", "mask"), ("cudnn", "padding"), ("flash", "padding")]


def dead_rows(structure, dead):
    """`[B, S]` true where a query row keeps no key."""
    if dead is not None:
        rows = np.zeros((2, 256), bool)
        rows[:, dead] = True
        return rows
    return np.asarray(structure["segment_ids"]) == 0


def run(implementation, case, dtype):
    structure, dead = CASES[case]()
    query, key, value = qkv((2, 256, 4, 64), dtype)
    extra = {"max_segments": 4} if implementation == "flash" else {}
    found = value_and_grads(implementation, query, key, value, **structure, **extra)
    truth = exact_value_and_grads(query, key, value, **structure)
    rows = dead_rows(structure, dead)
    assert rows.any()
    for name, array in zip(("output", "query gradient"), found[:2], strict=True):
        assert not np.any(array[rows]), f"{implementation} {case}: a row seeing no key has a nonzero {name}"
    return query, key, value, structure, found, truth


@pytest.mark.parametrize("implementation, case", CPU, ids=[f"{i}-{c}" for i, c in CPU])
def test_a_row_that_sees_no_key_reads_nothing(implementation, case):
    query, key, value, structure, found, truth = run(implementation, case, jnp.float32)
    expected = reference(query, key, value, **structure)
    for name, mine, theirs, want in zip(("output", "query", "key", "value"), found, expected, truth,
                                        strict=True):
        assert_as_exact_as_the_reference(mine, theirs, want, f"{implementation} {case} {name}")


@pytest.mark.gpu(reason="cuDNN's and flash's fused attention")
@pytest.mark.parametrize("implementation, case", GPU, ids=[f"{i}-{c}" for i, c in GPU])
def test_a_row_that_sees_no_key_reads_nothing_on_the_fused_gpu_kernels(implementation, case):
    _, _, _, _, found, truth = run(implementation, case, jnp.bfloat16)
    for name, mine, want in zip(("output", "query", "key", "value"), found, truth, strict=True):
        assert np.abs(mine - want).max() <= TOLERANCE[jnp.bfloat16] * np.abs(want).max(), (
            f"{implementation} {case} {name}")
