"""Check 2's native step witness for the Mosaic attention forward (tests/attention_witness.py, row 2084 (f)),
on Mosaic GPU's interpreter: the certify launch's states and output inside their step enclosures (nulls),
and each fault the witness exists for refused at its elements: a dropped rescale, a maximum from the wrong
tile, sums rounded to bf16 where the walk holds fp32, and a mask off by one key. A GPU's native launches
run the same checks in the native certificates' flow (tests/test_native_certificates.py)."""

import attention_oracle
import jax
import numpy as np
import pytest
from attention_witness import SHAPE, assert_faults_refused, assert_witnessed, launched, operands


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("seed", [0, 1])
def test_each_stored_state_and_the_output_lie_in_their_step_enclosures(causal, seed):
    assert_witnessed(seed, causal, interpret=True)


def test_each_fault_is_refused_at_its_elements():
    assert_faults_refused(interpret=True)


def test_the_certify_launch_writes_every_record_it_declares():
    """The certify launch's records past each causal query block's walked key tiles hold the empty state its
    refs start from (`certified_walk`): under JAX's NaN check (`jax.debug_nans`), which the interpreter's
    unwritten outputs (NaN) trip, run op by op, its records bit for bit the unchecked run's (NSNAN-2)."""
    q, k, v = operands(0)
    causal = attention_oracle.kept(SHAPE[1], SHAPE[1], causal=True)
    unchecked = launched(q, k, v, causal, interpret=True)
    with jax.debug_nans(new_val=True):
        checked = launched(q, k, v, causal, interpret=True)
    for name in ("out", "m", "l", "acc"):
        assert np.asarray(checked[name]).tobytes() == np.asarray(unchecked[name]).tobytes(), name
