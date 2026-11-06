"""Check 2's native step witness for the Mosaic attention forward (tests/attention_witness.py, row 2084 (f)),
on Mosaic GPU's interpreter: the certify launch's states and output inside their step enclosures (nulls),
and each fault the witness exists for refused at its elements: a dropped rescale, a maximum from the wrong
tile, sums rounded to bf16 where the walk holds fp32, and a mask off by one key. A GPU's native launches
run the same checks in the native certificates' flow (tests/test_native_certificates.py)."""

import pytest
from attention_witness import assert_faults_refused, assert_witnessed


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("seed", [0, 1])
def test_each_stored_state_and_the_output_lie_in_their_step_enclosures(causal, seed):
    assert_witnessed(seed, causal, interpret=True)


def test_each_fault_is_refused_at_its_elements():
    assert_faults_refused(interpret=True)
