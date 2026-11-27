"""DC-AE's GPU tier (the owner's rule: no unit tests on GPUs, so its GPU case a file of its own, apart from
tests/test_dc_ae.py's, which a GPU tier part's compile-only prepare does not run): its eager linear attention
on a process whose default device is the accelerator (train 1403's F7)."""

import pytest
import test_dc_ae as dc_ae

pytestmark = pytest.mark.gpu(reason="a process whose default device is the accelerator the bug took")


@pytest.mark.compiles_at_run(reason="an eager call, its programs compiled where it reaches them")
def test_the_linear_attention_called_eagerly_on_cpu_operands_sums_where_they_lie_on_a_gpu():
    """On a GPU tier, whose default device is the accelerator, DC-AE's eager linear attention on operands
    committed to the CPU takes the CPU's whole sums (`dc_ae.linear_attention_sums_where_its_operands_lie`)."""
    dc_ae.linear_attention_sums_where_its_operands_lie("gpu")
