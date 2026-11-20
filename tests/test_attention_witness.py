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


def test_native_attention_witness_cells_prepare_without_deferring(tmp_path, monkeypatch):
    """Row 9546: under a compile-only sm90 GPU-tier prepare (`truths_ahead.prepared`), the real native
    attention certificate cells (`test_native_certificates`) compile both causal and full certify programs
    through `ahead_arm` into completed `site.json` records with zero `DEFERRED`/`UNPREPARED`."""
    import json

    import generation_support
    import model_flows
    import test_native_certificates as certs
    import tier_wire
    from gpu_flow import run_cells
    from truths_ahead import DEFERRED, UNPREPARED, prepared

    topology = generation_support.node_topology(tmp_path / "topology.bin", gpu="sm90")
    monkeypatch.setenv(model_flows.TOPOLOGY, str(topology))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    store_dir = tmp_path / "store"
    certify_step = certs.the_attention_templates_states_certify_natively_a_step_at_a_time
    with prepared(store_dir, stamp="a", target=generation_support.node_target()):
        skipped = run_cells([
            certs.cell(certify_step, "sm90", "causal"),
            certs.cell(certify_step, "sm90", "full"),
            certs.cell(certs.the_attention_witness_refuses_its_faults_natively, "sm90"),
        ])
    assert skipped == []
    assert not (store_dir / DEFERRED).exists(), (store_dir / DEFERRED).read_text()
    assert not (store_dir / UNPREPARED).exists(), (store_dir / UNPREPARED).read_text()
    labels = {json.loads(path.read_text())["label"] for path in store_dir.rglob("site.json")}
    assert any("attention witness, causal," in label for label in labels), labels
    assert any("attention witness, full," in label for label in labels), labels
