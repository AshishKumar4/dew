"""Check 2's native witness for the Mosaic attention template (tests/attention_witness.py, rows 2084 (f) and
6130), on Mosaic GPU's interpreter: the forward's certify launch's states and output inside their step
enclosures (nulls), full, causal, windowed, over packed documents and capped, and each fault the witness
exists for refused at its elements: a dropped rescale, a maximum from the wrong tile, sums rounded to bf16
where the walk holds fp32, a mask off by one key, a window's or the documents' records held as one
undivided call's, and a cap's read; and the fast backward's gradients inside their enclosures, capped
too, its faults (a wrong scale, a sixty-fourth off, a mask, a cap) refused. A GPU's native launches run
the same checks in the native certificates' flow (tests/test_native_certificates.py)."""

import jax
import numpy as np
import pytest
from attention_witness import (
    BACKWARD_FAULTS,
    FAULTS,
    FORMS,
    TWO_LAUNCH_FAULTS,
    assert_backward_faults_refused,
    assert_backward_witnessed,
    assert_equal_totals_held,
    assert_faults_refused,
    assert_two_launches_held_to_their_own_d,
    assert_witnessed,
    launched,
    operands,
)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("seed", [0, 1])
def test_each_stored_state_and_the_output_lie_in_their_step_enclosures(form, seed):
    assert_witnessed(seed, form, interpret=True)


@pytest.mark.parametrize("fault", FAULTS)
def test_each_fault_is_refused_at_its_elements(fault):
    assert_faults_refused(interpret=True, faults=(fault,))


def test_the_certify_launch_writes_every_record_it_declares():
    """The certify launch's records past each causal query block's walked key tiles hold the empty state its
    refs start from (`certified_walk`): under JAX's NaN check (`jax.debug_nans`), which the interpreter's
    unwritten outputs (NaN) trip, run op by op, its records bit for bit the unchecked run's (NSNAN-2)."""
    q, k, v = operands(0)
    unchecked = launched(q, k, v, "causal", interpret=True)
    with jax.debug_nans(new_val=True):
        checked = launched(q, k, v, "causal", interpret=True)
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
    from tier_wire import DEFERRED
    from truths_ahead import UNPREPARED, prepared

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
    for expected in ("attention witness, witness, causal,", "attention witness, witness, full,",
                     "attention witness, faults, causal,", "attention witness, faults, full,"):
        assert any(expected in label for label in labels), (expected, labels)


@pytest.mark.parametrize("accurate", [False, True], ids=["fast", "accurate"])
@pytest.mark.parametrize("form", FORMS)
def test_each_gradient_of_the_backward_lies_in_its_enclosure(form, accurate):
    assert_backward_witnessed(0, form, interpret=True, accurate=accurate)


@pytest.mark.parametrize("form", FORMS)
def test_each_gradient_of_the_two_launch_backward_lies_in_its_enclosure(form, with_deterministic_ops):
    """Row 13697: under --xla_gpu_deterministic_ops, the GPU tier's lane, the fast backward takes two
    launches (`MosaicBackward.legal` refuses the fused one), its dK/dV launch over the dQ launch's own
    denominator and D refined from its sums (`gradient.dq_finish`): each gradient inside the enclosure of
    the schedule that ran, as its rows state it."""
    assert_backward_witnessed(0, form, interpret=True)


@pytest.mark.parametrize("fault", [None, *TWO_LAUNCH_FAULTS], ids=["held", *TWO_LAUNCH_FAULTS])
def test_two_launches_are_held_to_their_own_refined_d(fault, with_deterministic_ops):
    """Two launches' gradients held to their own schedule, and each fault of them refused, as the fused
    launch's (its D and denominator) at dK, apart (`attention_witness.TWO_LAUNCH_FAULTS`)."""
    assert_two_launches_held_to_their_own_d(interpret=True, held=fault is None,
                                            faults=() if fault is None else (fault,))


def test_two_launches_of_the_forwards_own_denominator_are_held_by_their_schedule(with_deterministic_ops):
    """Two launches whose denominators equal the forward's to the bit, their D refined all the same, held."""
    assert_equal_totals_held(interpret=True)


@pytest.mark.parametrize("fault", BACKWARD_FAULTS)
def test_each_backward_fault_is_refused_at_its_elements(fault):
    assert_backward_faults_refused(interpret=True, faults=(fault,))
