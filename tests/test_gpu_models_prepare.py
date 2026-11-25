"""A GPU tier part's prepare of `tests/test_gpu_models.py`'s cells, emulated on a CPU: each arm of
`a_bf16_decoder_trains_and_serves_through_its_gpu_kernels` and `modernbert_states_in_bf16_on_the_gpu_kernels`
stated alike in the prepare and on another device's run, so the run finds what its prepare compiled (rows
11967, 11013).
"""

import os
import subprocess
import sys
from pathlib import Path

WORKER = r"""
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import model_flows
import numpy as np
import test_causal_transformer as causal
import test_hf_modernbert as modernbert
import tier_wire
from arm_arithmetic import drawn_on_the_host, widened
from arm_store import _Ahead, _compile_arms, _stored_arm, armed
from generation_support import node_target, node_topology
from gpu_flow import Cell, run_cells
from truths_ahead import (
    DEFERRED,
    PROGRAMS,
    UNPREPARED,
    CompiledTruths,
    arm_role,
    compiled_truths,
    named,
    prepared,
)

from dew.training.ahead import AheadPrograms

base = Path(sys.argv[1])
cpus = jax.devices("cpu")
assert len(cpus) >= 2, f"expected >=2 cpu devices, got {cpus}"

CAUSAL_CELL = "a bf16 decoder trains on a packed batch and serves its cache through the GPU's kernels"

# 1. Real sm90 prepare of both GPU model cells (pattern of test_flow_cells.py).
sm90_store = base / "sm90_store"
os.environ[model_flows.TOPOLOGY] = str(node_topology(base / "topology.bin", gpu="sm90"))
os.environ[tier_wire.PREPARE] = "1"
os.environ["PYTEST_CURRENT_TEST"] = (
    "tests/test_gpu_models.py::test_the_models_run_their_gpu_kernels_as_a_users_process_does (call)"
)
with prepared(sm90_store, stamp="a", target=node_target()):
    run_cells([
        Cell(CAUSAL_CELL, causal.a_bf16_decoder_trains_and_serves_through_its_gpu_kernels),
        Cell("ModernBERT's states in bf16 on its GPU kernels",
             modernbert.modernbert_states_in_bf16_on_the_gpu_kernels),
    ])

ro = CompiledTruths(sm90_store, prepare=False, stamp="a")
assert ro.holds("a bf16 decoder on its GPU kernels"), "causal truths not prepared"
assert ro.holds(armed("ModernBERT bf16 states")), "ModernBERT bf16 dew arm not prepared"

# 2. Capture both cells' arms in PREPARE (cpus[0]) and RUN (cpus[1]) and prove
#    every arm has a statement and keeps the same arm_role across devices.
captured = []

def capture(dew_fn, ref_fn, draw, label, **kw):
    captured.append((dew_fn, ref_fn, draw, label, kw))

causal.assert_as_exact = capture
modernbert.assert_as_exact = capture

os.environ[tier_wire.PREPARE] = "1"
with jax.default_device(cpus[0]):
    causal.a_bf16_decoder_trains_and_serves_through_its_gpu_kernels()
    modernbert.modernbert_states_in_bf16_on_the_gpu_kernels()
prep_causal, prep_mb = captured[0], captured[1]

captured.clear()
os.environ.pop(tier_wire.PREPARE, None)
with jax.default_device(cpus[1]):
    causal.a_bf16_decoder_trains_and_serves_through_its_gpu_kernels()
    modernbert.modernbert_states_in_bf16_on_the_gpu_kernels()
run_causal, run_mb = captured[0], captured[1]

c_label = prep_causal[3]
c_dew_prep = arm_role("dew", prep_causal[0], c_label)
c_dew_run = arm_role("dew", run_causal[0], c_label)
assert c_dew_prep is not None, "causal dew arm has no statement"
assert c_dew_prep == c_dew_run, f"causal dew arm_role drifted across devices: {c_dew_prep} != {c_dew_run}"

c_ref_prep = arm_role("reference", prep_causal[1], c_label)
c_ref_run = arm_role("reference", run_causal[1], c_label)
assert c_ref_prep is not None, "causal reference arm has no statement"
assert c_ref_prep == c_ref_run, f"causal reference arm_role drifted across devices: {c_ref_prep}, {c_ref_run}"

m_label = prep_mb[3]
m_dew_prep = arm_role("dew", prep_mb[0], m_label)
m_dew_run = arm_role("dew", run_mb[0], m_label)
assert m_dew_prep is not None, "ModernBERT dew arm has no statement"
assert m_dew_prep == m_dew_run, f"ModernBERT dew arm_role drifted across devices: {m_dew_prep} != {m_dew_run}"

# The causal site's sm90 programs: where its prepare compiled them ahead, both its arms' by the keys the run
# looks them up by, never one arm's without its twin's; or its site marked compiled at run, each arm compiled
# on the GPU (`CompiledTruths.at_run`: its cuDNN launch lowering on no deviceless prepare).
# Or its cell deferred whole to its run, recorded so (`gpu_flow.run_cells`, `DEFERRED`: its cuDNN attention
# lowering on no deviceless prepare), its arms' site never reached, each arm compiled on the GPU.
c_sample = drawn_on_the_host(run_causal[2], 0)
if not ro.holds(armed(c_label)):
    deferred = json.loads((sm90_store / DEFERRED).read_text())
    assert any(reason["error"].startswith(CAUSAL_CELL) for reason in deferred["reasons"]), deferred
elif ro.at_run(armed(c_label)) is None:
    c_programs = AheadPrograms.load(ro.path(armed(c_label)) / PROGRAMS).programs
    for role_name in (c_dew_run, c_ref_run):
        assert named(role_name, c_sample) in c_programs, (named(role_name, c_sample), sorted(c_programs))

# Confirm ModernBERT's run-side program name matches what sm90 prepare stored.
mb_sample = drawn_on_the_host(run_mb[2], 0)
mb_programs = AheadPrograms.load(ro.path(armed(m_label)) / PROGRAMS).programs
assert named(m_dew_run, mb_sample) in mb_programs, (
    f"ModernBERT sm90 prepare missed run key {named(m_dew_run, mb_sample)} in {sorted(mb_programs)}"
)

# 3. End-to-end CPU prepare (cpus[0]) -> CPU run (cpus[1]) for causal reference arm + truth
#    and ModernBERT dew arm: hits stored executables with zero MissingTruth and zero UNPREPARED.
cpu_store = base / "cpu_store"
os.environ[tier_wire.PREPARE] = "1"
with jax.default_device(cpus[0]), compiled_truths(cpu_store, prepare=True, stamp="cpu") as store:
    _compile_arms(store, {"reference": (prep_causal[1], True)}, prep_causal[2], c_label,
                  seeds=1, orders=None, compile=jax.jit)
    _compile_arms(store, {"dew": (prep_mb[0], False)}, prep_mb[2], m_label,
                  seeds=1, orders=None, compile=jax.jit)

os.environ.pop(tier_wire.PREPARE, None)
with jax.default_device(cpus[1]), compiled_truths(cpu_store, prepare=False, stamp="cpu") as store:
    stored_ref = _stored_arm(store, c_label, run_causal[1], run_causal[1], "reference")
    assert isinstance(stored_ref, _Ahead), "causal reference arm ran unprepared"
    ref_out = stored_ref(*c_sample)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(ref_out))

    # Also verify float64 truth executes cleanly with the host NumPy batch/cotangent.
    with jax.enable_x64(True):
        truth_out = run_causal[4]["truth_fn"](*widened(c_sample))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(truth_out))

    stored_mb = _stored_arm(store, m_label, run_mb[0], run_mb[0], "dew")
    assert isinstance(stored_mb, _Ahead), "ModernBERT dew arm ran unprepared"
    mb_out = stored_mb(*mb_sample)
    assert np.all(np.isfinite(np.asarray(mb_out)))

assert not (cpu_store / UNPREPARED).exists(), (cpu_store / UNPREPARED).read_text()
"""


def test_gpu_models_cells_prepare_and_match_arm_roles_across_devices(tmp_path):
    """Both `a_bf16_decoder_trains_and_serves_through_its_gpu_kernels` and
    `modernbert_states_in_bf16_on_the_gpu_kernels` state every arm without device sharding so prepare
    (cpu:0) and run (a distinct device) compute identical `arm_role` fingerprints with no MissingTruth."""
    flags = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2").strip()
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, str(tmp_path)],
        cwd=Path(__file__).parent,
        env={**os.environ, "XLA_FLAGS": flags},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout[-1500:]}\nSTDERR:\n{proc.stderr[-1500:]}"
