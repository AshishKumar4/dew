"""Policy loss terms and replicated-reader rollout contracts."""

from pathlib import Path

import numpy as np
import pytest
from process_support import run_pool
from test_indexer_training import SEQ, deepseek_stack

from dew.nn.backbones.decoder_block import Mixture
from dew.objectives.lm import IndexerTraining
from dew.objectives.rl import DPOObjective, GRPOObjective

WORKER = Path(__file__).with_name("rl_contracts_worker.py")


@pytest.mark.parametrize("objective", [GRPOObjective, DPOObjective])
@pytest.mark.parametrize("options", [
    {"balance_rate": 1e-3}, {"aux_loss_alpha": .01}, {"mtp_weight": .1},
    {"z_loss": 1e-4}, {"router_z_loss": .01},
    {"indexer": IndexerTraining("warmup")}, {"indexer": IndexerTraining("sparse")},
])
def test_policy_losses_refuse_unapplied_lm_training_terms(objective, options):
    indexer = options.get("indexer")
    model = deepseek_stack(4 if indexer is not None and indexer.phase == "sparse" else None,
                           mixture=Mixture(experts=4, top_k=2), num_nextn_predict_layers=1)
    with pytest.raises(ValueError):
        objective(model, SEQ, **options)


@pytest.mark.parametrize("objective", [GRPOObjective, DPOObjective])
def test_policy_losses_accept_explicit_lm_training_defaults(objective):
    built = objective(deepseek_stack(None), SEQ, balance_rate=None, aux_loss_alpha=None,
                      mtp_weight=None, z_loss=0., router_z_loss=0., indexer=None)
    assert built.phase is None and built.z_loss == built.router_z_loss == 0


@pytest.mark.mesh
def test_native_episodes_refuse_replicated_readers_before_any_tool_runs(tmp_path):
    reports = run_pool("native", tmp_path, 2, script=WORKER, devices=1, timeout=120)
    for report in reports:
        assert report["readers"] == 2 and report["shares"] == 1
        assert report["refused"] and report["opened"] == report["closed"] == 0
        assert report["verified"] == 0


@pytest.mark.mesh
def test_ppo_whitening_counts_each_replicated_share_once(tmp_path):
    single, = run_pool("ppo", tmp_path / "single", 1, script=WORKER, devices=2, timeout=120)
    reports = run_pool("ppo", tmp_path / "pool", 2, script=WORKER, devices=1, timeout=120)
    assert single["readers"] == 1 and all(report["readers"] == 2 for report in reports)
    assert np.max(np.abs(single["actor_gradient"])) > 1e-6
    for report in reports:
        assert not report["refused"]
        for name in ("advantages", "returns", "old_values", "actor_gradient"):
            np.testing.assert_allclose(report[name], single[name], rtol=2e-5, atol=2e-6)


@pytest.mark.mesh
def test_ppo_refuses_one_unique_action_even_with_two_readers(tmp_path):
    reports = run_pool("single_action", tmp_path, 2, script=WORKER, devices=1, timeout=120)
    assert all(report["refused"] and report["readers"] == 2 for report in reports)
