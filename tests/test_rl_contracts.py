"""Policy loss terms and replicated-reader rollout contracts."""

from pathlib import Path

import jax
import pytest
from process_support import run_pool
from test_indexer_training import SEQ, deepseek_stack

from dew.nn.backbones.decoder_block import Mixture
from dew.objectives.rl import DPOObjective, GRPOObjective

WORKER = Path(__file__).with_name("rl_contracts_worker.py")


@pytest.mark.parametrize("objective", [GRPOObjective, DPOObjective])
def test_policy_losses_take_the_scorers_options_and_train_the_whole_decoder(objective):
    """A policy or preference loss over a routed decoder with prediction depths takes the token
    scorer's options alone: the head's chunks, tile, pad id and reports, and every parameter
    trains, its reference frozen beside them where it reads one."""
    model = deepseek_stack(None, mixture=Mixture(experts=4, top_k=2), num_nextn_predict_layers=1)
    built = objective(model, SEQ, head_chunks=2, head_tile="tiled", pad_id=0, qk_stats=False,
                      token_accuracy=False)
    assert (built.head_chunks, built.pad_id, built.token_accuracy) == (2, 0, False)
    variables = built.init(jax.random.key(0))
    assert set(variables) == {"params"}


@pytest.mark.mesh
@pytest.mark.parametrize("mode", ["native", "ppo"])
def test_episode_rollouts_refuse_replicated_readers_before_any_tool_runs(tmp_path, mode):
    reports = run_pool(mode, tmp_path, 2, script=WORKER, devices=1, timeout=120)
    for report in reports:
        assert report["readers"] == 2 and report["shares"] == 1
        assert report["refused"] and report["opened"] == report["closed"] == 0
        assert report["verified"] == 0
