"""tools/rl_acceptance.py's environment runs: each acceptance run a record that reads back as itself, and the
report's curve and the returns of its scored splits read from the tracking its run wrote."""

import dataclasses
import json
import time

import numpy as np
import pytest
from test_env_runs import ppo
from tool_support import load

from dew.objectives.rl.config import EnvRunConfig, EpisodesConfig
from dew.training import LocalTracker

rl_acceptance = load("rl_acceptance")


@pytest.mark.parametrize("make", [rl_acceptance.ppo_run, rl_acceptance.dreamer_run],
                         ids=lambda make: make.__name__)
def test_an_acceptance_run_reads_back_from_its_record_as_itself(tmp_path, make):
    """The PPO and DreamerV3 acceptance runs, at their defaults and at another task and size, each read
    back from their JSON record as the run they were."""
    for run in (make(tmp_path), make(tmp_path, task="CartpoleSwingup", envs=4)):
        assert EnvRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run


def test_a_runs_report_reads_its_curve_and_each_splits_return_from_the_tracking_it_wrote(tmp_path):
    """A run's training reports are read as the acceptance report holds them: one row a logged update,
    its step and the seconds since the run began, its scalars by their own names with `train/` and
    `rollout/` dropped; and the return each scored split logged once at the run's end, in the order asked,
    which no row of the curve holds."""
    run, _, _ = ppo(tmp_path)
    splits = {rl_acceptance.DRAWN: EpisodesConfig(16, drawn=True), rl_acceptance.GREEDY: EpisodesConfig(16)}
    prepared = dataclasses.replace(run, episodes=splits).prepare()
    began = time.time()
    prepared.train("curve")
    ended = time.time()
    curve, (drawn, greedy) = rl_acceptance.tracked(tmp_path / "curve", began, rl_acceptance.DRAWN,
                                                   rl_acceptance.GREEDY)
    with LocalTracker(tmp_path / "curve" / "tracking") as journal:
        logged = journal.scalars()
    trained = [(step, scalars) for step, scalars in logged if "train/loss" in scalars]
    assert len(curve) == 3
    assert [row["step"] for row in curve] == [step for step, _ in trained]
    for row, (_, scalars) in zip(curve, trained, strict=True):
        assert 0 <= row["seconds"] <= ended - began
        np.testing.assert_equal(row["loss"], scalars["train/loss"])
        np.testing.assert_equal(row["episode_return"], scalars["rollout/episode_return"])
    assert [row["seconds"] for row in curve] == sorted(row["seconds"] for row in curve)
    scored = {name: value for _, scalars in logged for name, value in scalars.items()
              if name.endswith("/return")}
    assert scored == {"drawn/return": drawn, "greedy/return": greedy}
    assert all(not set(row) & set(scored) for row in curve)
