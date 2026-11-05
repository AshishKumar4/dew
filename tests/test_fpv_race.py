"""tools/fpv_race.py's runs: a teacher's weights wrapped as the teacher phase's run of no updates, read back
as a trained teacher is, and the world phase's stages, each a record that reads back as itself."""

import json

import jax
import numpy as np
import pytest
from flax import serialization
from tool_support import load

from dew.checkpoints import Checkpoints
from dew.config import RunConfig
from dew.config.env import EnvRunConfig

fpv_race = load("fpv_race")


def test_a_wrapped_teacher_is_the_teacher_phases_run_holding_its_weights_at_step_0(tmp_path):
    """A teacher's weights written whole, wrapped (`wrap`), are the teacher phase's run at its level
    (`teacher_run`) whose step 0 holds them, every leaf bit for bit: its record reads back as that run, its
    environment, model, objective and rollout the phase's own, and its report is the phase's, unscored."""
    phase = fpv_race.teacher_run(tmp_path, hz=30.0, level=1.0, drones=16)
    weights = phase.prepare().objective.init(jax.random.key(3))
    (tmp_path / "teacher.msgpack").write_bytes(serialization.to_bytes(jax.device_get(weights)))
    report = fpv_race.wrap(tmp_path, weights=tmp_path / "teacher.msgpack", hz=30.0, level=1.0, drones=16)
    assert report["wrapped"]["step"] == 0
    saved = RunConfig.load(str(tmp_path / "teacher"))
    assert isinstance(saved, EnvRunConfig)
    for field in ("env", "model", "objective", "rollout", "optim"):
        assert getattr(saved, field) == getattr(phase, field)
    with Checkpoints(str(tmp_path / "teacher")) as checkpoints:
        assert checkpoints.latest == 0
        held = checkpoints.variables(step=0, ema=False)
    for one, other in zip(jax.tree.leaves(jax.device_get(held)), jax.tree.leaves(jax.device_get(weights)),
                          strict=True):
        np.testing.assert_array_equal(one, other)
    # The teacher phase's own report, unscored, which the phase goes on from.
    written = json.loads((tmp_path / "teacher.json").read_text())
    assert (written["curve"], written["best"], written["level"]) == ([], None, 1.0)


@pytest.mark.parametrize("stage", fpv_race.STAGES)
def test_each_world_stage_is_a_record_that_reads_back_as_itself(tmp_path, stage):
    """Each of the world phase's stages (`World.run`), its race's renderer a record HyperLux builds, reads
    back from its record as itself, at another curriculum level too (`env.difficulty`, which a stage's chunk
    sets), and every stage after the first starts from the run before it."""
    world = fpv_race.World(tmp_path, tmp_path / "teacher", tmp_path / "assets")
    run = world.run(stage, 1.0)
    assert EnvRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run
    harder = run.override({"env.difficulty": 0.5, "env.ramp": 0.5})
    assert harder.env is not None and harder.env.fields["difficulty"] == 0.5
    assert harder == world.run(stage, 0.5)
    index = fpv_race.STAGES.index(stage)
    assert run.pretrained == (None if index == 0 else str(tmp_path / fpv_race.STAGES[index - 1]))
