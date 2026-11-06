"""The native release of a trained model (`dew.inference.models.SavedModel`, row 4313): a JEPA encoder, a
control policy and a DreamerV3 agent, objectives with no inference task of their own, export as a release
their run's record states and load back as the model and weights the run trained, where their checkpoints
recorded no declaration and `SavedRun.at` refused them."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recording import bits
from run_kinds import jepa, trained
from test_env_runs import TEACHER, student, teacher_run

from dew.checkpoints import Checkpoints
from dew.config.release import RELEASE_FILE, Release
from dew.inference.loading import pipeline
from dew.inference.tasks import SavedRun
from dew.interop.saved_sources import export_run


def stored(directory: Path, *, ema: bool) -> dict:
    """The variables the run in `directory` kept at its last step, as its checkpoints read them."""
    with Checkpoints(str(directory)) as checkpoints:
        return checkpoints.variables(step=None, ema=ema)


def assert_same(one, other) -> None:
    """Two variable trees of one structure, leaf for leaf bit for bit."""
    assert jax.tree.structure(one) == jax.tree.structure(other)
    for left, right in zip(bits(one), bits(other), strict=True):
        np.testing.assert_array_equal(left, right)


def test_a_jepa_run_exports_and_loads_as_its_encoder(tmp_path):
    """A JEPA run pins as a declaration (its encoder's record and its numerics, its settings none) and loads
    as a `SavedModel` of its encoder, the target encoder's average where the run kept one and the encoder's
    own part of the trained tree, not the predictor's; `dew export` writes it, and `dew.pipeline` of the
    directory loads the model rebuilt from the release record and the weights written, bit for bit, which
    encode a batch as the run's encoder does."""
    from dew.inference.models import SavedModel

    trained(jepa(tmp_path, drawn=False), tmp_path, "encoder", trainer={"name": "encoder",
                                                                       "checkpoint_final": True})
    run = SavedRun.pinned(str(tmp_path / "runs" / "encoder"))
    assert run.record["model"]["name"].endswith("JepaEncoder")
    loaded = SavedModel.from_saved(run, ema=True)
    assert set(loaded.variables["params"]) != {"context_encoder", "predictor"}
    assert_same(loaded.variables, {collection: tree["context_encoder"] for collection, tree in
                                   stored(tmp_path / "runs" / "encoder", ema=True).items()
                                   if "context_encoder" in tree})
    export_run(run, tmp_path / "release", ema=True)
    again = pipeline(str(tmp_path / "release"))
    assert isinstance(again, SavedModel) and again.model == loaded.model
    assert again.objective is loaded.objective and again.settings == loaded.settings
    assert_same(again.variables, loaded.variables)
    sample = jnp.ones((2, 16, 16, 3))
    indices = jnp.arange(4, dtype=jnp.int32)[None].repeat(2, 0)
    np.testing.assert_array_equal(again.model.apply(again.variables, sample, indices),
                                  loaded.model.apply(loaded.variables, sample, indices))


def test_a_control_policy_run_exports_with_the_observation_policy_it_acts_by(tmp_path):
    """A control policy's release keeps its model whole (its actor and critic, the whole trained tree) and
    the observation policy its objective acts by, the width of the observation and the views it joins
    (`ControlSettings`), beside the action width its model states, so the loaded policy acts as the trained
    one does on the same observations: the environment run's own reconstruction of its agent
    (`EnvRunConfig.agent`) is no longer the only way back from its checkpoints."""
    from dew.inference.models import SavedModel
    from dew.objectives.rl.control import ControlSettings

    teacher_run(tmp_path).prepare().train("teacher")
    directory = tmp_path / "teacher"
    run = SavedRun.pinned(str(directory))
    loaded = SavedModel.from_saved(run, ema=False)
    assert loaded.settings == ControlSettings(observation_size=TEACHER, reads=("teacher",))
    assert_same(loaded.variables, stored(directory, ema=False))
    export_run(run, tmp_path / "policy", ema=False)
    again = pipeline(str(tmp_path / "policy"), param_dtype="auto")
    assert isinstance(again, SavedModel) and again.settings == loaded.settings
    observation = jnp.linspace(-1.0, 1.0, 3 * again.settings.observation_size).reshape(3, -1)
    assert_same(again.model.apply(again.variables, observation),
                loaded.model.apply(loaded.variables, observation))


def test_a_dreamer_run_exports_its_world_model_and_agent_whole(tmp_path):
    """A DreamerV3 agent's release is its world model with the actor, critic and the returns' percentiles
    its tree holds, rebuilt from the record its run states (the observation widths and action width its
    environment derived are in the model's own record), its settings none: the loaded tree is the run's,
    bit for bit."""
    from dew.inference.models import SavedModel

    student(tmp_path, 2).prepare().train("dreamer")
    directory = tmp_path / "dreamer"
    run = SavedRun.pinned(str(directory))
    loaded = SavedModel.from_saved(run, ema=False)
    assert "returns" in loaded.variables and loaded.settings == type(loaded.settings)()
    assert_same(loaded.variables, stored(directory, ema=False))
    export_run(run, tmp_path / "agent", ema=False)
    again = pipeline(str(tmp_path / "agent"))
    assert isinstance(again, SavedModel) and again.model == loaded.model
    assert_same(again.variables, loaded.variables)
    # Stored in bfloat16, its parameters are, and the returns' percentiles keep the dtype written, as a
    # load of the run's own checkpoint keeps them.
    half = pipeline(str(tmp_path / "agent"), param_dtype="bfloat16")
    direct = SavedModel.from_saved(run, ema=False, param_dtype="bfloat16")
    assert {leaf.dtype for leaf in jax.tree.leaves(half.variables["params"])} == {jnp.dtype(jnp.bfloat16)}
    assert ({leaf.dtype for leaf in jax.tree.leaves(half.variables["returns"])}
            == {leaf.dtype for leaf in jax.tree.leaves(direct.variables["returns"])}
            == {leaf.dtype for leaf in jax.tree.leaves(loaded.variables["returns"])})


def test_a_native_release_loads_in_the_dtype_asked_and_refuses_a_source_it_is_not(tmp_path):
    """A native release stores its weights in the `param_dtype` a load asks, as every task's load does,
    and computes in the `dtype` it asks where its model declares one: a control policy, which declares
    none (its rollouts' actions are float32 throughout), is refused a bfloat16 compute. A record that
    states no run (a source's own files) is no native release: `SavedModel.from_pretrained` of it is
    refused."""
    from dew.inference.models import SavedModel

    teacher_run(tmp_path).prepare().train("teacher")
    export_run(SavedRun.pinned(str(tmp_path / "teacher")), tmp_path / "policy", ema=False)
    half = pipeline(str(tmp_path / "policy"), param_dtype="bfloat16")
    assert {leaf.dtype for leaf in jax.tree.leaves(half.variables)
            if jnp.issubdtype(leaf.dtype, jnp.floating)} == {jnp.dtype(jnp.bfloat16)}
    with pytest.raises(ValueError):
        pipeline(str(tmp_path / "policy"), dtype="bfloat16")
    # Its parameters' storage is no compute of its: exported again and reloaded, it acts on float32
    # observations in float32.
    half.save_pretrained(tmp_path / "again")
    again = pipeline(str(tmp_path / "again"))
    observation = jnp.ones((2, again.settings.observation_size), jnp.float32)
    assert {leaf.dtype for leaf in jax.tree.leaves(again.model.apply(again.variables, observation))
            if jnp.issubdtype(leaf.dtype, jnp.floating)} == {jnp.dtype(jnp.float32)}
    (tmp_path / "source").mkdir()
    Release(max_length=64).write(tmp_path / "source")
    with pytest.raises(ValueError):
        SavedModel.from_pretrained(tmp_path / "source")



def test_a_model_release_written_before_it_stated_its_run_loads_as_it_did(tmp_path):
    """A native model release written before the release stated its run (`task` holding the objective, the
    model's record, its settings and numerics) loads as the release of today writes it: the same model and
    weights, the same settings (review of the run release)."""
    from dew.inference.models import SavedModel

    teacher_run(tmp_path).prepare().train("teacher")
    run = SavedRun.pinned(str(tmp_path / "teacher"))
    export_run(run, tmp_path / "policy", ema=False)
    today = pipeline(str(tmp_path / "policy"))
    release = Release.load(tmp_path / "policy")
    assert release is not None and release.task is not None
    record = dict(release.task["run"])
    settings = {name: record.pop(name) for name in list(record)
                if name not in ("objective", "model", "precision")}
    earlier = {"objective": record["objective"], "model": record["model"], "settings": settings,
               "numerics": record.get("precision"), "dtypes": {}}
    (tmp_path / "policy" / RELEASE_FILE).write_text(json.dumps({"task": earlier}))
    before = pipeline(str(tmp_path / "policy"))
    assert isinstance(before, SavedModel) and before.model == today.model
    assert before.settings == today.settings
    assert_same(before.variables, today.variables)


def test_a_run_whose_average_is_its_reference_never_reads_as_averaged():
    """An objective whose average is its frozen reference (DPO's) releases its live weights whatever `ema`
    asks (`SavedRun.variables`), so its run's `averaged` says so: never the average, asked or stored, which a
    release records of the weights it holds (review of the run release)."""
    from dew.objectives.rl.preference import DPOObjective
    from dew.registry import import_path

    run = SavedRun("a run its objective alone answers for", {"objective": import_path(DPOObjective)}, 0)
    assert run.averaged(True) is False
    assert run.averaged(None) is False
