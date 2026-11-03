"""Dreamer 4's world model as a run over video (`dew.config.world_model`): the tokenizer's stage and the
dynamics' stage over the saved tokenizer, each from its record."""

import dataclasses
import itertools
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from test_env_runs import assert_bitwise

from dew.config import ObjectiveConfig, RunConfig, TrainerConfig
from dew.config.world_model import TokenizerConfig, WorldModelRunConfig
from dew.data import Files, Memory
from dew.data.dataset import Loader
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.lora import LoRA
from dew.nn.kernels import generation
from dew.objectives.world_model import CausalTokenizer, Dynamics, TokenizerObjective, WorldModelObjective
from dew.optim import OptimConfig
from dew.registry import ModelConfig
from dew.state import TrainState
from dew.training import Trainer

WORLD = "dew.objectives.world_model"
WINDOW, SIZE, BATCH = 4, 8, 8
"""A clip's frames, a frame's side in pixels and a batch's clips."""
TOKENIZER: dict[str, Any] = {"patch": 4, "width": 16, "depth": 1, "heads": 2, "latents": 4, "bottleneck": 4}
DYNAMICS: dict[str, Any] = {"spatial": 2, "width": 16, "depth": 1, "heads": 2, "levels": 3, "frames": WINDOW}
"""The tokenizer's and the dynamics model's fields at a test's size; the run derives the rest."""
OPTIM = OptimConfig(optimizer="adam", learning_rate=1e-3)
TOKENIZER_STEPS, DYNAMICS_STEPS = 3, 2


def videos(rows: int = 16, size: int = SIZE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1, 1, (rows, WINDOW, size, size, 1)).astype(np.float32)


def stored(directory: Path, size: int = SIZE, *, states: bool = True, actions: int = 0) -> Pipeline:
    """Clips written as one ArrayRecord shard under `directory` and read as batches of `BATCH` in the order
    they are stored, repeating; the export states the clips' shape (and their actions') where `states`."""
    clips = videos(size=size)
    columns = {"video": list(clips)}
    shapes = {"video": [WINDOW, size, size, 1]}
    if actions:
        columns["actions"] = list(
            np.random.default_rng(1).normal(size=(len(clips), WINDOW, actions)).astype(np.float32))
        shapes["actions"] = [WINDOW, actions]
    export(Memory(columns), str(directory), format="array_record", rows_per_shard=len(clips),
           metadata={"fields": shapes} if states else None)
    return Files(str(directory)).repeat().batch(BATCH)


def trainer(root: Path, steps: int, key: int) -> TrainerConfig:
    return TrainerConfig(checkpoint_dir=str(root), steps=steps, log_every=1, eval_every=None,
                         checkpoint_every=None, key=key, multi_host=False)


def tokenizer_run(root: Path, clips: Pipeline) -> WorldModelRunConfig:
    return WorldModelRunConfig(data=clips, model=ModelConfig(f"{WORLD}:CausalTokenizer", TOKENIZER),
                               optim=OPTIM, trainer=trainer(root, TOKENIZER_STEPS, 0))


def dynamics_run(root: Path, clips: Pipeline | None, tokenizer: Path, **fields) -> WorldModelRunConfig:
    run = WorldModelRunConfig(
        data=clips, model=ModelConfig(f"{WORLD}:Dynamics", DYNAMICS),
        objective=ObjectiveConfig(f"{WORLD}:WorldModelObjective"), tokenizer=TokenizerConfig(str(tokenizer)),
        optim=OPTIM, trainer=trainer(root, DYNAMICS_STEPS, 1))
    return dataclasses.replace(run, **fields)


class Stages(NamedTuple):
    root: Path
    clips: Pipeline
    tokenizer: TrainState
    dynamics: TrainState


@pytest.fixture(scope="module")
def stages(tmp_path_factory) -> Stages:
    """The tokenizer's stage trained from its record, then the dynamics' stage over it, from its record."""
    root = tmp_path_factory.mktemp("stages")
    clips = stored(root / "clips")
    tokenizer = tokenizer_run(root, clips).prepare().train("tokenizer")
    dynamics = dynamics_run(root, clips, root / "tokenizer").prepare().train("dynamics")
    return Stages(root, clips, tokenizer, dynamics)


def batches(clips: np.ndarray) -> Loader:
    """The clips as the stored chain reads them: in order, `BATCH` at a time, again from the first."""
    return Loader(train=lambda partition: ({"video": clips[i:i + BATCH]}
                                           for i in itertools.cycle(range(0, len(clips), BATCH))),
                  val=None, records=len(clips), batch=BATCH)


def test_each_stage_from_its_record_trains_as_the_stage_composed_in_python(stages):
    """The tokenizer's stage, and the dynamics' stage over the saved tokenizer at its last step, each end,
    every leaf of their states bit for bit, where the same networks and objectives composed in Python end
    over the same clips: the run derives the tokenizer's frame from its data, and the dynamics' widths
    from the tokenizer."""
    tokenizer = CausalTokenizer((SIZE, SIZE, 1), **TOKENIZER)
    dynamics = Dynamics(latents=4, bottleneck=4, **DYNAMICS)
    run = tokenizer_run(stages.root, stages.clips)
    with generation.run_precision(run.trainer.precision):
        tokens = Trainer(TokenizerObjective(tokenizer), OPTIM, key=0).fit(
            batches(videos()), steps=TOKENIZER_STEPS, log_every=1)
        objective = WorldModelObjective(dynamics, tokenizer, tokens.variables)
        trained = Trainer(objective, OPTIM, key=1).fit(batches(videos()), steps=DYNAMICS_STEPS, log_every=1)
    assert_bitwise(stages.tokenizer, tokens)
    assert_bitwise(stages.dynamics, trained)


def test_a_dynamics_stage_names_the_step_it_read_of_its_tokenizer_and_reads_back_as_itself(stages):
    """A dynamics stage that names its tokenizer's run alone resolves, as it starts, the tokenizer's latest
    step, which its record then names; the run.json it writes reads back as the stage, as does each
    stage's record through JSON."""
    prepared = dynamics_run(stages.root, stages.clips, stages.root / "tokenizer").prepare()
    assert isinstance(prepared.run, WorldModelRunConfig)
    assert prepared.run.tokenizer is not None and prepared.run.tokenizer.step == TOKENIZER_STEPS
    saved = RunConfig.load(str(stages.root / "dynamics"))
    assert isinstance(saved, WorldModelRunConfig)
    assert saved.tokenizer == TokenizerConfig(str(stages.root / "tokenizer"), TOKENIZER_STEPS)
    for run in (saved, RunConfig.load(str(stages.root / "tokenizer")), prepared.run):
        assert WorldModelRunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run


def test_a_dynamics_stage_over_labeled_clips_derives_its_action_width_from_them(tmp_path):
    """Clips that state `actions` of two a frame train a dynamics model of two actions a step, whose
    action projection the run draws, with no width in its record."""
    clips = stored(tmp_path / "clips", actions=2)
    tokenizer_run(tmp_path, clips).prepare().train("tokenizer")
    state = dynamics_run(tmp_path, clips, tmp_path / "tokenizer").prepare().train("labeled")
    assert state.variables["params"]["action_projection"]["kernel"].shape == (2, DYNAMICS["width"])


@pytest.fixture(scope="module")
def other_frames(tmp_path_factory) -> Pipeline:
    """Clips of another frame size than the tokenizer's stage trained on."""
    return stored(tmp_path_factory.mktemp("other") / "clips", 2 * SIZE)


@pytest.mark.parametrize("change", [
    "a tokenizer stage that states its frame", "a dynamics stage that states its latents",
    "a dynamics stage with no tokenizer", "a tokenizer stage that names a tokenizer",
    "a dynamics model that holds fewer frames than the clips",
    "a dynamics model under a tokenizer's objective",
    "a tokenizer of other frames than the clips", "a dynamics stage's run as its tokenizer",
    "clips whose chain states no shapes", "an adapter"])
def test_a_world_model_run_refuses_a_setup_it_cannot_train(stages, other_frames, tmp_path, change):
    """A stage that states a width the run derives, or names no tokenizer or a second one, whose model is
    another class than its objective trains, whose dynamics cannot hold its clips' frames, whose
    tokenizer, or whose saved tokenizer's run, is no tokenizer or was trained on other frames, over
    clips that state no shape, or with an adapter, is refused as its run is prepared."""
    root, clips = stages.root, stages.clips
    tokenizer = tokenizer_run(root, clips)
    dynamics = dynamics_run(root, clips, root / "tokenizer")
    model = dynamics.model
    runs = {
        "a tokenizer stage that states its frame": dataclasses.replace(tokenizer, model=dataclasses.replace(
            tokenizer.model, fields={**tokenizer.model.fields, "frame": [SIZE, SIZE, 1]})),
        "a dynamics stage that states its latents": dataclasses.replace(dynamics, model=dataclasses.replace(
            model, fields={**model.fields, "latents": 4})),
        "a dynamics stage with no tokenizer": dataclasses.replace(dynamics, tokenizer=None),
        "a tokenizer stage that names a tokenizer": dataclasses.replace(
            tokenizer, tokenizer=dynamics.tokenizer),
        "a dynamics model that holds fewer frames than the clips": dataclasses.replace(
            dynamics, model=dataclasses.replace(model, fields={**model.fields, "frames": WINDOW - 1})),
        "a dynamics model under a tokenizer's objective": dataclasses.replace(
            tokenizer, model=model),
        "a tokenizer of other frames than the clips": dataclasses.replace(dynamics, data=other_frames),
        "a dynamics stage's run as its tokenizer": dataclasses.replace(
            dynamics, tokenizer=TokenizerConfig(str(root / "dynamics"))),
        "clips whose chain states no shapes": dataclasses.replace(
            tokenizer, data=stored(tmp_path / "unstated", states=False)),
        "an adapter": dataclasses.replace(tokenizer, lora=LoRA(rank=2, modules=("dense",))),
    }
    with pytest.raises(ValueError):
        runs[change].prepare()


def test_a_command_line_names_the_stage_its_record_does(tmp_path):
    """A dynamics stage named on the command line, its model's fields one flag each and its tokenizer turned
    on by its flags (`--tokenizer.`), is the stage its record states."""
    args = ["--model", f"{WORLD}:Dynamics", "--model.spatial", "2", "--model.width", "16",
            "--model.depth", "1", "--model.heads", "2", "--model.levels", "3",
            "--model.frames", str(WINDOW), "--objective", f"{WORLD}:WorldModelObjective",
            "--tokenizer.run", str(tmp_path / "tokenizer"), "--tokenizer.step", "3",
            "--optim.optimizer", "adam", "--optim.learning-rate", "1e-3",
            "--trainer.checkpoint-dir", str(tmp_path), "--trainer.steps", "2", "--trainer.log-every", "1",
            "--trainer.multi-host", "False"]
    stated = dataclasses.replace(
        dynamics_run(tmp_path, None, tmp_path / "tokenizer"), data=None,
        tokenizer=TokenizerConfig(str(tmp_path / "tokenizer"), 3),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=2, log_every=1, multi_host=False))
    assert WorldModelRunConfig.cli(args) == stated
