"""What a run admits, at one boundary (row 1872, audit TH-07): a configured model is built from its
effective record, and a trainer makes its key only after the process is ready."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import flax.linen as nn
import jax
import numpy as np
import optax
import pytest
from affine_run import Replaying
from objective_cases import IMAGES

from dew.batch import VALID_ROWS
from dew.config import TrainerConfig
from dew.config.jepa import JepaRunConfig
from dew.data import TFDS, Loading, Memory, presets
from dew.data.dataset import DataPartition, Loader, rows_of
from dew.data.pipeline import DeviceStep, Pipeline
from dew.inputs import Field, InputSpec
from dew.objectives.jepa.objective import JepaObjective
from dew.objectives.supervised import Supervised
from dew.registry import ModelConfig
from dew.training import Trainer
from dew.training.observe import Observe

ROOT = Path(__file__).resolve().parents[1]


class Built(JepaRunConfig):
    """A JEPA run whose training hands back the objective it built, untrained."""

    def train(self, objective, dataset, **_):
        return objective


def test_a_jepa_predictor_takes_the_width_its_encoder_was_built_at(tmp_path):
    """An encoder whose width its record pins as a default (a run recorded at another class default)
    is built at that width, and the predictor it feeds takes the same width: before, the predictor
    read the stated fields alone and took its class's 384 (AT-06)."""
    rows = TFDS("dew_images", path=str(IMAGES), split="all").pipe(presets.images, 16, batch=8)
    encoder = ModelConfig("jepa_encoder", {"patch_size": 2, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2,
                                           "dtype": "float32", "attention_impl": "xla"},
                          defaults={"emb_features": 16})
    run = Built(model=encoder, predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
                data=rows.read("memory").loading(Loading(workers=0)),
                trainer=TrainerConfig(checkpoint_dir=str(tmp_path), steps=1))
    objective = run.prepare().train("built")
    assert isinstance(objective, JepaObjective)
    assert (objective.encoder.emb_features, objective.predictor.emb_features) == (16, 16)


KEYED = """
import json
import optax
from dew._devices import backend_open
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Trainer
from dew.training.observe import Observe

model = models.build("causal_transformer", vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                     mlp_features=16, max_seq_len=8)
trainer = Trainer(LMObjective(model, 8), optax.sgd(0.1), key=3)
built = backend_open()
key = trainer.key
print(json.dumps({"built": built, "read": backend_open()}))
"""
"""A fresh process that builds a trainer from a seed, then reads its key."""


def test_a_trainer_opens_no_backend_until_its_key_is_read():
    """XLA reads its flags and compilation cache as JAX opens its backend, which `fit` readies first:
    a trainer built from a seed leaves the backend closed, and its key is made when first read
    (AT-09). Before, the constructor made the key and opened the backend."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(ROOT / "src"),
                                                                     os.environ.get("PYTHONPATH"))))}
    done = subprocess.run([sys.executable, "-c", KEYED], capture_output=True, text=True, env=env, cwd=ROOT,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"built": False, "read": True}


class Pooled(nn.Module):
    """A map that reads its input's mean, so it takes any width its batch has."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x.mean(axis=-1, keepdims=True))


def test_a_batch_of_a_shape_the_run_has_not_stepped_is_checked_against_its_declaration():
    """An objective declares its input three wide; a stream whose second batch is five wide, as a
    later stage of a ramp or a schedule may make it, is refused at that batch, before it compiles a
    step for it. Before, only the first batch was checked, and a model that reads any width trained
    on the undeclared one (AT-03)."""
    def batch_at(step: int) -> dict:
        x = np.random.default_rng(step).normal(size=(8, 3 if step == 0 else 5)).astype(np.float32)
        return {"x": x, "y": x[:, :2]}

    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    trainer = Trainer(objective, optax.sgd(0.1), key=0)
    data = Loader(train=lambda partition: Replaying(batch_at), val=None, batch=8, records=None)
    with pytest.raises(ValueError):
        trainer.fit(data, steps=2, log_every=1)


@pytest.mark.parametrize("eval_every", [None, 5], ids=["at-the-end", "midway"])
def test_a_validation_split_of_undeclared_inputs_is_refused_before_the_run_trains(eval_every):
    """A model that reads any width trains on the three-wide inputs its objective declares, and its
    validation split is five wide: the split is read before the run updates, whether it is scored
    every five steps or only at the end, and the run is refused there, before it completes a step.
    Before, only training batches were checked, and an end-only split was first read after training
    (AT-03)."""
    def batch(width: int, seed: int) -> dict:
        x = np.random.default_rng(seed).normal(size=(8, width)).astype(np.float32)
        return {"x": x, "y": x[:, :2]}

    stream = Replaying(lambda step: batch(3, step))
    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    data = Loader(train=lambda partition: stream, val=None, batch=8, records=None)
    held = {"held": lambda partition: iter([batch(5, 0), batch(5, 1)])}
    stepped = Observe(1, lambda state: int(state.step))
    with pytest.raises(ValueError):
        Trainer(objective, optax.sgd(0.1), key=0).fit(data, validation=held, steps=40, log_every=1,
                                                      eval_every=eval_every, observe=stepped)
    assert stepped.values == []


@pytest.mark.parametrize("metric", ["mean", "own"])
@pytest.mark.parametrize("eval_every", [None, 2], ids=["at-the-end", "midway"])
def test_a_validation_split_holding_no_field_its_metric_reads_is_refused_before_the_run_trains(eval_every,
                                                                                                metric):
    """A metric that reads a field beside its scores (`Mean.fields`, as a CLIP score reads the captions it
    compares, or a metric of a user's own that states it, as FID states its images) over a validation split
    whose records hold none: the split's first batch is read before the run updates, whether it is scored
    every two steps or only at the end, and the run is refused there, on every process at once, before it
    completes a step. Before, an end-only split met the missing field at its first scoring, after training
    (row 9063), and a metric no `Mean` was never asked (row 10561)."""
    from dew.artifacts import TokenScores
    from dew.eval.common import Mean
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import LMObjective

    tokens = np.tile(np.asarray([[0, 1, 2, 3, 0]], np.int32), (8, 1))
    rows = Memory({"text": tokens}).read("memory").loading(Loading(workers=0, threads=1, read_buffer=1))
    model = CausalTransformer(vocab_size=4, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                              max_seq_len=4, attention_impl="reference")
    class Captions:
        """A metric of a user's own, no `Mean`, that reads the captions beside its scores (`Metric`'s
        `fields`), as FID reads its reference images: the captions it was handed, counted."""

        name, reads, fields = "captions", TokenScores, ("caption",)

        def __call__(self, scores, batch) -> int:
            return len(batch["caption"])

        def merge(self, accumulated: int, contribution: int) -> int:
            return accumulated + contribution

        def finalize(self, accumulated: int) -> float:
            return float(accumulated)

    captioned = Captions() if metric == "own" else Mean(
        lambda scores, batch: [len(caption) for caption in batch["caption"]], name="captioned",
        better="higher", reads=TokenScores, fields=("caption",))
    stepped = Observe(1, lambda state: int(state.step))
    with pytest.raises(ValueError):
        Trainer(LMObjective(model, seq_len=4, ema_decay=None), optax.adam(.05), key=0).fit(
            rows.shuffle().repeat().batch(8), steps=4, log_every=1, eval_every=eval_every,
            metrics=[captioned], validation=rows.batch(8), observe=stepped)
    assert stepped.values == []


def regression_rows(rows: int, seed: int) -> dict:
    x = np.random.default_rng(seed).normal(size=(rows, 3)).astype(np.float32)
    return {"x": x, "y": x[:, :2]}


@pytest.mark.parametrize("handed", ["grain", "arrays"])
def test_data_handed_as_it_is_steps_as_given(handed):
    """A run takes a Grain dataset batched by Grain, its elements the batches it steps on, and NumPy
    arrays, one batch every step, each never batched again: the objective's three-wide declaration
    holds for each step's batch, a batch of batches would not, and the run takes its steps and its
    validation from them (AT-07). Before, fit took either for a loader and failed on what it lacked."""
    import grain.python as grain

    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    if handed == "grain":
        columns = regression_rows(32, 0)
        records = [{name: column[row] for name, column in columns.items()} for row in range(32)]
        data = grain.MapDataset.source(records).batch(8)
        validation = grain.MapDataset.source(records[:16]).batch(8)
    else:
        data, validation = regression_rows(8, 0), {"val": regression_rows(8, 1)}
    state = Trainer(objective, optax.sgd(0.1), key=0).fit(data, validation=validation, steps=6, log_every=1)
    assert int(state.step) == 6


def test_a_handed_validation_batch_is_shared_alike_by_every_reader():
    """Five rows of handed arrays read by two shares: each reads three, the one short filled with a
    repeat its `VALID_ROWS` marks, so the shares make one global batch and every real row scores once;
    read by one host of two row shards, they are six, the repeat marked."""
    from dew.data.compile import native

    columns = regression_rows(5, 0)
    read = native(columns, training=False)
    shares = [next(read(DataPartition(index=index, count=2))) for index in range(2)]
    assert [rows_of(share) for share in shares] == [3, 3]
    # A share that holds only real rows marks none, as a reader's batch does (`FilledBatch`).
    assert sum(int(np.sum(share.get(VALID_ROWS, np.ones(rows_of(share), bool)))) for share in shares) == 5
    # One host feeding two row shards reads a batch of a whole number of them.
    whole = next(read(DataPartition(row_multiple=2)))
    assert (rows_of(whole), int(np.sum(whole[VALID_ROWS]))) == (6, 5)


def test_a_trainer_validated_before_it_fits_trains_from_its_seed():
    """Validation traces the trainer's state, its key among it: the key it makes there is concrete, and
    the fit after it trains from it (`Trainer.key`)."""
    objective = Supervised(Pooled(), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                           inputs=InputSpec(Field("x", (3,))))
    trainer = Trainer(objective, optax.sgd(0.1), key=3)
    trainer.validate(steps=2)
    state = trainer.fit(regression_rows(8, 0), steps=2, log_every=1)
    assert int(state.step) == 2


@dataclasses.dataclass(frozen=True)
class Narrowing:
    """A step that states each row's `aux` holds three values (`Declares`) and, from the row `after` on,
    hands two: a step that broke the shape its chain fixes."""

    after: int

    def declared(self, facts):
        from dew.data.facts import shaped

        return shaped(facts, "aux", [3])

    def __call__(self, row):
        return {**row, "aux": row["aux"][:2]} if int(row["i"]) >= self.after else row


def test_a_batch_breaking_a_field_shape_its_chain_fixes_is_refused_before_it_is_placed():
    """A chain that fixes each row's `aux` at three values (`Loader.fields`) and hands its second step rows of
    two is refused at that batch, on its host before it is placed (row 9017: only its rows were held to the
    loader, and the step compiled again for the new shape); one that hands what it states trains on."""
    rows = 2 * jax.device_count()
    columns = {"x": np.ones((4 * rows, 8), np.float32), "y": np.zeros((4 * rows, 4), np.float32),
               "aux": np.ones((4 * rows, 3), np.float32), "i": np.arange(4 * rows, dtype=np.int32)}

    def fit(after: int):
        chain = Pipeline(Memory(columns)).map(Narrowing(after), identity="narrowing@1").repeat().batch(rows)
        objective = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x")
        return Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(chain, steps=3, log_every=100)

    with pytest.raises(ValueError):
        fit(after=rows)
    assert int(fit(after=8 * rows).step) == 3


class Cropped(DeviceStep):
    """A device step that keeps each row's first four `x` values, and says so (`Declares`)."""

    def declared(self, facts):
        from dew.data.facts import shaped

        return shaped(facts, "x", [4])

    def __call__(self, batch):
        return {**batch, "x": batch["x"][:, :4]}


def test_a_device_steps_shape_is_no_shape_its_host_batch_is_held_to():
    """A chain whose device step crops each row's `x` from eight values to four hands its host batch at
    eight, which the step crops after the host's check: the host batch is held to the shapes of what it
    holds, no device step's (row 9533: refused at the shape the device step would make)."""
    rows = 2 * jax.device_count()
    columns = {"x": np.ones((4 * rows, 8), np.float32), "y": np.zeros((4 * rows, 4), np.float32)}
    chain = Pipeline(Memory(columns)).repeat().batch(rows).map(Cropped())
    objective = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x")
    assert int(Trainer(objective, optax.sgd(0.1), key=jax.random.key(0)).fit(chain, steps=2,
                                                                             log_every=100).step) == 2


def test_a_stream_with_no_batch_ends_its_run_where_it_starts_its_final_checkpoint_written(tmp_path):
    """A finite stream that holds no batch, on every process, ends its run at step 0 as its length would,
    its state the one it started from and its final checkpoint written, where its objective declares the
    input it reads (row 9068: refused at step 0); one that reads its input's shape off its first batch is
    refused, as there is none to read it off."""
    from dew.checkpoints import Checkpoints

    rows = 2 * jax.device_count()
    empty = Loader(lambda partition: iter(()), None, batch=rows, records=None)
    declared = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2,
                          inputs=InputSpec(Field("x", (8,))))
    trainer = Trainer(declared, optax.sgd(0.1), key=jax.random.key(0), checkpoints=Checkpoints(str(tmp_path)))
    assert int(trainer.fit(empty, steps=5, log_every=100).step) == 0
    assert trainer.checkpoints.latest == 0
    named = Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x")
    with pytest.raises(ValueError):
        Trainer(named, optax.sgd(0.1), key=jax.random.key(0)).fit(empty, steps=5, log_every=100)
