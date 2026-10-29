"""A validation pass scores every record of its split once, whatever the split's size.

The last batch is filled out to the batch's rows with repeats that
`VALID_ROWS` marks, and every loss and metric counts the real rows alone
(`Objective.row_mean`).
"""

from __future__ import annotations

import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_support import record_batches
from diffusion_stubs import batch_for
from diffusion_support import images_of
from objective_cases import cases
from objective_data import corpus_windows
from reference_error import assert_computes_the_oracle, chain_roundings
from test_guidance_distillation import FIXTURES, FLUX

from dew import Evaluation
from dew.batch import VALID_ROWS, padded_rows
from dew.config import ObjectiveConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data import Loading
from dew.data.chat import Role
from dew.data.dataset import DataPartition, rows_of
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import TEACHER, Step
from dew.objectives.diffusion import GuidanceDistillationObjective
from dew.objectives.lm import LMObjective, Perplexity
from dew.registry import ModelConfig, objectives
from dew.sampling import Euler

ROWS = 8
"""A batch's rows: one per device of the lane's eight."""
TOKEN_ACCURACY = FIXTURES / "token_accuracy"
"""TRL's token-accuracy references (`tools/token_accuracy_reference.py`)."""
RECORDS = 9
"""The split: a batch and one record, so on two processes one share runs out
a batch before the other's and covers it with a copy of its last."""
ROLLOUT = {"grpo", "ppo", "flow_grpo"}
"""Trained on rollouts: a validation split holds prompts, which their loss does not read."""


def runnable(name: str, windows, tmp_path):
    """`name`'s recipe case, or for guidance distillation, whose student must
    read a guidance input, a guidance-embedded Flux on the committed tiny
    pipeline's towers distilling its own initialization."""
    if name != "guidance_distillation":
        return cases(windows)[name]()
    with tarfile.open(FIXTURES / "flux_source.tar.xz") as archive:
        archive.extractall(tmp_path / "flux", filter="data")
    teacher = DiffusionRunConfig(
        model=ModelConfig("flux_transformer", {**FLUX, "dtype": "float32", "attention_impl": "xla"}),
        data=images_of("tfds://oxford_flowers102", 8), preset=Flow(), val_metrics=(),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(tmp_path / "flux" / "pipeline")),
        objective=ObjectiveConfig("diffusion",
                                  {"solver": Euler(), "guidance": None, "steps": 2,
                                   "ema_decay": None})).build()
    objective = GuidanceDistillationObjective(teacher.model, teacher.process, teacher.inputs,
                                              variables={TEACHER: teacher.init(jax.random.key(0))}, steps=2)
    return objective, batch_for(teacher, 8)


@pytest.fixture(scope="module")
def windows(tmp_path_factory):
    return corpus_windows(tmp_path_factory.mktemp("corpus"))


def records_of(batch, count: int) -> list:
    """`count` records cycled out of `batch`'s rows."""
    rows = rows_of(batch)
    return [jax.tree.map(lambda value, index=index: np.asarray(value)[index % rows], batch)
            for index in range(count)]


def stacked(records: list) -> dict:
    return jax.tree.map(lambda *values: np.stack(values), *records)


def wide(tree):
    """`tree` with every floating leaf in float64; a host field, a record's captions, as it is."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                        if jnp.issubdtype(np.asarray(leaf).dtype, jnp.floating) else leaf, tree)


def moved(tree, seed: int):
    """`tree` in float64 with every floating leaf moved by a draw of its own,
    so a student is not its teacher nor a policy its reference, and no loss is
    a constant the repeats could not move."""
    leaves, structure = jax.tree.flatten(wide(tree))
    keys = jax.random.split(jax.random.key(seed), len(leaves))
    return jax.tree.unflatten(structure, [
        leaf + 0.05 * jax.random.normal(key, jnp.shape(leaf), jnp.float64)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf
        for leaf, key in zip(leaves, keys, strict=True)])


@pytest.mark.parametrize("held", [7, 1])
def test_the_last_batch_is_filled_out_and_marked_on_every_process_count(held):
    """Main's repro: eight training records in batches of four, and seven or
    one to validate. The pass reads each record once, on one process or two,
    and the repeats that fill the last batch out are marked; with one record,
    process 1's share is empty and reads one batch of repeats alone."""
    data = record_batches({"id": np.arange(8)}, 4, {"id": np.arange(held)},
                          loading=Loading(workers=0, threads=1, read_buffer=1))
    for count in (1, 2):
        seen = []
        for index in range(count):
            batches = list(data.val(DataPartition(index, count)))
            assert batches, (count, index)
            for batch in batches:
                assert rows_of(batch) == 4 // count
                valid = np.asarray(batch.get(VALID_ROWS, np.ones(4 // count, bool)))
                seen.extend(np.asarray(batch["id"])[valid].tolist())
        assert sorted(seen) == list(range(held)), count


def test_a_batch_fills_out_with_its_own_rows_in_turn_and_never_from_none():
    """A batch of three rows filled out to seven repeats them in turn and marks the four repeats; a
    batch of no rows has nothing to repeat, so filling one out is refused rather than made up."""
    indices, repeats = padded_rows(3, 7)
    assert indices.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert repeats.tolist() == [False] * 3 + [True] * 4
    with pytest.raises(ValueError):
        padded_rows(0, 4)


EVALUATED = sorted(set(objectives) - ROLLOUT)
"""Every registered objective with an evaluation loss."""

POOLED = (("block_diffusion", "distillation", "lm", "mean_flow", "masked_diffusion", "supervised"),
          ("guidance_distillation",), ("ladd", "decision", "shortcut"), ("jepa", "diffusion", "rcm", "dpo"))
"""`EVALUATED` in the parts `tests/test_multiprocess.py` runs its passes on a pool in, each about as long as
the others, by each objective's seconds alone on one process (guidance distillation's Flux about 119,
LADD's 47, JEPA's 40, the decision objective's 36, the rest under 30), so no part passes CI's group
target; the first, the lightest, also scores `Perplexity`."""


def evaluated(name: str, windows, tmp_path) -> dict:
    """`name`'s pass over a split of `RECORDS` in batches of `ROWS`, run on
    every process of the pool: once through a run's compiled records (`record_batches`), whose
    repeats fill the last batch with its own rows, and once with the repeats
    another record's. Each pass's scores and counts, and for a loss that draws
    nothing (the same under two keys) the loss of every record in one batch
    with the roundings of its computation (`chain_roundings`)."""
    objective, batch = runnable(name, windows, tmp_path)
    records = records_of(batch, RECORDS)
    with jax.enable_x64(new_val=True):
        drawn = objective.init(jax.random.key(0))
        variables, averaged = moved(drawn, 1), moved(drawn, 2)
        records = wide(records)
        data = record_batches(records, ROWS, records, loading=Loading(workers=0, threads=1, read_buffer=1))
        tail = len(records) - ROWS
        batches = [{**stacked(records[:ROWS]), VALID_ROWS: np.ones(ROWS, bool)},
                   {**stacked(records[ROWS:] + [records[0]] * (ROWS - tail)),
                    VALID_ROWS: np.arange(ROWS) < tail}]

        def other_repeats(partition: DataPartition):
            """The same pass, its repeats another record's: each global batch's
            share as `Dataset` cuts it, `index :: count`."""
            return iter([jax.tree.map(lambda value: value[partition.index::partition.count], held)
                         for held in batches])

        passes = [Evaluation.run(objective, variables, reader, key=3, averaged=averaged, loss=True)
                  for reader in (data.val, other_repeats)]
        report = {"scores": [result.scores for result in passes],
                  "records": [result.records for result in passes],
                  "batches": [result.coordinated_batches for result in passes], "unbatched": None}
        # The pass scores the averaged weights, unless the average is the
        # objective's frozen reference.
        scored = variables if objective.ema_is_reference else averaged

        def statistics(rows, key: int):
            return objective._loss(scored, rows, Step(jnp.asarray(0), jax.random.key(key), averaged))[0]

        # Whether the loss draws anything, on a batch the objective takes
        # whatever its own batch constraints (LADD's virtual batch of 8).
        first = [statistics(stacked(records[:ROWS]), key) for key in (1, 2)]
        if all(np.array_equal(left, right) for left, right in zip(
                jax.tree.leaves(first[0]), jax.tree.leaves(first[1]), strict=True)):
            whole = stacked(records)
            unbatched, _ = objective.reduce_loss(statistics(whole, 1))
            report["unbatched"] = float(unbatched)
            report["roundings"] = chain_roundings(jax.make_jaxpr(lambda tree: objective._loss(
                tree, whole, Step(jnp.asarray(0), jax.random.key(1), averaged))[0])(scored))
    return report


def assert_every_record_once(name: str, report: dict) -> None:
    """The pass read each record once, its loss is the same whatever the
    repeats hold, bit for bit, and a loss that draws nothing is the loss of
    every record in one batch, within float64 rounding (`assert_computes_the_oracle`)."""
    assert report["records"] == [RECORDS, RECORDS] and report["batches"] == [2, 2], (name, report)
    assert report["scores"][0] == report["scores"][1], name
    if report["unbatched"] is not None:
        assert_computes_the_oracle(np.asarray([report["scores"][0]["val/loss"]]),
                                   np.asarray([report["unbatched"]]), name, roundings=report["roundings"])


@pytest.mark.parametrize("name", EVALUATED)
def test_a_pass_scores_every_record_once_and_a_deterministic_loss_is_the_unbatched_one(
        name, windows, tmp_path):
    """Every registered objective with an evaluation loss, on one process
    (`tests/test_multiprocess.py` runs the same on two)."""
    assert_every_record_once(name, evaluated(name, windows, tmp_path))


def test_a_validation_token_accuracy_is_the_unpadded_one_and_at_most_one():
    """TRL's count on tests/fixtures/token_accuracy's five-row batch, padded
    to eight with three repeats of its row with the most right targets: the
    loss reports TRL's right over counted, which the repeats' hits pushed
    above it before they were left out of the count's numerator."""
    reference = dict(np.load(TOKEN_ACCURACY / "decoder.npz"))
    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2, mlp_features=32,
                              max_seq_len=16, attention_impl="reference")
    params = model.init(jax.random.key(0), jnp.ones((1, 12), jnp.int32))
    objective = LMObjective(model, seq_len=11, ema_decay=None, loss_role=Role.ASSISTANT, token_accuracy=True)
    variables = {**objective.init(jax.random.key(0)), "params": params["params"]}
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    tokens, roles = reference["1/tokens"], reference["1/roles"]
    real = {"text": jnp.asarray(tokens), "text_roles": jnp.asarray(roles)}
    scores = objective.evaluate(variables, real, step)
    best = int(np.argmax(np.sum(np.asarray(scores.correct) * np.asarray(scores.weights), axis=1)))
    repeats = [best] * 3
    padded = {"text": jnp.asarray(np.concatenate([tokens, tokens[repeats]])),
              "text_roles": jnp.asarray(np.concatenate([roles, roles[repeats]])),
              VALID_ROWS: np.arange(len(tokens) + 3) < len(tokens)}
    _, aux = objective.loss(variables, padded, step)
    expected = int(reference["1/correct"]) / int(reference["1/total"])
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(expected, rel=1e-6)
    assert float(aux.metrics["token_accuracy"]) <= 1.0


def test_a_masked_diffusion_accuracy_leaves_its_repeat_rows_out(windows):
    """With every weight zero the model predicts token 0 at every masked
    position, so repeat rows of zeros are all right and the real text, which
    holds no zero byte, all wrong: the pass's masked accuracy is the real
    rows' 0, where counting the repeats reported more."""
    objective, batch = cases(windows)["masked_diffusion"]()
    records = records_of(batch, 5)
    variables = jax.tree.map(jnp.zeros_like, objective.init(jax.random.key(0)))
    zeros = jax.tree.map(np.zeros_like, records[0])
    padded = {**stacked(records + [zeros] * (ROWS - 5)), VALID_ROWS: np.arange(ROWS) < 5}
    aux = objective._loss(variables, padded, Step(jnp.asarray(0), jax.random.key(1), None))[1]
    assert float(aux.metrics["masked_fraction"]) > 0
    assert float(aux.metrics["masked_accuracy"]) == 0.0


def test_a_dpo_accuracy_is_the_real_rows_whatever_the_repeats_hold(windows):
    """DPO's preference accuracy over five pairs the policy ranks right,
    padded with repeats that swap chosen and rejected, which it ranks wrong:
    the pass reports the five pairs' accuracy, as their batch alone does."""
    objective, batch = cases(windows)["dpo"]()
    records = records_of(batch, 5)
    swapped = [jax.tree.map(lambda value: value[::-1], record) for record in records[:ROWS - 5]]
    with jax.enable_x64(new_val=True):
        variables = moved(objective.init(jax.random.key(0)), 3)
        step = Step(jnp.asarray(0), jax.random.key(1), moved(objective.init(jax.random.key(0)), 13))
        alone = float(objective._loss(variables, stacked(records), step)[1].metrics["accuracy"])
        padded = {**stacked(records + swapped), VALID_ROWS: np.arange(ROWS) < 5}
        reported = float(objective._loss(variables, padded, step)[1].metrics["accuracy"])
    assert alone == 1.0 and reported == alone


def perplexity(windows, count: int = RECORDS) -> dict:
    """`Perplexity` over the LM case's split of `count` records in batches of
    `ROWS`, run on every process of the pool, with the cross entropy of every
    record's tokens scored in one batch and the roundings of that computation."""
    objective, batch = cases(windows)["lm"]()
    records = records_of(batch, count)
    with jax.enable_x64(new_val=True):
        variables = moved(objective.init(jax.random.key(0)), 1)
        data = record_batches(records_of(batch, ROWS), ROWS, records,
                              loading=Loading(workers=0, threads=1, read_buffer=1))
        result = Evaluation.run(objective, variables, data.val, key=3, metrics=(Perplexity(),))
        whole = stacked(records)
        step = Step(jnp.asarray(0), jax.random.key(1), None)
        scores = objective.evaluate(variables, whole, step)
        roundings = chain_roundings(jax.make_jaxpr(lambda tree: objective.evaluate(tree, whole, step).losses)(
            variables))
    total, mass = Perplexity()(scores, whole)
    return {"perplexity": result.scores["val/perplexity"], "records": result.records,
            "unbatched": total / mass, "roundings": roundings}


def assert_the_metric_over_every_record(report: dict, count: int = RECORDS) -> None:
    """The pass's perplexity is every one of `count` records', within float64
    rounding: the repeats filling the last batch reach no metric."""
    assert report["records"] == count
    assert_computes_the_oracle(np.log([report["perplexity"]]), np.asarray([report["unbatched"]]),
                               "the cross entropy", roundings=report["roundings"])


@pytest.mark.parametrize("count", [RECORDS, 1])
def test_a_metric_over_a_pass_is_the_metric_over_every_record(windows, count):
    """On one process (`tests/test_multiprocess.py` runs the same on two,
    where one record leaves process 1's share empty)."""
    assert_the_metric_over_every_record(perplexity(windows, count), count)


def named_records(mesh=None) -> list[tuple[str, int, int]]:
    """Each record a validation pass scored, as a metric read it: its name, a string no accelerator
    holds, and its tags, a list of a length of its own, which no batch stacks, so a batch carries
    a column of them as a list, beside the label the objective produced from what the devices
    hold. The pass runs on `mesh`, a pool's spanning its processes (`tests/test_multiprocess.py`),
    or this process's."""
    from affine_run import FEATURES, Features

    from dew.artifacts import Representations
    from dew.data.compile import compile
    from dew.data.pipeline import Pipeline
    from dew.data.sources import Memory

    records = Memory(tuple({"x": np.full(FEATURES, index, np.float32), "y": np.full(2, index, np.float32),
                            "index": np.int32(index), "name": f"record {index}",
                            "tags": [f"tag {index}"] * (index % 3 + 1)}
                           for index in range(RECORDS)))
    data = compile(Pipeline(records).repeat().batch(ROWS), validation=Pipeline(records).batch(ROWS),
                   objective=Features().data(), read="memory",
                   loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))

    class Names:
        name, reads = "names", Representations

        def __init__(self):
            self.seen = []

        def __call__(self, artifact, batch):
            self.seen += [(str(name), len(tags), int(label)) for name, tags, label in zip(
                batch["name"], batch["tags"], artifact.labels, strict=True)]
            return 0.0, 1

        def merge(self, accumulated, contribution):
            return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

        def finalize(self, accumulated):
            return accumulated[0] / accumulated[1]

    objective, names = Features(), Names()
    Evaluation.run(objective, objective.init(jax.random.key(0)), data.val, key=0, metrics=(names,), mesh=mesh)
    return names.seen


def test_a_metric_reads_the_fields_no_device_holds():
    """A held record's name, a string no accelerator holds, reaches the
    metrics with the rest of its batch: the objective's programs read what a
    device holds, and a metric reads the whole batch, its last batch's
    repeats left out, from its arrays and its columns of opaque records alike."""
    assert named_records() == [(f"record {index}", index % 3 + 1, index) for index in range(RECORDS)]
