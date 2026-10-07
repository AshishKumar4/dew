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
from diffusion_stubs import batch_for
from reference_error import assert_computes_the_oracle, chain_roundings
from test_guidance_distillation import FIXTURES, FLUX
from test_objective_inputs import cases, corpus_windows

from dew import Dataset, Evaluation
from dew.config import ModelConfig
from dew.data import DataPartition, Loading, TFDSImages
from dew.data.chat import Role
from dew.data.dataset import rows_of
from dew.diffusion.presets import Flow
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import VALID_ROWS, Objective, Step
from dew.objectives.diffusion import DiffusionRunConfig, GuidanceDistillationObjective, TextCondition
from dew.objectives.lm import LMObjective, Perplexity
from dew.registry import objectives
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
        data=TFDSImages(image_size=8), preset=Flow(), solver=Euler(), guidance=None, sampling_steps=2,
        ema_decay=None, val_metrics=(),
        text=TextCondition(encoder="diffusion_text", checkpoint=str(tmp_path / "flux" / "pipeline"))).build()
    objective = GuidanceDistillationObjective(teacher.model, teacher.process, teacher.inputs, teacher=teacher,
                                              teacher_variables=teacher.init(jax.random.key(0)), steps=2)
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
    """`tree` with every floating leaf in float64."""
    return jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf, tree)


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
    data = Dataset.from_records({"id": np.arange(8)}, batch=4, validation={"id": np.arange(held)},
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


EVALUATED = sorted(set(objectives) - ROLLOUT)
"""Every registered objective with an evaluation loss."""


def evaluated(name: str, windows, tmp_path) -> dict:
    """`name`'s pass over a split of `RECORDS` in batches of `ROWS`, run on
    every process of the pool: once through `Dataset.from_records`, whose
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
        data = Dataset.from_records(records, batch=ROWS, validation=records,
                                    loading=Loading(workers=0, threads=1, read_buffer=1))
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
        scored = variables if objective._ema_is_reference else averaged

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


def test_an_accuracy_counts_neither_the_hits_nor_the_answers_of_a_repeat_row():
    """`Objective.accuracy` over a batch whose last row repeats the first:
    the repeat's hits and answers both drop out, a target mask weighs each
    answer, and a batch with nothing counted reports 0."""
    correct = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    weights = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    padded = {VALID_ROWS: np.asarray([True, True, False])}
    accuracy, counted = Objective.accuracy(correct, padded, weights).mean()
    assert bool(counted) and accuracy == np.float32(2) / np.float32(5)
    assert Objective.accuracy(correct, {}).mean()[0] == np.float32(6) / np.float32(9)
    assert float(Objective.accuracy(correct, {}, jnp.zeros_like(correct)).mean()[0]) == 0.0


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
        data = Dataset.from_records(records_of(batch, ROWS), batch=ROWS, validation=records,
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
