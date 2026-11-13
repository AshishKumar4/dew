"""What a training step reports: its loss and metrics as the rows it consumed make them, whatever split the
planner reads it in, and what its input costs it."""

import dataclasses
import logging
import time

import jax
import numpy as np
import pytest
from affine_run import BATCH, FEATURES, dataset
from recording import RecordingTracker
from test_trainer import LM_CONTEXT, LM_ROWS, make_trainer

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec


@pytest.mark.parametrize("split", [2, None], ids=["two-microbatches", "planned"])
def test_a_steps_logged_loss_is_its_rows_whatever_its_split(split):
    """A run accumulating two steps an update logs at each step the loss of the rows that step consumed:
    read in two microbatches a step, or in the split its plan chooses, it logs every step's loss as the run
    read whole logs it, its commit steps' the update's and its others' their own batch's (docs/design/
    training.md, the logged loss), where a split step once logged its last microbatch's alone."""
    from reference_error import assert_fp32_reduction_bound

    def logged(**kwargs) -> list[float]:
        tracker = RecordingTracker()
        trainer = make_trainer(tracker=tracker, accumulation=2, **kwargs)
        trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
        trainer.fit(dataset(), steps=6, log_every=1)
        return [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]

    whole = np.asarray(logged(split=1))
    assert len(whole) == 6
    # Each step's loss, a mean of BATCH squared errors, pooled in another order: one fp32 reduction apart.
    assert_fp32_reduction_bound(logged(split=split), whole, np.abs(whole), terms=BATCH)


def test_a_steps_logged_metrics_are_its_rows_whatever_its_split():
    """A language model's run over rows that hold other counts of targets, split in two microbatches a step,
    logs each step's cross entropy, perplexity and token accuracy as the step read whole logs them, each
    pooled by its targets (`Ratio`, docs/design/training.md, the logged metrics), where a split step logged
    its last microbatch's; its logged scalars are the whole step's by name, none its last microbatch's."""
    from reference_error import assert_fp32_reduction_bound

    rows = (np.arange(2 * LM_ROWS * (LM_CONTEXT + 1), dtype=np.int32) % 12 + 1).reshape(-1, LM_CONTEXT + 1)
    # The second half of each batch's rows hold two targets, the first half six: microbatches of unequal mass.
    rows[np.arange(len(rows)) % LM_ROWS >= LM_ROWS // 2, 3:] = 0
    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=12, dtype="float32")
    names = ("train/loss", "train/ce", "train/perplexity", "train/token_accuracy")

    def logged(split: int) -> dict[str, np.ndarray]:
        tracker = RecordingTracker()
        trainer = make_trainer(tracker=tracker, accumulation=2, split=split,
                               objective=LMObjective(model, seq_len=LM_CONTEXT, pad_id=0))
        trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
        trainer.fit(Pipeline(Memory({"text": rows})).repeat().batch(LM_ROWS), steps=4, log_every=1)
        steps = [scalars for _, scalars in tracker.scalars if "train/loss" in scalars]
        assert not any(name.endswith("/last_microbatch") for scalars in steps for name in scalars)
        return {name: np.asarray([scalars[name] for scalars in steps]) for name in names}

    whole, split = logged(1), logged(2)
    for name in names:
        assert_fp32_reduction_bound(split[name], whole[name], np.abs(whole[name]), terms=LM_ROWS * LM_CONTEXT)


def kept(row) -> bool:
    """Every record kept."""
    del row
    return True


@dataclasses.dataclass(frozen=True)
class Slow:
    """A record step that takes `seconds` a record."""

    seconds: float

    def __call__(self, row):
        time.sleep(self.seconds)
        return row


@pytest.mark.parametrize("workers", [None, 2], ids=["in-process", "stated-workers"])
def test_a_per_share_stream_that_starves_the_step_in_process_is_said_once(caplog, workers):
    """A per-share stream read in the training process (its automatic count none, as its order is its
    workers' count's, row 5852) whose records take longer than the step: the run says once that its input
    starves the step, naming `.workers(n)` and `.elastic()`; a stream that states its workers is not said
    to."""

    from dew.data.compile import compile

    rows = np.random.default_rng(0).normal(size=(64, FEATURES)).astype(np.float32)
    # A filter, which may drop records, so its stream's order is its workers' count's.
    chain = (Pipeline(Memory({"x": rows, "y": 2 * rows[:, :2]}, shard_rows=8)).map(Slow(0.01))
             .filter(kept, reason="never").repeat())
    chain = chain if workers is None else chain.workers(workers)
    loader = compile(chain.batch(BATCH), read="streamed")
    with caplog.at_level(logging.WARNING, logger="dew.training.input"):
        make_trainer().fit(loader, steps=6, log_every=1)
    said = [record for record in caplog.records if record.name == "dew.training.input"]
    assert len(said) == (1 if workers is None else 0)
