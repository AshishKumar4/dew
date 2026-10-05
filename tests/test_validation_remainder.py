"""A validation pass scores every record of its split once, whatever the batch size and mesh."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew import Dataset, Evaluation, Mean, Trainer
from dew.artifacts import ImageGrid, TokenScores
from dew.objectives.base import Aux, Objective, Ratio
from dew.training import MeshSpec

RECORDS, BATCH = 70, 16
"""Four whole batches and six records over: fewer than the eight devices of
the mesh the pass is sharded over."""


class Indices(Objective):
    """Scores each row with its record's index, so a mean over the pass is
    the mean of the indices it counted."""

    artifact = TokenScores

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        index = jnp.asarray(batch["index"], jnp.float32)
        return Ratio(jnp.sum(index + variables["params"]["w"]), jnp.asarray(index.shape[0], jnp.float32)), Aux({})

    def evaluate(self, params, batch, step):
        index = jnp.asarray(batch["index"], jnp.float32)[:, None]
        return TokenScores(index, jnp.ones_like(index), correct=index < 10)


def scored(artifact, batch):
    return artifact.losses[:, 0]


def read(artifact, batch):
    return batch["index"]


def below_ten(artifact, batch):
    return artifact.correct[:, 0]


METRICS = (Mean(scored, name="scored", better="lower", reads=TokenScores),
           Mean(read, name="read", better="lower", reads=TokenScores),
           Mean(below_ten, name="below_ten", better="higher", reads=TokenScores))


def split(length=RECORDS):
    return Dataset.from_records({"index": np.arange(200, dtype=np.int32)}, batch=BATCH,
                                validation={"index": np.arange(length, dtype=np.int32)})


@pytest.mark.mesh
def test_metrics_cover_every_record_once_on_a_mesh():
    objective = Indices()
    data = split()
    result = Evaluation.run(objective, objective.init(jax.random.key(0)), data.val, key=jax.random.key(1),
                            metrics=METRICS, mesh=MeshSpec().build())

    mean = (RECORDS - 1) / 2
    assert result.scores == {"val/scored": mean, "val/read": mean, "val/below_ten": 10 / RECORDS}
    assert result.records == RECORDS and result.coordinated_batches == 5


def test_the_loss_never_counts_a_copy():
    """The objective's loss sums every row it is given, so the filled batch
    is left out of it, and the loss is the mean over the whole batches."""
    objective = Indices()
    result = Evaluation.run(objective, objective.init(jax.random.key(0)), split().val,
                            key=jax.random.key(1), loss=True)

    whole = RECORDS // BATCH * BATCH
    assert result.scores == {"val/loss": (whole - 1) / 2}


def test_a_split_smaller_than_a_batch_is_scored_whole():
    objective = Indices()
    result = Evaluation.run(objective, objective.init(jax.random.key(0)), split(5).val,
                            key=jax.random.key(1), metrics=METRICS[:1])

    assert result.scores == {"val/scored": 2.0} and result.records == 5


class Recording:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, artifact, step):
        pass


def test_fit_validates_on_every_record():
    tracker = Recording()
    trainer = Trainer(Indices(), optax.sgd(0.0), key=jax.random.key(0), tracker=tracker)
    trainer.fit(split(), steps=1, eval_every=1, log_every=1, metrics=METRICS[:1])

    logged = next(scalars for scalars in tracker.scalars if "val/scored" in scalars)
    assert logged["val/scored"] == (RECORDS - 1) / 2
    assert logged["evaluation/records"] == RECORDS


class Pictures(Objective):
    """Returns one image a row, and its captions, so a metric over images and
    captions sees the counted rows alone."""

    artifact = ImageGrid

    def init(self, key, variables=None):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, variables, batch, step):
        return variables["params"]["w"]

    def evaluate(self, params, batch, step):
        index = jnp.asarray(batch["index"], jnp.float32)
        captions = tuple(str(int(value)) for value in np.asarray(batch["index"]))
        return ImageGrid(jnp.broadcast_to(index[:, None, None, None], (index.shape[0], 2, 2, 1)), captions)


def test_image_artifacts_and_their_captions_keep_only_counted_rows():
    def captioned(artifact, batch):
        assert len(artifact.captions) == len(artifact.images)
        return [float(caption) for caption in artifact.captions]

    objective = Pictures()
    result = Evaluation.run(objective, objective.init(jax.random.key(0)), split().val, key=jax.random.key(1),
                            metrics=[Mean(captioned, name="captioned", better="lower", reads=ImageGrid),
                                     Mean(lambda artifact, batch: artifact.images[:, 0, 0, 0], name="pixel",
                                          better="lower", reads=ImageGrid)])

    mean = (RECORDS - 1) / 2
    assert result.scores == {"val/captioned": mean, "val/pixel": mean}
