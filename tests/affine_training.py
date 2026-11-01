"""Training helpers for the affine regression of tests/affine_run.py."""

import jax
import optax
from affine_run import Regression, batches

from dew.data.dataset import Loader
from dew.training import Checkpoints, Layout, Trainer


def fit_one_step() -> None:
    """One step of a plain run over `batches`, the least `Trainer.fit` that readies its process."""
    Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0)).fit(
        Loader(lambda partition: batches(), None, None, 8), steps=1, log_every=1)


def frozen_trainer(tmp_path=None, **kwargs) -> Trainer:
    """The regression's trainer with its average frozen at unit decay, the
    reference a preference or RL objective reads, checkpointing under
    `tmp_path` when one is given."""
    checkpoints = None if tmp_path is None else Checkpoints(str(tmp_path / "run"))
    return Trainer(Regression(ema_decay=1.0), optax.sgd(0.1), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), checkpoints=checkpoints, **kwargs)
