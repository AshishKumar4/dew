"""The programs a user's path compiles at Dew's compile point (`dew.nn.kernels.options.OuterProgram`) are read
by the algebra pass (docs/design/algebra.md §2, row 6137): a run whose optimizer state lives on the host,
with its evaluations; a text generation; and an RL rollout each record their passes as they trace
(`report.passes`), and none with the pass switched off (`algebra.passes(on=False)`)."""

import jax
import numpy as np
import optax
import pytest
from affine_run import BATCH, Counting, Features, Spread, val_batches
from flax import linen as nn
from model_support import decoder

from dew import algebra
from dew.data.dataset import Loader
from dew.inference import TextGeneration
from dew.nn.kernels.templates import report
from dew.objectives.rl import GRPOObjective, RolloutScheduler
from dew.sampling.text import Sampling
from dew.training import Checkpoints, Layout, Trainer


@pytest.fixture
def passed():
    """The passes the algebra pass records within the test (`report.passes`), apart from the process's."""
    with report.refusals():
        yield report.passes


def test_a_host_master_runs_steps_and_evaluations_are_read_by_the_pass(passed, tmp_path):
    """A run whose optimizer state and average live in host memory, evaluated every two steps."""
    checkpoints = Checkpoints(str(tmp_path / "run"), keep=3)
    trainer = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0), checkpoints=checkpoints,
                      layout=Layout(min_shard=1, tolerance=1.0, host=("opt_state", "ema")))
    data = Loader(train=lambda partition: Counting(), val=lambda partition: val_batches()(), records=None,
                  batch=BATCH)
    trainer.fit(data, steps=4, log_every=1, eval_every=2, checkpoint_every=2, metrics=(Spread([]),))
    checkpoints.wait()
    assert passed()


def test_a_text_generation_and_an_rl_rollout_are_read_by_the_pass_unless_it_is_off(passed):
    """A generation's programs, then a policy's sampled rollout over prompts; the same generation under
    another key traced with the pass off records none."""
    model = decoder()
    objective = GRPOObjective(model, seq_len=7)
    params = nn.unbox(objective.init(jax.random.key(0)))
    sampling = Sampling(temperature=1.2, eos_token_ids=(3, 7), pad_token_id=12)
    with algebra.passes(on=False):
        TextGeneration(model, params, sampling=sampling)([[1, 2, 3]], 3, key=0)
    assert not passed()
    TextGeneration(model, params, sampling=sampling)([[1, 2, 3]], 3, key=0)
    generated = len(passed())
    assert generated
    rollout = RolloutScheduler.sampled(objective, lambda text, record: float(len(text) % 3), groups=2,
                                       max_new_tokens=4, sampling=sampling)
    state = Trainer(objective, optax.sgd(.01), key=jax.random.key(1)).initial_state()
    prompts = {"prompt": np.array([[0, 0, 1, 2], [3, 4, 5, 6]], np.int32),
               "prompt_length": np.array([2, 4], np.int32),
               "data_source": np.array([[97], [98]], np.int32),
               "ground_truth": np.array([[49], [50]], np.int32),
               "extra_info": np.zeros((2, 1), np.int32)}
    rollout(state, prompts, jax.random.key(3), objective=objective)
    assert len(passed()) > generated
