"""The programs a user's path compiles at Dew's compile point (`dew.nn.kernels.options.OuterProgram`) are read
by the algebra pass (docs/design/algebra.md §2, row 6137): a host-master step's phases, an evaluation, a
text generation and an RL rollout each compile their programs there (`options.keeping`), each recording its
pass as it traces (`report.passes`, each path's window apart), and none with the pass switched off
(`algebra.passes(on=False)`)."""

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
from dew.nn.kernels import options
from dew.nn.kernels.templates import report
from dew.objectives.rl import GRPOObjective, RolloutScheduler
from dew.sampling.text import Sampling
from dew.training import Layout, Trainer, identity
from dew.training.evaluation import Evaluation


@pytest.fixture
def passed():
    """The passes the algebra pass records within the test (`report.passes`), apart from the process's."""
    with report.refusals():
        yield report.passes


def compiled_in(passed, run) -> tuple[list[str], int]:
    """The programs `run` compiles at Dew's compile point (`options.keeping`), by their functions' names,
    and the passes it records, its own window of the test's: each compile traces its program once, through
    the pass, so a path records at least a pass a program."""
    names: list[str] = []
    before = len(passed())
    with options.keeping(lambda program, *heard: names.append(identity.called(program))):
        run()
    return names, len(passed()) - before


def test_a_host_master_step_compiles_its_phases_through_the_pass_and_an_evaluation_its_program(passed):
    """A host-master step, its variables in host memory (`Layout.host`, which runs the transaction's
    phases on the host, `Transaction.step(host=True)`), compiles its phases at the compile point, its local
    reduction, its attempt's preparation and its finish among them, each recording its pass; an evaluation
    of the run's weights its own program, with its pass. (A layout keeping only the optimizer state on the
    host steps on the devices, one program, which shows no phase.)"""
    trainer = Trainer(Features(), optax.adam(0.1), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0, host=("variables",)))
    state = trainer.initial_state()
    batch = next(iter(Counting()))
    names, passes = compiled_in(passed, lambda: trainer.compile(state, batch)(state, batch))
    phases = {name.rsplit(".", 1)[-1] for name in names}
    assert {"local_reduction", "prepare_attempt", "finish_attempt"} <= phases, names
    assert passes >= len(names), (passes, names)
    data = Loader(train=lambda partition: Counting(), val=lambda partition: val_batches()(), records=None,
                  batch=BATCH)
    names, passes = compiled_in(passed, lambda: Evaluation.run(trainer.objective, state.variables, data.val,
                                                               metrics=(Spread([]),), key=0))
    assert names and passes >= len(names), (passes, names)


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
    generate = TextGeneration(model, params, sampling=sampling)
    names, passes = compiled_in(passed, lambda: generate([[1, 2, 3]], 3, key=0))
    assert names and passes >= len(names), (passes, names)
    rollout = RolloutScheduler.sampled(objective, lambda text, record: float(len(text) % 3), groups=2,
                                       max_new_tokens=4, sampling=sampling)
    state = Trainer(objective, optax.sgd(.01), key=jax.random.key(1)).initial_state()
    prompts = {"prompt": np.array([[0, 0, 1, 2], [3, 4, 5, 6]], np.int32),
               "prompt_length": np.array([2, 4], np.int32),
               "data_source": np.array([[97], [98]], np.int32),
               "ground_truth": np.array([[49], [50]], np.int32),
               "extra_info": np.zeros((2, 1), np.int32)}
    names, passes = compiled_in(passed,
                                lambda: rollout(state, prompts, jax.random.key(3), objective=objective))
    assert names and passes >= len(names), (passes, names)
