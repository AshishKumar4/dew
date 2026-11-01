"""A trainer over a rollout that checkpoints its state, made before its process joins its pool, then fit
there; a split scored by its episodes evaluated in a pool, its scorer once failing, once reporting fewer
scores than it declares, on one process alone."""

import itertools
from collections.abc import Callable

import jax
import numpy as np
import optax
from process_support import serve_mode, worker_parser
from test_env_rollout import Walk

from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout
from dew.rl.env import Batched, TimeLimit
from dew.training import Trainer


def unjoined(directory):
    """A trainer over a ControlRollout, checkpointed to `directory`, made from an integer seed before this
    process joins its pool, as a program that readies its process after making its trainer makes it."""
    from dew.checkpoints import Checkpoints

    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 4)
    return Trainer(objective, optax.adam(1e-3), key=0, checkpoints=Checkpoints(str(directory)),
                   rollout=ControlRollout(objective, walks, unroll_length=4))


def fitted(trainer) -> dict:
    """The pool's size and the type of the ValueError a step's fit of `trainer` raised, or None."""
    from dew.data.dataset import Loader

    idle = Loader(train=lambda partition: itertools.repeat({"step": np.zeros((8,), np.float32)}), val=None,
                  records=8, batch=8)
    try:
        trainer.fit(idle, steps=1)
        refused = None
    except ValueError as error:
        refused = type(error).__name__
    return {"processes": jax.process_count(), "refused": refused}


def flown(args):
    """A pool's evaluation of a split scored by its episodes, each process flying its own walks from one
    key: every process's scores and event key; then the same split whose scorer fails, and whose scorer
    reports one of its two scores, on the last process alone, each process's error's type."""
    from dew.objectives.base import Shown
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, Evaluation

    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    variables = objective.init(jax.random.key(0))
    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 8)
    scored = episodes(objective.greedy, walks, 8)
    split = Episodes(scored, {"return": Shown(better="higher"), "steps": Shown()})
    result = Evaluation.run(objective, variables, split, key=jax.random.key(3), step=2, split="flight")

    def raising(variables, key):
        if jax.process_index() == jax.process_count() - 1:
            raise RuntimeError("this process's environments failed")
        return scored(variables, key)

    def incomplete(variables, key):
        reported = scored(variables, key)
        return {"return": reported["return"]} if jax.process_index() == jax.process_count() - 1 else reported

    failed = {}
    for name, scorer in (("raising", raising), ("incomplete", incomplete)):
        try:
            Evaluation.run(objective, variables, Episodes(scorer, split.scores), key=jax.random.key(3),
                           step=2, split="flight")
            failed[name] = None
        except Exception as error:
            failed[name] = type(error).__name__
    return {"scores": result.scores, "event": list(result.event_key), "failed": failed}


if __name__ == "__main__":
    arguments = worker_parser({"fitted": fitted, "flown": flown}, __doc__).parse_args()
    modes: dict[str, Callable] = {"flown": flown}
    if arguments.mode == "fitted":
        # Made before serve_mode joins the pool, which it could not join had making the trainer opened JAX.
        made = unjoined(arguments.out.parent / "run")
        modes["fitted"] = lambda arguments: fitted(made)
    serve_mode(modes, arguments)
