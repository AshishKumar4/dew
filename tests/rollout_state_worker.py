"""A trainer over a rollout that checkpoints its state, made before its process joins its pool, then fit
there, or placed from a run one process checkpointed; a split scored by its episodes evaluated in a pool,
its scorer once failing, once reporting fewer scores than it declares, on one process alone."""

import dataclasses
from collections.abc import Callable

import jax
import optax
from flax import linen as nn
from process_support import serve_mode, worker_parser
from test_env_rollout import Walk

from dew.config.env import idle
from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout
from dew.rl.env import Batched, TimeLimit
from dew.training import Trainer


def unjoined(directory, *, stateful: bool = True):
    """A trainer over a ControlRollout, or where not `stateful` a plain function that keeps no state,
    checkpointed to `directory`, made from an integer seed before this process joins its pool, as a program
    that readies its process after making its trainer makes it."""
    from dew.checkpoints import Checkpoints

    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 4)
    rollout = (ControlRollout(objective, walks, unroll_length=4) if stateful
               else (lambda state, batch, key: batch))
    return Trainer(objective, optax.adam(1e-3), key=0, checkpoints=Checkpoints(str(directory)),
                   rollout=rollout)


def fitted(trainer) -> dict:
    """The pool's size and the type of the ValueError a step's fit of `trainer` raised, or None. The run's
    loader is the idle one read through a generator, which reports no position, so its checkpoint holds
    none: a pool of another size places it, and only the rollout's state refuses it."""
    held = idle()
    unpositioned = dataclasses.replace(
        held, train=lambda partition: (batch for batch in held.train(partition)))
    try:
        trainer.fit(unpositioned, steps=1)
        refused = None
    except ValueError as error:
        refused = type(error).__name__
    return {"processes": jax.process_count(), "refused": refused}


def placed(trainers) -> dict:
    """The pool's size and, for each of `trainers` by name, the type of the ValueError its placement of
    the run its checkpoints hold raised, or None."""
    refused = {}
    for name, trainer in trainers.items():
        try:
            trainer.place()
            refused[name] = None
        except ValueError as error:
            refused[name] = type(error).__name__
    return {"processes": jax.process_count(), "refused": refused}


def flown(args):
    """A pool's evaluation of a split scored by its episodes, each process flying its own walks from one
    key: every process's scores and event key; then the same split whose scorer fails, and whose scorer
    reports one of its two scores, on the last process alone, each process's error's type."""
    from dew.objectives.base import Shown
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, Evaluation

    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    variables = nn.unbox(objective.init(jax.random.key(0)))
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
    arguments = worker_parser({"fitted": fitted, "placed": placed, "flown": flown}, __doc__).parse_args()
    modes: dict[str, Callable] = {"flown": flown}
    run = arguments.out.parent / "run"
    # Made before serve_mode joins the pool, which it could not join had making a trainer opened JAX.
    if arguments.mode == "fitted":
        made = unjoined(run)
        modes["fitted"] = lambda arguments: fitted(made)
    elif arguments.mode == "placed":
        trainers = {"stateful": unjoined(run), "stateless": unjoined(run, stateful=False)}
        modes["placed"] = lambda arguments: placed(trainers)
    serve_mode(modes, arguments)
