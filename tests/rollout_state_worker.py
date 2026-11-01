"""A Trainer made in a pool over a rollout that checkpoints its state, and over one that keeps none; a
split scored by its episodes evaluated in a pool, its scorer once failing on one process alone."""

import jax
import optax
from process_support import serve_mode, worker_parser
from test_env_rollout import Walk

from dew.objectives.rl.control import ActorCritic, ControlPPOObjective, ControlRollout
from dew.rl.env import Batched, TimeLimit
from dew.training import Trainer


def made(args):
    """Whether a pool of `args.processes` makes a trainer over a stateful ControlRollout and over a plain
    function that returns the batch it is given, each refusal's message kept."""
    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 4)
    found = {}
    for name, rollout in (("stateful", ControlRollout(objective, walks, unroll_length=4)),
                          ("stateless", lambda state, batch, key: batch)):
        try:
            Trainer(objective, optax.adam(1e-3), key=jax.random.key(0), rollout=rollout)
            found[name] = None
        except ValueError as refused:
            found[name] = str(refused)
    found["processes"] = jax.process_count()
    return found


def flown(args):
    """A pool's evaluation of a split scored by its episodes, each process flying its own walks from one
    key: every process's scores and event key; then the same split whose scorer fails on the last process
    alone, each process's error's type."""
    from dew.objectives.base import Shown
    from dew.rl.evaluation import episodes
    from dew.training import Episodes, Evaluation

    objective = ControlPPOObjective(ActorCritic(1, (8,), (8,)), 1, discount=0.9)
    variables = objective.init(jax.random.key(0))
    walks = Batched(TimeLimit(Walk(limit=50.0), episode_length=8), 8)
    scored = episodes(objective.greedy, walks, 8)
    split = Episodes(scored, {"return": Shown(better="higher"), "steps": Shown()})
    result = Evaluation.run(objective, variables, split, key=jax.random.key(3), step=2, split="flight")

    def failing(variables, key):
        if jax.process_index() == jax.process_count() - 1:
            raise RuntimeError("this process's environments failed")
        return scored(variables, key)

    try:
        Evaluation.run(objective, variables, Episodes(failing, split.scores), key=jax.random.key(3), step=2,
                       split="flight")
        failed = None
    except Exception as error:
        failed = type(error).__name__
    return {"scores": result.scores, "event": list(result.event_key), "failed": failed}


if __name__ == "__main__":
    modes = {"made": made, "flown": flown}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
