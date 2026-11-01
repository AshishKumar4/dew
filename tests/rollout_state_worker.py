"""A Trainer made in a pool over a rollout that checkpoints its state, and over one that keeps none."""

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


if __name__ == "__main__":
    modes = {"made": made}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
