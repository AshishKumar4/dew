"""A pool's serving from one server over its processes' devices, or its refusal of serving from one
process's engine: the modes tests/test_multiprocess.py runs over two processes (`run_pool`'s `script`)."""

import itertools
import threading
from collections.abc import Callable

import numpy as np
from flax import linen as nn
from multiprocess_worker import TINY
from process_support import serve_mode, worker_parser
from rl_support import pooled_rollout


def mode_served_refusals(args) -> dict:
    """What a pool refuses to serve from one process's engine, on every process and before any program
    compiles: an engine over a scoring server whose weights span the pool (`ScoringServer.pooled`). The
    outcome is the error's text, or None for one that served, and the programs compiled meanwhile."""
    import jax
    import jax.numpy as jnp

    from dew.inference import Engine
    from dew.inference.scoring import ScoringServer
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.training import MeshSpec

    model = CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=12,
                              dtype="float32", attention_impl="xla")
    params = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)))
    mesh = MeshSpec(fsdp=jax.device_count()).build()
    pooled = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    weights = jax.tree.map(lambda leaf: jax.make_array_from_callback(
        leaf.shape, pooled, lambda index: np.asarray(leaf)[index]), params)
    compiled = []

    def counted(event: str, duration: float, **_) -> None:
        if event.endswith("backend_compile_duration"):
            compiled.append(duration)

    jax.monitoring.register_event_duration_secs_listener(counted)

    class Unscored:
        """A scoring task whose passes none reaches: the engine refuses its server first."""

    try:
        Engine(ScoringServer(Unscored(), weights))
        scoring = None
    except ValueError as error:
        scoring = str(error)
    return {"scoring": scoring, "compiled": len(compiled)}


def served_run(records: list) -> tuple:
    """A GRPO run over the pool whose rollouts one server over every process's devices draws ahead of the
    trainer (`RolloutScheduler.served`), its weights pushed every second update: the rollout, the trainer
    and each process's prompt stream, its call records logged into `records`."""
    import jax
    import optax

    from dew.data.dataset import Loader
    from dew.objectives.rl import GRPOObjective, RolloutScheduler
    from dew.sampling import Sampling
    from dew.training import Layout, MeshSpec, Trainer

    processes = jax.process_count()
    model, params, _, local = pooled_rollout()
    rows = len(local["prompt"])
    objective = GRPOObjective(model, seq_len=7, variables=params, behavior_importance=2.0)
    rollout = RolloutScheduler.served(objective, lambda text, record: float(len(text)), tasks=rows, slots=8,
                                      groups=2, max_new_tokens=4, sampling=Sampling(temperature=1.0),
                                      ahead=1, max_lag=2, sync_every=2, log=records.append)
    trainer = Trainer(objective, optax.sgd(0.01), key=jax.random.key(5),
                      mesh=MeshSpec(fsdp=jax.device_count()), layout=Layout(min_shard=TINY), rollout=rollout)
    data = Loader(train=lambda partition: itertools.repeat(local), val=None, records=rows * processes,
                  batch=rows * processes)
    return rollout, trainer, rollout.tasks(data)


def mode_served_rollout(args) -> dict:
    """A served GRPO run over the pool (`served_run`) whose server every process steps together as each
    call admits its groups and reloads at each push (`NativeRolloutServer.waited`, row 4279): three
    updates, so a batch submitted a step ahead trails the trainer. Each process's report: the steps
    trained, whether its server spans the pool, the engine threads left (a pooled server has none), and
    each call's admitted groups and lag."""
    from dew.inference.rollouts import NativeRolloutServer

    records = []
    rollout, trainer, tasks = served_run(records)
    final = trainer.fit(tasks, steps=3, log_every=1, checkpoint_every=None)
    server = rollout.acquisition.weights
    assert isinstance(server, NativeRolloutServer)
    pooled = server.pooled
    rollout.close()
    return {"step": int(final.step), "pooled": pooled,
            "engines": [thread.name for thread in threading.enumerate()
                        if thread.name.startswith("dew-rollout-server")],
            "groups": [record.groups for record in records], "lags": [record.lag for record in records]}


def mode_served_step_failure(args) -> dict:
    """A served GRPO run over the pool (`served_run`) whose server's second step fails on process 1
    alone, after the step's own agreements (as its settling of a request may, `Server._settle`): every
    process hears it at that step's agreement and stops its server, none waiting on a round the other
    never reads (`NativeRolloutServer.waited`). Each process's report: the error it raised and whether its
    server then refuses a draw as stopped."""
    import jax

    from dew.inference.rollouts import NativeRolloutServer
    from dew.inference.serving import Server

    stepped = Server.step
    calls = []

    def failing(server, *step_args, **step_kwargs):
        calls.append(None)
        stepped(server, *step_args, **step_kwargs)
        if jax.process_index() == 1 and len(calls) == 2:
            raise RuntimeError("an injected step failure")

    Server.step = failing
    rollout, trainer, tasks = served_run([])
    try:
        trainer.fit(tasks, steps=3, log_every=1, checkpoint_every=None)
        raised = None
    except RuntimeError as error:
        raised = f"{type(error).__name__}: {error}"
    server = rollout.acquisition.weights
    assert isinstance(server, NativeRolloutServer)
    try:
        server.submit([3, 4], 2, key=0)
        stopped = None
    except RuntimeError as error:
        stopped = str(error)
    return {"raised": raised, "stopped": stopped}


MODES: dict[str, Callable[..., dict]] = {"served_refusals": mode_served_refusals,
                                         "served_rollout": mode_served_rollout,
                                         "served_step_failure": mode_served_step_failure}


if __name__ == "__main__":
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args())
