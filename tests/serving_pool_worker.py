"""A pool's serving from one server over its processes' devices, or its refusal of serving from one
process's engine: the modes tests/test_multiprocess.py runs over two processes (`run_pool`'s `script`)."""

from collections.abc import Callable

import numpy as np
from flax import linen as nn
from process_support import serve_mode, worker_parser


def mode_served_refusals(args) -> dict:
    """What a pool refuses to serve from one process's engine, on every process and before any program
    compiles: an engine over a scoring server whose weights span the pool (`ScoringServer.pooled`), and a
    served GRPO rollout, whose engine steps one process's server (`RolloutScheduler.served`, row 4279).
    Each outcome is the error's text, or None for one that served, and the programs compiled meanwhile."""
    import jax
    import jax.numpy as jnp

    from dew.inference import Engine
    from dew.inference.scoring import ScoringServer
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.rl import GRPOObjective, RolloutScheduler
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

    def refused(call) -> str | None:
        try:
            call()
        except ValueError as error:
            return str(error)
        return None

    class Unscored:
        """A scoring task whose passes none reaches: the engine refuses its server first."""

    scoring = refused(lambda: Engine(ScoringServer(Unscored(), weights)))
    served = refused(lambda: RolloutScheduler.served(GRPOObjective(model, seq_len=7, variables=params),
                                                     lambda text, record: 0.0, tasks=1, slots=2))
    return {"scoring": scoring, "served": served, "compiled": len(compiled)}


MODES: dict[str, Callable[..., dict]] = {"served_refusals": mode_served_refusals}


if __name__ == "__main__":
    serve_mode(MODES, worker_parser(MODES, __doc__).parse_args())
