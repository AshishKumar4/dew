"""Which kernels a model flow's programs launch split over two devices (`split_routes`), read in a process of
the flow's own XLA flags (`SPLIT_ROUTES`, tests/test_per_shard.py's
`test_a_kernel_every_split_flow_launches_has_a_cell`): a module of its own, so that process imports the
flows alone and none of the per-shard cells tests/test_per_shard.py states as it is collected (20 s of
every flow's 100, row 6846)."""

import jax
import numpy as np
from model_flows import KINDS, flow_named, generic_kernels

GENERATIONS = ("sm80", "sm90")
"""The GPU generations the model flows run on (an A100's, a GH200's: tests/gpu_flow.py's tiers), whose
split programs name the kernels that must run per shard."""


SPLIT_ROUTES = """import json, sys
import dew  # before jaxlib loads, so XLA_FLAGS' flags are proven held (`dew._devices.flags_hold`)
import split_flows
print(json.dumps(split_flows.split_routes(sys.argv[1])))
"""
"""A flow's split routes' process (`split_routes`, `test_a_kernel_every_split_flow_launches_has_a_cell`)."""


def split_routes(name: str) -> list[list[str]]:
    """The kernels off the reference path (`generic_kernels`) that flow `name`'s programs (`KINDS`: its
    training step, its decode loop) launch traced as each of `GENERATIONS`, their rows split over a run's
    two-device data axis under a run's layout (`placing`), at the flow's matmul precision, as a run traces
    them: each `[choice, kernel]`."""
    from dew.distributed.placement import Layout, MeshSpec
    from dew.nn.kernels import generation
    from dew.nn.sharding import placing

    flow = flow_named(name)
    generic = generic_kernels()
    mesh = MeshSpec().build(jax.devices()[:2])
    model = flow.model
    params = jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=jax.NamedSharding(
            mesh, jax.sharding.PartitionSpec())),
        jax.eval_shape(lambda: model.init(jax.random.key(0), np.zeros((flow.rows, flow.length), np.int32)))[
            "params"])
    ids = jax.ShapeDtypeStruct((flow.rows, flow.length), np.int32,
                               sharding=jax.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
    found = set()
    for target in GENERATIONS:
        for kind in KINDS:
            # At the flow's own matmul precision (`Flow.precision`), which a kernel's legality may read (an
            # fp32 SSD's Triton port takes the default precision alone), as its run multiplies.
            with (generation.assumed(target), jax.default_matmul_precision(flow.precision),
                  jax.set_mesh(mesh), placing(Layout()), generation.routes() as routes):
                jax.jit(getattr(flow, kind)(model)).trace(params, ids)
            found |= {(str(route.choice), route.kernel) for route in routes
                      if route.kernel not in generic[str(route.choice)]}
    return sorted(map(list, found))
