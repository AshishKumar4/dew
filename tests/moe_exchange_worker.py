"""Compare local shards of real-process expert exchange with whole-array routing."""

import functools
import itertools
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.moe import ExpertMLP
from dew.training import MeshSpec, build_mesh


def main() -> None:
    rank, coordinator, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2,
                               process_id=rank, local_device_ids=[0], initialization_timeout=30)
    from test_moe_exchange import objective, routing_case

    mesh = build_mesh(MeshSpec(expert=2))
    split = NamedSharding(mesh, P('expert'))
    replicated = NamedSharding(mesh, P())
    errors: dict[str, list[float]] = {}
    for dtype, activation, skewed in itertools.product(
            (jnp.float32, jnp.bfloat16), ('swiglu', 'geglu_exact'), (False, True)):
        inputs, routing_weights, choices = routing_case(16, 4, 2, skewed)
        model = ExpertMLP(4, 12, 8, dtype=dtype, activation=activation)
        with jax.default_device(jax.local_devices()[0]):
            x = jnp.asarray(inputs, dtype)
            weights, ids = jnp.asarray(routing_weights), jnp.asarray(choices)
            parameters = model.init(jax.random.key(19), x, weights, ids)
            expected = jax.jit(jax.value_and_grad(functools.partial(objective, model, ids),
                                                (0, 1, 2), has_aux=True))(parameters, x, weights)
        parameter_specs = jax.tree.map(lambda _: split, parameters)

        def distribute(value: jax.Array, sharding: NamedSharding) -> jax.Array:
            local = np.asarray(value)
            return jax.make_array_from_callback(local.shape, sharding, lambda index: local[index])

        parameters = jax.tree.map(distribute, parameters, parameter_specs)
        x, weights, ids = (distribute(value, split) for value in (x, weights, ids))
        out_specs = ((replicated, split), (parameter_specs, split, split))
        with jax.set_mesh(mesh):
            def loss(p, x, weights, ids):
                return objective(model.clone(dispatch='exchange'), ids, p, x, weights)
            actual = jax.jit(jax.value_and_grad(loss, (0, 1, 2), has_aux=True),
                             out_shardings=out_specs)(parameters, x, weights, ids)
        assert actual[0][1].dtype == dtype
        maxima = []
        for ours, theirs in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            for shard in ours.addressable_shards:
                a = np.asarray(shard.data).astype(np.float64)
                b = np.asarray(theirs)[shard.index].astype(np.float64)
                np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
                maxima.append(float(np.max(np.abs(a - b), initial=0)))
        errors[f'{jnp.dtype(dtype).name}/{activation}/{skewed}'] = maxima
    output.write_text(json.dumps({'processes': jax.process_count(), 'errors': errors}))
    jax.distributed.shutdown()


if __name__ == '__main__':
    main()
