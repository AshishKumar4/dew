"""Compare local shards of real-process expert exchange with whole-array routing."""

import functools
import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.moe import ExpertMLP
from dew.training import MeshSpec


def main() -> None:
    rank, coordinator, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2,
                               process_id=rank, local_device_ids=[0], initialization_timeout=30)
    from moe_support import objective, routing_case

    mesh = MeshSpec(expert=2).build()
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
            parameters = nn.unbox(model.init(jax.random.key(19), x, weights, ids))
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
            def loss(p, x, weights, ids, *, model=model):
                return objective(model.clone(dispatch='exchange'), ids, p, x, weights)
            actual = jax.jit(jax.value_and_grad(loss, (0, 1, 2), has_aux=True),
                             out_shardings=out_specs)(parameters, x, weights, ids)
        assert actual[0][1].dtype == dtype
        maxima = []
        (value, out), (d_parameters, d_x, d_weights) = actual
        (value_, out_), (d_parameters_, d_x_, d_weights_) = expected
        compared = [*((ours, theirs, False) for ours, theirs in zip(
            jax.tree.leaves((value, out, d_parameters, d_weights)),
            jax.tree.leaves((value_, out_, d_parameters_, d_weights_)), strict=True)), (d_x, d_x_, True)]
        for ours, theirs, routed in compared:
            whole = np.asarray(theirs).astype(np.float64)
            # The exchange rounds each route's cotangent to the rows' dtype as it crosses back, before its
            # token sums them wide; the one-process dispatch sums them unrounded (row 12396): a narrow
            # input's cotangent apart by a rounding a route, at its largest entry's ulp.
            narrow = routed and jnp.finfo(theirs.dtype).bits < 32
            rtol = float(jnp.finfo(theirs.dtype).eps) if narrow else 3e-5
            atol = rtol * float(np.max(np.abs(whole), initial=0)) if narrow else 3e-5
            for shard in ours.addressable_shards:
                a = np.asarray(shard.data).astype(np.float64)
                b = whole[shard.index]
                np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
                # Each error as its share of its leaf's tolerance (`assert_exchange_across_processes`).
                maxima.append(float(np.max(np.abs(a - b) / (atol + rtol * np.abs(b)), initial=0)))
        errors[f'{jnp.dtype(dtype).name}/{activation}/{skewed}'] = maxima
    output.write_text(json.dumps({'processes': jax.process_count(), 'errors': errors}))
    jax.distributed.shutdown()


if __name__ == '__main__':
    main()
