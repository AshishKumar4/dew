"""Pooled GPT OSS updates with real all-to-all traffic between CPU processes."""

import itertools
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.gpt_oss import GptOssMLP
from dew.training import MeshSpec, build_mesh


def place(value: jax.Array, sharding: NamedSharding) -> jax.Array:
    if value.is_fully_addressable:
        host = np.asarray(value)
        return jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])
    return jax.device_put(value, sharding)


def compare_local(actual, expected) -> float:
    maximum = 0.0
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        for first, second in zip(a.addressable_shards, b.addressable_shards, strict=True):
            left, right = np.asarray(first.data, np.float64), np.asarray(second.data, np.float64)
            np.testing.assert_allclose(left, right, atol=3e-5, rtol=3e-5)
            maximum = max(maximum, float(np.max(abs(left - right))))
    return maximum


def main() -> None:
    rank, coordinator, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    jax.distributed.initialize(coordinator_address=coordinator, num_processes=2,
                               process_id=rank, local_device_ids=[0], initialization_timeout=30)
    from test_moe_biased_exchange import reference_case, training_shardings, training_step

    try:
        mesh = build_mesh(MeshSpec(expert=2))
        tokens, replicated = NamedSharding(mesh, P('expert')), NamedSharding(mesh, P())
        specs = training_shardings(mesh)
        errors: dict[str, list[float]] = {}
        for dtype, skewed in itertools.product((jnp.float32, jnp.bfloat16), (False, True)):
            arrays, parameters, reference_gradients = reference_case(skewed)
            x = place(jnp.asarray(arrays['hidden'].reshape(-1, 8), dtype), tokens)
            parameters = jax.tree.map(place, parameters, specs)
            optimizer = optax.adam(1e-3)
            state = optimizer.init(parameters)
            state_specs = jax.tree.map(lambda value: replicated if value.ndim == 0 else value.sharding, state)
            state = jax.tree.map(place, state, state_specs)
            outputs = ((replicated, tokens), (specs, tokens), specs, state_specs)
            model = GptOssMLP(8, 12, 4, 2, dtype=dtype)
            baseline = []
            maxima = []
            for dispatch in ('global', 'exchange'):
                step = training_step(model.clone(dispatch=dispatch), optimizer)
                with jax.set_mesh(mesh):
                    compiled = jax.jit(step, out_shardings=outputs)
                    p, s = parameters, state
                    for index in range(3):
                        result = jax.block_until_ready(compiled(p, s, x))
                        assert result[0][1].dtype == dtype
                        if dispatch == 'global':
                            baseline.append(result)
                        else:
                            maxima.append(compare_local(result, baseline[index]))
                        if index == 0 and dtype == jnp.float32:
                            expected = ((arrays['loss'], arrays['output'].reshape(-1, 8)),
                                        (reference_gradients, arrays['input_grad'].reshape(-1, 8)))
                            for actual, target in zip(jax.tree.leaves(result[:2]),
                                                       jax.tree.leaves(expected), strict=True):
                                for shard in actual.addressable_shards:
                                    np.testing.assert_allclose(np.asarray(shard.data),
                                        np.asarray(target)[shard.index], atol=3e-5, rtol=3e-5)
                        p, s = result[2:]
            errors[f'{jnp.dtype(dtype).name}/{skewed}'] = maxima
        output.write_text(json.dumps({'processes': jax.process_count(), 'errors': errors}))
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
