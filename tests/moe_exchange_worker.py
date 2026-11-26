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
            parameters_, x_, weights_, ids_ = parameters, x, weights, ids
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
        compared = list(zip(jax.tree.leaves((value, out, d_parameters, d_weights, d_x)),
                            jax.tree.leaves((value_, out_, d_parameters_, d_weights_, d_x_)), strict=True))
        for at, (ours, theirs) in enumerate(compared):
            whole = np.asarray(theirs).astype(np.float64)
            allowed = 3e-5 + 3e-5 * np.abs(whole)
            if at == len(compared) - 1 and jnp.finfo(dtype).bits < 32:
                # The input's narrow cotangent, its routes rounded on the wire (`routes_rounded_bound`).
                with jax.default_device(jax.local_devices()[0]):
                    routes = route_cotangents(model, parameters_, x_, weights_, ids_, out_)
                allowed = routes_rounded_bound(routes, jnp.finfo(dtype).eps / 2)
            maxima += _held(ours, whole, allowed)
        errors[f'{jnp.dtype(dtype).name}/{activation}/{skewed}'] = maxima
    output.write_text(json.dumps({'processes': jax.process_count(), 'errors': errors}))
    jax.distributed.shutdown()


def _held(ours: jax.Array, whole: np.ndarray, allowed: np.ndarray) -> list[float]:
    """Each local shard of `ours` within `allowed` of the whole reference, entry by entry, each shard's
    largest error as its share of what its entry allows."""
    shares = []
    for shard in ours.addressable_shards:
        error = np.abs(np.asarray(shard.data).astype(np.float64) - whole[shard.index])
        bound = np.broadcast_to(allowed, whole.shape)[shard.index]
        assert np.all(error <= bound), (error, bound)
        shares.append(float(np.max(error / np.maximum(bound, np.finfo(np.float64).tiny), initial=0)))
    return shares


def route_cotangents(model, parameters, x, weights, ids, out) -> np.ndarray:
    """Each route's share of the input's cotangent under the objective's (`moe_support.objective`: the
    output's cosine, in its dtype), `[top_k, *x.shape]` float64: the experts' layer of each route alone, at
    the one-process dispatch, pulled back from the one-process output's cotangent."""
    cotangent = jnp.cos(out.astype(jnp.float32)).astype(out.dtype)
    routes = []
    for k in range(ids.shape[-1]):
        def routed(rows, k=k):
            return model.apply(parameters, rows, weights[:, k:k + 1], ids[:, k:k + 1])
        routes.append(np.asarray(jax.vjp(routed, x)[1](cotangent)[0], np.float64))
    return np.stack(routes)


def routes_rounded_bound(routes: np.ndarray, unit: float) -> np.ndarray:
    """How far the exchange's input cotangent may lie from the one-process dispatch's, entry by entry: the
    exchange rounds each route's cotangent to the rows' dtype as it crosses back (`unit`, its unit roundoff,
    2**-8 for bfloat16, of that route's magnitude), then sums a token's routes wide and rounds once, where
    the one-process dispatch sums them unrounded and rounds once (row 12396), so the two are apart by at
    most `unit` of each route's magnitude (`top_k` roundings, one a route's cotangent on the wire) and
    `unit` of the sum's for each side's one rounding of it (`routes`, `[top_k, *x.shape]`, each route's
    cotangent at the one-process dispatch)."""
    return unit * (np.sum(np.abs(routes), axis=0) + 2 * np.abs(np.sum(routes, axis=0)))


if __name__ == '__main__':
    main()
