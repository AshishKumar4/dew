"""Router parity readings, the router fixtures the balance tests share (`favour_expert_zero` and
`uniform_gates`), and the two-process expert-exchange launch the MoE tests share. The routers stay
with their tests: V2's softmax and best-expert group score and V3's sigmoid, two-best group score and
bias differ."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
from process_support import start_process


def by_expert(indices, weights):
    """One token's slots ordered by expert id, indices and weights together."""
    order = np.argsort(np.asarray(indices), axis=-1)
    return (np.take_along_axis(np.asarray(indices), order, axis=-1),
            np.take_along_axis(np.asarray(weights), order, axis=-1))


def router_variables(tensors, bias=False):
    """The reference gate weight as a `Router` parameter tree.

    torch Linear holds [out, in] and Dew keeps [in, out], the transpose every
    kernel takes in dew.interop.hf_decoders.
    """
    variables = {"params": {"kernel": jnp.asarray(tensors["mlp.gate.weight"].T)}}
    if bias:
        variables["moe"] = {"e_score_correction_bias": jnp.asarray(
            tensors["mlp.gate.e_score_correction_bias"])}
    return variables


def favour_expert_zero(variables, amount):
    """`variables` with expert 0's balancing bias raised in every router, so every token favours it."""
    moe = flatten_dict(variables["moe"])
    moe = {path: leaf.at[0].add(amount) if path[-1] == "e_score_correction_bias" else leaf
           for path, leaf in moe.items()}
    return {**variables, "moe": unflatten_dict(moe)}


def uniform_gates(variables):
    """`variables` with every router's gate at zero. Its scores are then the same for every token, so its
    choice is its balancing bias alone, and no mask a microbatch draws can change what it counts."""
    params = flatten_dict(variables["params"])
    params = {path: jnp.zeros_like(leaf) if path[-2:] == ("gate", "kernel") else leaf
              for path, leaf in params.items()}
    return {**variables, "params": unflatten_dict(params)}


def exchange_worker(script, out, processes, process_id, coordinator):
    """One rank of an expert-exchange script, as run_pool's `start`: one CPU
    device, its own session, and the rank, coordinator and report as argv."""
    return start_process(
        [sys.executable, str(script), str(process_id), coordinator, str(out)], devices=1)


def assert_exchange_across_processes(worker: Path, tmp_path: Path, bound: float) -> None:
    """`worker` (a script beside the tests) run in two real CPU processes (`exchange_worker`):
    each reports two processes, and every error it measured against its one-process reference
    within `bound`."""
    from process_support import run_pool

    for report in run_pool(worker, tmp_path, 2, timeout=120, start=exchange_worker):
        assert report['processes'] == 2
        for errors in report['errors'].values():
            assert max(errors) < bound, errors


def routing_case(tokens: int, experts: int, top_k: int, skewed: bool):
    rng = np.random.default_rng(219)
    x = rng.normal(size=(tokens, 8)).astype(np.float32)
    weights = rng.uniform(0.1, 0.9, size=(tokens, top_k)).astype(np.float32)
    choices = (np.tile(np.arange(top_k), (tokens, 1)) if skewed
               else np.argsort(rng.normal(size=(tokens, experts)), axis=-1)[:, :top_k])
    return x, weights, choices.astype(np.int32)


def objective(module, indices, parameters, x, weights):
    out = module.apply(parameters, x, weights, indices)
    return jnp.sum(jnp.sin(out.astype(jnp.promote_types(out.dtype, jnp.float32)))), out


def reference_case(skewed: bool):
    name = 'exchange_skewed' if skewed else 'exchange_random'
    with np.load(Path(__file__).parent / 'fixtures' / 'gpt_oss' / (name + '.npz')) as fixture:
        arrays = {name: np.asarray(value) for name, value in fixture.items()}

    def parameters(prefix):
        return {'router': {'kernel': jnp.asarray(arrays[prefix + 'router.weight'].T),
                           'bias': jnp.asarray(arrays[prefix + 'router.bias'])},
                'experts': {name.removeprefix(prefix + 'experts.'): jnp.asarray(value)
                            for name, value in arrays.items() if name.startswith(prefix + 'experts.')}}

    return arrays, parameters(''), parameters('grad.')


def training_shardings(mesh):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return {
        'router': {'kernel': NamedSharding(mesh, P('fsdp', 'expert')),
                   'bias': NamedSharding(mesh, P('expert'))},
        'experts': {'gate_up_proj': NamedSharding(mesh, P('expert', None, 'fsdp')),
                    'gate_up_proj_bias': NamedSharding(mesh, P('expert', 'fsdp')),
                    'down_proj': NamedSharding(mesh, P('expert', 'fsdp')),
                    'down_proj_bias': NamedSharding(mesh, P('expert'))}}


def training_step(model, optimizer):
    def objective(parameters, x):
        output = jnp.asarray(model.apply({'params': parameters}, x))
        return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

    def step(parameters, state, x):
        result, gradients = jax.value_and_grad(objective, (0, 1), has_aux=True)(parameters, x)
        updates, state = optimizer.update(gradients[0], state, parameters)
        return result, gradients, optax.apply_updates(parameters, updates), state

    return step
