"""Router parity readings and the two-process expert-exchange launch the MoE
tests share. The routers stay with their tests: V2's softmax and best-expert
group score and V3's sigmoid, two-best group score and bias differ."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


def exchange_worker(script, out, processes, process_id, coordinator):
    """One rank of an expert-exchange script, as run_pool's `start`: one CPU
    device, its own session, and the rank, coordinator and report as argv."""
    return start_process(
        [sys.executable, str(script), str(process_id), coordinator, str(out)], devices=1)


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
