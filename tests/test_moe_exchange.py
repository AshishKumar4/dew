"""Dropless exchange and capacity dropping against the global sort/gather path on CPU."""

import functools
import math
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture
from dew.nn.moe import ExpertMLP, capacity_positions
from dew.training import MeshSpec, build_mesh

pytestmark = pytest.mark.mesh


def routing_case(tokens: int, experts: int, top_k: int, skewed: bool):
    rng = np.random.default_rng(219)
    x = rng.normal(size=(tokens, 8)).astype(np.float32)
    weights = rng.uniform(0.1, 0.9, size=(tokens, top_k)).astype(np.float32)
    choices = (np.tile(np.arange(top_k), (tokens, 1)) if skewed
               else np.argsort(rng.normal(size=(tokens, experts)), axis=-1)[:, :top_k])
    return x, weights, choices.astype(np.int32)


def objective(module, indices, parameters, x, weights):
    out = module.apply(parameters, x, weights, indices)
    return jnp.sum(jnp.sin(out.astype(jnp.float32))), out


@pytest.mark.parametrize("shards,fsdp,tokens,top_k,skewed,scale_inputs,activation,limit", [
    (2, 4, 24, 3, False, False, 'swiglu', None),
    (4, 2, 24, 2, True, False, 'swiglu', None),
    (8, 1, 24, 3, False, True, 'geglu_exact', 0.7),
    (4, 2, 7, 2, True, True, 'geglu', None),
    (4, 1, 1, 1, True, False, 'swiglu', None),
    (2, 1, 0, 2, True, False, 'swiglu', None),
])
def test_exchange_retains_outputs_and_all_differentiable_inputs(
        shards, fsdp, tokens, top_k, skewed, scale_inputs, activation, limit):
    x, weights, indices = map(jnp.asarray, routing_case(tokens, 8, top_k, skewed))
    reference = ExpertMLP(8, 12, 8, scale_inputs=scale_inputs, activation=activation,
                          swiglu_limit=limit)
    parameters = reference.init(jax.random.key(19), x, weights, indices)
    expected = jax.jit(jax.value_and_grad(functools.partial(objective, reference, indices),
                                        (0, 1, 2), has_aux=True))(parameters, x, weights)
    mesh = build_mesh(MeshSpec(expert=shards, fsdp=fsdp))
    with jax.set_mesh(mesh):
        actual = jax.jit(jax.value_and_grad(functools.partial(
            objective, reference.clone(dispatch='exchange'), indices),
            (0, 1, 2), has_aux=True))(parameters, x, weights)
    # Chunking changes the order of fp32 expert-weight cotangent sums.
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf.astype(jnp.float32), expected_leaf.astype(jnp.float32),
                                   atol=3e-5, rtol=3e-5)
    if skewed:
        for projection in actual[1][0]['params'].values():
            np.testing.assert_array_equal(projection['kernel'][top_k:], 0)


def test_exchange_preserves_placed_expert_and_width_gradients():
    x, weights, indices = map(jnp.asarray, routing_case(24, 8, 2, False))
    model = ExpertMLP(8, 16, 8)
    parameters = model.init(jax.random.key(19), x, weights, indices)
    mesh = build_mesh(MeshSpec(expert=4, fsdp=2))
    columns = NamedSharding(mesh, P('expert', None, 'fsdp'))
    parameter_specs = {'params': {
        'gate_proj': {'kernel': columns}, 'up_proj': {'kernel': columns},
        'down_proj': {'kernel': NamedSharding(mesh, P('expert', 'fsdp'))}}}
    token_spec = NamedSharding(mesh, P(('expert', 'fsdp')))
    parameters = jax.device_put(parameters, parameter_specs)
    x, weights, indices = (jax.device_put(value, token_spec) for value in (x, weights, indices))
    specs = ((NamedSharding(mesh, P()), token_spec), (parameter_specs, token_spec, token_spec))
    with jax.set_mesh(mesh):
        expected = jax.jit(jax.value_and_grad(functools.partial(objective, model, indices),
                            (0, 1, 2), has_aux=True), out_shardings=specs)(parameters, x, weights)
        actual = jax.jit(jax.value_and_grad(functools.partial(
            objective, model.clone(dispatch='exchange'), indices),
            (0, 1, 2), has_aux=True), out_shardings=specs)(parameters, x, weights)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)



@pytest.mark.parametrize("parallel", [False, True])
def test_decoder_exchange_retains_logits_and_expert_bias_observations(parallel):
    mixture = Mixture(experts=4, top_k=2, parallel=parallel, bias=not parallel,
                      shared_features=0 if parallel else 12)
    reference = CausalTransformer(vocab_size=16, emb_features=8, num_heads=2,
                                  num_layers=1, mlp_features=12, max_seq_len=4,
                                  qk_norm=False, mixture=mixture)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    parameters = reference.init(jax.random.key(10), tokens)
    expected, observations = reference.apply(parameters, tokens, mutable=['router'])
    from dataclasses import replace
    exchanged = reference.clone(mixture=replace(mixture, dispatch='exchange'))
    mesh = build_mesh(MeshSpec(expert=2))
    with jax.set_mesh(mesh):
        actual, actual_observations = jax.jit(lambda p, t: exchanged.apply(
            p, t, mutable=['router']))(parameters, tokens)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual_observations),
                                          jax.tree.leaves(observations), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=1e-7, rtol=1e-6)


def test_exchange_requires_an_expert_axis_that_owns_whole_experts():
    x, weights, indices = map(jnp.asarray, routing_case(8, 6, 2, False))
    module = ExpertMLP(6, 12, 8, dispatch='exchange')
    parameters = module.init(jax.random.key(0), x, weights, indices)
    for shards in (1, 4):
        mesh = build_mesh(MeshSpec(expert=shards))
        with jax.set_mesh(mesh), pytest.raises(ValueError, match='divides num_experts'):
            module.apply(parameters, x, weights, indices)


def test_unknown_dispatch_is_rejected_at_the_mixture():
    with pytest.raises(ValueError, match='dispatch'):
        Mixture(experts=4, dispatch='drop')
    with pytest.raises(ValueError, match='capacity_factor'):
        Mixture(experts=4, capacity_factor=0.0)


def maxtext_positions(indices: np.ndarray, experts: int) -> np.ndarray:
    """Each slot's place in its expert's queue as MaxText's `generate_masks`
    counts it: a running count over a sequence's tokens and, within a
    token, over its choices in order."""
    *_, length, top_k = indices.shape
    flat = indices.reshape(-1, length * top_k)
    running = np.cumsum(np.eye(experts, dtype=np.int64)[flat], axis=1)
    return (np.take_along_axis(running, flat[..., None], axis=-1)[..., 0] - 1).reshape(indices.shape)


def maxtext_capacity(length: int, top_k: int, experts: int, factor: float) -> int:
    return int(max(math.ceil(length * top_k / experts) * factor, factor))


@pytest.mark.parametrize('experts,top_k,factor', [(8, 2, 1.0), (4, 1, 1.25), (16, 4, 0.5)])
def test_capacity_queues_each_sequence_in_token_order(experts, top_k, factor):
    scores = np.random.default_rng(3).normal(size=(3, 10, experts))
    scores[..., :2] += 1.0
    indices = np.argsort(-scores, axis=-1)[..., :top_k].astype(np.int32)
    positions, capacity = capacity_positions(jnp.asarray(indices), experts, factor)
    np.testing.assert_array_equal(positions, maxtext_positions(indices, experts))
    assert capacity == maxtext_capacity(10, top_k, experts, factor)


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize('spec,batch,length,top_k', [
    (MeshSpec(expert=2, fsdp=2), 4, 12, 2),
    (MeshSpec(expert=2), 16, 4, 2),
    (MeshSpec(expert=4), 2, 12, 2),
    (MeshSpec(expert=4), 3, 5, 1),
], ids=['whole-sequences', 'two-sequences-a-device', 'part-of-a-sequence', 'padded-straddling'])
@pytest.mark.parametrize('dispatch', ['global', 'exchange'])
def test_capacity_drops_the_same_slots_on_every_layout(spec, batch, length, top_k, dispatch):
    """With a capacity, a layer computes what the dropless layer computes
    with the dropped slots' weights zeroed: outputs, parameter and token
    gradients, and the kept slots' weight gradients. A dropped slot's
    weight has none. The cases split sequences across devices, put several
    on one device, and pad a batch the devices do not divide."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(batch, length, 8)).astype(np.float32)
    weights = rng.uniform(0.1, 0.9, size=(batch, length, top_k)).astype(np.float32)
    scores = rng.normal(size=(batch, length, 8))
    scores[..., :2] += 1.5
    indices = np.argsort(-scores, axis=-1)[..., :top_k].astype(np.int32)
    kept = maxtext_positions(indices, 8) < maxtext_capacity(length, top_k, 8, 1.0)
    assert not kept.all()
    routing = jnp.asarray(indices)
    dropless = ExpertMLP(8, 12, 8)
    parameters = dropless.init(jax.random.key(19), x, weights, routing)
    gradient = functools.partial(jax.value_and_grad, argnums=(0, 1, 2), has_aux=True)
    expected = jax.jit(gradient(functools.partial(objective, dropless, routing)))(
        parameters, x, np.where(kept, weights, 0).astype(np.float32))
    dropping = dropless.clone(dispatch=dispatch, capacity_factor=1.0)
    with jax.set_mesh(build_mesh(spec, jax.devices()[:4])):
        actual = jax.jit(gradient(functools.partial(objective, dropping, routing)))(
            parameters, x, weights)
    (value, out), (d_parameters, d_x, d_weights) = actual
    (expected_value, expected_out), (expected_parameters, expected_x, expected_weights) = expected
    for a, b in zip(jax.tree.leaves((value, out, d_parameters, d_x)),
                    jax.tree.leaves((expected_value, expected_out, expected_parameters, expected_x)),
                    strict=True):
        np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(np.where(kept, d_weights, 0), np.where(kept, expected_weights, 0),
                               atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(np.where(kept, 0, d_weights), 0)


def test_exchange_collectives_work_across_two_real_processes(tmp_path):
    from test_multiprocess import free_port, report_of, terminate, worker_env

    coordinator = f"127.0.0.1:{free_port()}"
    worker = Path(__file__).with_name('moe_exchange_worker.py')
    outputs = [tmp_path / f'rank{rank}.json' for rank in range(2)]
    running = [subprocess.Popen(
        [sys.executable, str(worker), str(rank), coordinator, str(output)],
        env=worker_env(1), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True) for rank, output in enumerate(outputs)]
    try:
        reports = [report_of(process, output, timeout=120)
                   for process, output in zip(running, outputs, strict=True)]
    finally:
        for process in running:
            if process.poll() is None:
                terminate(process)
    for report in reports:
        assert report['processes'] == 2
        for errors in report['errors'].values():
            assert max(errors) < 3e-5, errors
