"""Biased, interleaved experts use the same dropless transport as gated experts."""

import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.nn.gpt_oss import GptOssExperts, GptOssMLP
from dew.training import MeshSpec, build_mesh


@pytest.mark.mesh
def test_biased_exchange_keeps_every_selected_expert_and_requires_its_mesh():
    rng = np.random.default_rng(351)
    x = jnp.asarray(rng.normal(size=(16, 8)), jnp.float32)
    weights = jnp.asarray(rng.uniform(.1, .9, size=(16, 2)), jnp.float32)
    # All routes land on one owner; later rounds must retain the excess.
    indices = jnp.tile(jnp.asarray([0, 1], jnp.int32), (16, 1))
    reference = GptOssExperts(8, 12, 4)
    variables = reference.init(jax.random.key(0), x, weights, indices)
    variables['params']['gate_up_proj_bias'] = jnp.asarray(rng.normal(size=(4, 24)), jnp.float32)
    variables['params']['down_proj_bias'] = jnp.asarray(rng.normal(size=(4, 8)), jnp.float32)
    expected = jnp.asarray(reference.apply(variables, x, weights, indices))
    exchanged = GptOssExperts(8, 12, 4, dispatch='exchange')
    with pytest.raises(ValueError, match='expert mesh axis'):
        exchanged.apply(variables, x, weights, indices)
    with jax.set_mesh(build_mesh(MeshSpec(expert=4))):
        actual = jax.jit(exchanged.apply)(variables, x, weights, indices)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.mesh
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16, jnp.float64])
@pytest.mark.parametrize('expert,fsdp', [(2, 4), (4, 2)])
@pytest.mark.parametrize('skewed', [False, True])
def test_biased_exchange_preserves_all_gradients_and_idle_experts(dtype, expert, fsdp, skewed):
    from jax.sharding import NamedSharding, PartitionSpec as P

    with jax.enable_x64(dtype == jnp.float64):
        rng = np.random.default_rng(352)
        x = jnp.asarray(rng.normal(size=(24, 8)), dtype)
        weights = jnp.asarray(rng.uniform(.1, .9, size=(24, 2)), dtype)
        ids = (np.tile([0, 1], (24, 1)) if skewed else
               np.argsort(rng.normal(size=(24, 4)), axis=-1)[:, :2])
        indices = jnp.asarray(ids, jnp.int32)
        model = GptOssExperts(8, 12, 4, dtype=dtype)
        variables = model.init(jax.random.key(0), x, weights, indices)
        master = jnp.float64 if dtype == jnp.float64 else jnp.float32
        variables = jax.tree.map(lambda value: value.astype(master), variables)
        bias = rng.normal(scale=2, size=(4, 24))
        bias[0, :2] = [12, -12]
        variables['params']['gate_up_proj_bias'] = jnp.asarray(bias, master)
        variables['params']['down_proj_bias'] = jnp.asarray(rng.normal(size=(4, 8)), master)
        mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
        specs = {'params': {
            'gate_up_proj': NamedSharding(mesh, P('expert', None, 'fsdp')),
            'gate_up_proj_bias': NamedSharding(mesh, P('expert', 'fsdp')),
            'down_proj': NamedSharding(mesh, P('expert', 'fsdp')),
            'down_proj_bias': NamedSharding(mesh, P('expert'))}}
        tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
        variables = jax.device_put(variables, specs)
        x, weights, indices = (jax.device_put(value, tokens) for value in (x, weights, indices))
        results = []
        for dispatch in ('global', 'exchange'):
            layer = model.clone(dispatch=dispatch)

            def loss(p, x, weights, indices):
                y = jnp.asarray(layer.apply(p, x, weights, indices))
                work = y.astype(jnp.promote_types(y.dtype, jnp.float32))
                return jnp.mean(jnp.sin(work)), y

            with jax.set_mesh(mesh):
                results.append(jax.jit(jax.value_and_grad(loss, (0, 1, 2), has_aux=True),
                    out_shardings=((NamedSharding(mesh, P()), tokens), (specs, tokens, tokens)))(
                        variables, x, weights, indices))
        tolerance = 1e-12 if dtype == jnp.float64 else 3e-5
        for a, b in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
            np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                       atol=tolerance, rtol=tolerance)
        if skewed:
            for gradient in results[1][1][0]['params'].values():
                np.testing.assert_array_equal(gradient[2:], 0)


@pytest.mark.parametrize('compute, cotangents, residue', [
    (jnp.bfloat16, [1., 2**-8, 2**-8, -1., 8.], 2**-7),
    (jnp.float64, [1. + 2**-30, -1., 0., 0., 8.], 2**-30)])
def test_expert_bias_gradients_sum_before_returning_to_master_dtype(compute, cotangents, residue):
    from dew.nn.moe import gather_expert_bias

    with jax.enable_x64():
        bias = jnp.asarray([[.2], [.4]], jnp.float32)
        ids = jnp.asarray([0, 0, 0, 0, 2], jnp.int32)
        cotangent = jnp.asarray(cotangents, compute)[:, None]

        def loss(bias):
            return jnp.sum(gather_expert_bias(bias, ids, compute).astype(jnp.float64) * cotangent)

        gradient = jax.jit(jax.grad(loss))(bias)
        # The final ID is padding, and expert 1 is idle. Neither takes any
        # cotangent. The exact residue rules out per-shard/per-round rounding.
        np.testing.assert_array_equal(gradient, np.asarray([[residue], [0]], np.float32))
        assert gradient.dtype == jnp.float32


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


@pytest.mark.parametrize('skewed', [False, True])
@pytest.mark.parametrize('dispatch', ['global', pytest.param('exchange', marks=pytest.mark.mesh)])
def test_biased_mlp_matches_transformers_outputs_and_every_parameter_gradient(skewed, dispatch):
    # Generated with transformers5.16.1/torch2.14.0+cpu by
    # tools/gpt_oss_reference.py, including saturated gates/up values.
    arrays, parameters, expected_gradients = reference_case(skewed)
    model = GptOssMLP(8, 12, 4, 2, dispatch=dispatch)
    x = jnp.asarray(arrays['hidden'])

    def loss(parameters, x):
        y = jnp.asarray(model.apply({'params': parameters}, x))
        return jnp.mean(jnp.sin(y)), y

    context = (jax.set_mesh(build_mesh(MeshSpec(expert=2, fsdp=4)))
               if dispatch == 'exchange' else contextlib.nullcontext())
    with context:
        (loss_value, output), (gradient, input_gradient) = jax.jit(
            jax.value_and_grad(loss, (0, 1), has_aux=True))(parameters, x)
    np.testing.assert_allclose(output, arrays['output'], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(loss_value, arrays['loss'], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(input_gradient, arrays['input_grad'], atol=3e-5, rtol=3e-5)
    for actual, expected in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_gradients), strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.mesh
def test_decoder_mixture_can_select_biased_expert_exchange():
    from dataclasses import replace
    from dew.nn.backbones.causal_transformer import CausalTransformer, Mixture

    mixture = Mixture(experts=4, top_k=2)
    model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2,
                              mlp_features=12, max_seq_len=4, qk_norm=False,
                              mlp='swigluoai', mixture=mixture)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    variables = model.init(jax.random.key(19), tokens)
    expected = jnp.asarray(model.apply(variables, tokens))
    exchanged = model.clone(mixture=replace(mixture, dispatch='exchange'))
    with jax.set_mesh(build_mesh(MeshSpec(expert=2, fsdp=4))):
        actual = jax.jit(exchanged.apply)(variables, tokens)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)




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


@pytest.mark.mesh
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('skewed', [False, True])
@pytest.mark.parametrize('expert,fsdp', [(2, 4), (4, 2), (4, 1)])
def test_full_biased_router_and_experts_take_identical_pooled_adam_steps(dtype, skewed, expert, fsdp):
    from jax.sharding import NamedSharding, PartitionSpec as P

    arrays, parameters, _ = reference_case(skewed)
    model = GptOssMLP(8, 12, 4, 2, dtype=dtype)
    mesh = build_mesh(MeshSpec(expert=expert, fsdp=fsdp))
    specs = training_shardings(mesh)
    tokens = NamedSharding(mesh, P(('expert', 'fsdp')))
    x = jax.device_put(jnp.asarray(arrays['hidden'].reshape(-1, 8), dtype), tokens)
    parameters = jax.device_put(parameters, specs)
    optimizer = optax.adam(1e-3)
    state = optimizer.init(parameters)
    state_specs = jax.tree.map(lambda value: NamedSharding(mesh, P())
                               if value.ndim == 0 else value.sharding, state)
    state = jax.device_put(state, state_specs)
    outputs = ((NamedSharding(mesh, P()), tokens), (specs, tokens), specs, state_specs)
    baseline = []
    for dispatch in ('global', 'exchange'):
        step = training_step(model.clone(dispatch=dispatch), optimizer)
        with jax.set_mesh(mesh):
            compiled = jax.jit(step, out_shardings=outputs)
            p, s = parameters, state
            for index in range(3):
                result = compiled(p, s, x)
                if dispatch == 'global':
                    baseline.append(result)
                else:
                    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(baseline[index]), strict=True):
                        np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                                   atol=3e-5, rtol=3e-5)
                p, s = result[2:]



@pytest.mark.mesh
def test_biased_router_updates_pool_across_two_real_cpu_processes(tmp_path):
    import subprocess
    import sys
    from test_multiprocess import free_port, report_of, terminate, worker_env

    coordinator = f'127.0.0.1:{free_port()}'
    worker = Path(__file__).with_name('moe_biased_exchange_worker.py')
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


@pytest.mark.mesh
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16, jnp.float64])
@pytest.mark.parametrize('skewed', [False, True])
def test_biased_router_and_bias_parameters_keep_forward_mode(dtype, skewed):
    from jax.sharding import NamedSharding, PartitionSpec as P

    with jax.enable_x64(dtype == jnp.float64):
        arrays, parameters, _ = reference_case(skewed)
        master = jnp.float64 if dtype == jnp.float64 else jnp.float32
        parameters = jax.tree.map(lambda value: value.astype(master), parameters)
        model = GptOssMLP(8, 12, 4, 2, dtype=dtype)
        x = jnp.asarray(arrays['hidden'].reshape(-1, 8), dtype)
        rng = np.random.default_rng(354)
        direction = jax.tree.map(lambda value: jnp.asarray(rng.normal(scale=.02, size=value.shape),
                                                          value.dtype), parameters)
        input_direction = jnp.asarray(rng.normal(scale=.02, size=x.shape), dtype)
        mesh = build_mesh(MeshSpec(expert=4, fsdp=2))
        specs, tokens = training_shardings(mesh), NamedSharding(mesh, P(('expert', 'fsdp')))
        parameters, direction = (jax.device_put(value, specs) for value in (parameters, direction))
        x, input_direction = (jax.device_put(value, tokens) for value in (x, input_direction))
        results = []
        for dispatch in ('global', 'exchange'):
            layer = model.clone(dispatch=dispatch)

            def run(parameters, x, direction, input_direction):
                def forward(parameters, x):
                    return jnp.asarray(layer.apply({'params': parameters}, x))
                return jax.jvp(forward, (parameters, x), (direction, input_direction))

            with jax.set_mesh(mesh):
                results.append(jax.jit(run, out_shardings=(tokens, tokens))(
                    parameters, x, direction, input_direction))
        tolerance = 1e-12 if dtype == jnp.float64 else 3e-5
        for a, b in zip(*results, strict=True):
            np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                       atol=tolerance, rtol=tolerance)

