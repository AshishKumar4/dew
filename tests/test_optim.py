"""The optimizer parameter groups: which parameters Muon steps and which AdamW.

The split is the production Muon recipe (docs/research/frontier-training.md:183):
AdamW keeps the embeddings, the head and the norms, Muon takes the matrices.
Each group's update is asserted against the transform it is supposed to be,
because a parameter in the wrong group still trains, only worse.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.config import OptimConfig
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
from dew.training import distributed
from dew.training.optim import build_optimizer, muon_weight_dimension_numbers

LR = 1e-3


def decoder_params():
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1,
        mlp_features=64, max_seq_len=8, tie_embeddings=False)
    return model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def dit_params():
    model = SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2,
                      mlp_ratio=1)
    return model.init(jax.random.key(0), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))


def muon_solver(**kwargs):
    return build_optimizer(OptimConfig(optimizer='muon', learning_rate=LR, **kwargs),
                           steps_per_epoch=10)


def fixed_gradients(params):
    """A different gradient per parameter, the same one on every call."""
    return jax.tree_util.tree_map_with_path(
        lambda path, x: jax.random.normal(
            jax.random.key(abs(hash(jax.tree_util.keystr(path))) % 2**31), x.shape),
        params)


def group_updates(params, **kwargs):
    """One step of the split solver on the fixed gradients."""
    grads = fixed_gradients(params)
    solver = muon_solver(**kwargs)
    updates, _ = solver.update(grads, solver.init(params), params)
    return updates, grads


def at(tree, path):
    for name in path:
        tree = tree[name]
    return tree


def leaf_paths(tree):
    return {tuple(entry.key for entry in path)
            for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]}


def moment_owners(group_state, params):
    """Parameter paths this optimizer group holds a moment array for.

    A group's masked state carries the parameter tree with `MaskedNode` where
    the group does not apply, so the paths of its real leaves are the group's
    membership as the state itself records it.
    """
    wanted = leaf_paths(params)
    owned = set()
    is_leaf = lambda leaf: isinstance(leaf, optax.MaskedNode)
    for path, leaf in jax.tree_util.tree_flatten_with_path(
            group_state, is_leaf=is_leaf)[0]:
        if is_leaf(leaf):
            continue
        names = []
        for entry in reversed(path):
            if not isinstance(entry, jax.tree_util.DictKey):
                break
            names.append(entry.key)
        candidate = tuple(reversed(names))
        if candidate in wanted:
            owned.add(candidate)
    return owned


@pytest.mark.parametrize("build", [decoder_params, dit_params],
                         ids=["causal_transformer", "simple_dit"])
def test_the_groups_partition_the_parameter_tree(build):
    """Every parameter is stepped by one group and no parameter by two: a
    parameter in both groups would take two updates in one step, and one in
    neither would never move."""
    params = build()
    solver = muon_solver()
    state = solver.init(params)

    groups = {name: moment_owners(group, params)
              for name, group in state.inner_states.items()}
    assert not groups['muon'] & groups['adam']
    assert groups['muon'] | groups['adam'] == leaf_paths(params)
    assert groups['muon'] and groups['adam']


@pytest.mark.parametrize("build", [decoder_params, dit_params],
                         ids=["causal_transformer", "simple_dit"])
def test_every_matrix_the_table_declares_is_a_muon_parameter(build):
    """The coverage the recipe asks for: each parameter of rank two or more
    that is neither a bias nor a lookup table nor the output head carries
    dimension numbers, and those numbers name every one of its axes."""
    params = build()
    spec = muon_weight_dimension_numbers(params)

    for path, param in jax.tree_util.tree_flatten_with_path(params)[0]:
        names = tuple(entry.key for entry in path)
        dimension_numbers = at(spec, names)
        excluded = (param.ndim < 2 or names[-1] in ('bias', 'embedding')
                    or names[-2] in ('lm_head', 'final_proj'))
        if excluded:
            assert dimension_numbers is None, names
            continue
        assert dimension_numbers is not None, names
        axes = (tuple(np.atleast_1d(dimension_numbers.reduction_axis))
                + tuple(np.atleast_1d(dimension_numbers.output_axis)))
        assert sorted(axes) == list(range(param.ndim)), names


def test_a_matrix_of_rank_above_two_with_undeclared_axes_is_rejected():
    """The failure a new module has to hit. Rank two takes Linen's kernel
    convention, but above it the spec would have to guess which axes are the
    matrix, and a wrong guess shows up only as a worse loss curve."""
    params = decoder_params()
    params['params']['layers_0']['mixer'] = {'gate': jnp.zeros((4, 8, 16))}

    with pytest.raises(ValueError, match="mixer.*rank 3.*declared logical axes"):
        muon_weight_dimension_numbers(params)


def test_the_embedding_the_head_and_the_norms_take_the_adamw_update():
    """AdamW's own update for the three kinds the labs keep out of Muon.
    Anything orthogonalized here would carry Newton-Schulz's shape scaling
    instead, which is a factor, not a rounding difference.

    Equality is at fp32, not bitwise: the same transform inside the whole
    tree fuses differently from the same transform on one parameter. Largest
    observed absolute difference 3.5e-10, on updates of 1.5e-3.
    """
    params = decoder_params()
    updates, grads = group_updates(params)
    # optax.adamw decays by 1e-4 unless told otherwise, and this solver was
    # built with no weight decay at all.
    reference = optax.adamw(LR, nesterov=True, weight_decay=0.0)

    for path in [('params', 'embed_tokens', 'embedding'),
                 ('params', 'lm_head', 'kernel'),
                 ('params', 'norm', 'scale'),
                 ('params', 'layers_0', 'self_attn', 'q_norm', 'scale')]:
        param, grad = at(params, path), at(grads, path)
        expected, _ = reference.update(grad, reference.init(param), param)
        np.testing.assert_allclose(np.asarray(at(updates, path)),
                                   np.asarray(expected), atol=1e-8)


def test_the_projections_and_the_mlp_take_the_muon_update():
    """The other side of the split, against Muon run on that matrix alone.
    Largest observed absolute difference 8.4e-10, on updates of 5e-4."""
    params = decoder_params()
    updates, grads = group_updates(params)
    reference = optax.contrib.muon(LR)

    for path in [('params', 'layers_0', 'self_attn', 'q_proj', 'kernel'),
                 ('params', 'layers_0', 'self_attn', 'o_proj', 'kernel'),
                 ('params', 'layers_1', 'mlp', 'down_proj', 'kernel')]:
        param, grad = at(params, path), at(grads, path)
        expected, _ = reference.update(grad, reference.init(param), param)
        np.testing.assert_allclose(np.asarray(at(updates, path)),
                                   np.asarray(expected), atol=1e-8)


def test_a_head_expanded_projection_orthogonalizes_its_flattened_head_side():
    """The axes the recipe cares about. A DiT query kernel is
    [embed, heads, head_dim] and a decoder's is [embed, heads * head_dim], so
    the update has to be the same either way: the head dimensions are one side
    of the matrix, not a batch of matrices.

    Newton-Schulz reduces over the same elements in a different order once the
    head axes are separate, so this is equality at fp32 and not bitwise. The
    largest observed absolute difference is 5.3e-10, on updates of order
    2.5e-4.
    """
    params = dit_params()
    updates, grads = group_updates(params)
    reference = optax.contrib.muon(LR)

    query = ('params', 'dit_block_0', 'attention', 'to_q', 'kernel')
    grad = at(grads, query)
    flat = {'kernel': grad.reshape(grad.shape[0], -1)}
    expected, _ = reference.update(flat, reference.init(flat), flat)
    np.testing.assert_allclose(
        np.asarray(at(updates, query)).reshape(flat['kernel'].shape),
        np.asarray(expected['kernel']), atol=1e-8)

    out = ('params', 'dit_block_0', 'attention', 'to_out_0', 'kernel')
    out_grad = at(grads, out)
    out_flat = {'kernel': out_grad.reshape(-1, out_grad.shape[-1])}
    expected_out, _ = reference.update(out_flat, reference.init(out_flat), out_flat)
    np.testing.assert_allclose(
        np.asarray(at(updates, out)).reshape(out_flat['kernel'].shape),
        np.asarray(expected_out['kernel']), atol=1e-8)


def test_weight_decay_reaches_the_norm_scales():
    """Decay on the norm scale is the piece of the recipe that lives in the
    AdamW group (docs/research/frontier-training.md:184), so the config's
    decay has to reach that group and not only Muon's."""
    params = decoder_params()
    decayed, _ = group_updates(params, weight_decay=0.1)
    plain, _ = group_updates(params)

    path = ('params', 'norm', 'scale')
    difference = np.asarray(at(decayed, path)) - np.asarray(at(plain, path))
    np.testing.assert_allclose(difference, -LR * 0.1 * np.asarray(at(params, path)),
                               rtol=1e-5)


def test_both_groups_step_with_the_one_schedule():
    """One schedule multiplies both groups. Neither group's state depends on
    the learning rate, so a scheduled run is the unscaled run times the
    schedule, for a Muon parameter and an AdamW one alike. A second constant
    on either group would break that on the group that kept it.
    """
    params = decoder_params()
    grads = fixed_gradients(params)
    schedule = dict(learning_rate=1e-4, learning_rate_peak=4e-3,
                    learning_rate_end=1e-3, learning_rate_schedule='cosine',
                    learning_rate_warmup_steps=1, learning_rate_decay_epochs=1)
    scheduled = build_optimizer(OptimConfig(optimizer='muon', **schedule),
                                steps_per_epoch=4)
    unscaled = build_optimizer(OptimConfig(optimizer='muon', learning_rate=1.0),
                               steps_per_epoch=4)
    rate = optax.warmup_cosine_decay_schedule(
        init_value=schedule['learning_rate'], peak_value=schedule['learning_rate_peak'],
        warmup_steps=schedule['learning_rate_warmup_steps'], decay_steps=4,
        end_value=schedule['learning_rate_end'])

    scheduled_state, unscaled_state = scheduled.init(params), unscaled.init(params)
    for step in range(5):
        scheduled_updates, scheduled_state = scheduled.update(
            grads, scheduled_state, params)
        unscaled_updates, unscaled_state = unscaled.update(
            grads, unscaled_state, params)
        for path in [('params', 'embed_tokens', 'embedding'),
                     ('params', 'layers_0', 'mlp', 'up_proj', 'kernel')]:
            np.testing.assert_allclose(
                np.asarray(at(scheduled_updates, path)),
                float(rate(step)) * np.asarray(at(unscaled_updates, path)),
                rtol=2e-4, atol=1e-12, err_msg=f"step {step} {path}")


def test_an_expert_stack_is_orthogonalized_one_expert_at_a_time(monkeypatch):
    """A routed expert kernel is [experts, embed, mlp], one matrix per expert
    stacked on the leading dimension (wave/moe's declaration). Muon has to
    treat that dimension as a batch and orthogonalize each expert on its own,
    so the update equals the updates of the single matrices stacked back up.
    Contracting the expert dimension instead would mix the experts, and the
    loss curve is the only place it would show.
    """
    monkeypatch.setitem(distributed.DEFAULT_LOGICAL_PARAM_AXES,
                        ("experts", "gate_proj"), ("exp", "embed", "mlp"))
    experts, embed, mlp = 3, 8, 16
    params = {'params': {'layers_0': {'mlp': {'experts': {'gate_proj': {
        'kernel': jnp.zeros((experts, embed, mlp))}}}}}}
    path = ('params', 'layers_0', 'mlp', 'experts', 'gate_proj', 'kernel')
    spec = at(muon_weight_dimension_numbers(params), path)
    assert spec.reduction_axis == (1,) and spec.output_axis == (2,)

    updates, grads = group_updates(params)
    grad = at(grads, path)
    reference = optax.contrib.muon(LR)
    for expert in range(experts):
        one = {'kernel': grad[expert]}
        expected, _ = reference.update(one, reference.init(one), one)
        np.testing.assert_allclose(np.asarray(at(updates, path)[expert]),
                                   np.asarray(expected['kernel']), atol=1e-8)


def test_the_router_gate_takes_the_adamw_update(monkeypatch):
    """A router gate is declared ('embed', 'exp'): one column per expert, so
    its output side counts choices rather than features, and the labs keep
    the router on AdamW along with the embeddings and the head. The same axis
    name leads the expert kernels, where it stacks matrices, so position is
    what tells the two apart."""
    monkeypatch.setitem(distributed.DEFAULT_LOGICAL_PARAM_AXES,
                        ("gate",), ("embed", "exp"))
    params = {'params': {'layers_0': {'mlp': {'gate': {
        'kernel': jnp.zeros((8, 4))}}}}}
    path = ('params', 'layers_0', 'mlp', 'gate', 'kernel')
    assert at(muon_weight_dimension_numbers(params), path) is None

    updates, grads = group_updates(params)
    reference = optax.adamw(LR, nesterov=True, weight_decay=0.0)
    param, grad = at(params, path), at(grads, path)
    expected, _ = reference.update(grad, reference.init(param), param)
    np.testing.assert_allclose(np.asarray(at(updates, path)),
                               np.asarray(expected), atol=1e-8)
