"""The optimizer parameter groups: which parameters Muon steps and which AdamW.

The split is the production Muon recipe (docs/research/frontier-training.md:183):
AdamW keeps the embeddings, the head and the norms, Muon takes the matrices.
Each group's update is asserted against the transform it is supposed to be,
because a parameter in the wrong group still trains, only worse.
"""
import jax
from dew.objectives.base import scalar_loss
import jax.numpy as jnp
import json
import numpy as np
import optax
import pytest
from pathlib import Path

from dew.config import OptimConfig
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
import dew.nn.moe  # declares the expert and gate axes the two MoE cases read
from dew.training.optim import (
    build_optimizer, muon_weight_dimension_numbers, scale_by_qk_clip)
from tools.muonclip_reference import clip_qk_kernel, clip_scale

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
                           steps=10)


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
    decay has to reach that group as well as Muon's."""
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
                    learning_rate_warmup_steps=1, learning_rate_decay_steps=4)
    scheduled = build_optimizer(OptimConfig(optimizer='muon', **schedule),
                                steps=4)
    unscaled = build_optimizer(OptimConfig(optimizer='muon', learning_rate=1.0),
                               steps=4)
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


def test_an_expert_stack_is_orthogonalized_one_expert_at_a_time():
    """A routed expert kernel is [experts, embed, mlp], one matrix per expert
    stacked on the leading dimension (SparseMLP's declaration). Muon has to
    treat that dimension as a batch and orthogonalize each expert on its own,
    so the update equals the updates of the single matrices stacked back up.
    Contracting the expert dimension instead would mix the experts, and the
    loss curve is the only place it would show.
    """
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


def test_the_router_gate_takes_the_adamw_update():
    """A router gate is declared ('embed', 'exp'): one column per expert, so
    its output side counts choices, not features, and the labs keep the
    router on AdamW along with the embeddings and the head. The same axis
    name leads the expert kernels, where it stacks matrices, so position is
    what tells the two apart."""
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


# --------------------------------------------------------------------------
# MuonClip: Muon plus the QK-Clip (Kimi K2, arXiv 2507.20534)
# --------------------------------------------------------------------------

from tools.muonclip_reference import clip_qk_kernel, clip_scale

QK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "muonclip" / "maxtext_mla.json"


def muonclip_solver(**kwargs):
    return build_optimizer(
        OptimConfig(optimizer='muonclip', learning_rate=LR, **kwargs), steps=10)


def qk_tree(layers=1, heads=4, kv_heads=4, dim=8, seed=0):
    """A trainable tree of query/key kernels and its fixed gradients."""
    rng = np.random.default_rng(seed)
    params, grads = {}, {}
    for layer in range(layers):
        params[f'layers_{layer}'] = {
            'self_attn': {
                'q_proj': {'kernel': jnp.asarray(
                    rng.normal(0, 0.5, (16, heads * dim)), jnp.float32)},
                'k_proj': {'kernel': jnp.asarray(
                    rng.normal(0, 0.5, (16, kv_heads * dim)), jnp.float32)}}}
        grads[f'layers_{layer}'] = {
            'self_attn': {
                'q_proj': {'kernel': jnp.asarray(
                    rng.normal(0, 0.01, (16, heads * dim)), jnp.float32)},
                'k_proj': {'kernel': jnp.asarray(
                    rng.normal(0, 0.01, (16, kv_heads * dim)), jnp.float32)}}}
    return params, grads


def qk_stats(layers=1, heads=4, rows=3, values=None, nope=None):
    """The `qk` collection with per-layer maxima, as the model sows it."""
    stats = {}
    for layer in range(layers):
        sown = {'max_logits': (jnp.asarray(
            np.full((rows, heads), 10.0, np.float32) if values is None
            else np.asarray(values, np.float32).reshape(rows, heads), jnp.float32),)}
        if nope is not None:
            sown['qk_nope'] = jnp.asarray(nope)
        stats[f'layers_{layer}'] = {'self_attn': sown}
    return stats


def largest_update_difference(left, right) -> float:
    return max(float(jnp.max(jnp.abs(a - b)))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def test_muonclip_without_stats_steps_like_muon():
    """No maxima, no clip: the transform steps aside bitwise, leaving every
    run whose loss never opened the collection on Muon's update."""
    params, grads = qk_tree()
    muon = muon_solver()
    clipped = muonclip_solver()
    expected, _ = muon.update(grads, muon.init(params), params)
    updates, _ = clipped.update(grads, clipped.init(params), params)
    assert largest_update_difference(updates, expected) == 0.0


def test_a_quiet_clip_steps_like_muon():
    """Maxima below the threshold rescale nothing: bitwise the Muon update."""
    params, grads = qk_tree()
    muon = muon_solver()
    clipped = muonclip_solver()
    expected, _ = muon.update(grads, muon.init(params), params)
    updates, _ = clipped.update(
        grads, clipped.init(params), params,
        qk_stats=qk_stats(values=[[10.0] * 4] * 3))
    assert largest_update_difference(updates, expected) == 0.0


def test_the_clip_matches_the_paper_equations():
    """Per-head query and key rescale against the paper's equations in
    tools/muonclip_reference.py: one head fires at half, the rest hold.
    Observed on CPU: 6.0e-08."""
    params, grads = qk_tree()
    stats = qk_stats(values=[[200.0, 50.0, 10.0, 5.0],
                             [10.0, 10.0, 10.0, 10.0],
                             [10.0, 10.0, 10.0, 10.0]])
    tx = scale_by_qk_clip(100.0)
    updates, _ = tx.update(grads, tx.init(params), params, qk_stats=stats)
    s_max = np.max(np.asarray(stats['layers_0']['self_attn']['max_logits'][0]),
                   axis=0)
    gamma = clip_scale(s_max, 100.0)
    for proj in ('q_proj', 'k_proj'):
        stepped = (np.asarray(params['layers_0']['self_attn'][proj]['kernel'])
                   + np.asarray(grads['layers_0']['self_attn'][proj]['kernel']))
        expected = clip_qk_kernel(stepped, np.asarray(gamma), 4)
        applied = (np.asarray(params['layers_0']['self_attn'][proj]['kernel'])
                   + np.asarray(updates['layers_0']['self_attn'][proj]['kernel']))
        assert float(np.max(np.abs(applied - expected))) < 1e-6


def test_grouped_keys_clip_by_the_strongest_head():
    """Two key heads behind four query heads: one firing query head rescales
    the whole key projection by its gamma, the conservative side, while the
    quiet query heads keep Muon's update bitwise. Observed on CPU: key at
    half, quiet query slices bitwise."""
    params, grads = qk_tree(kv_heads=2)
    stats = qk_stats(values=[[200.0, 10.0, 10.0, 10.0],
                             [10.0, 10.0, 10.0, 10.0],
                             [10.0, 10.0, 10.0, 10.0]])
    tx = scale_by_qk_clip(100.0)
    updates, _ = tx.update(grads, tx.init(params), params, qk_stats=stats)
    key, key_update = (params['layers_0']['self_attn']['k_proj']['kernel'],
                       updates['layers_0']['self_attn']['k_proj']['kernel'])
    grad_update = grads['layers_0']['self_attn']['k_proj']['kernel']
    assert float(jnp.max(jnp.abs(
        (np.asarray(key) + np.asarray(key_update))
        - 0.5 * (np.asarray(key) + np.asarray(grad_update))))) < 1e-6
    query_update = updates['layers_0']['self_attn']['q_proj']['kernel']
    plain = grads['layers_0']['self_attn']['q_proj']['kernel']
    assert largest_update_difference(
        query_update.reshape(16, 4, 8)[:, 1:, :],
        plain.reshape(16, 4, 8)[:, 1:, :]) == 0.0


def test_the_latent_branches_match_maxtext():
    """The MLA branches against MaxText 0.2.4's own outputs, committed as
    tests/fixtures/muonclip/maxtext_mla.json: `q_proj` stands in for `wq_b`
    and `kv_b_proj` for `wkv_b`, the same per-head tensors under Dew's
    names. Observed on CPU: at most 4.7e-09."""
    fixture = json.loads(QK_FIXTURE.read_text())
    assert fixture['maxtext'] == '0.2.4'
    worst = 0.0
    tx = scale_by_qk_clip(100.0)
    for case in fixture['cases']:
        proj = 'q_proj' if case['layer'] == 'wq_b' else 'kv_b_proj'
        flat = np.asarray(case['param']).reshape(
            len(case['param']), -1)
        params = {'l': {'m': {proj: {'kernel': jnp.asarray(flat)}}}}
        grads = jax.tree.map(jnp.zeros_like, params)
        stats = {'l': {'m': {
            'max_logits': (jnp.asarray(case['max_logits'], jnp.float32),),
            'qk_nope': jnp.asarray(case['qk_nope'])}}}
        updates, _ = tx.update(grads, tx.init(params), params, qk_stats=stats)
        applied = (flat + np.asarray(updates['l']['m'][proj]['kernel'])
                   ).reshape(np.asarray(case['expected']).shape)
        worst = max(worst, float(np.max(np.abs(applied - case['expected']))))
    assert worst < 1e-6, worst


def test_nonpositive_maxima_hold_their_weights():
    """A head whose logits never rose above zero clips nothing: the raw
    transform is the identity there, bitwise. MaxText's formula would hand
    such a head a negative rescale, so Dew holds it at 1.0 instead."""
    params, grads = qk_tree()
    tx = scale_by_qk_clip(100.0)
    updates, _ = tx.update(
        grads, tx.init(params), params,
        qk_stats=qk_stats(values=[[-3.0, 0.0, -0.5, -100.0]] * 3))
    assert largest_update_difference(updates, grads) == 0.0


def test_an_unsown_projection_is_refused():
    """A named projection whose layer sowed nothing raises naming the layer,
    and a non-positive threshold raises too: both would train a different
    model than the maxima describe."""
    params, grads = qk_tree()
    tx = scale_by_qk_clip(100.0)
    with pytest.raises(ValueError, match="sowed no max logits"):
        tx.update(grads, tx.init(params), params, qk_stats={})
    with pytest.raises(ValueError, match="bounds positive logits"):
        scale_by_qk_clip(0.0)


def test_the_threshold_rides_optimizer_opts():
    """`--optim.optimizer-opts '{"qk_clip_threshold": 5.0}'` reaches the
    transform: maxima of 50 clip nothing at the default 100, and rescale
    each side by sqrt(5/50) at 5, so their product carries the tenth."""
    params, grads = qk_tree()
    stats = qk_stats(values=[[50.0] * 4] * 3)
    default = muonclip_solver()
    muon = muon_solver()
    plain, _ = muon.update(grads, muon.init(params), params)
    updates, _ = default.update(grads, default.init(params), params,
                                qk_stats=stats)
    assert largest_update_difference(updates, plain) == 0.0
    low = build_optimizer(OptimConfig(
        optimizer='muonclip', learning_rate=LR,
        optimizer_opts={'qk_clip_threshold': 5.0}), steps=10)
    clipped, _ = low.update(grads, low.init(params), params, qk_stats=stats)
    factor = float(np.sqrt(5.0 / 50.0))
    stepped = (np.asarray(params['layers_0']['self_attn']['q_proj']['kernel'])
               + np.asarray(plain['layers_0']['self_attn']['q_proj']['kernel']))
    assert float(np.max(np.abs(
        np.asarray(params['layers_0']['self_attn']['q_proj']['kernel'])
        + np.asarray(clipped['layers_0']['self_attn']['q_proj']['kernel'])
        - factor * stepped))) < 1e-6


def tiny_decoder(**overrides):
    fields = dict(vocab_size=32, emb_features=16, num_layers=2, num_heads=4,
                  num_kv_heads=2, mlp_features=32, max_seq_len=8,
                  qk_norm=False)
    fields.update(overrides)
    return CausalTransformer(**fields)


def test_the_sow_reports_the_kernels_logits():
    """The `qk` collection holds one fp32 `[rows, heads]` maximum per layer,
    finite and tracking the query kernel: doubling layer 0's queries doubles
    its maxima, which a stale or miswired sow would not. A closed collection
    sows nothing, and the plain forward skips the extra matmul."""
    model = tiny_decoder()
    variables = model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], jnp.int32)
    _, sown = model.apply(variables, ids, mutable=["qk"])
    assert set(sown.get("qk", {})) == {"layers_0", "layers_1"}
    for layer in ("layers_0", "layers_1"):
        (max_logits,) = sown["qk"][layer]["self_attn"]["max_logits"]
        assert max_logits.shape == (1, 4) and max_logits.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(max_logits)))
        assert bool(jnp.all(max_logits != 0.0))

    def double_layer_zero(path, leaf):
        names = tuple(entry.key for entry in path
                      if isinstance(entry, jax.tree_util.DictKey))
        if (len(names) >= 3 and names[-3:] == (
                'self_attn', 'q_proj', 'kernel')
                and 'layers_0' in names):
            return leaf * 2
        return leaf

    doubled = jax.tree_util.tree_map_with_path(double_layer_zero, variables)
    _, sown_doubled = model.apply(doubled, ids, mutable=["qk"])
    before = sown["qk"]["layers_0"]["self_attn"]["max_logits"][0]
    after = sown_doubled["qk"]["layers_0"]["self_attn"]["max_logits"][0]
    assert float(jnp.max(jnp.abs(after / before - 2.0))) < 1e-5
    # Layer 1 reads layer 0's output, so its maxima legitimately move; what
    # has to hold there is presence and finiteness, not equality.
    (downstream,) = sown_doubled["qk"]["layers_1"]["self_attn"]["max_logits"]
    assert downstream.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(downstream)))

    _, shut = model.apply(variables, ids, mutable=[])
    assert shut == {}


def test_muonclip_moves_a_real_step():
    """Three muonclip steps on the tiny decoder, stats from its own forward:
    finite losses, and different weights than Muon at the same seed, which is
    what proves the clip fired on the real tree. Observed on CPU: maxima peak
    at 4.49 against the threshold of 1.0, losses finite around 3.9, weights
    0.56 from Muon's."""
    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective
    model = tiny_decoder()
    variables = model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    rows = np.random.default_rng(0).integers(0, 32, size=(4, 9)).astype(np.int32)
    batch = {"text": rows}
    inputs = jnp.asarray(rows[:, :-1])
    objective = LMObjective(model, 8)
    info = Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(3), ema=None)

    def run(solver, stats):
        params, opt_state = variables["params"], solver.init(variables["params"])
        losses = []
        for _ in range(3):
            (loss, _), grads = jax.value_and_grad(
                lambda p: scalar_loss(objective, {**variables, "params": p}, batch, info),
                has_aux=True)(params)
            updates, opt_state = solver.update(
                grads, opt_state, params, **stats)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
        return params, losses

    muon = muon_solver()
    clipped = build_optimizer(OptimConfig(
        optimizer='muonclip', learning_rate=LR,
        optimizer_opts={'qk_clip_threshold': 1.0}), steps=10)
    _, plain_losses = run(muon, {})
    _, sown = model.apply(variables, inputs, mutable=["qk"])
    stats = {"qk_stats": sown.get("qk")}
    params, losses = run(clipped, stats)
    assert all(np.isfinite(losses)), losses
    assert all(np.isfinite(plain_losses)), plain_losses
    muon_params, _ = run(muon, {})
    assert largest_update_difference(params, muon_params) > 1e-6
