"""The optimizer parameter groups: which parameters Muon steps and which AdamW.

The split is the production Muon recipe (docs/research/frontier-training.md:183):
AdamW keeps the embeddings, the head and the norms, Muon takes the matrices.
Each group's update is asserted against the transform it is supposed to be,
because a parameter in the wrong group still trains, only worse.
"""
import dataclasses
import importlib
import json
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal
from jax.sharding import NamedSharding, PartitionSpec as P
from learning_gate import assert_learns, within
from model_support import largest_difference
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, equations, without_ynn

from dew.config import OptimConfig
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.dit import SimpleDiT
from dew.optim import (
    OPTIMIZER_MAP,
    Cosine,
    Exponential,
    Linear,
    OneCycle,
    Optimizer,
    ParamGroup,
    Power,
    PowerTail,
    ScheduleBase,
    learning_rate,
    muon_weight_dimension_numbers,
    rates_of,
    scale_by_qk_clip,
    stepped,
    weight_decay_mask,
)
from dew.training import MeshSpec
from tools.muonclip_reference import clip_mla_reference, clip_qk_kernel, clip_scale

LR = 1e-3


@pytest.mark.parametrize("warmup, updates", [(3, 3), (5, 3)])
def test_cosine_names_the_warmup_and_decay_horizon_it_cannot_fit(warmup, updates):
    with pytest.raises(ValueError):
        Cosine(peak_value=1e-3, warmup_steps=warmup).schedule(updates)


def test_cosine_uses_its_explicit_horizon_when_the_run_has_no_step_cap():
    schedule = Cosine(peak_value=1.0, warmup_steps=2, decay_steps=4).schedule(None)
    np.testing.assert_array_equal([float(schedule(i)) for i in (0, 2, 4)], [0, 1, 0])
    with pytest.raises(ValueError):
        Cosine(peak_value=1.0, warmup_steps=2).schedule(None)


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
    return OptimConfig(optimizer='muon', learning_rate=LR, **kwargs).build(10)


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


UPDATE_SEEDS = 8
"""Check 1's declared seeds for a group's update against optax's transform on the parameter alone
(`seeds_for`)."""


def drawn_tree(params, seed):
    """`params`'s tree drawn afresh for a seed, its parameters and a gradient of each, both standard
    normal scaled as the decoder's own init and its gradients run (a gradient a hundredth)."""
    keys = iter(jax.random.split(jax.random.key(seed), 2 * len(jax.tree.leaves(params))))

    def normal(scale):
        return lambda leaf: np.asarray(scale * jax.random.normal(next(keys), leaf.shape, leaf.dtype))

    return jax.tree.map(normal(1.0), params), jax.tree.map(normal(1e-2), params)


def assert_updates_as(solver, reference, params, paths, label, *, flatten=None, seeds=UPDATE_SEEDS):
    """Check 1: `solver`'s update of drawn parameters under drawn gradients, at each of `paths`, on par with
    `reference`'s update of each parameter alone, both from `reference`'s float64 run. `flatten` reshapes a
    path's gradient to the matrix the reference steps, and the update back."""
    shape = flatten or (lambda path, value: value)

    def dew(params, grads):
        updates, _ = solver.update(grads, solver.init(params), params)
        return [shape(path, at(updates, path)) for path in paths]

    def alone(params, grads):
        steps = []
        for path in paths:
            grad, param = shape(path, at(grads, path)), shape(path, at(params, path))
            steps.append(reference.update(grad, reference.init(param), param)[0])
        return steps

    assert_as_exact(without_ynn(dew), without_ynn(alone), lambda seed: drawn_tree(params, seed), label,
                    seeds=seeds, truth_fn=alone)


def test_the_embedding_the_head_and_the_norms_take_the_adamw_update():
    """AdamW's own update for the three kinds the labs keep out of Muon, on
    par with optax's AdamW on each parameter alone (check 1). Anything
    orthogonalized here would carry Newton-Schulz's shape scaling instead,
    which is a factor, not a rounding difference."""
    # optax.adamw decays by 1e-4 unless told otherwise, and this solver was
    # built with no weight decay at all.
    assert_updates_as(muon_solver(), optax.adamw(LR, nesterov=True, weight_decay=0.0), decoder_params(),
                      [('params', 'embed_tokens', 'embedding'), ('params', 'lm_head', 'kernel'),
                       ('params', 'norm', 'scale'), ('params', 'layers_0', 'self_attn', 'q_norm', 'scale')],
                      "the AdamW group")


def test_the_projections_and_the_mlp_take_the_muon_update():
    """The other side of the split, on par with Muon run on that matrix alone (check 1)."""
    assert_updates_as(muon_solver(), optax.contrib.muon(LR), decoder_params(),
                      [('params', 'layers_0', 'self_attn', 'q_proj', 'kernel'),
                       ('params', 'layers_0', 'self_attn', 'o_proj', 'kernel'),
                       ('params', 'layers_1', 'mlp', 'down_proj', 'kernel')], "the Muon group")


def test_a_head_expanded_projection_orthogonalizes_its_flattened_head_side():
    """The axes the recipe cares about. A DiT query kernel is
    [embed, heads, head_dim] and a decoder's is [embed, heads * head_dim], so
    the update has to be the same either way: the head dimensions are one side
    of the matrix, not a batch of matrices. Over drawn kernels and gradients,
    each update on par with optax's Muon of the flattened matrix (check 1)."""
    query = ('params', 'dit_block_0', 'attention', 'to_q', 'kernel')
    out = ('params', 'dit_block_0', 'attention', 'to_out_0', 'kernel')

    def flattened(path, value):
        # The query's heads are its output side, the output projection's its input side.
        return value.reshape(value.shape[0], -1) if path == query else value.reshape(-1, value.shape[-1])

    assert_updates_as(muon_solver(), optax.contrib.muon(LR), dit_params(), [query, out],
                      "the head-expanded projections", flatten=flattened)



def test_a_schedule_that_marks_no_rate_is_its_own_shape():
    """A ScheduleBase subclass implementing `values(steps)` alone, with fields of its own and
    none marked a rate, builds under OptimConfig and steps at the values it gives: it holds no rate
    fields as operands (`rates`), so its values are constants of its program."""

    @dataclasses.dataclass(frozen=True)
    class Halving(ScheduleBase):
        start: float = 0.4

        def values(self, steps):
            return lambda count: self.start * 0.5 ** count

    assert Halving().rates() == {}
    solver = OptimConfig(optimizer="adam", learning_rate=Halving(), clip_grads=0).build(4)
    params = {"w": jnp.ones(3)}
    state = solver.init(params)
    for step in range(3):
        _, state = solver.update({"w": jnp.ones(3)}, state, params)
        # A power of two times the float32 start: exact.
        np.testing.assert_array_equal(learning_rate(state), np.float32(0.4) * np.float32(0.5) ** step)

def test_weight_decay_reaches_the_norm_scales():
    """Decay on the norm scale is the piece of the recipe that lives in the
    AdamW group (docs/research/frontier-training.md:184), so the config's
    decay has to reach that group as well as Muon's."""
    assert_updates_as(muon_solver(weight_decay=0.1), optax.adamw(LR, nesterov=True, weight_decay=0.1),
                      decoder_params(), [('params', 'norm', 'scale')], "the norm scale's decay")


@pytest.mark.parametrize('optimizer, state_dtype', [('adam', 'float32'), ('adam', 'bfloat16'),
                                                    ('adamw', 'float32'), ('adamw', 'bfloat16'),
                                                    ('lamb', 'float32')])
def test_weight_decay_spares_the_biases_and_the_norms_unless_a_group_claims_them(optimizer, state_dtype):
    """On a zero gradient an update is the decay alone: every optimizer but Muon (above) moves a DiT's
    kernels as one all-parameter group does, and spares the norm and biases (a per-head one too)."""
    params = fixed_gradients(dit_params()['params'])

    def decay_step(**groups):
        solver = OptimConfig(optimizer=optimizer, learning_rate=LR, weight_decay=0.1,
                             state_dtype=state_dtype, **groups).build(1)
        return solver.update(jax.tree.map(jnp.zeros_like, params), solver.init(params), params)[0]

    spared, grouped = decay_step(), decay_step(param_groups=(ParamGroup('all', ('*',)),))
    for path in [('output', 'final_norm', 'scale'), ('output', 'final_norm', 'bias'),
                 ('dit_block_0', 'attention', 'to_q', 'bias'), ('dit_block_0', 'mlp', 'up_proj', 'bias')]:
        assert not np.any(at(spared, path)) and np.all(at(grouped, path)), path
    for path in [('dit_block_0', 'attention', 'to_q', 'kernel'), ('embed', 'patch_embed', 'Conv_0', 'kernel'),
                 ('output', 'final_proj', 'kernel')]:
        np.testing.assert_array_equal(at(spared, path), at(grouped, path), err_msg=str(path))


def test_both_groups_step_with_the_one_schedule():
    """One schedule multiplies both groups. Neither group's state depends on
    the learning rate, so a scheduled run is the unscaled run times the
    schedule, for a Muon parameter and an AdamW one alike. A second constant
    on either group would break that on the group that kept it. Over drawn
    parameters and gradients, each of five scheduled updates on par with the
    unscaled run's times the rate the scheduled run reports at its step
    (check 1), both from the unscaled run's float64 run times optax's warmup
    cosine in float64. The rate's own rounding is the schedules' tests'."""
    cosine = Cosine(init_value=1e-4, peak_value=4e-3, end_value=1e-3, warmup_steps=1, decay_steps=4)
    scheduled = OptimConfig(optimizer='muon', learning_rate=cosine).build(4)
    unscaled = OptimConfig(optimizer='muon', learning_rate=1.0).build(4)
    rate = optax.warmup_cosine_decay_schedule(
        init_value=cosine.init_value, peak_value=cosine.peak_value,
        warmup_steps=cosine.warmup_steps, decay_steps=4, end_value=cosine.end_value)
    paths = [('params', 'embed_tokens', 'embedding'), ('params', 'layers_0', 'mlp', 'up_proj', 'kernel')]
    params = decoder_params()
    state, reported = scheduled.init(params), []
    for _ in range(5):
        _, state = scheduled.update(jax.tree.map(jnp.zeros_like, params), state, params)
        reported.append(np.float32(learning_rate(state)))

    def steps(solver, scale):
        def walked(params, grads):
            state, taken = solver.init(params), []
            for step in range(5):
                updates, state = solver.update(grads, state, params)
                taken += [scale(step) * at(updates, path) for path in paths]
            return taken
        return walked

    assert_as_exact(without_ynn(steps(scheduled, lambda step: 1.0)),
                    without_ynn(steps(unscaled, lambda step: reported[step])),
                    lambda seed: drawn_tree(decoder_params(), seed), "one schedule over both groups",
                    seeds=UPDATE_SEEDS,
                    truth_fn=jax.jit(steps(unscaled, lambda step: rate(jnp.asarray(step, jnp.float64)))))


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

    assert_updates_as(muon_solver(), ExpertByExpert(optax.contrib.muon(LR)), params, [path],
                      "an expert stack, one expert at a time")


class ExpertByExpert(typing.NamedTuple):
    """`transform` stepping each matrix of an expert stack, its leading axis, alone: the reference of a
    stack Muon steps one expert at a time."""

    transform: optax.GradientTransformation

    def init(self, params):
        return [self.transform.init(one) for one in params]

    def update(self, grads, state, params):
        steps = [self.transform.update(grad, one, param)[0]
                 for grad, one, param in zip(grads, state, params, strict=True)]
        return jnp.stack(steps), state


@pytest.mark.mesh(devices=4)
@pytest.mark.parametrize("spec", [{"expert": 4}, {"expert": 2, "fsdp": 2}], ids=["expert", "expert-fsdp"])
def test_an_expert_stack_placed_over_the_mesh_is_orthogonalized_one_expert_at_a_time(spec):
    """The contract above with the stack placed as a run places it, its
    experts over the expert axis and each matrix split over fsdp, and the
    step compiled there. The experts' gradients are one matrix at four
    scales, so each normalized alone takes optax's update for that matrix
    alone, where normalizing the stack as one would share a scale among them."""
    experts, embed, mlp = 4, 8, 16
    mesh = MeshSpec(**spec).build(jax.devices()[:4])
    placement = NamedSharding(mesh, P("expert", "fsdp" if "fsdp" in spec else None, None))
    scales = np.asarray([1.0, 16.0, 0.25, 2.0], np.float32)[:, None, None]

    def draw(seed):
        grad = np.asarray(jax.random.normal(jax.random.key(seed), (embed, mlp)))
        return (np.asarray(jax.random.normal(jax.random.key(seed + (1 << 20)), (experts, embed, mlp))),
                scales * grad)

    solver = muon_solver()

    def placed(params, grads):
        def stacked(value):
            return {'params': {'layers_0': {'mlp': {'experts': {'gate_proj': {
                'kernel': jax.device_put(value, placement)}}}}}}

        with jax.set_mesh(mesh):
            updates, _ = jax.jit(solver.update)(stacked(grads), solver.init(stacked(params)), stacked(params))
        return np.asarray(updates['params']['layers_0']['mlp']['experts']['gate_proj']['kernel'])

    alone = ExpertByExpert(optax.contrib.muon(LR))

    def reference(params, grads):
        return alone.update(grads, alone.init(params), params)[0]

    assert_as_exact(placed, without_ynn(reference), draw, f"an expert stack over {spec}", seeds=UPDATE_SEEDS,
                    truth_fn=reference)


def test_a_matrix_declared_with_no_axis_named_is_orthogonalized_as_one_matrix():
    """Gemma 3n's AltUp declares its coefficient matrices (None, None), whole on every device: that
    says how they shard, not which axes form the matrix, so Muon steps the kernel as optax steps one
    undeclared matrix, contracting its rows into its columns."""
    # Gemma 3n's modules declare AltUp's axes as they are defined.
    importlib.import_module("dew.nn.gemma3n")

    params = {'params': {'layers_0': {'altup': {'prediction_coefs': {'kernel': jnp.zeros((16, 4))}}}}}
    path = ('params', 'layers_0', 'altup', 'prediction_coefs', 'kernel')
    updates, grads = group_updates(params)
    reference = optax.contrib.muon(LR)
    one = {'kernel': at(grads, path)}
    expected, _ = reference.update(one, reference.init(one), one)
    np.testing.assert_array_equal(np.asarray(at(updates, path)), np.asarray(expected['kernel']))


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

    assert_updates_as(muon_solver(), optax.adamw(LR, nesterov=True, weight_decay=0.0), params, [path],
                      "the router gate")


# --------------------------------------------------------------------------
# MuonClip: Muon plus the QK-Clip (Kimi K2, arXiv 2507.20534)
# --------------------------------------------------------------------------

QK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "muonclip" / "maxtext_mla.json"
MEGATRON = np.load(QK_FIXTURE.parent / "megatron.npz")


def muonclip_solver(**kwargs):
    return OptimConfig(optimizer='muonclip', learning_rate=LR, **kwargs).build(10)


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


def qk_stats(layers=1, heads=4, rows=3, values=None, nope=None, kv_heads=None, head_dim=8):
    """The `qk` collection with per-layer maxima, as the model sows it: the
    latent query's nope width, or the key-head count (`heads` unless given)
    and the head width."""
    stats = {}
    for layer in range(layers):
        sown = {'max_logits': (jnp.asarray(
            np.full((rows, heads), 10.0, np.float32) if values is None
            else np.asarray(values, np.float32).reshape(rows, heads), jnp.float32),)}
        if nope is not None:
            sown['qk_nope'] = jnp.asarray(nope)
        else:
            sown['kv_heads'] = jnp.asarray(heads if kv_heads is None else kv_heads)
            sown['head_dim'] = jnp.asarray(head_dim)
        stats[f'layers_{layer}'] = {'self_attn': sown}
    return stats


def test_muonclip_without_stats_steps_like_muon():
    """No maxima, no clip: the transform steps aside bitwise, leaving every
    run whose loss never opened the collection on Muon's step."""
    params, grads = qk_tree()
    muon = muon_solver()
    clipped = muonclip_solver()
    expected, _ = muon.update(grads, muon.init(params), params)
    initial = clipped.init(params)
    updates, state = clipped.update(grads, initial, params)
    moved = stepped(params, updates, initial, state)
    assert largest_difference(moved, optax.apply_updates(params, expected)) == 0.0


def test_a_quiet_clip_steps_like_muon():
    """Maxima below the threshold rescale nothing: bitwise the Muon step."""
    params, grads = qk_tree()
    muon = muon_solver()
    clipped = muonclip_solver()
    expected, _ = muon.update(grads, muon.init(params), params)
    initial = clipped.init(params)
    updates, state = clipped.update(
        grads, initial, params,
        qk_stats=qk_stats(values=[[10.0] * 4] * 3))
    moved = stepped(params, updates, initial, state)
    assert largest_difference(moved, optax.apply_updates(params, expected)) == 0.0


def clip_step(solver, stats):
    """One step of `solver` from fresh state over drawn kernels and their gradients, under the sown
    maxima `stats`, the weights as the step stores them (`stepped`)."""
    def step(params, grads):
        initial = solver.init(params)
        updates, state = solver.update(grads, initial, params, qk_stats=stats)
        return stepped(params, updates, initial, state)
    return step


def test_a_clip_rescales_the_weights_only_after_the_updates_it_takes_part_in():
    """`stepped` rescales after an update the clip ran in and no other. Under `optax.MultiSteps`, which
    runs its optimizer every second update and returns its state as it was on the others, a clip that
    fires on every update it runs rescales the weights on the second update alone: the first and third
    move them by the zero update, bit for bit."""
    params, grads = qk_tree()
    stats = qk_stats(values=[[200.0] * 4] * 3)
    solver = optax.MultiSteps(muonclip_solver(), every_k_schedule=2)
    state = solver.init(params)
    for update in range(3):
        updates, after = solver.update(grads, state, params, qk_stats=stats)
        moved = stepped(params, updates, state, after)
        unclipped = optax.apply_updates(params, updates)
        if update == 1:
            assert largest_difference(moved, unclipped) > 0
        else:
            assert largest_difference(moved, unclipped) == 0.0 and largest_difference(moved, params) == 0.0
        params, state = moved, after


def test_a_bounded_clipped_group_clamps_after_the_clip():
    """A bounded group's clamp is its weights' last word: under MuonClip, which rescales the stepped
    kernels, the clamp acts on the rescaled weights (`stepped`, in the chain's order), so every stored
    weight is within a bound that excludes zero, where clamping first would leave the clip's rescale
    under it. The clip fired: the weights are not the clamp of the unclipped step."""
    params, grads = qk_tree()
    stats = qk_stats(values=[[200.0] * 4] * 3)
    lower, upper = 0.1, 1.0
    solver = OptimConfig(optimizer='muonclip', learning_rate=LR, param_groups=(
        ParamGroup('all', ('*',), bounds=(lower, upper)),)).build(10)
    initial = solver.init(params)
    updates, state = solver.update(grads, initial, params, qk_stats=stats)
    moved = stepped(params, updates, initial, state)
    for leaf in jax.tree.leaves(moved):
        assert np.all((np.asarray(leaf) >= lower) & (np.asarray(leaf) <= upper))
    clamped = jax.tree.map(lambda leaf: jnp.clip(leaf, lower, upper), optax.apply_updates(params, updates))
    assert largest_difference(moved, clamped) > 0


CLIP_SEEDS = 8
"""Check 1's declared seeds for the QK-Clip against the paper's equations (`seeds_for`)."""


def test_the_clip_matches_the_paper_equations():
    """Per-head query and key rescale against the paper's equations in
    tools/muonclip_reference.py, over drawn kernels and gradients (check 1):
    one head fires at half, the rest hold."""
    stats = qk_stats(values=[[200.0, 50.0, 10.0, 5.0],
                             [10.0, 10.0, 10.0, 10.0],
                             [10.0, 10.0, 10.0, 10.0]])
    tx = scale_by_qk_clip(100.0)
    applied = clip_step(tx, stats)

    def paper(params, grads):
        s_max = jnp.max(jnp.asarray(stats['layers_0']['self_attn']['max_logits'][0], params[
            'layers_0']['self_attn']['q_proj']['kernel'].dtype), axis=0)
        gamma = clip_scale(s_max, 100.0)
        return {'layers_0': {'self_attn': {proj: {'kernel': clip_qk_kernel(
            params['layers_0']['self_attn'][proj]['kernel'] + grads['layers_0']['self_attn'][proj]['kernel'],
            gamma, 4)} for proj in ('q_proj', 'k_proj')}}}

    assert_as_exact(without_ynn(applied), without_ynn(paper), lambda seed: qk_tree(seed=seed),
                    "the clip against the paper", seeds=CLIP_SEEDS, truth_fn=paper)


@pytest.mark.parametrize("case", ["gqa", "mha"])
def test_a_query_group_clips_as_megatron_clips_it(case):
    """Grouped-query and multi-head attention against Megatron Core's own
    `SelfAttention.clip_qk` and `_clip_linear_qkv` (core_v0.19.2), run as
    published by tools/muonclip_reference.py: per query group, eta is
    tau over the group's largest logit, at most 1, and the group's query
    heads and its key head each scale by sqrt(eta). Under grouped queries a
    group with one firing head clips all four of its query heads and its
    key, and the quiet group keeps its weights; multi-head attention is the
    group of one. The kernels a zero gradient's update lands on are held to
    Megatron's float64 run by the float64 rule, and the values are
    untouched."""
    heads, width = MEGATRON[f"{case}/max_logits"].shape[0], MEGATRON[f"{case}/k"].shape[1]
    parts = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
    params = {"attn": {proj: {"kernel": jnp.asarray(MEGATRON[f"{case}/{part}"])}
                       for proj, part in parts.items()}}
    head_dim = MEGATRON[f"{case}/q"].shape[1] // heads
    stats = {"attn": {"max_logits": (jnp.asarray(MEGATRON[f"{case}/max_logits"])[None],),
                      "kv_heads": jnp.asarray(width // head_dim), "head_dim": jnp.asarray(head_dim)}}
    tx = scale_by_qk_clip(100.0)
    initial = tx.init(params)
    updates, state = tx.update(jax.tree.map(jnp.zeros_like, params), initial, params, qk_stats=stats)
    applied = stepped(params, updates, initial, state)
    for proj, part in parts.items():
        got = np.asarray(applied["attn"][proj]["kernel"])
        if part == "v":
            np.testing.assert_array_equal(got, MEGATRON[f"{case}/clipped_v_f64"])
        else:
            assert_as_exact_as_the_reference(got, MEGATRON[f"{case}/clipped_{part}"],
                                             MEGATRON[f"{case}/clipped_{part}_f64"], f"{case} {proj}")


def test_a_gated_query_clips_its_query_half_and_bounds_the_logits():
    """Qwen3.5's output gate doubles each query head to [query | gate]. The
    paper's QK-Clip rescales the query and key weights and nothing else, so
    the gate half keeps its kernel bitwise, and the layer's logits, read
    again on the same tokens, are each head's own times its group's eta:
    the firing group lands on tau and the other keeps its maxima, tau
    between the two groups' largest. One layer, so the clip's input is the
    embeddings both times."""
    model = tiny_decoder(output_gate=True, num_layers=1)
    variables = model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    ids = jnp.asarray(np.random.default_rng(1).integers(0, 32, (2, 8)), jnp.int32)
    _, sown = model.apply(variables, ids, mutable=["qk"])
    stats = sown["qk"]
    before = np.max(np.asarray(stats["layers_0"]["self_attn"]["max_logits"][0]), axis=0)
    tau = float(np.mean(before.reshape(2, 2).max(axis=1)))
    tx = scale_by_qk_clip(tau)
    params = variables["params"]
    initial = tx.init(params)
    updates, state = tx.update(jax.tree.map(jnp.zeros_like, params), initial, params, qk_stats=stats)
    clipped = stepped(params, updates, initial, state)
    query, query_after = (np.asarray(tree["layers_0"]["self_attn"]["q_proj"]["kernel"]).reshape(16, 4, 2, -1)
                          for tree in (params, clipped))
    np.testing.assert_array_equal(query_after[:, :, 1], query[:, :, 1])
    eta = np.minimum(1.0, tau / before.reshape(2, 2).max(axis=1)).repeat(2)
    assert 0 < np.sum(eta < 1) < 4

    # Over drawn decoders and tokens, the maxima read after the clip on par with the maxima before it
    # times each group's eta (check 1), the truth both of the decoder in float64. The threshold is the
    # draw's, between the two groups' float32 maxima; the clip takes it as a number.
    initialized = jax.jit(lambda key: model.init(key, jnp.ones((1, 8), jnp.int32)))

    def maxima(of, variables, ids):
        _, sown = of.apply(variables, ids, mutable=["qk"])
        return sown["qk"], jnp.max(sown["qk"]["layers_0"]["self_attn"]["max_logits"][0], axis=0)

    def draw(seed):
        variables = initialized(jax.random.key(seed))
        tokens = jax.random.randint(jax.random.key(seed + (1 << 20)), (2, 8), 0, 32)
        groups = np.asarray(maxima(model, variables, tokens)[1]).reshape(2, 2).max(axis=1)
        return variables, tokens, np.float32(np.mean(groups))

    def read_after(variables, ids, tau):
        stats, _ = maxima(model, variables, ids)
        tx = scale_by_qk_clip(float(tau))
        params = variables["params"]
        initial = tx.init(params)
        updates, state = tx.update(jax.tree.map(jnp.zeros_like, params), initial, params,
                                   qk_stats=stats)
        return maxima(model, {**variables, "params": stepped(params, updates, initial, state)}, ids)[1]

    def paper(of):
        def clipped_then_read(variables, ids, tau):
            # The paper's clip, written on the weights: each group's query heads and its key head by
            # the root of its eta, the gate half of each query head untouched; then the same tokens read.
            before = maxima(of, variables, ids)[1]
            root = jnp.sqrt(clip_scale(before.reshape(2, 2).max(axis=1), tau))
            attention = dict(variables["params"]["layers_0"]["self_attn"])
            query = jnp.asarray(attention["q_proj"]["kernel"]).reshape(16, 2, 2, 2, -1)
            query = query.at[:, :, :, 0].multiply(root[None, :, None, None].astype(query.dtype))
            key = jnp.asarray(attention["k_proj"]["kernel"]).reshape(16, 2, -1) * root[None, :, None].astype(
                query.dtype)
            attention["q_proj"] = {**attention["q_proj"], "kernel": query.reshape(16, -1)}
            attention["k_proj"] = {**attention["k_proj"], "kernel": key.reshape(16, -1)}
            layer = {**variables["params"]["layers_0"], "self_attn": attention}
            return maxima(of, {**variables, "params": {**variables["params"], "layers_0": layer}}, ids)[1]
        return clipped_then_read

    assert_as_exact(read_after, paper(model), draw, "the gated query's clip", seeds=CLIP_SEEDS,
                    truth_fn=paper(model.clone(dtype=jnp.float64)))


def test_a_biased_projection_clips_its_bias_with_its_kernel_and_bounds_the_logits():
    """With `attention_bias`, a query is `x W + b`, so the clip scales the bias with the kernel. Here
    the kernels are zero and the learned biases large, so the logits are the biases' alone: read
    again on the same tokens after the clip, the firing group's maxima shrink and the other group's
    hold, as the kernels' clip moves an unbiased layer's."""
    model = tiny_decoder(attention_bias=True, num_layers=1)
    variables = model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    rng = np.random.default_rng(2)
    attention = dict(variables["params"]["layers_0"]["self_attn"])
    for proj in ("q_proj", "k_proj"):
        attention[proj] = {"kernel": jnp.zeros_like(attention[proj]["kernel"]),
                           "bias": jnp.asarray(rng.normal(0, 3, attention[proj]["bias"].shape), jnp.float32)}
    params = {**variables["params"], "layers_0": {**variables["params"]["layers_0"], "self_attn": attention}}
    ids = jnp.asarray(rng.integers(0, 32, (2, 8)), jnp.int32)
    _, sown = model.apply({**variables, "params": params}, ids, mutable=["qk"])
    stats = sown["qk"]
    before = np.max(np.asarray(stats["layers_0"]["self_attn"]["max_logits"][0]), axis=0)
    tau = float(np.mean(before.reshape(2, 2).max(axis=1)))
    tx = scale_by_qk_clip(tau)
    initial = tx.init(params)
    updates, state = tx.update(jax.tree.map(jnp.zeros_like, params), initial, params, qk_stats=stats)
    clipped = stepped(params, updates, initial, state)
    _, resown = model.apply({**variables, "params": clipped}, ids, mutable=["qk"])
    firing = (before.reshape(2, 2).max(axis=1) > tau).repeat(2)
    assert 0 < np.sum(firing) < 4
    after = np.max(np.asarray(resown["qk"]["layers_0"]["self_attn"]["max_logits"][0]), axis=0)
    # The firing group's logits shrink toward zero, the clip having reached the biases they come
    # from; the other group's weights, and so its logits, stay as they were, bit for bit.
    assert np.all(np.abs(after[firing]) < np.abs(before[firing]))
    np.testing.assert_array_equal(after[~firing], before[~firing])


def test_the_latent_branches_match_maxtext():
    """The MLA branches against MaxText 0.2.4's own outputs, committed as
    tests/fixtures/muonclip/maxtext_mla.json: `q_proj` stands in for `wq_b`
    and `kv_b_proj` for `wkv_b`, the same per-head tensors under Dew's
    names. Each case's clipped kernel held by the float64 rule to MaxText's
    own float32 output, its truth the fixture's transcription of MaxText
    (`clip_mla_reference`, recorded bit for bit equal to it) in float64."""
    fixture = json.loads(QK_FIXTURE.read_text())
    assert fixture['maxtext'] == '0.2.4'
    tx = scale_by_qk_clip(100.0)
    for index, case in enumerate(fixture['cases']):
        proj = 'q_proj' if case['layer'] == 'wq_b' else 'kv_b_proj'
        param = np.asarray(case['param'], np.float32)
        flat = param.reshape(len(param), -1)
        params = {'l': {'m': {proj: {'kernel': jnp.asarray(flat)}}}}
        grads = jax.tree.map(jnp.zeros_like, params)
        max_logits = np.asarray(case['max_logits'], np.float32)
        stats = {'l': {'m': {'max_logits': (jnp.asarray(max_logits),),
                             'qk_nope': jnp.asarray(case['qk_nope'])}}}
        initial = tx.init(params)
        updates, state = tx.update(grads, initial, params, qk_stats=stats)
        applied = np.asarray(stepped(params, updates, initial, state)['l']['m'][proj]['kernel'])
        applied = applied.reshape(param.shape)
        with jax.enable_x64(new_val=True):
            scale = np.asarray(clip_scale(np.max(max_logits.astype(np.float64), axis=0), case['tau']))
        truth = clip_mla_reference(param.astype(np.float64), scale, case['qk_nope'], case['layer'])
        assert_as_exact_as_the_reference(applied, np.asarray(case['expected'], np.float32), truth,
                                         f"MaxText's {case['layer']}, case {index}")


THRESHOLD_SEEDS = 8
"""Check 1's declared seeds for the clip at a configured threshold against the paper's (`seeds_for`)."""


def test_the_threshold_rides_optimizer_opts():
    """`--optim.optimizer-opts '{"qk_clip_threshold": 5.0}'` reaches the
    transform: maxima of 50 clip nothing at the default 100, which steps as
    Muon bit for bit, and at 5 rescale each side of the Muon-stepped kernels
    by sqrt(5/50), so their product carries the tenth: on par with the
    paper's rescale of the stepped kernels over drawn kernels and gradients
    (check 1), the clip rounding `gamma * (W + update)` once."""
    stats = qk_stats(values=[[50.0] * 4] * 3)
    params, grads = qk_tree()
    default, muon = muonclip_solver(), muon_solver()
    plain, _ = muon.update(grads, muon.init(params), params)
    initial = default.init(params)
    updates, state = default.update(grads, initial, params, qk_stats=stats)
    moved = stepped(params, updates, initial, state)
    assert largest_difference(moved, optax.apply_updates(params, plain)) == 0.0
    clipped = clip_step(muonclip_solver(optimizer_opts={'qk_clip_threshold': 5.0}), stats)

    def paper(solver):
        def rescaled(params, grads):
            moved = optax.apply_updates(params, solver.update(grads, solver.init(params), params)[0])
            s_max = jnp.max(jnp.asarray(stats['layers_0']['self_attn']['max_logits'][0], params[
                'layers_0']['self_attn']['q_proj']['kernel'].dtype), axis=0)
            gamma = clip_scale(s_max, 5.0)
            return {'layers_0': {'self_attn': {proj: {'kernel': clip_qk_kernel(
                moved['layers_0']['self_attn'][proj]['kernel'], gamma, 4)} for proj in ('q_proj', 'k_proj')}}}
        return rescaled

    # The truth's Muon keeps its momentum in the parameters' float64, where the config's holds float32.
    wide = muon_solver(optimizer_opts={'mu_dtype': None})
    assert_as_exact(without_ynn(clipped), without_ynn(paper(muon)), lambda seed: qk_tree(seed=seed),
                    "the clip at a configured threshold", seeds=THRESHOLD_SEEDS, truth_fn=paper(wide))


def tiny_decoder(**overrides):
    fields = {'vocab_size': 32, 'emb_features': 16, 'num_layers': 2, 'num_heads': 4,
                  'num_kv_heads': 2, 'mlp_features': 32, 'max_seq_len': 8,
                  'qk_norm': False}
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
    # Doubling a binary float is exact, and the projection, the rotary and the scaled dot product
    # commute with it, so the doubled queries' maxima are the maxima doubled, bit for bit.
    np.testing.assert_array_equal(after, 2 * before)
    # Layer 1 reads layer 0's output, so its maxima legitimately move; what
    # has to hold there is presence and finiteness, not equality.
    (downstream,) = sown_doubled["qk"]["layers_1"]["self_attn"]["max_logits"]
    assert downstream.shape == (1, 4)
    assert bool(jnp.all(jnp.isfinite(downstream)))

    _, shut = model.apply(variables, ids, mutable=[])
    assert shut == {}


@pytest.mark.parametrize("kind", ["grpo", "grpo-kl", "dpo", "ppo"])
def test_policy_steps_carry_live_qk_statistics_to_muonclip(kind):
    from flax import linen as nn

    from dew.objectives.base import Step, part
    from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective
    from dew.training import Trainer
    from dew.training.transaction import compact_qk

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, tokens, **packing):
            return nn.Embed(32, 1)(tokens)[..., 0]

    model = tiny_decoder(num_layers=1)
    if kind == "dpo":
        objective = DPOObjective(model, 5, qk_stats=True)
    elif kind == "ppo":
        objective = PPOObjective(model, 5, critic=Critic(), qk_stats=True, beta=.01)
    else:
        objective = GRPOObjective(model, 5, qk_stats=True, beta=.01 if kind == "grpo-kl" else 0.)
    rows = jax.device_count()
    ids = jnp.asarray(np.random.default_rng(7).integers(0, 32, (rows, 6)), jnp.int32)
    if kind == "dpo":
        paired = jnp.stack((ids, (ids + 3) % 32), axis=1)
        batch = {"input_ids": paired, "completion_mask": jnp.ones_like(paired)}
        scoring_ids = jnp.concatenate((paired[:, 0], paired[:, 1]))
        packing = {}
    else:
        mask = jnp.broadcast_to(jnp.arange(6) >= 3, ids.shape).astype(jnp.float32)
        batch = {"input_ids": ids, "text_segment_ids": jnp.ones_like(ids),
                 "text_positions": jnp.broadcast_to(jnp.arange(6), ids.shape),
                 "response_mask": mask, "advantages": mask,
                 "behavior_log_probs": jnp.full(ids.shape, -3.),
                 "old_values": jnp.zeros_like(mask), "returns": mask}
        scoring_ids = ids
        packing = {"segment_ids": batch["text_segment_ids"][:, :-1],
                   "positions": batch["text_positions"][:, :-1]}
    threshold = .01
    optimizer = OptimConfig(optimizer="muonclip", learning_rate=0.,
                            optimizer_opts={"qk_clip_threshold": threshold}).build(1)
    trainer = Trainer(objective, optimizer, key=jax.random.key(0))
    initial = trainer.initial_state()
    variables = initial.variables
    policy = part(variables, "policy") if kind == "ppo" else variables
    _, sown = model.apply(policy, scoring_ids[:, :-1], method="hidden_states", mutable=["qk"], **packing)
    expected = compact_qk(sown["qk"])
    peaks = expected["layers_0"]["self_attn"]["max_logits"][0]
    assert np.max(peaks) > threshold
    # The reference must not determine the clip, even when its logits are much larger.
    reference = None if initial.ema is None else jax.tree.map(lambda leaf: leaf * 3, initial.averaged)
    _, aux = objective.loss(variables, batch, Step(jnp.array(0), jax.random.key(1), reference))
    observed = aux.qk_stats["policy"] if kind == "ppo" and aux.qk_stats is not None else aux.qk_stats
    assert observed is not None
    for got, want in zip(jax.tree.leaves(compact_qk(observed)), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)
    start = jax.tree.map(np.asarray, policy["params"])
    compiled = trainer.compile(initial, batch)
    final, *_ = compiled(initial, batch)
    assert int(final.updates) == 1
    after = part(final.variables, "policy") if kind == "ppo" else final.variables
    for name in ("q_proj", "k_proj"):
        before = start["layers_0"]["self_attn"][name]["kernel"]
        updated = np.asarray(after["params"]["layers_0"]["self_attn"][name]["kernel"])
        assert np.linalg.norm(updated) < np.linalg.norm(before)


# --- bf16 optimizer state ---------------------------------------------------

def bf16_state_adamw(**kwargs):
    return OptimConfig(optimizer='adamw', learning_rate=LR, weight_decay=0.0,
                                       state_dtype='bfloat16', **kwargs).build(10)


BF16_SECOND_SEEDS = 16
"""Check 1's declared seeds for the bf16 state's second step against optax's from its moments
(`seeds_for`, paired)."""


def test_bf16_state_takes_optax_adamw_steps_from_the_fp32_moments():
    """The update is computed from the fp32 moments before they are rounded.
    The first starts from zero moments, which bf16 holds exactly, so it is
    the fp32-state AdamW update (`OPTIMIZER_MAP`), its options (nesterov,
    eps_root, b2) included; the second is AdamW's update from the bf16 state's
    own moments, widened. Each over drawn parameters and gradients, on par
    with optax's (check 1), its truth optax's float64 run from the same
    moments. What the state stores is each fp32 moment of the independent
    fp32-state run rounded to one of its two bf16 neighbours, entry for
    entry: from zero moments both runs compute each moment by the same
    products and nothing added, which no contraction can fuse otherwise, so
    the storage is held exactly; with the second step's arithmetic it holds
    the second update within the storage's rounding of the fp32 run's."""
    opts = {'nesterov': True, 'eps_root': 1e-8}
    reference = OPTIMIZER_MAP['adamw'](LR, b2=0.99, weight_decay=0.0, **opts)
    solver = bf16_state_adamw(b2=0.99, optimizer_opts=opts)
    first = jax.jit(lambda grads, params: solver.update(grads, solver.init(params), params)[1])

    def draw(seed):
        params, grads = drawn_tree(decoder_params(), seed)
        moments = first(grads, params).inner_state[0]
        # The bf16 state's moments after one step, widened exactly, and its count.
        return params, grads, np.asarray(moments.count), *(jax.tree.map(
            lambda leaf: np.asarray(leaf, np.float32), tree) for tree in (moments.mu, moments.nu))

    def second(params, grads, count, mu, nu):
        state = first(grads, params)
        update, after = solver.update(grads, state, params)
        for held in (state, after):
            moments = held.inner_state[0]
            assert {leaf.dtype for leaf in jax.tree.leaves((moments.mu, moments.nu))} == {
                jnp.dtype(jnp.bfloat16)}
        return update

    def from_moments(params, grads, count, mu, nu):
        state = reference.init(params)
        adam = state[0]._replace(count=jnp.asarray(count), mu=mu, nu=nu)
        return reference.update(grads, (adam, *state[1:]), params)[0]

    def fresh(params, grads, *_):
        return reference.update(grads, reference.init(params), params)[0]

    assert_as_exact(jax.jit(lambda params, grads, *_: solver.update(grads, solver.init(params), params)[0]),
                    jax.jit(fresh), draw, "the bf16 state's first step", seeds=UPDATE_SEEDS, truth_fn=fresh)
    assert_as_exact(second, jax.jit(from_moments), draw, "the bf16 state's second step",
                    seeds=BF16_SECOND_SEEDS, truth_fn=from_moments)

    params, grads = drawn_tree(decoder_params(), 0)
    stored = first(grads, params).inner_state[0]
    exact = reference.update(grads, reference.init(params), params)[1][0]
    for name in ("mu", "nu"):
        for moment, held in zip(jax.tree.leaves(getattr(exact, name)), jax.tree.leaves(getattr(stored, name)),
                                strict=True):
            bits = np.asarray(moment, np.float32).view(np.uint32)
            # bf16 keeps a float32's top 16 bits: the neighbour toward zero drops the rest, the one
            # away from zero is the next, and a moment bf16 holds exactly has only itself.
            toward = bits & np.uint32(0xFFFF0000)
            away = np.where(bits & np.uint32(0xFFFF), toward + np.uint32(0x10000), toward)
            kept = np.asarray(held, np.float32).view(np.uint32)
            assert np.all((kept == toward) | (kept == away)), name


def test_bf16_state_keeps_the_second_moments_small_increments():
    """At b2 = 0.999 an increment of the second moment is under half a bf16
    spacing, which round to nearest drops every step: nu started at 1 would
    never decay. The stochastic rounding keeps each increment in expectation,
    so the mean over many entries follows the fp32 moment."""
    params = {'w': jnp.zeros((8192,), jnp.float32)}
    grads = {'w': jnp.full((8192,), 1e-3, jnp.float32)}
    reference, solver = optax.scale_by_adam(), bf16_state_adamw()
    expected = reference.init(params)
    expected = expected._replace(nu={'w': jnp.ones(8192, jnp.float32)})
    state = solver.init(params)
    moments = state.inner_state[0]._replace(nu={'w': jnp.ones(8192, jnp.bfloat16)})
    state = state._replace(inner_state=(moments, *state.inner_state[1:]))
    reference_update, solver_update = jax.jit(reference.update), jax.jit(solver.update)
    for _ in range(1000):
        _, expected = reference_update(grads, expected, params)
        _, state = solver_update(grads, state, params)
    want = float(expected.nu['w'][0])
    assert want < 0.4
    # Each entry's rounding is its own draw: over the entries, their mean within a hundredth of the fp32
    # moment (the learning gate's two one-sided contrasts, each entry a unit).
    entries = np.asarray(state.inner_state[0].nu['w'], np.float64)
    assert_learns(lambda numbers: within("the second moment", entries[numbers], want, 1e-2 * want),
                  "stochastically rounded second moments", units=entries.size)


def program_of(closed) -> tuple:
    """A traced program as its equations' primitives and parameters, in order, and every constant it
    holds, its literals and its captured arrays: two traces are one program where these agree."""
    def constants(values):
        return tuple((np.asarray(value).dtype, np.asarray(value).tobytes()) for value in values)

    steps = []
    for equation in equations(closed.jaxpr):
        literals = [operand.val for operand in equation.invars if isinstance(operand, Literal)]
        # The sub-programs a parameter holds are walked as equations of their own.
        params = tuple(sorted((name, value) for name, value in equation.params.items()
                              if not isinstance(value, (Jaxpr, ClosedJaxpr))))
        steps.append((equation.primitive.name, params, constants(literals)))
    return tuple(steps), constants(closed.consts)


@pytest.mark.parametrize("optimizer", typing.get_args(Optimizer))
@pytest.mark.parametrize("form", ["constant", "schedule", "groups"])
def test_rates_of_one_schedule_lower_to_one_update(optimizer, form):
    """Two runs at different rates of one schedule's shape, a group's
    multiplier and its own schedule included, lower the same update: the
    rate is an operand in the optimizer's state, not a constant in its
    program. The updates still differ, by the rates."""
    params = decoder_params()["params"]
    grads = fixed_gradients(params)

    def config(rate):
        schedule = (None if form == "constant"
                    else Cosine(peak_value=rate, warmup_steps=2, end_value=rate / 10, init_value=rate / 4))
        groups = () if form != "groups" else (
            ParamGroup("kernels", ("*kernel",), learning_rate_multiplier=rate * 100,
                       schedule=Linear(peak_value=rate * 2, warmup_steps=1, end_value=rate / 8)),
            ParamGroup("rest", ("*",)))
        return OptimConfig(optimizer=optimizer, learning_rate=rate if schedule is None else schedule,
                           param_groups=groups)

    programs, updates = [], []
    for rate in (1e-3, 4e-3):
        solver = config(rate).build(10)
        state = solver.init(params)
        programs.append(program_of(jax.make_jaxpr(solver.update)(grads, state, params)))
        updates.append(solver.update(grads, state, params)[0])
    assert programs[0] == programs[1]
    assert any(not np.array_equal(left, right) for left, right in
               zip(jax.tree.leaves(updates[0]), jax.tree.leaves(updates[1]), strict=True))


SCHEDULES = {
    "cosine": Cosine(peak_value=1e-2, warmup_steps=2, end_value=1e-3, init_value=1e-4),
    "linear": Linear(peak_value=3e-3, warmup_steps=2, decay_start=3, end_value=1e-4),
    # Rates forty decades apart: their ratio overflows float32, though each rate is a float32.
    "linear_wide": Linear(peak_value=1e-20, warmup_steps=2, decay_start=3, end_value=1e20),
    "exponential": Exponential(init_value=1e-2, end_value=1e-3, offset=1e-4),
    "power": Power(peak_value=2e-2, warmup_steps=2, a=4e-2, tail=PowerTail(start=4, end_value=1e-3)),
    "one_cycle": OneCycle(peak_value=1e-2),
}


@pytest.mark.parametrize(("optimizer", "schedule"), [
    ("lamb", SCHEDULES["cosine"]), ("adam", SCHEDULES["linear"]), ("adamw", SCHEDULES["exponential"]),
    ("muon", SCHEDULES["power"]), ("muonclip", SCHEDULES["one_cycle"]), ("adamw", None),
], ids=["lamb-cosine", "adam-linear", "adamw-exponential", "muon-power", "muonclip-one_cycle",
        "adamw-constant"])
def test_a_configured_optimizer_is_optaxs_own_at_the_rate_its_state_holds(optimizer, schedule):
    """`OptimConfig` runs the optax optimizer it names (`OPTIMIZER_MAP`)
    behind the global-norm clip, its betas and `optimizer_opts` reaching
    it, at the rate its state holds (`scheduled_solver`). Six steps on
    changing gradients, through the warmups, decays and tails, are optax's
    own optimizer at the schedule's own values bit for bit, and each step
    reports the rate it stepped at. How near float64 that rate is, the
    schedule's, is `test_a_scheduled_rate_is_as_near_float64_as_optaxs`."""
    params = decoder_params()["params"]
    adam = optimizer in ("adam", "adamw", "lamb")
    betas, opts = ({"b1": 0.8, "b2": 0.95}, {"eps": 1e-5, "eps_root": 1e-9}) if adam else ({}, {})
    config = OptimConfig(optimizer=optimizer, **betas, optimizer_opts=opts,
                         learning_rate=3e-3 if schedule is None else schedule, clip_grads=0.5)
    solver = config.build(10)
    given = {**betas, **opts, **({"mask": weight_decay_mask} if adam else {})}
    # The reference steps at the rate each of Dew's updates reports, read as it is reported.
    reported = []
    reference = optax.chain(optax.clip_by_global_norm(0.5), OPTIMIZER_MAP[optimizer](
        lambda count: jnp.stack(reported)[count], **given))
    state, expected_state = solver.init(params), reference.init(params)
    for step in range(6):
        grads = jax.tree.map(lambda grad, step=step: grad * (step + 1) * 0.3, fixed_gradients(params))
        updates, state = solver.update(grads, state, params)
        reported.append(jnp.asarray(learning_rate(state)))
        expected, expected_state = reference.update(grads, expected_state, params)
        for have, want in zip(jax.tree.leaves(updates), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(np.asarray(have), np.asarray(want))
        params = optax.apply_updates(params, updates)


def rate_fields(name: str, multiplier: float) -> dict[str, float]:
    """`SCHEDULES[name]`'s rate-valued fields times `multiplier`, the products in double: the values a
    run injects, one-cycle's derived init and end resolved as torch derives them."""
    schedule, m = SCHEDULES[name], multiplier
    if isinstance(schedule, Cosine):
        return {"init": schedule.init_value * m, "peak": schedule.peak_value * m}
    if isinstance(schedule, Linear):
        return {"peak": schedule.peak_value * m, "end": schedule.end_value * m}
    if isinstance(schedule, Exponential):
        return {"init": schedule.init_value * m, "end": schedule.end_value * m, "offset": schedule.offset * m}
    if isinstance(schedule, Power):
        assert schedule.tail is not None
        return {"peak": schedule.peak_value * m, "a": schedule.a * m, "tail": schedule.tail.end_value * m}
    peak = schedule.peak_value * m
    return {"peak": peak, "init": peak / 25, "end": peak / 25 / 1e4}


def optax_schedule(name: str, fields, steps: int = 10) -> optax.Schedule:
    """`SCHEDULES[name]` over `steps` at the rate `fields` (`rate_fields`), as optax's own pieces build
    it: Python floats for a float64 truth, or a run's injected values. What optax's builders branch on
    (a cosine's floor, a decay's rate) is a ratio of the configured rates, which a multiplier leaves as
    it is, so it stays a constant; and for the two schedules optax has none of, lm-engine's power law
    and torch's one-cycle, their published formulas in optax's pieces (`optax.join_schedules`). The
    count is the default float, float32 for the reference and float64 for its truth."""
    schedule, f = SCHEDULES[name], fields
    if isinstance(schedule, Cosine):
        # `optax.warmup_cosine_decay_schedule`, its floor the configured fields' ratio.
        floor = schedule.end_value / schedule.peak_value
        return optax.join_schedules([optax.linear_schedule(f["init"], f["peak"], schedule.warmup_steps),
                                     optax.cosine_decay_schedule(f["peak"], steps - schedule.warmup_steps,
                                                                 alpha=floor)], [schedule.warmup_steps])
    if isinstance(schedule, Linear):
        start = schedule.decay_start
        return optax.join_schedules([optax.linear_schedule(0.0, f["peak"], schedule.warmup_steps),
                                     optax.constant_schedule(f["peak"]),
                                     optax.linear_schedule(f["peak"], f["end"], steps - start)],
                                    [schedule.warmup_steps, start])
    if isinstance(schedule, Exponential):
        decay = optax.exponential_decay(f["init"], steps, schedule.end_value / schedule.init_value,
                                        end_value=f["end"])
        return lambda count: f["offset"] + decay(count)
    if isinstance(schedule, Power):
        assert schedule.tail is not None
        warmup, start = schedule.warmup_steps, schedule.tail.start

        def law(step):
            counted = jnp.asarray(step, jnp.result_type(float))
            return jnp.minimum(f["peak"], f["a"] * (counted * schedule.c) ** schedule.b)
        warm = jnp.minimum(f["peak"], f["a"] * (warmup * schedule.c) ** schedule.b)
        return optax.join_schedules([optax.linear_schedule(0.0, warm, warmup),
                                     lambda count: law(count + warmup),
                                     optax.linear_schedule(law(start), f["tail"], steps - start)],
                                    [warmup, start])
    # torch's two half cosines (`OneCycle`) as optax's cosine decays, from `init` up to the peak at
    # `rise` and down to `end` at the run's last step, where this run's phase ends fall on steps.
    rise, last = round(schedule.warmup_fraction * steps - 1), steps - 1
    configured = rate_fields(name, 1.0)
    return optax.join_schedules([
        optax.cosine_decay_schedule(f["init"], rise, alpha=configured["peak"] / configured["init"]),
        optax.cosine_decay_schedule(f["peak"], last - rise, alpha=configured["end"] / configured["peak"]),
    ], [rise])


RATE_SEEDS = 8
"""Check 1's declared seeds for a scheduled rate against optax's float32 schedule (`seeds_for`)."""

RATE_MULTIPLIERS = 64
"""The multipliers a draw scales a schedule's rates by, one rate of each step's leaf apiece."""


@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_a_scheduled_rate_is_as_near_float64_as_optaxs(name):
    """The rate a configured optimizer reports at each of a ten-step run's updates, its schedule's
    rates times a multiplier, is on par with the rate optax's own form for rates set at run time
    steps at (check 1): the rates injected as hyperparameters (`optax.inject_hyperparams`), the
    schedule built from their values at each update (`optax_schedule`) and read as the update of a
    unit gradient (`optax.scale_by_learning_rate`), the truth that schedule in float64. Dew holds the
    rate fields as operands, each rounded once as optax's injected values are, where a scale times
    the schedule's ratios to it rounded three times and overflowed where two rates are far apart
    (ledger 1660). Against optax's schedule of Python floats, whose constants fold in double, both
    forms sit further from float64, the cosine's 2.4 times and the linear decay's 3.5 times at worst:
    a float32 schedule's own arithmetic on values a run sets, which every one-executable form shares
    (the owner's exception to the gate's reference, ledger 1660). Each step's rate is a leaf over a
    draw's multipliers; every multiplier's state holds its fields (`rates_of`, as a `ParamGroup`'s
    multiplier sets them) and steps through the one compiled update, which the fields being operands
    makes possible."""
    schedule, params, unit = SCHEDULES[name], {"w": jnp.ones(3)}, {"w": jnp.ones(())}
    solver = OptimConfig(optimizer="adam", learning_rate=schedule, clip_grads=0).build(10)
    initial = solver.init(params)
    update = jax.jit(jax.vmap(solver.update, in_axes=(None, 0, None)))

    def factory(init=None, peak=None, end=None, offset=None, a=None, tail=None):
        held = {"init": init, "peak": peak, "end": end, "offset": offset, "a": a, "tail": tail}
        return optax.scale_by_learning_rate(optax_schedule(name, {key: value for key, value in held.items()
                                                                   if value is not None}), flip_sign=False)

    injected = optax.inject_hyperparams(factory, hyperparam_dtype=jnp.float32)(**rate_fields(name, 1.0))
    injected_state = injected.init(unit)
    injected_update = jax.jit(jax.vmap(injected.update, in_axes=(None, 0)))

    def draw(seed):
        return (np.exp(np.random.default_rng(seed).uniform(-3.0, 3.0, RATE_MULTIPLIERS)),)

    def batched(state, fields, held):
        """`state` once for each multiplier, its rate fields at `fields`' values where `held` names them."""
        def at(path, leaf):
            name_at = held(path)
            if name_at is not None:
                return jnp.asarray([values[name_at] for values in fields], leaf.dtype)
            return jnp.broadcast_to(leaf, (len(fields), *jnp.shape(leaf)))
        return jax.tree_util.tree_map_with_path(at, state)

    def dew(multipliers):
        fields = [rates_of(schedule, float(multiplier)) for multiplier in multipliers]
        state = batched(initial, fields, lambda path: path[-1].key if len(path) >= 2 and path[-2] == (
            jax.tree_util.GetAttrKey("rates")) else None)
        rates = {}
        for step in range(10):
            _, state = update({"w": jnp.ones(3)}, state, params)
            rates[f"step {step}"] = jnp.asarray(learning_rate(state))
        return rates

    def optax_rates(multipliers):
        fields = [rate_fields(name, float(multiplier)) for multiplier in multipliers]
        state = batched(injected_state, fields, lambda path: path[-1].key if len(path) >= 2 and path[-2] == (
            jax.tree_util.GetAttrKey("hyperparams")) else None)
        rates = {}
        for step in range(10):
            stepped_rate, state = injected_update(unit, state)
            rates[f"step {step}"] = stepped_rate["w"]
        return rates

    def truth(multipliers):
        built = [optax_schedule(name, rate_fields(name, float(multiplier))) for multiplier in multipliers]
        # optax counts in int32, which divides to float32: a float64 count keeps the schedule float64.
        return {f"step {step}": jnp.stack([values(jnp.asarray(step, jnp.float64)) for values in built])
                for step in range(10)}

    assert_as_exact(dew, optax_rates, draw, f"the {name} rate", seeds=RATE_SEEDS, truth_fn=truth)


def test_adamw_steps_land_as_near_float64_as_torchs():
    """Three AdamW steps from zero parameters, where each update is the rate times the gradient's
    sign, land as near their float64 values as torch's AdamW does (`reference_error`): the bias
    corrections are formed in double, as torch forms them, where optax's own forms 1 - b2**count
    from b2 rounded to float32 first, 1.3e-5 off at the first step (google-deepmind/optax#1827)."""
    import torch

    rate, decay, b1, b2, eps = 1e-3, 1e-2, 0.9, 0.999, 1e-8
    grads = [np.random.default_rng(step).normal(size=4096).astype(np.float32) for step in range(3)]
    params, first, second = np.zeros(4096), np.zeros(4096), np.zeros(4096)
    for count, grad in enumerate(grads, start=1):
        grad = grad.astype(np.float64)
        first, second = b1 * first + (1 - b1) * grad, b2 * second + (1 - b2) * grad ** 2
        clean = first / (1 - b1 ** count) / (np.sqrt(second / (1 - b2 ** count)) + eps)
        params = params - rate * (clean + decay * params)
    tensor = torch.zeros(4096, dtype=torch.float32, requires_grad=True)
    torch_adamw = torch.optim.AdamW([tensor], lr=rate, betas=(b1, b2), eps=eps, weight_decay=decay)
    solver = OPTIMIZER_MAP['adamw'](rate, b1=b1, b2=b2, eps=eps, weight_decay=decay)
    ours = jnp.zeros(4096, jnp.float32)
    state = solver.init(ours)
    for grad in grads:
        tensor.grad = torch.from_numpy(grad)
        torch_adamw.step()
        updates, state = solver.update(jnp.asarray(grad), state, ours)
        ours = optax.apply_updates(ours, updates)
    assert_as_exact_as_the_reference(np.asarray(ours), tensor.detach().numpy(), params, "adamw steps")


@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_narrow_weights_step_on_float32_moments_as_float32_weights_do(optimizer, dtype):
    """With `state_dtype='float32'`, the default, weights in float16 or bfloat16 keep their moments in
    float32: a gradient of 1e-4, whose square is below float16's smallest number, steps them as the
    same gradient steps float32 weights, by the rate and not by the rate over eps. (A decay of the
    weights themselves reads them in their own dtype, so none is taken here.)"""
    solver = OptimConfig(optimizer=optimizer, learning_rate=1e-3, clip_grads=0, weight_decay=0.0).build(4)
    weights = {"w": jnp.full((4, 4), 0.5, dtype)}
    gradient = {"w": jnp.full((4, 4), 1e-4, dtype)}
    narrow, _ = solver.update(gradient, solver.init(weights), weights)
    wide_weights = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), weights)
    wide, _ = solver.update(jax.tree.map(lambda leaf: leaf.astype(jnp.float32), gradient),
                            solver.init(wide_weights), wide_weights)
    np.testing.assert_array_equal(np.asarray(narrow["w"], np.float32), np.asarray(wide["w"]))
