"""Mixture of experts: router parity, the grouped matmul, the balancing bias
and the expert mesh axis.

The parity fixtures come from transformers 5.16.1 through
tools/moe_reference.py: `MixtralSparseMoeBlock` for softmax routing and the
block output, and the router of `DeepseekV3MoE` for sigmoid scores, the
selection bias and the group limit. Everything runs at fp32 on CPU, and each
parity test states its tolerance and the largest difference observed.

Slot order inside a token's top-k carries no meaning: the reference calls
`torch.topk(sorted=False)` and both implementations sum over the k slots, so
the comparisons sort each token's slots by expert id first.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from dew.nn.backbones.causal_transformer import CausalTransformer, GatedMLP, Mixture
from dew.nn.moe import (
    ExpertMLP, Router, SparseMLP, calculate_load_balance_updates,
)
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective, TEXT_KEY
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer, build_mesh
from dew.training.distributed import batch_sharding

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())

VOCAB = 64
SEQ_LEN = 16
BATCH = 8
# The training models here hold thousands of parameters, orders below the
# production shard threshold, so lower it or "sharded" would mean "replicated".
TINY_SHARD = 256


def fixture(name: str) -> dict:
    with np.load(FIXTURES / f"{name}.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


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


def sparse_variables(tensors, num_experts):
    """A checkpoint's per-expert tensors as a `SparseMLP` parameter tree.

    This is the whole translation a Qwen3.5-MoE or DeepSeek V4 checkpoint
    needs for its feed-forward: stack the experts of one projection onto a
    leading dimension, transpose each expert's matrix.
    """
    def stack(projection):
        return jnp.asarray(np.stack([
            tensors[f"mlp.experts.{expert}.{projection}.weight"].T
            for expert in range(num_experts)]))

    return {"params": {
        "gate": {"kernel": jnp.asarray(tensors["mlp.gate.weight"].T)},
        "experts": {projection: {"kernel": stack(projection)}
                    for projection in ("gate_proj", "up_proj", "down_proj")},
    }}


def mixtral_router(**overrides) -> Router:
    config = CONFIG["mixtral"]
    return Router(num_experts=config["num_local_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **overrides)


def deepseek_router(**overrides) -> Router:
    config = CONFIG["deepseek"]
    settings = dict(score_function='sigmoid',
                    normalize_weights=config["norm_topk_prob"],
                    routed_scaling_factor=config["routed_scaling_factor"],
                    expert_groups=config["n_group"],
                    groups_per_token=config["topk_group"],
                    expert_bias=True)
    settings.update(overrides)
    return Router(num_experts=config["n_routed_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **settings)


# --------------------------------------------------------------------------
# Router parity against transformers 5.16.1
# --------------------------------------------------------------------------

def test_router_reproduces_the_mixtral_choice_and_gate_values():
    """MixtralSparseMoeBlock: softmax over the experts, top-k, renormalise."""
    tensors = fixture("mixtral")
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, CONFIG["mixtral"]["hidden_size"])
    weights, indices = mixtral_router().apply(router_variables(tensors), hidden)

    theirs_indices, theirs_weights = by_expert(
        tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(indices, weights)
    assert np.array_equal(ours_indices, theirs_indices)
    # Largest observed difference 1.79e-07 at fp32 on CPU, from the gate
    # matmul reading a transposed kernel.
    assert np.max(np.abs(ours_weights - theirs_weights)) < 1e-6


def test_mixtral_parity_needs_the_renormalisation():
    """The mutation the router's weights would survive silently: keep the
    top-k softmax mass without dividing by it."""
    tensors = fixture("mixtral")
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, CONFIG["mixtral"]["hidden_size"])
    weights, indices = mixtral_router(normalize_weights=False).apply(
        router_variables(tensors), hidden)

    _, theirs_weights = by_expert(tensors["router_indices"], tensors["router_weights"])
    _, ours_weights = by_expert(indices, weights)
    assert np.max(np.abs(ours_weights - theirs_weights)) > 0.1


def test_router_reproduces_the_deepseek_group_limited_choice():
    """DeepseekV3MoE's router: sigmoid scores, a per-expert selection bias, the
    node limit over expert groups, renormalise, scale."""
    tensors = fixture("deepseek")
    hidden = jnp.asarray(tensors["hidden"])
    weights, indices = deepseek_router().apply(
        router_variables(tensors, bias=True), hidden)

    theirs_indices, theirs_weights = by_expert(
        tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(
        indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))
    assert np.array_equal(ours_indices, theirs_indices)
    # Largest observed difference 2.38e-07 at fp32 on CPU, on weights the
    # routed_scaling_factor of 2.5 has already multiplied.
    assert np.max(np.abs(ours_weights - theirs_weights)) < 1e-6


def test_deepseek_parity_needs_the_group_limit():
    """Dropping the node limit leaves a plain top-k, which picks other experts."""
    tensors = fixture("deepseek")
    hidden = jnp.asarray(tensors["hidden"])
    _, indices = deepseek_router(expert_groups=1, groups_per_token=1).apply(
        router_variables(tensors, bias=True), hidden)

    flat = np.asarray(indices).reshape(-1, indices.shape[-1])
    assert not np.array_equal(np.sort(flat, axis=-1),
                              np.sort(tensors["router_indices"], axis=-1))


def test_the_selection_bias_never_reaches_the_gate_values():
    """DeepSeek's balancing bias decides which experts a token gets and has no
    say in what they contribute, so the weights are the unbiased scores."""
    tensors = fixture("deepseek")
    config = CONFIG["deepseek"]
    hidden = jnp.asarray(tensors["hidden"])
    router = deepseek_router()
    variables = router_variables(tensors, bias=True)
    weights, indices = router.apply(variables, hidden)

    scores = router.apply(variables, hidden, method=Router.scores)
    biased = scores + jnp.asarray(tensors["mlp.gate.e_score_correction_bias"])
    scale = config["routed_scaling_factor"]

    def gathered(source):
        picked = jnp.take_along_axis(source, indices, axis=-1)
        return picked / jnp.sum(picked, axis=-1, keepdims=True) * scale

    assert np.max(np.abs(np.asarray(weights) - np.asarray(gathered(scores)))) < 1e-6
    # And the same gather off the biased scores is the bug this rules out.
    assert np.max(np.abs(np.asarray(weights) - np.asarray(gathered(biased)))) > 0.01


def test_the_expert_block_reproduces_the_mixtral_block_output():
    """The whole sparse feed-forward, which is the router plus the grouped
    matmul plus the weighted sum, against the reference block."""
    tensors = fixture("mixtral")
    config = CONFIG["mixtral"]
    block = SparseMLP(num_experts=config["num_local_experts"],
                      top_k=config["num_experts_per_tok"],
                      hidden_features=config["intermediate_size"],
                      out_features=config["hidden_size"])
    output = block.apply(sparse_variables(tensors, config["num_local_experts"]),
                         jnp.asarray(tensors["hidden"]))

    # Largest observed difference 7.63e-06 at fp32 on CPU, on outputs that
    # reach 24.7, so 3.1e-07 of the scale. The reference sums each expert's
    # contribution into a zeroed buffer while the ragged path sums a token's
    # k slots, so the two differ in summation order alone.
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < 2e-5


# --------------------------------------------------------------------------
# The aux-loss-free balancing bias
# --------------------------------------------------------------------------

def test_the_load_balance_update_pushes_against_the_busy_experts():
    """Lifted from maxtext layers/moe.py:238, checked on counts by hand: over
    four experts, three tokens on expert 0 and one on expert 1 averages one
    per expert, so expert 0 loses bias, expert 1 holds and the idle two gain."""
    indices = jnp.asarray([[[0, 0], [0, 1]]])
    update = calculate_load_balance_updates(indices, num_experts=4, rate=0.001)

    assert update.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(update), np.array([-0.001, 0.0, 0.001, 0.001], np.float32))


EXPERTS = 8


def balanced_run(steps=40, rate=0.01, direction=1.0):
    """A skewed router run with the bias update applied every step.

    The gate reads a constant first feature, so every expert has a standing
    preference on top of what a token asks for, which is the imbalance a real
    router starts with. `direction` of -1 is the mutation: an update that
    follows the load instead of opposing it.
    """
    router = Router(num_experts=EXPERTS, in_features=16, top_k=2, expert_bias=True)
    tokens = jax.random.normal(jax.random.key(3), (128, 16)).at[:, 0].set(1.0)
    kernel = (jax.random.normal(jax.random.key(4), (16, EXPERTS)) * 0.5).at[0].set(
        jnp.asarray([2.0, 1.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]))
    variables = {"params": {"kernel": kernel},
                 "moe": {"e_score_correction_bias": jnp.zeros(EXPERTS, jnp.float32)}}

    def load(variables):
        _, indices = router.apply(variables, tokens)
        return np.bincount(np.asarray(indices).ravel(), minlength=EXPERTS)

    start = load(variables)
    for _ in range(steps):
        _, indices = router.apply(variables, tokens)
        update = calculate_load_balance_updates(indices, EXPERTS, rate) * direction
        variables = {**variables, "moe": {
            "e_score_correction_bias":
                variables["moe"]["e_score_correction_bias"] + update}}
    return start, load(variables), np.asarray(
        variables["moe"]["e_score_correction_bias"])


def test_the_bias_update_evens_out_the_expert_load():
    """The mechanism end to end: the update applied to the bias every step,
    and the load spread has to close.

    The router reads the bias out of the `moe` collection and never writes it,
    which is where transformers keeps it and how MaxText hands the update
    back, so the step that applies it is this loop.
    """
    start, final, bias = balanced_run()

    # Observed 51 tokens between the busiest and the idlest expert at the
    # start and 12 after 40 steps, over 128 tokens and 8 experts.
    assert np.ptp(final) < np.ptp(start) / 3, (start, final)
    assert bias[np.argmax(start)] < 0 < bias[np.argmin(start)]
    assert final.min() > 0, final


def test_a_balancing_update_that_follows_the_load_makes_it_worse():
    """The mutation: the same loop with the update's sign flipped concentrates
    the load, so the balanced run above is the update's doing."""
    start, final, bias = balanced_run(direction=-1.0)

    assert np.ptp(final) > np.ptp(start), (start, final)
    assert bias[np.argmax(start)] > 0 > bias[np.argmin(start)]


@pytest.mark.parametrize("settings,message", [
    ({"score_function": 'softplus'}, "score_function"),
    ({"top_k": 9}, "top_k"),
    ({"expert_groups": 3}, "divide"),
    ({"expert_groups": 4, "groups_per_token": 5}, "groups_per_token"),
    ({"expert_groups": 8}, "two best"),
    ({"expert_groups": 4, "groups_per_token": 1, "top_k": 3}, "fewer than"),
])
def test_a_router_that_cannot_choose_top_k_experts_is_rejected(settings, message):
    with pytest.raises(ValueError, match=message):
        Router(**{"num_experts": 8, "in_features": 8, "top_k": 2, **settings}).init(
            jax.random.key(0), jnp.zeros((2, 8)))


# --------------------------------------------------------------------------
# The grouped matmul
# --------------------------------------------------------------------------

def routed_experts(num_experts=8, top_k=1, tokens=6, width=8, hidden=12,
                   implementation='xla'):
    """An ExpertMLP with routing that leaves some experts idle."""
    experts = ExpertMLP(num_experts=num_experts, hidden_features=hidden,
                        out_features=width, implementation=implementation)
    x = jax.random.normal(jax.random.key(0), (tokens, width))
    indices = jnp.asarray(
        np.arange(tokens * top_k).reshape(tokens, top_k) % 3, jnp.int32)
    weights = jnp.full((tokens, top_k), 1.0 / top_k, jnp.float32)
    variables = experts.init(jax.random.key(1), x, weights, indices)
    return experts, variables, x, weights, indices


def test_every_expert_initialises_like_the_dense_projection_it_replaces():
    """The expert dimension stacks whole matrices, so fan-in is one expert's
    input width. Counting the stack in it would scale every expert's weights
    down by sqrt(num_experts) and a from-scratch run would start quiet.
    """
    experts = ExpertMLP(num_experts=8, hidden_features=64, out_features=64)
    variables = experts.init(
        jax.random.key(1), jnp.zeros((4, 64)), jnp.ones((4, 1)),
        jnp.zeros((4, 1), jnp.int32))
    stacked = np.asarray(variables["params"]["gate_proj"]["kernel"])
    dense = np.asarray(nn.initializers.lecun_normal()(
        jax.random.key(1), (64, 64), jnp.float32))

    assert stacked.shape == (8, 64, 64)
    # Observed 0.1255 against the dense 0.1252 and the 1/sqrt(64) of 0.1250,
    # where a fan-in over the whole stack would give 0.0442.
    assert abs(stacked.std() - 1 / np.sqrt(64)) < 0.02 / np.sqrt(64)
    assert abs(stacked.std() - dense.std()) < 0.05 * dense.std()


def test_an_expert_no_token_reached_cannot_change_the_output():
    """Group sizes and the sort have to agree: if a token's rows land in the
    wrong expert's group, an idle expert's weights start showing up."""
    experts, variables, x, weights, indices = routed_experts()
    baseline = experts.apply(variables, x, weights, indices)
    used = set(np.asarray(indices).ravel().tolist())

    for expert in range(8):
        zeroed = {"params": {name: {"kernel": value["kernel"].at[expert].set(0.0)}
                             for name, value in variables["params"].items()}}
        output = experts.apply(zeroed, x, weights, indices)
        changed = not np.array_equal(np.asarray(output), np.asarray(baseline))
        assert changed == (expert in used), expert


def test_the_grouped_matmul_matches_a_per_expert_loop():
    """jax.lax.ragged_dot over sorted tokens against the same contractions
    written out one expert at a time."""
    experts, variables, x, weights, indices = routed_experts(top_k=2)
    output = experts.apply(variables, x, weights, indices)

    kernels = {name: np.asarray(value["kernel"])
               for name, value in variables["params"].items()}
    reference = np.zeros(x.shape, np.float32)
    for token in range(x.shape[0]):
        row = np.asarray(x[token])
        for slot in range(indices.shape[-1]):
            expert = int(indices[token, slot])
            gate = jax.nn.silu(row @ kernels["gate_proj"][expert])
            hidden = np.asarray(gate) * (row @ kernels["up_proj"][expert])
            reference[token] += float(weights[token, slot]) * (
                hidden @ kernels["down_proj"][expert])

    # Largest observed difference 1.19e-07 at fp32 on CPU.
    assert np.max(np.abs(np.asarray(output) - reference)) < 1e-6


def test_the_tokamax_grouped_matmul_agrees_with_the_xla_one():
    """The optional kernel path, on a machine that has tokamax installed."""
    pytest.importorskip("tokamax")
    experts, variables, x, weights, indices = routed_experts(top_k=2)
    expected = experts.apply(variables, x, weights, indices)
    other = experts.clone(implementation='tokamax').apply(
        variables, x, weights, indices)

    assert np.max(np.abs(np.asarray(other) - np.asarray(expected))) < 1e-6


def test_an_unknown_grouped_matmul_is_rejected():
    with pytest.raises(ValueError, match="tokamax"):
        routed_experts(implementation='megablox')


def test_routing_that_does_not_describe_the_tokens_is_rejected():
    experts, variables, x, weights, indices = routed_experts(top_k=2)
    with pytest.raises(ValueError, match="does not describe"):
        experts.apply(variables, x[:-1], weights, indices)


# --------------------------------------------------------------------------
# The decoder that grows experts
# --------------------------------------------------------------------------

def decoder_fields(**overrides) -> dict:
    settings = dict(vocab_size=VOCAB, emb_features=32, num_layers=4, num_heads=2,
                    num_kv_heads=1, mlp_features=64, max_seq_len=SEQ_LEN)
    settings.update(overrides)
    return settings


def decoder(**overrides) -> CausalTransformer:
    return CausalTransformer(**decoder_fields(**overrides))


def leaf_names(variables):
    leaves, _ = jax.tree_util.tree_flatten_with_path(variables)
    return sorted('/'.join(str(entry.key) for entry in path) for path, _ in leaves)


@pytest.mark.parametrize("settings,expected", [
    ({"mixture": {"experts": 4}}, (0, 1, 2, 3)),
    ({"mixture": {"experts": 4, "every": 2}}, (1, 3)),
    ({"mixture": {"experts": 4, "every": 4}}, (3,)),
    ({"mixture": {"experts": 4, "layers": (0, 2)}}, (0, 2)),
    ({}, ()),
])
def test_the_sparse_layers_are_the_ones_the_mixture_names(settings, expected):
    assert models.build("causal_transformer", **decoder_fields(**settings)).sparse_layers == expected


def test_a_sparse_layer_replaces_only_its_own_feed_forward():
    """The frozen leaf names: a model with experts on one layer keeps every
    other leaf of the dense model, including the dense layers' mlp."""
    dense = decoder()
    sparse = decoder(mixture=Mixture(experts=4, top_k=2, layers=(1,)))
    tokens = jnp.zeros((2, SEQ_LEN), jnp.int32)
    dense_names = leaf_names(dense.init(jax.random.key(0), tokens))
    sparse_names = leaf_names(sparse.init(jax.random.key(0), tokens))

    assert [name for name in dense_names if 'layers_1/mlp' not in name] == \
           [name for name in sparse_names if 'layers_1/mlp' not in name]
    assert [name for name in sparse_names if 'layers_1/mlp' in name] == [
        'params/layers_1/mlp/experts/down_proj/kernel',
        'params/layers_1/mlp/experts/gate_proj/kernel',
        'params/layers_1/mlp/experts/up_proj/kernel',
        'params/layers_1/mlp/gate/kernel',
    ]


def test_the_expert_leaves_hold_every_expert_of_the_reference_layout():
    """One leaf per projection, stacked over the experts, which is what a
    checkpoint's `mlp.experts.N.gate_proj.weight` tensors translate into."""
    model = decoder(emb_features=32, mixture=Mixture(experts=8, top_k=2, layers=(0,)))
    variables = model.init(jax.random.key(0), jnp.zeros((2, SEQ_LEN), jnp.int32))
    experts = variables["params"]["layers_0"]["mlp"]["experts"]

    assert experts["gate_proj"]["kernel"].shape == (8, 32, 64)
    assert experts["up_proj"]["kernel"].shape == (8, 32, 64)
    assert experts["down_proj"]["kernel"].shape == (8, 64, 32)
    assert variables["params"]["layers_0"]["mlp"]["gate"]["kernel"].shape == (32, 8)
    dense = variables["params"]["layers_1"]["mlp"]
    assert dense["gate_proj"]["kernel"].shape == (32, 64)


def test_an_expert_layer_and_a_dense_layer_agree_at_one_expert():
    """A single expert taking every token is the dense feed-forward, which is
    what makes the router's weight the only difference between them."""
    tokens = jax.random.normal(jax.random.key(0), (2, 3, 8))
    sparse = SparseMLP(num_experts=1, top_k=1, hidden_features=16, out_features=8)
    variables = sparse.init(jax.random.key(1), tokens)
    kernels = variables["params"]["experts"]

    gated = GatedMLP(hidden_features=16, out_features=8)
    dense_variables = {"params": {
        name: {"kernel": kernels[name]["kernel"][0]}
        for name in ("gate_proj", "up_proj", "down_proj")}}

    assert np.max(np.abs(np.asarray(sparse.apply(variables, tokens))
                         - np.asarray(gated.apply(dense_variables, tokens)))) < 1e-6


def test_the_router_runs_in_fp32_under_a_bfloat16_model():
    """Routing decides which experts a token trains, so a bf16 run must not
    decide it on bf16 scores. The experts stay in the compute dtype."""
    tokens = jnp.asarray(jax.random.normal(jax.random.key(0), (2, 3, 8)), jnp.bfloat16)
    sparse = SparseMLP(num_experts=4, top_k=2, hidden_features=16, out_features=8,
                       dtype=jnp.bfloat16)
    variables = sparse.init(jax.random.key(1), tokens)
    weights, _ = sparse.apply(variables, tokens, method=lambda module, x: module.gate(x))

    assert variables["params"]["gate"]["kernel"].dtype == jnp.float32
    assert weights.dtype == jnp.float32
    assert sparse.apply(variables, tokens).dtype == jnp.bfloat16

    # The same tokens at fp32 choose the same experts and weight them the
    # same, which is what running the gate in fp32 is for.
    exact = SparseMLP(num_experts=4, top_k=2, hidden_features=16, out_features=8)
    wide = exact.apply(variables, jnp.asarray(tokens, jnp.float32),
                       method=lambda module, x: module.gate(x))
    assert np.array_equal(np.asarray(weights), np.asarray(wide[0]))


@pytest.mark.parametrize("mixture,message", [
    ({"experts": 0}, "dense model has no mixture"),
    ({"experts": 4, "every": 2, "layers": (1,)}, "only"),
    ({"experts": 4, "layers": (4,)}, "outside"),
    ({"experts": 4, "every": 0}, "positive"),
    ({"experts": 4, "top_k": 5}, "top_k"),
])
def test_a_misconfigured_mixture_is_rejected(mixture, message):
    """A mixture's own dials are checked where they are set, and the ones
    that read the model are checked when it builds; either way the message
    names the dial."""
    with pytest.raises(ValueError, match=message):
        models.build("causal_transformer", **decoder_fields(mixture=mixture)).init(
            jax.random.key(0), jnp.zeros((1, SEQ_LEN), jnp.int32))


# --------------------------------------------------------------------------
# The expert mesh axis
# --------------------------------------------------------------------------

def expert_specs(expert_size, fsdp_size, num_experts=8, min_shard_size=TINY_SHARD):
    model = SparseMLP(num_experts=num_experts, top_k=2, hidden_features=64,
                      out_features=32)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 4, 32)))
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size, expert=expert_size))
    shardings = Layout(min_shard=min_shard_size).shardings(mesh, variables)
    return mesh, jax.tree.map(lambda sharding: sharding.spec, shardings)["params"]


@pytest.mark.parametrize("expert_size,fsdp_size", [(1, 8), (2, 4), (4, 2)])
def test_the_expert_dimension_takes_the_expert_axis(expert_size, fsdp_size):
    """Eight experts over one, two and four expert shards. The expert axis
    carries the expert dimension and nothing else, so the widths keep fsdp."""
    _, specs = expert_specs(expert_size, fsdp_size)
    expert_axis = 'expert' if expert_size > 1 else None

    assert specs["experts"]["gate_proj"]["kernel"] == P(expert_axis, None, 'fsdp')
    assert specs["experts"]["up_proj"]["kernel"] == P(expert_axis, None, 'fsdp')
    assert specs["experts"]["down_proj"]["kernel"] == P(expert_axis, 'fsdp')
    # The router's expert dimension rides the axis too; its width keeps fsdp.
    assert specs["gate"]["kernel"] == (
        P('fsdp', 'expert') if expert_size > 1 else P('fsdp'))


def test_the_expert_axis_shards_experts_without_an_fsdp_axis():
    """Expert parallelism alone: with fsdp at one there is still a dimension
    to split, which the old two-axis rule replicated."""
    _, specs = expert_specs(expert_size=8, fsdp_size=1)

    assert specs["experts"]["gate_proj"]["kernel"] == P('expert')
    assert specs["experts"]["down_proj"]["kernel"] == P('expert')


def test_an_expert_count_the_axis_cannot_split_keeps_the_widths_sharded():
    """Six experts over four shards divides nothing, so the expert name is
    dropped and the parameter still shards on the dimension that can."""
    _, specs = expert_specs(expert_size=4, fsdp_size=2, num_experts=6)

    assert specs["experts"]["gate_proj"]["kernel"] == P(None, None, 'fsdp')


@pytest.mark.parametrize("expert_size,fsdp_size", [(1, 8), (2, 4), (4, 2)])
def test_every_expert_parallel_layout_stays_inside_the_sharding_tolerance(
        expert_size, fsdp_size):
    model = models.build("causal_transformer", **moe_config())
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size, expert=expert_size))
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables)

    for (path, leaf), sharding in zip(
            jax.tree_util.tree_flatten_with_path(variables)[0],
            jax.tree.leaves(shardings), strict=True):
        for dimension, entry in enumerate(sharding.spec):
            if entry is None:
                continue
            axes = (entry,) if isinstance(entry, str) else entry
            size = int(np.prod([mesh.shape[axis] for axis in axes]))
            assert leaf.shape[dimension] % size == 0, (path, sharding.spec)
    layout.check(variables["params"], shardings["params"], mesh)


def test_a_mostly_dense_model_on_expert_only_parallelism_is_rejected():
    """Expert parallelism splits the experts and nothing else, so a model
    whose experts are a fifth of it runs mostly replicated. The check has to
    see that, which the fsdp-only rule could not: it returned as soon as the
    fsdp axis was one.
    """
    model = models.build("causal_transformer", **moe_config())
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, SEQ_LEN), jnp.int32))
    mesh = build_mesh(MeshSpec(fsdp=1, expert=8))
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables)

    with pytest.raises(ValueError, match="replicated"):
        layout.check(variables["params"], shardings["params"], mesh)


def test_build_mesh_rejects_an_expert_size_the_devices_cannot_hold():
    with pytest.raises(ValueError, match="expert 4"):
        build_mesh(MeshSpec(fsdp=4, expert=4))


def test_the_batch_is_split_over_the_expert_axis_too():
    """Expert parallelism must not cost data parallelism: every device holds a
    slice of the batch whichever axis it sits on."""
    mesh = build_mesh(MeshSpec(fsdp=2, expert=4))
    batch = jax.make_array_from_process_local_data(
        batch_sharding(mesh), np.zeros((jax.device_count(), 4), np.float32))

    assert len(batch.addressable_shards) == jax.device_count()
    assert batch.addressable_shards[0].data.shape == (1, 4)


# --------------------------------------------------------------------------
# Training on the simulated eight-device mesh
# --------------------------------------------------------------------------

def moe_config() -> dict:
    """Eight experts, so the expert dimension divides every mesh below."""
    return {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2,
            "num_heads": 2, "num_kv_heads": 1, "mlp_features": 64,
            "max_seq_len": SEQ_LEN,
            "mixture": {"experts": 8, "top_k": 2, "layers": (1,)}}


def token_batches():
    rng = np.random.default_rng(0)
    batch = {"text": rng.integers(0, VOCAB, size=(BATCH, SEQ_LEN + 1)).astype(np.int32)}
    while True:
        yield batch


class Data:
    def __init__(self, train):
        self._train, self.val, self.batch, self.records = train, None, BATCH, None

    def train(self):
        return self._train()

    steps_per_epoch = None


class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


def moe_trainer(expert_size, fsdp_size, tracker=None, bias=False):
    config = moe_config()
    model = models.build("causal_transformer",
                         **{**config, "mixture": {**config["mixture"], "bias": bias}})
    return Trainer(
        LMObjective(model, SEQ_LEN, balance_rate=0.01 if bias else None),
        optax.adam(1e-3), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp_size, expert=expert_size),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)


def run_losses(trainer, steps):
    """Per-step losses of a fit, as the tracker receives them."""
    trainer.tracker = tracker = RecordingTracker()
    trainer.fit(Data(token_batches), steps=steps, log_every=1)
    return [entry["train/loss"] for entry in tracker.scalars]


def test_the_expert_shards_train_the_same_model():
    """Fifty steps of the same sparse decoder at the same seed, with the
    experts on one shard and on four. Expert parallelism moves where the
    experts live, so the losses have to be the same run."""
    steps = 50
    one = run_losses(moe_trainer(1, 2), steps)
    four = run_losses(moe_trainer(4, 2), steps)

    assert len(one) == steps and np.all(np.isfinite(one))
    assert one[-1] < one[0] / 2, one
    difference = np.max(np.abs(np.array(one) - np.array(four)))
    # Observed equal on all 50 steps, 4.649611 down to 0.594418. The
    # tolerance is 1e-6 rather than zero because a different collective order
    # is allowed to round differently.
    assert difference < 1e-6, difference


def test_the_experts_are_really_split_across_the_expert_axis():
    state = moe_trainer(expert_size=4, fsdp_size=2).fit(Data(token_batches), steps=0)
    experts = state.params["params"]["layers_1"]["mlp"]["experts"]
    kernel = experts["gate_proj"]["kernel"]

    assert 'expert' in str(kernel.sharding.spec)
    assert kernel.addressable_shards[0].data.shape == (2, 32, 32)
    moments = state.opt_state[0].mu["layers_1"]["mlp"]["experts"]
    assert moments["gate_proj"]["kernel"].sharding.spec == kernel.sharding.spec


# --------------------------------------------------------------------------
# The balancing bias through Aux.variables
# --------------------------------------------------------------------------

def test_a_from_scratch_run_logs_the_load_and_moves_the_deepseek_bias():
    """The aux-loss-free balancing end to end: the routers sow their loads,
    the loss reports them and hands the bias update back through
    Aux.variables, and the trainer writes it into the `moe` collection.

    The bias starts at zero on every expert and the router is skewed by the
    tokens, so after a run the busiest experts carry a negative bias, the
    idlest a positive one, and the bias is not what a fresh init holds.
    """
    steps = 30
    tracker = RecordingTracker()
    trainer = moe_trainer(1, 2, tracker=tracker, bias=True)
    fresh = trainer.initial_state()
    assert set(fresh.params) == {"params", "moe"}
    np.testing.assert_array_equal(
        np.asarray(fresh.params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]), 0.0)

    state = trainer.fit(Data(token_batches), steps=steps, log_every=1)

    bias = np.asarray(state.params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"])
    assert np.any(bias != 0), "the bias never moved"
    assert bias.min() < 0 < bias.max(), bias
    # Every step moved every expert by the rate, one way or the other.
    np.testing.assert_allclose(np.abs(bias) / 0.01, np.round(np.abs(bias) / 0.01), atol=1e-4)
    loads = [entry["train/moe/max_load"] for entry in tracker.scalars]
    assert len(loads) == steps and all(1 / 8 <= load <= 1.0 for load in loads)
    assert all(entry["train/moe/min_load"] <= 1 / 8 for entry in tracker.scalars)
    # The bias is state, not a parameter: the optimizer holds no moment for it.
    assert "moe" not in state.opt_state[0].mu


def test_balancing_needs_a_router_with_a_bias():
    trainer = moe_trainer(1, 2)
    trainer.objective.balance_rate = 0.01
    params = trainer.initial_state().params
    with pytest.raises(ValueError, match="bias=True"):
        trainer.objective.loss(params, next(token_batches()),
                               Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None))


def test_the_balancing_bias_is_one_replicated_value_across_every_shard():
    """The bias is state every device reads, and its update counts tokens
    over the whole global batch, not over a device's slice.

    The step is `jax.jit` with in_shardings and out_shardings
    (dew/training/trainer.py), so the histogram inside it runs on the global
    array and the compiler inserts the reduction; this pins that. Every
    addressable shard has to hold the same whole bias, and the bias a mesh
    that splits the batch eight ways produces has to be the bias one device
    produces from the same batch. A per-shard count would move the bias by a
    per-shard direction and the shards would disagree.
    """
    steps = 3
    state = moe_trainer(4, 2, bias=True).fit(Data(token_batches), steps=steps)
    bias = state.params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]

    assert bias.sharding.spec == jax.sharding.PartitionSpec(), bias.sharding
    assert len(bias.addressable_shards) == jax.device_count()
    whole = np.asarray(jax.device_get(bias))
    for shard in bias.addressable_shards:
        assert np.array_equal(np.asarray(jax.device_get(shard.data)), whole)
    assert np.any(whole != 0), "the bias never moved"
    # Every step moved every expert by the rate, one way or the other.
    np.testing.assert_allclose(np.abs(whole) / 0.01, np.round(np.abs(whole) / 0.01),
                               atol=1e-4)
    assert np.abs(whole).max() <= steps * 0.01 + 1e-6

    single = single_device_bias(steps)
    np.testing.assert_array_equal(whole, single)


def single_device_bias(steps):
    """The same run on one device, as the value the sharded run must match.

    A fresh process is the only way to change the device count, so this
    subprocess runs the same trainer at one device and prints the bias.
    """
    script = """
import numpy as np, jax, optax
from dew.registry import models
import dew.nn.backbones
from dew.objectives.lm import LMObjective
from dew.training import Trainer, MeshSpec, Layout
import test_moe as suite

model = models.build("causal_transformer", **{
    **suite.moe_config(),
    "mixture": {**suite.moe_config()["mixture"], "bias": True}})
trainer = Trainer(LMObjective(model, suite.SEQ_LEN, balance_rate=0.01),
                  optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec(), layout=Layout(min_shard=suite.TINY_SHARD))
state = trainer.fit(suite.Data(suite.token_batches), steps=%d)
bias = state.params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]
print(",".join(repr(float(value)) for value in np.asarray(bias)))
""" % steps
    environment = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                   "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join(
                       [str(Path(__file__).resolve().parent),
                        str(Path(__file__).resolve().parents[1] / "src")])}
    finished = subprocess.run([sys.executable, "-c", script], check=True,
                              capture_output=True, text=True, env=environment)
    return np.asarray([float(value) for value in finished.stdout.strip().splitlines()[-1].split(",")])
