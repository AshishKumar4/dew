"""Mixture of experts: router parity, the grouped matmul and the balancing
bias. The decoder that grows experts and the expert mesh axis are in
tests/test_moe_decoder.py.

The parity fixtures come from transformers 5.16.1 through
tools/moe_reference.py: `MixtralSparseMoeBlock` for softmax routing and the
block output, `DeepseekV3MoE` for sigmoid scores, the selection bias, the
group limit and the shared expert, and DeepSeek V4's router and experts for
sqrt(softplus) scores and the swiglu limit. Everything runs at fp32 on CPU,
and each parity test states its tolerance and the largest difference
observed.

Slot order inside a token's top-k carries no meaning: the reference calls
`torch.topk(sorted=False)` and both implementations sum over the k slots, so
the comparisons sort each token's slots by expert id first.
"""

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from moe_support import by_expert, router_variables

from dew.nn.backbones.decoder_block import GatedMLP
from dew.nn.moe import ExpertMLP, Router, SparseMLP, load_balance_update
from dew.training import MeshSpec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())


def fixture(name: str) -> dict:
    with np.load(FIXTURES / f"{name}.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


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
    settings = {"score_function": 'sigmoid',
                    "normalize_weights": config["norm_topk_prob"],
                    "routed_scaling_factor": config["routed_scaling_factor"],
                    "expert_groups": config["n_group"],
                    "groups_per_token": config["topk_group"],
                    "expert_bias": True}
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


def test_a_router_with_all_zero_sigmoid_scores_keeps_finite_zero_weights():
    router = Router(num_experts=4, in_features=2, top_k=2, score_function="sigmoid")
    variables = {"params": {"kernel": jnp.full((2, 4), -1000., jnp.float32)}}
    hidden = jnp.ones((3, 2), jnp.float32)

    weights, indices = jax.jit(router.apply)(variables, hidden)

    np.testing.assert_array_equal(weights, np.zeros((3, 2), np.float32))
    assert np.isfinite(weights).all()
    assert np.all((np.asarray(indices) >= 0) & (np.asarray(indices) < 4))


def test_mixtral_parity_needs_the_renormalisation():
    """The mutation the router's weights would survive: keep the top-k
    softmax mass without dividing by it."""
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
    (weights, indices), sown = router.apply(variables, hidden, mutable=["router"])

    scores = sown["router"]["scores"][0]
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


def deepseek_block(shared: bool = True) -> SparseMLP:
    """`DeepseekV3MoE` as a SparseMLP: its router, experts and shared branch."""
    config = CONFIG["deepseek"]
    width = config["moe_intermediate_size"]
    return SparseMLP(
        num_experts=config["n_routed_experts"],
        top_k=config["num_experts_per_tok"],
        hidden_features=width, out_features=config["hidden_size"],
        score_function='sigmoid',
        routed_scaling_factor=config["routed_scaling_factor"],
        expert_groups=config["n_group"], groups_per_token=config["topk_group"],
        expert_bias=True,
        shared=None if not shared else functools.partial(
            GatedMLP, hidden_features=width * config["n_shared_experts"],
            out_features=config["hidden_size"]))


def deepseek_variables(tensors, shared: bool = True) -> dict:
    config = CONFIG["deepseek"]
    variables = sparse_variables(tensors, config["n_routed_experts"])
    variables["moe"] = {"gate": {"e_score_correction_bias": jnp.asarray(
        tensors["mlp.gate.e_score_correction_bias"])}}
    if shared:
        variables["params"]["shared_experts"] = {
            projection: {"kernel": jnp.asarray(
                tensors[f"mlp.shared_experts.{projection}.weight"].T)}
            for projection in ("gate_proj", "up_proj", "down_proj")}
    return variables


def test_the_expert_block_reproduces_the_deepseek_block_output():
    """`DeepseekV3MoE` end to end: the group-limited router, the routed
    experts scaled by 2.5, and the shared expert every token takes, summed.

    Tolerance 2e-5; largest observed difference 7.63e-06 at fp32 on CPU, on
    outputs that reach 34.9, the same summation-order spread as Mixtral's.
    """
    tensors = fixture("deepseek")
    output = deepseek_block().apply(deepseek_variables(tensors),
                                    jnp.asarray(tensors["hidden"]))
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < 2e-5


def test_deepseek_parity_needs_the_shared_branch():
    """The mutation: the same block without its shared expert is the routed
    sum alone, which the reference output is not."""
    tensors = fixture("deepseek")
    output = deepseek_block(shared=False).apply(
        deepseek_variables(tensors, shared=False), jnp.asarray(tensors["hidden"]))
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) > 0.1


def v4_router(**overrides) -> Router:
    config = CONFIG["deepseek_v4"]
    settings = {"score_function": 'sqrtsoftplus',
                    "routed_scaling_factor": config["routed_scaling_factor"],
                    "expert_bias": True}
    settings.update(overrides)
    return Router(num_experts=config["num_local_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **settings)


def test_router_reproduces_the_deepseek_v4_sqrt_softplus_choice():
    """`DeepseekV4TopKRouter`: sqrt(softplus) scores, the selection bias,
    renormalise, scale.

    Tolerance 1e-6; largest observed difference 1.19e-07 at fp32 on CPU.
    """
    tensors = fixture("deepseek_v4")
    hidden = jnp.asarray(tensors["hidden"])
    weights, indices = v4_router().apply(router_variables(tensors, bias=True), hidden)

    theirs_indices, theirs_weights = by_expert(
        tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(
        indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))
    assert np.array_equal(ours_indices, theirs_indices)
    assert np.max(np.abs(ours_weights - theirs_weights)) < 1e-6


def test_v4_parity_needs_the_sqrt_softplus():
    """A sigmoid over the same logits renormalises to other gate values."""
    tensors = fixture("deepseek_v4")
    hidden = jnp.asarray(tensors["hidden"])
    weights, indices = v4_router(score_function='sigmoid').apply(
        router_variables(tensors, bias=True), hidden)
    _, theirs_weights = by_expert(tensors["router_indices"], tensors["router_weights"])
    _, ours_weights = by_expert(
        indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))
    assert np.max(np.abs(ours_weights - theirs_weights)) > 0.01


def v4_experts(swiglu_limit) -> ExpertMLP:
    config = CONFIG["deepseek_v4"]
    return ExpertMLP(num_experts=config["num_local_experts"],
                     hidden_features=config["intermediate_size"],
                     out_features=config["hidden_size"], swiglu_limit=swiglu_limit)


def test_the_experts_reproduce_the_deepseek_v4_clamped_output():
    """`DeepseekV4Experts` on the router's choice: the gate capped at
    swiglu_limit from above and the up projection on both sides, then silu.

    Tolerance 2e-5; largest observed difference 4.77e-07 at fp32 on CPU, on
    outputs that reach 3.1; without the clamp the same weights are off by 25.
    """
    tensors = fixture("deepseek_v4")
    config = CONFIG["deepseek_v4"]
    variables = {"params": sparse_variables(
        tensors, config["num_local_experts"])["params"]["experts"]}
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, config["hidden_size"])
    output = v4_experts(config["swiglu_limit"]).apply(
        variables, hidden, jnp.asarray(tensors["router_weights"]),
        jnp.asarray(tensors["router_indices"]))
    assert np.max(np.abs(np.asarray(output) - tensors["experts_output"])) < 2e-5


def test_v4_parity_needs_the_clamp():
    """The unclamped experts on the same weights disagree, so the limit is
    what the fixture tests."""
    tensors = fixture("deepseek_v4")
    config = CONFIG["deepseek_v4"]
    variables = {"params": sparse_variables(
        tensors, config["num_local_experts"])["params"]["experts"]}
    hidden = jnp.asarray(tensors["hidden"]).reshape(-1, config["hidden_size"])
    output = v4_experts(None).apply(
        variables, hidden, jnp.asarray(tensors["router_weights"]),
        jnp.asarray(tensors["router_indices"]))
    assert np.max(np.abs(np.asarray(output) - tensors["experts_output"])) > 0.1


def test_a_swiglu_limit_that_clamps_nothing_is_rejected():
    with pytest.raises(ValueError, match="swiglu_limit"):
        v4_experts(0.0).init(jax.random.key(0), jnp.zeros((4, 16)),
                             jnp.ones((4, 2)), jnp.zeros((4, 2), jnp.int32))


# --------------------------------------------------------------------------
# The aux-loss-free balancing bias
# --------------------------------------------------------------------------

def test_the_load_balance_update_pushes_against_the_busy_experts():
    """Lifted from maxtext layers/moe.py:238, checked on counts by hand: over
    four experts, three tokens on expert 0 and one on expert 1 averages one
    per expert, so expert 0 loses bias, expert 1 holds and the idle two gain."""
    indices = jnp.asarray([[[0, 0], [0, 1]]])
    counts = jnp.sum(jax.nn.one_hot(indices.ravel(), 4, dtype=jnp.int32), axis=0)
    update = load_balance_update(counts, rate=0.001)

    assert update.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(update), np.array([-0.001, 0.0, 0.001, 0.001], np.float32))


EXPERTS = 8


def balanced_run(steps=40, rate=0.01, direction=1.0):
    """A skewed router run with the bias update applied every step.

    The gate reads a constant first feature, so every expert has a standing
    preference on top of what a token asks for, which is the imbalance a real
    router starts with. `direction` of -1 is the mutation: an update that
    follows the load, where the reference opposes it.
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
        counts = jnp.sum(jax.nn.one_hot(indices.ravel(), EXPERTS, dtype=jnp.int32), axis=0)
        update = load_balance_update(counts, rate) * direction
        variables = {**variables, "moe": {
            "e_score_correction_bias":
                variables["moe"]["e_score_correction_bias"] + update}}
    return start, load(variables), np.asarray(
        variables["moe"]["e_score_correction_bias"])


def test_the_bias_update_evens_out_the_expert_load():
    """The mechanism end to end: the update applied to the bias every step,
    and the load spread has to close.

    The router reads the bias out of the `moe` collection and never writes it
    (transformers keeps it there, and MaxText hands the update back the same
    way), so the step that applies it is this loop.
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


def test_float32_expert_routing_with_x64_keeps_its_output_and_gradients():
    """Enabling a float64 oracle must not prevent the float32 model running.

    bincount defaults to int64 under x64; TPU ragged-dot cannot lower those
    group sizes. Exercise the routed experts, not just a hand-typed count.
    """
    with jax.enable_x64(new_val=False):
        experts, variables, x, weights, indices = routed_experts(top_k=2)

    def step(variables, x):
        output = experts.apply(variables, x, weights, indices)
        return jnp.square(output).sum(), output

    results = []
    for enabled in (False, True):
        with jax.enable_x64(enabled):
            results.append(jax.device_get(jax.jit(
                jax.value_and_grad(step, argnums=(0, 1), has_aux=True))(variables, x)))
    for expected, actual in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        assert actual.dtype == expected.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

def random_routing(key, tokens, top_k, num_experts=8):
    """Distinct experts per token, as a router chooses them, under random
    positive weights; some experts stay idle and the loads differ."""
    choice_key, weight_key = jax.random.split(key)
    _, indices = jax.lax.top_k(jax.random.normal(choice_key, (tokens, num_experts)), top_k)
    return jax.random.uniform(weight_key, (tokens, top_k)), indices


def test_the_tokamax_grouped_matmul_agrees_with_the_xla_one():
    """The optional kernel path, on a machine that has tokamax installed."""
    pytest.importorskip("tokamax")
    experts, variables, x, _, _ = routed_experts(top_k=2, tokens=24)
    weights, indices = random_routing(jax.random.key(2), 24, 2)
    expected = experts.apply(variables, x, weights, indices)
    other = experts.clone(implementation='tokamax').apply(
        variables, x, weights, indices)

    # Observed 0 on CPU, where tokamax lowers to the same ragged_dot; the
    # bound leaves room for a Mosaic or Triton kernel's accumulation order.
    assert np.max(np.abs(np.asarray(other) - np.asarray(expected))) < 1e-6


def test_the_tokamax_grouped_matmul_backs_the_same_gradients():
    """Both kernels differentiate: the expert weights and the tokens get the
    same gradient through either, under routing that loads experts unevenly."""
    pytest.importorskip("tokamax")
    experts, variables, x, _, _ = routed_experts(top_k=3, tokens=24)
    weights, indices = random_routing(jax.random.key(3), 24, 3)
    probe = jax.random.normal(jax.random.key(4), x.shape)

    def gradients(module):
        def loss(variables, x):
            return jnp.sum(module.apply(variables, x, weights, indices) * probe)
        return jax.grad(loss, argnums=(0, 1))(variables, x)

    expected = gradients(experts)
    other = gradients(experts.clone(implementation='tokamax'))

    for theirs, ours in zip(jax.tree.leaves(expected), jax.tree.leaves(other), strict=True):
        assert np.max(np.abs(np.asarray(ours) - np.asarray(theirs))) < 1e-6
    assert all(float(jnp.max(jnp.abs(leaf))) > 0 for leaf in jax.tree.leaves(expected))


def test_an_unknown_grouped_matmul_is_rejected():
    with pytest.raises(ValueError, match="tokamax"):
        routed_experts(implementation='megablox')


def test_routing_that_does_not_describe_the_tokens_is_rejected():
    experts, variables, x, weights, indices = routed_experts(top_k=2)
    with pytest.raises(ValueError, match="does not describe"):
        experts.apply(variables, x[:-1], weights, indices)


@pytest.mark.parametrize('generation,gpu,chosen', [
    ('sm89', True, 'pallas'), ('sm86', True, 'pallas'), ('sm80', True, 'pallas'), ('sm75', False, 'xla'),
    ('sm90', True, 'xla'), ('v6e', False, 'xla'), ('cpu', False, 'xla')])
def test_auto_takes_the_measured_grouped_matmul_and_xla_elsewhere(monkeypatch, generation, gpu,
                                                                  chosen):
    """'auto' runs the Pallas kernels only on a generation they were measured
    on and can compile for; an unmeasured or older one runs XLA."""
    import dew.nn.moe as moe
    from dew.nn import kernels
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: generation)
    monkeypatch.setattr(moe, 'triton_runs', lambda: gpu)
    assert moe.grouped_matmul_kernel('auto', jnp.bfloat16, (jnp.bfloat16, jnp.float32),
                                     None) == chosen


def test_pallas_steps_aside_for_a_product_its_kernels_would_change(monkeypatch):
    """The kernels multiply at the operands' dtype and ignore precision, so
    fp32 at HIGHEST runs XLA even where they compile."""
    import dew.nn.moe as moe
    from dew.nn import kernels
    monkeypatch.setattr(kernels.generation, 'device_generation', lambda: 'sm89')
    monkeypatch.setattr(moe, 'triton_runs', lambda: True)
    assert moe.grouped_matmul_kernel('pallas', jnp.float32, (jnp.float32, jnp.float32),
                                     'highest') == 'xla'
    assert moe.grouped_matmul_kernel('pallas', jnp.float32, (jnp.float32, jnp.float32),
                                     'default') == 'pallas'


@pytest.mark.mesh
@pytest.mark.parametrize("dispatch", ["exchange", "global"])
@pytest.mark.parametrize("experts", ["gpt_oss", "mixture"])
def test_a_forward_reads_the_expert_kernels_as_stored(experts, dispatch):
    """bf16 expert kernels under an fp32 stream, as gpt-oss-20b is served over
    an expert axis. The forward multiplies them as stored: the stream rounds
    to bf16 and the products accumulate in fp32. Widened, by the dispatch for
    its gradient sums or by the stream's promotion, each layer's experts were
    copied to fp32: 14.35 GiB of live temporaries on four RTX 3090s. Read off
    the program JAX traces, since XLA's CPU backend runs a bf16 dot in fp32
    on its own."""
    from dew.nn.gpt_oss import GptOssMLP

    width, hidden = 64, 96
    if experts == "gpt_oss":
        module = GptOssMLP(width, hidden, 8, 2, dispatch=dispatch)
    else:
        module = SparseMLP(num_experts=8, top_k=2, hidden_features=hidden,
                           out_features=width, dispatch=dispatch)
    x = jnp.zeros((4, 16, width), jnp.float32)
    shapes = jax.eval_shape(module.init, jax.random.key(0), x)
    variables = jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, jnp.bfloat16), shapes)
    with jax.set_mesh(MeshSpec(expert=4, fsdp=2).build()), nn.logical_axis_rules(()):
        program = str(jax.make_jaxpr(module.apply)(variables, x))
    kernels = [f"{experts},{rows},{columns}" for experts in (8, 2)
               for rows, columns in ((width, hidden), (hidden, width),
                                     (width, 2 * hidden), (2 * hidden, width))]
    widened = [line for line in program.splitlines()
               if any(f"f32[{shape}]" in line for shape in kernels)]
    assert not widened, widened[:3]


def test_experts_keep_a_stored_dtype_only_when_narrower_than_the_stream():
    """fp32 over bf16 experts computes in bf16, as stored; bf16 over fp16
    experts promotes to fp32, since fp16 would overflow bf16 values past
    65504; one dtype throughout keeps it."""
    from dew.nn.moe import expert_compute_dtype

    def compute(stream, experts):
        return expert_compute_dtype(jnp.zeros(2, stream), jnp.zeros(2, experts), dtype=None)

    assert compute(jnp.float32, jnp.bfloat16) == jnp.bfloat16
    assert compute(jnp.bfloat16, jnp.float16) == jnp.float32
    assert compute(jnp.bfloat16, jnp.bfloat16) == jnp.bfloat16
    assert expert_compute_dtype(jnp.zeros(2, jnp.float32), jnp.zeros(2, jnp.bfloat16),
                                dtype=jnp.float32) == jnp.float32
