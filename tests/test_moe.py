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

import collections
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from interop_support import fixture_arrays
from moe_support import by_expert, router_variables

from dew.nn.mlp import GatedMLP
from dew.nn.moe import ExpertMLP, Router, SparseMLP, load_balance_update

pytestmark = pytest.mark.kernel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "moe"
CONFIG = json.loads((FIXTURES / "config.json").read_text())
ROUTER_TOLERANCE = 1e-6
"""A router's gate values against the reference's, at fp32 on CPU: the largest observed
difference is 1.79e-07, from the gate matmul reading a transposed kernel."""
OUTPUT_TOLERANCE = 2e-5
"""An expert block's output against the reference's, at fp32 on CPU: the largest observed
difference is 7.63e-06 on outputs that reach 34.9, the summation order's spread."""


def fixture(name: str) -> dict:
    return fixture_arrays(FIXTURES / f"{name}.npz")


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

    # Largest observed difference 1.79e-07 at fp32 on CPU, from the gate
    # matmul reading a transposed kernel.
    assert_routes_as_recorded(tensors, weights, indices)


def assert_routes_as_recorded(tensors, weights, indices) -> None:
    """The router chose each token's recorded experts and gated them as recorded, expert by expert."""
    theirs_indices, theirs_weights = by_expert(tensors["router_indices"], tensors["router_weights"])
    ours_indices, ours_weights = by_expert(
        indices.reshape(-1, indices.shape[-1]), weights.reshape(-1, weights.shape[-1]))
    assert np.array_equal(ours_indices, theirs_indices)
    assert np.max(np.abs(ours_weights - theirs_weights)) < ROUTER_TOLERANCE


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
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < OUTPUT_TOLERANCE


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
    assert np.max(np.abs(np.asarray(output) - tensors["block_output"])) < OUTPUT_TOLERANCE


def v4_router(**overrides) -> Router:
    config = CONFIG["deepseek_v4"]
    settings = {"score_function": 'sqrtsoftplus',
                    "routed_scaling_factor": config["routed_scaling_factor"],
                    "expert_bias": True}
    settings.update(overrides)
    return Router(num_experts=config["num_local_experts"],
                  in_features=config["hidden_size"],
                  top_k=config["num_experts_per_tok"], **settings)


@pytest.mark.parametrize(("name", "router"), [
    pytest.param("deepseek", deepseek_router, id="v3-group-limited"),
    pytest.param("deepseek_v4", v4_router, id="v4-sqrt-softplus")])
def test_router_reproduces_the_deepseek_choice(name, router):
    """DeepseekV3MoE's router (sigmoid scores, a per-expert selection bias, the
    node limit over expert groups) and `DeepseekV4TopKRouter`'s (sqrt(softplus)
    scores, the selection bias), each renormalising and scaling: largest
    observed weight differences 2.38e-07 and 1.19e-07 at fp32 on CPU, on
    weights the routed scaling factor has already multiplied."""
    tensors = fixture(name)
    weights, indices = router().apply(router_variables(tensors, bias=True), jnp.asarray(tensors["hidden"]))
    assert_routes_as_recorded(tensors, weights, indices)


def test_a_sqrt_softplus_routers_gradient_is_finite_where_softplus_underflows():
    """DeepSeek V4's router scores `sqrt(softplus(logits))`: where an expert's logit takes softplus below
    float32's least normal, the plain formula's derivative, `sqrt`'s at zero, is infinite, and the
    router's gradient came out NaN from an unselected expert's zero cotangent (row 3919); its score
    there is the formula's tail (`test_the_sqrt_softplus_score_is_the_formula_and_its_tail`), and the
    gradient finite."""
    router = v4_router(expert_bias=False)
    hidden = jax.random.normal(jax.random.key(0), (4, CONFIG["deepseek_v4"]["hidden_size"]))
    variables = router.init(jax.random.key(1), hidden)
    # Logits far down for every expert but a few, softplus underflowing for most.
    pushed = jax.tree.map(lambda leaf: leaf * 4000.0, variables)

    def routed(variables):
        weights, _ = router.apply(variables, hidden)
        return jnp.sum(weights)

    gradient = jax.grad(routed)(pushed)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))


def test_the_sqrt_softplus_score_is_the_formula_and_its_tail():
    """`moe.sqrt_softplus` on par with the plain formula's float32, `sqrt(softplus(logits))`, from the
    float64 formula over logits from far down its tail to large (check 1), and below `exp(logits)` at
    float32's epsilon its tail `exp(logits / 2)`: an expert down the tail keeps a positive score where the
    plain formula's float32 is zero."""
    from reference_error import assert_as_exact

    from dew.nn import moe

    def plain(logits):
        return jnp.sqrt(jax.nn.softplus(logits))

    def draw(seed):
        return (jnp.asarray(np.random.default_rng(seed).uniform(-170.0, 30.0, 4096), jnp.float32),)

    assert_as_exact(jax.jit(moe.sqrt_softplus), jax.jit(plain), draw, "the sqrt-softplus score", seeds=8)
    logits = jnp.asarray([-170.0, -110.0, -40.0, -16.0, -15.0, 0.0, 30.0], jnp.float32)
    scores, formula = np.asarray(moe.sqrt_softplus(logits)), np.asarray(plain(logits))
    assert np.all(scores > 0) and not np.all(formula > 0)


@pytest.mark.parametrize("fill", [0.0, None], ids=["axk2-zero", "deepseek-inf"])
def test_a_router_chooses_an_expert_outside_its_best_groups_by_the_score_its_reference_fills_in(fill):
    """A grouped router whose best group's experts all choose below zero: A.X-K2's fills every expert
    outside that group at 0.0 (modeling_axk2.py:340), so those outrank the group's own, and DeepSeek V3's
    at -inf, so they never do (`Router.excluded_group_score`). Of eight experts in four groups, two
    groups are a token's best and the four experts outside them are its choice, or never are."""
    import torch
    from transformers import AutoConfig
    from transformers.models.axk2.modeling_axk2 import AXK2TopkRouter
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3TopkRouter

    fields = {"hidden_size": 16, "n_routed_experts": 8, "num_experts_per_tok": 4, "n_group": 4,
              "topk_group": 2, "routed_scaling_factor": 1.0, "norm_topk_prob": True}
    reference = (AXK2TopkRouter(AutoConfig.for_model("axk2", **fields)) if fill is not None
                 else DeepseekV3TopkRouter(AutoConfig.for_model("deepseek_v3", **fields)))
    generator = np.random.default_rng(0x3080)
    weight = generator.standard_normal((8, 16)).astype(np.float32)
    # Every expert's score for choice falls below zero, the excluded experts' fill above it or not.
    bias = np.full((8,), -5.0, np.float32)
    hidden = generator.standard_normal((6, 16)).astype(np.float32)
    reference.weight.data = torch.from_numpy(weight)
    reference.e_score_correction_bias.data = torch.from_numpy(bias)
    with torch.no_grad():
        _, _, theirs = reference(torch.from_numpy(hidden))
    router = Router(num_experts=8, in_features=16, top_k=4, score_function='sigmoid',
                    normalize_weights=True, expert_groups=4, groups_per_token=2, expert_bias=True,
                    excluded_group_score=fill)
    variables = {"params": {"kernel": jnp.asarray(weight.T)},
                 "moe": {"e_score_correction_bias": jnp.asarray(bias)}}
    _, indices = router.apply(variables, jnp.asarray(hidden))
    np.testing.assert_array_equal(np.sort(np.asarray(indices), -1), np.sort(theirs.numpy(), -1))


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
    assert np.max(np.abs(np.asarray(output) - tensors["experts_output"])) < OUTPUT_TOLERANCE


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
    variables = nn.unbox(experts.init(jax.random.key(1), x, weights, indices))
    return experts, variables, x, weights, indices


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


@pytest.mark.parametrize("scale_inputs", [False, True])
def test_experts_served_from_packed_banks_are_on_par_with_their_values_dense(scale_inputs):
    """An expert layer whose matrices are packed NVFP4 banks in its `QUANTIZATION` collection, no parameter
    beside them (a served 4-bit checkpoint's, row 4516: `dew.nn.dense.packed_matrix`), multiplied as NVFP4
    products (`gemm.nvfp4_product`): on par with the layer holding the banks' values as dense parameters in
    the same compute dtype, bf16 (check 1, truth the values in float64). A layer scaling its inputs by their
    routing weights (Llama 4's) hands the packed products its rows scaled, as no packed product takes a
    prologue (`moe._streamed`)."""
    from arm_arithmetic import float64_held
    from packed_experts import packed_banks
    from reference_error import assert_as_exact

    from dew.quantization import QUANTIZATION

    # An NVFP4 matrix's depth is whole stages (`formats.NVFP4_STAGE`, 256).
    experts, width, hidden, tokens, top_k = 4, 256, 256, 24, 2
    shapes = (("gate_proj", (width, hidden)), ("up_proj", (width, hidden)), ("down_proj", (hidden, width)))
    banks = {name: packed_banks(experts, *shape, seed) for seed, (name, shape) in enumerate(shapes)}
    dense = {"params": {name: {"kernel": jnp.asarray(values)} for name, (_, values) in banks.items()}}
    packed = {QUANTIZATION: {name: {"kernel": bank} for name, (bank, _) in banks.items()}}

    def layer(dtype):
        return ExpertMLP(num_experts=experts, hidden_features=hidden, out_features=width, dtype=dtype,
                         scale_inputs=scale_inputs)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        x = jax.random.normal(streams[0], (tokens, width), jnp.bfloat16)
        indices = jax.random.randint(streams[1], (tokens, top_k), 0, experts)
        return x, jax.random.uniform(streams[2], (tokens, top_k), jnp.float32, 0.25, 1.0), indices

    assert_as_exact(jax.jit(lambda *inputs: layer(None).apply(packed, *inputs)),
                    jax.jit(lambda *inputs: layer(jnp.bfloat16).apply(dense, *inputs)), draw,
                    f"packed experts{', inputs scaled' if scale_inputs else ''}", seeds=8,
                    truth_fn=lambda *inputs: layer(None).apply(float64_held(dense), *inputs))


def test_a_scanned_runs_stacked_packed_banks_are_read_at_its_layer_as_that_layers_banks():
    """An expert projection over a stack of three layers' packed NVFP4 banks read in place at a traced layer
    (`Layered`, the banked decode of a scanned packed load: row 4861) is, bit for bit, the projection over
    that layer's bank alone, its gathered rows read gathered and the output scattered to its slots alike."""
    from packed_experts import packed_banks

    from dew.nn.kernels import formats
    from dew.nn.moe import Gathered, Layered, expert_projection

    banks = [packed_banks(4, 256, 128, seed)[0] for seed in range(3)]
    stack = jax.tree.map(lambda *fields: jnp.stack([jnp.asarray(field) for field in fields]), *banks)
    sizes = jnp.asarray([5, 0, 7, 4], jnp.int32)
    rng = np.random.default_rng(4861)
    tokens = jnp.asarray(rng.standard_normal((12, 256)), jnp.bfloat16)
    rows = jnp.asarray(rng.permutation(16) % 12, jnp.int32)
    slots = jnp.asarray(rng.permutation(16), jnp.int32)
    inputs = Gathered(tokens, rows, slots)

    def project(kernel):
        return expert_projection(inputs, kernel, sizes, jnp.bfloat16, "auto", None, scatter=slots)

    for layer, bank in enumerate(banks):
        assert isinstance(bank, formats.NvFp4)
        got = jax.jit(project)(Layered(stack, jnp.asarray(layer)))
        want = jax.jit(project)(jax.tree.map(jnp.asarray, bank))
        assert np.asarray(got).tobytes() == np.asarray(want).tobytes(), layer


def test_packed_banks_decode_into_the_parameters_an_export_writes():
    """A served layer's packed banks decoded for an export or a merge (`quantization.decoded_banks`):
    each its values exactly, as the layer's `params` under its name, and the bank gone from `QUANTIZATION`;
    the layer applied to them reads what the packed layer reads, to the dense product's rounding."""
    from packed_experts import packed_banks

    from dew.quantization import QUANTIZATION, decoded_banks

    bank, values = packed_banks(2, 256, 256, seed=3)
    decoded = decoded_banks({"params": {"router": {"kernel": jnp.zeros((4, 2))}},
                                    QUANTIZATION: {"experts": {"up_proj": {"kernel": bank}}}})
    assert QUANTIZATION not in decoded
    np.testing.assert_array_equal(np.asarray(decoded["params"]["experts"]["up_proj"]["kernel"]), values)
    np.testing.assert_array_equal(np.asarray(decoded["params"]["router"]["kernel"]), np.zeros((4, 2)))


def test_a_runs_stacked_packed_banks_decode_as_each_layers_bank_does():
    """A scanned run's packed banks stacked on its layer axis, as a banked store holds them for serving
    (`dew.inference.banks`: `[layers, experts, in, out]`), decoded for an export
    (`quantization.decoded_banks`, before the export reads the layers apart): each layer's values
    exactly as its bank alone decodes, over every leading axis."""
    from packed_experts import packed_banks

    from dew.quantization import QUANTIZATION, decoded_banks

    layers = [packed_banks(2, 256, 256, seed=seed) for seed in (3, 4)]
    stacked = jax.tree.map(lambda *fields: np.stack(fields), *(bank for bank, _ in layers))
    decoded = decoded_banks({QUANTIZATION: {"experts": {"up_proj": {"kernel": stacked}}}})
    np.testing.assert_array_equal(np.asarray(decoded["params"]["experts"]["up_proj"]["kernel"]),
                                  np.stack([values for _, values in layers]))


@pytest.mark.parametrize("part", [
    pytest.param("values", id="values"),
    pytest.param("gradients", id="gradients", marks=pytest.mark.check1_shards(seeds=8, per_task=2)),
])
def test_a_layers_paired_and_scaled_expert_products_are_one_launch_as_exact_as_apart(part, request):
    """A stack's layer's gate and up projections over a scaled stream (`Gathered.scales`, Llama 4's scaled
    input) are, on an H100's routes, one paired launch (`paired_projections`) taking both whole stacks, the
    layer's offset into them, the rows' table and the scale in its prologue: no copy of the rows gathered or
    scaled, nor of the layer's matrices sliced (`grouped_matmul.weighed`, its route the template's). Its
    outputs and its gradients of the tokens, both stacks and the scales are as exact as the products of the
    rows gathered and scaled and the layer's matrices sliced, each multiplied apart, both from float64
    truth (check 1; row 12306). The kernels run interpreted on this CPU, as the H100's: the binding and the
    values one case (16 s), the gradients another, their draws in map tasks of two (128 s a task of four).
    """
    from generation_support import here
    from map_judge import marked_seeds
    from reference_error import as_given, assert_as_exact, with_gradients

    from dew.nn import moe
    from dew.nn.kernels import gemm, generation, grouped_matmul

    layers, experts, depth, width, tokens, top_k = 2, 2, 128, 256, 68, 2
    slots = tokens * top_k
    sizes = jnp.asarray([70, 66], jnp.int32)
    order = np.random.default_rng(12306).permutation(slots).astype(np.int32)
    rows, where = jnp.asarray(order // top_k), jnp.arange(slots, dtype=jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(tokens, depth)), jnp.bfloat16),
                *(jnp.asarray(rng.normal(size=(layers, experts, depth, width)) / 8, jnp.bfloat16)
                  for _ in range(2)),
                jnp.asarray(rng.uniform(0.25, 1.0, size=slots), jnp.float32),
                tuple(jnp.asarray(rng.normal(size=(slots, width)), jnp.bfloat16) for _ in range(2)))

    def dew(x, gate, up, scales):
        layer = jnp.asarray(1, jnp.int32)
        return moe.paired_projections(moe.Gathered(x, rows, where, scales), moe.Layered(gate, layer),
                                      moe.Layered(up, layer), sizes, jnp.bfloat16, "auto", None)

    # This process's own platform, read before `generation.assumed` states another.
    on_a_cpu = here() == "cpu"

    def launched(x, gate, up, scales):
        # The paired call `dew` makes (its binding asserted below) run as the H100's launch: on a GPU the
        # projections themselves; on this CPU the template's interpreter over the same hand-off
        # (`moe._streamed`), which product code never asks for (`interpret_on_cpu`).
        if not on_a_cpu:
            return dew(x, gate, up, scales)
        tokens, gather, prologue = moe._streamed(moe.Gathered(x, rows, where, scales), jnp.bfloat16)
        return gemm.gemm(tokens, gate, sizes, compute=jnp.bfloat16, out_dtype=jnp.bfloat16,
                         out_dtypes=(jnp.bfloat16, jnp.bfloat16), paired=up, layer=jnp.asarray(1, jnp.int32),
                         gather=gather, **prologue, interpret_on_cpu=True)

    def plain(x, gate, up, scales, wide=jnp.float32, out=jnp.bfloat16):
        # The rows scaled as the prologue scales them, in the products' dtype; float64 truth rounds nothing.
        scaled = (x[rows] * scales.astype(x.dtype)[:, None] if wide == jnp.float32
                  else x[rows].astype(wide) * scales.astype(wide)[:, None]).astype(wide)

        def projected(stack):
            return jnp.concatenate([scaled[:70] @ stack[1, 0].astype(wide),
                                    scaled[70:] @ stack[1, 1].astype(wide)]).astype(out)
        return projected(gate), projected(up)

    def valued(f):
        """`f` of the drawn inputs but the cotangent."""
        return lambda x, gate, up, scales, cotangent: f(x, gate, up, scales)

    truth = functools.partial(plain, wide=jnp.float64, out=jnp.float64)
    with generation.assumed("sm90", interpreted=True), generation.chosen(grouped_matmul="mosaic"):
        with grouped_matmul.weighed() as calls:
            jax.eval_shape(dew, *draw(0)[:4])
        paired = [(next(iter(call.bound.values()))[1].arguments, kernel) for call, kernel in calls
                  if next(iter(call.bound.values()))[1].arguments["paired"] is not None]
        assert paired, [kernel for _, kernel in calls]
        for arguments, kernel in paired:
            assert kernel == "mosaic", kernel
            assert arguments["kernel"].shape == arguments["paired"].shape == (layers * experts, depth, width)
            assert arguments["first"] is not None and arguments["gather"].shape == (slots,)
            assert arguments["tokens"].shape == (tokens, depth) and arguments["hooks"].prologue is not None
        label = "a layer's paired experts over scaled rows"
        if part == "gradients":
            # The gradients eagerly, as the Mosaic interpreters' ordered callbacks need (`as_given`).
            assert_as_exact(as_given(with_gradients(launched)), jax.jit(with_gradients(plain)), draw,
                            f"{label}, gradients", seeds=marked_seeds(request),
                            truth_fn=with_gradients(truth))
            return
        with generation.routes() as routes:
            assert_as_exact(jax.jit(valued(launched)), jax.jit(valued(plain)), draw, label, seeds=8,
                            truth_fn=valued(truth))
        assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}

    # Stacks read at two indices are each read at its own: no pair, the second stack at its layer.
    with grouped_matmul.weighed() as calls:
        jax.eval_shape(lambda x, gate, up, scales: moe.paired_projections(
            moe.Gathered(x, rows, where, scales), moe.Layered(gate, jnp.asarray(0, jnp.int32)),
            moe.Layered(up, jnp.asarray(1, jnp.int32)), sizes, jnp.bfloat16, "auto", None), *draw(0)[:4])
    assert all(next(iter(call.bound.values()))[1].arguments["paired"] is None for call, _ in calls)
    x, gate, up, scales, _ = draw(0)
    _, apart = moe.paired_projections(moe.Gathered(x, rows, where, scales), moe.Layered(gate, jnp.asarray(0)),
                                      moe.Layered(up, jnp.asarray(1)), sizes, jnp.bfloat16, "xla", None)
    np.testing.assert_array_equal(np.asarray(apart), np.asarray(plain(x, gate, up, scales)[1]))


def _gated_case():
    """A gated expert layer's rows, routing and three kernel stacks (two layers of two experts each) for
    `moe.gated_experts`, the layer index each projection reads: [slots, depth] rows of 68 tokens at top 2,
    sorted by expert, [layers, experts, depth, width] gate and up stacks and a [layers, experts, width,
    depth] down stack."""
    from dew.nn import moe

    layers, experts, depth, width, tokens, top_k = 2, 2, 128, 256, 68, 2
    slots = tokens * top_k
    order = np.random.default_rng(1376).permutation(slots).astype(np.int32)
    rows, where = jnp.asarray(order // top_k), jnp.arange(slots, dtype=jnp.int32)
    sizes = jnp.asarray([70, 66], jnp.int32)
    layer = jnp.asarray(1, jnp.int32)

    def draw(seed: int, dtype=jnp.bfloat16):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(tokens, depth)), dtype),
                *(jnp.asarray(rng.normal(size=shape) / 8, dtype)
                  for shape in ((layers, experts, depth, width),) * 2 + ((layers, experts, width, depth),)))

    def gated(x, gate, up, down, limit=None, activation="swiglu"):
        kernels = (moe.Layered(kernel, layer) for kernel in (gate, up, down))
        return moe.gated_experts(moe.Gathered(x, rows, where), *kernels, sizes, x.dtype, "auto", None,
                                 activation, limit)

    def apart(x, gate, up, down, limit=None):
        # Each projection on its own, the gated product between them, as before the fused form.
        stream = moe.Gathered(x, rows, where)
        gate_rows, up_rows = (moe.expert_projection(stream, moe.Layered(kernel, layer), sizes, x.dtype,
                                                    "auto", None) for kernel in (gate, up))
        hidden = moe.gated_product("swiglu")(*moe.swiglu_clamped(gate_rows, up_rows, limit))
        return moe.expert_projection(hidden, moe.Layered(down, layer), sizes, x.dtype, "auto", None)

    return draw, gated, apart, (slots, depth, width)


@pytest.mark.parametrize("limit", [None, 7.0], ids=["swiglu", "clamped swiglu"])
def test_a_gated_expert_layer_is_bitwise_its_projections_apart(limit):
    """A gated expert layer as one chain (`moe.gated_experts`, row 1376: the gate and up projections one
    paired product, their gated product its epilogue and the down projection's prologue) computes its output
    and each kernel's gradient bitwise its three projections apart with the gated product between them,
    clamped or not (GPT-OSS's limit), and the rows' gradient as exactly (check 1)."""
    from reference_error import assert_as_exact, with_gradients

    draw, gated, apart, _ = _gated_case()
    x, gate, up, down = draw(0)
    cotangent = jax.random.normal(jax.random.key(3), (x.shape[0] * 2, x.shape[1]), jnp.bfloat16)
    ran = [jax.jit(with_gradients(functools.partial(form, limit=limit)))(x, gate, up, down, cotangent)
           for form in (gated, apart)]
    # The output and each kernel's gradient bitwise (the leaves: output, rows', gate's, up's, down's); the
    # rows' gradient below.
    fused, separate = (jax.tree.leaves(each) for each in ran)
    for mine, theirs in zip([fused[0], *fused[2:]], [separate[0], *separate[2:]], strict=True):
        held = np.asarray(mine), np.asarray(theirs)
        np.testing.assert_array_equal(*(each.view(np.uint16) for each in held))

    # The rows' gradient sums the gate's and up's in the paired product's one accumulation, rounded once,
    # where apart rounds each and adds: as exact as apart's, from their float64 truth (check 1).
    def rows_gradient(form, dtype=None):
        def gradient(x, gate, up, down, cotangent):
            def loss(x):
                out = form(x, gate, up, down, limit=limit)
                wide = jnp.promote_types(out.dtype, jnp.float32)
                return jnp.sum(out.astype(wide) * cotangent.astype(wide))
            return jax.grad(loss)(x if dtype is None else x.astype(dtype))
        return gradient

    def drawn(seed):
        operands = draw(seed)
        return (*operands, jax.random.normal(jax.random.key(seed), cotangent.shape, jnp.bfloat16))

    with jax.enable_x64(new_val=True):
        def truth(x, gate, up, down, cotangent):
            wide = [jnp.asarray(each, jnp.float64) for each in (x, gate, up, down, cotangent)]
            return rows_gradient(apart)(*wide)
        assert_as_exact(jax.jit(rows_gradient(gated)), jax.jit(rows_gradient(apart)), drawn,
                        "a gated expert layer's rows' gradient", seeds=8, truth_fn=truth)


def test_a_gated_expert_layer_runs_its_chain_on_mosaic_gpu_under_a_declared_swiglu():
    """On an H100 whose grouped matmul is Mosaic GPU's and whose SwiGLU is declared Pallas's (a dense layer's
    `norm_swiglu` launch), a gated expert layer's chain selects Mosaic GPU's launches, forward and backward,
    refusing none (`generation.routes`), every launch's operands bf16: its gated product the epilogue's and
    the prologue's arithmetic on the tile (`activations.gated_hook`), no launch of its own inside them; one
    paired call with an epilogue and a down call reading what it formed (`grouped_matmul.weighed`)."""
    from dew.nn.kernels import generation, grouped_matmul

    draw, gated, _, _ = _gated_case()
    operands = draw(0)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", swiglu="pallas"),
          generation.routes() as routes, grouped_matmul.weighed() as calls):
        loss = jax.grad(lambda *operands: gated(*operands).astype(jnp.float32).sum(), argnums=(0, 1, 2, 3))
        jax.eval_shape(loss, *operands)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}, routes
    bound = [next(iter(call.bound.values()))[1].arguments for call, _ in calls]
    # Every launch, forward and backward, multiplies bf16 operands: none widened to take another route.
    assert all(arguments[name].dtype == jnp.bfloat16 for arguments in bound
               for name in ("tokens", "kernel") if arguments.get(name) is not None), bound
    # The forward's paired call carries the epilogue; the backward's (the rows' gradient through both stacks)
    # none.
    assert sum(arguments["hooks"].epilogue is not None for arguments in bound
               if arguments.get("paired") is not None) == 1, bound
    # The down projection's forward reads the epilogue's formed product (`gemm.gemm`'s `applied`), its
    # prologue the backward's: its call takes the [slots, width] rows through the down stack.
    down = operands[3]
    assert any(arguments.get("kernel") is not None and tuple(arguments["kernel"].shape) == (
        down.shape[0] * down.shape[1], *down.shape[2:]) and arguments["tokens"].shape[-1] == down.shape[2]
        for arguments in bound), bound


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
def test_a_gated_expert_layers_checkpointed_gradient_runs_its_pair_once(dtype):
    """A gated expert layer checkpointed under the `minimal` remat policy (its blocks' `nn.remat`: the gate's,
    up's and down's outputs saved by name), its gradient of the rows and all three kernels holds the paired
    product's forward and any replay of it, each wrapper and launch, two outputs or three, as often as the
    unchecked gradient does (`moe.gated_experts`, row 1376): the gate's and up's roundings named where the
    policy reads them, the paired product's backward reading its kept roundings and the down projection's
    its prologue's operands, neither replays the pair; at bf16 on Mosaic GPU's launch and at fp32 (where the
    accumulators' rounding is the accumulators themselves) on the route it selects."""
    from reference_error import equations

    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.nn.kernels import generation

    draw, gated, _, (slots, _, width) = _gated_case()
    operands = draw(0, dtype)

    def loss(*operands):
        return gated(*operands).astype(jnp.float32).sum()

    def paired(f) -> collections.Counter:
        # Every paired product's equations, each wrapper and its launch, forward or recomputed: those whose
        # outputs are the gate's and up's `[slots, width]` (two, a hookless replay of the pair) or those and
        # their gated product (three, the forward's epilogue), the primal and every gradient live.
        chosen = {"grouped_matmul": "mosaic"} if dtype == jnp.bfloat16 else {}
        with generation.assumed("sm90"), generation.chosen(**chosen):
            graph = jax.make_jaxpr(jax.value_and_grad(f, argnums=(0, 1, 2, 3)))(*operands)
        return collections.Counter(
            (equation.primitive.name, len(equation.outvars)) for equation in equations(graph)
            if len(equation.outvars) in (2, 3)
            and all(tuple(var.aval.shape) == (slots, width) for var in equation.outvars))

    plain = paired(loss)
    assert plain, "the gated expert layer runs its pair"
    # Replayed, the checkpointed gradient would hold each of them twice.
    assert paired(jax.checkpoint(loss, policy=REMAT_POLICIES["minimal"].checkpoint_policy())) == plain


def test_a_gated_expert_layer_whose_hook_no_epilogue_lowers_runs_its_projections_apart():
    """On an H100 whose grouped matmul is Mosaic GPU's, a gated expert layer under exact GeGLU (its GELU's
    erfc, which no Mosaic launch lowers, `hooks.hook_lowers`) runs its projections apart, forward and
    backward, each a Mosaic launch with no hook, refusing none (`moe.gated_experts`), as a dense gated MLP
    hands off (`dew.nn.gated_mlp`)."""
    from dew.nn.kernels import generation, grouped_matmul

    draw, gated, _, _ = _gated_case()
    operands = draw(0)
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes, grouped_matmul.weighed() as calls):
        def exact(*operands):
            return gated(*operands, activation="geglu_exact").astype(jnp.float32).sum()

        loss = jax.grad(exact, argnums=(0, 1, 2, 3))
        jax.eval_shape(loss, *operands)
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}, routes
    bound = [next(iter(call.bound.values()))[1].arguments for call, _ in calls]
    assert not any(arguments.get("paired") is not None and arguments["hooks"].epilogue is not None
                   for arguments in bound), bound
