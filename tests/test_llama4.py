"""Llama 4's attention rules and expert weighting against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one
`Llama4TextAttention` run as a local layer (interleaved rope, L2 q/k norm,
chunk 4) and as a global layer (no rope, temperature tuning at floor_scale
4), and one `Llama4TextMoe`. Everything runs at fp32 on CPU, and each
parity test states its tolerance and the largest difference observed.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.backbones.causal_transformer import GatedMLP
from dew.nn.llama4 import Llama4Attention, chunk_mask, temperature_scale
from dew.nn.moe import SparseMLP

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llama4"
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 8


def fixture(name: str) -> dict:
    with np.load(FIXTURES / f"{name}.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


def attention(**overrides: Any) -> Llama4Attention:
    settings: dict[str, Any] = dict(
        emb_features=HIDDEN, num_heads=HEADS, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, max_seq_len=32, rope_theta=500000.0,
        floor_scale=4.0, attn_scale=0.1)
    settings.update(overrides)
    return Llama4Attention(**settings)


def attention_variables(tensors: dict) -> dict:
    return {"params": {
        name: {"kernel": jnp.asarray(tensors[f"self_attn.{name}.weight"].T)}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")}}


def test_a_local_layer_matches_the_reference():
    """Interleaved rope, the scale-free L2 norm and the chunk of 4 over 12
    positions. Tolerance 1e-5; observed 4.8e-07."""
    tensors = fixture("attention")
    module = attention(use_rope=True, attention_chunk_size=4)
    output = module.apply(attention_variables(tensors), jnp.asarray(tensors["hidden"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors["local_output"]))) < 1e-5


def test_a_global_layer_matches_the_reference():
    """No rope, queries scaled by log1p(floor((p + 1) / 4)) / 10 + 1, the
    whole sequence attended. Tolerance 1e-5; observed 4.8e-07."""
    tensors = fixture("attention")
    module = attention(use_rope=False)
    output = module.apply(attention_variables(tensors), jnp.asarray(tensors["hidden"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors["global_output"]))) < 1e-5


@pytest.mark.parametrize("mutation, against", [
    (dict(use_rope=True, attention_chunk_size=None), "local_output"),
    (dict(use_rope=True, attention_chunk_size=4, use_qk_norm=False), "local_output"),
    (dict(use_rope=False, attn_temperature_tuning=False), "global_output"),
])
def test_each_rule_is_what_the_parity_tests(mutation, against):
    """Without the chunk, without the norm, or without the temperature, the
    same weights disagree with the reference."""
    tensors = fixture("attention")
    output = attention(**mutation).apply(attention_variables(tensors), jnp.asarray(tensors["hidden"]))
    assert float(np.max(np.abs(np.asarray(output) - tensors[against]))) > 1e-3


def test_the_chunk_mask_and_temperature_follow_the_reference_formulas():
    positions = jnp.arange(6)
    mask = chunk_mask(positions, positions, 4)[0, 0]
    assert mask.tolist() == [[p // 4 == k // 4 for k in range(6)] for p in range(6)]
    scale = temperature_scale(jnp.asarray([0, 3, 4, 7, 8]), 4.0, 0.1)
    expected = np.log1p(np.floor((np.array([0, 3, 4, 7, 8]) + 1) / 4)) * 0.1 + 1
    np.testing.assert_allclose(np.asarray(scale), expected, rtol=1e-6)


def test_decode_matches_prefill_on_both_rules():
    """The cache path: a prefill of 7 tokens then 5 single steps agree with
    the whole sequence on a chunked local layer and a global layer, the
    chunk and the temperature reading the cache slot as the position.
    Tolerance 1e-5; observed 2.4e-07 and 3.6e-07."""
    tensors = fixture("attention")
    hidden = jnp.asarray(tensors["hidden"])
    for settings in (dict(use_rope=True, attention_chunk_size=4), dict(use_rope=False)):
        module = attention(**settings)
        variables = attention_variables(tensors)
        whole = jnp.asarray(module.apply(variables, hidden))
        # The first decode call allocates the cache without writing it.
        state = {**module.init(jax.random.key(0), hidden[:, :1], decode=True), **variables}
        steps = []
        for start, stop in [(0, 7)] + [(position, position + 1) for position in range(7, 12)]:
            step, mutated = module.apply(state, hidden[:, start:stop], decode=True, mutable=["cache"])
            state = {**state, "cache": mutated["cache"]}
            steps.append(step)
        incremental = jnp.concatenate(steps, axis=1)
        assert float(jnp.max(jnp.abs(incremental - whole))) < 1e-5


def test_the_input_scaled_experts_match_the_reference_block():
    """`Llama4TextMoe`: sigmoid of the top-2 logits scales each token's
    input to its experts, the shared expert adds. Tolerance 1e-5; observed
    9.5e-07. Scaling the outputs instead disagrees."""
    tensors = fixture("moe")
    gate_up = tensors["feed_forward.experts.gate_up_proj"]
    width = gate_up.shape[-1] // 2

    def block(scale_inputs: bool) -> SparseMLP:
        return SparseMLP(num_experts=4, top_k=2, hidden_features=width, out_features=HIDDEN,
                         score_function="sigmoid", normalize_weights=False,
                         scale_inputs=scale_inputs,
                         shared=lambda name: GatedMLP(hidden_features=width, out_features=HIDDEN,
                                                      name=name))

    variables = {"params": {
        "gate": {"kernel": jnp.asarray(tensors["feed_forward.router.weight"].T)},
        "experts": {"gate_proj": {"kernel": jnp.asarray(gate_up[..., :width])},
                    "up_proj": {"kernel": jnp.asarray(gate_up[..., width:])},
                    "down_proj": {"kernel": jnp.asarray(tensors["feed_forward.experts.down_proj"])}},
        "shared_experts": {
            name: {"kernel": jnp.asarray(tensors[f"feed_forward.shared_expert.{name}.weight"].T)}
            for name in ("gate_proj", "up_proj", "down_proj")},
    }}
    hidden = jnp.asarray(tensors["hidden"])
    reference = tensors["output"].reshape(hidden.shape)
    output = block(True).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(output) - reference))) < 1e-5
    other = block(False).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(other) - reference))) > 1e-3


def test_input_scaling_reaches_the_router_gradient():
    """The routing weight sits inside the expert, so the router kernel gets
    gradient through the experts' nonlinearity."""
    module = SparseMLP(num_experts=4, top_k=2, hidden_features=16, out_features=HIDDEN,
                       score_function="sigmoid", normalize_weights=False, scale_inputs=True)
    hidden = jax.random.normal(jax.random.PRNGKey(0), (2, 5, HIDDEN))
    variables = module.init(jax.random.PRNGKey(1), hidden)
    grads = jax.grad(lambda v: jnp.square(jnp.asarray(module.apply(v, hidden))).sum())(variables)
    assert float(jnp.abs(grads["params"]["gate"]["kernel"]).max()) > 0
