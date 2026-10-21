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
from interop_support import fixture_arrays

from dew.inference import TextGeneration
from dew.interop import Pretrained
from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.kv_cache import KVCache
from dew.nn.llama4 import Llama4Mixer
from dew.nn.mixer_base import MixerContext
from dew.nn.mlp import GatedMLP
from dew.nn.moe import SparseMLP
from dew.sampling import Sampling

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llama4"
LLAMA4_CHECKPOINT = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama4-tiny"
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 8


def fixture(name: str) -> dict:
    return fixture_arrays(FIXTURES / f"{name}.npz")


def attention(*, attention_chunk_size: int | None = None, **overrides):
    """The layer the `llama4` kind builds, with the fixture's geometry."""
    settings: dict[str, Any] = {"floor_scale": 4.0, "attn_scale": 0.1, **overrides}
    context = MixerContext(emb_features=HIDDEN, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                           max_seq_len=32, rope_theta=500000.0, attention_chunk=attention_chunk_size,
                           qk_norm=False)
    return Llama4Mixer(**settings).build(context)()


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


def test_decode_matches_prefill_on_both_rules():
    """The cache path: a prefill of 7 tokens then 5 single steps agree with
    the whole sequence on a chunked local layer and a global layer, the
    chunk and the temperature reading the cache slot as the position.
    Tolerance 1e-5; observed 2.4e-07 and 3.6e-07."""
    tensors = fixture("attention")
    hidden = jnp.asarray(tensors["hidden"])
    for settings in ({"use_rope": True, "attention_chunk_size": 4}, {"use_rope": False}):
        module = attention(**settings)
        variables = attention_variables(tensors)
        whole = jnp.asarray(module.apply(variables, hidden))
        # The first decode call allocates the cache without writing it.
        state = {**module.init(jax.random.key(0), hidden[:, :1], MixerCall(decode=True)), **variables}
        steps = []
        for start, stop in [(0, 7)] + [(position, position + 1) for position in range(7, 12)]:
            step, mutated = module.apply(state, hidden[:, start:stop], MixerCall(decode=True),
                                         mutable=["cache"])
            state = {**state, "cache": mutated["cache"]}
            steps.append(step)
        incremental = jnp.concatenate(steps, axis=1)
        assert float(jnp.max(jnp.abs(incremental - whole))) < 1e-5


def test_a_left_padded_row_scores_its_tokens_as_the_row_alone_cached_or_not():
    """Three padding slots before five real tokens: the chunk of 4 and the rotation read each real
    token's logical place, its rank among the row's real tokens (`Coordinates.positions`), so the
    padded row's real outputs are the row's alone, uncached and through a cache's prefill alike.
    Tolerance 1e-5."""
    tensors = fixture("attention")
    real = jnp.asarray(tensors["hidden"])[:, :5]
    padded = jnp.concatenate([jnp.zeros((real.shape[0], 3, HIDDEN), real.dtype), real], axis=1)
    valid = jnp.broadcast_to(jnp.arange(8) >= 3, padded.shape[:2])
    metadata = AttentionMetadata(coordinates=coordinates(valid=valid))
    for settings in ({"use_rope": True, "attention_chunk_size": 4}, {"use_rope": False}):
        module = attention(**settings)
        variables = attention_variables(tensors)
        alone = np.asarray(module.apply(variables, real))
        uncached = module.apply(variables, padded, MixerCall(metadata))
        state = {**module.init(jax.random.key(0), padded[:, :1], MixerCall(decode=True)), **variables}
        cached, _ = module.apply(state, padded, MixerCall(metadata, decode=True), mutable=["cache"])
        for got in (uncached, cached):
            assert float(np.max(np.abs(np.asarray(got)[:, 3:] - alone))) < 1e-5, settings


def test_the_temperature_reads_logical_positions_through_a_cache():
    """A global layer's queries scale by their logical places (`floor_scale` 4), which the caller
    gives as 8 onwards: a cached call scales them as an uncached one does, not by the cache slots
    0 onwards they are written at. Tolerance 1e-5."""
    tensors = fixture("attention")
    hidden = jnp.asarray(tensors["hidden"])[:, :6]
    metadata = AttentionMetadata(coordinates=coordinates(
        positions=jnp.broadcast_to(jnp.arange(6) + 8, hidden.shape[:2])))
    module = attention(use_rope=False)
    variables = attention_variables(tensors)
    uncached = np.asarray(module.apply(variables, hidden, MixerCall(metadata)))
    state = {**module.init(jax.random.key(0), hidden[:, :1], MixerCall(decode=True)), **variables}
    cached, _ = module.apply(state, hidden, MixerCall(metadata, decode=True), mutable=["cache"])
    assert float(np.max(np.abs(np.asarray(cached) - uncached))) < 1e-5


@pytest.mark.parametrize("dense, layout", [
    (KVCache(), KVCache(page_size=4)),
    (KVCache(quantized="int8"), KVCache(quantized="int8", page_size=4)),
    (KVCache(quantized="float8_e4m3fn"), KVCache(quantized="float8_e4m3fn", page_size=4))],
    ids=["paged", "int8-paged", "float8-paged"])
def test_a_paged_cache_draws_what_the_dense_cache_draws(dense, layout):
    """The tiny Llama 4 checkpoint (three chunked rotated layers, one global layer without rope)
    decodes greedily through each storage layout of the shared attention: a paged cache draws the
    dense cache's tokens at full precision and at each quantized format."""
    pretrained = Pretrained.load(str(LLAMA4_CHECKPOINT), dtype="float32", attention_impl="reference")
    prompts = np.load(LLAMA4_CHECKPOINT / "input_ids.npy")[:, :7]

    def drawn(cache: KVCache) -> np.ndarray:
        generate = TextGeneration(pretrained.model.with_kv_cache(cache), pretrained.variables,
                                  sampling=Sampling(temperature=0))
        return np.asarray(generate(prompts, 16, key=0).tokens)

    np.testing.assert_array_equal(drawn(layout), drawn(dense))


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
    output = block(scale_inputs=True).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(output) - reference))) < 1e-5
    other = block(scale_inputs=False).apply(variables, hidden)
    assert float(np.max(np.abs(np.asarray(other) - reference))) > 1e-3
