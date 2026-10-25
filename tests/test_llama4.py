"""Llama 4's attention rules and expert weighting against transformers 5.16.1.

The block fixtures come from tools/hf_reference_b.py: one
`Llama4TextAttention` run as a local layer (interleaved rope, L2 q/k norm,
chunk 4) and as a global layer (no rope, temperature tuning at floor_scale
4), and one `Llama4TextMoe`. Everything runs at fp32 on CPU, and each
parity test states its tolerance and the largest difference observed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_arrays
from reference_error import assert_as_exact

from dew.inference import TextGeneration
from dew.interop import Pretrained
from dew.nn.inputs import AttentionMetadata, MixerCall, coordinates
from dew.nn.kv_cache import KVCache
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.mlp import GatedMLP
from dew.nn.moe import SparseMLP
from dew.sampling import Sampling

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "llama4"
LLAMA4_CHECKPOINT = Path(__file__).resolve().parent / "fixtures" / "hf" / "llama4-tiny"
HIDDEN, HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 8


def fixture(name: str) -> dict:
    return fixture_arrays(FIXTURES / f"{name}.npz")


def attention(*, use_rope: bool, attention_chunk_size: int | None = None):
    """A layer of the kinds Llama 4's translator emits, with the fixture's geometry: a rotated one
    turning adjacent pairs under the weightless L2 norm, or one without positions scaling its queries
    by their place at floor_scale 4."""
    context = MixerContext(emb_features=HIDDEN, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                           max_seq_len=32, rope_theta=500000.0, attention_chunk=attention_chunk_size,
                           qk_norm=use_rope)
    mixer = (AttentionMixer(rotary_pairs="adjacent", qk_norm_weight=False) if use_rope
             else AttentionMixer(nope=True, temperature_tuning=(4.0, 0.1)))
    return mixer.build(context)()


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


TUNED_SEEDS = 32
"""The declared seeds for check 1 of the tuned windowed layer (`assert_as_exact`), sized on armada."""
TUNED_WINDOW, TUNED_FLOOR, TUNED_SCALE, TUNED_THETA = 4, 3.0, 0.5, 10000.0


def tuned_layer(dtype=jnp.float32):
    """A windowed kind's attention rotating halves and tuning its temperature, which no Llama 4 layer
    combines: the options compose as any attention's."""
    context = MixerContext(emb_features=HIDDEN, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                           max_seq_len=32, rope_theta=TUNED_THETA, sliding_window=TUNED_WINDOW, qk_norm=False,
                           dtype=dtype)
    return AttentionMixer(temperature_tuning=(TUNED_FLOOR, TUNED_SCALE)).build(context)()


def tuned_equation(hidden, params, dtype):
    """The layer written out: the projections, half-pair rotary at theta, each query times
    log1p(floor((p + 1) / floor_scale)) * attn_scale + 1, a causal window of `TUNED_WINDOW` keys
    (its own included), grouped key-value heads, 1/sqrt(d) and the output projection, in `dtype`."""
    p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params["params"])
    x = jnp.asarray(hidden, dtype)
    batch, length, _ = x.shape

    def heads(name, count):
        return (x @ p[name]["kernel"].reshape(HIDDEN, -1)).reshape(batch, length, count, HEAD_DIM)

    q, k, v = heads("q_proj", HEADS), heads("k_proj", KV_HEADS), heads("v_proj", KV_HEADS)
    places = jnp.arange(length, dtype=dtype)
    angles = places[:, None] * TUNED_THETA ** (-jnp.arange(0, HEAD_DIM, 2, dtype=dtype) / HEAD_DIM)
    cos, sin = jnp.cos(angles)[None, :, None], jnp.sin(angles)[None, :, None]

    def rotated(t):
        first, second = t[..., :HEAD_DIM // 2], t[..., HEAD_DIM // 2:]
        return jnp.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)

    temperature = jnp.log1p(jnp.floor((places + 1) / TUNED_FLOOR)) * TUNED_SCALE + 1
    q = rotated(q) * temperature[None, :, None, None]
    k = jnp.repeat(rotated(k), HEADS // KV_HEADS, axis=2)
    v = jnp.repeat(v, HEADS // KV_HEADS, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.asarray(HEAD_DIM, dtype))
    query, key = jnp.arange(length)[:, None], jnp.arange(length)[None, :]
    seen = (key <= query) & (key > query - TUNED_WINDOW)
    weights = jax.nn.softmax(jnp.where(seen, logits, -jnp.inf), axis=-1)
    attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(batch, length, -1)
    return attended @ p["o_proj"]["kernel"].reshape(-1, HIDDEN)


def test_a_windowed_kind_tunes_its_temperature_as_written():
    """A windowed attention kind rotating halves and tuning its query temperature computes its written
    equation (`tuned_equation`), full and through a cache (a prefill, then a token a call), on par
    with the equation in float32 against float64 (check 1)."""
    module = tuned_layer()
    length = 12

    def draw(seed):
        hidden = np.random.default_rng([1614, seed]).standard_normal((2, length, HIDDEN)).astype(np.float32)
        params = jax.tree.map(np.asarray, module.init(jax.random.key(seed), jnp.zeros((1, length, HIDDEN))))
        return hidden, params

    @jax.jit
    def full(hidden, params):
        return module.apply(params, hidden)

    @jax.jit
    def cached(hidden, params):
        state = {**module.init(jax.random.key(0), hidden[:, :1], MixerCall(decode=True)), **params}
        steps = []
        for start, stop in [(0, 7)] + [(position, position + 1) for position in range(7, length)]:
            step, mutated = module.apply(state, hidden[:, start:stop], MixerCall(decode=True),
                                         mutable=["cache"])
            state = {**state, "cache": mutated["cache"]}
            steps.append(step)
        return jnp.concatenate(steps, axis=1)

    def written(hidden, params):
        return tuned_equation(hidden, params, jnp.float32)

    def truth(hidden, params):
        return tuned_equation(hidden, params, jnp.float64)

    for ours, label in ((full, "tuned windowed layer"), (cached, "tuned windowed layer, cached")):
        assert_as_exact(ours, jax.jit(written), draw, label, seeds=TUNED_SEEDS, truth_fn=truth)


def test_a_llama4_of_unrotated_layers_alone_translates_as_llama4():
    """A Llama 4 config whose every layer carries no rope (`no_rope_layer_interval` 1) is still Llama 4:
    its record translates its weights by Llama 4's names with no model type stated, so the checkpoint's
    tensors land where the stated-type translation puts them."""
    import json

    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.interop.safetensors_io import read_weights

    config = json.loads((LLAMA4_CHECKPOINT / "config.json").read_text())
    config = {**config, "no_rope_layers": [0] * config["num_hidden_layers"]}
    tensors = read_weights(LLAMA4_CHECKPOINT)
    record = translate_config(config)
    assert {kind.split("_")[0] for kind in record["kinds"]} == {"full"}
    inferred = translate_weights(tensors, record)
    stated = translate_weights(tensors, record, model_type="llama4_text")
    assert jax.tree.structure(inferred) == jax.tree.structure(stated)
    for ours, theirs in zip(jax.tree.leaves(inferred), jax.tree.leaves(stated), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


def ngpt_layers(dtype=jnp.float32):
    """An nGPT attention with the temperature tuned, and the same attention untuned, at `dtype`."""
    from dew.nn.ngpt import NGPT

    context = MixerContext(emb_features=HIDDEN, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                           max_seq_len=32, rope_theta=TUNED_THETA, qk_norm=False,
                           ngpt=NGPT(std=0.02, alpha=1.0), dtype=dtype)
    return (AttentionMixer(temperature_tuning=(TUNED_FLOOR, TUNED_SCALE)).build(context)(),
            AttentionMixer().build(context)())


def test_a_tuned_temperature_holds_through_an_ngpt_query():
    """nGPT puts a rotated query on the unit sphere times its gain squared, so the temperature scales
    it after that: a tuned nGPT layer's row at place p is the untuned layer's row p with its query gain
    raised by the square root of p's temperature (each row reads its own query alone), on par against
    float64 (check 1)."""
    from dew.nn.mixers.attention import temperature_scale

    tuned, _ = ngpt_layers()
    length = 12

    def draw(seed):
        hidden = np.random.default_rng([1615, seed]).standard_normal((2, length, HIDDEN)).astype(np.float32)
        params = jax.tree.map(np.asarray, tuned.init(jax.random.key(seed), jnp.zeros((1, length, HIDDEN))))
        return hidden, params

    def regained(dtype):
        _, untuned = ngpt_layers(dtype)

        def rows(hidden, params):
            wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params)
            temperature = temperature_scale(jnp.arange(length), TUNED_FLOOR, TUNED_SCALE, dtype=dtype)
            picked = []
            for place in range(length):
                gained = {**wide, "params": {**wide["params"],
                                             "sqk": wide["params"]["sqk"] * jnp.sqrt(temperature[place])}}
                picked.append(untuned.apply(gained, jnp.asarray(hidden, dtype))[:, place])
            return jnp.stack(picked, axis=1)
        return rows

    ours = jax.jit(lambda hidden, params: tuned.apply(params, hidden))
    assert_as_exact(ours, jax.jit(regained(jnp.float32)), draw, "tuned nGPT layer", seeds=TUNED_SEEDS,
                    truth_fn=regained(jnp.float64))
