"""Multi-head latent attention on par with transformers 5.16.1, run live.

The blocks are tools/mla_reference.py's: tiny DeepseekV3Attention blocks with
and without the query LoRA (YaRN rope at the released spelling, interleaved
pairs) and a tiny DeepseekV32Attention with its DSA indexer and biased
projections, their weights the fixtures' (tests/fixtures/mla). Dew's block, its
input gradient and its decode are on par with the reference's float32 from the
reference run in float64 over drawn hidden states (check 1,
tests/reference_error.py); where two of Dew's paths compute one block (the
fused kernels against the reference one) they are on par with the reference
path from its float64 run, every parameter's gradient among them.

The V3.2 indexer's scores are relu'd sums, which tie at exactly zero at the
top-k boundary on most draws; `torch.topk` promises no order among equal
scores, so the reference arm and its float64 truth select by the order JAX
documents for `jax.lax.top_k`, the lower index first (`stable_top_k`), one of
the behaviours the reference permits, which changes no selection a score
decides. A draw keeps only inputs whose decided selections the reference's
own scores separate by more than MARGIN (`boundary_gap`), so float32 and
float64 select alike. The last section trains the V3.2 stack, mixer and
routed experts together, on the simulated mesh.
"""

import contextlib
import functools
import json
import math
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from interop_support import fixture_arrays
from jax.sharding import PartitionSpec as P
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, torch_with_gradients, with_gradients
from selection_margins import indexer_margins
from tool_support import load

from dew.data.dataset import Loader
from dew.interop.decoder_parts import yarn_record
from dew.interop.hf_decoders import translate_weights
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import MixerCall
from dew.nn.kernels import generation
from dew.nn.kv_cache import KVCache
from dew.nn.mla import MultiHeadLatentAttention
from dew.nn.precision import FP32_SUMS
from dew.nn.rope import (
    LongRopeScaling,
    RopeScaling,
    YarnScaling,
    rotary_freqs,
    yarn_attention_factor,
    yarn_inv_freq,
    yarn_query_scale,
)
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
from dew.training import Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mla"
CONFIG = json.loads((FIXTURES / "config.json").read_text())
GENERATOR = load("mla_reference")
"""tools/mla_reference.py, whose configs and transformers classes build the reference blocks."""

SEEDS = {"v3 block": 16, "v3n block": 32, "v32 block": 8, "auto": 64, "decode": 8, "value columns": 8,
         "yarn": 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada: the most any of
a site's cases took."""

LENGTH = 28
"""The tokens of a drawn row: four times the fixtures' seven, so a draw's input gradient sums over
enough positions for the reference compared with itself to resolve (at seven, the block without the
query LoRA did not within 512 seeds)."""

MARGIN = 1e-3
"""The least selection margin a V3.2 draw admits (`boundary_gap`): four orders above the scores' fp32
rounding, so float64 and fp32 select the same keys."""


def fixture(name: str) -> dict:
    return fixture_arrays(FIXTURES / f"{name}.npz")


def yarn_of(settings: dict) -> YarnScaling:
    """The fixture's rope spelling as the mixer's yarn record."""
    return YarnScaling(**yarn_record(settings["rope_scaling"], "rope_scaling",
                                     int(settings["max_position_embeddings"])))


def mla_module(settings: dict, max_seq_len: int = 64,
               attention_impl: str = "reference", kv_cache: KVCache | None = None
               ) -> MultiHeadLatentAttention:
    """The fixture's reference config as a dew mixer."""
    return MultiHeadLatentAttention(
        emb_features=settings["hidden_size"],
        num_heads=settings["num_attention_heads"],
        max_seq_len=max_seq_len,
        q_lora_rank=settings["q_lora_rank"],
        kv_lora_rank=settings["kv_lora_rank"],
        qk_nope_head_dim=settings["qk_nope_head_dim"],
        qk_rope_head_dim=settings["qk_rope_head_dim"],
        v_head_dim=settings["v_head_dim"],
        rope_theta=float(settings["rope_theta"]),
        rope_interleave=settings.get("rope_interleave", True),
        rope_scaling=yarn_of(settings),
        norm_eps=float(settings["rms_norm_eps"]),
        attention_bias=bool(settings["attention_bias"]),
        index_topk=settings.get("index_topk"),
        index_n_heads=settings.get("index_n_heads"),
        index_head_dim=settings.get("index_head_dim"),
        attention_impl=attention_impl,
        kv_cache=KVCache() if kv_cache is None else kv_cache)


def block_variables(tensors: dict) -> dict:
    """Block-relative fixture tensors as the mixer's parameter tree.

    The fixtures hold one attention block's tensors under their torch
    names; a checkpoint carries them under model.layers.N.self_attn.*,
    so the test prefixes that path and runs the real weight translation.
    """
    names = {f"model.layers.0.self_attn.{name}": tensor
             for name, tensor in tensors.items()
             if name not in ("hidden", "output")}
    tree = translate_weights(names, {"tie_embeddings": False})["params"]
    return {"params": tree["layers_0"]["self_attn"]}


@contextlib.contextmanager
def stable_top_k():
    """`Tensor.topk` choosing among equal scores by the lower index, as `jax.lax.top_k` documents:
    a stable descending sort's first k. `torch.topk` promises no order among ties, so this is one of
    its permitted behaviours; it moves no selection a score decides."""
    def topk(self, k, dim=-1, largest=True, sorted=True):
        assert largest, "the indexer ranks its largest scores"
        order = torch.sort(self, dim=dim, descending=True, stable=True).indices.narrow(dim, 0, k)
        return torch.return_types.topk((self.gather(dim, order), order))

    with patch.object(torch.Tensor, "topk", topk):
        yield


REFERENCE = {"mla_v3": ("V3", "DeepseekV3Config", "DeepseekV3Attention", "DeepseekV3RotaryEmbedding"),
             "mla_v3n": ("V3N", "DeepseekV3Config", "DeepseekV3Attention", "DeepseekV3RotaryEmbedding"),
             "mla_v32": ("V32", "DeepseekV32Config", "DeepseekV32Attention", "rotary32")}
"""Each fixture's reference: its config's fields, config class, block class and rotary class, by
their names in tools/mla_reference.py."""


def reference_block(name: str, wide: bool = False):
    """The fixture's transformers block and rotary embedding, its weights the fixture's; built in
    float64, every float32 pin widened (`float64`), where `wide`."""
    from tools.torch_precision import float64

    fields, config, block, rotary = (getattr(GENERATOR, part) for part in REFERENCE[name])
    weights = {key: torch.from_numpy(value) for key, value in fixture(name).items()
               if key not in ("hidden", "output")}
    with float64() if wide else contextlib.nullcontext():
        settings = config(**fields, attn_implementation="eager")
        built = block(settings, layer_idx=0)
        built.load_state_dict(weights)
        return built.eval(), rotary(settings)


def reference_walk(name: str, wide: bool = False):
    """The reference block as a function of a draw `(hidden, cotangent)`, run in float32, or in
    float64 where `wide` (the truth): its output and the hidden states' gradient under the drawn
    cotangent (`torch_with_gradients`)."""
    from tools.torch_precision import float64

    block, rotary = reference_block(name, wide)

    def forward(states):
        batch, length = states.shape[:2]
        positions = torch.arange(length).unsqueeze(0).expand(batch, length)
        mask = torch.triu(torch.full((length, length), float("-inf"), dtype=states.dtype), diagonal=1)
        with float64() if wide else contextlib.nullcontext(), stable_top_k():
            output, _ = block(states, rotary(states, positions), mask.expand(batch, 1, length, length),
                              position_ids=positions)
        return output
    return torch_with_gradients(forward)


def dew_walk(name: str, settings: dict, attention_impl: str = "reference"):
    """Dew's block on the fixture's weights as a function of a draw: what `reference_walk` returns."""
    module = mla_module(settings, attention_impl=attention_impl)
    variables = block_variables(fixture(name))
    run = jax.jit(with_gradients(lambda states: module.apply(variables, states)))
    return lambda hidden, cotangent: run(jnp.asarray(hidden), jnp.asarray(cotangent))


def boundary_gap(name: str, hidden: np.ndarray) -> float:
    """The smallest margin, over every query, of the reference's top-k (modeling_deepseek_v32.py,
    `DeepseekV32Indexer.forward`, replayed) in its scores' units (`indexer_margins`): the gap between
    its last selected index score and its best rejected one, or at an exact zero tie, how near zero
    a tied key's best head agreement sits. A query that sees no more keys than the budget has no
    margin to keep."""
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import apply_rotary_pos_emb

    block, rotary = reference_block(name)
    indexer = block.indexer
    with torch.no_grad():
        states = torch.from_numpy(hidden)
        batch, length = states.shape[:2]
        positions = torch.arange(length).unsqueeze(0).expand(batch, length)
        cos, sin = rotary(states, positions)
        query = indexer.wq_b(block.q_a_layernorm(block.q_a_proj(states))).view(
            batch, length, indexer.n_heads, indexer.head_dim)
        rope = [indexer.qk_rope_head_dim, indexer.head_dim - indexer.qk_rope_head_dim]
        q_rot, q_pass = torch.split(query, rope, dim=-1)
        k_rot, k_pass = torch.split(indexer.k_norm(indexer.wk(states)).unsqueeze(2), rope, dim=-1)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        query, keys = torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1).squeeze(2)
        agreements = (torch.matmul(query.float(), keys.transpose(-1, -2).float().unsqueeze(1))
                      * indexer.softmax_scale)
        weights = indexer.weights_proj(states).float() * indexer.n_heads ** -0.5
        index = torch.matmul(weights.unsqueeze(-2), torch.relu(agreements)).squeeze(-2)
        seen = torch.arange(length)[None, None, :] <= positions[:, :, None]
    return float(np.min(indexer_margins(index.numpy(), agreements.numpy(), seen.numpy(), indexer.index_topk,
                                        relative=False)))


def block_draw(name: str):
    """Check 1's draw for a block: a seed's hidden states and cotangent, rows LENGTH tokens long, for the
    V3.2 block the first of the seed's candidates its reference selects by more than MARGIN
    (`boundary_gap`)."""
    shape = (CONFIG["batch"], LENGTH, CONFIG["v3"]["hidden_size"])

    def draw(seed):
        for attempt in range(64):
            rng = np.random.default_rng([seed, attempt])
            hidden = rng.standard_normal(shape).astype(np.float32)
            if name != "mla_v32" or boundary_gap(name, hidden) > MARGIN:
                return hidden, rng.standard_normal(shape).astype(np.float32)
        raise AssertionError(f"no candidate of seed {seed} has its selection decided by {MARGIN}")
    return draw


def test_yarn_matches_the_reference_derivation():
    """The released YaRN spelling against transformers' rotary embedding, live: the inverse
    frequencies, the cos/sin amplitude and the query scale exactly (as computed, the same float32 and
    Python arithmetic), and cos and sin at drawn positions on par with the reference's float32 from
    its float64 run (check 1)."""
    from tools.torch_precision import float64

    yarn = yarn_of(CONFIG["v3"])
    _, rotary = reference_block("mla_v3")
    np.testing.assert_array_equal(np.asarray(yarn_inv_freq(8, 10000.0, yarn, dtype=np.float32)),
                                  rotary.inv_freq.numpy())
    assert yarn_attention_factor(yarn) == rotary.attention_scaling
    assert yarn_query_scale(yarn) == (0.1 * math.log(40.0) + 1.0) ** 2
    _, wide = reference_block("mla_v3", wide=True)

    @as_given
    def reference(positions, wide_run=False):
        with float64() if wide_run else contextlib.nullcontext():
            rows = torch.from_numpy(np.asarray(positions)).long()[None]
            cos, sin = (wide if wide_run else rotary)(torch.zeros(1, dtype=torch.float64 if wide_run else
                                                                  torch.float32), rows)
        half = cos.shape[-1] // 2
        return {"cos": cos[0, :, :half].numpy(), "sin": sin[0, :, :half].numpy()}

    def ours(positions):
        cos, sin = rotary_freqs(jnp.asarray(positions), 8, 10000.0, rope_scaling=yarn, dtype=np.float32)
        return {"cos": cos, "sin": sin}

    def draw(seed):
        return (np.random.default_rng(seed).integers(0, 4096 * 40, 64).astype(np.int32),)

    assert_as_exact(ours, reference, draw, "yarn rotary", seeds=SEEDS["yarn"],
                    truth_fn=lambda positions: reference(positions, wide_run=True))


def test_yarn_over_a_partial_rotary_ramps_the_rotated_width_and_refuses_a_proportional_one():
    """YaRN ramps the rotated width: a 'default' partial rotary of 8 of a head's 16 dims rotates as a
    whole head of 8 does, the reference's `dim = head_dim * partial_rotary_factor`, bit for bit, while
    a proportional one, which has no YaRN form, is refused."""
    yarn = yarn_of(CONFIG["v3"])
    positions = jnp.arange(0, 4096 * 40, 997)
    whole = rotary_freqs(positions, 8, 10000.0, rope_scaling=yarn, dtype=np.float32)
    partial = rotary_freqs(positions, 16, 10000.0, rot_dim=8, partial_rotary_type="default",
                           rope_scaling=yarn, dtype=np.float32)
    for ours, theirs in zip(partial, whole, strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    with pytest.raises(ValueError):
        rotary_freqs(positions, 16, 10000.0, rot_dim=8, partial_rotary_type="proportional",
                     rope_scaling=yarn, dtype=np.float32)


def decode_loop(module, variables, tokens):
    """Prefill plus one-token steps, the sampler's decode protocol."""
    state = module.init(jax.random.key(0), tokens[:, :1], MixerCall(decode=True))
    state = {**state, "params": variables["params"]}
    outputs = []
    for position in range(tokens.shape[1]):
        out, mutated = module.apply(
            state, tokens[:, position:position + 1], MixerCall(decode=True), mutable=["cache"])
        state = {**state, "cache": mutated["cache"]}
        outputs.append(out)
    return jnp.concatenate(outputs, axis=1)


SETTINGS = {"mla_v3": "v3", "mla_v3n": "v3n", "mla_v32": "v32"}


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_decode_matches_prefill(name):
    """Prefill plus one-token steps on par with the block's parallel pass from that pass in float64
    (check 1), over drawn hidden states."""
    settings = CONFIG[SETTINGS[name]]
    module = mla_module(settings)
    variables = block_variables(fixture(name))
    full = jax.jit(lambda variables, hidden: module.apply(variables, hidden))

    decoded = jax.jit(lambda variables, hidden: decode_loop(module, variables, hidden))

    draw = block_draw(name)
    assert_as_exact(decoded, full, lambda seed: (variables, draw(seed)[0]), f"{name} decode",
                    seeds=SEEDS["decode"])


@pytest.mark.parametrize("quantized", [None, "int8", "float8_e4m3fn"])
@pytest.mark.parametrize("name", ["mla_v3", "mla_v32"])
def test_a_paged_latent_cache_decodes_what_the_compact_cache_decodes(name, quantized):
    """A latent attention's cache takes the layer's layout: paged, it decodes every step as the compact
    cache does at the same format, bit for bit, full precision or quantized, over the latents (V3) and
    over the expanded keys the indexer selects among (V3.2), the quantized keys read back in the basis
    their queries or their expansion take."""
    settings = CONFIG[SETTINGS[name]]
    variables = block_variables(fixture(name))
    hidden = block_draw(name)(0)[0]

    def decoded(layout):
        module = mla_module(settings, kv_cache=layout)
        return np.asarray(jax.jit(lambda variables, hidden: decode_loop(module, variables, hidden))(
            variables, hidden))

    np.testing.assert_array_equal(decoded(KVCache(quantized=quantized, page_size=4)),
                                  decoded(KVCache(quantized=quantized)))


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n"])
def test_mla_reproduces_the_v3_block(name):
    """DeepseekV3Attention, with the query LoRA and without (the plain-q_proj variant no released
    checkpoint uses): q LoRA, compressed KV, interleaved YaRN rope. The output and the hidden states'
    gradient on par with the reference's (check 1)."""
    assert_as_exact(dew_walk(name, CONFIG[SETTINGS[name]]), reference_walk(name), block_draw(name),
                    f"{name} block", seeds=SEEDS[f"{SETTINGS[name]} block"],
                    truth_fn=reference_walk(name, wide=True))


def test_mla_reproduces_the_v32_block():
    """DeepseekV32Attention: biased projections, the indexer's top-k folded into the mask. The output
    and the hidden states' gradient on par with the reference's (check 1), and the selection matters:
    the dense mixer on the same weights moves the output past rounding (`assert_moves`).

    A fixture once held the dense block, not the sparse one: the reference folds the indexer's top-k
    into the mask only when its config names the eager or sdpa path, and a standalone
    DeepseekV32Config leaves `_attn_implementation` None. The reference here names the eager path."""
    from reference_error import assert_moves

    dense = dict(CONFIG["v32"], index_topk=None, index_n_heads=None, index_head_dim=None)
    draw, truth = block_draw("mla_v32"), reference_walk("mla_v32", wide=True)
    assert_as_exact(dew_walk("mla_v32", CONFIG["v32"]), reference_walk("mla_v32"), draw, "mla_v32 block",
                    seeds=SEEDS["v32 block"], truth_fn=truth)
    assert_moves(dew_walk("mla_v32", CONFIG["v32"]), dew_walk("mla_v32", dense), draw, "dense mla_v32",
                 seeds=SEEDS["v32 block"], truth_fn=truth)


# --------------------------------------------------------------------------
# The kernel the head widths select
# --------------------------------------------------------------------------


def walked(module):
    """`module`'s output and the gradients of its every parameter and the hidden states under the
    drawn cotangent, as a function of a draw `(variables, hidden, cotangent)`, compiled."""
    return jax.jit(with_gradients(module.apply))


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_auto_takes_the_unequal_query_and_value_head_widths(name):
    """`attention_impl='auto'` computes the block on the shape it used to refuse, on par with the
    reference path, its output and every gradient, the parameters' among them, from the reference
    path's float64 run (check 1); a crop that kept the padded columns instead of the value's own moves
    them by their own size.

    MLA's values are narrower than its queries (`v_head_dim` 8 against
    `qk_nope_head_dim + qk_rope_head_dim` 16 in these fixtures, 128 against
    192 in every released V3 config), and `jax.nn.dot_product_attention`
    checks the value against the key's whole shape, so both fused paths
    raised before the value was padded to the query's width. On CPU `auto`
    resolves to the xla kernel.
    """
    settings = CONFIG[SETTINGS[name]]
    variables, draw = block_variables(fixture(name)), block_draw(name)
    assert_as_exact(walked(mla_module(settings, attention_impl="auto")), walked(mla_module(settings)),
                    lambda seed: (variables, *draw(seed)), f"{name} auto", seeds=SEEDS["auto"])


def test_the_fused_path_returns_the_value_columns_and_not_the_padding():
    """A value narrower than the query rides into the fused kernel padded to
    the query's width, so the call has to hand back the value's own columns:
    the same numbers as the call where the caller writes that padding out,
    and on par with the reference kernel from its float64 run (check 1). A
    value wider than the query has no such rewrite, since padding the query
    would move the kernel's own 1/sqrt(d) scale, and is refused.
    """
    def draw(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal(shape), jnp.float32)
                     for shape in ((2, 7, 4, 16), (2, 7, 4, 16), (2, 7, 4, 8)))

    # A CPU's calls take the reference path, whose float64 sums and exact softmax derivative the
    # fused kernel's are not; routed as an A100's, the call runs the fused kernel this test is about.
    query, key, value = draw(0)
    fused = functools.partial(scaled_dot_product_attention, query, key, implementation="xla", causal=True)
    with generation.assumed("sm80"):
        narrow = fused(value)
        assert narrow.shape == (2, 7, 4, 8)
        assert jnp.array_equal(
            narrow, fused(jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, 8))))[..., :8])
        with pytest.raises(ValueError):
            fused(jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, 16))))
        xla, reference = (functools.partial(scaled_dot_product_attention, implementation=name, causal=True)
                          for name in ("xla", "reference"))

        def summed(*arrays):
            # The fused kernel's products sum as an A100's do, so the reference states those sums.
            with jax.default_matmul_precision(FP32_SUMS):
                return reference(*arrays)

        assert_as_exact(jax.jit(xla), jax.jit(summed), draw, "narrow value columns",
                        seeds=SEEDS["value columns"])


def mla_record(settings: dict) -> dict:
    """The fixture's reference config as the `mla` kind's record."""
    return {
        "class": "mla", "fields": {
        "q_lora_rank": settings["q_lora_rank"],
        "kv_lora_rank": settings["kv_lora_rank"],
        "qk_nope_head_dim": settings["qk_nope_head_dim"],
        "qk_rope_head_dim": settings["qk_rope_head_dim"],
        "v_head_dim": settings["v_head_dim"],
        "rope_interleave": settings.get("rope_interleave", True),
        "index_topk": settings.get("index_topk"),
        "index_n_heads": settings.get("index_n_heads"),
        "index_head_dim": settings.get("index_head_dim"),
    }}


def mla_model(settings: dict, **overrides) -> CausalTransformer:
    """A one-layer decoder whose mixer is the fixture's block."""
    fields = {
        "vocab_size": 37, "emb_features": settings["hidden_size"], "num_layers": 1,
        "num_heads": settings["num_attention_heads"],
        "head_dim": settings["qk_nope_head_dim"] + settings["qk_rope_head_dim"],
        "mlp_features": 48, "max_seq_len": 64, "rope_theta": float(settings["rope_theta"]),
        "yarn": yarn_record(settings["rope_scaling"], "rope_scaling",
                            int(settings["max_position_embeddings"])),
        "norm_eps": float(settings["rms_norm_eps"]), "qk_norm": False,
        "attention_bias": bool(settings["attention_bias"]),
        "mixer": mla_record(settings), "attention_impl": "reference"}
    return CausalTransformer(**{**fields, **overrides})


@pytest.mark.parametrize("name", ["mla_v3", "mla_v32"])
def test_a_decoder_on_the_mla_kind_runs_the_reference_block(name):
    """The kind lands the block at layers_0/self_attn with the translated
    tree's names, and there it computes what the standalone block does on
    the same weights, bit for bit, so it is on par with the reference where
    the standalone block is (`test_mla_reproduces_the_v3_block`,
    `test_mla_reproduces_the_v32_block`).
    """
    settings = CONFIG[SETTINGS[name]]
    tensors = fixture(name)
    model = mla_model(settings)
    tokens = jnp.zeros((2, 7), jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    translated = block_variables(tensors)["params"]
    layer = variables["params"]["layers_0"]
    assert jax.tree_util.tree_structure(layer["self_attn"]) == (
        jax.tree_util.tree_structure(translated))
    variables = {**variables, "params": {
        **variables["params"],
        "layers_0": {**layer, "self_attn": translated}}}
    hidden = jnp.asarray(tensors["hidden"])
    inside = model.apply(
        variables, hidden, method=lambda m, x: m.layers[0].self_attn(x))
    standalone = mla_module(settings).apply(block_variables(tensors), hidden)
    assert jnp.array_equal(inside, standalone)
    logits = model.apply(variables, jnp.arange(14, dtype=jnp.int32).reshape(2, 7))
    assert logits.shape == (2, 7, 37)


@pytest.mark.parametrize("scaling", [
    RopeScaling(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=4),
    LongRopeScaling(short_factor=(1.0, 1.5, 2.0, 2.5), long_factor=(3.0, 4.0, 5.0, 6.0),
                    original_max_position_embeddings=4)], ids=["llama3", "longrope"])
def test_the_mla_kind_rotates_by_the_models_rope_scaling(scaling):
    """A model's Llama 3.1 or LongRoPE scaling reaches its latent attention as
    it reaches its ordinary attention (`MixerContext.rotary_scaling`): the
    kind's layer computes, bit for bit, what the standalone block given that
    scaling computes, where a kind that forwarded YaRN alone rotated at the
    plain frequencies (audit NN-04). The seven positions pass LongRoPE's four,
    so it reads its long factors."""
    settings = CONFIG["v3"]
    tensors = fixture("mla_v3")
    model = mla_model(settings, yarn=None, rope_scaling=scaling)
    variables = model.init(jax.random.key(0), jnp.zeros((2, 7), jnp.int32))
    layer = variables["params"]["layers_0"]
    variables = {**variables, "params": {
        **variables["params"], "layers_0": {**layer, "self_attn": block_variables(tensors)["params"]}}}
    hidden = jnp.asarray(tensors["hidden"])
    inside = model.apply(variables, hidden, method=lambda m, x: m.layers[0].self_attn(x))
    standalone = mla_module(settings).clone(rope_scaling=scaling).apply(block_variables(tensors), hidden)
    np.testing.assert_array_equal(inside, standalone)


# --------------------------------------------------------------------------
# The stack on the mesh
# --------------------------------------------------------------------------

TINY_SHARD = 256
"""The architecture sweep's threshold: these models hold thousands of
parameters, orders below the production shard threshold."""


def deepseek_stack() -> CausalTransformer:
    """DeepSeek V3.2 at toy width: the sparse mla mixer on both layers, the
    second layer routed with the balancing bias, the group limit and a
    shared expert; the architecture sweep trains the same case."""
    return CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
        head_dim=16, mlp_features=64, max_seq_len=16,
        mixer={"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
               "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
               "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16}},
        mixture={"experts": 8, "top_k": 2, "layers": (1,),
                 "score_function": "sigmoid", "groups": 4, "groups_per_token": 2,
                 "bias": True, "expert_features": 16, "shared_features": 16})


def token_batches():
    rng = np.random.default_rng(0)
    while True:
        yield {"text": rng.integers(0, 64, size=(8, 17)).astype(np.int32)}


@pytest.mark.mesh
@pytest.mark.parametrize("fsdp,expert", [(2, 1), (2, 2)])
def test_the_deepseek_stack_fits_on_the_mesh(fsdp, expert):
    """Two steps through the trainer on the simulated eight devices, with
    the parameters two ways over fsdp and, in the second case, the experts
    two ways over the expert axis as well. The loss is finite and falls, the
    balancing bias moves, and the layout check passes: every rank two or
    more parameter of the mixer, the indexer and the experts is placed by
    its declaration, so nothing above the shard threshold stays replicated.
    Expert parallelism alone leaves the mixer replicated by design, which
    test_moe covers, so the expert case rides an fsdp axis."""
    model = deepseek_stack()
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(model, 16, balance=ExpertBalance(balance_rate=0.01)), optax.adam(1e-3),
        key=jax.random.key(0), mesh=MeshSpec(fsdp=fsdp, expert=expert),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)

    state = trainer.fit(Loader(lambda partition: token_batches(), None, batch=8, records=None),
                        steps=2, log_every=1)

    assert dict(trainer.device_mesh.shape) == {
        "data": 8 // (fsdp * expert), "expert": expert, "fsdp": fsdp,
        "tensor": 1, "sequence": 1, "stage": 1}
    losses = [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert losses[1] < losses[0], losses
    bias = state.variables["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]
    assert np.any(np.asarray(bias) != 0), "the bias never moved"
    attention = state.variables["params"]["layers_0"]["self_attn"]
    experts = state.variables["params"]["layers_1"]["mlp"]["experts"]
    assert attention["kv_b_proj"]["kernel"].sharding.spec == P("fsdp")
    assert attention["indexer"]["wq_b"]["kernel"].sharding.spec == P("fsdp")
    assert experts["gate_proj"]["kernel"].sharding.spec == P("expert" if expert > 1 else None, "fsdp")
    abstract = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 16), jnp.int32))
    shardings = trainer.layout.shardings(trainer.device_mesh, abstract)
    trainer.layout.check(abstract["params"], shardings["params"], trainer.device_mesh)
