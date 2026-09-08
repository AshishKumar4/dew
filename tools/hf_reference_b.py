#!/usr/bin/env python3
"""Fixtures for the decoder families that needed new primitives.

Runs in the same venv as tools/hf_reference.py and reuses its writers:
`write_tiny` for a random-weight checkpoint with the reference's fp32
logits, `write_released_config` for a released config.json alone.

    PYTHONPATH=src .venv/bin/python tools/hf_reference_b.py

What lands in tests/fixtures/hf:

- gpt-oss-tiny/: two alternating layers with attention sinks, four fused
  biased experts per layer and YaRN over grouped-query heads (the tiny
  config is tools/gpt_oss_reference.py's).
- deepseek-v2-tiny/: DeepSeek V2 Lite at toy width, softmax routing under
  group_limited_greedy with no renormalisation, the shared expert and a
  dense first layer, MLA without the query LoRA.
- kimi-k2-tiny/: Kimi K2 at toy width under its own model_type, released
  rope spelling and provenance in source.json. Its config carries the
  release's routing (twelve experts in one group, sigmoid scores under
  noaux_tc with the balancing bias, one shared expert, routed_scaling_factor
  2.827), its rope (theta 50000 with YaRN factor 32 at beta_fast and
  beta_slow 1.0) and its MLA widths (qk_nope twice qk_rope, values as wide
  as qk_nope), not DeepSeek V3's. Its weights are the released per-expert
  tensor names transformers reads through V3's conversion.
- glm4-moe-tiny/: GLM 4.5 at toy width, biased q/k/v over a bias-free
  o_proj, the q/k norms of GLM 4.6, a half rotary, a dense first layer over
  a routed one with the balancing bias and a shared expert, scaled by 1.5,
  and one MTP depth under the released names (model.layers.2.*). transformers
  builds no depth, so its logits (mtp_logits.npy) come from the depth's own
  tensors composed the way the engines that run the released weights do:
  eh_proj over enorm(embeddings) and hnorm(hidden) in that order, one
  Glm4MoeDecoderLayer, shared_head.norm, then the trunk's head.
- glm-moe-dsa-tiny/: GLM-5.3 at toy width, DeepSeek V3.2's sparse MLA with
  the indexer rotating interleaved pairs, three dense layers over routed
  ones with the balancing bias and a shared expert, IndexShare's
  full/shared schedule (index_topk_freq 4 off index_skip_topk_offset 3)
  over eight layers, and one MTP depth (model.layers.8.*) with its own
  indexer, composed as glm4-moe-tiny's is (mtp_logits.npy).
- llama4-tiny/: Llama 4 at toy width, three chunked local layers with the
  interleaved rope and the L2 q/k norm around one global layer with
  temperature tuning, every other layer routed with the shared expert.
- gemma3n-tiny/: Gemma 3n at toy width, three copies of the residual
  stream under AltUp, the LAuReL block, gaussian top-k sparsity on the
  first two layers, feed-forward widths that differ per layer, per-layer
  inputs and the last layer sharing the second's keys and values.
- gemma4-moe-tiny/: Gemma 4's 26B-A4B shape at toy width, the routed
  branch beside every layer's dense MLP under Gemma4TextRouter, global
  layers reading their values off the keys with fewer key/value heads and
  a wider head, and the per-layer output scalar.
- gpt-oss-20b/, deepseek-v2-lite/, kimi-k2/, glm-4.5-air/, glm-5/, glm-5.3/,
  llama-4-scout/, gemma4-26b-a4b/: released configs only. Llama-4-Scout and
  gemma-4-26B-A4B are gated, so their configs come from mirrors and drop
  the mirror's own marker key.

The gemma4-ple, gemma4-kvshare and gemma4-e2b fixtures predate the
persistent layer_scalar buffer transformers 5.16.1 saves, so
add_layer_scalars writes the ones the reference initialises into them; the
logits are unchanged.

What lands in tests/fixtures/llama4, the block-level references the
primitive tests read before the family loads:

- attention.npz: one `Llama4TextAttention` as a local layer (rope, L2
  norm, chunk 4) and as a global layer (no rope, temperature tuning at
  floor_scale 4) on the same random weights and hidden states.
- moe.npz: one `Llama4TextMoe` on random weights, its output and the
  per-expert tensors the checkpoint layout carries fused.

What lands in tests/fixtures/gemma3n, for tests/test_gemma3n.py:

- blocks.npz: one `Gemma3nTextAltUp` predicting and correcting a random
  stream of four copies, one `Gemma3nTextLaurelBlock`, and one
  `Gemma3nTextMLP` at sparsity 0.95, each on random weights with its
  inputs and outputs.

What lands in tests/fixtures/gemma4, for tests/test_gemma4_moe.py:

- moe.npz: one `Gemma4TextDecoderLayer` feed-forward half on random
  weights: the residual it reads, the dense MLP's output, the router's
  weights and choices, and the summed branch output before the block's
  post_feedforward_layernorm.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.numpy import load_file, save_file
from transformers import (
    DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM,
    Gemma3nTextConfig, Gemma4TextConfig, Glm4MoeConfig, Glm4MoeForCausalLM,
    GlmMoeDsaConfig, GlmMoeDsaForCausalLM, Llama4TextConfig,
    Qwen3NextConfig, Qwen3NextForCausalLM,
)
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM, Gemma3nTextAltUp, Gemma3nTextLaurelBlock, Gemma3nTextMLP,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextDecoderLayer
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer, Glm4MoeRMSNorm
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer, GlmMoeDsaRMSNorm,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM, Llama4TextAttention, Llama4TextMoe, Llama4TextRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt_oss_reference import tiny_gpt_oss  # noqa: E402
from hf_reference import (  # noqa: E402
    DEEPSEEK_YARN, FIXTURES, reference_logits, scatter_weights, write_released_config,
    write_tiny,
)
from moe_reference import expert_tensors  # noqa: E402
from qwen_mtp_reference import QwenMTP  # noqa: E402


def tiny_deepseek_v2() -> DeepseekV2ForCausalLM:
    """V2 Lite's shape at toy width: q_lora_rank None, group_limited_greedy
    over four groups of two experts keeping two groups, which is exactly the
    four experts a token takes, so the group limit decides the choice."""
    config = DeepseekV2Config.from_dict(dict(
        vocab_size=256, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, n_shared_experts=2, n_routed_experts=8,
        routed_scaling_factor=2.5, q_lora_rank=None, kv_lora_rank=8,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_group=4,
        topk_group=2, num_experts_per_tok=4, first_k_dense_replace=1,
        topk_method="group_limited_greedy", norm_topk_prob=False,
        scoring_func="softmax", hidden_act="silu", max_position_embeddings=64,
        rms_norm_eps=1e-6, tie_word_embeddings=False, rope_theta=10000.0,
        rope_scaling={**DEEPSEEK_YARN, "mscale": 0.707, "mscale_all_dim": 0.707},
        attention_bias=False, aux_loss_alpha=0.001, seq_aux=True))
    torch.manual_seed(0)
    return DeepseekV2ForCausalLM(config)


# moonshotai/Kimi-K2-Instruct at revision fd1984e2, scaled to a fixture.
# The release's proportions this rounds to a hidden width of 32:
# intermediate_size 2.571x that width (80 here), moe_intermediate_size
# 0.286x (8), the query LoRA 0.214x (8), and qk_nope twice qk_rope with the
# values as wide as qk_nope, which is the one proportion that survives
# exactly. The 0.071x latent would round to 2, too narrow to carry the
# compressed keys and values, so kv_lora_rank stays at the 8 the other MLA
# fixtures use.
# The release's own values, unscaled: routed_scaling_factor 2.827,
# rope_theta 50000, YaRN factor 32 with beta_fast and beta_slow both 1.0
# over original_max_position_embeddings 4096, one group holding every
# expert, one dense layer, one shared expert, no prediction depth, and
# bos/eos at the release's own offsets below the top of the vocabulary
# (163584 and 163585 of 163840). Kimi's 384 experts and the 48:1 ratio they
# keep against num_experts_per_tok do not survive a fixture; twelve experts
# with two per token keep a routed layer whose choice the group limit
# cannot decide, which is what n_group 1 means.
KIMI_K2_YARN = {
    "type": "yarn", "factor": 32.0, "beta_fast": 1.0, "beta_slow": 1.0,
    "mscale": 1.0, "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}
KIMI_K2_TINY = dict(
    vocab_size=256, hidden_size=32, intermediate_size=80,
    moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=12,
    routed_scaling_factor=2.827, q_lora_rank=8, kv_lora_rank=8,
    qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, n_group=1,
    topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1,
    moe_layer_freq=1, norm_topk_prob=True, scoring_func="sigmoid",
    topk_method="noaux_tc", num_nextn_predict_layers=0, hidden_act="silu",
    max_position_embeddings=64, rms_norm_eps=1e-6, tie_word_embeddings=False,
    rope_theta=50000.0, rope_scaling=dict(KIMI_K2_YARN), attention_bias=False,
    attention_dropout=0.0, aux_loss_alpha=0.001, seq_aux=True,
    pretraining_tp=1, bos_token_id=0, eos_token_id=1,
)


def tiny_kimi_k2() -> DeepseekV3ForCausalLM:
    """Kimi K2's shape at toy width, built by the class its release names.

    The config keeps transformers' own model_type here so that
    `save_pretrained` reverses its per-expert conversion and writes the
    released tensor names; `write_kimi_k2_config` puts Kimi's model_type and
    its released rope spelling back over the saved config.

    beta_fast and beta_slow both 1.0 place the YaRN correction range on one
    frequency of the eight-wide rope slice, so three of its four frequencies
    extrapolate and the fourth interpolates by the factor. A fixture whose
    ramp came out all one way would agree with an implementation that
    dropped either branch.

    The rope entry is copied per call because config standardization writes
    `rope_type` and `rope_theta` into the entry it is handed, and the config
    written beside the weights is the release's own seven fields.
    """
    config = DeepseekV3Config.from_dict(
        dict(KIMI_K2_TINY, rope_scaling=dict(KIMI_K2_YARN), rope_interleave=True))
    torch.manual_seed(0)
    return DeepseekV3ForCausalLM(config)


def write_kimi_k2_config(name: str, repo: str) -> None:
    """The fixture's config as Kimi K2 releases one, and its provenance.

    The release ships `model_type: kimi_k2` with `architectures:
    [DeepseekV3ForCausalLM]`, `rope_theta` beside a `rope_scaling` of
    `type: yarn`, and the training fields transformers ignores. It also
    ships an `auto_map` onto its own `modeling_deepseek.py` and an fp8
    `quantization_config`; neither is written here, since the fixture
    carries no remote code and its weights are floating.

    transformers 5.16.1 registers no `kimi_k2` config, so the reload below
    names the class the release's auto_map does and registers V3's own
    per-expert conversion under Kimi's model_type. It fails the fixture if
    the released spelling reaches other logits than the saved config did.
    """
    from huggingface_hub import model_info
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping,
    )

    directory = FIXTURES / name
    saved = json.loads((directory / "config.json").read_text())
    for field in ("head_dim", "qk_head_dim", "rope_parameters", "rope_interleave"):
        saved.pop(field, None)
    config = {"architectures": ["DeepseekV3ForCausalLM"], **saved, **KIMI_K2_TINY,
              "model_type": "kimi_k2", "transformers_version": transformers.__version__}
    (directory / "config.json").write_text(
        json.dumps(dict(sorted(config.items())), indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({
        "released": {"repo": repo, "revision": model_info(repo).sha},
        "transformers": {"version": transformers.__version__,
                         "revision": "93c8b7b485963a10800c91f55304db6be211c2bd"},
    }, indent=1) + "\n")

    register_checkpoint_conversion_mapping(
        "kimi_k2", get_checkpoint_conversion_mapping("deepseek_v3"), overwrite=True)
    model, report = DeepseekV3ForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    unread = {key: sorted(report[key]) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if report[key]}
    if unread:
        raise SystemExit(f"{directory}: the released config does not read its own weights: {unread}")
    ids = np.load(directory / "input_ids.npy")
    difference = float(np.max(np.abs(reference_logits(model, ids)
                                     - np.load(directory / "logits.npy"))))
    if difference != 0.0:
        raise SystemExit(f"{directory}: the released rope spelling moved the logits by {difference:.3e}")
    print(f"{directory}: model_type {config['model_type']}, {len(config)} fields, "
          f"released spelling reloads bit for bit")


def tiny_glm4_moe() -> Glm4MoeForCausalLM:
    config = Glm4MoeConfig.from_dict(dict(
        vocab_size=256, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        partial_rotary_factor=0.5, use_qk_norm=True, attention_bias=True,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1, n_group=1,
        topk_group=1, routed_scaling_factor=1.5, norm_topk_prob=True,
        first_k_dense_replace=1, num_nextn_predict_layers=1, max_position_embeddings=64,
        rope_theta=1e6, rms_norm_eps=1e-5, tie_word_embeddings=False))
    torch.manual_seed(0)
    return Glm4MoeForCausalLM(config)


class GlmMTP(torch.nn.Module):
    """One GLM MTP depth as vLLM's Glm4MoeMultiTokenPredictorLayer composes
    it, over the family's own decoder block and norm."""

    def __init__(self, config, norm: type[torch.nn.Module], block: torch.nn.Module) -> None:
        super().__init__()
        self.enorm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = torch.nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = block
        self.shared_head_norm = norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, model, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        fused = self.eh_proj(torch.cat(
            [self.enorm(model.model.embed_tokens(ids)), self.hnorm(hidden)], dim=-1))
        positions = torch.arange(fused.shape[1])[None]
        mask = create_causal_mask(config=model.config, inputs_embeds=fused, attention_mask=None,
                                  past_key_values=None, position_ids=positions)
        out = self.block(fused, attention_mask=mask, position_ids=positions,
                         position_embeddings=model.model.rotary_emb(fused, positions))
        # GlmMoeDsaDecoderLayer hands its top-k indices up beside the states.
        if isinstance(out, tuple):
            out = out[0]
        return model.lm_head(self.shared_head_norm(out))


def glm4_moe_mtp(config: Glm4MoeConfig) -> GlmMTP:
    return GlmMTP(config, Glm4MoeRMSNorm, Glm4MoeDecoderLayer(config, layer_idx=config.num_hidden_layers))


def glm_moe_dsa_mtp(config: GlmMoeDsaConfig) -> GlmMTP:
    """The depth's block reads its own indexer mode and MLP kind at its
    index, past the trunk's lists: the released depth ships indexer weights
    (GLM-5.3's model.layers.78.self_attn.indexer.*) and routes. The strict
    config holds the lists to num_hidden_layers, so the block is built from
    a copy one layer deeper."""
    depth = config.num_hidden_layers
    # __post_init__ fills the three lists; the strict validator refuses a
    # short one, so an unset list fails loudly here rather than silently.
    extended = GlmMoeDsaConfig.from_dict(dict(
        config.to_dict(), num_hidden_layers=depth + 1,
        indexer_types=[*(config.indexer_types or ()), "full"],
        mlp_layer_types=[*(config.mlp_layer_types or ()), "sparse"],
        layer_types=[*(config.layer_types or ()), "deepseek_sparse_attention"]))
    extended._attn_implementation = "eager"
    return GlmMTP(config, GlmMoeDsaRMSNorm, GlmMoeDsaDecoderLayer(extended, layer_idx=depth))


def write_glm_mtp(name: str, model, depth: GlmMTP, seed: int = 2026) -> None:
    """The depth's tensors into the fixture checkpoint, and its reference logits."""
    directory = FIXTURES / name
    depth = depth.eval()
    scatter_weights(depth, seed)
    # The layer's submodules sit behind a class decorator that hides them
    # from a checker, so the routed block's parts are fetched by their paths.
    experts = depth.block.get_submodule("mlp.experts")
    bias = depth.block.get_buffer("mlp.gate.e_score_correction_bias")
    ids = torch.from_numpy(np.load(directory / "input_ids.npy").astype(np.int64))
    with torch.no_grad():
        bias.copy_(torch.linspace(-0.4, 0.4, model.config.n_routed_experts))
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        logits = depth(model, hidden[:, :-1], ids[:, 1:])
    prefix = f"model.layers.{model.config.num_hidden_layers}."
    tensors = load_file(str(directory / "model.safetensors"))
    for tensor_name, tensor in depth.state_dict().items():
        if tensor_name.startswith("block.mlp.experts."):
            continue
        released = tensor_name.replace("block.", "").replace("shared_head_norm", "shared_head.norm")
        tensors[prefix + released] = tensor.to(torch.float32).numpy()
    for tensor_name, tensor in expert_tensors(experts).items():
        tensors[prefix + tensor_name] = tensor
    tensors[prefix + "embed_tokens.weight"] = tensors["model.embed_tokens.weight"]
    tensors[prefix + "shared_head.head.weight"] = tensors["lm_head.weight"]
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    np.save(directory / "mtp_logits.npy", logits.to(torch.float32).numpy())
    print(f"{directory}: depth {prefix}* with {len(tensors)} tensors, "
          f"mtp logits {tuple(logits.shape)}")


# zai-org/GLM-5.3 at toy width. The release's proportions on a hidden width
# of 32: intermediate_size twice the width (64), moe_intermediate_size and
# the query LoRA a third of it (12 each), qk_nope three times qk_rope with
# the values four times it (12, 4, 16), the indexer head twice the rope
# width (8). The 1/12 latent would round to 3, too narrow to carry the
# compressed keys and values, so kv_lora_rank stays at the 8 the other MLA
# fixtures use. The release's own values: routed_scaling_factor 2.5,
# rope_theta 8e6 under rope_parameters, one group holding every expert,
# three dense layers, one shared expert, one prediction depth, and the
# stale head_dim the release ships at qk_nope_head_dim, which the config
# points back at the rope slice. 256 experts with 8 per token become 8
# with 2 per token. IndexShare keeps the release's schedule, index_topk_freq
# 4 from index_skip_topk_offset 3, which over eight layers reads
# full, full, full, shared, shared, shared, full, shared: the last layer
# shares the seventh's top-k and not the third's, so a reader that took
# the first full layer's selection would disagree. The depth is a full
# layer, as the release's model.layers.78.self_attn.indexer.* says.
#
# The release's indexer has half as many heads as the attention, which
# would be two here; the relu zeroes a key's score whenever every head's
# agreement is negative, and at two heads no seed in 400 kept the fourth
# and fifth scores apart on every row of the five indexer layers (the
# trunk's four full layers and the depth). Four heads with seed 3850 keep
# them at least 0.0149 apart everywhere, so the fixture's top-k is one
# selection and not a tie torch and jax break differently.
GLM_MOE_DSA_SEED = 3850
GLM_MOE_DSA_TINY = dict(
    vocab_size=256, hidden_size=32, intermediate_size=64, moe_intermediate_size=12,
    num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4,
    n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5,
    kv_lora_rank=8, q_lora_rank=12, qk_rope_head_dim=4, v_head_dim=16,
    qk_nope_head_dim=12, n_group=1, topk_group=1, num_experts_per_tok=2,
    norm_topk_prob=True, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-5, tie_word_embeddings=False,
    rope_parameters={"rope_theta": 8000000.0, "rope_type": "default"},
    attention_bias=False, attention_dropout=0.0, index_topk=4, index_head_dim=8,
    index_n_heads=4, head_dim=12, first_k_dense_replace=3,
    num_nextn_predict_layers=1, indexer_rope_interleave=True, rope_interleave=True,
    index_topk_freq=4, index_skip_topk_offset=3, index_topk_pattern=None,
    index_share_for_mtp_iteration=True, moe_router_dtype="float32",
    scoring_func="sigmoid", topk_method="noaux_tc", moe_layer_freq=1, ep_size=1,
    pretraining_tp=1, use_cache=True, bos_token_id=0, eos_token_id=1,
)


def tiny_glm_moe_dsa() -> GlmMoeDsaForCausalLM:
    torch.manual_seed(0)
    return GlmMoeDsaForCausalLM(GlmMoeDsaConfig.from_dict(dict(GLM_MOE_DSA_TINY)))


def write_glm_moe_dsa_head_dim(name: str) -> None:
    """The release's head_dim back over the saved config, and the proof it
    changes nothing.

    GLM-5.2 and 5.3 ship head_dim at qk_nope_head_dim (192), which
    GlmMoeDsaConfig.__post_init__:152 overwrites with the rope width before
    anything reads it, so `save_pretrained` writes the rope width. The
    fixture keeps the release's spelling, and fails if the reloaded model
    reaches other logits than the saved one.
    """
    directory = FIXTURES / name
    saved = json.loads((directory / "config.json").read_text())
    saved["head_dim"] = GLM_MOE_DSA_TINY["head_dim"]
    (directory / "config.json").write_text(json.dumps(saved, indent=2) + "\n")
    loaded = GlmMoeDsaForCausalLM.from_pretrained(
        str(directory), dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if not isinstance(loaded, tuple):
        raise SystemExit("output_loading_info returns the model and its report")
    model, report = loaded
    unread = {key: sorted(report[key]) for key in
              ("missing_keys", "mismatched_keys", "error_msgs") if report[key]}
    depth = f"model.layers.{model.config.num_hidden_layers}."
    stray = sorted(key for key in report["unexpected_keys"] if not key.startswith(depth))
    if unread or stray:
        raise SystemExit(f"{directory}: the config does not read its own weights: {unread} {stray}")
    ids = np.load(directory / "input_ids.npy")
    difference = float(np.max(np.abs(reference_logits(model, ids) - np.load(directory / "logits.npy"))))
    if difference != 0.0:
        raise SystemExit(f"{directory}: the released head_dim moved the logits by {difference:.3e}")
    print(f"{directory}: head_dim {saved['head_dim']} over a rope width of "
          f"{saved['qk_rope_head_dim']} reloads bit for bit")


# Qwen/Qwen3-Next-80B-A3B-Instruct at revision 9c7f2fbe, scaled to a
# fixture. The release's proportions at a hidden width of 64: the dense
# intermediate_size 2.5x that width (160), the routed and shared expert
# widths 0.25x (16), the attention's q heads spanning 2x the width (8 heads
# of 16) over one key/value head (the release's 16:2), the delta net with
# twice as many value heads as key heads and a 3:1 linear-to-full pattern
# derived from full_attention_interval 4, as the release leaves layer_types
# unset. Its own values, unscaled: rope_theta 1e7 with partial_rotary_factor
# 0.25, every layer routed (mlp_only_layers [], decoder_sparse_step 1),
# norm_topk_prob, rms_norm_eps 1e-6, untied embeddings. 512 experts with 10
# per token do not survive a fixture; eight with two per token keep a
# routed layer whose renormalised top-k the reference computes.
# `num_nextn_predict_layers` declares the prediction layer the release ships
# as mtp.* tensors (vllm qwen3_next_mtp.py:59 reads it, defaulting to 1);
# the released config leaves it unset and transformers ignores the tensors.
QWEN3_NEXT_TINY = dict(
    vocab_size=256, hidden_size=64, intermediate_size=160, moe_intermediate_size=16,
    shared_expert_intermediate_size=16, num_hidden_layers=4, num_attention_heads=8,
    num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=8, linear_value_head_dim=12, linear_conv_kernel_dim=4,
    num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
    norm_topk_prob=True, full_attention_interval=4, rope_theta=1e7, partial_rotary_factor=0.25,
    hidden_act="silu", max_position_embeddings=64, rms_norm_eps=1e-6,
    tie_word_embeddings=False, attention_bias=False, num_nextn_predict_layers=1,
)


def tiny_qwen3_next() -> Qwen3NextForCausalLM:
    torch.manual_seed(0)
    return Qwen3NextForCausalLM(Qwen3NextConfig.from_dict(dict(QWEN3_NEXT_TINY)))


def write_qwen3_next_mtp(name: str, model: Qwen3NextForCausalLM, repo: str,
                         seed: int = 2027) -> None:
    """The prediction layer's tensors under vLLM's mtp.* names beside the
    fixture's, its logits over the trunk's hidden states, and the release
    the fixture was scaled from.

    The trunk's experts are one tensor per expert, the release's layout,
    which `save_pretrained` writes by reversing its load-time packing; the
    depth's are written the same way. `Qwen3NextConfig` neither declares nor
    writes `num_nextn_predict_layers`, so the saved config gets it back, and
    `full_attention_interval` beside the derived layer_types, the way the
    release spells its pattern.
    """
    from huggingface_hub import model_info

    directory = FIXTURES / name
    depth = QwenMTP(model.config).eval()
    scatter_weights(depth, seed)
    ids = torch.from_numpy(np.load(directory / "input_ids.npy").astype(np.int64))
    with torch.no_grad():
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        embeddings = model.model.embed_tokens(ids)
        positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
        valid = torch.ones(ids.shape[0], ids.shape[1] - 1, dtype=torch.bool)
        logits = model.lm_head(depth(hidden[:, :-1], embeddings[:, 1:], positions[:, 1:], valid))
    tensors = load_file(str(directory / "model.safetensors"))
    for tensor_name, tensor in depth.state_dict().items():
        if not tensor_name.startswith("layers.0.mlp.experts."):
            tensors["mtp." + tensor_name] = tensor.to(torch.float32).numpy()
    for tensor_name, tensor in expert_tensors(depth.layers[0].get_submodule("mlp.experts")).items():
        tensors["mtp.layers.0." + tensor_name] = tensor
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    np.savez(directory / "mtp_reference.npz", logits=logits.numpy(), hidden=hidden.numpy(),
             embeddings=embeddings.numpy(), positions=positions.numpy(), valid=valid.numpy())
    config = json.loads((directory / "config.json").read_text())
    config.update(num_nextn_predict_layers=QWEN3_NEXT_TINY["num_nextn_predict_layers"],
                  full_attention_interval=QWEN3_NEXT_TINY["full_attention_interval"])
    (directory / "config.json").write_text(json.dumps(dict(sorted(config.items())), indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({
        "released": {"repo": repo, "revision": model_info(repo).sha},
        "transformers": {"version": transformers.__version__,
                         "revision": "93c8b7b485963a10800c91f55304db6be211c2bd"},
        "vllm": {"revision": "51da0ca66c8065619c79e35dff97aa99aeaf5644",
                 "mtp_file": "vllm/model_executor/models/qwen3_next_mtp.py"},
    }, indent=1) + "\n")
    print(f"{directory}: mtp.* depth with {len(tensors)} tensors, mtp logits {tuple(logits.shape)}")


def llama4_tiny_config() -> Llama4TextConfig:
    """Every fourth layer global, so the pattern holds one of each kind;
    floor_scale 4 makes the temperature bite inside twelve positions."""
    return Llama4TextConfig(
        vocab_size=96, hidden_size=32, intermediate_size=48, intermediate_size_mlp=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_local_experts=4, num_experts_per_tok=2, interleave_moe_layer_step=2,
        attention_chunk_size=4, max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        floor_scale=4, attn_scale=0.1, use_qk_norm=True, rms_norm_eps=1e-5,
        tie_word_embeddings=False)


def tiny_llama4() -> Llama4ForCausalLM:
    torch.manual_seed(0)
    return Llama4ForCausalLM(llama4_tiny_config())


def write_llama4_blocks(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = llama4_tiny_config()
    config._attn_implementation = "eager"
    generator = torch.Generator().manual_seed(41)
    hidden = torch.randn(2, 12, config.hidden_size, generator=generator)
    positions = torch.arange(12)[None]
    rotary = Llama4TextRotaryEmbedding(config)
    arrays = {"hidden": hidden.numpy()}
    # Layer 0 rotates and chunks, layer 3 is the global layer of the pattern.
    for name, index in (("local", 0), ("global", 3)):
        attention = Llama4TextAttention(config, layer_index := index).eval()
        scatter_weights(attention, seed=42)
        mask_builder = create_chunked_causal_mask if index == 0 else create_causal_mask
        with torch.no_grad():
            mask = mask_builder(config=config, inputs_embeds=hidden, attention_mask=None,
                                past_key_values=None, position_ids=positions)
            output, _ = attention(hidden, rotary(hidden, positions), mask)
        arrays[f"{name}_output"] = output.numpy()
        if name == "local":
            arrays.update({f"self_attn.{tensor_name}": tensor.detach().numpy()
                           for tensor_name, tensor in attention.named_parameters()})
        assert attention.layer_idx == layer_index
    np.savez(directory / "attention.npz", allow_pickle=False, **arrays)

    moe = Llama4TextMoe(config).eval()
    scatter_weights(moe, seed=43)
    with torch.no_grad():
        output, _ = moe(hidden)
    arrays = {"hidden": hidden.numpy(), "output": output.numpy()}
    arrays.update({f"feed_forward.{tensor_name}": tensor.detach().numpy()
                   for tensor_name, tensor in moe.named_parameters()})
    np.savez(directory / "moe.npz", allow_pickle=False, **arrays)
    print(f"{directory}: attention and moe blocks, {sorted(p.name for p in directory.iterdir())}")


def gemma4_moe_tiny_config() -> Gemma4TextConfig:
    """Two sliding layers around a global one: the global kind keeps one
    key/value head of 16 while the sliding kind keeps two of 8, and every
    layer routes two of four experts of width 16 beside its dense MLP."""
    return Gemma4TextConfig.from_dict(dict(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
        num_global_key_value_heads=1, attention_k_eq_v=True, enable_moe_block=True,
        num_experts=4, top_k_experts=2, moe_intermediate_size=16, sliding_window=4,
        hidden_size_per_layer_input=0, num_kv_shared_layers=0, max_position_embeddings=64,
        rms_norm_eps=1e-6, final_logit_softcapping=30.0, tie_word_embeddings=True,
        rope_parameters={"full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                                            "partial_rotary_factor": 0.25},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}}))


def tiny_gemma4_moe() -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    return Gemma4ForCausalLM(gemma4_moe_tiny_config())


def write_gemma4_moe_block(directory: Path) -> None:
    """The feed-forward half of a routed Gemma 4 layer, run the way the
    layer runs it (modeling_gemma4.py, Gemma4TextDecoderLayer.forward)."""
    directory.mkdir(parents=True, exist_ok=True)
    layer = Gemma4TextDecoderLayer(gemma4_moe_tiny_config(), layer_idx=0).eval()
    scatter_weights(layer, seed=44)
    generator = torch.Generator().manual_seed(45)
    residual = torch.randn(2, 6, layer.hidden_size, generator=generator)
    with torch.no_grad():
        mlp_out = layer.mlp(layer.pre_feedforward_layernorm(residual))
        flat = residual.reshape(-1, layer.hidden_size)
        probabilities, weights, indices = layer.router(flat)
        routed = layer.experts(layer.pre_feedforward_layernorm_2(flat), indices, weights)
        output = (layer.post_feedforward_layernorm_1(mlp_out)
                  + layer.post_feedforward_layernorm_2(routed.reshape(residual.shape)))
    arrays = {"hidden": residual.numpy(), "mlp_out": mlp_out.numpy(),
              "router_probabilities": probabilities.numpy(),
              "router_weights": weights.numpy(), "router_indices": indices.numpy(),
              "output": output.numpy()}
    for prefix in ("router", "experts", "pre_feedforward_layernorm_2",
                   "post_feedforward_layernorm_1", "post_feedforward_layernorm_2"):
        arrays.update({f"{prefix}.{tensor_name}": tensor.detach().numpy()
                       for tensor_name, tensor in getattr(layer, prefix).named_parameters()})
    np.savez(directory / "moe.npz", allow_pickle=False, **arrays)
    print(f"{directory}: moe block, {sorted(arrays)}")


def gemma3n_tiny_config() -> Gemma3nTextConfig:
    """Three copies of the residual stream, sparsity on the first two layers,
    widths of 48 and 64, one layer sharing K/V. The per-layer table has as
    many rows as the vocabulary, since the reference indexes it with the
    token ids as they are (the released 262144 rows serve text ids below the
    multimodal ones)."""
    return Gemma3nTextConfig.from_dict(dict(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32,
        intermediate_size=[48, 48, 64, 64], num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention",
                     "sliding_attention"],
        sliding_window=4, max_position_embeddings=64, rms_norm_eps=1e-6,
        rope_theta=1e6, rope_local_base_freq=1e4, final_logit_softcapping=30.0,
        hidden_size_per_layer_input=8, altup_num_inputs=3, altup_active_idx=0,
        altup_coef_clip=120.0, altup_correct_scale=True, num_kv_shared_layers=1,
        laurel_rank=8, activation_sparsity_pattern=[0.95, 0.95, 0.0, 0.0],
        tie_word_embeddings=True))


def tiny_gemma3n() -> Gemma3nForCausalLM:
    torch.manual_seed(0)
    return Gemma3nForCausalLM(gemma3n_tiny_config())


def write_gemma3n_blocks(directory: Path) -> None:
    """AltUp, the LAuReL block and the sparse MLP, each alone on random
    weights, the way the layer calls them (modeling_gemma3n.py)."""
    directory.mkdir(parents=True, exist_ok=True)
    config = Gemma3nTextConfig.from_dict(dict(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
        hidden_size_per_layer_input=8, altup_num_inputs=4, laurel_rank=8,
        activation_sparsity_pattern=[0.95, 0.0], num_kv_shared_layers=0))
    generator = torch.Generator().manual_seed(46)
    arrays = {}
    altup = Gemma3nTextAltUp(config).eval()
    scatter_weights(altup, seed=47)
    stream = torch.randn(4, 2, 6, config.hidden_size, generator=generator)
    activated = torch.randn(2, 6, config.hidden_size, generator=generator)
    with torch.no_grad():
        predictions = altup.predict(stream)
        corrected = altup.correct(predictions, activated)
        scaled = altup.scale_corrected_output(corrected[config.altup_active_idx])
    arrays.update(stream=stream.numpy(), activated=activated.numpy(),
                  predictions=predictions.numpy(), corrected=corrected.numpy(),
                  scaled=scaled.numpy())
    arrays.update({f"altup.{name}": tensor.detach().numpy()
                   for name, tensor in altup.named_parameters()})
    laurel = Gemma3nTextLaurelBlock(config).eval()
    scatter_weights(laurel, seed=48)
    with torch.no_grad():
        arrays["laurel_output"] = laurel(activated).numpy()
    arrays.update({f"laurel.{name}": tensor.detach().numpy()
                   for name, tensor in laurel.named_parameters()})
    mlp = Gemma3nTextMLP(config, layer_idx=0).eval()
    scatter_weights(mlp, seed=49)
    with torch.no_grad():
        arrays["mlp_output"] = mlp(activated).numpy()
    arrays.update({f"mlp.{name}": tensor.detach().numpy()
                   for name, tensor in mlp.named_parameters()})
    np.savez(directory / "blocks.npz", allow_pickle=False, **arrays)
    print(f"{directory}: altup, laurel and sparse mlp blocks, {sorted(arrays)}")


def add_layer_scalars(name: str) -> None:
    """The ones of the reference's layer_scalar buffer into an older fixture."""
    directory = FIXTURES / name
    tensors = load_file(str(directory / "model.safetensors"))
    layers = {int(tensor_name.split(".")[2]) for tensor_name in tensors
              if tensor_name.startswith("model.layers.")}
    for index in layers:
        tensors[f"model.layers.{index}.layer_scalar"] = np.ones((1,), np.float32)
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})


def write_mirrored_config(name: str, repo: str) -> None:
    """A gated release's config from a mirror that carries it identically
    plus its own marker keys, which are dropped: meta-llama/Llama-4-Scout,
    google/gemma-4-26B-A4B and google/gemma-3n-E2B."""
    from huggingface_hub import hf_hub_download
    import json

    directory = FIXTURES / name
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    config.pop("unsloth_fixed", None)
    config.get("text_config", config).pop("for_llm_compressor", None)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({"repo": repo}) + "\n")


def main() -> None:
    write_tiny("gpt-oss-tiny", tiny_gpt_oss())
    write_tiny("deepseek-v2-tiny", tiny_deepseek_v2())
    write_tiny("kimi-k2-tiny", tiny_kimi_k2())
    write_kimi_k2_config("kimi-k2-tiny", "moonshotai/Kimi-K2-Instruct")
    glm = tiny_glm4_moe()
    write_tiny("glm4-moe-tiny", glm)
    write_glm_mtp("glm4-moe-tiny", glm, glm4_moe_mtp(glm.config))
    dsa = tiny_glm_moe_dsa()
    write_tiny("glm-moe-dsa-tiny", dsa, seed=GLM_MOE_DSA_SEED)
    write_glm_mtp("glm-moe-dsa-tiny", dsa, glm_moe_dsa_mtp(dsa.config))
    write_glm_moe_dsa_head_dim("glm-moe-dsa-tiny")
    qwen3_next = tiny_qwen3_next()
    write_tiny("qwen3-next-tiny", qwen3_next)
    write_qwen3_next_mtp("qwen3-next-tiny", qwen3_next, "Qwen/Qwen3-Next-80B-A3B-Instruct")
    write_released_config("qwen3-next-80b-a3b", "Qwen/Qwen3-Next-80B-A3B-Instruct")
    write_tiny("llama4-tiny", tiny_llama4())
    write_llama4_blocks(FIXTURES.parent / "llama4")
    write_mirrored_config("llama-4-scout", "unsloth/Llama-4-Scout-17B-16E")
    write_mirrored_config("gemma4-26b-a4b", "unsloth/gemma-4-26B-A4B-it")

    write_tiny("gemma4-moe-tiny", tiny_gemma4_moe())
    write_gemma4_moe_block(FIXTURES.parent / "gemma4")
    write_tiny("gemma3n-tiny", tiny_gemma3n())
    write_gemma3n_blocks(FIXTURES.parent / "gemma3n")
    for name in ("gemma4-ple", "gemma4-kvshare", "gemma4-e2b"):
        add_layer_scalars(name)
    write_released_config("gpt-oss-20b", "openai/gpt-oss-20b")
    write_released_config("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite")
    write_released_config("kimi-k2", "moonshotai/Kimi-K2-Instruct")
    write_released_config("glm-4.5-air", "zai-org/GLM-4.5-Air")
    write_released_config("glm-5", "zai-org/GLM-5")
    write_released_config("glm-5.3", "zai-org/GLM-5.3")
    write_mirrored_config("gemma-3n-e2b", "unsloth/gemma-3n-E2B")


if __name__ == "__main__":
    main()
