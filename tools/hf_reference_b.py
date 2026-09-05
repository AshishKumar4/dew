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
- glm4-moe-tiny/: GLM 4.5 at toy width, biased q/k/v over a bias-free
  o_proj, the q/k norms of GLM 4.6, a half rotary, a dense first layer over
  a routed one with the balancing bias and a shared expert, scaled by 1.5,
  and one MTP depth under the released names (model.layers.2.*). transformers
  builds no depth, so its logits (mtp_logits.npy) come from the depth's own
  tensors composed the way the engines that run the released weights do:
  eh_proj over enorm(embeddings) and hnorm(hidden) in that order, one
  Glm4MoeDecoderLayer, shared_head.norm, then the trunk's head.
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
- gpt-oss-20b/, deepseek-v2-lite/, kimi-k2/, glm-4.5-air/, llama-4-scout/,
  gemma4-26b-a4b/: released configs only. Llama-4-Scout and gemma-4-26B-A4B
  are gated, so their configs come from mirrors and drop the mirror's own
  marker key.

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

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import (
    DeepseekV2Config, DeepseekV2ForCausalLM, Gemma3nTextConfig, Gemma4TextConfig,
    Glm4MoeConfig, Glm4MoeForCausalLM, Llama4TextConfig,
)
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM, Gemma3nTextAltUp, Gemma3nTextLaurelBlock, Gemma3nTextMLP,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextDecoderLayer
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer, Glm4MoeRMSNorm
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM, Llama4TextAttention, Llama4TextMoe, Llama4TextRotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt_oss_reference import tiny_gpt_oss  # noqa: E402
from hf_reference import (  # noqa: E402
    DEEPSEEK_YARN, FIXTURES, scatter_weights, write_released_config, write_tiny,
)
from moe_reference import expert_tensors  # noqa: E402


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


class Glm4MoeMTP(torch.nn.Module):
    """One GLM MTP depth as vLLM's Glm4MoeMultiTokenPredictorLayer composes it."""

    def __init__(self, config: Glm4MoeConfig) -> None:
        super().__init__()
        self.enorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = torch.nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = Glm4MoeDecoderLayer(config, layer_idx=config.num_hidden_layers)
        self.shared_head_norm = Glm4MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, model: Glm4MoeForCausalLM, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        fused = self.eh_proj(torch.cat(
            [self.enorm(model.model.embed_tokens(ids)), self.hnorm(hidden)], dim=-1))
        positions = torch.arange(fused.shape[1])[None]
        mask = create_causal_mask(config=model.config, inputs_embeds=fused, attention_mask=None,
                                  past_key_values=None, position_ids=positions)
        out = self.block(fused, attention_mask=mask, position_ids=positions,
                         position_embeddings=model.model.rotary_emb(fused, positions))
        return model.lm_head(self.shared_head_norm(out))


def write_glm4_moe_mtp(name: str, model: Glm4MoeForCausalLM, seed: int = 2026) -> None:
    """The depth's tensors into the fixture checkpoint, and its reference logits."""
    directory = FIXTURES / name
    depth = Glm4MoeMTP(model.config).eval()
    scatter_weights(depth, seed)
    # The layer's submodules sit behind a class decorator that hides them
    # from a checker, so the routed block's parts are read by name.
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
    glm = tiny_glm4_moe()
    write_tiny("glm4-moe-tiny", glm)
    write_glm4_moe_mtp("glm4-moe-tiny", glm)
    write_tiny("llama4-tiny", tiny_llama4())
    write_llama4_blocks(FIXTURES.parent / "llama4")
    write_mirrored_config("llama-4-scout", "unsloth/Llama-4-Scout-17B-16E")
    write_mirrored_config("gemma4-26b-a4b", "unsloth/gemma-4-26B-A4B-it")

    write_gemma4_moe_block(FIXTURES.parent / "gemma4")
    write_tiny("gemma3n-tiny", tiny_gemma3n())
    write_gemma3n_blocks(FIXTURES.parent / "gemma3n")
    for name in ("gemma4-ple", "gemma4-kvshare", "gemma4-e2b"):
        add_layer_scalars(name)
    write_released_config("gpt-oss-20b", "openai/gpt-oss-20b")
    write_released_config("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite")
    write_released_config("kimi-k2", "moonshotai/Kimi-K2-Instruct")
    write_released_config("glm-4.5-air", "zai-org/GLM-4.5-Air")
    write_mirrored_config("gemma-3n-e2b", "unsloth/gemma-3n-E2B")


if __name__ == "__main__":
    main()
