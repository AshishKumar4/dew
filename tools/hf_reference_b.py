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
- gpt-oss-20b/, deepseek-v2-lite/, kimi-k2/, glm-4.5-air/, llama-4-scout/:
  released configs only. Llama-4-Scout is gated, so its config comes from
  a mirror and drops the mirror's own marker key.

What lands in tests/fixtures/llama4, the block-level references the
primitive tests read before the family loads:

- attention.npz: one `Llama4TextAttention` as a local layer (rope, L2
  norm, chunk 4) and as a global layer (no rope, temperature tuning at
  floor_scale 4) on the same random weights and hidden states.
- moe.npz: one `Llama4TextMoe` on random weights, its output and the
  per-expert tensors the checkpoint layout carries fused.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from transformers import (
    DeepseekV2Config, DeepseekV2ForCausalLM, Glm4MoeConfig, Glm4MoeForCausalLM,
    Llama4TextConfig,
)
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
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
        attention_chunk_size=4, max_position_embeddings=64, rope_theta=500000.0,
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


def write_llama4_scout_config() -> None:
    """meta-llama/Llama-4-Scout-17B-16E is gated; the mirror carries the
    identical config plus its own marker, which is dropped."""
    from huggingface_hub import hf_hub_download
    import json

    directory = FIXTURES / "llama-4-scout"
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(hf_hub_download("unsloth/Llama-4-Scout-17B-16E", "config.json")).read_text())
    config["text_config"].pop("for_llm_compressor", None)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    (directory / "source.json").write_text(json.dumps({"repo": "unsloth/Llama-4-Scout-17B-16E"}) + "\n")


def main() -> None:
    write_tiny("gpt-oss-tiny", tiny_gpt_oss())
    write_tiny("deepseek-v2-tiny", tiny_deepseek_v2())
    glm = tiny_glm4_moe()
    write_tiny("glm4-moe-tiny", glm)
    write_glm4_moe_mtp("glm4-moe-tiny", glm)
    write_tiny("llama4-tiny", tiny_llama4())
    write_llama4_blocks(FIXTURES.parent / "llama4")
    write_llama4_scout_config()
    write_released_config("gpt-oss-20b", "openai/gpt-oss-20b")
    write_released_config("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite")
    write_released_config("kimi-k2", "moonshotai/Kimi-K2-Instruct")
    write_released_config("glm-4.5-air", "zai-org/GLM-4.5-Air")


if __name__ == "__main__":
    main()
