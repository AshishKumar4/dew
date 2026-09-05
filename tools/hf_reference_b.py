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
- gpt-oss-20b/, deepseek-v2-lite/, kimi-k2/: released configs only.
"""

import sys
from pathlib import Path

import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt_oss_reference import tiny_gpt_oss  # noqa: E402
from hf_reference import DEEPSEEK_YARN, write_released_config, write_tiny  # noqa: E402


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


def main() -> None:
    write_tiny("gpt-oss-tiny", tiny_gpt_oss())
    write_tiny("deepseek-v2-tiny", tiny_deepseek_v2())
    write_released_config("gpt-oss-20b", "openai/gpt-oss-20b")
    write_released_config("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite")
    write_released_config("kimi-k2", "moonshotai/Kimi-K2-Instruct")


if __name__ == "__main__":
    main()
