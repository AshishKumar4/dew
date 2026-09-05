#!/usr/bin/env python3
"""Regenerate GPT OSS primitive fixtures with transformers 5.16.1 on CPU.

Run PYTHONPATH=src python tools/gpt_oss_reference.py from the checkout.
"""

from pathlib import Path

import numpy as np
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention, GptOssForCausalLM, GptOssMLP, eager_attention_forward,
)
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gpt_oss"


def write_attention() -> None:
    config = GptOssConfig(hidden_size=32, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=8, num_hidden_layers=1)
    attention = GptOssAttention(config, layer_idx=0).eval()
    generator = torch.Generator().manual_seed(137)
    query = torch.randn(2, 4, 7, 8, generator=generator)
    key = torch.randn(2, 2, 7, 8, generator=generator)
    value = torch.randn(2, 2, 7, 8, generator=generator)
    sinks = torch.tensor([-2.0, 0.3, 3.0, 10.0])
    positions = torch.arange(7)
    mask = (positions[:, None] >= positions) & (positions[:, None] - positions < 3)
    additive = torch.where(mask, 0.0, torch.finfo(torch.float32).min)[None, None]
    with torch.no_grad():
        attention.get_parameter("sinks").copy_(sinks)
        output, _ = eager_attention_forward(
            attention, query, key, value, additive, scaling=8**-0.5)
    np.savez(FIXTURES / "attention.npz", query=query.transpose(1, 2).numpy(),
             key=key.transpose(1, 2).numpy(), value=value.transpose(1, 2).numpy(),
             sinks=sinks.numpy(), mask=mask.numpy(), output=output.numpy())

def write_moe() -> None:
    config = GptOssConfig(hidden_size=16, intermediate_size=24, num_local_experts=4,
                          num_experts_per_tok=2)
    config._experts_implementation = 'eager'
    block = GptOssMLP(config).eval()
    generator = torch.Generator().manual_seed(138)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.8)
        hidden = torch.randn(2, 7, 16, generator=generator) * 2
        output, _ = block(hidden)
    arrays = {name: parameter.detach().numpy() for name, parameter in block.named_parameters()}
    arrays.update(hidden=hidden.numpy(), output=output.numpy())
    np.savez(FIXTURES / "moe.npz", allow_pickle=False, **arrays)


def write_mxfp4() -> None:
    generator = torch.Generator().manual_seed(139)
    blocks = torch.randint(0, 256, (2, 8, 2, 16), generator=generator, dtype=torch.uint8)
    scales = torch.randint(110, 142, blocks.shape[:-1], generator=generator, dtype=torch.uint8)
    output = convert_moe_packed_tensors(blocks, scales).float()
    np.savez(FIXTURES / "mxfp4.npz", blocks=blocks.numpy(), scales=scales.numpy(),
             output=output.numpy())


def tiny_gpt_oss() -> GptOssForCausalLM:
    config = GptOssConfig(
        # Both expert input widths are multiples of the 32-value MXFP4 group.
        hidden_size=32, intermediate_size=64, vocab_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0,
                         "factor": 4.0, "original_max_position_embeddings": 16,
                         "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False})
    torch.manual_seed(140)
    return GptOssForCausalLM(config)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    write_attention()
    write_moe()
    write_mxfp4()


if __name__ == "__main__":
    main()
