#!/usr/bin/env python3
"""Regenerate GPT OSS primitive fixtures with transformers 5.16.1 on CPU.

Run PYTHONPATH=src python tools/gpt_oss_reference.py from the checkout.
"""

from pathlib import Path

import numpy as np
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, eager_attention_forward

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


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    write_attention()


if __name__ == "__main__":
    main()
