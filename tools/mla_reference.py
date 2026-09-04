#!/usr/bin/env python3
"""Write the MLA fixtures tests/test_mla.py checks against.

Everything here runs under torch and transformers, which Dew does not
depend on, so this is the only place the reference attention is executed.
The fixtures it writes are what the suite compares against.

Set up the venv and run it (the same room tools/moe_reference.py needs):

    uv venv /tmp/hfref --python 3.12
    uv pip install --python /tmp/hfref/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/hfref/bin/python transformers==5.16.1 numpy
    /tmp/hfref/bin/python tools/mla_reference.py

What lands in tests/fixtures/mla:

- config.json: the fields of each reference config Dew's modules are built
  from, so the test repeats no numbers of its own.
- mla_v3.npz: hidden states, every weight of a tiny `DeepseekV3Attention`
  (q LoRA down/up plus its norm, the compressed KV projection plus its
  norm, the expansion, the output projection), the YaRN rope spelling both
  released DeepSeek configs carry, and the block's fp32 output.
- mla_v32.npz: the same for a tiny `DeepseekV32Attention` with its DSA
  indexer weights, the eager top-k mask fold, and biased projections, so
  the bias path is exercised somewhere.
- yarn.npz: the inverse frequencies, cos/sin amplitude and attention scale
  the reference rotary embedding derives from the released YaRN spelling,
  against which dew's closed form is checked directly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
    DeepseekV3Config,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
)
from transformers.models.deepseek_v32.configuration_deepseek_v32 import (
    DeepseekV32Config,
)
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32RotaryEmbedding as rotary32,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mla"

BATCH, LENGTH, HIDDEN = 2, 7, 32
HEADS = 4
Q_LORA, KV_LORA = 8, 8
NOPE, ROPE, V = 8, 8, 8

# The released rope spelling: YaRN factor 40 off 4096 base positions, mscale
# on every dim, theta 10000. Old `rope_scaling` key, as both released configs
# ship it; transformers standardises it into rope_parameters.
YARN = {
    "type": "yarn",
    "factor": 40.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}

V3 = dict(
    hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=HEADS,
    q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=NOPE,
    qk_rope_head_dim=ROPE, v_head_dim=V, rope_theta=10000.0,
    rope_scaling=dict(YARN), rope_interleave=True, attention_bias=False,
    rms_norm_eps=1e-6, max_position_embeddings=256)

# Biased projections, so the bias path is exercised; the indexer keeps two
# heads of width 16 over the rope width of 8, and a top-k the sequence
# exceeds, so selection actually drops keys.
V32 = dict(
    hidden_size=HIDDEN, num_attention_heads=HEADS, num_key_value_heads=HEADS,
    q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA, qk_nope_head_dim=NOPE,
    qk_rope_head_dim=ROPE, v_head_dim=V, rope_theta=10000.0,
    rope_scaling=dict(YARN), attention_bias=True, rms_norm_eps=1e-6,
    max_position_embeddings=256, index_topk=4, index_n_heads=2,
    index_head_dim=16)


def hidden_states() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn((BATCH, LENGTH, HIDDEN), generator=generator)


def scatter_weights(module: torch.nn.Module, seed: int) -> None:
    """Random weights with something in every tensor.

    Freshly constructed norms sit at their identity, which would pass a
    parity test that had a norm backwards; every tensor gets noise instead.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in module.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator) * 0.4
            if "norm" in name or "layernorm" in name:
                tensor.copy_(1.0 + torch.randn(tensor.shape, generator=generator) * 0.05)
            elif "bias" in name:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.2)
            else:
                tensor.copy_(noise)


def causal_mask(batch: int, length: int) -> torch.Tensor:
    """Additive fp32 causal mask: 0 kept, -inf dropped, `[B, 1, S, T]`."""
    mask = torch.full((length, length), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, length, length)


def block_weights(module: torch.nn.Module) -> dict:
    """Every parameter of the attention block as fp32 numpy, by torch name."""
    return {name: tensor.detach().to(torch.float32).numpy()
            for name, tensor in module.named_parameters()}


def run_v3(directory: Path) -> None:
    config = DeepseekV3Config(**V3)
    block = DeepseekV3Attention(config, layer_idx=0).to(torch.float32)
    scatter_weights(block, seed=12)
    states = hidden_states()
    rotary = DeepseekV3RotaryEmbedding(config)
    positions = torch.arange(LENGTH).unsqueeze(0).expand(BATCH, LENGTH)
    with torch.no_grad():
        embeddings = rotary(states, positions)
        output, _ = block(states, embeddings, causal_mask(BATCH, LENGTH))
    arrays = {"hidden": states.numpy(), "output": output.numpy(),
              **block_weights(block)}
    np.savez(directory / "mla_v3.npz", **arrays)
    print(f"mla_v3.npz: {len(arrays)} arrays")


def run_v32(directory: Path) -> None:
    config = DeepseekV32Config(**V32)
    block = DeepseekV32Attention(config, layer_idx=0).to(torch.float32)
    scatter_weights(block, seed=13)
    block.eval()
    states = hidden_states()
    positions = torch.arange(LENGTH).unsqueeze(0).expand(BATCH, LENGTH)
    with torch.no_grad():
        embeddings = rotary32(config)(states, positions)
        output, _ = block(states, embeddings, causal_mask(BATCH, LENGTH),
                          position_ids=positions)
    arrays = {"hidden": states.numpy(), "output": output.numpy(),
              **block_weights(block)}
    np.savez(directory / "mla_v32.npz", **arrays)
    print(f"mla_v32.npz: {len(arrays)} arrays")


def run_yarn(directory: Path) -> None:
    """The reference's YaRN derivation of the released spelling, standalone."""
    config = DeepseekV3Config(**V3)
    rotary = DeepseekV3RotaryEmbedding(config)
    inv_freq = rotary.inv_freq.detach().to(torch.float32).numpy()
    print(f"rope_type={rotary.rope_type} attention_scaling={rotary.attention_scaling}")
    states = hidden_states()
    positions = torch.arange(LENGTH).unsqueeze(0).expand(BATCH, LENGTH)
    with torch.no_grad():
        cos, sin = rotary(states, positions)
    arrays = {"inv_freq": inv_freq,
              "attention_scaling": np.float32(rotary.attention_scaling),
              "cos": cos.numpy(), "sin": sin.numpy()}
    np.savez(directory / "yarn.npz", **arrays)
    print(f"yarn.npz: inv_freq {inv_freq.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps({"v3": V3, "v32": V32, "yarn": YARN,
                    "batch": BATCH, "length": LENGTH}, indent=2) + "\n")
    run_v3(out)
    run_v32(out)
    run_yarn(out)
    size = sum(path.stat().st_size for path in out.iterdir())
    print(f"{out}: {size / 1e3:.0f} kB, {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
