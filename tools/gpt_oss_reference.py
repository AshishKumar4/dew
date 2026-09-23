#!/usr/bin/env python3
"""Regenerate GPT OSS primitive fixtures with transformers 5.16.1 on CPU.

The exchange-gradient fixtures additionally require torch2.14.0+cpu. The
MXFP4 encoder fixture runs the kernel transformers loads for MXFP4,
kernels-community/gpt-oss-triton-kernels at `KERNELS`, whose module imports
triton; its torch path, the one transformers calls, runs on CPU.

Run PYTHONPATH=src python tools/gpt_oss_reference.py from the checkout.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers.integrations.mxfp4 import convert_moe_packed_tensors, quantize_to_mxfp4
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssForCausalLM,
    GptOssMLP,
    eager_attention_forward,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gpt_oss"

KERNELS = ("kernels-community/gpt-oss-triton-kernels", "0f351046d799bc4d2c9dbb2c7cf36753204b5916")
"""The kernel transformers 5.16.1 loads for MXFP4 (`get_kernel(..., version=1)`,
integrations/mxfp4.py:567), at the commit its v1 branch names."""

MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
"""Halfway between neighbouring E2M1 magnitudes."""


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


def mxfp4_domains() -> dict[str, np.ndarray]:
    """[expert, input, output] float32 weights for the MXFP4 encoder, by name."""
    rng = np.random.default_rng(4242)
    ties = np.concatenate([[6.0], MIDPOINTS, [-6.0], -MIDPOINTS])
    groups = np.zeros((8, 32), np.float32)
    # 6 sets each scale, so every midpoint of either sign is its own
    # quotient; at 2 ** -126 the smaller ones are bf16 subnormals.
    for row, exponent in enumerate((0, -3, 10, -126)):
        groups[row, :16] = np.ldexp(ties, exponent)
    # A largest value of 5 rounds its scale up to 1, so it is a midpoint too.
    groups[4, :8] = [5.0, -5.0, 2.5, -2.5, 1.25, -1.25, 0.25, -0.25]
    # Values bf16 carries onto a midpoint before the kernel sees them.
    groups[5, :10] = [6.0, 0.2499, -0.2499, 1.2499, -1.2499, 2.495, -2.495, 4.99, -5.01, 0.7495]
    # Nothing but a -0.0: scale byte 0, codes 0 and 8.
    groups[6, 3] = -0.0
    # bf16's smallest subnormal: the scale still rounds up to 2 ** -126.
    groups[7, :3] = [2.0 ** -133, -(2.0 ** -133), 2.0 ** -133]
    patterns = rng.integers(0, 1 << 16, size=1_000 * 32, dtype=np.uint16)
    random_bf16 = patterns.view(ml_dtypes.bfloat16).astype(np.float32)
    random_bf16[~np.isfinite(random_bf16)] = 0
    small = np.ldexp(rng.integers(1, 256, size=(500, 31)).astype(np.float32), -141)
    large = np.ldexp(np.float32(1), rng.integers(-132, 8, size=(500, 1)))
    mixed = np.concatenate([large, small * rng.choice([-1, 1], size=(500, 31))], axis=1)
    magnitudes = np.arange(1 << 15, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)
    return {
        "midpoints": np.ascontiguousarray(groups.T[None]),
        "random_bf16": random_bf16.reshape(1, -1, 1),
        "mixed": mixed.astype(np.float32).reshape(1, -1, 1),
        "trained": (rng.standard_normal((4, 128, 9)) * 0.08).astype(np.float32),
        "every_bf16_magnitude": np.repeat(magnitudes[np.isfinite(magnitudes)], 32).reshape(1, -1, 1),
    }


def write_mxfp4_encoder() -> None:
    """The blocks and scales transformers' own MXFP4 encoder writes, by domain.

    transformers' `quantize_to_mxfp4` (integrations/mxfp4.py:231-234) runs
    `downcast_to_mxfp_torch` of the kernel's numerics_details/mxfp.py on the
    weight rounded to bf16; `hub` stands in for the module `get_kernel`
    returns, which is where transformers finds that function. The weights
    are [expert, input, output] and quantize along the input axis, as the
    checkpoint groups them; the blocks and scales are stored in the
    checkpoint's [expert, output, group, 16] and [expert, output, group].
    """
    root = Path(snapshot_download(KERNELS[0], revision=KERNELS[1],
                                  allow_patterns=["build/torch-cpu/numerics_details/**"]))
    sys.path.insert(0, str(root / "build" / "torch-cpu"))
    hub = SimpleNamespace(numerics_details=SimpleNamespace(mxfp=importlib.import_module("numerics_details.mxfp")))
    arrays = {}
    for domain, weight in mxfp4_domains().items():
        experts, inputs, outputs = weight.shape
        packed, scales = quantize_to_mxfp4(torch.from_numpy(weight), hub)
        arrays[f"{domain}_weight"] = weight
        arrays[f"{domain}_blocks"] = packed.transpose(1, 2).reshape(experts, outputs, inputs // 32, 16).numpy()
        arrays[f"{domain}_scales"] = scales.transpose(1, 2).contiguous().numpy()
    np.savez_compressed(FIXTURES / "mxfp4_encoder.npz", **arrays)


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


def write_exchange_cases() -> None:
    """Tiny fp32 full-MLP VJPs from the pinned transformers implementation."""
    import json
    from importlib.metadata import version

    reference = {'transformers': '5.16.1', 'torch': '2.14.0+cpu'}
    for package, expected in reference.items():
        if version(package) != expected:
            raise RuntimeError(f"exchange fixtures require {package}=={expected}")
    for skewed in (False, True):
        config = GptOssConfig(hidden_size=8, intermediate_size=12, num_local_experts=4,
                             num_experts_per_tok=2)
        config._experts_implementation = 'eager'
        block = GptOssMLP(config).eval()
        generator = torch.Generator().manual_seed(141)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .3)
            biases = block.get_parameter('experts.gate_up_proj_bias')
            biases[0, :2] = torch.tensor([12., -12.])
            biases[1, :2] = torch.tensor([-12., 12.])
            if skewed:
                block.get_parameter('router.weight').mul_(.05)
                block.get_parameter('router.bias').copy_(torch.tensor([2., 1., -20., -20.]))
        hidden = torch.randn(2, 8, 8, generator=generator).requires_grad_()
        output, scores = block(hidden)
        loss = output.sin().mean()
        loss.backward()
        with torch.no_grad():
            _, _, indices = block.router(hidden.reshape(-1, 8))
        if hidden.grad is None:
            raise RuntimeError('reference input gradient is absent')
        arrays = {'hidden': hidden.detach().numpy(), 'output': output.detach().numpy(),
                  'loss': loss.detach().numpy(), 'input_grad': hidden.grad.numpy(),
                  'router_scores': scores.detach().numpy().reshape(2, 8, 2),
                  'router_indices': indices.numpy().reshape(2, 8, 2)}
        for name, parameter in block.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f'reference gradient is absent for {name}')
            arrays[name] = parameter.detach().numpy()
            arrays['grad.' + name] = parameter.grad.numpy()
        name = 'exchange_skewed' if skewed else 'exchange_random'
        np.savez(FIXTURES / (name + '.npz'), **arrays)
    (FIXTURES / 'exchange_reference.json').write_text(json.dumps({
        **reference, 'seed': 141, 'dtype': 'float32', 'loss': 'mean(sin(output))',
        'hidden_size': 8, 'intermediate_size': 12, 'experts': 4, 'top_k': 2}, indent=2) + '\n')



def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    write_exchange_cases()
    write_attention()
    write_moe()
    write_mxfp4()
    write_mxfp4_encoder()


if __name__ == "__main__":
    main()
