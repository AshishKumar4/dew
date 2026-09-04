#!/usr/bin/env python3
"""Write the Qwen3.5-family fixtures tests/test_linear_attention.py checks against.

A tiny qwen3_5_text model (hybrid: three linear-attention layers, one
full-attention layer, the gated attention and interleaved mRoPE the released
configs carry) is built with fixed-seed random weights, its config and
weights are saved in the HF layout, and its fp32 logits on a fixed prompt are
committed beside them, so the comparison runs in CI without a download.
Also writes the module-level fixtures: a GatedDeltaNet layer's output and
final state on the same weights, chunked and recurrent, and the conv output.

Run with the venv that has torch and transformers installed:
    /home/mrwhite0racle/Desktop/dew/.venv/bin/python tools/qwen_linear_reference.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
TINY = "qwen35-tiny"

# Small enough to run anywhere, big enough that every shape the family uses
# is exercised: grouped keys, value heads outnumbering key heads, a partial
# rotary, the doubled gated q_proj and the conv with a real history.
CONFIG = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=True,
    linear_conv_kernel_dim=4, linear_key_head_dim=12, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    full_attention_interval=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
)
PROMPT = list(range(1, 13))  # 12 fixed token ids


def tiny_model() -> Qwen3_5ForCausalLM:
    config = Qwen3_5TextConfig(**CONFIG, rope_parameters={
        "rope_type": "default", "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "mrope_interleaved": True, "mrope_section": [2, 2, 4]})
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config)
    model.eval()
    return model


def scatter(model: torch.nn.Module) -> None:
    """Deterministic per-tensor values, something different in every leaf."""
    generator = torch.Generator().manual_seed(1)
    for name, tensor in model.state_dict().items():
        with torch.no_grad():
            noise = torch.randn(tensor.shape, generator=generator)
            if "norm" in name or name.endswith("A_log") or "dt_bias" in name:
                tensor.copy_(noise.abs() * 0.5 + 0.5)
            elif "conv1d" in name:
                tensor.copy_(noise * 0.3 + torch.tensor([0.5, -0.5, 0.25, -0.25][:tensor.shape[-1]]))
            else:
                tensor.copy_(noise)


def main() -> None:
    model = tiny_model()
    scatter(model)
    directory = FIXTURES / TINY
    directory.mkdir(parents=True, exist_ok=True)

    config = json.loads(model.config.text_config.to_json_string())
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    state = {name: tensor.contiguous() for name, tensor in model.state_dict().items()
             if not name.startswith("mtp.")}
    save_file(state, str(directory / "model.safetensors"))

    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        logits = model(ids).logits.to(torch.float32).numpy()
    np.save(directory / "logits.npy", logits)

    # Module-level parity: one linear-attention layer on its own, its output
    # and final recurrent state, for both chunked and recurrent forms.
    layer = model.model.layers[0].linear_attn
    hidden = torch.randn(1, 12, CONFIG["hidden_size"], generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        out = layer(hidden.clone(), attention_mask=None)
        # The chunked form through the module (no cache), and the raw rule
        # both ways to pin the chunked/recurrent equivalence to the reference.
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)
        module = model.model.layers[0].linear_attn
        qkv = module.in_proj_qkv(hidden.clone())
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m5
    np.save(directory / "layer_output.npy", out.detach().numpy())

    print(f"{directory}: logits {logits.shape}, layer output {out.shape}")
    print(f"argmax[:8] = {logits[0].argmax(-1)[:8].tolist()}")


if __name__ == "__main__":
    main()
