#!/usr/bin/env python3
"""Write the Qwen3.5 fixtures tests/test_linear_attention.py and
tests/test_hf_decoders.py check against.

Everything here runs under torch and transformers 5.16.1, which dew does not
depend on, so this is the only place the reference gated delta net and the
reference Qwen3.5 decoder are executed. The fixtures it writes are what the
suite compares against.

Run it with a venv that has torch and transformers:

    uv venv /tmp/hfref --python 3.12
    uv pip install --python /tmp/hfref/bin/python torch \\
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/hfref/bin/python transformers==5.16.1 \\
        safetensors numpy huggingface_hub
    /tmp/hfref/bin/python tools/qwen_linear_reference.py

What lands:

- tests/fixtures/linear_attention/gated_delta_net.npz: random operands of
  the gated delta rule over 70 tokens (two chunks of 64, the second padded)
  with `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`
  run on them, from a zero state and from a random initial state; a
  depthwise causal conv input, its taps and `F.conv1d`'s output; and one
  `Qwen3_5GatedDeltaNet` layer's weights under their checkpoint names, its
  input and its output. config.json beside it holds the geometry the layer
  was built with, so the test repeats no numbers of its own.
- tests/fixtures/hf/qwen35-tiny/: a random-weight hybrid checkpoint in the
  HF layout (three linear-attention layers, one gated full-attention layer,
  a partial rotary of 0.25 with the interleaved mRoPE sections the released
  configs carry), the 2 x 12 token ids it was run on and its fp32 logits.
- tests/fixtures/hf/qwen35-0.8b/: no weights. config.json is the released
  config of Qwen/Qwen3.5-0.8B, the smallest checkpoint of the family, and
  its text decoder is what the translation is tested on. inv_freq.npy is
  the rotary inverse frequencies the reference builds from that config,
  which is how the test pins the partial rotary convention without torch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding, torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
HF = FIXTURES / "hf"
LINEAR = FIXTURES / "linear_attention"
TINY = "qwen35-tiny"
REAL_MODEL = "Qwen/Qwen3.5-0.8B"
BATCH, LENGTH = 2, 12

# Small enough to live in git, shaped so every gap the family opens is
# exercised: value heads outnumbering key heads, key and value dims apart,
# a partial rotary of a quarter of the head (the released factor), the
# doubled gated q_proj, and a conv with a real history.
CONFIG: dict = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, hidden_act="silu", max_position_embeddings=64,
    rms_norm_eps=1e-6, tie_word_embeddings=True,
    linear_conv_kernel_dim=4, linear_key_head_dim=12, linear_value_head_dim=16,
    linear_num_key_heads=2, linear_num_value_heads=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    # A quarter of head_dim 32 rotates: 8 dims, 4 frequency pairs, which the
    # three mRoPE sections split as 2 + 1 + 1 (the released [11, 11, 10]
    # splits the 32 pairs of a 64-dim rope the same way).
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                     "partial_rotary_factor": 0.25,
                     "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
)

# The rule fixture: two chunks of 64 with the second one padded, three heads
# so a head index mistake shows, and key and value dims apart.
RULE_BATCH, RULE_LENGTH, RULE_HEADS, KEY_DIM, VALUE_DIM = 2, 70, 3, 12, 16
CONV_BATCH, CONV_FEATURES, CONV_LENGTH, CONV_KERNEL = 2, 8, 13, 4
LAYER_LENGTH = 70


def tiny_model() -> Qwen3_5ForCausalLM:
    config = Qwen3_5TextConfig(**CONFIG)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config)
    scatter_weights(model)
    model.eval()
    return model


def scatter_weights(model: torch.nn.Module) -> None:
    """Random weights with something in every tensor.

    The norms move off their identity (Qwen3.5's RMSNorm scales by 1 + w
    from a zero init), so a parity test with the offset backwards fails.
    A_log stays in the reference's own init range (log of 0.01..16) and
    dt_bias near its one, so the decay gate is neither saturated nor dead.
    """
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator)
            if name.endswith("A_log"):
                tensor.copy_(torch.log(0.5 + 3.5 * torch.rand(tensor.shape, generator=generator)))
            elif name.endswith("dt_bias"):
                tensor.copy_(1.0 + 0.1 * noise)
            elif "norm" in name:
                tensor.copy_(tensor + 0.05 * noise)
            elif "conv1d" in name:
                tensor.copy_(0.3 * noise)
            else:
                tensor.copy_(0.2 * noise)


def rule_fixture(generator: torch.Generator) -> dict:
    """The two forms of the rule on random operands, from zero and from a state.

    The reference normalises q and k inside the rule
    (use_qk_l2norm_in_kernel=True, as Qwen3_5GatedDeltaNet calls it), so the
    operands here are raw and dew's l2norm runs in the test. g is a log
    decay, so it is negative, and beta a sigmoid, so it sits in (0, 1).
    """
    shape = (RULE_BATCH, RULE_LENGTH, RULE_HEADS)
    query = torch.randn((*shape, KEY_DIM), generator=generator)
    key = torch.randn((*shape, KEY_DIM), generator=generator)
    value = torch.randn((*shape, VALUE_DIM), generator=generator)
    g = -F.softplus(torch.randn(shape, generator=generator))
    beta = torch.sigmoid(torch.randn(shape, generator=generator))
    state = 0.5 * torch.randn((RULE_BATCH, RULE_HEADS, KEY_DIM, VALUE_DIM), generator=generator)
    arrays = {"rule.query": query, "rule.key": key, "rule.value": value,
              "rule.g": g, "rule.beta": beta, "rule.initial_state": state}
    for name, form in (("chunk", torch_chunk_gated_delta_rule),
                       ("recurrent", torch_recurrent_gated_delta_rule)):
        for carried, initial in (("", None), ("_carried", state)):
            with torch.no_grad():
                out, final = form(query, key, value, g, beta, initial_state=initial,
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)
            arrays[f"{name}{carried}.output"] = out
            arrays[f"{name}{carried}.state"] = final
    return arrays


def conv_fixture(generator: torch.Generator) -> dict:
    """F.conv1d as the reference calls it: depthwise, padded K - 1 to the
    left and cut to the sequence length, then silu."""
    x = torch.randn((CONV_BATCH, CONV_FEATURES, CONV_LENGTH), generator=generator)
    weight = 0.3 * torch.randn((CONV_FEATURES, 1, CONV_KERNEL), generator=generator)
    with torch.no_grad():
        out = F.silu(F.conv1d(x, weight, padding=CONV_KERNEL - 1,
                              groups=CONV_FEATURES)[..., :CONV_LENGTH])
    return {"conv.input": x, "conv.weight": weight, "conv.output": out}


def layer_fixture(model: Qwen3_5ForCausalLM, generator: torch.Generator) -> dict:
    """One Qwen3_5GatedDeltaNet with the tiny model's first-layer weights."""
    layer = model.model.layers[0].linear_attn
    # torch types a submodule attribute as Tensor | Module; this one is the
    # Qwen3_5GatedDeltaNet of a linear_attention layer.
    assert isinstance(layer, torch.nn.Module)
    hidden = torch.randn((RULE_BATCH, LAYER_LENGTH, CONFIG["hidden_size"]), generator=generator)
    with torch.no_grad():
        out = layer(hidden, attention_mask=None)
    arrays = {f"layer.{name}": tensor for name, tensor in layer.state_dict().items()}
    arrays.update({"layer.hidden": hidden, "layer.output": out})
    return arrays


def real_config() -> dict:
    """The released config.json of the smallest Qwen3.5 checkpoint."""
    path = hf_hub_download(REAL_MODEL, "config.json")
    return json.loads(Path(path).read_text())


def rope_inv_freq(config: dict) -> np.ndarray:
    """The inverse frequencies the reference builds for the real decoder:
    a rope of int(head_dim * partial_rotary_factor) dims, exponents over
    that width (Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters,
    modeling_qwen3_5.py:117-124)."""
    text = Qwen3_5TextConfig(**config["text_config"])
    rotary = Qwen3_5TextRotaryEmbedding(text)
    return rotary.inv_freq.detach().to(torch.float32).numpy()


def write_linear(model: Qwen3_5ForCausalLM) -> None:
    LINEAR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(2)
    arrays = {**rule_fixture(generator), **conv_fixture(generator),
              **layer_fixture(model, generator)}
    np.savez(LINEAR / "gated_delta_net.npz",
             **{name: tensor.to(torch.float32).numpy() for name, tensor in arrays.items()})
    geometry = {key: CONFIG[key] for key in (
        "hidden_size", "rms_norm_eps", "linear_conv_kernel_dim", "linear_key_head_dim",
        "linear_value_head_dim", "linear_num_key_heads", "linear_num_value_heads")}
    (LINEAR / "config.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"{LINEAR}: {len(arrays)} arrays, layer output "
          f"{tuple(arrays['layer.output'].shape)}")


def write_tiny(model: Qwen3_5ForCausalLM) -> None:
    directory = HF / TINY
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    ids = np.random.RandomState(7).randint(0, CONFIG["vocab_size"], (BATCH, LENGTH)).astype(np.int32)
    np.save(directory / "input_ids.npy", ids)
    model.set_attn_implementation("eager")
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(ids), use_cache=False).logits
    np.save(directory / "logits.npy", logits.to(torch.float32).numpy())
    size = sum(path.stat().st_size for path in directory.iterdir())
    print(f"{directory}: {size / 1e3:.0f} kB, argmax[0, :8]="
          f"{logits[0].argmax(-1)[:8].tolist()}")


def write_real(config: dict) -> None:
    directory = HF / "qwen35-0.8b"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    inv_freq = rope_inv_freq(config)
    np.save(directory / "inv_freq.npy", inv_freq)
    text = config["text_config"]
    print(f"{directory}: {text['model_type']}, {text['num_hidden_layers']} layers, "
          f"{len(text)} text fields, inv_freq {inv_freq.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    model = tiny_model()
    write_linear(model)
    write_tiny(model)
    write_real(real_config())


if __name__ == "__main__":
    main()
