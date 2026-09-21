#!/usr/bin/env python3
"""The Mamba-2 fixtures tests/test_mamba2.py checks against, from
transformers 5.16.1 on CPU.

    PYTHONPATH=src:tools .venv/bin/python tools/mamba2_reference.py

What lands:

- tests/fixtures/mamba2/ssd.npz: random operands of the SSD scan over 70
  tokens (three chunks of 32, the last padded) with `mamba2_chunk_scan`
  (modeling_mamba2.py:254-357) run on them from a zero state and from a
  random initial state, `mamba2_selective_state_update` (192-251) on the
  first token from that state, and one `Mamba2Mixer` layer's weights under
  their checkpoint names with its input and its output in fp32 and in
  bfloat16. config.json beside it holds the geometry.
- tests/fixtures/hf/mamba2-tiny/: a random `Mamba2ForCausalLM` in the HF
  layout, two SSD layers of two heads (head_dim 8 over a hidden width of 8
  with expand 2), a state of 4 per head, one B/C group so both heads read
  the same B and C, the conv with its bias, and a chunk of 4 so the
  12-token reference crosses two chunk boundaries. `logits.npy` is the fp32
  forward of tools/hf_reference.py's `write_tiny`, `logits_bf16.npy` the
  same forward with weights and activations in bfloat16. `source.json`
  pins the reference release.
"""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from hf_reference import FIXTURES, write_tiny
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2ForCausalLM,
    Mamba2Mixer,
    mamba2_chunk_scan,
    mamba2_selective_state_update,
)

SSD = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2"
TRANSFORMERS_REVISION = "93c8b7b485963a10800c91f55304db6be211c2bd"
SEED = 2024
# The scan fixture's geometry: batch, tokens, heads, head dim, groups, state, chunk.
B, S, H, P, G, N, CHUNK = 2, 70, 4, 6, 2, 5, 32


def mamba2_tiny_config() -> Mamba2Config:
    return Mamba2Config(
        vocab_size=32, hidden_size=8, expand=2, num_heads=2, head_dim=8, state_size=4,
        n_groups=1, conv_kernel=4, chunk_size=4, num_hidden_layers=2,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        tie_word_embeddings=False, pad_token_id=1, bos_token_id=0, eos_token_id=2)


def scan_fixture(generator: torch.Generator) -> dict[str, torch.Tensor]:
    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)
    x, dt = randn(B, S, H, P), randn(B, S, H)
    A = -torch.exp(randn(H))
    Bm, Cm = randn(B, S, G, N), randn(B, S, G, N)
    D, dt_bias = randn(H), randn(H)
    initial = randn(B, H, P, N)
    with torch.no_grad():
        out, final = mamba2_chunk_scan(x, dt, A, Bm, Cm, chunk_size=CHUNK, D=D, dt_bias=dt_bias,
                                       dt_softplus=True, return_final_states=True)
        out_carried, final_carried = mamba2_chunk_scan(
            x, dt, A, Bm, Cm, chunk_size=CHUNK, D=D, dt_bias=dt_bias, dt_softplus=True,
            initial_states=initial, return_final_states=True)
        state = initial.clone()
        step = mamba2_selective_state_update(
            state, x[:, 0], dt[:, 0][..., None].expand(-1, -1, P), A[:, None, None].expand(-1, P, N),
            Bm[:, 0], Cm[:, 0], D[:, None].expand(-1, P), dt_bias[:, None].expand(-1, P), dt_softplus=True)
    return {
        "scan.x": x, "scan.dt": dt, "scan.A": A, "scan.B": Bm, "scan.C": Cm, "scan.D": D,
        "scan.dt_bias": dt_bias, "scan.initial": initial,
        "scan.output": out, "scan.final": final,
        "scan.output_carried": out_carried, "scan.final_carried": final_carried,
        "step.output": step, "step.state": state,
    }


def layer_fixture(generator: torch.Generator) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    config = Mamba2Config(hidden_size=12, expand=2, num_heads=H, head_dim=P, state_size=N, n_groups=G,
                          conv_kernel=4, chunk_size=CHUNK, num_hidden_layers=1, vocab_size=32,
                          layer_norm_epsilon=1e-5)
    layer = Mamba2Mixer(config, layer_idx=0).eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
        hidden = torch.randn(B, S, config.hidden_size, generator=generator)
        output = layer(hidden)
        half = Mamba2Mixer(config, layer_idx=0).eval()
        half.load_state_dict(layer.state_dict())
        output_bf16 = half.to(torch.bfloat16)(hidden.to(torch.bfloat16)).to(torch.float32)
    arrays = {f"layer.{name}": tensor for name, tensor in layer.state_dict().items()}
    arrays.update({"layer.hidden": hidden, "layer.output": output, "layer.output_bf16": output_bf16})
    geometry = {"hidden_size": config.hidden_size, "num_heads": H, "head_dim": P, "state_size": N,
                "n_groups": G, "conv_kernel": 4, "chunk_size": CHUNK, "layer_norm_epsilon": 1e-5}
    return arrays, geometry


def write_ssd() -> None:
    SSD.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(SEED)
    arrays = scan_fixture(generator)
    layer, geometry = layer_fixture(generator)
    arrays.update(layer)
    np.savez(SSD / "ssd.npz", allow_pickle=False, **{name: tensor.detach().numpy() for name, tensor in arrays.items()})
    (SSD / "config.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"{SSD / 'ssd.npz'}: {(SSD / 'ssd.npz').stat().st_size / 1e3:.0f} kB")


def write_mamba2_tiny(name: str = "mamba2-tiny") -> None:
    model = Mamba2ForCausalLM(mamba2_tiny_config())
    write_tiny(name, model, seed=SEED)
    directory = FIXTURES / name
    ids = np.load(directory / "input_ids.npy")
    half = Mamba2ForCausalLM.from_pretrained(str(directory), dtype=torch.bfloat16, local_files_only=True).eval()
    with torch.no_grad():
        logits = half(input_ids=torch.from_numpy(ids), use_cache=False).logits
    np.save(directory / "logits_bf16.npy", logits.to(torch.float32).numpy())
    (directory / "source.json").write_text(json.dumps({
        "transformers": {"version": transformers.__version__, "revision": TRANSFORMERS_REVISION},
        "seed": SEED,
    }, indent=1) + "\n")
    (directory / "model.safetensors").chmod(0o644)
    print(f"{directory}: bf16 logits and source.json written")


if __name__ == "__main__":
    write_ssd()
    write_mamba2_tiny()
