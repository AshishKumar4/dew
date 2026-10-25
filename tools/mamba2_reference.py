#!/usr/bin/env python3
"""The Mamba-2 fixtures tests/test_mamba2.py checks against, from
transformers 5.16.1 on CPU.

    PYTHONPATH=src:tools .venv/bin/python tools/mamba2_reference.py
    PYTHONPATH=src:tools .venv/bin/python tools/mamba2_reference.py float64

What lands:

- tests/fixtures/mamba2/ssd.npz: random operands of the SSD scan over 70
  tokens (three chunks of 32, the last padded) with `mamba2_chunk_scan`
  (modeling_mamba2.py:254-357) run on them from a zero state and from a
  random initial state, `mamba2_selective_state_update` (192-251) on the
  first token from that state, and one `Mamba2Mixer` layer's weights under
  their checkpoint names with its input and its output in fp32 and in
  bfloat16, and every output again in float64 on the same operands
  (`write_float64`, the `_f64` arrays), the truth the tests measure each
  side's rounding from. config.json beside it holds the geometry.
- tests/fixtures/hf/mamba2-tiny/: a random `Mamba2ForCausalLM` in the HF
  layout, two SSD layers of two heads (head_dim 8 over a hidden width of 8
  with expand 2), a state of 4 per head, one B/C group so both heads read
  the same B and C, the conv with its bias, and a chunk of 4 so the
  12-token reference crosses two chunk boundaries. `logits.npy` is the fp32
  forward of tools/hf_reference.py's `write_tiny`; its bf16 and float64
  forwards are tools/numerics_reference.py's. `source.json` pins the
  reference release.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from hf_reference import FIXTURES, write_tiny
from torch_precision import float64
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
    np.savez(SSD / "ssd.npz", **{name: tensor.detach().numpy() for name, tensor in arrays.items()})
    (SSD / "config.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(f"{SSD / 'ssd.npz'}: {(SSD / 'ssd.npz').stat().st_size / 1e3:.0f} kB")


def write_float64() -> None:
    """ssd.npz's outputs again in float64 on its own stored operands and weights (`float64`), each
    beside its fp32 output as `<name>_f64`; nothing it held before changes."""
    arrays = dict(np.load(SSD / "ssd.npz"))
    wide = {name: torch.from_numpy(arrays[f"scan.{name}"]).double()
            for name in ("x", "dt", "A", "B", "C", "D", "dt_bias", "initial")}
    geometry = json.loads((SSD / "config.json").read_text())
    config = Mamba2Config(hidden_size=geometry["hidden_size"], expand=2, num_heads=H, head_dim=P,
                          state_size=N, n_groups=G, conv_kernel=geometry["conv_kernel"], chunk_size=CHUNK,
                          num_hidden_layers=1, vocab_size=32,
                          layer_norm_epsilon=geometry["layer_norm_epsilon"])
    with torch.no_grad(), float64():
        scan = {"chunk_size": CHUNK, "D": wide["D"], "dt_bias": wide["dt_bias"], "dt_softplus": True,
                "return_final_states": True}
        operands = (wide["x"], wide["dt"], wide["A"], wide["B"], wide["C"])
        out, final = mamba2_chunk_scan(*operands, **scan)
        carried, carried_final = mamba2_chunk_scan(*operands, initial_states=wide["initial"], **scan)
        state = wide["initial"].clone()
        step = mamba2_selective_state_update(
            state, wide["x"][:, 0], wide["dt"][:, 0][..., None].expand(-1, -1, P),
            wide["A"][:, None, None].expand(-1, P, N), wide["B"][:, 0], wide["C"][:, 0],
            wide["D"][:, None].expand(-1, P), wide["dt_bias"][:, None].expand(-1, P), dt_softplus=True)
        layer = Mamba2Mixer(config, layer_idx=0).eval().double()
        layer.load_state_dict({name[len("layer."):]: torch.from_numpy(value).double()
                               for name, value in arrays.items() if name.startswith("layer.")
                               and name[len("layer."):] in layer.state_dict()})
        output = layer(torch.from_numpy(arrays["layer.hidden"]).double())
    wides = {"scan.output": out, "scan.final": final, "scan.output_carried": carried,
             "scan.final_carried": carried_final, "step.output": step, "step.state": state,
             "layer.output": output}
    for name, tensor in wides.items():
        assert tensor.dtype == torch.float64, f"{name} left float64"
        arrays[f"{name}_f64"] = tensor.detach().numpy()
    np.savez(SSD / "ssd.npz", **arrays)
    print(f"{SSD / 'ssd.npz'}: {len(wides)} float64 outputs")


def write_mamba2_tiny(name: str = "mamba2-tiny") -> None:
    model = Mamba2ForCausalLM(mamba2_tiny_config())
    write_tiny(name, model, seed=SEED)
    directory = FIXTURES / name
    (directory / "source.json").write_text(json.dumps({
        "transformers": {"version": transformers.__version__, "revision": TRANSFORMERS_REVISION},
        "seed": SEED,
    }, indent=1) + "\n")
    (directory / "model.safetensors").chmod(0o644)
    print(f"{directory}: source.json written")


def write_seeded() -> None:
    """Fresh native-input/orbit draws with transformers' fp32 and torch-float64 outputs.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba2_reference.py seeded
    """
    import jax
    from recorded_references import provenance, record_case, save
    from reference_error import assert_computes_the_oracle, chain_roundings, widened
    from ssd_support import mamba2_draw, mamba2_run

    def draw(seed):
        inputs = mamba2_draw(seed, (B, S, H, P, G, N))
        redrawn = mamba2_draw(seed)
        assert inputs.keys() == redrawn.keys(), f"transformers seed {seed}: regenerated inputs differ"
        for name, value in inputs.items():
            other = redrawn[name]
            assert (value.dtype == other.dtype and value.shape == other.shape
                    and value.tobytes() == other.tobytes()), f"transformers seed {seed}: {name} differs"
        return inputs

    def reference(site, arrays):
        x, dt, A, Bm, Cm, D, bias, initial = (arrays[name] for name in
                                             ("x", "dt", "A", "B", "C", "D", "dt_bias", "h0"))
        if site == "step":
            state = initial.clone()
            output = mamba2_selective_state_update(
                state, x[:, 0], dt[:, 0][..., None].expand(-1, -1, P),
                A[:, None, None].expand(-1, P, N), Bm[:, 0], Cm[:, 0],
                D[:, None].expand(-1, P), dt_bias=bias[:, None].expand(-1, P), dt_softplus=True)
        else:
            output, state = mamba2_chunk_scan(
                x, dt, A, Bm, Cm, chunk_size=CHUNK, D=D, dt_bias=bias, dt_softplus=True,
                initial_states=initial if site == "carried" else None, return_final_states=True)
        return {"output": output.detach().numpy(), "final": state.detach().numpy()}

    stored = {}
    for site in ("scan", "carried", "step"):
        exact_run = jax.jit(lambda inputs, site=site: mamba2_run(site, inputs))
        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(draw(0))))

        def evaluate(inputs, site=site, exact_run=exact_run, roundings=roundings):
            """The reference's fp32 outputs and its torch float64 truth, Dew's float64 path held to that
            truth at this seed: the recording keeps the reference's evidence alone, and the test's truth
            is Dew's float64 path (`indexed_recording`'s `truth_fn`)."""
            tensors = {name: torch.from_numpy(value) for name, value in inputs.items()}
            with torch.no_grad():
                narrow = reference(site, tensors)
                with float64():
                    exact = reference(site, {name: value.double() for name, value in tensors.items()})
            with jax.enable_x64(new_val=True):
                twin = jax.tree.map(np.asarray, exact_run(widened(inputs)))
            for name, value in exact.items():
                assert_computes_the_oracle(twin[name], value, f"transformers float64 {site} {name}",
                                           roundings=roundings)
            return narrow, exact

        stored.update(record_case(site, draw, evaluate, lambda inputs, site=site: mamba2_run(site, inputs),
                                  store_inputs=False, summary=True))
    stored["provenance"] = provenance(
        f"transformers {transformers.__version__}", TRANSFORMERS_REVISION, "cpu",
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba2_reference.py seeded",
        truth="transformers reference in torch float64 through tools/torch_precision.py")
    destination = SSD / "ssd_seeded.npz"
    written = save(destination, stored, reference=f"transformers {transformers.__version__}",
                   license="Apache-2.0")
    print(f"wrote {destination.name}: {written} ({written.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["seeded"]:
        write_seeded()
    elif sys.argv[1:] == ["float64"]:
        write_float64()
    else:
        write_ssd()
        write_float64()
        write_mamba2_tiny()
