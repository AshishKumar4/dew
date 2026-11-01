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

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from hf_reference import FIXTURES, transformers_record, write_tiny
from torch_precision import float64
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2ForCausalLM,
    Mamba2Mixer,
    mamba2_chunk_scan,
    mamba2_selective_state_update,
)

SSD = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2"
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
        "transformers": transformers_record(),
        "seed": SEED,
    }, indent=1) + "\n")
    (directory / "model.safetensors").chmod(0o644)
    print(f"{directory}: source.json written")


def _stepped(forward, length: int, prefill: int) -> np.ndarray:
    """`model_support.stepped_outputs` in NumPy, a float64 run's outputs kept wide: `forward(n)` over the
    first `n` positions, a prefill of `prefill` positions, then one position a step."""
    calls = [(0, prefill), *((position, position + 1) for position in range(prefill, length))]
    return np.concatenate([forward(hi)[:, lo:hi] for lo, hi in calls], axis=1)


def _layer_config() -> Mamba2Config:
    """One `Mamba2Mixer` at the layer fixture's geometry (`ssd_support.LAYER_GEOMETRY`)."""
    from ssd_support import LAYER_GEOMETRY as g

    return Mamba2Config(hidden_size=g["hidden_size"], expand=2, num_heads=g["num_heads"],
                        head_dim=g["head_dim"], state_size=g["state_size"], n_groups=g["n_groups"],
                        conv_kernel=g["conv_kernel"], chunk_size=g["chunk_size"], num_hidden_layers=1,
                        vocab_size=32, layer_norm_epsilon=g["layer_norm_epsilon"])


def _mixer(arrays, dtype) -> Mamba2Mixer:
    """Transformers' mixer on a drawn layer's weights (`ssd_support.mamba2_layer_draw`), in `dtype`."""
    from ssd_support import LAYER_SHAPES

    layer = Mamba2Mixer(_layer_config(), layer_idx=0).eval()
    layer.load_state_dict({name: torch.from_numpy(value) for name, value in arrays.items()
                           if name in LAYER_SHAPES and name != "hidden"})
    return layer.to(dtype)


def scan_reference(site, inputs, dtype) -> dict[str, np.ndarray]:
    """Transformers' scan on a draw's native inputs (`ssd_support.mamba2_draw`): 'scan' and 'carried' its
    chunk scan from zero or from the drawn state, 'step' its selective step's first token, 'starts' its chunk
    scan on each document alone (`ssd_support.SCAN_STARTS`), each row's first from the drawn state."""
    from ssd_support import SCAN_STARTS, documents_of

    t = {name: torch.from_numpy(value).to(dtype) for name, value in inputs.items()}
    if site == "step":
        state = t["h0"].clone()
        output = mamba2_selective_state_update(
            state, t["x"][:, 0], t["dt"][:, 0][..., None].expand(-1, -1, P),
            t["A"][:, None, None].expand(-1, P, N), t["B"][:, 0], t["C"][:, 0], t["D"][:, None].expand(-1, P),
            dt_bias=t["dt_bias"][:, None].expand(-1, P), dt_softplus=True)
        return {"output": output.numpy(), "final": state.numpy()}

    def scanned(rows, span, initial):
        pieces = [t[name][rows, span] for name in ("x", "dt", "B", "C")]
        return mamba2_chunk_scan(pieces[0], pieces[1], t["A"], pieces[2], pieces[3], chunk_size=CHUNK,
                                 D=t["D"], dt_bias=t["dt_bias"], dt_softplus=True, initial_states=initial,
                                 return_final_states=True)

    if site != "starts":
        output, state = scanned(slice(None), slice(None), t["h0"] if site == "carried" else None)
        return {"output": output.numpy(), "final": state.numpy()}
    outputs, finals = [], []
    for row, spans in enumerate(documents_of(SCAN_STARTS)):
        pieces = []
        for index, (start, end) in enumerate(spans):
            piece, final = scanned(slice(row, row + 1), slice(start, end),
                                   t["h0"][row:row + 1] if index == 0 else None)
            pieces.append(piece)
        outputs.append(torch.cat(pieces, dim=1))
        finals.append(final)
    return {"output": torch.cat(outputs).numpy(), "final": torch.cat(finals).numpy()}


def layer_reference(site, inputs, dtype) -> dict[str, np.ndarray]:
    """Transformers' mixer on a drawn layer as each site compares it (`ssd_support.mamba2_layer_site_run`'s
    shapes): 'layer' the mixer over the hidden states, 'layer_bf16' with its weights cast too, 'decoded' its
    forwards at a cached decode's widths, 'padded' each row closed, 'packed' each document alone, 'tangent'
    its JVP along the drawn direction (`torch.func.jvp`)."""
    import ssd_support as support

    # The bf16 site's narrow run in bfloat16 end to end; its truth, as every site's, in `dtype` (float64).
    dtype = torch.bfloat16 if site == "layer_bf16" and dtype == torch.float32 else dtype
    layer = _mixer(inputs, dtype)
    hidden = torch.from_numpy(inputs["hidden"]).to(dtype)

    def on(x):
        return layer(x).double().numpy()

    if site in ("layer", "layer_bf16"):
        return {"output": layer(hidden).double().numpy()}
    if site == "decoded":
        return {"output": _stepped(lambda length: on(hidden[:, :length]), hidden.shape[1], support.PREFILL)}
    if site == "padded":
        second = hidden[1, :12][torch.from_numpy(support.PAUSED[1])][None]
        return {"first": on(hidden[:1, :12])[0], "second": on(second)[0]}
    if site == "packed":
        return {"output": np.concatenate([on(hidden[:1, start:end])
                                          for start, end in itertools.pairwise(support.CUTS)], axis=1)}
    direction = torch.from_numpy(inputs["direction"]).to(dtype)
    tangent = torch.func.jvp(layer, (hidden[:1, :support.TANGENT_TOKENS],), (direction,))[1]
    return {"output": tangent.double().numpy()}


def model_reference(site, inputs, dtype) -> dict[str, np.ndarray]:
    """Transformers' `Mamba2ForCausalLM` of `ssd_support.HF_TINY`'s architecture on a model draw, its weights
    Dew's tree exported (`Pretrained.export`): 'model_padded' and 'model_scored' the final states of each
    document of the padded row alone over its real tokens, the scored row's first 12; 'model_decoded' the
    logits at a cached decode's widths, from the fourth."""
    import ssd_support as support

    tensors = support.mamba2_model().export(support.mamba2_model_variables(inputs))
    model = Mamba2ForCausalLM(Mamba2Config.from_pretrained(support.HF_TINY)).eval()
    model.load_state_dict({name: torch.from_numpy(np.asarray(value)) for name, value in tensors.items()})
    model = model.to(dtype)
    ids = torch.from_numpy(inputs["ids"].astype(np.int64))
    if site == "model_decoded":
        def logits(length):
            return model(ids[:, :length]).logits.double().numpy()

        return {"logits": _stepped(logits, 12, 4)[:, 3:]}
    length = {"model_padded": 16, "model_scored": 17}[site]
    valid, segments = support.padded_layout(length)
    real, closed = ids[:1, :length][:, torch.from_numpy(valid[0])], segments[:, valid[0]]
    if site == "model_scored":
        real, closed = real[:, :12], closed[:, :12]
    starts = np.concatenate([[True], closed[0, 1:] != closed[0, :-1]])[None]
    return {"hidden": np.concatenate([model.backbone(real[:, start:end]).last_hidden_state[0].double().numpy()
                                      for start, end in support.documents_of(starts)[0]])}


PAIRED = frozenset({"starts", "padded"})
"""The sites sized paired with Dew's arm (`record_case`'s `dew`, Dew's twin in the inputs' dtype), which
transformers compared with itself resolved within no 512 draws: the resets' final state, only the last
document's tokens writing it, and the padded rows' outputs, twelve tokens and seven (m2-seeded-rec4 at
c868cbbc6)."""


def _recorded(site, draw, twin, reference) -> dict[str, np.ndarray]:
    """A site's recording (`record_case`): `reference(site, inputs, dtype)` in fp32 and in torch float64 on
    each draw, Dew's float64 `twin(site, inputs)` held to the latter at every seed (the test's truth, Dew's
    float64 path, `indexed_recording`'s `truth_fn`); sized paired with Dew's arm at `PAIRED`'s sites."""
    import jax
    from recorded_references import record_case
    from reference_error import assert_computes_the_oracle, chain_roundings, widened

    exact_run = jax.jit(lambda inputs: twin(site, inputs))
    with jax.enable_x64(new_val=True):
        roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(draw(0))))

    def evaluate(inputs):
        with torch.no_grad():
            narrow = reference(site, inputs, torch.float32)
            with float64():
                exact = reference(site, inputs, torch.float64)
        with jax.enable_x64(new_val=True):
            found = jax.tree.map(np.asarray, exact_run(widened(inputs)))
        for name, value in exact.items():
            assert_computes_the_oracle(found[name], value, f"transformers float64 {site} {name}",
                                       roundings=roundings)
        return narrow, exact

    if site in PAIRED:
        return record_case(site, draw, evaluate, store_inputs=False, summary=True,
                           dew=jax.jit(lambda inputs: twin(site, inputs)))
    return record_case(site, draw, evaluate, lambda inputs: twin(site, inputs), store_inputs=False,
                       summary=True)


def write_seeded() -> None:
    """Fresh draws with transformers' fp32 and torch-float64 outputs at each site: the scan's native inputs
    under an orbit ('scan', 'carried', 'step', 'starts'), a mixer's weights and hidden states ('layer',
    'layer_bf16', 'decoded', 'padded', 'packed', 'tangent') and a model's ('model_padded', 'model_scored',
    'model_decoded').

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/mamba2_reference.py seeded
    """
    import ssd_support as support
    from recorded_references import provenance, save

    def draw(seed):
        inputs = support.mamba2_draw(seed, (B, S, H, P, G, N))
        redrawn = support.mamba2_draw(seed)
        assert inputs.keys() == redrawn.keys(), f"transformers seed {seed}: regenerated inputs differ"
        for name, value in inputs.items():
            other = redrawn[name]
            assert (value.dtype == other.dtype and value.shape == other.shape
                    and value.tobytes() == other.tobytes()), f"transformers seed {seed}: {name} differs"
        return inputs

    stored = {}
    references = {**dict.fromkeys(support.SCAN_SITES, scan_reference),
                  **dict.fromkeys(support.LAYER_SITES, layer_reference),
                  **dict.fromkeys(support.MODEL_SITES, model_reference)}
    for site, reference in references.items():
        inputs_of, twin = support.mamba2_site(site)
        # The scan's draws checked against their regeneration at the fixture's geometry.
        stored.update(_recorded(site, draw if site in support.SCAN_SITES else inputs_of, twin, reference))
    stored["provenance"] = provenance(
        f"transformers {transformers.__version__}", transformers_record()["revision"], "cpu",
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
