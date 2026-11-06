"""EDM2's U-Net and its magnitude-preserving layers over seeds, for tests/fixtures/edm2/seeds.npz.

The reference is `UNet` in NVlabs/edm2's `training/networks_edm2.py`, read at
a pinned commit and executed as published, with its two `torch_utils`
imports stubbed: `persistence.persistent_class` is only pickling metadata
and `misc.const_like` a tensor constructor. Its code is not vendored here
(the repository's licence is CC BY-NC-SA 4.0).

A tiny network (8x8 input, two levels, attention at 4x4, a 4-wide label):
a seed draws its weights as the reference initializes them, and every scalar
gain the reference initializes to zero too, so no branch is switched off,
with its inputs and a probe (`draw`, which the test draws again; the names
and shapes from layout.json, which this writes). Over the seeds check 1 draws
(`record_case`, tools/recorded_references.py), two sites, each in float32 and
in the float64 twin, summarized (each leaf's error from its truth and its
truth's sketch): `unet`, the output in eval mode; and `train`, the same
network trained in train mode, where `MPConv.forward` writes the normalized
weight back into the parameter before using it (forced weight normalization,
Equation 66), for STEPS steps of torch's SGD on `sum(output * probe)`: each
step's output, and after the first every stored weight and gain in the
test's layout (`converted`), in one vector. SGD rather than EDM2's Adam: the
normalization does not depend on the optimizer, and Adam's first steps are
near sign(gradient) steps, which turn a near-zero gradient's float32 rounding
into a whole step's difference.

    PYTHONPATH=src:tests python tools/edm2_reference.py
"""

from __future__ import annotations

import contextlib
import json
import types
from pathlib import Path

import jax
import numpy as np
import torch
from reference_source import fetched, stub_torch_utils
from torch_precision import float64

COMMIT = "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"
SOURCE = f"https://raw.githubusercontent.com/NVlabs/edm2/{COMMIT}/training/networks_edm2.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm2" / "seeds.npz"
LAYOUT = FIXTURE.with_name("layout.json")
"""The reference's state, each name's shape, in its order: what `draw` draws."""
STEPS = 3
# A rate that moves the weights and the output visibly in STEPS steps.
SGD = {"lr": 0.002}
CONFIG = {"img_resolution": 8, "img_channels": 3, "label_dim": 4, "model_channels": 8,
          "channel_mult": [1, 2], "num_blocks": 1, "attn_resolutions": [4],
          "channels_per_head": 4}


def networks(dtype: str) -> types.ModuleType:
    """The published module, with the two `torch_utils` names it imports,
    computing in `dtype`: float32 as published, and its float64 twin with the
    float32 casts it makes retargeted (its norms' `vector_norm(dtype=...)`
    among them, which the shared `float64` context does not widen), whose
    distance from the first is the rounding error a float32 evaluation makes.
    The twin is built inside `float64`, so its constructor computes in
    float64 too."""
    stub_torch_utils()
    module = types.ModuleType("networks_edm2")
    source = fetched(SOURCE)
    exec(source.replace("torch.float32", f"torch.{dtype}"), module.__dict__)
    return module


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's weights under their reference names (`weights/`), as `UNet` initializes them: each
    `MPConv` weight standard normal, each `MPFourier` frequency 2 pi times a standard normal and phase 2 pi
    times a uniform, and each scalar gain normal about 1 at 0.5, where the reference's zero would switch
    its branch off; and the inputs, the noise labels, a 4-wide text vector and the probe, all float32."""
    rng = np.random.default_rng([1572, seed])
    arrays = {}
    for name, shape in json.loads(LAYOUT.read_text()).items():
        if name.endswith(".freqs"):
            value = 2 * np.pi * rng.standard_normal(shape)
        elif name.endswith(".phases"):
            value = 2 * np.pi * rng.uniform(size=shape)
        elif not shape:
            value = rng.normal(1.0, 0.5, shape)
        else:
            value = rng.standard_normal(shape)
        arrays[f"weights/{name}"] = value
    arrays |= {"x": rng.standard_normal((2, 3, 8, 8)), "noise_labels": rng.standard_normal(2),
               "text": rng.standard_normal((2, 4)), "probe": rng.standard_normal((2, 3, 8, 8))}
    return {name: np.asarray(value, np.float32) for name, value in arrays.items()}


def built(published, twin, inputs: dict[str, np.ndarray], mode: str):
    """The reference in float32 and its float64 twin, each holding the draw's weights, in `mode`."""
    state = {name.removeprefix("weights/"): torch.as_tensor(value) for name, value in inputs.items()
             if name.startswith("weights/")}
    net = getattr(published.UNet(**CONFIG), mode)()
    net.load_state_dict(state)
    with float64():
        exact = getattr(twin.UNet(**CONFIG), mode)().double()
    exact.load_state_dict({name: value.double() for name, value in state.items()})
    return net, exact


def labels(twin, inputs: dict[str, np.ndarray]) -> torch.Tensor:
    """The class labels that reach the embedding a text vector does: the model normalizes a text vector
    to unit magnitude where the reference multiplies its one-hot by sqrt(label_dim)."""
    return twin.normalize(torch.as_tensor(inputs["text"]).double()) / np.sqrt(CONFIG["label_dim"])


def unet(published, twin, inputs: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """The `unet` site's output in float32 and float64, in eval mode."""
    net, exact = built(published, twin, inputs, "eval")
    x, noise = (torch.as_tensor(inputs[name]) for name in ("x", "noise_labels"))
    label = labels(twin, inputs)
    with torch.no_grad():
        single = net(x, noise, label.float()).numpy()
    with float64(), torch.no_grad():
        wide = exact(x.double(), noise.double(), label).numpy()
    return {"output": single}, {"output": wide}


def train(published, twin, inputs: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """The `train` site in float32 and float64: STEPS SGD steps in train mode, each step's output and,
    after the first, the stored weights and gains in the test's layout (`converted`) in one vector."""
    from test_edm2 import converted

    nets = built(published, twin, inputs, "train")
    runs = []
    for net, dtype in zip(nets, (torch.float32, torch.float64), strict=True):
        optimizer = torch.optim.SGD(net.parameters(), **SGD)
        x, noise, probe = (torch.as_tensor(inputs[name]).to(dtype) for name in ("x", "noise_labels", "probe"))
        label = labels(twin, inputs).to(dtype)
        out = {}
        with float64() if dtype == torch.float64 else contextlib.nullcontext():
            for step in range(STEPS):
                output = net(x, noise, label)
                out[f"output_{step}"] = output.detach().numpy().copy()
                if step:
                    # In float64 where the run is: `converted` holds its arrays as JAX does.
                    with jax.enable_x64(dtype == torch.float64):
                        stored = converted({f"weights/{name}": value.numpy().copy()
                                            for name, value in net.state_dict().items()})["params"]
                        out[f"stored_{step}"] = np.concatenate([np.ravel(leaf) for leaf in
                                                                jax.tree_util.tree_leaves(stored)])
                optimizer.zero_grad()
                (output * probe).sum().backward()
                optimizer.step()
        runs.append(out)
    return runs[0], runs[1]


def main() -> None:
    from recorded_references import provenance, record_case, save
    from test_edm2 import trained, unet_output

    published, twin = networks("float32"), networks("float64")
    LAYOUT.write_text(json.dumps({name: list(value.shape) for name, value in
                                  published.UNet(**CONFIG).state_dict().items()}, indent=1) + "\n")
    arrays = {}
    for site, evaluate, dew in (("unet", unet, unet_output), ("train", train, trained)):
        compiled = jax.jit(lambda inputs, dew=dew: dew(jax.numpy.float32, inputs))
        arrays.update(record_case(site, draw,
                                  lambda inputs, evaluate=evaluate: evaluate(published, twin, inputs),
                                  store_inputs=False, summary=True, dew=compiled))
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "NVlabs/edm2", COMMIT, "cpu", "python tools/edm2_reference.py", torch=torch.__version__)},
        reference=f"NVlabs/edm2 {COMMIT}", license="CC BY-NC-SA 4.0 (results only)")
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
