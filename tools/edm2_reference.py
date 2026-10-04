"""EDM2's U-Net and its magnitude-preserving layers, for tests/fixtures/edm2.

The reference is `UNet` in NVlabs/edm2's `training/networks_edm2.py`, read at
a pinned commit and executed as published, with its two `torch_utils`
imports stubbed: `persistence.persistent_class` is only pickling metadata
and `misc.const_like` a tensor constructor. Its code is not vendored here
(the repository's licence is CC BY-NC-SA 4.0).

A tiny network (8x8 input, two levels, attention at 4x4, a 4-wide label)
draws its weights, and every scalar gain the reference initializes to zero is
drawn too, so no branch is switched off. What lands: every weight and buffer
under its reference name, the inputs, the class labels as the reference
reads them, and the output computed in float64 and in float32.

`--train` writes train.npz beside it: the same network trained in train
mode, where `MPConv.forward` writes the normalized weight back into the
parameter before using it (forced weight normalization, Equation 66), for
STEPS steps of torch's SGD on `sum(output * probe)`. Each step records the
output and every stored weight and gain after its forward, in float32 and
in the float64 twin, from one initial state. SGD rather than EDM2's Adam:
the normalization does not depend on the optimizer, and Adam's first steps
are near sign(gradient) steps, which turn a near-zero gradient's float32
rounding into a whole step's difference.

    PYTHONPATH=src python tools/edm2_reference.py [--train]
"""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np
import torch

COMMIT = "4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc"
SOURCE = f"https://raw.githubusercontent.com/NVlabs/edm2/{COMMIT}/training/networks_edm2.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm2" / "unet.npz"
TRAIN = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "edm2" / "train.npz"
STEPS = 3
# A rate that moves the weights and the output visibly in STEPS steps.
SGD = {"lr": 0.002}
CONFIG = {"img_resolution": 8, "img_channels": 3, "label_dim": 4, "model_channels": 8,
          "channel_mult": [1, 2], "num_blocks": 1, "attn_resolutions": [4],
          "channels_per_head": 4}


def networks(dtype: str) -> types.ModuleType:
    """The published module, with the two `torch_utils` names it imports,
    computing in `dtype`: float32 as published, and its float64 twin with the
    float32 casts it makes retargeted, whose distance from the first is the
    rounding error a float32 evaluation makes."""
    persistence = types.ModuleType("torch_utils.persistence")
    persistence.persistent_class = lambda cls: cls
    misc = types.ModuleType("torch_utils.misc")
    misc.const_like = lambda ref, value: torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    package = types.ModuleType("torch_utils")
    package.persistence, package.misc = persistence, misc
    sys.modules.update({"torch_utils": package, "torch_utils.persistence": persistence,
                        "torch_utils.misc": misc})
    module = types.ModuleType("networks_edm2")
    source = urllib.request.urlopen(SOURCE).read().decode()
    exec(source.replace("torch.float32", f"torch.{dtype}"), module.__dict__)
    return module


def main() -> None:
    published, twin = networks("float32"), networks("float64")
    torch.manual_seed(0)
    net = published.UNet(**CONFIG).eval()
    with torch.no_grad():
        for value in net.parameters():
            if value.ndim == 0:
                value.copy_(torch.randn(()) * 0.5 + 1.0)
    exact = twin.UNet(**CONFIG).eval().double()
    exact.load_state_dict({name: value.double() for name, value in net.state_dict().items()})
    generator = torch.Generator().manual_seed(1)
    x, noise_labels, text = (torch.randn(*shape, generator=generator)
                             for shape in ((2, 3, 8, 8), (2,), (2, 4)))
    # The model normalizes a text vector to unit magnitude where the reference
    # multiplies its one-hot by sqrt(label_dim), so this is the label that
    # reaches the same embedding.
    labels = twin.normalize(text.double()) / np.sqrt(CONFIG["label_dim"])
    arrays = {"x": x.numpy(), "noise_labels": noise_labels.numpy(), "text": text.numpy(),
              "config": np.asarray(json.dumps(CONFIG))}
    for name, value in net.state_dict().items():
        arrays[f"weights/{name}"] = value.numpy()
    with torch.no_grad():
        arrays["output"] = exact(x.double(), noise_labels.double(), labels).numpy()
        arrays["output32"] = net(x, noise_labels, labels.float()).double().numpy()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE, **arrays)
    print(f"{FIXTURE}: output |max| {np.abs(arrays['output']).max():.3f}")


def train() -> None:
    published, twin = networks("float32"), networks("float64")
    torch.manual_seed(0)
    initial = published.UNet(**CONFIG)
    with torch.no_grad():
        for value in initial.parameters():
            if value.ndim == 0:
                value.copy_(torch.randn(()) * 0.5 + 1.0)
    generator = torch.Generator().manual_seed(2)
    x, noise_labels, text = (torch.randn(*shape, generator=generator)
                             for shape in ((2, 3, 8, 8), (2,), (2, 4)))
    probe = torch.randn((2, 3, 8, 8), generator=generator)
    arrays = {"x": x.numpy(), "noise_labels": noise_labels.numpy(), "text": text.numpy(),
              "probe": probe.numpy(), "config": np.asarray(json.dumps(CONFIG)),
              "sgd": np.asarray(json.dumps(SGD)), "steps": np.asarray(STEPS)}
    for name, value in initial.state_dict().items():
        arrays[f"weights/{name}"] = value.numpy()
    for precision, module, dtype in (("fp32", published, torch.float32), ("fp64", twin, torch.float64)):
        net = module.UNet(**CONFIG).train().to(dtype)
        net.load_state_dict({name: value.to(dtype) for name, value in initial.state_dict().items()})
        optimizer = torch.optim.SGD(net.parameters(), **SGD)
        labels = twin.normalize(text.double()).to(dtype) / np.sqrt(CONFIG["label_dim"])
        for step in range(STEPS):
            output = net(x.to(dtype), noise_labels.to(dtype), labels)
            arrays[f"{precision}/{step}/output"] = output.detach().numpy()
            for name, value in net.state_dict().items():
                arrays[f"{precision}/{step}/weights/{name}"] = value.numpy().copy()
            optimizer.zero_grad()
            (output * probe.to(dtype)).sum().backward()
            optimizer.step()
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRAIN, **arrays)
    print(f"{TRAIN}: {TRAIN.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    train() if sys.argv[1:] == ["--train"] else main()
