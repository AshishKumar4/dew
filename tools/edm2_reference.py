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

    PYTHONPATH=src python tools/edm2_reference.py
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


if __name__ == "__main__":
    main()
