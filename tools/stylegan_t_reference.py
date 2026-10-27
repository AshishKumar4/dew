"""StyleGAN-T's discriminator head by its official code, for tests/fixtures/stylegan_t.

autonomousvision/stylegan-t's `DiscHead`, `make_block`, `BatchNormLocal`
and `SpectralConv1d` (networks/discriminator.py) and `ResidualBlock` and
`FullyConnectedLayer` (networks/shared.py), read at a pinned commit and run
as published in training mode, score a batch of token sequences under a
condition. LADD reshapes the sequence to its 2D grid; at a grid height of one
a (1, k) kernel is the 1D kernel k, which is what the fixture compares. What
lands is the inputs, every weight in Flax's layout, the spectral norms'
`u` vectors before the call, and the logits, in float64: built, drawn and run inside
tools/torch_precision.py's `float64`, so every float32 pin of the published code is widened too.

    PYTHONPATH=src python tools/stylegan_t_reference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from reference_source import definitions
from torch.nn.utils.spectral_norm import SpectralNorm
from torch_precision import float64

ROOT = "https://raw.githubusercontent.com/autonomousvision/stylegan-t/36ab80ce76237fefe03e65e9b3161c040ae888e3/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "stylegan_t"
BATCH, LENGTH, CHANNELS, CONDITION = 16, 12, 8, 6


def main() -> None:
    with float64():
        written()


def written() -> None:
    """The fixture's arrays, written: run inside `float64` (`main`)."""
    scope = {"torch": torch, "nn": torch.nn, "np": np, "SpectralNorm": SpectralNorm, "F": torch.nn.functional,
             "misc": None, "Callable": object, "Any": object}
    definitions(ROOT + "networks/shared.py", ("ResidualBlock", "FullyConnectedLayer"), scope)
    definitions(ROOT + "networks/discriminator.py",
                ("SpectralConv1d", "BatchNormLocal", "make_block", "DiscHead"), scope)
    torch.manual_seed(0)
    head = scope["DiscHead"](CHANNELS, CONDITION).train()
    x = torch.randn(BATCH, CHANNELS, LENGTH)
    c = torch.randn(BATCH, CONDITION)
    arrays = {"x": x.permute(0, 2, 1).numpy()[:, None], "c": c.numpy()}
    for name, module in head.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            arrays[f"{name}.kernel"] = module.weight_orig.detach().numpy().transpose(2, 1, 0)[None]
            arrays[f"{name}.bias"] = module.bias.detach().numpy()
            arrays[f"{name}.u"] = module.weight_u.detach().numpy().copy()
        elif type(module).__name__ == "BatchNormLocal":
            arrays[f"{name}.weight"] = module.weight.detach().numpy()
            arrays[f"{name}.bias"] = module.bias.detach().numpy()
    arrays["cmapper.weight"] = head.cmapper.weight.detach().numpy().T * head.cmapper.weight_gain
    arrays["cmapper.bias"] = head.cmapper.bias.detach().numpy() * head.cmapper.bias_gain
    with torch.no_grad():
        arrays["logits"] = head(x, c)[:, 0].numpy()[:, None]
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "head.npz", **arrays)
    print(f"{FIXTURE}: a DiscHead over {BATCH} sequences of {LENGTH} tokens, {sorted(arrays)}")


if __name__ == "__main__":
    main()
