"""JiT's bottleneck patch embedding by its official code, for tests/fixtures/jit.

LTH14/JiT's `BottleneckPatchEmbed` (model_jit.py, read at a pinned commit,
the class extracted and run as published) embeds a batch of pixels; its
weights land in Flax's layout beside the tokens, in float64 and float32.

    PYTHONPATH=src python tools/jit_reference.py
"""

from __future__ import annotations

import ast
import urllib.request
from pathlib import Path

import numpy as np
import torch

MODEL = "https://raw.githubusercontent.com/LTH14/JiT/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/model_jit.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jit"


def main() -> None:
    text = urllib.request.urlopen(MODEL).read().decode()
    scope = {"nn": torch.nn}
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "BottleneckPatchEmbed":
            exec(ast.get_source_segment(text, node), scope)
    torch.manual_seed(0)
    embed = scope["BottleneckPatchEmbed"](img_size=16, patch_size=8, in_chans=3, pca_dim=5, embed_dim=12)
    pixels = torch.randn(2, 3, 16, 16)
    arrays = {"pixels": pixels.numpy().transpose(0, 2, 3, 1),
              "Conv_0.kernel": embed.proj1.weight.detach().double().numpy().transpose(2, 3, 1, 0),
              "Dense_0.kernel": embed.proj2.weight.detach().double().numpy()[:, :, 0, 0].T,
              "Dense_0.bias": embed.proj2.bias.detach().double().numpy()}
    with torch.no_grad():
        arrays["tokens"] = embed.double()(pixels.double()).numpy()
        arrays["tokens32"] = embed.float()(pixels).double().numpy()
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "bottleneck.npz", **arrays)
    print(f"{FIXTURE}: a bottleneck patch embedding of 2 images")


if __name__ == "__main__":
    main()
