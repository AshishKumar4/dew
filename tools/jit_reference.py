"""JiT's bottleneck patch embedding by its official code, for tests/fixtures/jit.

LTH14/JiT's `BottleneckPatchEmbed` (model_jit.py, read at a pinned commit,
the class extracted and run as published) embeds a batch of pixels a seed
(`draw`, which the test draws again); its weights land in Flax's layout, and
its tokens over check 1's seeds as the reference's evidence in float32 against
its float64 run (`recorded_references.record_case`, summarized).

    PYTHONPATH=src python tools/jit_reference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from reference_source import definitions

MODEL = "https://raw.githubusercontent.com/LTH14/JiT/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/model_jit.py"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "jit"


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's two 16x16 images, channels last, standard normal."""
    return {"pixels": np.random.default_rng([0x317, seed]).standard_normal((2, 16, 16, 3)).astype(np.float32)}


def main() -> None:
    from recorded_references import record_case

    scope = definitions(MODEL, ("BottleneckPatchEmbed",), {"nn": torch.nn})
    torch.manual_seed(0)
    embed = scope["BottleneckPatchEmbed"](img_size=16, patch_size=8, in_chans=3, pca_dim=5, embed_dim=12)
    arrays = {"Conv_0.kernel": embed.proj1.weight.detach().double().numpy().transpose(2, 3, 1, 0),
              "Dense_0.kernel": embed.proj2.weight.detach().double().numpy()[:, :, 0, 0].T,
              "Dense_0.bias": embed.proj2.bias.detach().double().numpy()}

    def evaluate(inputs):
        pixels = torch.from_numpy(inputs["pixels"].transpose(0, 3, 1, 2).copy())
        with torch.no_grad():
            return ({"tokens": embed.float()(pixels).numpy()},
                    {"tokens": embed.double()(pixels.double()).numpy()})

    arrays.update(record_case("bottleneck", draw, evaluate, store_inputs=False, summary=True))
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "bottleneck.npz", **arrays)
    print(f"{FIXTURE}: a bottleneck patch embedding of 2 images")


if __name__ == "__main__":
    main()
