"""DreamerV3's image encoder and decoder (`ConvEncoder`, `ConvDecoder`) against the code's own
(danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c, dreamerv3/rssm.py `Encoder` and
`Decoder`, configs.yaml's `simple` types at the 1M size): over drawn weights, images and latents, Dew's
float32 tokens and pixels are as exact as the code's float32 run against its float64 run (check 1,
tools/dreamerv3_reference.py's recording), Dew's float64 run held to that run's recorded sketch."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax.traverse_util import unflatten_dict
from reference_runs import assert_as_exact_as_recorded

from dew.objectives.rl.dreamer import ConvDecoder, ConvEncoder

FIXTURE = Path(__file__).parent / "fixtures" / "dreamerv3" / "conv.npz"
IMAGE, DEPTH, UNITS, DETER, STOCH, CLASSES, BLOCKS = (64, 64, 3), 4, 64, 512, 32, 4, 8
CONV_SEEDS = 8
"""The declared seeds, sized by `seeds_for` paired as the recording confirmed them."""
OUTPUTS = ("tokens", "pixels")
"""The leaves `dew` gives, the site's recorded leaves."""


def layout() -> dict[str, tuple[int, ...]]:
    """The code's parameters by ninjax path, Dew's by the same names (tools/dreamerv3_worker.py)."""
    depths = [DEPTH * mult for mult in (2, 3, 4, 4)]
    shapes = {}
    for stage, depth in enumerate(depths):
        inputs = IMAGE[-1] if stage == 0 else depths[stage - 1]
        shapes |= {f"enc/cnn{stage}/kernel": (5, 5, inputs, depth), f"enc/cnn{stage}/bias": (depth,),
                   f"enc/cnn{stage}norm/scale": (depth,)}
    size = 4 * 4 * depths[-1]
    shapes |= {"dec/sp0/kernel": (BLOCKS, DETER // BLOCKS, size // BLOCKS),
               "dec/sp0/bias": (size,),
               "dec/sp1/kernel": (STOCH * CLASSES, 2 * UNITS), "dec/sp1/bias": (2 * UNITS,),
               "dec/sp1norm/scale": (2 * UNITS,), "dec/sp2/kernel": (2 * UNITS, size),
               "dec/sp2/bias": (size,),
               "dec/spnorm/scale": (depths[-1],), "dec/imgout/kernel": (5, 5, depths[0], IMAGE[-1]),
               "dec/imgout/bias": (IMAGE[-1],)}
    for stage in range(len(depths) - 1):
        shapes |= {f"dec/conv{stage}/kernel": (5, 5, depths[stage + 1], depths[stage]),
                   f"dec/conv{stage}/bias": (depths[stage],), f"dec/conv{stage}norm/scale": (depths[stage],)}
    return shapes


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's weights at their initializer's scale, the norms' about one and the biases off zero, two
    rows of three steps of images, deterministic states in (-1, 1) and one-hot categoricals."""
    rng = np.random.default_rng([532, 1, seed])
    drawn = {}
    for path, shape in layout().items():
        if path.endswith("scale"):
            value = rng.normal(1.0, 0.3, shape)
        elif path.endswith("bias"):
            value = rng.normal(0.0, 0.3, shape)
        else:
            value = rng.normal(0.0, 1.0, shape) / np.sqrt(np.prod(shape[:-1]))
        drawn[f"param.{path}"] = value.astype(np.float32)
    classes = rng.integers(0, CLASSES, (2, 3, STOCH))
    return {**drawn, "image": rng.integers(0, 256, (2, 3, *IMAGE)).astype(np.uint8),
            "deter": rng.uniform(-1, 1, (2, 3, DETER)).astype(np.float32),
            "stoch": np.eye(CLASSES, dtype=np.float32)[classes]}


def dew(dtype, inputs) -> dict:
    """Dew's encoder tokens and decoder pixels at `dtype` under a draw's weights."""
    def tree(prefix):
        held = {tuple(path.removeprefix(prefix).split("/")): jnp.asarray(value, dtype)
                for path, value in inputs.items() if path.startswith(prefix)}
        return {"params": unflatten_dict(held)}

    encoder = ConvEncoder(depth=DEPTH, dtype=dtype)
    decoder = ConvDecoder(IMAGE, depth=DEPTH, units=UNITS, blocks=BLOCKS, dtype=dtype)
    return {"tokens": encoder.apply(tree("param.enc/"), inputs["image"]),
            "pixels": decoder.apply(tree("param.dec/"), jnp.asarray(inputs["deter"], dtype),
                                    jnp.asarray(inputs["stoch"], dtype))}


def test_the_image_encoder_and_decoder_compute_dreamerv3s():
    """Dew's tokens and pixels as exact as the code's float32 against float64 (check 1)."""
    assert_as_exact_as_recorded(FIXTURE, "conv", CONV_SEEDS, draw, dew,
                                "DreamerV3's image encoder and decoder")
