"""One all-parameter SGD step through a native multimodal family's wrapper over drawn media, for check 1
(tests/test_multimodal_model.py), whose reference tools/multimodal_reference.py's `sgd` records over the
seeds (a summarized recording, `recorded`): the draw both read (`drawn`), Dew's step at a dtype
(`stepped`) and the leaves check 1 judges of it (`judged`)."""

import dataclasses
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from multimodal_families import FAMILIES

from dew.interop.pretrained import Pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import ModelConfig
from tools.multimodal_reference import LEARNING_RATE

FIXTURES = Path(__file__).parent / "fixtures"


def recorded(name: str) -> Path:
    """`name`'s recorded reference over the seeds, its one site named `sgd`."""
    return FIXTURES / "multimodal_sgd" / f"{name}.npz"


@dataclasses.dataclass(frozen=True)
class Family:
    """A family's float32 source, the prompts its rows hold, the shape of its rows' three images and its
    checkpoint's audio clips (None where it takes no audio)."""

    loaded: Pretrained
    prompts: list[str]
    images: tuple[int, ...]
    audio: list[np.ndarray] | None


@functools.cache
def family_of(name: str) -> Family:
    references, checkpoint = FAMILIES[name]
    source = FIXTURES / "hf" / checkpoint
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    prompts = json.loads((FIXTURES / "hf" / references / "prompts.json").read_text())
    images = np.load(FIXTURES / "hf" / references / "raw_images.npy")
    assert images.dtype == np.uint8, images.dtype
    clips = [np.load(path) for path in sorted(source.glob("waveform_*.npy"))]
    return Family(loaded, prompts, images.shape, clips or None)


def media(name: str, seed: int) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """Seed `seed`'s media for `name`'s prompts: three images of uniform bytes, one in the first row and
    two in the second, and, for a family that takes audio, a normal clip at each of its checkpoint's
    clips' lengths and scale."""
    held = family_of(name)
    rng = np.random.default_rng([0x3332, seed])
    images = rng.integers(0, 256, held.images, dtype=np.uint8)
    clips = None if held.audio is None else [
        (rng.standard_normal(clip.shape) * clip.std()).astype(clip.dtype) for clip in held.audio]
    return images, clips


def rows(images: np.ndarray) -> list[list[np.ndarray]]:
    """The images as the prompts' rows take them: the first alone, the other two together."""
    return [[images[0]], [images[1], images[2]]]


@functools.cache
def _layout(name: str):
    """The tree of `name`'s model inputs at every seed (the prompts fix every shape), read off seed 0's."""
    images, clips = media(name, 0)
    held = family_of(name)
    return jax.tree_util.tree_structure(held.loaded.processor(held.prompts, images=rows(images), audio=clips))


def drawn(name: str):
    """Check 1's draw for `name`: seed `seed`'s media (`media`) and its model inputs through the family's own
    processor, as host arrays by name, which the recording's reference encodes from the media and Dew's
    step reads as model inputs (`inputs_of`)."""
    held = family_of(name)

    def draw(seed: int) -> dict[str, np.ndarray]:
        images, clips = media(name, seed)
        inputs = held.loaded.processor(held.prompts, images=rows(images), audio=clips)
        leaves = {f"inputs.{index}": np.asarray(leaf) for index, leaf in enumerate(jax.tree.leaves(inputs))}
        audio = {f"audio.{index}": clip for index, clip in enumerate(clips or ())}
        return {"images": images, **audio, **leaves}
    return draw


def inputs_of(name: str, arrays) -> ModelInputs:
    """The model inputs a draw holds (`drawn`)."""
    layout = _layout(name)
    return jax.tree_util.tree_unflatten(layout, [arrays[f"inputs.{index}"]
                                                 for index in range(layout.num_leaves)])


@functools.cache
def _real(name: str) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """`name`'s valid token slots, each row's image count and its images' grids (None without), the same
    at every seed."""
    inputs = inputs_of(name, drawn(name)(0))
    grid = inputs.conditioning.get("image_grid_thw")
    return (np.asarray(inputs.token_fields["attention_mask"]),
            np.asarray(inputs.conditioning["image_lengths"]), None if grid is None else np.asarray(grid))


def judged(real, loss, pixels, logits) -> dict:
    """The leaves check 1 judges of a step (`real`, its family's `_real`): its loss, the pixel gradient of
    each real image in the reference's flat image order, without padding, and the logits after the step at
    the valid token slots."""
    valid, lengths, grid = real
    if grid is None:
        real = jnp.concatenate([pixels[row, :int(count)] for row, count in enumerate(lengths)])
    else:
        real = jnp.concatenate([pixels[row, image, :int(np.prod(grid[row, image]))]
                                for row, count in enumerate(lengths) for image in range(int(count))])
    return {"loss": jnp.reshape(loss, (1,)), "pixel_gradient": real, "updated_logits": logits[valid]}


@functools.cache
def _wide(name: str):
    """`name`'s model and variables in float64 (`ModelConfig.with_dtype`: a wrapper's towers too)."""
    loaded = family_of(name).loaded
    with jax.enable_x64(new_val=True):
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                 if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, loaded.variables)
    return ModelConfig.from_model(loaded.model).with_dtype("float64").build(), variables


def first_layer_signed(gradient, sign):
    """`gradient` with its first decoder layer's subtree (the first `layers_0` a walk meets) times `sign`:
    at -1, a sign slip in one layer's backward that leaves every shape alone; at 1, the gradient's own."""
    def walk(node):
        if not isinstance(node, dict):
            return node, False
        if "layers_0" in node:
            return {**node, "layers_0": jax.tree.map(lambda leaf: leaf * sign, node["layers_0"])}, True
        out, found = {}, False
        for key, value in node.items():
            out[key], hit = walk(value) if not found else (value, False)
            found = found or hit
        return out, found

    signed, found = walk(gradient)
    assert found, "no layers_0 subtree in the gradient"
    return signed


def stepped(name: str, *, wide: bool = False):
    """Dew's step on a draw's arrays (`drawn`): the summed next-token loss over the rows, its gradient in
    every parameter and the pixels, one SGD step of LEARNING_RATE on every parameter, its first decoder
    layer's gradient times `sign` (`first_layer_signed`), and the logits after it, as `judged` leaves, with
    the variables it trains; with `wide`, in float64, the truth."""
    loaded = family_of(name).loaded
    model, variables = _wide(name) if wide else (loaded.model, loaded.variables)
    # Seed 0's inputs read through the processor on the host, before any trace reads them.
    real = _real(name)

    def step(arrays, sign=1.0):
        inputs = inputs_of(name, arrays)
        objective = LMObjective(model, inputs.tokens.shape[1] - 1, ema_decay=None, pad_id=0)
        params = variables["params"]

        def loss(params, pixels):
            data = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
            statistics, _ = objective.loss({**variables, "params": params}, {"text": data},
                                           Step(step=jnp.int32(0), key=jax.random.key(4), ema=None))
            return objective.reduce_loss(statistics)[0]

        value, (gradient, pixels) = jax.value_and_grad(loss, argnums=(0, 1))(
            params, inputs.conditioning["pixel_values"])
        optimizer = optax.sgd(LEARNING_RATE)
        updates, _ = optimizer.update(first_layer_signed(gradient, sign), optimizer.init(params), params)
        trained = {**variables, "params": optax.apply_updates(params, updates)}
        return judged(real, value, pixels, model.apply(trained, inputs.tokens, **inputs.kwargs())), trained
    return step
