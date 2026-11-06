"""Kimi K3's media bundle read whole (row 855): its MoonViT, with K3's options (RMS norms at the epsilon of
the dtype the tower computes in, unbiased maps, an attention wider than the tower), and its patch merger
(PatchMergerMLPV2: unbiased maps and an RMS norm after them), against the release's own
(moonshotai/Kimi-K3 at f831ab6, modeling_kimi_k3.py), recorded by tools/venv_reference.py's
`kimi-k3-vision` site (tools/kimi_k3_vision_worker.py) over kimi-k3-tiny's tower and projector. No model
weights are downloaded at test time."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_runs import assert_as_exact_as_recorded

from dew.interop import Pretrained
from dew.nn.multimodal import MultimodalTransformer

TINY = Path(__file__).parent / "fixtures" / "hf" / "kimi-k3-tiny"
FIXTURE = TINY / "vision_recording.npz"
GRIDS = {"image": (1, 4, 6), "video": (2, 4, 4)}
"""Each drawn item's `(t, h, w)` grid in 14-pixel patches: an image of six merge blocks, its table resampled
from the fixture's 4x4, and a two-frame video, its frames pooled into four (`sd2_tpool`)."""
OUTPUTS = tuple(GRIDS)
"""Each item's projected merge blocks, the site's recorded leaves."""
SEEDS = 8
"""The declared seeds, sized by `seeds_for` paired as the recording confirmed them."""


def draw(seed: int) -> dict[str, np.ndarray]:
    """A seed's items: each grid's patches `[t*h*w, 3, 14, 14]`, a stream of its own for each seed."""
    rng = np.random.default_rng([0x855, 3, seed])
    return {kind: rng.normal(size=(t * h * w, 3, 14, 14)).astype(np.float32)
            for kind, (t, h, w) in GRIDS.items()}


_LOADED: dict[str, tuple[MultimodalTransformer, dict]] = {}


def dew(dtype, inputs) -> dict[str, jax.Array]:
    """Dew's bundle at `dtype` (`Pretrained.load`, loaded once in float32 and widened), each item's
    projected merge blocks: its patches through the bundle's tower and projector, as a row's media
    conditioning carries them (`VisionConditioner`). Widened, its RMS norms keep the float32 run's
    epsilon (`MoonViT.rms_eps`), as the release's twin does (tools/kimi_k3_vision_worker.py): the truth is
    the function the float32 run computes."""
    name = jnp.dtype(dtype).name
    if name not in _LOADED:
        # Loaded where its weights stay concrete, though its first call traces (`recorded_dew`'s pattern).
        with jax.ensure_compile_time_eval():
            if "float32" not in _LOADED:
                source = Pretrained.load(TINY, dtype="float32", param_dtype="float32",
                                         attention_impl="reference")
                assert isinstance(source.model, MultimodalTransformer)
                _LOADED["float32"] = source.model, source.variables
            model, variables = _LOADED["float32"]
            vision = dataclasses.replace(model.vision, rms_eps=float(jnp.finfo(jnp.float32).eps))
            _LOADED[name] = model.clone(dtype=jnp.dtype(name), vision=vision), jax.tree.map(
                lambda leaf: jnp.asarray(leaf, name) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
                variables)
    model, variables = _LOADED[name]
    found = {}
    for kind, (t, h, w) in GRIDS.items():
        conditioning = {"pixel_values": jnp.asarray(inputs[kind], name).reshape(1, 1, t * h * w, -1),
                        "image_grid_thw": jnp.asarray([[[t, h, w]]], jnp.int32)}
        features = model.apply(variables, conditioning,
                               method=lambda module, given: module.conditioner(given))
        # The item's merge blocks, its frames pooled into its first frame's places.
        found[kind] = features[0, :h * w // 4]
    return found


def test_kimi_k3_reads_its_media_through_its_tower_and_merger_on_par_with_its_release():
    """Kimi K3 loads whole (row 855): its MoonViT, attending over an item's every frame and pooling them,
    with RMS norms at its computing dtype's epsilon, and its PatchMergerMLPV2 give the release's projected
    merge blocks for a drawn image and a drawn two-frame video, on par with the release's float32 run from
    its float64 twin (check 1, tools/venv_reference.py kimi-k3-vision), Dew's float64 run the truth."""
    assert_as_exact_as_recorded(FIXTURE, "kimi-k3-vision", SEEDS, draw, dew,
                                "Kimi K3's projected media")


def test_a_kimi_k3_bundle_loads_whole_and_its_language_model_alone_on_request():
    """A Kimi K3 bundle loads whole, its tower and projector read into the model (`MultimodalTransformer`),
    and `text_only` takes its language model alone, the decoder the text tests hold to the release, every
    tensor of it read the same."""
    whole = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    alone = Pretrained.load(TINY, dtype="float32", attention_impl="reference", text_only=True)
    assert isinstance(whole.model, MultimodalTransformer)
    assert not isinstance(alone.model, MultimodalTransformer)
    assert {"tower", "projector"} <= set(whole.variables["params"])
    language = whole.variables["params"]["language_model"]
    for (path, ours), theirs in zip(jax.tree_util.tree_leaves_with_path(language),
                                    jax.tree.leaves(alone.variables["params"]), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs),
                                      err_msg=jax.tree_util.keystr(path))


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32", "float64"])
def test_an_rms_norms_default_epsilon_is_the_one_torch_takes_for_its_input(dtype):
    """An RMS norm stating no epsilon (K3's `RMSNorm(eps=None)`, `RMSNorm.epsilon` None) takes the one
    torch's rms_norm takes for an input of `dtype`, the module stating no dtype of its own: over inputs
    small enough that the epsilon dominates, torch's norm at None is its norm at float32's machine
    epsilon for a half, bfloat16 or float32 input and float64's for a float64 one, bit for bit, where its
    input's own differs for a half or bfloat16 one; and Dew's norm at None is its norm at that epsilon."""
    import torch

    from dew.nn.norms import RMSNorm

    values = np.full((2, 16), 2.0 ** -12)
    values[1] = np.linspace(-1, 1, 16) * 2.0 ** -10
    narrow = getattr(torch, dtype)
    x = torch.tensor(values, dtype=narrow)
    taken = torch.finfo(torch.float64 if dtype == "float64" else torch.float32).eps
    default = torch.nn.functional.rms_norm(x, (16,), eps=None)
    assert torch.equal(default, torch.nn.functional.rms_norm(x, (16,), eps=taken))
    own = torch.nn.functional.rms_norm(x, (16,), eps=torch.finfo(narrow).eps)
    assert torch.equal(default, own) == (dtype in ("float32", "float64"))
    with jax.enable_x64(new_val=True):
        ours = jnp.asarray(values, dtype)
        variables = RMSNorm(epsilon=None).init(jax.random.key(0), ours)
        np.testing.assert_array_equal(np.asarray(RMSNorm(epsilon=None).apply(variables, ours), np.float64),
                                      np.asarray(RMSNorm(epsilon=taken).apply(variables, ours), np.float64))
