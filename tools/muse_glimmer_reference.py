"""Muse Glimmer's vision trunk and projector at a tiny scale, run by transformers 5.16.1, for
tests/test_vision.py: the reference config, the trunk's and projector's tensors as the release names
them (each name its prefix off), packed pixels with each row's patch grid, and `refs.npz`: the trunk
after ln_post in merge-block order, as Dew's tower returns it, and the adapter, projection and
scale-free norm's output, fp32 and float64, IEEE products. The first row is an NCHW image packed by
the image processor's own patchify, which the test feeds Dew as the image.

    python tools/muse_glimmer_reference.py
"""

from types import SimpleNamespace

import numpy as np
import torch
from torch_precision import float64
from mimo_reference import FIXTURES, packed_rows, padded, scatter, write
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionAdapter,
    MuseGlimmerVisionModel,
    get_vision_pixel_shuffle_index,
)

MUSE_VISION = dict(
    num_hidden_layers=4, hidden_size=32, intermediate_size=48, num_attention_heads=4, hidden_act="gelu",
    patch_size=2, patch_temporal=2, merge_size=2, pos_emb_height=4, pos_emb_width=4, layer_norm_eps=1e-5,
    layer_types=["window_attention", "window_attention", "full_attention", "window_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0})
MUSE_WRAPPER = dict(out_hidden_size=128, projector_hidden_size=24, projector_hidden_act="gelu")
MUSE_TEXT = dict(hidden_size=16, rms_norm_eps=1e-6)
MUSE_GRIDS = ((1, 6, 10), (1, 8, 4))
"""4-patch windows over a 6x10 grid, some partial, and an 8x4 one; the first row is an NCHW image
packed by the processor's own patchify."""


def write_muse_glimmer_vision_tiny() -> None:
    """Muse Glimmer's trunk (after ln_post, before the pixel shuffle the projector takes) and its
    adapter, projection and scale-free norm; the first row packed from an NCHW image by the image
    processor's patchify, which the test feeds Dew as the image."""
    vconf = MuseGlimmerVisionConfig(**MUSE_VISION)
    vconf._attn_implementation = "eager"
    wrapper = SimpleNamespace(**MUSE_WRAPPER)

    def build():
        return (MuseGlimmerVisionModel(vconf).eval(), MuseGlimmerVisionAdapter(wrapper).eval(),
                torch.nn.Linear(MUSE_WRAPPER["projector_hidden_size"], MUSE_TEXT["hidden_size"], bias=False),
                MuseGlimmerRMSNorm(eps=MUSE_TEXT["rms_norm_eps"], with_scale=False))

    torch.manual_seed(1234)
    tower, adapter, projection, norm = build()
    for index, module in enumerate((tower, adapter, projection)):
        scatter(module, 1234 + index)
    width = 3 * MUSE_VISION["patch_temporal"] * MUSE_VISION["patch_size"] ** 2
    batch, rows = packed_rows(MUSE_GRIDS, width, 13)
    _, height, wide = MUSE_GRIDS[0]
    patch = MUSE_VISION["patch_size"]
    image = np.random.RandomState(17).rand(1, 3, height * patch, wide * patch).astype(np.float32)
    packed, _, _ = MuseGlimmerImageProcessor.patchify(None, torch.from_numpy(image), patch,
                                                       MUSE_VISION["patch_temporal"])
    rows[0] = packed[0].numpy()
    batch[0, :len(rows[0])] = rows[0]

    def project(modules, features):
        _, adapt, linear, rms = modules
        return rms(linear(adapt(features)))

    def run(modules, dtype):
        trunks, outputs = [], []
        captured = {}
        hook = modules[0].ln_post.register_forward_hook(lambda module, inputs, output: captured.update(x=output))
        with torch.no_grad():
            for row, grid in zip(rows, MUSE_GRIDS, strict=True):
                shuffled = modules[0](torch.from_numpy(row).to(dtype), torch.tensor([grid])).last_hidden_state
                # The trunk in merge-block order, each block's patches consecutive, as Dew's tower returns it.
                order = get_vision_pixel_shuffle_index(torch.tensor([grid]), MUSE_VISION["merge_size"])
                trunks.append(captured["x"][order].numpy())
                outputs.append(project(modules, shuffled).numpy())
        hook.remove()
        return padded(trunks), padded(outputs)

    def shuffled(trunk):
        """The pixel shuffle of a merge-block-ordered trunk: each block's features channel by channel."""
        block = MUSE_VISION["merge_size"] ** 2
        return trunk.view(-1, block, trunk.shape[-1]).permute(0, 2, 1).reshape(-1, block * trunk.shape[-1])

    modules = (tower, adapter, projection, norm)
    trunk, merged = run(modules, torch.float32)
    with float64():
        widened = build()
        for source, target in zip(modules, widened, strict=True):
            target.load_state_dict({name: tensor.double() for name, tensor in source.state_dict().items()})
        trunk_f64, _ = run(widened, torch.float64)
        with torch.no_grad():
            merged_f64 = padded([project(widened, shuffled(torch.from_numpy(t[:len(r)]).double())).numpy()
                                 for t, r in zip(trunk.astype(np.float32), rows, strict=True)])
    write(FIXTURES / "muse-glimmer-vision-tiny",
          {"model_type": "muse_glimmer", "vision_config": vconf.to_dict(), **MUSE_WRAPPER,
           "text_config": MUSE_TEXT},
          tower.state_dict(),
          {**{f"vision_adapter.{name}": tensor for name, tensor in adapter.state_dict().items()},
           "vision_projection.weight": projection.weight.detach()},
          {"pixels": batch, "_grid": np.asarray(MUSE_GRIDS, np.int32), "_image": image, "tower": trunk,
           "tower_f64": trunk_f64, "projector": merged, "projector_f64": merged_f64})


def main() -> None:
    write_muse_glimmer_vision_tiny()


if __name__ == "__main__":
    main()
