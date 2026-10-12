"""MiMo-V2.6's and Muse Glimmer's vision towers and projectors at a tiny scale, run by their references.

The fixtures tests/test_vision.py holds Dew's shared `VisionTransformer` and its projectors to. MiMo's
reference is the vision half of XiaomiMiMo/MiMo-V2.6-Pro-MOPD's `modeling_mimo_v2.py`, fetched at
`MIMO_REVISION`; Muse Glimmer's is transformers 5.16.1's. Each directory holds the reference config,
the tower's and projector's tensors as the release names them (each name its prefix off), packed
pixels with each row's patch grid, and `refs.npz`: the trunk sequence and the projector's output in
fp32 and float64, IEEE products. The trunk is compared over each row's own patches; the projector
runs on the fp32 trunk, as the merger reads it.

    python tools/vit_reference.py
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers_wan_reference import float64
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
from transformers.models.muse_glimmer.image_processing_muse_glimmer import MuseGlimmerImageProcessor
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerVisionAdapter,
    MuseGlimmerVisionModel,
    get_vision_pixel_shuffle_index,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
MIMO_REPO = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"
MIMO_REVISION = "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8"

MIMO_VISION = dict(
    depth=4, hidden_size=32, intermediate_size=48, num_heads=4, num_key_value_heads=2, qk_channels=8,
    in_chans=3, patch_size=4, spatial_merge_size=2, temporal_patch_size=2, hidden_act="silu",
    rms_norm_eps=1e-6, fullatt_block_indexes=[0, 3], vit_window_attn_types=[-1, 0, 1, -1], use_sink=True,
    visual_token_window_size=3, out_hidden_size=16)
"""Two band layers, one in each token order, between two full ones, a band narrower than a frame's
rows, and sinks on the band layers."""

MIMO_GRIDS = ((1, 4, 6), (2, 4, 4))
"""A 4x6 image and a two-frame 4x4 clip, so the band crosses rows and the frames stay apart."""

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


def scatter(module: torch.nn.Module, seed: int) -> None:
    """Something in every tensor: norms near one, biases and sinks small, maps random."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in module.named_parameters():
            if "norm" in name or "ln_" in name:
                tensor.copy_(1.0 + torch.randn(tensor.shape, generator=generator) * 0.05)
            elif "bias" in name or "sinks" in name:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.2)
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.4)


def mimo_module():
    """MiMo's modeling module, fetched at MIMO_REVISION and imported as a package beside its config."""
    package = types.ModuleType("mimo_v2")
    package.__path__ = []
    sys.modules["mimo_v2"] = package
    for name in ("configuration_mimo_v2", "modeling_mimo_v2"):
        path = hf_hub_download(MIMO_REPO, f"{name}.py", revision=MIMO_REVISION)
        spec = importlib.util.spec_from_file_location(f"mimo_v2.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"mimo_v2.{name}"] = module
        spec.loader.exec_module(module)
    return sys.modules["mimo_v2.modeling_mimo_v2"]


def packed_rows(grids, width: int, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random packed pixels per row, `[patches, width]` each, and them padded into one batch."""
    generator = np.random.RandomState(seed)
    rows = [generator.randn(int(np.prod(grid)), width).astype(np.float32) for grid in grids]
    batch = np.zeros((len(rows), max(len(row) for row in rows), width), np.float32)
    for index, row in enumerate(rows):
        batch[index, :len(row)] = row
    return batch, rows


def padded(outputs: list[np.ndarray]) -> np.ndarray:
    """Per-row outputs `[n, ...]` as one zero-padded batch."""
    batch = np.zeros((len(outputs), max(len(x) for x in outputs), *outputs[0].shape[1:]), np.float64)
    for index, x in enumerate(outputs):
        batch[index, :len(x)] = x
    return batch


def write(directory: Path, config: dict, tower: dict, projector: dict, arrays: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    save_file({name: tensor.contiguous() for name, tensor in tower.items()}, directory / "model.safetensors")
    save_file({name: tensor.contiguous() for name, tensor in projector.items()}, directory / "projector.safetensors")
    np.savez(directory / "refs.npz", precision="ieee",
             **{name: np.asarray(value, np.float64 if name.endswith("_f64") else np.float32)
                for name, value in arrays.items() if not name.startswith("_")},
             **{name[1:]: value for name, value in arrays.items() if name.startswith("_")})
    print(directory, sorted(p.name for p in directory.iterdir()))


def write_mimo_vision_tiny() -> None:
    """MiMo ViT: the trunk sequence (the merger's input, caught by a hook) and the merger's output,
    the merger's biases zero as the release stores none."""
    modeling = mimo_module()
    config = SimpleNamespace(**MIMO_VISION)
    torch.manual_seed(1234)
    tower = modeling.MiMoVisionTransformer(config).float().eval()
    scatter(tower, 1234)
    with torch.no_grad():
        for name, tensor in tower.merger.named_parameters():
            if name.endswith("bias"):
                tensor.zero_()
    width = 3 * MIMO_VISION["temporal_patch_size"] * MIMO_VISION["patch_size"] ** 2
    batch, rows = packed_rows(MIMO_GRIDS, width, 11)

    def run(model, dtype):
        trunks, merged = [], []
        caught = {}
        hook = model.merger.register_forward_hook(lambda module, inputs, output: caught.update(x=inputs[0]))
        with torch.no_grad():
            for row, grid in zip(rows, MIMO_GRIDS, strict=True):
                merged.append(model(torch.from_numpy(row).to(dtype), torch.tensor([grid])).numpy())
                trunks.append(caught["x"].numpy())
        hook.remove()
        return padded(trunks), padded(merged)

    trunk, merged = run(tower, torch.float32)
    with float64():
        wide = modeling.MiMoVisionTransformer(config).eval()
        wide.load_state_dict({name: tensor.double() for name, tensor in tower.state_dict().items()})
        trunk_f64, _ = run(wide, torch.float64)
        with torch.no_grad():
            merged_f64 = padded([wide.merger(torch.from_numpy(t[:len(r)]).double()).numpy()
                                 for t, r in zip(trunk.astype(np.float32), rows, strict=True)])
    state = tower.state_dict()
    write(FIXTURES / "mimo-vision-tiny",
          {"model_type": "mimo_v2", "vision_config": MIMO_VISION, "text_config": {"hidden_size": 16}},
          {name: tensor for name, tensor in state.items() if not name.startswith("merger.")},
          {name: tensor for name, tensor in state.items()
           if name.startswith("merger.") and not name.endswith("bias")},
          {"pixels": batch, "_grid": np.asarray(MIMO_GRIDS, np.int32), "tower": trunk, "tower_f64": trunk_f64,
           "projector": merged, "projector_f64": merged_f64})


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
    torch.backends.cuda.matmul.allow_tf32 = False
    write_mimo_vision_tiny()
    write_muse_glimmer_vision_tiny()


if __name__ == "__main__":
    main()
