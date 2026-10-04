#!/usr/bin/env python3
"""Write the DiT-family and UNet fixture with FlaxDiff's own models.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of.
Its model modules, fetched at commit 15c55b001304604147a2a8002a76cf5d4c32092f
and checked against their SHA-256, build tiny SimpleDiT, SimpleUDiT,
SimpleMMDiT, HierarchicalMMDiT, Unet, VideoDiT and UNet3D models whose every
parameter is moved off its initialization (and rounded to bfloat16-
representable values, so the fixture compresses). The tool records each
model's output on images (or clips), times and text states and, against a
fixed cotangent, the gradients of the image, the text and every parameter:
once in float32 and once in float64, the truth tests/reference_error.py
measures both from.

The float64 walk runs a second copy of the same files with every
`jnp.float32` reading `jnp.float64` (FlaxDiff pins its Fourier frequencies,
time input, RoPE tables, output heads and UNet attention to float32), under
`jax.enable_x64`. Every one of them is a dtype (a cast, a default, an
initializer's dtype) but one comparison, common.py's weight-standardized
convolution taking `eps = 1e-5 if self.dtype == jnp.float32 else 1e-3`:
each walk compares against its own dtype, so the branch stays the same
(and no case here builds that convolution). The 2D sincos table, which
FlaxDiff builds in float32 NumPy, stays the constant it is. Attention runs FlaxDiff's reference path
(`attention_impl=None`) with `force_fp32_for_softmax=False`: flax's forced
softmax casts to float32 whenever the dtype is not, which would round the
float64 walk, and in float32 it changes nothing.

Cases, two per model:
- `published`: the constructor's defaults, narrowed (raster scan, no
  qk-norm, MLP ratio 4; the Unet with attention at every level);
- `variant`: the other paths: a zigzag or Hilbert scan, qk-norm, MLP ratio
  2, odd head counts, a rectangular image, three hierarchy stages, and
  UNets with a level without attention, RMS norms, and stages that project
  in and out and run the full self, cross and feed-forward block (whose
  middle stage FlaxDiff projects with 1x1 convolutions).

The video models take three-frame clips in `published` and two in
`variant`.

Run with the Dew test environment, on CPU:

    python tools/flaxdiff_family_reference.py OUTPUT.npz
"""

import hashlib
import json
import os
import sys
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

REPO, COMMIT = "AshishKumar4/FlaxDiff", "15c55b001304604147a2a8002a76cf5d4c32092f"
# In import order: each module's own imports come before it.
FILES = {
    "hilbert": "24b7c9bf77539c0b15ca54ac372d93eba4cdd4fd8ce85e5e95a3ab3d9f7faff7",
    "s5": "531ac17e10cef3255ab03d607ff3abb977711b55b62a5379cc0347a993045093",
    "common": "d06be6c24cb015696d8621dd6d2a55b73f86817e0e3afdf86474740b60ad048b",
    "attention": "4954deaa1d05d25b7e551ddafb1687a2ece2d5abd6add54cbb74a645d62c4c7c",
    "vit_common": "665d8e2d248855b662a90d4a960b105171f02d3bed351bad0ca4128ef83dfb13",
    "dit_common": "9beefef6e13f06be1b955be880a77e21b7528d6eb21c729bc9053dc57e50c7b4",
    "simple_unet": "9964f37d44362e99343f0e1d198837e0d38007046b800d27a9d8064a32220d31",
    "simple_vit": "579d6e981142d8aab267e8e7d30ef50f90332090148b925feeda465aa17dbab6",
    "simple_dit": "c3064f67c0da98add4656ea74e3aa3dd4fbbdc52d96f053dc90c4abb44263b79",
    "simple_mmdit": "1d67ca79eb12bdb5cd4ca58994b3909a0c8c642e6073c21cfba507b2eb891a7b",
    "video_dit": "cecd0c05fa53f2078cf5dc7fcc5e55232be3cc9bb722de43429997616d377bac",
    "unet_3d": "ec1323014ec4cedcccac84c7dfe6af46a1dbed9bb1fd8a5a9493a4063916ea6a",
}
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "FlaxDiff" / COMMIT
ATTENTION = {"attention_impl": None, "force_fp32_for_softmax": False}
# case: (module, class, config, image [(T,) H, W, C], text [tokens, width])
CASES = {
    "simple_dit/published": ("simple_dit", "SimpleDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_dit/variant": ("simple_dit", "SimpleDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    "simple_udit/published": ("simple_vit", "SimpleUDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 4, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_udit/variant": ("simple_vit", "SimpleUDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "use_hilbert": True, **ATTENTION}, (8, 8, 3), (4, 7)),
    "simple_mmdit/published": ("simple_mmdit", "SimpleMMDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 2, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "simple_mmdit/variant": ("simple_mmdit", "SimpleMMDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    "hierarchical_mmdit/published": ("simple_mmdit", "HierarchicalMMDiT", {
        "output_channels": 4, "base_patch_size": 2, "emb_features": (12, 16), "num_layers": (1, 2),
        "num_heads": (2, 2), "mlp_ratio": 4, **ATTENTION}, (8, 8, 4), (5, 12)),
    "hierarchical_mmdit/variant": ("simple_mmdit", "HierarchicalMMDiT", {
        "output_channels": 3, "base_patch_size": 1, "emb_features": (8, 12, 16), "num_layers": (1, 1, 1),
        "num_heads": (2, 3, 4), "mlp_ratio": 2, "qk_norm": True, **ATTENTION}, (8, 12, 3), (4, 7)),
    # FlaxDiff's decoder upsamples level i to feature_depths[-i], Dew's to
    # feature_depths[-i - 2]: the two agree for two levels, or equal depths.
    "unet/published": ("simple_unet", "Unet", {
        "output_channels": 4, "emb_features": 16, "feature_depths": (8, 16),
        "attention_configs": ({"heads": 2}, {"heads": 2}), "num_res_blocks": 2, "num_middle_res_blocks": 1,
        "norm_groups": 4, "attention_impl": None}, (8, 8, 4), (5, 12)),
    "unet/variant": ("simple_unet", "Unet", {
        "output_channels": 3, "emb_features": 12, "feature_depths": (8, 8, 8),
        "attention_configs": ({"heads": 2, "use_projection": True, "only_pure_attention": False}, None,
                              {"heads": 4, "use_projection": True, "only_pure_attention": False}),
        "num_res_blocks": 1, "num_middle_res_blocks": 2, "norm_groups": 0, "attention_impl": None},
        (8, 8, 3), (4, 7)),
    "video_dit/published": ("video_dit", "VideoDiT", {
        "output_channels": 4, "patch_size": 2, "emb_features": 16, "num_layers": 1, "num_heads": 2,
        "mlp_ratio": 4, **ATTENTION}, (3, 8, 8, 4), (5, 12)),
    "video_dit/variant": ("video_dit", "VideoDiT", {
        "output_channels": 3, "patch_size": 2, "emb_features": 12, "num_layers": 2, "num_heads": 3,
        "mlp_ratio": 2, "qk_norm": True, "use_zigzag": True, **ATTENTION}, (2, 8, 12, 3), (4, 7)),
    "unet_3d/published": ("unet_3d", "UNet3D", {
        "output_channels": 4, "emb_features": 16, "feature_depths": (8, 16),
        "attention_configs": ({"heads": 2}, {"heads": 2}), "num_res_blocks": 2, "num_middle_res_blocks": 1,
        "norm_groups": 4, "temporal_heads": 2, "attention_impl": None}, (3, 8, 8, 4), (5, 12)),
    "unet_3d/variant": ("unet_3d", "UNet3D", {
        "output_channels": 3, "emb_features": 12, "feature_depths": (8, 8, 8),
        "attention_configs": ({"heads": 2, "use_projection": True, "only_pure_attention": False}, None,
                              {"heads": 4, "use_projection": True, "only_pure_attention": False}),
        "num_res_blocks": 1, "num_middle_res_blocks": 2, "norm_groups": 0, "temporal_heads": 4,
        "attention_impl": None}, (2, 8, 8, 3), (4, 7)),
}
TIMES = (-1.3, 0.9)
SEED = 59


def flaxdiff(wide: bool) -> dict[str, types.ModuleType]:
    """FlaxDiff's model modules at `COMMIT` as the package `flaxdiff.models`,
    with every `jnp.float32` read as `jnp.float64` when `wide`."""
    for name in [name for name in sys.modules if name.split(".")[0] == "flaxdiff"]:
        del sys.modules[name]
    for package in ("flaxdiff", "flaxdiff.models"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    modules = {}
    for stem, digest in FILES.items():
        local = CACHE / "flaxdiff" / "models" / f"{stem}.py"
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/flaxdiff/models/{stem}.py"
            local.write_bytes(urllib.request.urlopen(url).read())
        found = hashlib.sha256(local.read_bytes()).hexdigest()
        if found != digest:
            raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
        source = local.read_text()
        if wide:
            source = source.replace("jnp.float32", "jnp.float64")
        name = f"flaxdiff.models.{stem}"
        module = types.ModuleType(name)
        module.__package__, module.__file__ = "flaxdiff.models", str(local)
        sys.modules[name] = module
        exec(compile(source, str(local), "exec"), module.__dict__)
        setattr(sys.modules["flaxdiff.models"], stem, module)
        modules[stem] = module
    return modules


def build(case: str, *, wide: bool):
    module, cls, config, _, _ = CASES[case]
    return getattr(flaxdiff(wide)[module], cls)(**config)


def walk(case: str, params, inputs, probe, *, wide: bool) -> dict[str, np.ndarray]:
    """One call, in float64 when `wide`, and the gradients of `sum(output *
    probe)` with respect to the image, the text states and every parameter."""
    dtype = np.float64 if wide else np.float32
    model = build(case, wide=wide)
    params = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), params)
    image, text = (jnp.asarray(inputs[name], dtype) for name in ("image", "text"))
    times = jnp.asarray(inputs["times"], dtype)

    def forward(params, image, text):
        return model.apply({"params": params}, image, times, text)

    output, pullback = jax.vjp(forward, params, image, text)
    grad_params, grad_image, grad_text = pullback(jnp.asarray(probe, dtype))
    arrays = {"output": output, "grad_image": grad_image, "grad_text": grad_text}
    arrays.update({f"grad_param.{'/'.join(path)}": leaf for path, leaf in flatten_dict(grad_params).items()})
    return {key: np.asarray(value) for key, value in arrays.items()}


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    # One thread and no XNNPACK, before JAX's first call: otherwise the
    # CPU's float32 walk of the Unet rounds differently from run to run.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_use_xnnpack=false"
                               " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    arrays: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(SEED)
    for case, (_, _, _, image_shape, text_shape) in CASES.items():
        inputs = {"image": rng.standard_normal((len(TIMES), *image_shape), dtype=np.float32),
                  "times": np.asarray(TIMES, np.float32),
                  "text": rng.standard_normal((len(TIMES), *text_shape), dtype=np.float32)}
        model = build(case, wide=False)
        variables = model.init(jax.random.key(SEED), inputs["image"], inputs["times"], inputs["text"])
        params = {}
        for path, leaf in flatten_dict(variables["params"]).items():
            moved = np.asarray(leaf) + 0.1 * rng.standard_normal(leaf.shape, dtype=np.float32)
            params[path] = moved.astype(ml_dtypes.bfloat16).astype(np.float32)
        params = unflatten_dict(params)
        # Eagerly: FlaxDiff's scan orders read their permutations as NumPy.
        shape = model.apply({"params": params}, inputs["image"], inputs["times"], inputs["text"]).shape
        probe = rng.standard_normal(shape, dtype=np.float32)
        # The Fourier table FlaxDiff's FourierEmbedding computes, which Dew
        # holds as a constant.
        table = np.asarray(flaxdiff(wide=False)["common"].FourierEmbedding(
            features=model.emb_features[0] if isinstance(model.emb_features, tuple)
            else model.emb_features).bind({}).freqs)
        record = {**inputs, "probe": probe, "fourier_table": table}
        record.update({f"param.{'/'.join(path)}": leaf.astype(ml_dtypes.bfloat16).view(np.uint16)
                       for path, leaf in flatten_dict(params).items()})
        single = walk(case, params, inputs, probe, wide=False)
        record.update({f"fp32.{key}": value for key, value in single.items()})
        with jax.enable_x64(new_val=True):
            truth = walk(case, params, inputs, probe, wide=True)
        record.update({f"fp64.{key}": value for key, value in truth.items()})
        assert all(value.dtype == np.float64 for value in truth.values()), case
        arrays.update({f"{case}/{key}": value for key, value in record.items()})
        gap = np.abs(record["fp32.output"] - record["fp64.output"]).max()
        print(f"{case}: output {tuple(shape)}, fp32 off float64 by {gap:.3g}")
    meta = {"repo": REPO, "commit": COMMIT, "files": FILES,
            "cases": {case: {"module": module, "class": cls, "config": config}
                      for case, (module, cls, config, _, _) in CASES.items()},
            "times": TIMES, "jax": jax.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
