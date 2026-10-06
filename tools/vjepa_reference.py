#!/usr/bin/env python3
"""Write the JEPA fixture with V-JEPA's own encoder and predictor.

facebookresearch/jepa's model sources, fetched at commit
51c59d518fc63c08464af6de585f78ac0c7ed4d5 and checked against their SHA-256,
build tiny image-mode (`num_frames=1`) encoders and mask-token predictors
whose every trainable parameter is moved off its initialization (and rounded
to bfloat16-representable values, so the fixture compresses). The encoder
embeds a batch of images restricted to their context patches; the predictor
takes those embeddings and predicts the target patches, as a JEPA step calls
them. The tool records both outputs and, against fixed cotangents on both,
the gradients of the images and of every parameter: once in float32 and
once in float64 (`diffusers_wan_reference.widened`, under which the sincos
tables are generated in float64 too), the truth tests/reference_error.py
measures both from. The blocks run V-JEPA's own exact (erf) GELU, as Dew's
JEPA does.

Cases:
- `published`: V-JEPA's `vit_*`/`vit_predictor` construction (LayerNorm
  epsilon 1e-6, MLP ratio 4), narrowed;
- `variant`: the constructors' own LayerNorm epsilon (1e-5), MLP ratio 2,
  odd head counts and a two-channel image.

Run with the Dew test environment's torch, on CPU:

    python tools/vjepa_reference.py OUTPUT.npz
"""

import hashlib
import importlib.util
import json
import sys
import types
import urllib.request
from functools import cache, partial
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

REPO, COMMIT = "facebookresearch/jepa", "51c59d518fc63c08464af6de585f78ac0c7ed4d5"
# In import order: each module's own imports come before it.
FILES = {
    "src/utils/tensors.py": "774a0f81eea52d2fcc4373818d8aab04e790c14f029e37e415ec0e5f6df2a8cb",
    "src/masks/utils.py": "2297fb30aabdbc635a7b9e9793806cebec62367f032d4b9bd57fbb4e1681a381",
    "src/models/utils/modules.py": "b4e2f4259a2255173d1059865691e7e45974d121781601035c44c589a1bf62e7",
    "src/models/utils/patch_embed.py": "8167c7a9a9a15dfff81a0e3b8cdfcbfab6881cc0da396cc130c4729ccba75c11",
    "src/models/utils/pos_embs.py": "3fff7d09a8e99eaaca0c3e436ecc5794dc9e62245fe3e5489e8cd37a23f771a8",
    "src/models/vision_transformer.py": "3d2dd861d535434c4202c1b488436433fc545d558899a48c392c240b4afe8abf",
    "src/models/predictor.py": "2a2d6febea26d5a670445f97bc9bc4c726617ee3d02a38f9aeff3b4b6db5a816",
}
CACHE = Path.home() / ".cache" / "dew" / "upstream" / "jepa" / COMMIT
CASES = {
    "published": {"img_size": 8, "patch_size": 2, "in_chans": 3, "embed_dim": 32, "depth": 2, "num_heads": 4,
                  "mlp_ratio": 4, "eps": 1e-6, "predictor_embed_dim": 16, "pred_depth": 2,
                  "pred_num_heads": 2, "context": 9, "targets": 4},
    "variant": {"img_size": 12, "patch_size": 3, "in_chans": 2, "embed_dim": 24, "depth": 3, "num_heads": 3,
                "mlp_ratio": 2, "eps": 1e-5, "predictor_embed_dim": 20, "pred_depth": 1,
                "pred_num_heads": 5, "context": 7, "targets": 5},
}
BATCH = 2
SEED = 53


@cache
def jepa():
    """V-JEPA's `src` at `COMMIT`, under its own package names."""
    for package in ("src", "src.models", "src.models.utils", "src.masks", "src.utils"):
        module = types.ModuleType(package)
        module.__path__ = []
        sys.modules[package] = module
    for path, digest in FILES.items():
        local = CACHE / path
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}"
            local.write_bytes(urllib.request.urlopen(url).read())
        found = hashlib.sha256(local.read_bytes()).hexdigest()
        if found != digest:
            raise RuntimeError(f"{local} has SHA-256 {found}, not the pinned {digest}")
        name = path.removesuffix(".py").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, local)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["src.models.vision_transformer"], sys.modules["src.models.predictor"]


def build(config: dict):
    """The case's encoder and predictor."""
    vision_transformer, predictor = jepa()
    norm = partial(torch.nn.LayerNorm, eps=config["eps"])
    encoder = vision_transformer.VisionTransformer(
        img_size=config["img_size"], patch_size=config["patch_size"], num_frames=1,
        in_chans=config["in_chans"], embed_dim=config["embed_dim"], depth=config["depth"],
        num_heads=config["num_heads"], mlp_ratio=config["mlp_ratio"], qkv_bias=True, norm_layer=norm)
    head = predictor.VisionTransformerPredictor(
        img_size=config["img_size"], patch_size=config["patch_size"], num_frames=1,
        embed_dim=config["embed_dim"], predictor_embed_dim=config["predictor_embed_dim"],
        depth=config["pred_depth"], num_heads=config["pred_num_heads"], mlp_ratio=config["mlp_ratio"],
        qkv_bias=True, norm_layer=norm, use_mask_tokens=True, num_mask_tokens=1,
        zero_init_mask_tokens=False)
    return encoder.eval(), head.eval()


def trainable(encoder, predictor) -> list[tuple[str, torch.nn.Parameter]]:
    """Every parameter but the two fixed sincos tables, named by model."""
    return [(f"{prefix}.{name}", value) for prefix, model in (("encoder", encoder), ("predictor", predictor))
            for name, value in model.named_parameters() if value.requires_grad]


def walk(encoder, predictor, inputs, probes, dtype):
    """Both outputs at `dtype`, and the gradients of `sum(context * probe) +
    sum(predictions * probe)` with respect to the images and every
    trainable parameter."""
    encoder, predictor = encoder.to(dtype), predictor.to(dtype)
    with torch.no_grad():
        encoder._init_pos_embed(encoder.pos_embed.data)
        predictor._init_pos_embed(predictor.predictor_pos_embed.data)
    images = inputs["images"].to(dtype).requires_grad_()
    context_idx, target_idx = inputs["context_idx"], inputs["target_idx"]
    named = trainable(encoder, predictor)
    context = encoder(images, masks=[context_idx])
    predictions = predictor(context, None, [context_idx], [target_idx])
    objective = ((context * probes["context"].to(dtype)).sum()
                 + (predictions * probes["predictions"].to(dtype)).sum())
    grads = torch.autograd.grad(objective, [images] + [value for _, value in named])
    arrays = {"context": context, "predictions": predictions, "grad_images": grads[0]}
    for (name, _), gradient in zip(named, grads[1:], strict=True):
        arrays[f"grad_param.{name}"] = gradient
    return {key: value.detach().numpy() for key, value in arrays.items()}


def main():
    from diffusers_wan_reference import widened

    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    torch.set_num_threads(2)
    arrays: dict[str, np.ndarray] = {}
    for case, config in CASES.items():
        torch.manual_seed(SEED)
        encoder, predictor = build(config)
        generator = torch.Generator().manual_seed(SEED + 1)
        with torch.no_grad():
            for _, parameter in trainable(encoder, predictor):
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
                parameter.copy_(parameter.to(torch.bfloat16).float())
        patches = (config["img_size"] // config["patch_size"]) ** 2
        # Each example's context and targets are disjoint, in no particular order.
        orders = [torch.randperm(patches, generator=generator) for _ in range(BATCH)]
        inputs = {
            "images": torch.randn((BATCH, config["in_chans"], config["img_size"], config["img_size"]),
                                  generator=generator),
            "context_idx": torch.stack([order[:config["context"]] for order in orders]),
            "target_idx": torch.stack([order[-config["targets"]:] for order in orders]),
        }
        probes = {
            "context": torch.randn((BATCH, config["context"], config["embed_dim"]), generator=generator),
            "predictions": torch.randn((BATCH, config["targets"], config["embed_dim"]), generator=generator),
        }
        record = {key: value.numpy() for key, value in inputs.items()}
        record.update({f"probe_{key}": value.numpy() for key, value in probes.items()})
        record.update({f"param.{name}": value.detach().numpy().astype(ml_dtypes.bfloat16).view(np.uint16)
                       for name, value in trainable(encoder, predictor)})
        single = walk(encoder, predictor, inputs, probes, torch.float32)
        record.update({f"fp32.{key}": value for key, value in single.items()})
        with widened():
            truth = walk(encoder, predictor, inputs, probes, torch.float64)
        record.update({f"fp64.{key}": value for key, value in truth.items()})
        arrays.update({f"{case}/{key}": value for key, value in record.items()})
        gap = max(np.abs(record[f"fp32.{key}"] - record[f"fp64.{key}"]).max()
                  for key in ("context", "predictions"))
        print(f"{case}: fp32 off float64 by {gap:.3g}")
    meta = {"repo": REPO, "commit": COMMIT, "files": FILES, "cases": CASES, "batch": BATCH,
            "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(sys.argv[1], **arrays)
    print(f"{sys.argv[1]}: {Path(sys.argv[1]).stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays")


if __name__ == "__main__":
    main()
