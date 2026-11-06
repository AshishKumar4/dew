"""V-JEPA's own encoder and predictor (facebookresearch/jepa at COMMIT, its model sources checked against
their SHA-256) run on drawn weights and inputs, in float32 and float64: the reference arm of
tools/venv_reference.py's `jepa-<case>` sites (its protocol tools/reference_worker.py), run in an
environment holding torch, on the CPU.

    python tools/vjepa_worker.py jepa-CASE    `IN OUT` lines on stdin, each `done`

IN holds each draw's configuration, `<draw>.config.<name>` each a 0-d array (`build`'s), its trainable
parameters `<draw>.param.<model>.<name>` (`encoder.` or `predictor.` first, every parameter but the two
fixed sincos tables), its `images` `[B, C, H, W]`, each image's disjoint `context_idx` and `target_idx`
patches, and cotangents `probe_context` and `probe_predictions` of the two outputs. OUT gets
`<draw>.reference.<leaf>` and `<draw>.truth.<leaf>` in float32 and float64: the encoder's embeddings of
each image's context patches (`context`), the mask-token predictor's `predictions` for its targets, as a
JEPA step calls them, and the `gradients` of `sum(context * probe) + sum(predictions * probe)` with
respect to the images and every trainable parameter in sorted name order, concatenated. Image-mode
(`num_frames=1`) models; the blocks run V-JEPA's own exact (erf) GELU, as Dew's JEPA does. The float64
run is under `torch_precision.widened`, under which the sincos tables are generated in float64 too.
"""

import importlib
import sys
from functools import cache, partial

import numpy as np
import torch
from reference_source import package, published_files
from reference_worker import serve

REPO, COMMIT = "facebookresearch/jepa", "51c59d518fc63c08464af6de585f78ac0c7ed4d5"
FILES = {
    "src/utils/tensors.py": "774a0f81eea52d2fcc4373818d8aab04e790c14f029e37e415ec0e5f6df2a8cb",
    "src/masks/utils.py": "2297fb30aabdbc635a7b9e9793806cebec62367f032d4b9bd57fbb4e1681a381",
    "src/models/utils/modules.py": "b4e2f4259a2255173d1059865691e7e45974d121781601035c44c589a1bf62e7",
    "src/models/utils/patch_embed.py": "8167c7a9a9a15dfff81a0e3b8cdfcbfab6881cc0da396cc130c4729ccba75c11",
    "src/models/utils/pos_embs.py": "3fff7d09a8e99eaaca0c3e436ecc5794dc9e62245fe3e5489e8cd37a23f771a8",
    "src/models/vision_transformer.py": "3d2dd861d535434c4202c1b488436433fc545d558899a48c392c240b4afe8abf",
    "src/models/predictor.py": "2a2d6febea26d5a670445f97bc9bc4c726617ee3d02a38f9aeff3b4b6db5a816",
}
@cache
def jepa():
    """V-JEPA's `src` at `COMMIT`, under its own package names."""
    package("src", published_files(REPO, COMMIT, FILES) / "src")
    return (importlib.import_module("src.models.vision_transformer"),
            importlib.import_module("src.models.predictor"))


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


def walked(encoder, predictor, inputs: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """Both outputs at `dtype`, and the gradients of `sum(context * probe) + sum(predictions * probe)` with
    respect to the images and every trainable parameter by sorted name, concatenated."""
    encoder, predictor = encoder.to(dtype), predictor.to(dtype)
    with torch.no_grad():
        encoder._init_pos_embed(encoder.pos_embed.data)
        predictor._init_pos_embed(predictor.predictor_pos_embed.data)
    images = torch.from_numpy(inputs["images"]).to(dtype).requires_grad_()
    context_idx, target_idx = (torch.from_numpy(inputs[name]) for name in ("context_idx", "target_idx"))
    named = sorted(trainable(encoder, predictor))
    context = encoder(images, masks=[context_idx])
    predictions = predictor(context, None, [context_idx], [target_idx])
    probes = {name: torch.from_numpy(inputs[f"probe_{name}"]).to(dtype)
              for name in ("context", "predictions")}
    objective = (context * probes["context"]).sum() + (predictions * probes["predictions"]).sum()
    grads = torch.autograd.grad(objective, [images] + [value for _, value in named])
    return {"context": context.detach().numpy(), "predictions": predictions.detach().numpy(),
            "gradients": np.concatenate([np.ravel(gradient.numpy()) for gradient in grads])}


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's outputs and gradients under its configuration and weights, in float32 and float64."""
    from torch_precision import widened

    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        inputs = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                  if key.startswith(f"{name}.")}
        config = {key.removeprefix("config."): value.item() for key, value in inputs.items()
                  if key.startswith("config.")}
        encoder, predictor = build(config)
        with torch.no_grad():
            for key, parameter in trainable(encoder, predictor):
                parameter.copy_(torch.from_numpy(inputs[f"param.{key}"]))
        for leaf, value in walked(encoder, predictor, inputs, torch.float32).items():
            out[f"{name}.reference.{leaf}"] = value
        with widened():
            for leaf, value in walked(encoder, predictor, inputs, torch.float64).items():
                out[f"{name}.truth.{leaf}"] = value
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("jepa-"):
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
