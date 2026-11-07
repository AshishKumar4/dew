"""JEPA's encoder and predictor against V-JEPA's own (facebookresearch/jepa).

Over drawn weights, images, context and target patches and cotangents, Dew's JepaEncoder's embeddings of
each image's context patches, JepaPredictor's predictions for its targets and the gradients of the
cotangents' products with respect to the images and every trainable parameter are as exact as V-JEPA's
float32 run against its float64 run (check 1), each configuration its own recording
(tools/venv_reference.py's `jepa-<case>` sites, tools/vjepa_worker.py), Dew's float64 run held to the
recorded sketch. Both run V-JEPA's own exact GELU.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_runs import assert_arms_as_recorded
from torch_layout import QKV, dew_leaf, join_qkv, node, split_qkv, torch_leaf

from dew.nn.backbones.jepa import JepaEncoder, JepaPredictor

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jepa"
CASES = {
    "published": {"img_size": 8, "patch_size": 2, "in_chans": 3, "embed_dim": 32, "depth": 2, "num_heads": 4,
                  "mlp_ratio": 4, "eps": 1e-6, "predictor_embed_dim": 16, "pred_depth": 2,
                  "pred_num_heads": 2, "context": 9, "targets": 4},
    "variant": {"img_size": 12, "patch_size": 3, "in_chans": 2, "embed_dim": 24, "depth": 3, "num_heads": 3,
                "mlp_ratio": 2, "eps": 1e-5, "predictor_embed_dim": 20, "pred_depth": 1,
                "pred_num_heads": 5, "context": 7, "targets": 5},
}
"""`published`: V-JEPA's `vit_*`/`vit_predictor` construction (LayerNorm epsilon 1e-6, MLP ratio 4),
narrowed; `variant`: the constructors' own LayerNorm epsilon (1e-5), MLP ratio 2, odd head counts and a
two-channel image. Each is drawn to its worker (`build`'s arguments), with its context and target counts."""
JEPA_SEEDS = {"published": 8, "variant": 16}
"""Each configuration's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""
OUTPUTS = ("context", "predictions", "gradients")
"""The leaves `dew` gives, each site's recorded leaves."""
BATCH = 2
TOP = {"patch_embed": ("embed", "patch_embed", "Conv_0"), "norm": ("norm",), "predictor_embed": ("proj_in",),
       "predictor_norm": ("norm",), "predictor_proj": ("proj_out",)}
NORM = {"weight": "scale", "bias": "bias"}
DENSE = {"weight": "kernel", "bias": "bias"}


def dew_path(name: str) -> tuple[str, ...]:
    """The Dew parameter a V-JEPA name (`encoder.` or `predictor.` first)
    lands on, under that model's own tree."""
    model, first, *rest = name.split(".")
    if first == "mask_tokens":
        return (model, "mask_token")
    if first in ("blocks", "predictor_blocks"):
        index, layer, *rest = rest
        block = (model, "stack", f"block_{index}")
        if layer.startswith("norm"):
            return (*block, layer, NORM[rest[-1]])
        if layer == "mlp":
            return (*block, "mlp", {"fc1": "up_proj", "fc2": "down_proj"}[rest[0]], DENSE[rest[-1]])
        if rest[0] == "proj":
            return (*block, "attention", "to_out_0", DENSE[rest[-1]])
        return (*block, "attention", "qkv", rest[-1])
    return (model, *TOP[first], (NORM if first.endswith("norm") else DENSE)[rest[-1]])


def to_dew(name: str, value: np.ndarray, heads: int) -> dict[tuple[str, ...], np.ndarray]:
    """One V-JEPA tensor in Dew's layout: kernels transposed to [in, out],
    q, k and v split per head, the output projection read per head."""
    path = dew_path(name)
    if path[-2] == "qkv":
        parts = split_qkv(value, heads)
        return {(*path[:-2], part, "kernel" if value.ndim == 2 else "bias"): parts[index]
                for index, part in enumerate(QKV)}
    return {path: dew_leaf(path, value, heads)}


def from_dew(name: str, tree, heads: int) -> np.ndarray:
    """The inverse of `to_dew`: a Dew gradient back in V-JEPA's layout."""
    path = dew_path(name)
    if path[-2] == "qkv":
        stored = "kernel" if path[-1] == "weight" else "bias"
        return join_qkv([node(tree, (*path[:-2], part, stored)) for part in QKV])
    return torch_leaf(path, node(tree, path))


def state_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """V-JEPA's trainable parameters' names and shapes under `config` (tools/vjepa_worker.py's
    `trainable`), every block's own and both models' embeddings, norms and projections."""
    width, inner, patch = config["embed_dim"], config["predictor_embed_dim"], config["patch_size"]
    shapes = {"encoder.patch_embed.proj.weight": (width, config["in_chans"], patch, patch),
              "encoder.patch_embed.proj.bias": (width,), "encoder.norm.weight": (width,),
              "encoder.norm.bias": (width,), "predictor.mask_tokens.0": (1, 1, inner),
              "predictor.predictor_embed.weight": (inner, width), "predictor.predictor_embed.bias": (inner,),
              "predictor.predictor_norm.weight": (inner,), "predictor.predictor_norm.bias": (inner,),
              "predictor.predictor_proj.weight": (width, inner), "predictor.predictor_proj.bias": (width,)}
    for prefix, blocks, features in (("encoder.blocks", config["depth"], width),
                                     ("predictor.predictor_blocks", config["pred_depth"], inner)):
        hidden = features * config["mlp_ratio"]
        for index in range(blocks):
            shapes |= {f"{prefix}.{index}.{name}": shape for name, shape in (
                ("norm1.weight", (features,)), ("norm1.bias", (features,)),
                ("attn.qkv.weight", (3 * features, features)), ("attn.qkv.bias", (3 * features,)),
                ("attn.proj.weight", (features, features)), ("attn.proj.bias", (features,)),
                ("norm2.weight", (features,)), ("norm2.bias", (features,)),
                ("mlp.fc1.weight", (hidden, features)), ("mlp.fc1.bias", (hidden,)),
                ("mlp.fc2.weight", (features, hidden)), ("mlp.fc2.bias", (features,)))}
    return shapes


def drawing(case: str):
    """`draw(seed)` of `case`: weights at their initializer's scale, the norms' about one and the biases and
    mask token off zero, images, each image's disjoint context and target patches in no particular order,
    and cotangents of both outputs, under V-JEPA's names and layouts."""
    config = CASES[case]
    patches = (config["img_size"] // config["patch_size"]) ** 2

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng([125, seed])
        drawn = {}
        for name, shape in state_shapes(config).items():
            if "norm" in name and name.endswith("weight"):
                value = rng.normal(1.0, 0.3, shape)
            elif name.endswith(("bias", "mask_tokens.0")):
                value = rng.normal(0.0, 0.3, shape)
            else:
                value = rng.normal(0.0, 1.0, shape) / np.sqrt(np.prod(shape[1:]))
            drawn[f"param.{name}"] = value.astype(np.float32)
        orders = [rng.permutation(patches) for _ in range(BATCH)]
        image = (BATCH, config["in_chans"], config["img_size"], config["img_size"])
        return {**drawn, **{f"config.{key}": np.asarray(value) for key, value in config.items()},
                "images": rng.standard_normal(image).astype(np.float32),
                "context_idx": np.stack([order[:config["context"]] for order in orders]),
                "target_idx": np.stack([order[-config["targets"]:] for order in orders]),
                "probe_context": rng.standard_normal((BATCH, config["context"], config["embed_dim"]))
                .astype(np.float32),
                "probe_predictions": rng.standard_normal((BATCH, config["targets"], config["embed_dim"]))
                .astype(np.float32)}
    return draw


def computing(case: str):
    """`dew(dtype, inputs)` of `case`: Dew's embeddings, predictions and the gradients tools/vjepa_worker.py's
    `walked` gives, in its order, at `dtype` under a draw's weights."""
    config = CASES[case]
    names = sorted(state_shapes(config))
    heads = {"encoder": config["num_heads"], "predictor": config["pred_num_heads"]}
    grid = config["img_size"] // config["patch_size"]

    def dew(dtype, inputs) -> dict:
        params: dict = {"encoder": {}, "predictor": {}}
        for name in names:
            for path, array in to_dew(name, inputs[f"param.{name}"], heads[name.split(".")[0]]).items():
                node = params
                for key in path[:-1]:
                    node = node.setdefault(key, {})
                node[path[-1]] = jnp.asarray(array, dtype)
        encoder = JepaEncoder(patch_size=config["patch_size"], emb_features=config["embed_dim"],
                              num_layers=config["depth"], num_heads=config["num_heads"],
                              mlp_ratio=config["mlp_ratio"], norm_epsilon=config["eps"], attention_impl="xla",
                              dtype=dtype)
        predictor = JepaPredictor(grid=(grid, grid), emb_features=config["embed_dim"],
                                  predictor_features=config["predictor_embed_dim"],
                                  num_layers=config["pred_depth"], num_heads=config["pred_num_heads"],
                                  mlp_ratio=config["mlp_ratio"], norm_epsilon=config["eps"],
                                  attention_impl="xla", dtype=dtype)
        images = jnp.asarray(inputs["images"], dtype).transpose(0, 2, 3, 1)
        context_idx, target_idx = inputs["context_idx"], inputs["target_idx"]
        probes = tuple(jnp.asarray(inputs[f"probe_{name}"], dtype) for name in ("context", "predictions"))

        def forward(params, images):
            context = encoder.apply({"params": params["encoder"]}, images, context_idx)
            return context, predictor.apply({"params": params["predictor"]}, context, context_idx, target_idx)

        def objective(params, images):
            outputs = forward(params, images)
            return sum(jnp.sum(out * probe) for out, probe in zip(outputs, probes, strict=True))

        context, predictions = forward(params, images)
        grads = jax.grad(objective, argnums=(0, 1))(params, images)
        native = [grads[1].transpose(0, 3, 1, 2)]
        native += [from_dew(name, grads[0], heads[name.split(".")[0]]) for name in names]
        return {"context": context, "predictions": predictions,
                "gradients": jnp.concatenate([jnp.ravel(part) for part in native])}
    return dew


def pipeline_arms(case: str):
    """tools/venv_reference.py's `jepa-<case>` site: its recording, draw, Dew run and leaves."""
    return FIXTURES / f"{case}.npz", drawing(case), computing(case), OUTPUTS


@pytest.mark.parametrize("case", list(CASES))
def test_the_embeddings_predictions_and_every_gradient_are_as_exact_as_v_jepas(case):
    """Dew's embeddings, predictions and gradients as exact as V-JEPA's float32 against float64 (check 1)."""
    assert_arms_as_recorded(pipeline_arms(case), f"jepa-{case}", JEPA_SEEDS[case], f"V-JEPA {case}")
