"""Checkpoint-file translation for native latent diffusion models.

Source configurations and safetensors are data. No external model or scheduler
implementation is imported by this module.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import json
from typing import TYPE_CHECKING, TypedDict
from flax.typing import Dtype
import numpy as np

from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
from dew.registry import resolve_dtype
from dew.interop.safetensors_io import load_params
if TYPE_CHECKING:
    from dew.interop.pretrained import WeightLayout

type TensorTree = dict[str, np.ndarray | TensorTree]


def _insert(tree: TensorTree, path: tuple[str, ...], value: np.ndarray) -> None:
    node = tree
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Tensor path crosses an existing leaf: {path}")
        node = child
    if path[-1] in node:
        raise ValueError(f"Two source tensors map to {path}")
    node[path[-1]] = value



class UNetFields(TypedDict):
    stages: tuple[UNetStage, ...]
    in_channels: int
    out_channels: int
    blocks_per_level: int
    linear_projection: bool
    additional_time_features: int
    middle_attention: bool
    frequency_shift: float
    cosine_first: bool
    dropout: float
    norm_groups: int
    norm_epsilon: float
    attention_norm_epsilon: float
    approximate_gelu: bool
    dtype: Dtype
    attention_impl: str | None


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def unet_fields(config: Mapping[str, object], *, dtype="float32", attention_impl="auto") -> UNetFields:
    """Interpret source geometry and reject operation-changing unsupported controls."""
    raw_widths = config["block_out_channels"]
    if not isinstance(raw_widths, (tuple, list)) or not raw_widths:
        raise ValueError("block_out_channels must be a nonempty sequence")
    widths = tuple(_integer(value, "block_out_channels") for value in raw_widths)
    count = len(widths)
    def per_stage(value, name):
        values = tuple(value) if isinstance(value, (list, tuple)) else (value,) * count
        if len(values) != count:
            raise ValueError(f"{name} must have one entry per UNet stage")
        return values
    heads = tuple(_integer(value, "attention heads") for value in per_stage(
        config.get("num_attention_heads") or config.get("attention_head_dim", 8), "attention heads"))
    depths = tuple(_integer(value, "transformer depth") for value in per_stage(config.get("transformer_layers_per_block", 1), "transformer depth"))
    only_cross = tuple(_boolean(value, "only_cross_attention") for value in per_stage(config.get("only_cross_attention", False), "only_cross_attention"))
    down, up = config["down_block_types"], config["up_block_types"]
    if not isinstance(down, (list, tuple)) or not isinstance(up, (list, tuple)) or len(down) != count or len(up) != count:
        raise ValueError("UNet down/up blocks must have one entry per stage")
    if any(kind not in ("DownBlock2D", "CrossAttnDownBlock2D") for kind in down):
        raise ValueError("Unsupported conditional UNet downsampling stage")
    if any(kind not in ("UpBlock2D", "CrossAttnUpBlock2D") for kind in up):
        raise ValueError("Unsupported conditional UNet upsampling stage")
    cross = tuple(kind == "CrossAttnDownBlock2D" for kind in down)
    if tuple(kind == "CrossAttnUpBlock2D" for kind in up) != cross[::-1]:
        raise ValueError("UNet decoder attention must mirror its encoder stages")
    addition = config.get("addition_embed_type")
    if addition not in (None, "text_time"):
        raise ValueError(f"Unsupported additional UNet condition: {addition}")
    middle = config.get("mid_block_type", "UNetMidBlock2DCrossAttn")
    if middle not in (None, "UNetMidBlock2DCrossAttn"):
        raise ValueError(f"Unsupported UNet middle block: {middle}")
    fixed = {
        "act_fn": "silu", "center_input_sample": False, "downsample_padding": 1,
        "mid_block_scale_factor": 1, "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False, "resnet_time_scale_shift": "default",
        "time_embedding_type": "positional", "time_embedding_act_fn": None,
        "timestep_post_act": None, "time_cond_proj_dim": None,
        "class_embed_type": None, "num_class_embeds": None, "class_embeddings_concat": False,
        "encoder_hid_dim": None, "encoder_hid_dim_type": None, "cross_attention_norm": None,
        "attention_type": "default", "dual_cross_attention": False,
        "reverse_transformer_layers_per_block": None, "upcast_attention": False,
        "conv_in_kernel": 3, "conv_out_kernel": 3,
    }
    for name, expected in fixed.items():
        if name in config and config[name] != expected:
            raise ValueError(f"Native UNet cannot honor active {name}={config[name]!r}")
    if config.get("time_embedding_dim") not in (None, widths[0] * 4):
        raise ValueError("Native UNet requires a time embedding four times its first width")
    groups = _integer(config.get("norm_num_groups", 32), "norm_num_groups")
    epsilon = _number(config.get("norm_eps", 1e-5), "norm_eps")
    if groups < 1 or epsilon <= 0 or any(width < 1 or width % groups for width in widths):
        raise ValueError("UNet normalization requires positive epsilon and widths divisible by its group count")
    if any(head < 1 or width % head or depth < 1 for width, head, depth in zip(widths, heads, depths)):
        raise ValueError("UNet stage heads must divide their width and transformer depth must be positive")
    source_name = config.get("_class_name", "UNet2DConditionModel")
    if source_name not in ("UNet2DConditionModel", "FlaxUNet2DConditionModel"):
        raise ValueError(f"Unsupported UNet source class: {source_name}")
    flax_semantics = source_name == "FlaxUNet2DConditionModel"
    if flax_semantics and (groups != 32 or epsilon != 1e-5):
        raise ValueError("The published Flax UNet has fixed normalization groups and epsilon")
    return UNetFields(
        stages=tuple(UNetStage(width, head, depth, attended, cross_only)
                     for width, head, depth, attended, cross_only in zip(widths, heads, depths, cross, only_cross)),
        in_channels=_integer(config["in_channels"], "in_channels"), out_channels=_integer(config["out_channels"], "out_channels"),
        blocks_per_level=_integer(config.get("layers_per_block", 2), "layers_per_block"),
        linear_projection=_boolean(config.get("use_linear_projection", False), "use_linear_projection"),
        additional_time_features=_integer(config["addition_time_embed_dim"], "addition_time_embed_dim") if addition else 0,
        middle_attention=middle is not None, frequency_shift=_number(config.get("freq_shift", 0), "freq_shift"),
        cosine_first=_boolean(config.get("flip_sin_to_cos", True), "flip_sin_to_cos"),
        dropout=_number(config.get("dropout", 0), "dropout"), norm_groups=groups, norm_epsilon=epsilon,
        attention_norm_epsilon=1e-5 if flax_semantics else 1e-6, approximate_gelu=flax_semantics,
        dtype=resolve_dtype(dtype), attention_impl=None if attention_impl == "reference" else attention_impl)


def _unet_path(name: str, rank: int) -> tuple[str, ...]:
    parts = name.split(".")
    leaf = parts.pop()
    if leaf not in ("weight", "bias"):
        raise ValueError(f"Unknown UNet tensor {name}")
    head = parts.pop(0)
    roots = {"conv_in": "input", "conv_out": "output", "conv_norm_out": "output_norm",
             "time_embedding": "time", "add_embedding": "additional_time"}
    if head in roots:
        path = [roots[head]]
    elif head in ("down_blocks", "up_blocks"):
        path = [f"{'down' if head == 'down_blocks' else 'up'}_{parts.pop(0)}"]
        kind, index = parts.pop(0), parts.pop(0)
        if kind == "resnets":
            path.append(f"residual_{index}")
        elif kind == "attentions":
            path.append(f"attention_{index}")
        elif kind in ("downsamplers", "upsamplers") and index == "0" and parts == ["conv"]:
            path.append("resize")
            parts = []
        else:
            raise ValueError(f"Unknown UNet stage tensor {name}")
    elif head == "mid_block":
        kind, index = parts.pop(0), parts.pop(0)
        if kind == "resnets" and index in ("0", "1"):
            path = ["middle_in" if index == "0" else "middle_out"]
        elif kind == "attentions" and index == "0":
            path = ["middle_attention"]
        else:
            raise ValueError(f"Unknown UNet middle tensor {name}")
    else:
        raise ValueError(f"Unknown UNet tensor {name}")
    attention = any(part.startswith("attention_") for part in path) or path[0] == "middle_attention"
    if attention and parts[:1] == ["transformer_blocks"]:
        path.append(f"layer_{parts[1]}")
        parts = parts[2:]
        kind = parts.pop(0)
        if kind in ("attn1", "attn2"):
            path.append("self_attention" if kind == "attn1" else "cross_attention")
            projection = parts.pop(0)
            if projection == "to_out" and parts == ["0"]:
                path.append("output")
                parts = []
            elif projection in ("to_q", "to_k", "to_v"):
                path.append(projection[-1])
            else:
                raise ValueError(f"Unknown UNet attention tensor {name}")
        elif kind == "ff":
            path.append("feed_forward")
            if parts[:2] == ["net", "0"]:
                path.append("net_0")
                parts = parts[2:]
            elif parts == ["net", "2"]:
                path.append("net_2")
                parts = []
            else:
                raise ValueError(f"Unknown UNet feed-forward tensor {name}")
        elif kind in ("norm1", "norm2", "norm3"):
            path.append({"norm1": "self_norm", "norm2": "cross_norm", "norm3": "ff_norm"}[kind])
        else:
            raise ValueError(f"Unknown UNet transformer tensor {name}")
    names = {"time_emb_proj": "temb_projection", "conv_shortcut": "residual_conv",
             "linear_1": "in_proj", "linear_2": "out_proj", "proj_in": "input", "proj_out": "output"}
    path.extend(names.get(part, part) for part in parts)
    path.append("bias" if leaf == "bias" else "scale" if rank == 1 else "kernel")
    return tuple(path)


def translate_unet_weights(tensors: Mapping[str, np.ndarray], model: UNet2DCondition) -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Native parameters and reversible source layouts, without random leaves."""
    from dew.interop.pretrained import WeightLayout
    parameters: TensorTree = {}
    layouts = []
    for name, tensor in tensors.items():
        tensor = np.asarray(tensor)
        path = _unet_path(name, tensor.ndim)
        value, transpose = tensor, None
        if path[-1] == "kernel":
            value = tensor.transpose(2, 3, 1, 0) if tensor.ndim == 4 else tensor.T
            transpose = (3, 2, 0, 1) if tensor.ndim == 4 else (1, 0)
            if "self_attention" in path or "cross_attention" in path:
                root = path[0]
                stage = (model.stages[int(root.removeprefix("down_"))] if root.startswith("down_") else
                         model.stages[::-1][int(root.removeprefix("up_"))] if root.startswith("up_") else model.stages[-1])
                if path[-2] in ("q", "k", "v"):
                    value = value.reshape(value.shape[0], stage.heads, stage.features // stage.heads)
                    transpose = (1, 2, 0)
                else:
                    value = value.reshape(stage.heads, stage.features // stage.heads, stage.features)
                    transpose = (2, 0, 1)
        _insert(parameters, path, value)
        layouts.append(WeightLayout("unet/" + name, (("params", *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)


def component_tensors(directory: Path, component: str) -> dict[str, np.ndarray]:
    """Read published safetensors, including sharded component directories."""
    folder = directory / component
    weights = "diffusion_pytorch_model" if component in ("unet", "vae") else "model"
    index = folder / f"{weights}.safetensors.index.json"
    def arrays(path: Path) -> dict[str, np.ndarray]:
        values = load_params(path)
        result: dict[str, np.ndarray] = {}
        for name, value in values.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Published component tensors must be flat named arrays: {path}")
            result[name] = value
        return result
    if index.is_file():
        record = json.loads(index.read_text())
        shards = sorted(set(record["weight_map"].values()))
        result = {}
        for name in shards:
            values = arrays(folder / name)
            if result.keys() & values.keys():
                raise ValueError(f"Duplicate tensors across {component} shards")
            result.update(values)
        return result
    return arrays(folder / f"{weights}.safetensors")


def record_layouts(component: str, tensors: Mapping[str, np.ndarray],
                   path_of: Callable[[str], tuple[str, ...] | None], prefix: tuple[str, ...]
                   ) -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Translate component tensors and retain their inverse storage layout."""
    from dew.interop.pretrained import WeightLayout
    parameters: TensorTree = {}
    layouts = []
    for name, tensor in tensors.items():
        path = path_of(name)
        if path is None:
            continue
        array = np.asarray(tensor, np.float32)
        transpose = None
        if path[-1] == "kernel":
            transpose = (3, 2, 0, 1) if array.ndim == 4 else (1, 0)
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        _insert(parameters, path, np.ascontiguousarray(array))
        layouts.append(WeightLayout(f"{component}/{name}", ((*prefix, *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)


def flax_component_parameters(component: str, tensors: Mapping[str, np.ndarray]) -> TensorTree:
    """Canonical tensor names to the declared Flax checkpoint's storage tree.

    This translates files only; no foreign model implementation is imported.
    """
    parameters: TensorTree = {}
    for name, tensor in tensors.items():
        array = np.asarray(tensor)
        parts = name.split(".")
        leaf = parts[-1]
        if component == "vae":
            from dew.nn.autoencoders.vae import _vae_path
            path = _vae_path(name, array.ndim)
        else:
            if leaf in ("weight", "bias"):
                parts.pop()
                leaf = ("bias" if leaf == "bias" else "embedding" if parts[-1] in ("token_embedding", "position_embedding")
                        else "scale" if array.ndim == 1 else "kernel")
            else:
                parts.pop()
            if component == "unet":
                combined = []
                index = 0
                while index < len(parts):
                    if index + 1 < len(parts) and parts[index + 1].isdigit():
                        combined.append(parts[index] + "_" + parts[index + 1])
                        index += 2
                    else:
                        combined.append(parts[index])
                        index += 1
                parts = combined
            path = (*parts, leaf)
        if path[-1] == "kernel":
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        _insert(parameters, path, np.ascontiguousarray(array))
    return parameters


def write_flax_component(directory: Path, component: str, tensors: Mapping[str, np.ndarray]) -> None:
    """Retain a source Flax model's declared weight format beside native safetensors."""
    from flax.serialization import to_bytes
    filename = "diffusion_flax_model.msgpack" if component in ("unet", "vae") else "flax_model.msgpack"
    (directory / component / filename).write_bytes(to_bytes(flax_component_parameters(component, tensors)))


def save_source(source, values, destination: Path) -> None:
    """Write a native diffusion bundle back to its published directory layout."""
    from dew.interop.safetensors_io import _safetensors
    destination.mkdir(parents=True, exist_ok=True)
    config = dict(source.config)
    index = dict(config.pop("model_index"))
    if source.inputs is not None:
        height, width = source.inputs.sample.shape[-3:-1]
        index.update(dew_height=height, dew_width=width)
    (destination / "model_index.json").write_text(json.dumps(index, indent=2))
    component_configs = {name: value for name, value in config.items() if isinstance(value, Mapping)}
    for name, component_config in component_configs.items():
        folder = destination / name
        folder.mkdir(exist_ok=True)
        file = {"scheduler": "scheduler_config.json", "feature_extractor": "preprocessor_config.json"}.get(name, "config.json")
        (folder / file).write_text(json.dumps(component_config, indent=2))
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for layout in source.weight_layouts:
        component, _, name = layout.name.partition("/")
        grouped.setdefault(component, {})[name] = layout.export(values)
    for component, tensors in grouped.items():
        weights = "diffusion_pytorch_model" if component in ("unet", "vae") else "model"
        _safetensors().save_file(tensors, destination / component / f"{weights}.safetensors", metadata={"format": "pt"})
        declared = index.get(component)
        if isinstance(declared, (list, tuple)) and len(declared) == 2 and isinstance(declared[1], str) and declared[1].startswith("Flax"):
            write_flax_component(destination, component, tensors)
    encoder = source.inputs.conditions["conditioning"].encoder
    for name, tokenizer in zip(encoder.names, encoder.tokenizers):
        folder = destination / ("tokenizer" + name.removeprefix("text_encoder"))
        tokenizer.save_pretrained(folder)
        tokenizer.backend_tokenizer.model.save(str(folder))
