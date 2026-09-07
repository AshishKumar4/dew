"""Load native models and their host processors from a Hugging Face source."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.interop import hf_decoders as decoders
from dew.interop.quantized import dequantize_checkpoint, fp8_block
from dew.diffusion.block import BlockProcess, CanvasGeneration
from dew.sampling.text import Sampling, Generation
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import MultimodalTransformer
from dew.nn.mixers.attention import AttentionMixer
from dew.nn.vision import projector_from_record, tower_from_record
from dew.registry import models, resolve_dtype, with_precision


class HostProcessor(Protocol):
    """The HF processor operations kept outside compiled model computation."""

    def __call__(self, **kwargs: object) -> Mapping[str, object]: ...
    def save_pretrained(self, save_directory: str) -> object: ...
    def batch_decode(self, sequences: list[list[int]], *, skip_special_tokens: bool) -> list[str]: ...


@dataclass(frozen=True)
class Processor:
    """Host text/image preprocessing followed by numeric layout normalization.

    The checkpoint processor owns resizing, normalization and special-token
    expansion. Dew organizes its outputs into row-aligned arrays; it does
    not reproduce the checkpoint's image preprocessing algorithms.
    """

    reference: HostProcessor
    config: Mapping[str, object]
    record: Mapping[str, object]

    def __call__(self, text: str | Sequence[str], *, images: object | None = None) -> ModelInputs:
        arguments: dict[str, object] = {
            "text": text if isinstance(text, str) else list(text),
            "padding": True, "return_tensors": "np"}
        if images is not None:
            arguments["images"] = images
        return self.from_hf(self.reference(**arguments))

    def from_hf(self, values: Mapping[str, object]) -> ModelInputs:
        """Validate and normalize actual processor outputs before device use."""
        known = {"input_ids", "attention_mask", "pixel_values", "token_type_ids", "mm_token_type_ids"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"processor fields {sorted(unknown)} have no native model input")
        tokens = np.asarray(values["input_ids"])
        if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer) or min(tokens.shape) < 1:
            raise ValueError("input_ids must be nonempty integer [B, S] rows")
        valid = np.asarray(values.get("attention_mask", np.ones(tokens.shape, bool)), dtype=bool)
        if valid.shape != tokens.shape:
            raise ValueError("attention_mask must align with input_ids")
        positions = np.maximum(np.cumsum(valid, axis=1) - 1, 0).astype(np.int32)
        token_fields = {"attention_mask": jnp.asarray(valid), "positions": jnp.asarray(positions)}
        conditioning: dict[str, jax.Array] = {}
        if "pixel_values" in values:
            count = self.record.get("tokens_per_image")
            if type(count) is not int or count < 1:
                raise ValueError("this processor layout requires a fixed positive tokens_per_image")
            image_id = self.record.get("image_token_id")
            if type(image_id) is not int:
                raise ValueError("image_token_id must be an integer")
            mask = tokens == image_id
            lengths = mask.sum(axis=1)
            if np.any(lengths % count):
                raise ValueError("image placeholder counts do not match the projector's feature count")
            image_counts = lengths // count
            pixels = np.asarray(values["pixel_values"])
            if pixels.ndim != 4 or pixels.shape[0] != int(image_counts.sum()):
                raise ValueError("pixel_values and image placeholder counts disagree")
            if not np.issubdtype(pixels.dtype, np.floating):
                raise ValueError("pixel_values must be floating processor output")
            width = int(image_counts.max())
            if width < 1:
                raise ValueError("pixels require image placeholders")
            padded = np.zeros((tokens.shape[0], width, *pixels.shape[1:]), pixels.dtype)
            indices = np.full(tokens.shape, -1, np.int32)
            groups = np.full(tokens.shape, -1, np.int32)
            offset = 0
            for row, images in enumerate(image_counts):
                n = int(images)
                padded[row, :n] = pixels[offset:offset + n]
                slots = np.flatnonzero(mask[row])
                indices[row, slots] = np.arange(n * count)
                groups[row, slots] = np.arange(n * count) // count
                offset += n
            token_fields.update(image_indices=jnp.asarray(indices), image_groups=jnp.asarray(groups))
            conditioning = {"pixel_values": jnp.asarray(padded),
                            "image_lengths": jnp.asarray(image_counts, jnp.int32)}
        result = ModelInputs(jnp.asarray(tokens, jnp.int32), token_fields, conditioning)
        result.validate()
        return result

    def decode(self, tokens: jax.typing.ArrayLike) -> list[str]:
        """Decode token rows with the tokenizer retained by the source processor."""
        array = np.asarray(tokens)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError("decode expects integer [B, S] token rows")
        return self.reference.batch_decode(array.tolist(), skip_special_tokens=True)

    def save_pretrained(self, directory: str | Path) -> None:
        """Save the same processor and tokenizer used by this source."""
        self.reference.save_pretrained(str(directory))


@dataclass(frozen=True)
class WeightLayout:
    """An existing source tensor's location and reversible storage layout."""

    name: str
    paths: tuple[tuple[str, ...], ...]
    shape: tuple[int, ...]
    transpose: tuple[int, ...] | None = None
    concatenate: int | None = None

    def export(self, variables: Mapping[str, object]) -> np.ndarray:
        leaves = []
        for path in self.paths:
            node: object = variables
            for part in path:
                if not isinstance(node, Mapping):
                    raise ValueError(f"parameter path {path} does not traverse a mapping")
                node = node[part]
            leaves.append(np.asarray(node))
        value = leaves[0] if self.concatenate is None else np.concatenate(leaves, axis=self.concatenate)
        if self.transpose is not None:
            value = value.transpose(self.transpose)
        return np.ascontiguousarray(value).reshape(self.shape)


def _wrapper_layouts(tensors, record):
    """Retain source names while borrowing the loader's internal leaf paths."""
    from dew.nn import vision

    tower_kind = record["tower"]["kind"]
    projector_kind = record["projector"]["kind"]
    tower_path: Callable[[str], tuple[str, ...] | None] = {"siglip": vision.siglip_vision_path, "llama4": vision.llama4_vision_path,
                  "gemma4": vision.gemma4_vision_path, "qwen3_5": vision.qwen35_vision_path,
                  "gemma3n": vision.gemma3n_vision_path}[tower_kind]
    tower_prefix = decoders._WRAPPER_TOWER_PREFIX[tower_kind]
    projector_prefix = decoders._WRAPPER_PROJECTOR_PREFIX[projector_kind]
    family = decoders._FAMILIES[record["text_model_type"]]
    bindings = []
    retained = {}
    for name, tensor in tensors.items():
        bare = name.removeprefix("model.")
        paths: tuple[tuple[str, ...], ...] = ()
        transpose = None
        concatenate = None
        if bare.startswith(projector_prefix):
            tail = bare.removeprefix(projector_prefix)
            path = vision.projector_weight_path(projector_kind, tail)
            paths = (("params", "projector", *path),)
            if path[-1] == "kernel" and tail != "mm_input_projection_weight":
                transpose = (1, 0)
        elif bare.startswith(tower_prefix):
            path = tower_path(bare.removeprefix(tower_prefix))
            if path is not None:
                paths = ((path[0], "tower", *path[1:]),) if tower_kind == "gemma4" else (("params", "tower", *path),)
                if path[-1] == "kernel":
                    transpose = (1, 0) if tensor.ndim == 2 else (3, 2, 0, 1)
                    if tensor.ndim == 5:
                        transpose = (1, 0)
        elif bare.startswith("language_model.") or bare == "lm_head.weight":
            tail = bare.removeprefix("language_model.")
            text_name = tail if tail.startswith(("model.", "lm_head.", "mtp.")) else "model." + tail
            if text_name == "lm_head.weight" and record["text"]["tie_embeddings"]:
                paths = (("params", "language_model", "embed_tokens", "embedding"),)
            elif text_name.endswith(".experts.gate_up_proj"):
                names = [text_name.removesuffix("gate_up_proj") + projection
                         for projection in ("gate_proj", "up_proj")]
                mapped = [family.weight_path(key, record["text"]) for key in names]
                resolved: list[tuple[str, ...]] = []
                for path in mapped:
                    if path is None:
                        raise ValueError(f"fused expert tensor {name!r} has no parameter path")
                    resolved.append((path[0], "language_model", *path[1:]))
                paths = tuple(resolved)
                concatenate = -1
                if record["text_model_type"] == "gemma4_text":
                    transpose = (0, 2, 1)
            else:
                path = family.weight_path(text_name, record["text"])
                if path is not None:
                    paths = ((path[0], "language_model", *path[1:]),)
                    if path[-1] == "kernel" and tensor.ndim == 2:
                        transpose = (1, 0)
                    elif text_name.endswith(".experts.down_proj") and record["text_model_type"] == "gemma4_text":
                        transpose = (0, 2, 1)
        else:
            raise ValueError(f"unknown source tensor {name!r}")
        if paths:
            bindings.append(WeightLayout(name, paths, tensor.shape, transpose, concatenate))
        else:
            # SigLIP's pooling head and reference-ignored auxiliary tensors
            # have no forward consumer; export preserves their source bytes.
            retained[name] = tensor
    return tuple(bindings), retained



@dataclass(frozen=True)
class Pretrained:
    """A native model, explicit variables and its checkpoint's host processor."""

    model: nn.Module
    variables: Mapping[str, Mapping[str, object]]
    processor: Processor | None
    config: Mapping[str, object]
    source: Path
    generation_config: Mapping[str, object] = field(default_factory=dict)
    weight_layouts: tuple[WeightLayout, ...] = ()
    retained_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)
    generation_adapter: Callable[[Pretrained, ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]], int, jax.Array, Sampling | BlockProcess | None], Generation | CanvasGeneration] | None = field(default=None, repr=False)
    export_adapter: Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, np.ndarray]] | None = field(default=None, repr=False)

    @property
    def model_config(self) -> dict[str, object]:
        """The native model's construction fields, read directly from its value."""
        return {item.name: getattr(self.model, item.name) for item in fields(self.model)
                if item.init and item.name not in ("parent", "name")}

    def generate(self, inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]], max_new_tokens: int, *,
                 key: jax.Array, generation: Sampling | BlockProcess | None = None) -> Generation | CanvasGeneration:
        """Generate through the model family's shared native sampling algorithm."""
        if self.generation_adapter is None:
            raise ValueError("this source has no native generation algorithm")
        return self.generation_adapter(self, inputs, max_new_tokens, key, generation)

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None) -> None:
        """Write trained variables back to the source layout with processor artifacts."""
        from dew.interop.safetensors_io import save_hf_layout
        values = self.variables if variables is None else variables
        destination = Path(directory)
        if self.export_adapter is not None:
            tensors = self.export_adapter(values, self.config)
        elif self.weight_layouts:
            tensors = {**self.retained_tensors,
                       **{layout.name: layout.export(values) for layout in self.weight_layouts}}
        elif isinstance(self.model, CausalTransformer):
            decoders.save_pretrained_decoder(self.model, values, destination)
            tensors = None
        else:
            raise ValueError("this source has no reversible weight layout")
        if tensors is not None:
            save_hf_layout(tensors, dict(self.config), destination)
        if self.processor is not None:
            self.processor.save_pretrained(destination)
        with open(destination / "generation_config.json", "w") as handle:
            json.dump(dict(self.generation_config), handle, indent=2)





def _native_variables(parts: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    collections: dict[str, dict[str, object]] = {}
    for component, variables in parts.items():
        for collection, tree in variables.items():
            collections.setdefault(collection, {})[component] = tree
    return collections


def _generation_value(bundle: Pretrained, name: str, default: object = None) -> object:
    text = bundle.config.get("text_config", bundle.config)
    if not isinstance(text, Mapping):
        raise ValueError("text_config must be a mapping")
    return bundle.generation_config.get(name, bundle.config.get(name, text.get(name, default)))


def _eos_ids(bundle: Pretrained) -> tuple[int, ...]:
    value = _generation_value(bundle, "eos_token_id")
    if value is None:
        return ()
    values = (value,) if type(value) is int else value
    if not isinstance(values, (tuple, list)) or any(type(item) is not int or item < 0 for item in values):
        raise ValueError("eos_token_id must be an integer or a sequence of integers")
    return tuple(values)


def _pad_id(bundle: Pretrained) -> int:
    value = _generation_value(bundle, "pad_token_id", 0)
    if value is None:
        value = 0
    if type(value) is not int or value < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    return value


def _generate_autoregressive(bundle: Pretrained, inputs, max_new_tokens: int,
                             key: jax.Array, generation: Sampling | BlockProcess | None):
    from dew.artifacts import agree_process_phase
    from dew.sampling.text import generate
    error = None
    try:
        if generation is None:
            temperature = _generation_value(bundle, "temperature", 1.0)
            if not isinstance(temperature, (float, int)) or isinstance(temperature, bool):
                raise ValueError("temperature must be numeric")
            top_k = _generation_value(bundle, "top_k")
            if top_k is not None and type(top_k) is not int:
                raise ValueError("top_k must be an integer")
            generation = Sampling(
                temperature=float(temperature) if _generation_value(bundle, "do_sample", False) else 0.0,
                top_k=top_k if top_k else None, eos_id=_eos_ids(bundle), pad_id=_pad_id(bundle))
        if not isinstance(generation, Sampling):
            raise ValueError("autoregressive generation requires Sampling")
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase="pretrained generation policy")
    assert isinstance(generation, Sampling)
    return generate(bundle.model, bundle.variables, inputs, max_new_tokens, key=key, sampling=generation)


def _generate_canvas(bundle: Pretrained, inputs, max_new_tokens: int,
                     key: jax.Array, generation: Sampling | BlockProcess | None):
    from dew.artifacts import agree_process_phase
    from dew.interop import diffusion_gemma
    from dew.nn.diffusion_gemma import DiffusionGemma
    error = None
    eos: tuple[int, ...] = ()
    pad = 0
    try:
        if not isinstance(bundle.model, DiffusionGemma):
            raise TypeError("canvas generation requires DiffusionGemma")
        if generation is None:
            generation = diffusion_gemma.generation_process(bundle.config, bundle.generation_config)
        if not isinstance(generation, BlockProcess):
            raise ValueError("DiffusionGemma generation requires BlockProcess")
        eos, pad = _eos_ids(bundle), _pad_id(bundle)
    except BaseException as failure:
        error = failure
    agree_process_phase(error, phase="pretrained canvas policy")
    assert isinstance(generation, BlockProcess) and isinstance(bundle.model, DiffusionGemma)
    return generation.generate(bundle.model, bundle.variables, inputs, max_new_tokens,
                               key=key, eos_token_ids=eos, pad_token_id=pad)



def load_pretrained(name_or_dir: str | Path, *, dtype: str = "bfloat16",
                    attention_impl: str = "auto", max_seq_len: int | None = None,
                    revision: str | None = None) -> Pretrained:
    """Load a source into a native Flax model with explicit parameter trees.

    ``name_or_dir`` is a local HF directory or a Hub model identifier. The
    decoder/tower/projector maps preserve their established internal paths;
    wrapper variables join under their existing component names. Processor
    artifacts are loaded only when the source contains them.
    """
    directory = decoders._snapshot(str(name_or_dir), revision)
    with open(directory / "config.json") as handle:
        config = json.load(handle)
    tensors = dequantize_checkpoint(decoders._load_shards(directory), fp8_block(config))
    family = config.get("model_type")
    layouts: tuple[WeightLayout, ...] = ()
    retained: dict[str, np.ndarray] = {}
    export_adapter = None
    generation_adapter = _generate_autoregressive
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma
        model = diffusion_gemma.build(config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len)
        variables = diffusion_gemma.translate_weights(tensors, config)
        record = config
        export_adapter = diffusion_gemma.export_weights
        generation_adapter = _generate_canvas
    elif "text_config" in config:
        record = decoders.translate_wrapper_config(config)
        text_fields = dict(record["text"])
        if max_seq_len is not None:
            text_fields["max_seq_len"] = max_seq_len
        if family == "gemma3":
            # Gemma3ForConditionalGeneration projects logits without the
            # causal-LM class's optional final tanh cap.
            text_fields["final_logit_softcap"] = None
            text_fields["mixer"] = AttentionMixer(bidirectional_images=True)
        language_model = models.build("causal_transformer", **with_precision(
            "causal_transformer", text_fields, dtype=dtype, attention_impl=attention_impl))
        if not isinstance(language_model, CausalTransformer):
            raise TypeError("causal_transformer registry entry must build CausalTransformer")
        model = MultimodalTransformer(
            language_model, tower_from_record(record["tower"]),
            projector_from_record(record["projector"]), family,
            record["image_token_id"], dtype=resolve_dtype(dtype),
            attention_impl=None if attention_impl == "reference" else attention_impl)
        variables = _native_variables(decoders.translate_wrapper_weights(tensors, record))
        layouts, retained = _wrapper_layouts(tensors, record)
    else:
        record = decoders.translate_config(config)
        if max_seq_len is not None:
            record["max_seq_len"] = max_seq_len
        model = models.build("causal_transformer", **with_precision(
            "causal_transformer", record, dtype=dtype, attention_impl=attention_impl))
        variables = decoders.translate_weights(tensors, record)
        decoders._check_tree(variables, model)
    processor = None
    if (directory / "processor_config.json").exists():
        from transformers import AutoProcessor
        reference = AutoProcessor.from_pretrained(str(directory), local_files_only=True)
        processor = Processor(reference, config, record)
    elif (directory / "tokenizer_config.json").exists():
        from transformers import AutoTokenizer
        reference = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        processor = Processor(reference, config, record)
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    return Pretrained(model, variables, processor, config, directory, generation_config,
                      layouts, retained, generation_adapter, export_adapter)
