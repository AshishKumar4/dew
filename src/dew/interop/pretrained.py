"""Load native models and their host processors from a Hugging Face source.

`Pretrained.load` is the front door: it reads a source directory or repo,
translates its config and weights through `dew.interop.hf_decoders`, and
returns the kind of `Pretrained` the source is, holding the model, its
variables and its processor.
`Pretrained` also carries the source's own decoding controls and the layouts
that write every tensor back, so `Pretrained.save` restores what it read.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import linen as nn
from jax.typing import DTypeLike

from dew import records
from dew._model_types import _QWEN35_TEXT_TYPES, _QWEN35_TYPES
from dew.coordination import agreed
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import Origin, SourceSchedule
from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
from dew.inference.pipeline import place
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.inputs.diffusion import (
    Composition,
    DiffusionConditioner,
    HiddenStatesConditioner,
    QwenImageConditioner,
    T5Segment,
    WanConditioner,
)
from dew.interop import gguf, hf_decoders as decoders, mamba2, sources, verify, weights as checkpoint_weights
from dew.interop.codecs import SourceQuantization, source_quantization
from dew.interop.components import bind_component
from dew.interop.config_records import NativeFields
from dew.interop.generation_config import (
    audit_masked,
    eos_ids,
    generation_config_of,
    generation_limit,
    pad_id,
    return_sequences,
    source_decoding,
)
from dew.interop.processors import (
    HostProcessor as HostProcessor,
    Processor as Processor,
    ProcessorCall as ProcessorCall,
    _hosts,
)
from dew.interop.safetensors_io import MAX_SHARD_SIZE
from dew.interop.streaming import LazyTree, SourceLeaf, WeightLayout
from dew.interop.weights import ParamTree
from dew.nn import audio as audio_nn
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer
from dew.objectives.base import Variables
from dew.registry import dtype_name, from_record, projectors, resolve_dtype, towers
from dew.sampling.guidance import CFG
from dew.sampling.pipelines import TextToImage
from dew.sampling.text import Sampling

if TYPE_CHECKING:

    from dew.lora import Adapter, LoRA
    from dew.training.distributed import Layout, MeshSpec



def _stacked_expert(path: tuple[str, ...]) -> tuple[tuple[str, ...], int | None]:
    """Return the leaf `hf_decoders._stack_experts` stacked a per-expert
    `experts/K/projection/kernel` path into, and K."""
    if (len(path) >= 4 and path[-4] == "experts" and path[-3].isdigit()
            and path[-1] == "kernel"):
        return (*path[:-3], path[-2], path[-1]), int(path[-3])
    return path, None


def _leading_axes(variables: Mapping[str, object], path: tuple[str, ...],
                  expert_index: int | None) -> int:
    """Return how many axes a bound leaf carries ahead of its stored matrix.

    A kernel stores `[in, out]` where its source stores `[out, in]`, and a
    grouped projection's leaf keeps one such matrix per group: DeepSeek
    V4's `[groups, in, rank]` is stored `[groups * rank, in]`, the group
    axis folded into the rows (modeling_deepseek_v4.py:294-323). The
    transpose is the same swap of the trailing pair under those axes, so
    the leaf says how many there are. A per-expert source tensor is one
    slice of its stacked leaf, which is taken before the transpose.
    """
    node: object = variables
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"the loaded tree holds no {path}, which {part!r} names")
        node = node[part]
    if not isinstance(node, np.ndarray | jax.Array | SourceLeaf):
        return 0
    rank = node.ndim - (0 if expert_index is None else 1)
    return max(rank - 2, 0)


def _language_layout(name: str, text_name: str, tensor: np.ndarray,
                     config, model_type: str, variables: Mapping[str, object],
                     component: str | None = None) -> WeightLayout | None:
    """Return the leaves the text family maps a source tensor to and the
    storage operations that rebuild it from them; a `packed` tensor is built
    from the layouts of its parts."""
    family = decoders.families()[model_type]
    packing = family.packing(text_name)
    if packing is None:
        return _leaf_layout(name, text_name, tensor, family, config, variables, component)
    parts = [_leaf_layout(part, part, value, family, config, variables, component)
             for part, value in packing.split(text_name, tensor, config).items()]
    if any(part is None for part in parts):
        raise ValueError(f"packed tensor {name!r} has no parameter path")
    return packing.layout(name, [part for part in parts if part is not None])


def _leaf_layout(name: str, text_name: str, tensor: np.ndarray, family: decoders.DecoderFamily,
                 config, variables: Mapping[str, object], component: str | None) -> WeightLayout | None:
    def nested(path: tuple[str, ...]) -> tuple[str, ...]:
        return path if component is None else (path[0], component, *path[1:])

    transpose = None
    expert_index = None
    head_name, embedding_name = family.tied_head_names
    if text_name == head_name and config["tie_embeddings"]:
        # The tied head has no leaf of its own, so its source name binds to
        # the embedding it copies, under whatever name that family stores.
        embedding = family.weight_path(embedding_name, config)
        if embedding is None:
            raise ValueError(f"{embedding_name!r} has no parameter path to tie {name!r} to")
        paths = (nested(embedding),)
    else:
        path = family.weight_path(text_name, config)
        if path is None:
            return None
        path, expert_index = _stacked_expert(path)
        paths = (nested(path),)
        if path[-1] == "kernel" and tensor.ndim == 2:
            lead = _leading_axes(variables, paths[0], expert_index)
            transpose = (*range(lead), lead + 1, lead)
    # A floating weight is written in its leaf's dtype; an index table
    # carries the width the checkpoint stored it in back.
    stored = None if np.issubdtype(tensor.dtype, np.floating) else tensor.dtype
    padded = tensor.shape[0] if text_name.endswith(family.zero_padded) else None
    return WeightLayout(name, paths, tensor.shape, transpose, None, expert_index, stored, padded)


def _wrapper_layouts(tensors, record, variables):
    """Return one `WeightLayout` per wrapper source tensor.

    Each layout keeps the source's own tensor name and points at the leaf path
    the loader built, so an export writes the names the source shipped.
    """
    from dew.nn import vision
    from dew.nn.vision.common import projector_weight_path

    tower_kind = record["tower"]["class"]
    audio_encoder = None if record["audio"] is None else towers.from_record(record["audio"])
    bindings = []
    retained = {}
    for name, tensor in tensors.items():
        group, local = decoders._wrapper_route(name, record)
        if group == "language_model":
            layout = _language_layout(name, local, tensor, record["text"], record["text_model_type"],
                                      variables, "language_model")
            if layout is None:
                retained[name] = tensor
            else:
                bindings.append(layout)
            continue
        paths: tuple[tuple[str, ...], ...] = ()
        transpose = None
        if group == "projector":
            path = projector_weight_path(record["projector"]["class"], local)
            paths = (("params", "projector", *path),)
            if path[-1] == "kernel" and local != "mm_input_projection_weight":
                transpose = (1, 0)
        elif group == "tower":
            path = vision.TOWER_PATHS[tower_kind](local)
            if path is not None:
                paths = (
                    ((path[0], "tower", *path[1:]),)
                    if tower_kind == "gemma4"
                    else (("params", "tower", *path),)
                )
                if path[-1] == "kernel":
                    transpose = (1, 0) if tensor.ndim in (2, 5) else (3, 2, 0, 1)
        elif group == "audio_projector":
            path = projector_weight_path(record["audio_projector"]["class"], local)
            paths = (("params", "audio_projector", *path),)
            if path[-1] == "kernel":
                transpose = (1, 0)
        else:
            if not isinstance(audio_encoder, (audio_nn.Gemma3nAudio, audio_nn.Gemma4Audio)):
                raise ValueError("source export requires a Gemma audio encoder")
            path = audio_nn.audio_weight_path(local, audio_encoder)
            paths = ((path[0], "audio_tower", *path[1:]),)
            if path[-1] == "kernel":
                # Kernels store [*window, in, out]; the source keeps [out, in, *window].
                transpose = {2: (1, 0), 3: (2, 1, 0), 4: (3, 2, 0, 1)}[tensor.ndim]
        if paths:
            bindings.append(WeightLayout(name, paths, tensor.shape, transpose))
        else:
            # SigLIP's pooling head and reference-ignored auxiliary tensors
            # have no forward consumer; export preserves their source bytes.
            retained[name] = tensor
    return tuple(bindings), retained


def _scalar_mode(layouts: tuple[WeightLayout, ...], variables: Mapping[str, object]) -> str | None:
    """Where the tree being written keeps the layer scalars `layouts` read
    ('trainable' in `params`, 'frozen' in `constants`), or None; the tree
    decides, as in `diffusion_gemma.scalar_placement`."""
    for layout in layouts:
        path = layout.paths[0]
        if path[-1] == "layer_scalar":
            node: object = variables.get("params")
            for part in path[1:]:
                node = node.get(part) if isinstance(node, Mapping) else None
            return "frozen" if node is None else "trainable"
    return None


def _share_quantized_aliases(tensors: dict[str, np.ndarray], aliases: tuple[tuple[str, str], ...],
                             quantized: tuple[str, ...]) -> None:
    """Share only aliases verified on original values before codec narrowing.

    A component may mix quantized and unquantized copies, or several MTP
    copies of a tied head. Fold the checked relationships transitively, then
    reuse the already decoded storage: no second cast or weight copy.
    """
    if not aliases:
        return
    links: dict[str, set[str]] = {}
    for left, right in aliases:
        links.setdefault(left, set()).add(right)
        links.setdefault(right, set()).add(left)
    remaining = set(links)
    while remaining:
        first = remaining.pop()
        group, pending = {first}, [first]
        while pending:
            fresh = links[pending.pop()] - group
            group.update(fresh)
            remaining.difference_update(fresh)
            pending.extend(fresh)
        representative = next((name for name in quantized if name in group), None)
        if representative is not None:
            value = tensors[representative]
            for name in group:
                tensors[name] = value


@dataclass(frozen=True)
class Pretrained:
    """A native model with explicit variables and its checkpoint's host processor.

    `Pretrained.load` returns a subclass for the kind of source it read, with the
    methods that work for that kind: `PretrainedDecoder` for an autoregressive
    decoder, `PretrainedMaskedDecoder` for a masked-diffusion one,
    `PretrainedBlockDecoder` for DiffusionGemma, `PretrainedPipeline` for a latent
    diffusion pipeline, and `PretrainedFallback` for transformers' own forward run
    through torchax. Every kind saves back to the source's format.

    `model_config` is the record the model was built from, in Dew's own
    vocabulary with the run's compute dtype and attention kernel, so a caller can
    log exactly the model it ran.
    """

    model: nn.Module
    variables: Variables
    processor: Processor | None
    config: Mapping[str, object]
    source: Path | None
    """The directory the source was read from.

    None for a model trained in Dew (`PretrainedDecoder.from_model`).
    """
    model_config: Mapping[str, object]
    generation_config: Mapping[str, object] = field(default_factory=dict)
    weight_layouts: tuple[WeightLayout, ...] = ()
    retained_tensors: Mapping[str, np.ndarray] = field(default_factory=dict)
    export_adapter: Callable[..., Mapping[str, np.ndarray]] | None = field(default=None, repr=False)
    """The encoder `export` writes through, installed by the route that built the model, or None
    for `weight_layouts`."""
    quantized_tensors: tuple[str, ...] = ()
    quantized_scale_dtype: str | None = None
    """The dtype a quantized source stored its scales in, when its format leaves that to the checkpoint.

    DeepSeek-V4's `.scale`, for example, is float8_e8m0fnu, and float32 in the
    Base releases. `save` writes the scales back in this dtype.
    """
    quantization_grid: Mapping[str, np.ndarray] = field(default_factory=dict, repr=False)
    """The scales and zeros an integer format (AWQ, GPTQ) encodes a saved weight against.

    They are kept as the source stored them.
    """
    revision: str | None = None
    """The Hub commit the source resolved to, whatever branch or tag was requested.

    None for a local directory.
    """
    adapter: Adapter | None = None
    """The low-rank adapter that `adapt` attached to the model, or None for the source as published.

    The variables hold its factors under `params`, next to the base weights under
    `frozen`.
    """
    tokenizer: str | decoders.ExportTokenizer | None = None
    """The vocabulary `save` writes next to the weights, by name or as an object.

    It is used by a bundle that has no source processor to write (`from_model`).
    """
    names_tokenizer: ClassVar[bool] = True
    """Whether generation_config.json may record the tokenizer's name.

    It is False for a family whose reader treats that file as a closed set of
    fields.
    """

    @property
    def text_processor(self) -> TaskProcessor | None:
        """Return what the bundle's tasks encode and decode text with.

        That is the source's processor, or a run processor over the vocabulary that
        `tokenizer` names (a run's export, including Dew's byte vocabulary).
        """
        if self.processor is not None or not isinstance(self.tokenizer, str):
            return self.processor
        from dew.data.text import tokenizer_for
        from dew.inference.pipeline import RunProcessor
        return RunProcessor(tokenizer_for(self.tokenizer, local_files_only=True))

    @classmethod
    def from_run(cls, directory: str | Path, *, step: int | str | None = None,
                 ema: bool | None = None, trust: Sequence[str] = ()) -> Self:
        """Return a trained run's selected checkpoint, rebuilt from the run's own inference record,
        in the layout of the architecture the record names. `trust` names the packages outside
        Dew the record may import (`TextGeneration.from_run`)."""
        from dew.config import ModelConfig
        from dew.inference.tasks import run_record
        from dew.records import record, text
        from dew.registry import models, objectives
        declaration = run_record(str(directory), step, trust)
        model_config = ModelConfig.from_dict(record(declaration['model'], 'model'))
        model = model_config.build()
        kind = text(declaration['objective'], 'objective')
        variables = objectives[kind]._saved_variables(str(directory), step=step, ema=ema)
        # The run's policy and budget, in the format the export's readers read.
        sampling = declaration.get('sampling')
        budget = declaration.get('sample_tokens')
        generation = None if not isinstance(sampling, dict) else generation_config_of(
            Sampling(**sampling), budget if isinstance(budget, int) else None)
        tokenizer = declaration.get('tokenizer')
        tokenizer = None if tokenizer is None else text(tokenizer, 'tokenizer')
        trained = models[model_config.name]
        if trained is CausalTransformer:
            decoder = from_record(CausalTransformer, model)
            bundle = PretrainedDecoder.from_model(decoder, variables, tokenizer=tokenizer,
                                                   generation_config=generation)
            if objectives[kind] is objectives['masked_diffusion']:
                bundle = PretrainedMaskedDecoder(
                    decoder, bundle.variables, bundle.processor, bundle.config, bundle.source,
                    bundle.model_config, bundle.generation_config,
                    export_adapter=bundle.export_adapter, tokenizer=bundle.tokenizer)
        elif trained is models["diffusion_gemma"]:
            from dew.interop import diffusion_gemma
            block = from_record(DiffusionGemma, model)
            bundle = PretrainedBlockDecoder(
                block, variables, None, diffusion_gemma.published_config(block), None,
                model_config.fields, {}, export_adapter=diffusion_gemma.export_weights, tokenizer=tokenizer)
        else:
            raise TypeError(f"{type(model).__name__} has no maintained exported bundle layout; "
                            "load diffusion runs with TextToImage.from_run")
        if not isinstance(bundle, cls):
            raise TypeError(f"{directory} is a {type(bundle).__name__} source, not a {cls.__name__}; "
                            f"load it with {type(bundle).__name__}.from_run or Pretrained.from_run")
        if model_config.adapter is None:
            return bundle
        # The run's adapter, as `Adapter.from_run` rebuilds it, so `save`
        # merges its factors and `adapter.save` writes them alone.
        from dew.lora import Adapter
        return replace(bundle, adapter=Adapter.recorded(model, variables, model_config.adapter))

    @classmethod
    def load(cls, name_or_dir: str | Path, *, dtype: DTypeLike = jnp.bfloat16,
             param_dtype: DTypeLike | Literal["auto"] = jnp.float32,
             attention_impl: str = "auto", max_seq_len: int | None = None,
             revision: str | None = None, gguf_file: str | None = None,
             single_file: str | None = None, dduf_file: str | None = None,
             mesh: MeshSpec | None = None, layout: Layout | None = None,
             fallback: str | None = None) -> Self:
        """Load a source as a native Flax model with explicit parameter trees, as the kind of source it is.

        `name_or_dir` is a local HF directory or a Hub model ID. Decoder, tower and
        projector weights keep their usual internal paths, and a wrapper's variables
        join under its existing component names. Processor files are loaded only when
        the source has them.

        `dtype` sets the compute dtype, and `param_dtype` separately sets the storage
        dtype of floating parameters. It defaults to FP32 master weights, and 'auto'
        keeps the checkpoint's own dtype. Each is a dtype (`jnp.bfloat16`) or its
        name, and the model's record keeps the name. Frozen components (text encoders
        and the VAE) follow `param_dtype` too, while router, clipping, positional and
        safety state keep their own FP32 or integer dtypes.

        Without `mesh` or `layout`, the variables are host arrays. With either, they
        are placed on that mesh (the default `MeshSpec()` when only `layout` is given)
        under that layout, one leaf at a time. A decoder's or a diffusion pipeline's
        leaves are read from the memory-mapped checkpoint one device shard at a time,
        and cast and transposed there (`dew.interop.streaming`), so the host never
        holds the whole translated model. Towers, projectors, a quantized source's
        dequantized tensors and a pipeline's convolution kernels are still built
        whole on the host first.

        `gguf_file` names a GGUF file in the repo or directory. Its metadata is the
        config, its block-quantized tensors are dequantized to float32
        (`dew.interop.gguf`), and its tokenizer is the processor when the repo ships
        no tokenizer.

        `single_file` names an original-format diffusion checkpoint in the repo or
        directory. Diffusers' own key maps convert it once into Dew's cache as the
        diffusers pipeline it describes (`dew.interop.single_file`), which then loads;
        the cache entry is published only after that load succeeds. The configs come
        from the repo or directory when it has a model_index.json, and otherwise from
        the diffusers repo that diffusers infers from the checkpoint, at the commit
        fetched. A component the file lacks gets its weights from the same place, or
        is refused by name.

        `dduf_file` names a DDUF file in the repo or directory, which is a diffusers
        pipeline packed into one archive. It is unpacked once into Dew's cache
        (`dew.interop.dduf`) and loads as the directory it packs, the same way
        diffusers' own `from_pretrained(..., dduf_file=)` reads it.

        `fallback="torchax"` opts into tier 3 for any causal LM that transformers can
        build, registered or not: transformers' PyTorch forward lowered to JAX by
        torchax (`dew.interop.torchax_fallback`), with no Dew kernels, sharding rules
        or cached generation.

        Called on a specific kind (`PretrainedDecoder.load`), it refuses a source of
        another kind and names that kind.
        """
        return cls._load(name_or_dir, dtype=dtype, param_dtype=param_dtype, attention_impl=attention_impl,
                         max_seq_len=max_seq_len, revision=revision, gguf_file=gguf_file,
                         single_file=single_file, dduf_file=dduf_file, mesh=mesh, layout=layout,
                         fallback=fallback)

    @classmethod
    def _load(cls, name_or_dir: str | Path, *, dtype: DTypeLike = jnp.bfloat16,
              param_dtype: DTypeLike | Literal["auto"] = jnp.float32,
              attention_impl: str = "auto", max_seq_len: int | None = None,
              revision: str | None = None, gguf_file: str | None = None, single_file: str | None = None,
              dduf_file: str | None = None,
              mesh: MeshSpec | None = None, layout: Layout | None = None, fallback: str | None = None,
              prepare: Callable[[nn.Module, Variables], Variables] | None = None) -> Self:
        """Share the source reader with inference's pre-placement projection packing."""
        if fallback not in (None, "torchax"):
            raise ValueError(f"fallback={fallback!r} names no loader; the one fallback is 'torchax', "
                             "tier 3 through transformers' PyTorch forward")
        streaming = mesh is not None or layout is not None

        def placed(variables: Variables) -> Variables:
            return place(variables, mesh, layout) if streaming else variables

        dtype = dtype_name(dtype)
        param_dtype = AUTO if param_dtype == AUTO else dtype_name(param_dtype)
        directory = sources.snapshot(str(name_or_dir), revision, weights=False)
        # A Hub snapshot directory is named by its commit.
        commit = None if os.path.isdir(name_or_dir) else directory.name
        if fallback is not None:
            from dew.interop import torchax_fallback
            loaded = torchax_fallback.load(name_or_dir, directory, commit, dtype=dtype,
                                           param_dtype=param_dtype, attention_impl=attention_impl,
                                           max_seq_len=max_seq_len)
            loaded = replace(loaded, variables=placed(loaded.variables))
        else:
            loaded = _pipeline_source(name_or_dir, directory, commit, single_file, dduf_file, placed,
                                      dtype=dtype, attention_impl=attention_impl, param_dtype=param_dtype,
                                      streaming=streaming)
        if loaded is None:
            loaded = _load_native_source(name_or_dir, directory, commit, gguf_file=gguf_file, placed=placed,
                                         streaming=streaming, dtype=dtype, param_dtype=param_dtype,
                                         attention_impl=attention_impl, max_seq_len=max_seq_len,
                                         prepare=prepare)
        if not isinstance(loaded, cls):
            raise TypeError(f"{name_or_dir} is a {type(loaded).__name__} source, not a {cls.__name__}; "
                            f"load it with {type(loaded).__name__}.load or Pretrained.load")
        return loaded

    @property
    def layouts(self) -> Mapping[str, WeightLayout]:
        """Map each source module name to the layout of its weight.

        The names are `model.<module>` for a decoder and `unet.<module>` for a
        pipeline component. A published adapter file uses these names, so `dew.lora`
        attaches a low-rank delta through them.
        """
        return {layout.name.removesuffix(".weight").replace("/", "."): layout
                for layout in self.weight_layouts if layout.name.endswith(".weight")}

    @property
    def _adaptable(self) -> Mapping[str, WeightLayout]:
        """The layouts an adapter binds: every one of a decoder's."""
        return self.layouts

    def adapt(self, lora: LoRA, *, key: int | jax.Array) -> Self:
        """Return this source with `lora` attached to its model and its factors drawn (`LoRA.apply`).

        The result holds the adapted model, the variables with the factors under
        `params` and the base weights under `frozen`, and the bound `adapter`. Any
        objective built over it trains only the factors, and since B starts at zero it
        computes what the source does. `adapter.save` writes the factors under the
        source's own names (PEFT's directory for a decoder, the Diffusers file for a
        pipeline), and `save` writes the source with them merged in. A pipeline
        adapts only its denoiser's projections, so the text towers and the VAE stay
        as published.
        """
        if self.adapter is not None:
            raise ValueError("this bundle already carries an adapter; adapt the source it was made from")
        adapter = lora.apply(self.model, self.variables, key=key, layouts=self._adaptable)
        return replace(self, model=adapter.model, variables=adapter.variables, adapter=adapter)

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """Return the tensors `save` writes, by their source names; `dew.inference.NCCLPush` sends these.

        An adapted bundle (`lora`) writes the source's own tensors with the factors
        merged into their kernels, as PEFT's `merge_and_unload` does, from its
        variables or a trainer's split of them; `adapter.save` writes the factors
        alone.
        """
        values = self.variables if variables is None else variables
        if self.adapter is not None:
            values = self.adapter.merge(values)
        quantization = self._quantization()
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            scalar_mode = _scalar_mode(self.weight_layouts, values)
            layouts = {layout.name: layout for layout in self.weight_layouts}
            if quantization is None:
                return decoders._layout_tensors(layouts, values, scalar_mode, self.retained_tensors)
            tensors = {**self.retained_tensors,
                       **{name: layout.export(values, scalar_mode) for name, layout in layouts.items()}}
        else:
            raise ValueError("this source has no reversible weight layout")
        if quantization is not None:
            tensors = quantization.requantize(tensors, self.quantized_tensors)
        return tensors

    def _quantization(self) -> SourceQuantization | None:
        """The config's quantization format, refused when the loader
        recorded no tensors to write back in it."""
        quantization = source_quantization(self.config, scale_dtype=self.quantized_scale_dtype,
                                           grid=self.quantization_grid)
        if quantization is not None and not self.quantized_tensors:
            raise ValueError(
                "this source's config declares a quantization_config and the loader recorded "
                "no quantized tensors to write back in it")
        return quantization

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write `export`'s tensors, the source's own config.json as published, and its tokenizer files.

        The tensors are written in shards of at most `max_shard_size`.
        """
        from dew.interop.safetensors_io import save_hf_layout
        values = self.variables if variables is None else variables
        destination = Path(directory)
        # Either half may refuse the bundle, each before it writes a file.
        tensors = self.export(values)
        decoders.save_export_assets(destination,
                                    tokenizer=self.processor if self.tokenizer is None else self.tokenizer,
                                    generation_config=dict(self.generation_config),
                                    named=self.names_tokenizer)
        save_hf_layout(tensors, dict(self.config), destination, max_shard_size)

    def push_to_hub(self, repo_id: str, *, variables: Mapping[str, object] | None = None,
                    private: bool = False, commit_message: str = "Upload dew export",
                    max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Upload what `save` writes to the Hub repo `repo_id`, creating the repo if it is missing.

        It calls `save` into a staging directory, then `huggingface_hub.HfApi`'s
        `create_repo` and `upload_folder`. The Hub client handles retries, progress
        and authentication.
        """
        import tempfile

        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory() as staged:
            self.save(staged, variables=variables, max_shard_size=max_shard_size)
            api = HfApi()
            api.create_repo(repo_id, private=private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, folder_path=staged, commit_message=commit_message)


@dataclass(frozen=True)
class PretrainedDecoder(Pretrained):
    """An autoregressive decoder, alone or inside a multimodal wrapper.

    It generates through its KV cache, fine-tunes on next-token prediction and
    takes a low-rank adapter.
    """

    model: CausalTransformer | MultimodalTransformer

    @classmethod
    def from_model(cls, model: nn.Module, variables: Variables, *,
                   tokenizer: str | decoders.ExportTokenizer | None = None,
                   generation_config: Mapping[str, object] | None = None) -> PretrainedDecoder:
        """Wrap a decoder trained in Dew as a bundle that `save` writes in its family's Hugging Face layout.

        The config is derived from the native computation, and every variable
        collection is encoded through the matching family. That is the same encoder a
        loaded source of the derived family exports through, so both write the same
        weights. Gemma 4 writes frozen or trainable layer-scalar values into HF
        buffers; reloading that layout reproduces the computation, but not which
        scalars were trainable. The config is derived from a `causal_transformer`'s
        fields, so any other model is refused.

        `tokenizer` is the vocabulary the weights were trained with, as an object or
        by name. `save` writes its files next to the weights, so the directory that
        `Pretrained.load` reads back includes its processor. `generation_config` is
        what generation_config.json records, `GENERATION_DEFAULTS` when None.
        """
        try:
            decoder = from_record(CausalTransformer, model)
        except ValueError as error:
            raise TypeError(f"{type(model).__name__} has no Hugging Face decoder layout") from error
        config = decoders._export_config(decoder)
        decoders._refuse_lossy_export(decoder, config)
        built = {entry.name: getattr(decoder, entry.name) for entry in dataclasses.fields(decoder)
                 if entry.init and entry.name not in ("parent", "name")}
        return cls(decoder, variables, None, config, None, built,
                   decoders.GENERATION_DEFAULTS if generation_config is None else generation_config,
                   export_adapter=decoders.export_decoder_weights, tokenizer=tokenizer)

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration:
        """Build the text generation task this source describes.

        Without an override, the task runs the source's policy (`task.sampling` holds
        every common control its config sets), any transform chain the rarer controls
        need, and the strategy its config names. An explicit `sampling` replaces the
        policy and drops that chain, because the chain was built for the policy the
        caller replaced; the source's EOS and pad ids fill the ones it leaves None,
        and `num_return_sequences` still comes from the source.

        The task shares the bundle's weights, and `dew.pipeline` packs its decoder
        when it places them. An adapted model whose factors are not merged keeps its
        projection paths and is not packed.
        """
        rows = return_sequences(self.config, self.generation_config)
        policy, logits, strategy = source_decoding(
            self.config, self.generation_config, self.model, rows, sampling)
        return TextGeneration(
            self.model,
            self.variables,
            self.text_processor,
            policy,
            max_new_tokens=generation_limit(self.config, self.generation_config, "max_new_tokens"),
            max_length=generation_limit(self.config, self.generation_config, "max_length"),
            n=rows,
            logits=logits,
            strategy=strategy,
        )



@dataclass(frozen=True)
class PretrainedMaskedDecoder(Pretrained):
    """A masked-diffusion decoder (LLaDA, Dream).

    It is a non-causal decoder with a mask id that generates by unmasking a whole
    response.
    """

    model: CausalTransformer

    def text_generation(self) -> MaskedGeneration:
        """Build the masked generation task this source describes.

        It uses Dew's MDLM sampler, which refines a whole response with Unmask, not
        the source family's own generation recipe.
        """
        from dew.diffusion.discrete import MDLM

        config, generation = self.config, self.generation_config
        mask_id = self.model.mask_token_id
        if mask_id is None:
            raise TypeError("a masked decoder generates by unmasking, and this model names no mask token")
        audit_masked(config, generation)
        return MaskedGeneration(self.model, self.variables, MDLM(mask_id=mask_id)(),
                                self.text_processor,
                                eos_token_ids=eos_ids(config, generation),
                                pad_token_id=pad_id(config, generation),
                                max_new_tokens=generation_limit(config, generation, "max_new_tokens"),
                                max_length=generation_limit(config, generation, "max_length"),
                                n=return_sequences(config, generation))


@dataclass(frozen=True)
class PretrainedBlockDecoder(Pretrained):
    """A DiffusionGemma source, which decodes whole canvases."""

    model: DiffusionGemma
    # transformers reads its generation_config.json into
    # DiffusionGemmaGenerationConfig, which raises on any field it does not
    # declare (generation_diffusion_gemma.py, 5.16.1).
    names_tokenizer: ClassVar[bool] = False

    def block_generation(self) -> BlockGeneration:
        """Build the DiffusionGemma canvas task, with the source's sampler config as its default."""
        from dew.interop import diffusion_gemma

        return BlockGeneration(
            self.model,
            self.variables,
            diffusion_gemma.generation_process(self.config, self.generation_config),
            self.text_processor,
            eos_ids(self.config, self.generation_config),
            pad_id(self.config, self.generation_config),
            max_new_tokens=generation_limit(self.config, self.generation_config, "max_new_tokens"),
            max_length=generation_limit(self.config, self.generation_config, "max_length"),
            n=return_sequences(self.config, self.generation_config),
        )



@dataclass(frozen=True, kw_only=True)
class PretrainedPipeline(Pretrained):
    """A latent diffusion pipeline: the denoiser as `model`, with its process, conditions and autoencoder.

    It also holds the policy the source samples with. It samples, fine-tunes by
    denoising and takes a low-rank adapter on its denoiser. `save` writes one
    tensor set per component, in the diffusers layout.
    """

    process: Process
    inputs: InputSpec
    autoencoder: AutoEncoder | None
    schedule: SourceSchedule
    task: SourceTask
    finish: Callable[[Mapping[str, object], jax.Array], jax.Array] | None = field(default=None, repr=False)

    def text_to_image(self) -> TextToImage:
        """Build the pipeline as a sampling task with its published policy.

        A video source's task (Wan's) samples clips of shape
        `[N, frames, height, width, 3]`.
        """
        return TextToImage(self.model, self.process, self.inputs, self.variables, self.autoencoder,
                           grid=self.task.grid, final_denoise=False, solver=self.schedule.solver,
                           steps=self.task.steps, guidance=self.task.guidance, finish=self.finish)

    @property
    def _adaptable(self) -> Mapping[str, WeightLayout]:
        """The denoiser's layouts alone, the `params`-rooted ones."""
        return {name: layout for name, layout in self.layouts.items() if layout.paths[0][0] == "params"}

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        raise ValueError("a diffusion source writes one tensor set per component; save it instead")

    def save(self, directory: str | Path, *, variables: Mapping[str, object] | None = None,
             max_shard_size: int | str = MAX_SHARD_SIZE) -> None:
        """Write each component's tensors and config in the diffusers layout.

        An adapted bundle's factors are merged into the denoiser's kernels (PEFT's
        `merge_and_unload`), from its variables or a trainer's split of them;
        `adapter.save` writes the factors alone.
        """
        from dew.interop import diffusion

        self._quantization()
        self._text_encoder("a saved pipeline writes it back")
        values = self.variables if variables is None else variables
        if self.adapter is not None:
            values = self.adapter.merge(values)
        diffusion.save_source(self, values, Path(directory))

    def _text_encoder(self, needed: str) -> None:
        """Refuse an operation the text encoder's weights take, on a pipeline
        `load_diffusion_source(text=False)` loaded without them."""
        held = self.variables.get("encoders", {})
        if any(keyword not in held for keyword in self.inputs.conditions):
            raise ValueError(f"this pipeline was loaded without its text encoder (text=False), and {needed}; "
                             "load it with its text encoder")


@dataclass(frozen=True)
class PretrainedFallback(Pretrained):
    """A causal LM run as transformers' PyTorch forward, lowered to JAX by torchax (tier 3).

    It fine-tunes on next-token prediction, without Dew's kernels, sharding rules
    or cached generation; to generate, use transformers' own
    `AutoModelForCausalLM.from_pretrained(source).generate`. Train it like any
    source, `LMObjective(fallback, seq_len)`.
    """


def _native_variables(parts: Mapping[str, Mapping[str, ParamTree]]) -> dict[str, dict[str, ParamTree]]:
    collections: dict[str, dict[str, ParamTree]] = {}
    for component, variables in parts.items():
        for collection, tree in variables.items():
            collections.setdefault(collection, {})[component] = tree
    return collections


class _Call(NamedTuple):
    """Holds one pinned pipeline's own `__call__` policy, read from Diffusers
    0.34.0 (Qwen-Image 2.1 from 6256aa76): the family it belongs to, the
    steps and guidance scale it defaults to, whether that scale guides two
    branches or is the value the model embeds, and the text sequence budget
    it pads its T5 tower to. Qwen-Image's pipeline pads to the longest prompt
    of a call, so its budget is the prompt window Dew pads each row to."""

    family: Literal["sd", "sdxl", "sd3", "flux", "flux2", "qwen_image", "z_image", "wan"]
    steps: int
    guidance: float
    guided: bool
    sequence: int = 0


# The class a file declares is the one whose defaults it gets, so no class is
# normalized into another: the XL inpainting pipeline keeps 7.5 where the
# other two XL ones lowered to 5.0, every Flax pipeline kept its own 7.5, and
# Flux's 3.5 is the guidance its transformer embeds while its own true
# classifier-free guidance is off at the pinned default.
_PIPELINE_POLICY: Mapping[str, _Call] = MappingProxyType({
    "StableDiffusionPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionImg2ImgPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionInpaintPipeline": _Call("sd", 50, 7.5, guided=True),
    "StableDiffusionXLPipeline": _Call("sdxl", 50, 5.0, guided=True),
    "StableDiffusionXLImg2ImgPipeline": _Call("sdxl", 50, 5.0, guided=True),
    "StableDiffusionXLInpaintPipeline": _Call("sdxl", 50, 7.5, guided=True),
    "StableDiffusion3Pipeline": _Call("sd3", 28, 7.0, guided=True, sequence=256),
    "FluxPipeline": _Call("flux", 28, 3.5, guided=False, sequence=512),
    # `true_cfg_scale` defaults to 1.0: the release samples unguided.
    "QwenImage21Pipeline": _Call("qwen_image", 40, 1.0, guided=True, sequence=512),
    # FLUX.2 [dev] embeds its 4.0; [klein] guides two branches at 4.0 unless
    # its index marks it step-distilled, which `_call_policy` reads.
    "Flux2Pipeline": _Call("flux2", 50, 4.0, guided=False, sequence=512),
    "Flux2KleinPipeline": _Call("flux2", 50, 4.0, guided=True, sequence=512),
    # Z-Image guides as `pos + 5.0 (pos - neg)`, which is Dew's
    # `neg + 6.0 (pos - neg)`.
    "ZImagePipeline": _Call("z_image", 50, 6.0, guided=True, sequence=512),
    "WanPipeline": _Call("wan", 50, 5.0, guided=True, sequence=512),
    "FlaxStableDiffusionPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionImg2ImgPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionInpaintPipeline": _Call("sd", 50, 7.5, guided=True),
    "FlaxStableDiffusionXLPipeline": _Call("sdxl", 50, 7.5, guided=True),
})


def _call_policy(index: Mapping[str, object], denoiser: _Denoiser) -> _Call:
    """Return the call policy this file's own pipeline carries.

    A directory that declares no pipeline - a bare component tree - takes its
    family's reference pipeline. A directory that declares one Dew does not
    implement is refused rather than run under another pipeline's defaults,
    and a declared pipeline of another family is refused too: a component
    this loader reads does not qualify a workflow it does not.
    """
    published = index.get("_class_name")
    expected = _PIPELINE_POLICY[denoiser.pipeline]
    if published is None:
        return expected
    found = _PIPELINE_POLICY.get(published) if isinstance(published, str) else None
    if found is None:
        raise ValueError(f"Native diffusion does not implement the published pipeline "
                         f"{published!r}")
    if found.family != expected.family:
        raise ValueError(f"The declared pipeline {published!r} is a {found.family} pipeline, and "
                         f"this directory's denoiser belongs to {denoiser.pipeline!r}")
    if published == "Flux2KleinPipeline" and records.boolean(
        index.get("is_distilled", False), "is_distilled"
    ):
        # A step-distilled [klein] ignores its guidance scale.
        return found._replace(guided=False)
    return found


@dataclass(frozen=True)
class SourceTask:
    """Holds a published pipeline's own call policy.

    `steps` and `guidance` are the defaults its `__call__` signature carries,
    and `grid` prepares the sampling grid the way that pipeline prepares it,
    with the sigma origin it uses and the latent geometry it lays out already
    bound. A pipeline whose guidance is a model input rather than two branches
    carries `guidance=None`.
    """

    steps: int
    guidance: CFG | None
    grid: Callable[[int], tuple[Process, jax.Array]]


@dataclass(frozen=True)
class _TextTowers:
    """The CLIP towers a UNet, SD3 or Flux denoiser reads, the T5 tower where
    its family has one, and the composition `DiffusionConditioner` builds
    from them. `embeds_guidance` marks a transformer that takes the guidance
    scale as a model input rather than as two guided branches."""

    composition: Composition
    towers: tuple[str, ...]
    t5_tower: str | None = None
    embeds_guidance: bool = False
    conditioner: ClassVar[type[DiffusionConditioner]] = DiffusionConditioner

    def components(self, index: Mapping[str, object]) -> tuple[str, ...]:
        """The text components this directory holds, which a conditioner load fetches."""
        return tuple(name for name in (*self.towers, self.t5_tower)
                     if name is not None and _present(index, name))

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: tuple[int, int], *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False
              ) -> tuple[DiffusionConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct the published text composition from metadata and either weight source."""
        names = tuple(name for name in self.towers if _present(index, name))
        if not names:
            raise ValueError("A latent diffusion source needs at least one text encoder")
        towers, tokenizers, text_params, layouts = _clip_towers(
            directory, names, compute, param_dtype=param_dtype, params=params, lazy=lazy)
        components: dict[str, Mapping[str, object]] = {
            name: _component_config(directory, name) for name in names}
        t5 = None
        if self.t5_tower is not None and _present(index, self.t5_tower):
            t5, t5_params, t5_layouts, components[self.t5_tower] = _t5_tower(
                directory, compute, self.t5_tower, policy.sequence, param_dtype=param_dtype,
                params=None if params is None else params[self.t5_tower], lazy=lazy)
            if params is None:
                text_params = {**text_params, self.t5_tower: t5_params}
            layouts += t5_layouts
        height, width = _geometry(index, size)
        encoder = DiffusionConditioner(
            towers, tokenizers, names, text_params, str(directory), height, width,
            denoiser.context_width, composition=self.composition, t5=t5,
            guidance=policy.guidance if self.embeds_guidance and not policy.guided else None,
            aesthetics=bool(index.get("requires_aesthetics_score", False)), param_dtype=param_dtype)
        return encoder, layouts, components

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty-prompt row a file's own pipeline guides against: the XL
        pipelines zero it where their index says so, and the SD3 pipeline
        encodes it with its towers, having no such control."""
        zero = self.composition == "clip_pooled" and bool(index.get("force_zeros_for_empty_prompt", True))
        return {"text": "", "negative": True, "zero": zero}


@dataclass(frozen=True)
class _QwenImageText:
    """Qwen-Image's Qwen3-VL text encoder, which `QwenImageConditioner` runs
    over its pipeline's chat template, padded to the call's token budget."""

    conditioner: ClassVar[type[QwenImageConditioner]] = QwenImageConditioner

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: tuple[int, int], *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False
              ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct `QwenImageConditioner` at the pipeline's prompt budget."""
        return _qwen_image_conditioning(directory, index, compute, size, tokens=policy.sequence,
                                        param_dtype=param_dtype, attention_impl=attention_impl,
                                        params=params, lazy=lazy)

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty prompt, encoded through the same template."""
        return {"text": ""}


@dataclass(frozen=True)
class _HiddenStatesText:
    """The text encoder FLUX.2 (`pipeline="flux2"`: Mistral-3 for [dev],
    Qwen3 for [klein]) or Z-Image (`"z_image"`: Qwen3) reads hidden states
    from, which `HiddenStatesConditioner` runs, padded to the call's token
    budget. `embeds_guidance` marks FLUX.2 [dev]'s transformer, which reads
    the guidance scale as an input."""

    pipeline: Literal["flux2", "z_image"]
    embeds_guidance: bool = False
    conditioner: ClassVar[type[HiddenStatesConditioner]] = HiddenStatesConditioner

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: tuple[int, int], *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False
              ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct `HiddenStatesConditioner` at the pipeline's prompt budget."""
        return _hidden_states_conditioning(
            directory, index, compute, size, pipeline=self.pipeline, tokens=policy.sequence,
            guidance=policy.guidance if self.embeds_guidance else None, param_dtype=param_dtype,
            attention_impl=attention_impl, params=params, lazy=lazy)

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty prompt a guided call encodes as its negative."""
        return {"text": ""}


@dataclass(frozen=True)
class _WanText:
    """Wan's UMT5 encoder, which `WanConditioner` runs at its pipeline's
    prompt budget; it reads no geometry."""

    conditioner: ClassVar[type[WanConditioner]] = WanConditioner

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: tuple[int, int], *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False
              ) -> tuple[WanConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
        """Construct `WanConditioner` at the pipeline's prompt budget."""
        return _wan_conditioning(directory, compute, tokens=policy.sequence, param_dtype=param_dtype,
                                 params=params, lazy=lazy)

    def unconditional(self, index: Mapping[str, object]) -> dict:
        """The empty negative prompt `WanPipeline` encodes by default."""
        return {"text": ""}


@dataclass(frozen=True)
class _Denoiser:
    """Holds what one architecture contributes to a diffusion source, from
    metadata alone: only a complete source load calls `weights`, and `text`
    is how the family conditions on its prompt. `sample_size` is the
    (rows, columns) of latent positions its pipeline renders by default, and
    `frames` the clip length in frames a video pipeline renders, None for an
    image one.
    """

    component: str
    model: nn.Module
    weights: Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]]
    built: Mapping[str, object]
    config: Mapping[str, object]
    text: _TextTowers | _QwenImageText | _HiddenStatesText | _WanText
    patch: int
    latent_input: int
    sample_size: tuple[int, int]
    context_width: int
    pipeline: str
    origin: Origin = "scheduler"
    frames: int | None = None


def _load_diffusion_source(directory: Path, index: Mapping[str, object], *, dtype: str,
                           attention_impl: str, param_dtype: str = "float32",
                           variables: Variables | None = None, lazy: bool = False,
                           text: bool = True) -> PretrainedPipeline:
    """Read a published latent diffusion directory into native modules and variables.

    The directory's own denoiser component selects the family (`_denoiser`),
    and everything the families share - the autoencoder, the text towers,
    the geometry, the conditioning, the safety head a file declares, the
    schedule and the call policy - is read once here.

    Supplied `variables` are a saved tree in this layout, bound as they are:
    every module is built from the directory's metadata and no weight file is
    read, so the directory needs only its configs and tokenizers.

    With `lazy` every component's leaves are `SourceLeaf` recipes over the
    mapped files, for a placement to read one device shard at a time
    (`dew.inference.pipeline.place`), but for the convolution kernels and a
    UNet's per-head attention kernels, which are read whole
    (`dew.interop.weights.record_layouts`). `text=False` binds the text
    encoder to no weights and leaves it out of the variables.
    """
    compute = resolve_dtype(dtype)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    if variables is None:
        denoiser_variables, denoiser_layouts = denoiser.weights(param_dtype, lazy)
    else:
        denoiser_variables = {name: value for name, value in variables.items()
                              if name not in ("encoders", "autoencoder")}
        denoiser_layouts = ()
    held = {} if variables is None else variables["encoders"]
    if not text:
        held = {"conditioning": {name: {} for name in _text_components(denoiser.text, index)}}
    policy = _call_policy(index, denoiser)
    autoencoder, vae_params, vae_layouts, vae_config = _diffusion_vae(
        directory, compute, param_dtype=param_dtype,
        params=None if variables is None else variables["autoencoder"], lazy=lazy)
    rows, columns = denoiser.sample_size
    size = (rows * autoencoder.downscale_factor, columns * autoencoder.downscale_factor)
    encoder, text_layouts, components = denoiser.text.build(
        directory, index, denoiser, policy, compute, size, param_dtype=param_dtype,
        attention_impl=attention_impl, params=held.get("conditioning"), lazy=lazy)
    components.update({denoiser.component: denoiser.config, "vae": vae_config})
    height, width = _geometry(index, size)
    # AutoencoderKLWan declares no channel count; it reads RGB.
    channels = records.integer(vae_config.get("in_channels", 3), "in_channels")
    if denoiser.frames is None:
        sample = Field("image", (height, width, channels))
    else:
        frames = index.get("dew_frames", denoiser.frames)
        if type(frames) is not int or frames < 1:
            raise ValueError("A clip's frame count must be a positive integer")
        sample = Field("video", (frames, height, width, channels))
        # The VAE states which clip lengths it encodes and decodes whole.
        autoencoder.latent_shape(sample.shape)
    inpaint = denoiser.latent_input == autoencoder.latent_channels * 2 + 1
    inputs = InputSpec(
        sample, {encoder.keyword: Condition(encoder, unconditional=denoiser.text.unconditional(index))},
        mask=Field("mask", (height, width, 1)) if inpaint else None,
    )
    encoders: dict[str, object] = {encoder.keyword: encoder.params} if text else {}
    finish, safety_layouts = None, ()
    if _present(index, "safety_checker"):
        finish, encoders["safety"], safety_layouts, safety_configs = _image_safety(
            directory, compute, param_dtype=param_dtype, params=held.get("safety"), lazy=lazy)
        components.update(safety_configs)
    schedule = SourceSchedule.from_config(_component_config(directory, "scheduler"))
    components["scheduler"] = dict(schedule.config)
    patch = denoiser.patch * autoencoder.downscale_factor
    tokens = (height // patch) * (width // patch)
    task = SourceTask(min(policy.steps, schedule.train_steps),
                      CFG(policy.guidance) if policy.guided and policy.guidance > 1 else None,
                      functools.partial(schedule.sampling, origin=denoiser.origin, tokens=tokens))
    variables = {**denoiser_variables, "encoders": encoders, "autoencoder": vae_params}
    geometry = {"dew_height": height, "dew_width": width}
    if denoiser.frames is not None:
        geometry["dew_frames"] = sample.shape[0]
    config = {"model_index": {**index, **geometry}, **components}
    return PretrainedPipeline(model=denoiser.model, variables=variables, processor=None, config=config,
                              source=directory, model_config=denoiser.built,
                              weight_layouts=denoiser_layouts + vae_layouts + text_layouts + safety_layouts,
                              process=schedule.training_process(tokens), inputs=inputs,
                              autoencoder=autoencoder, schedule=schedule, finish=finish, task=task)


def load_diffusion_source(checkpoint: str, *, dtype: str = "bfloat16", param_dtype: str = "float32",
                          revision: str | None = None, attention_impl: str = "auto",
                          size: tuple[int, ...] | None = None, variables: Variables | None = None,
                          mesh: MeshSpec | None = None, layout: Layout | None = None,
                          text: bool = True) -> PretrainedPipeline:
    """A published diffusion pipeline, to train from its own weights.

    `size` is the (height, width) in pixels the pipeline runs at instead of
    its own, or a video pipeline's (frames, height, width): the geometry its
    conditioning, its training shift and its sampling grid are bound to.
    Supplied `variables` are a saved tree of the same pipeline, as a run
    that fine-tuned it wrote them: the modules are built from the
    directory's metadata and bind those variables, and no weight downloads.
    `mesh` and `layout` place the weights it reads as `Pretrained.load`'s
    do, streamed one leaf at a time.

    `text=False` reads and downloads every component's weights but the text
    encoder's, which then need not share the host or the device with the
    denoiser: the conditioner is built from metadata and holds none, a call
    takes the prompts its conditioner encoded alone
    (`TextToImage.prepare(conditions=...)`), and a prompt, training or
    `save`, which read the text encoder, are refused.
    """
    directory = sources.snapshot(checkpoint, revision, weights=False)
    if not (directory / "model_index.json").is_file():
        raise ValueError(f"{checkpoint} is not a diffusion pipeline: it has no model_index.json")
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    if variables is not None and not text:
        raise ValueError("supplied variables bind every component; text=False skips reading one")
    if variables is None:
        skipped = () if text else _text_components(
            _denoiser(directory, dtype=dtype, attention_impl=attention_impl).text, index)
        # Both fetches at the commit the metadata resolved to.
        directory = sources.snapshot(checkpoint, directory.name, weights=tuple(
            name for name in index if _present(index, name) and name not in skipped))
    if size is not None:
        if len(size) not in (2, 3):
            raise ValueError(f"size is (height, width) or (frames, height, width), not {size}")
        index = {**index, "dew_height": size[-2], "dew_width": size[-1]}
        if len(size) == 3:
            index["dew_frames"] = size[0]
    streaming = variables is None and (mesh is not None or layout is not None)
    loaded = _load_diffusion_source(directory, index, dtype=dtype, attention_impl=attention_impl,
                                    param_dtype=param_dtype, variables=variables, lazy=streaming, text=text)
    if streaming:
        loaded = replace(loaded, variables=place(loaded.variables, mesh, layout))
    return replace(loaded, revision=None if os.path.isdir(checkpoint) else directory.name)


def load_diffusion_conditioner[C: ConditionEncoder](
        checkpoint: str, kind: type[C], *, dtype: str | None = "bfloat16", param_dtype: str = "float32",
        revision: str | None = None, attention_impl: str = "auto", tokens: int | None = None,
        params: Variables | None = None, mesh: MeshSpec | None = None, layout: Layout | None = None) -> C:
    """Load the text conditioning a pipeline's denoiser reads, which must be a
    `kind`, or bind supplied parameters using metadata only. `tokens` replaces
    the pipeline's own prompt budget. `mesh` and `layout` place the weights it
    reads as `Pretrained.load`'s do, streamed one leaf at a time, so a
    pipeline's prompts can be encoded with its text encoder alone."""
    from dew.nn.autoencoders import AutoencoderKL

    compute = resolve_dtype(dtype)
    resolve_dtype(param_dtype)
    directory = sources.snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    denoiser = _denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    text = denoiser.text
    if text.conditioner is not kind:
        raise ValueError(f"{checkpoint} conditions through a {text.conditioner.__name__}; "
                         f"build it with {text.conditioner.__name__}.from_pretrained")
    towers = isinstance(text, _TextTowers)
    if params is None:
        # snapshot_download returns a commit directory. Keep both fetches on
        # that commit even when the requested Hub branch moves between them.
        directory = sources.snapshot(checkpoint, directory.name, weights=_text_components(text, index))
    policy = _call_policy(index, denoiser)
    # The CLIP families' geometry is their VAE's; the language-model encoders
    # are bound at 16 pixels per latent position.
    scale = (AutoencoderKL(channels=tuple(_component_config(directory, "vae")["block_out_channels"]))
             .downscale_factor if towers else 16)
    rows, columns = denoiser.sample_size
    streaming = params is None and (mesh is not None or layout is not None)
    encoder, _, _ = text.build(
        directory, index, denoiser, policy if tokens is None else policy._replace(sequence=tokens), compute,
        (rows * scale, columns * scale), param_dtype=param_dtype, attention_impl=attention_impl,
        params=params, lazy=streaming)
    if not isinstance(encoder, kind):
        raise TypeError(f"{type(text).__name__} built a {type(encoder).__name__}, not a {kind.__name__}")
    if streaming:
        encoder.params = place(encoder.params, mesh, layout)
    return encoder


def _unet_denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build the published UNet: cross attention over one or two CLIP towers, whose
    pooled text conditioning is the one its added time features ask for."""
    from dew.interop import diffusion

    config = _component_config(directory, "unet")
    fields = diffusion.unet_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value

    def weights(param_dtype: str, lazy: bool) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = diffusion.translate_unet_weights(
            diffusion.component_tensors(directory, "unet"), model, param_dtype=param_dtype, lazy=lazy)
        return {"params": params}, layouts

    pooled = model.additional_time_features > 0
    built = {"class": "unet_2d_condition",
             "fields": {**fields, "dtype": dtype,
                        "stages": [asdict(stage) for stage in model.stages]}}
    return _Denoiser(
        component="unet",
        model=model,
        weights=weights,
        built=built,
        config=config,
        text=_TextTowers("clip_pooled" if pooled else "clip", ("text_encoder", "text_encoder_2")),
        patch=1,
        latent_input=model.in_channels,
        sample_size=_square(config),
        context_width=records.integer(config.get("cross_attention_dim", 1280), "cross_attention_dim"),
        pipeline="StableDiffusionXLPipeline" if pooled else "StableDiffusionPipeline",
    )


def _denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build the published denoiser this directory holds: its transformer, by the
    class it names, or else its UNet."""
    if not (directory / "transformer" / "config.json").is_file():
        return _unet_denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    config = _component_config(directory, "transformer")
    published = config.get("_class_name")
    builder = _DENOISERS.get(published) if isinstance(published, str) else None
    if builder is None:
        raise ValueError(f"Native diffusion does not implement the published transformer "
                         f"{published!r}")
    return builder(config, directory, dtype=dtype, attention_impl=attention_impl)


def _transformer_weights(directory: Path, translate: Callable[..., tuple[LazyTree, tuple[WeightLayout, ...]]]
                         ) -> Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]]:
    """Read a transformer denoiser's parameters, and their layouts, from its
    component; `lazy` leaves them `SourceLeaf`s for a placement to read."""
    from dew.interop import diffusion

    def weights(param_dtype: str, lazy: bool) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = translate(diffusion.component_tensors(directory, "transformer"),
                                    param_dtype=param_dtype, lazy=lazy)
        return {"params": params}, layouts
    return weights


def _transformer_denoiser(name: str, model: nn.Module, fields: Mapping[str, object], config: dict,
                         dtype: str | None,
                         weights: Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]],
                         text: _TextTowers | _QwenImageText | _HiddenStatesText | _WanText, *,
                         patch: int, latent_input: int, sample_size: tuple[int, int], context_width: int,
                         pipeline: str, origin: Origin = "scheduler", frames: int | None = None) -> _Denoiser:
    """Build the native record with the pipeline's already resolved conditioning and geometry."""
    built = {"class": name, "fields": {**{key: list(value) if isinstance(value, tuple) else value
                                         for key, value in fields.items()}, "dtype": dtype}}
    return _Denoiser(
        component="transformer", model=model, weights=weights, built=built, config=config,
        text=text, patch=patch, latent_input=latent_input, sample_size=sample_size,
        context_width=context_width, pipeline=pipeline, origin=origin, frames=frames)


def _sd3_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build SD3's MM-DiT: both CLIP towers and the T5 tower read jointly, with the
    stored position buffer in its own frozen collection."""
    from dew.interop import diffusion

    fields = diffusion.sd3_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value

    def weights(param_dtype: str, lazy: bool) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, buffers, layouts = diffusion.translate_sd3_weights(
            diffusion.component_tensors(directory, "transformer"), param_dtype=param_dtype, lazy=lazy)
        return {"params": params, "buffers": buffers}, layouts

    return _transformer_denoiser(
        "sd3_transformer", model, fields, config, dtype, weights,
        _TextTowers("sd3", ("text_encoder", "text_encoder_2"), t5_tower="text_encoder_3"),
        patch=model.patch_size, latent_input=model.in_channels, sample_size=_square(config),
        context_width=model.joint_attention_dim, pipeline="StableDiffusion3Pipeline")


def _flux_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build Flux's 2x2-packed latent with pooled CLIP and sequence T5 text.

    The class declares no sample size. A directory overrides its pipeline's
    default of 128 latent positions with its geometry; the pipeline supplies
    the scheduler's starting sigmas.
    """
    from dew.interop import diffusion

    fields = diffusion.flux_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value
    weights = _transformer_weights(directory, diffusion.translate_flux_weights)
    return _transformer_denoiser(
        "flux_transformer", model, fields, config, dtype, weights,
        _TextTowers("flux", ("text_encoder",), t5_tower="text_encoder_2",
                    embeds_guidance=model.guidance_embeds),
        patch=2, latent_input=model.in_channels // 4, sample_size=_square(config, 128),
        context_width=model.joint_attention_dim, pipeline="FluxPipeline", origin="linspace")


def _qwen_image_denoiser(config: dict, directory: Path, *, dtype: str | None,
                         attention_impl: str) -> _Denoiser:
    """Build Qwen-Image 2.1's stream over Qwen3-VL text and latent positions.

    Its config declares no sample size. The pipeline's 1024-pixel resolution
    is 64 positions through its VAE's 16x, with pipeline-supplied sigmas.
    """
    from dew.interop import diffusion

    fields = diffusion.qwen_image_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value
    return _transformer_denoiser(
        "qwen_image_transformer", model, fields, config, dtype,
        _transformer_weights(directory, diffusion.translate_qwen_image_weights), _QwenImageText(),
        patch=1, latent_input=model.in_channels, sample_size=(64, 64),
        context_width=model.context_in_dim, pipeline="QwenImage21Pipeline", origin="linspace")


def _flux2_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build FLUX.2's folded latent on stacked text-encoder states.

    Its pipelines' default 128 positions through the VAE's 8x are 64 folded
    positions. They hand the scheduler `linspace(1, 1/N, N)` with their own
    empirical mu.
    """
    from dew.interop import diffusion

    fields = diffusion.flux2_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value
    guided = model.guidance_embeds
    weights = _transformer_weights(directory, diffusion.translate_flux2_weights)
    return _transformer_denoiser(
        "flux2_transformer", model, fields, config, dtype, weights,
        _HiddenStatesText("flux2", embeds_guidance=guided), patch=1, latent_input=model.in_channels,
        sample_size=(64, 64), context_width=model.joint_attention_dim,
        pipeline="Flux2Pipeline" if guided else "Flux2KleinPipeline", origin="empirical")


def _z_image_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build Z-Image's 2x2-patched latent on its encoder's second-to-last layer.

    The default 1024 pixels are 128 positions through the Flux VAE's 8x.
    The pipeline supplies `linspace(1, 1/N, N)` to its static-shift scheduler.
    """
    from dew.interop import diffusion

    fields = diffusion.z_image_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value
    weights = _transformer_weights(directory, diffusion.translate_z_image_weights)
    return _transformer_denoiser(
        "z_image_transformer", model, fields, config, dtype, weights,
        _HiddenStatesText("z_image"), patch=2, latent_input=model.in_channels, sample_size=(128, 128),
        context_width=model.cap_feat_dim, pipeline="ZImagePipeline", origin="linspace")


def _wan_denoiser(config: dict, directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
    """Build Wan 2.1's video latent on UMT5 text, with 1x2x2 patches.

    Its default 81 frames at 480x832 are 60x104 latent positions through the
    VAE's 8x, with the scheduler's own starting sigmas.
    """
    from dew.interop import diffusion

    fields = diffusion.wan_fields(config, dtype=dtype, attention_impl=attention_impl)
    model = fields.value
    weights = _transformer_weights(directory, diffusion.translate_wan_weights)
    return _transformer_denoiser(
        "wan_transformer", model, fields, config, dtype, weights,
        _WanText(), patch=model.patch_size[-1], latent_input=model.in_channels, sample_size=(60, 104),
        context_width=model.text_dim, pipeline="WanPipeline", frames=81)


_DENOISERS: Mapping[str, Callable[..., _Denoiser]] = MappingProxyType({
    "SD3Transformer2DModel": _sd3_denoiser,
    "FluxTransformer2DModel": _flux_denoiser,
    "QwenImage21Transformer2DModel": _qwen_image_denoiser,
    "Flux2Transformer2DModel": _flux2_denoiser,
    "ZImageTransformer2DModel": _z_image_denoiser,
    "WanTransformer3DModel": _wan_denoiser,
})


def _text_components(text: _TextTowers | _QwenImageText | _HiddenStatesText | _WanText,
                     index: Mapping[str, object]) -> tuple[str, ...]:
    """The components a pipeline's text conditioning reads its weights from."""
    return text.components(index) if isinstance(text, _TextTowers) else ("text_encoder",)


def _component_config(directory: Path, name: str) -> dict:
    """Read one published component's own config file."""
    file = "scheduler_config.json" if name == "scheduler" else "config.json"
    with open(directory / name / file) as handle:
        return json.load(handle)


def _square(config: Mapping[str, object], default: int | None = None) -> tuple[int, int]:
    """A config's `sample_size`, one side of the square its pipeline renders."""
    side = records.integer(config["sample_size"] if default is None else config.get("sample_size", default),
                           "sample_size")
    return side, side


def _geometry(index: Mapping[str, object], size: tuple[int, int]) -> tuple[int, int]:
    """The (height, width) a pipeline is bound to: the index's own, or `size`."""
    height, width = index.get("dew_height", size[0]), index.get("dew_width", size[1])
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    return height, width


def _present(index: Mapping[str, object], name: str) -> bool:
    """Return whether the index declares a component rather than declaring it absent."""
    entry = index.get(name)
    return isinstance(entry, list) and entry[0] is not None


def _diffusion_vae(directory: Path, compute, *, param_dtype: str = "float32",
                   params: Variables | None = None, lazy: bool = False
                   ) -> tuple[AutoEncoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build the published autoencoder, its parameters and their source layouts;
    supplied `params` are bound without a weight read, and `lazy` leaves read
    ones `SourceLeaf`s."""
    from dew.interop import diffusion
    from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
    from dew.nn.autoencoders.vae import _vae_path

    config = _component_config(directory, "vae")
    if config.get("_class_name") == "AutoencoderKLWan":
        from dew.nn.autoencoders.wan import load_wan_vae
        return load_wan_vae(directory, compute, param_dtype=param_dtype, params=params, lazy=lazy)
    if config.get("_class_name") == "AutoencoderKLQwenImage21":
        from dew.nn.autoencoders.qwen_image import load_qwen_image_vae
        return load_qwen_image_vae(directory, compute, param_dtype=param_dtype, params=params, lazy=lazy)
    if config.get("_class_name") == "AutoencoderKLFlux2":
        from dew.nn.autoencoders.flux2 import load_flux2_vae
        return load_flux2_vae(directory, compute, param_dtype=param_dtype, params=params, lazy=lazy)
    model = AutoencoderKL(
        channels=tuple(config["block_out_channels"]),
        latent_channels=config["latent_channels"],
        image_channels=config["in_channels"],
        blocks_per_level=config["layers_per_block"],
        norm_groups=config["norm_num_groups"],
        quantize=diffusion.flag(config, "use_quant_conv", default=True),
        post_quantize=diffusion.flag(config, "use_post_quant_conv", default=True),
        dtype=compute,
    )
    return bind_component(
        directory / "vae", "vae", config, model, _vae_path,
        lambda bound: StableDiffusionVAE(str(directory), dtype=compute, params=bound, model=model,
                                         latent_shift=config.get("shift_factor") or 0.0,
                                         latent_scale=config.get("scaling_factor", 0.18215)),
        prefix=("autoencoder",), params=params, param_dtype=param_dtype, lazy=lazy)


def _clip_towers(directory: Path, names: tuple[str, ...], compute, *, param_dtype: str = "float32",
                 params: Variables | None = None, lazy: bool = False):
    """Build the published CLIP text towers, their tokenizers, their parameters and
    the layouts those parameters came from."""
    from transformers import CLIPTokenizer

    from dew.nn.text_encoders import translate_config

    towers, tokenizers, layouts = [], [], ()
    bound = {} if params is None else params
    for name in names:
        config = _component_config(directory, name)
        model = translate_config(config).value.clone(dtype=compute)
        tower, tree, recorded, _ = bind_component(
            directory / name, name, config, model, lambda name, rank: _text_head_path(name),
            lambda bound, model=model: model, prefix=("encoders", "conditioning", name),
            params=params, param_dtype=param_dtype, lazy=lazy)
        towers.append(tower)
        if params is None:
            bound = {**bound, name: tree}
            layouts += recorded
        tokenizers.append(CLIPTokenizer.from_pretrained(
            directory / ("tokenizer" + name.removeprefix("text_encoder"))))
    return tuple(towers), tuple(tokenizers), bound, layouts


def _t5_tower(directory: Path, compute, component: str, tokens: int, *, param_dtype: str = "float32",
              params: Variables | None = None, lazy: bool = False):
    """Build the published T5 encoder as the conditioner's segment, with its
    parameters, their layouts and its config.

    `component` is where the family keeps it: an SD3 directory's third text
    encoder, a Flux directory's second one; `tokens` is the sequence budget
    the pipeline pads to.
    """
    from dew.data.text import load_tokenizer
    from dew.nn.text_encoders import _t5_path, t5_embedding, translate_t5_config

    config = _component_config(directory, component)
    tower = translate_t5_config(config).value.clone(dtype=compute)
    return bind_component(
        directory / component, component, config, tower, lambda name, rank: _t5_path(name),
        lambda bound: T5Segment(tower, load_tokenizer(str(directory / (
            "tokenizer" + component.removeprefix("text_encoder")))), component, tokens),
        prefix=("encoders", "conditioning", component), params=params,
        param_dtype=param_dtype, lazy=lazy, validate=t5_embedding)


def _wan_conditioning(directory: Path, compute, *, tokens: int, param_dtype: str,
                      params: Variables | None = None, lazy: bool = False
                      ) -> tuple[WanConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
    """Build Wan's conditioner: the UMT5 encoder, its tokenizer, the
    parameters and their layouts."""
    from dew.data.text import load_tokenizer
    from dew.nn.text_encoders import _t5_path, t5_embedding, translate_t5_config

    config = _component_config(directory, "text_encoder")
    if config.get("model_type") != "umt5":
        raise ValueError(f"Wan's text encoder is a umt5 model, not {config.get('model_type')!r}")
    tower = translate_t5_config(config).value.clone(dtype=compute)
    encoder, _, layouts, _ = bind_component(
        directory / "text_encoder", "text_encoder", config, tower, lambda name, rank: _t5_path(name),
        lambda bound: WanConditioner(tower, load_tokenizer(str(directory / "tokenizer")),
                                     bound if params is not None else {"text_encoder": bound},
                                     str(directory), tokens=tokens, param_dtype=param_dtype),
        prefix=("encoders", "conditioning", "text_encoder"), params=params,
        param_dtype=param_dtype, lazy=lazy, validate=t5_embedding)
    return encoder, layouts, {"text_encoder": config}


def _qwen_vl_text_config(config: Mapping[str, object]) -> dict:
    """The Qwen3-VL encoder's text_config as the Qwen3 decoder it computes for a prompt.

    Qwen-Image encodes its text-to-image prompt with no image, and a
    text-only row puts all three of Qwen3-VL's rotary axes at the token's
    position, so the sections of `mrope_section` - interleaved or not -
    rotate every channel pair at that one position: the plain rotary table.
    The rest of the text_config is Qwen3's, down to the per-head query and
    key norms, and the Qwen3 translator reads and checks it.
    """
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"Qwen-Image's text encoder is a qwen3_vl model, not {config.get('model_type')!r}")
    text = dict(records.record(config["text_config"], "text_config"))
    if text.get("model_type") != "qwen3_vl_text":
        raise ValueError(f"A qwen3_vl text_config is qwen3_vl_text, not {text.get('model_type')!r}")
    half = records.integer(text["head_dim"], "head_dim") // 2
    for key in ("rope_parameters", "rope_scaling"):
        entry = text.get(key)
        if isinstance(entry, Mapping):
            entry = dict(entry)
            section = records.integers(entry.pop("mrope_section"), "mrope_section")
            records.boolean(entry.pop("mrope_interleaved", False), "mrope_interleaved")
            if sum(section) != half:
                raise ValueError(f"mrope_section {section} must cover the {half} channel pairs")
            text[key] = entry
    return {**text, "model_type": "qwen3",
            "tie_word_embeddings": records.boolean(config.get("tie_word_embeddings", False),
                                                   "tie_word_embeddings")}


def _qwen_text_path(record: decoders.DecoderFields):
    """Map a Qwen3-VL checkpoint's tensors: its language model as the Qwen3
    decoder's, its head too, and its vision tower held as stored, which the
    text-to-image prompt never reads and an export writes back."""
    family = decoders.families()["qwen3"]

    def path(name: str) -> tuple[str, ...] | None:
        if name.startswith("model.language_model."):
            return family.weight_path("model." + name.removeprefix("model.language_model."), record)
        if name == "lm_head.weight":
            return family.weight_path(name, record)
        if name.startswith("model.visual."):
            # One flat leaf per stored tensor: these are carried, not run,
            # so no module path, and no sharding rule, reads them.
            return ("visual", name.removeprefix("model.visual."))
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def _qwen_image_conditioning(directory: Path, index: Mapping[str, object], compute, size: tuple[int, int], *,
                             tokens: int, param_dtype: str, attention_impl: str,
                             params: Variables | None = None, lazy: bool = False
                             ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...],
                                        dict[str, Mapping[str, object]]]:
    """Build Qwen-Image's conditioner: the Qwen3-VL language model, its
    processor's tokenizer and chat template, the parameters and their layouts."""
    from dew.data.text import load_tokenizer
    from dew.interop import diffusion

    config = _component_config(directory, "text_encoder")
    record = decoders.translate_config(_qwen_vl_text_config(config))
    named = dtype_name(compute)
    if named is None:
        raise ValueError("Qwen-Image's Qwen3-VL encoder computes in a named dtype; pass dtype")
    decoder = from_record(CausalTransformer, {**record, "dtype": named, "attention_impl": attention_impl})
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tower, layouts = checkpoint_weights.record_layouts(
            "text_encoder", diffusion.component_tensors(directory, "text_encoder"),
            _qwen_text_path(record), ("encoders", "conditioning", "text_encoder"),
            param_dtype=param_dtype, lazy=lazy)
        params = {"text_encoder": tower}
    height, width = _geometry(index, size)
    encoder = QwenImageConditioner(
        decoder, load_tokenizer(str(directory / "processor")), params, str(directory),
        height, width, tokens=tokens, param_dtype=param_dtype)
    return encoder, layouts, {"text_encoder": config}


_FLUX2_TEXT: Mapping[str, tuple[Literal["qwen3", "mistral3"], tuple[int, ...]]] = MappingProxyType({
    "qwen3": ("qwen3", (9, 18, 27)), "mistral3": ("mistral3", (10, 20, 30))})
"""Each FLUX.2 text encoder's `model_type`, the template its pipeline
formats a prompt with, and the `hidden_states` it stacks."""


def _hidden_states_path(record: decoders.DecoderFields, family: str, multimodal: bool):
    """Map a hidden-states text encoder's tensors: a Qwen3 language model's
    own names, or a Mistral-3's language model as the Mistral decoder's with
    its vision tower and projector held as stored, which a text prompt never
    reads and an export writes back. A Mistral-3 checkpoint names its
    language model `language_model.model.` and its head
    `language_model.lm_head` (transformers 4.50 and 5 write these), or
    `model.language_model.` and `lm_head` (4.52 to 4.57)."""
    decoder = decoders.families()[family]

    def path(name: str) -> tuple[str, ...] | None:
        if not multimodal or name == "lm_head.weight":
            return decoder.weight_path(name, record)
        if name == "language_model.lm_head.weight":
            return decoder.weight_path("lm_head.weight", record)
        for prefix in ("model.language_model.", "language_model.model."):
            if name.startswith(prefix):
                return decoder.weight_path("model." + name.removeprefix(prefix), record)
        if name.removeprefix("model.").startswith(("vision_tower.", "multi_modal_projector.")):
            return ("visual", name)
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def _hidden_states_conditioning(directory: Path, index: Mapping[str, object], compute,
                                size: tuple[int, int], *,
                                pipeline: Literal["flux2", "z_image"], tokens: int, guidance: float | None,
                                param_dtype: str, attention_impl: str, params: Variables | None = None,
                                lazy: bool = False
                                ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...],
                                           dict[str, Mapping[str, object]]]:
    """Build FLUX.2's or Z-Image's conditioner: the text encoder's language
    model, the tokenizer and chat template, the parameters and their layouts.
    Z-Image reads the output of the encoder's second-to-last layer with the
    template's thinking on."""
    from dew.data.text import load_tokenizer
    from dew.interop import diffusion

    config = _component_config(directory, "text_encoder")
    kind = records.text(config.get("model_type"), "model_type")
    known = _FLUX2_TEXT if pipeline == "flux2" else {"qwen3": _FLUX2_TEXT["qwen3"]}
    if kind not in known:
        raise ValueError(f"The {pipeline} text encoder is one of {sorted(known)}, not {kind!r}")
    template, layers = known[kind]
    multimodal = kind == "mistral3"
    text = dict(records.record(config["text_config"], "text_config")) if multimodal else config
    if multimodal:
        text["tie_word_embeddings"] = records.boolean(config.get("tie_word_embeddings", False),
                                                      "tie_word_embeddings")
    record = decoders.translate_config(text)
    named = dtype_name(compute)
    if named is None:
        raise ValueError(f"The {pipeline} text encoder computes in a named dtype; pass dtype")
    decoder = from_record(CausalTransformer, {**record, "dtype": named, "attention_impl": attention_impl})
    if pipeline == "z_image":
        layers = (decoder.num_layers - 1,)
    if max(layers) >= decoder.num_layers:
        # transformers' last hidden state is the final norm's output, which
        # none of these pipelines reads of its released encoder.
        raise ValueError(f"{pipeline} reads hidden state {max(layers)}, which a {decoder.num_layers}-layer "
                         "encoder does not have before its final norm")
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tower, layouts = checkpoint_weights.record_layouts(
            "text_encoder", diffusion.component_tensors(directory, "text_encoder"),
            _hidden_states_path(record, records.text(text["model_type"], "model_type"), multimodal),
            ("encoders", "conditioning", "text_encoder"), param_dtype=param_dtype, lazy=lazy)
        params = {"text_encoder": tower}
    height, width = _geometry(index, size)
    encoder = HiddenStatesConditioner(
        decoder, load_tokenizer(str(directory / "tokenizer")), params, str(directory), height, width,
        template=template, layers=layers, thinking=pipeline == "z_image", tokens=tokens, guidance=guidance,
        param_dtype=param_dtype)
    return encoder, layouts, {"text_encoder": config}


def _image_safety(directory: Path, compute, *, param_dtype: str = "float32",
                  params: Variables | None = None, lazy: bool = False):
    """Build the safety head a file declares: the finish, its parameters, their
    layouts and the two configs it ships; supplied `params` are bound without
    a weight read."""
    from dew.inputs.diffusion import CLIPImageTransform, CLIPSafetyHead, ImageSafety
    from dew.interop import diffusion
    from dew.nn.text_encoders import translate_vision_config

    config = _component_config(directory, "safety_checker")
    with open(directory / "feature_extractor" / "preprocessor_config.json") as handle:
        transform = json.load(handle)
    layouts: tuple[WeightLayout, ...] = ()
    if params is None:
        tensors = diffusion.component_tensors(directory, "safety_checker")
        # The root scoring vectors and thresholds are state, not tower/projection
        # weights. Preserve their FP32 contract without post-casting a whole tree.
        state = {name: value for name, value in tensors.items()
                 if (path := _safety_path(name)) is not None and len(path) == 1}
        weights = {name: value for name, value in tensors.items() if name not in state}
        params, layouts = checkpoint_weights.record_layouts(
            "safety_checker", weights, _safety_path, ("encoders", "safety"), param_dtype=param_dtype,
            lazy=lazy)
        scoring, state_layouts = checkpoint_weights.record_layouts(
            "safety_checker", state, _safety_path, ("encoders", "safety"), param_dtype="float32", lazy=lazy)
        params.update(scoring)
        layouts += state_layouts
    head = CLIPSafetyHead(translate_vision_config(config).value.clone(dtype=compute),
                          int(config["projection_dim"]), dtype=compute)
    return (ImageSafety(head, CLIPImageTransform.from_config(transform)), params, layouts,
            {"safety_checker": config, "feature_extractor": transform})


def _text_head_path(name: str):
    from dew.nn.text_encoders import _text_path
    if name == "text_projection.weight":
        return ("text_projection", "kernel")
    path = _text_path(name)
    return None if path is None else ("text_model", *path)


def _safety_path(name: str):
    from dew.nn.text_encoders import _clip_path
    if name.startswith(("concept_embeds", "special_care_embeds")):
        return (name,)
    return _clip_path(name.removeprefix("vision_model."))


def _wrapper_text_fields(config: Mapping[str, object], record: decoders.WrapperFields,
                         max_seq_len: int | None) -> decoders.DecoderFields:
    """Read the decoder fields a wrapper's text_config states, with the corrections
    its own family makes to them.

    Gemma 3 projects its logits without the causal-LM class's final tanh cap
    and attends its image spans both ways; Gemma 4 does the same on its
    sliding layers when the config asks for it; Qwen3.5 splits its rotary
    into the three mrope sections its wrapper positions rows with.
    """
    family = config.get("model_type")
    text_config = records.record(config["text_config"], "text_config")
    text_fields = record["text"].copy()
    if max_seq_len is not None:
        text_fields["max_seq_len"] = max_seq_len
    if family == "gemma3":
        text_fields["final_logit_softcap"] = None
        text_fields["mixer"] = {"class": "attention", "fields": {"bidirectional_images": True}}
    if family == "gemma4" and text_config.get("use_bidirectional_attention") == "vision":
        kinds = decoders._kinds_of(text_fields).copy()
        sliding = NativeFields(decoders.LayerKind, {
            **kinds.get("sliding_attention", {}),
            "mixer": {"class": "attention", "fields": {"bidirectional_images": True}}})
        kinds["sliding_attention"] = sliding
        text_fields["kinds"] = kinds
    if family in _QWEN35_TYPES:
        rope = records.record(text_config.get("rope_parameters") or {}, "rope_parameters")
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(type(value) is not int or value < 0 for value in sections)):
            raise ValueError("mrope_section must contain three nonnegative integer widths")
        kinds = decoders._kinds_of(text_fields).copy()
        full = NativeFields(decoders.LayerKind, {
            **kinds.get("full_attention", {}),
            "mixer": {"class": "attention", "fields": {
                      "mrope_section": [sections[0], sections[1], sections[2]]}}})
        kinds["full_attention"] = full
        text_fields["kinds"] = kinds
    return text_fields


def _wrapper_model(config: Mapping[str, object], record: decoders.WrapperFields,
                   language_model: CausalTransformer, *, dtype: str) -> MultimodalTransformer:
    """Build the wrapper its record describes: the decoder above, the towers and
    projectors it names, and the placeholder ids its prompts carry."""
    family = records.text(config["model_type"], "model_type")
    text_config = records.record(config["text_config"], "text_config")
    audio_record = record["audio"]
    audio_projector = record["audio_projector"]
    return MultimodalTransformer(
        language_model, towers.from_record(record["tower"]),
        projectors.from_record(record["projector"]), family,
        record["image_token_id"], dtype=resolve_dtype(dtype),
        # A text_config that states a null pad id states none, and the
        # wrapper's own field defaults to 0 for exactly that.
        pad_token_id=records.integer(text_config.get("pad_token_id") or 0, "pad_token_id"),
        extra_placeholder_ids=(tuple(records.integer(config.get(name, default), name) for name, default in
            (("video_token_id", 258884), ("audio_token_id", 258881))) if family == "gemma4" else ()),
        audio=None if audio_record is None else towers.from_record(audio_record),
        audio_projection=(None if audio_projector is None
                          else projectors.from_record(audio_projector)),
        audio_soft_tokens=record["audio_soft_tokens"])


def _source_processor(directory: Path, config: Mapping[str, object], record: Mapping[str, object],
                      model: nn.Module, gguf_path: Path | None = None, *, media: bool = False
                      ) -> Processor | None:
    """Build the host preprocessing a source ships, or None where it ships none.

    Only a `media` source, a wrapper whose towers the loader built beside its
    decoder (`_media_wrapper`), reads images or audio, and only through the
    processor its repo ships; a text decoder takes its tokenizer whatever
    processor files sit beside it (DiffusionGemma publishes a Gemma 4
    processor config beside a text-only model), and a tiny multimodal
    fixture without processor files tokenizes text only.
    """
    processor_files = any((directory / name).exists()
                          for name in ("processor_config.json", "preprocessor_config.json"))
    if media and processor_files:
        from transformers import AutoProcessor
        options = {"backend": "pil"} if config.get("model_type") == "gemma3" else {}
        reference = AutoProcessor.from_pretrained(str(directory), local_files_only=True, **options)
        return Processor(reference, config, record, model.vocab_size)
    if (directory / "tokenizer_config.json").exists() or gguf_path is not None:
        from dew.data.text import load_tokenizer
        # A GGUF repo ships none; the file carries its tokenizer.
        tokenizer = (gguf.tokenizer(gguf_path) if gguf_path is not None
                     and not (directory / "tokenizer_config.json").exists()
                     else load_tokenizer(str(directory), local_files_only=True))
        if not _hosts(tokenizer):
            raise TypeError(f"the tokenizer in {directory} lacks a host processor operation")
        return Processor(tokenizer, config, record, model.vocab_size)
    return None


def split_revision(source: str) -> tuple[str, str | None]:
    """Split a `repo@revision` reference into the repo and the revision.

    This is Levanter's `RepoRef` spelling: a branch, tag or commit after the last
    '@'. A local directory, or a reference without '@', names no revision. Hub
    repo IDs cannot contain '@'.
    """
    if os.path.isdir(source) or "@" not in source:
        return source, None
    name, revision = source.rsplit("@", 1)
    if not name or not revision:
        raise ValueError(f"{source!r} is not a repo@revision reference")
    return name, revision


AUTO = "auto"
"""The param_dtype that stores a checkpoint's parameters in its own dtype."""


def _checkpoint_dtype(config: Mapping[str, object], tensors: Mapping[str, np.ndarray]) -> str:
    """Return the storage dtype a checkpoint states, for param_dtype 'auto'.

    transformers' dtype='auto' rule (modeling_utils.py `_get_dtype`, 5.16.1):
    config.json's `dtype` (`torch_dtype` before 5.0), else the dtype of the
    first floating tensor. Packed FP8 or FP4 payloads are no storage dtype,
    so the first tensor stored in one is what a quantized checkpoint without
    a stated dtype resolves to. A diffusers pipeline states none, so its
    denoiser's tensors decide.
    """
    stated = config.get("dtype", config.get("torch_dtype"))
    if stated is not None:
        storage = dtype_name(resolve_dtype(records.text(stated, "dtype")))
        if storage is None:
            raise ValueError(f"dtype={stated!r} names no floating parameter storage")
        return storage
    storable = {np.dtype(np.float32): "float32", np.dtype(np.float16): "float16",
                np.dtype(ml_dtypes.bfloat16): "bfloat16"}
    for tensor in tensors.values():
        if tensor.dtype in storable:
            return storable[tensor.dtype]
    raise ValueError("param_dtype 'auto' found neither a stated dtype nor a float32, bfloat16 or "
                     "float16 tensor in the checkpoint")


def _pipeline_source(name_or_dir: str | Path, directory: Path, commit: str | None, single_file: str | None,
                     dduf_file: str | None, placed: Callable[[Variables], Variables], *, dtype: str,
                     attention_impl: str, param_dtype: str, streaming: bool) -> PretrainedPipeline | None:
    """The source as a latent diffusion pipeline, or None when it is a decoder.

    A single file converts into the pipeline it describes, and a DDUF file
    unpacks into the one it packs; a directory with
    a model_index.json and no config.json of its own is one. A decoder that
    also ships a pipeline index for its sampler (DiffusionGemma) is loaded as
    the decoder its config names. `placed` puts the pipeline's variables on
    the mesh, and `streaming` has them read one leaf at a time as it does; a
    single file's conversion is kept only once that succeeds too.
    """

    def pipeline(directory: Path) -> PretrainedPipeline:
        with open(directory / "model_index.json") as handle:
            index = json.load(handle)
        storage = param_dtype
        if storage == AUTO:
            from dew.interop import diffusion
            denoiser = "transformer" if (directory / "transformer" / "config.json").is_file() else "unet"
            storage = _checkpoint_dtype({}, diffusion.component_tensors(directory, denoiser))
        loaded = _load_diffusion_source(directory, index, dtype=dtype, attention_impl=attention_impl,
                                        param_dtype=storage, lazy=streaming)
        return replace(loaded, variables=placed(loaded.variables), revision=commit)

    if single_file is not None and dduf_file is not None:
        raise ValueError(f"single_file={single_file!r} and dduf_file={dduf_file!r} each name a whole "
                         "pipeline; pass one")
    if dduf_file is not None:
        from dew.interop import dduf
        return pipeline(dduf.unpacked(sources.repo_file(name_or_dir, directory, dduf_file)))
    if single_file is not None:
        from dew.interop import single_file as original
        # A repo or directory that describes the pipeline (model_index.json
        # and its component configs) is the configs, as from_single_file(config=)
        # takes; a Hub repo's are its metadata snapshot, so the weights of a
        # component the file lacks come from the repo at that commit.
        configs = directory if (directory / "model_index.json").is_file() else None
        hub = None if configs is None or commit is None else (str(name_or_dir), commit)
        with original.unpacked(sources.repo_file(name_or_dir, directory, single_file), configs, hub) as (
                converted, published):
            return replace(pipeline(converted), source=published)
    if (directory / "model_index.json").is_file() and not (directory / "config.json").is_file():
        with open(directory / "model_index.json") as handle:
            index = json.load(handle)
        # The metadata fetch returns its commit directory, so the weights
        # come from that commit even if the requested branch moves.
        return pipeline(sources.snapshot(str(name_or_dir), directory.name, weights=tuple(
            name for name in index if _present(index, name))))
    return None


def _source_config(name_or_dir: str | Path, directory: Path, commit: str | None,
                   gguf_path: Path | None) -> tuple[Mapping[str, object], Mapping[str, np.ndarray] | None]:
    """The decoder's config, and its tensors where the config came with them
    (a GGUF file holds both); refused, naming what the source ships instead,
    before any weight downloads when there is no config.json."""
    if gguf_path is not None:
        return gguf.read(gguf_path)
    if not (directory / "config.json").is_file():
        # A GGUF repo often ships none, and says which argument reads it.
        files = (sources.repo_files(str(name_or_dir), directory) if commit is not None else
                 {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()})
        shipped = sources.missing_weights(str(name_or_dir), files)
        raise FileNotFoundError(f"{name_or_dir} has no config.json, which says what model its weights "
                                f"are; {shipped}")
    with open(directory / "config.json") as handle:
        config = records.record(json.load(handle), "config.json")
    # A GGUF file holds no kimi_k25 wrapper, so only a config.json can carry this.
    text_config = config.get("text_config")
    if (config.get("model_type") == "kimi_k25" and isinstance(text_config, Mapping)
            and text_config.get("quantization_config") is not None):
        raise ValueError(
            "text_config.quantization_config is not supported for the kimi_k25 text-only loader; "
            "provide dequantized text weights and remove that quantization descriptor")
    return config, None


def _decoded(tensors: Mapping[str, np.ndarray], config: Mapping[str, object], param_dtype: str
             ) -> tuple[Mapping[str, np.ndarray], tuple[str, ...], str | None, dict[str, np.ndarray]]:
    """The tensors with a quantized source's weights decoded, and what `save`
    needs to write them back: the quantized names, the scales' dtype and the
    integer formats' grid."""
    quantization = source_quantization(config)
    if quantization is None:
        return tensors, (), None, {}
    quantized_tensors, scale_dtype = quantization.names(tensors), quantization.scale_dtype(tensors)
    grid = {part: tensors[part] for name in quantized_tensors for part in quantization.grid(name)
            if part in tensors}
    aliases: tuple[tuple[str, str], ...] = ()
    if param_dtype != "float32":
        aliases = decoders.validate_source_aliases(
            quantization.tensor_names(tensors), partial(quantization.read, tensors), config)
    tensors = quantization.dequantize(tensors, param_dtype=param_dtype)
    _share_quantized_aliases(tensors, aliases, quantized_tensors)
    return tensors, quantized_tensors, scale_dtype, grid


def _input_quantization(model: nn.Module, layouts: tuple[WeightLayout, ...],
                        config: Mapping[str, object], grid: Mapping[str, np.ndarray]) -> nn.Module:
    """Bind each stored input scale to its Linear's native Qwix scope.

    Expert stacks and concatenated projections need one quantizer per
    member, which this per-Linear binding cannot compute, so they are refused
    before returning a model with the wrong activation forward.
    """
    from dew.training.quantization import FP8Input, NVFP4Input, checkpoint_input_quantization

    codec = source_quantization(config)
    if codec is None or codec.input_scale_dtype is None:
        return model
    inputs: dict[str, FP8Input | NVFP4Input] = {}
    for layout in layouts:
        part = layout.name.removesuffix('.weight') + codec.input_suffix
        if part not in grid:
            continue
        if (len(layout.paths) != 1 or layout.paths[0][-1] != 'kernel'
                or layout.expert_index is not None or layout.concatenate is not None):
            raise ValueError(f"{layout.name} needs NVFP4 input QDQ per expert or projection member, "
                             "which this checkpoint provider does not compute")
        scale = np.asarray(grid[part])
        if scale.dtype != np.float32 or scale.size != 1:
            raise ValueError(f"{part} must be one float32 stored global scale, "
                             f"got {scale.dtype} {scale.shape}")
        path = '/'.join(layout.paths[0][1:-1])
        if codec.input_kind(layout.name) == 'fp8':
            inputs[path] = FP8Input(float(scale.reshape(())))
        else:
            inputs[path] = NVFP4Input(float(scale.reshape(())), codec.input_scale_dtype == 'float8_e4m3fn',
                                     format=codec.input_format)
    if len(inputs) != sum(name.endswith(codec.input_suffix) for name in grid):
        raise ValueError("NVFP4 input scales must each bind one Linear scope; "
                         "an unbound scale would drop QDQ")
    return checkpoint_input_quantization(model, inputs)


def _decoder_layouts(tensors: Mapping[str, np.ndarray], record: decoders.DecoderFields, family: str,
                     variables: Variables) -> tuple[tuple[WeightLayout, ...], dict[str, np.ndarray]]:
    """Each source tensor's binding into the tree, and the tensors none binds."""
    bindings, retained = [], {}
    for name, tensor in tensors.items():
        binding = _language_layout(name, name, tensor, record, family, variables)
        if binding is None:
            retained[name] = tensor
        else:
            bindings.append(binding)
    return tuple(bindings), retained


def _generation_config(directory: Path) -> Mapping[str, object]:
    """The source's generation_config.json, or {} when it has none."""
    generation_path = directory / "generation_config.json"
    generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}

    def policy_read() -> None:
        """Refuse a generation_config.json that is not an object.

        Loading for training or export does not opt into the source sampler;
        active policy support is checked when the caller creates its task.
        """
        if not isinstance(generation_config, dict):
            raise ValueError("generation_config.json must contain an object")

    agreed("pretrained generation policy", policy_read)
    return generation_config


class _Built(NamedTuple):
    """A decoder source built and bound: the model, its variables, the
    record it was translated to, what it was built from, and the source
    tensors' bindings and the ones none binds."""

    model: nn.Module
    variables: Variables
    record: Mapping[str, object]
    built: Mapping[str, object]
    layouts: tuple[WeightLayout, ...]
    retained: dict[str, np.ndarray]


def _wrapper_source(config: Mapping[str, object], tensors: Mapping[str, np.ndarray], directory: Path, *,
                    dtype: str, attention_impl: str, max_seq_len: int | None, param_dtype: str,
                    lazy: bool) -> _Built:
    """A multimodal wrapper: the decoder under its text_config and the towers beside it."""
    record = decoders.translate_wrapper_config(config)
    text_fields = _wrapper_text_fields(config, record, max_seq_len)
    # The Transformers conditional classes ignore auxiliary prediction
    # layers. A released config advertises a depth even when its checkpoint
    # contains only the trunk; a source with mtp.* retains its actual depth.
    if (record['text_model_type'] in _QWEN35_TEXT_TYPES
            and not any(name.startswith(('mtp.', 'model.mtp.')) for name in tensors)):
        text_fields['num_nextn_predict_layers'] = 0
        record['text']['num_nextn_predict_layers'] = 0
    text = NativeFields(CausalTransformer, {**text_fields, "dtype": dtype, "attention_impl": attention_impl})
    wrapper: decoders.WrapperFields = {**record, "text": text}
    language_model = from_record(CausalTransformer, wrapper["text"])
    model = _wrapper_model(config, record, language_model, dtype=dtype)
    parts = decoders.translate_wrapper_weights(tensors, record, param_dtype=param_dtype, lazy=lazy)
    variables = _native_variables({**parts, "language_model": decoders.with_constants(
        parts["language_model"], record["text"], directory)})
    layouts, retained = _wrapper_layouts(tensors, record, variables)
    return _Built(model, variables, record, wrapper, layouts, retained)


def _media_wrapper(config: Mapping[str, object]) -> bool:
    """Whether a source is a media wrapper, its decoder under text_config.

    Where its model_type is a registered decoder family, the text half is
    the model unless the family reads its media bundle whole
    (`DecoderFamily.wrapper`). DiffusionGemma nests its text config too.
    """
    family = config.get("model_type")
    return (family != "diffusion_gemma" and "text_config" in config
            and (family not in decoders.families() or decoders._bundles(config)))


def _derived_weights(model: CausalTransformer, variables: Mapping[str, object],
                     config: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Write a loaded decoder through the decoder export's own encoder, under
    the config its fields derive rather than the `config` it shipped."""
    return decoders.export_decoder_weights(model, variables, decoders._export_config(model))


def _decoder_source(config: Mapping[str, object], tensors: Mapping[str, np.ndarray], directory: Path,
                    verified: verify.VerifiedMapping | None, *, dtype: str, attention_impl: str,
                    max_seq_len: int | None, param_dtype: str, lazy: bool) -> _Built:
    """A decoder of a registered family, or of one it was verified as (tier 2)."""
    if verified is None:
        record = decoders.translate_config(config)
        # translate_config refused every model_type but a registered family's name.
        family = records.text(config.get("model_type"), "model_type")
    else:
        record = verified.translate(config, tensors)
        family = verified.family
    if max_seq_len is not None:
        record["max_seq_len"] = max_seq_len
    built = {**record, "dtype": dtype, "attention_impl": attention_impl}
    model = from_record(CausalTransformer, built)
    variables = decoders.with_constants(decoders.translate_weights(
        tensors, record, family, param_dtype=param_dtype, lazy=lazy), record, directory)
    decoders._check_tree(variables, model)
    # The bindings are what an adapter loader resolves source names through
    # and what a quantized source is written back through, so a
    # derived-export family binds too; `save` picks its writer by
    # preserve_source_layout and quantization, not by whether bindings exist.
    # A family whose tensors are rewritten before the path map reads them
    # (GPT-2's and GPT-NeoX's prepare) has no raw-name bindings.
    entry = decoders.families()[family]
    layouts, retained = ((), {})
    if entry.preserve_source_layout or entry.prepare is decoders.DecoderFamily.prepare:
        layouts, retained = _decoder_layouts(tensors, record, family, variables)
    return _Built(model, variables, record, built, layouts, retained)


def _load_native_source(name_or_dir: str | Path, directory: Path, commit: str | None, *,
                        gguf_file: str | None, placed: Callable[[Variables], Variables], streaming: bool,
                        dtype: str, param_dtype: str, attention_impl: str,
                        max_seq_len: int | None,
                        prepare: Callable[[nn.Module, Variables], Variables] | None = None) -> Pretrained:
    """Decode native source weights and bind their processors and export layout.

    Diffusers pipelines and the explicit torchax fallback have already taken
    their own routes; this boundary handles native decoders and media wrappers.
    """
    gguf_path = None if gguf_file is None else sources.repo_file(name_or_dir, directory, gguf_file)
    config, tensors = _source_config(name_or_dir, directory, commit, gguf_path)
    mamba_ssm = mamba2.is_mamba_ssm(config)
    if mamba_ssm:
        config = mamba2.config_from_mamba_ssm(config)
    family = config.get("model_type")
    # Before any weight downloads: a format the codec cannot read is refused
    # on the config alone.
    source_quantization(config)
    # An unregistered decoder is checked against transformers on the config
    # alone, before its weights download (tier 2, dew.interop.verify).
    verified = (verify.verify_mapping(config) if isinstance(family, str) and family not in decoders.families()
                and family != "diffusion_gemma" and "text_config" not in config else None)
    if tensors is None:
        directory = sources.snapshot(str(name_or_dir), directory.name)
        tensors = sources.load_shards(directory)
    if mamba_ssm:
        tensors = mamba2.tensors_from_mamba_ssm(tensors)
    if param_dtype == AUTO:
        param_dtype = _checkpoint_dtype(config, tensors)
    tensors, quantized_tensors, scale_dtype, grid = _decoded(tensors, config, param_dtype)
    export_adapter = None
    media = _media_wrapper(config)
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma

        model = diffusion_gemma.build(
            config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len
        )
        variables = diffusion_gemma.translate_weights(tensors, config, param_dtype=param_dtype)
        record, layouts, retained = config, (), {}
        built: Mapping[str, object] = {**config, "dtype": dtype, "attention_impl": attention_impl}
        export_adapter = diffusion_gemma.export_weights
    elif media:
        model, variables, record, built, layouts, retained = _wrapper_source(
            config, tensors, directory, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len,
            param_dtype=param_dtype, lazy=streaming)
    else:
        model, variables, record, built, layouts, retained = _decoder_source(
            config, tensors, directory, verified, dtype=dtype, attention_impl=attention_impl,
            max_seq_len=max_seq_len, param_dtype=param_dtype, lazy=streaming)
        if (isinstance(family, str) and family in decoders.families()
                and not decoders.families()[family].preserve_source_layout
                and source_quantization(config) is None):
            # The decoder export's own encoder, so this and `PretrainedDecoder.from_model`
            # leave the same weights. A quantized source keeps its packed format
            # by going back over its source names (`weight_layouts`).
            export_adapter = _derived_weights
    model = _input_quantization(model, layouts, config, grid)
    processor = _source_processor(directory, config, record, model, gguf_path, media=media)
    generation_config = _generation_config(directory)

    # Dew's byte vocabulary has no files: an export names it in
    # generation_config.json, and the name is the whole record of it.
    named = generation_config.get("tokenizer_name") if processor is None else None

    def bundle[K: Pretrained](kind: type[K]) -> K:
        prepared = variables if prepare is None else prepare(model, variables)
        return kind(model, placed(prepared), processor, config, directory, built, generation_config,
                    layouts, retained, export_adapter, quantized_tensors=quantized_tensors,
                    quantized_scale_dtype=scale_dtype, quantization_grid=grid,
                    # The weights' commit: a pickle repo's may be its conversion's.
                    revision=None if commit is None else directory.name,
                    tokenizer="byte" if named == "byte" else None)

    if family == "diffusion_gemma":
        return bundle(PretrainedBlockDecoder)
    # The masked-diffusion families' records attend both ways and reserve a
    # mask id to denoise; a wrapper's record nests its decoder's.
    if record.get("causal") is False and record.get("mask_token_id") is not None:
        return bundle(PretrainedMaskedDecoder)
    return bundle(PretrainedDecoder)
