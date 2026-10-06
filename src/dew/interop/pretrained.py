"""Load native models and their host processors from a Hugging Face source.

`Pretrained.load` is the front door: it reads a source directory or repo,
translates its config and weights through `dew.interop.hf_decoders`, and
returns the kind of `Pretrained` the source is, holding the model, its
variables and its processor.
`Pretrained` also carries the source's own decoding controls and the layouts
that write every tensor back, so `Pretrained.save` restores what it read.
A latent diffusion source is assembled by `dew.interop.diffusion_pipelines`,
from the components `dew.interop.diffusion_components` builds.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import DTypeLike

from dew import records
from dew._model_types import _QWEN35_TEXT_TYPES, _QWEN35_TYPES
from dew.coordination import agreed
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import SourceSchedule
from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
from dew.inference.pipeline import place
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import InputSpec
from dew.interop import gguf, hf_decoders as decoders, mamba2, sources, verify
from dew.interop.codecs import SourceQuantization, source_quantization
from dew.interop.config_records import NativeFields
from dew.interop.decoder_config import _kinds_of
from dew.interop.decoder_export import _export_config, _layout_tensors, _refuse_lossy_export
from dew.interop.decoder_family import _check_tree
from dew.interop.diffusion_components import _present
from dew.interop.diffusion_pipelines import (
    SourceTask as SourceTask,
    _load_diffusion_source,
    load_diffusion_conditioner as load_diffusion_conditioner,
    load_diffusion_source as load_diffusion_source,
)
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
from dew.interop.streaming import SourceLeaf, WeightLayout
from dew.interop.weights import ParamTree, auto_storage_dtype
from dew.nn import audio as audio_nn
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer
from dew.objectives.base import Variables
from dew.registry import (
    dtype_name,
    from_record,
    precision_fields,
    projectors,
    resolve_dtype,
    towers,
    with_precision,
)
from dew.sampling.pipelines import TextToImage
from dew.sampling.text import Sampling

if TYPE_CHECKING:

    from dew.lora import Adapter, LoRA
    from dew.training.distributed import Layout, MeshSpec



def _stacked_expert(path: tuple[str, ...]) -> tuple[tuple[str, ...], int | None]:
    """Return the leaf `decoder_paths._stack_experts` stacked a per-expert
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

    tower_kind = record["tower"]["name"]
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
            path = projector_weight_path(record["projector"]["name"], local)
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
            path = projector_weight_path(record["audio_projector"]["name"], local)
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
    export_adapter: (
        Callable[[nn.Module, Mapping[str, object], Mapping[str, object]], Mapping[str, np.ndarray]] | None
    ) = field(default=None, repr=False)
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
                 ema: bool | None = None) -> Self:
        """Return a trained run's selected checkpoint, rebuilt from the run's own inference record."""
        from dew.config import ModelConfig
        from dew.inference.tasks import run_record
        from dew.records import record, text
        from dew.registry import objectives
        declaration = run_record(str(directory), step)
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
        if isinstance(model, CausalTransformer):
            bundle = PretrainedDecoder.from_model(model, variables, tokenizer=tokenizer,
                                                   generation_config=generation)
            if kind == 'masked_diffusion':
                bundle = PretrainedMaskedDecoder(
                    model, bundle.variables, bundle.processor, bundle.config, bundle.source,
                    bundle.model_config, bundle.generation_config,
                    export_adapter=bundle.export_adapter, tokenizer=bundle.tokenizer)
        elif isinstance(model, DiffusionGemma):
            from dew.interop import diffusion_gemma
            config = diffusion_gemma.published_config(model)
            bundle = PretrainedBlockDecoder(model, variables, None, config, None, model_config.fields(), {},
                                             export_adapter=diffusion_gemma.export_weights,
                                             tokenizer=tokenizer)
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
        family = self.config.get("model_type")
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif (
            isinstance(self.model, CausalTransformer)
            and isinstance(family, str)
            and family in decoders.families()
            and not decoders.families()[family].preserve_source_layout
            and quantization is None
        ):
            # The decoder export's own encoder, so this and `PretrainedDecoder.from_model`
            # leave the same weights. A quantized source keeps its packed format
            # by going back over its source names, below.
            return decoders.export_decoder_weights(self.model, values, _export_config(self.model))
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            text = self.model.language_model if isinstance(self.model, MultimodalTransformer) else self.model
            scalar_mode = text.layer_scalar if isinstance(text, CausalTransformer) else None
            layouts = {layout.name: layout for layout in self.weight_layouts}
            if quantization is None:
                return _layout_tensors(layouts, values, scalar_mode, self.retained_tensors)
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
    def from_model(cls, model: CausalTransformer, variables: Variables, *,
                   tokenizer: str | decoders.ExportTokenizer | None = None,
                   generation_config: Mapping[str, object] | None = None) -> PretrainedDecoder:
        """Wrap a decoder trained in Dew as a bundle that `save` writes in its family's Hugging Face layout.

        The config is derived from the native computation, and every variable
        collection is encoded through the matching family. That is the same encoder a
        loaded source of the derived family exports through, so both write the same
        weights. Gemma 4 writes frozen or trainable layer-scalar values into HF
        buffers; reloading that layout reproduces the computation, but not which
        scalars were trainable.

        `tokenizer` is the vocabulary the weights were trained with, as an object or
        by name. `save` writes its files next to the weights, so the directory that
        `Pretrained.load` reads back includes its processor. `generation_config` is
        what generation_config.json records, `GENERATION_DEFAULTS` when None.
        """
        config = _export_config(model)
        _refuse_lossy_export(model, config)
        built = {entry.name: getattr(model, entry.name) for entry in dataclasses.fields(model)
                 if entry.init and entry.name not in ("parent", "name")}
        return cls(model, variables, None, config, None, built,
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
        text_fields["mixer"] = {"name": "attention", "fields": {"bidirectional_images": True}}
    if family == "gemma4" and text_config.get("use_bidirectional_attention") == "vision":
        kinds = _kinds_of(text_fields).copy()
        sliding = NativeFields(LayerKind, {
            **kinds.get("sliding_attention", {}),
            "mixer": {"name": "attention", "fields": {"bidirectional_images": True}}})
        kinds["sliding_attention"] = sliding
        text_fields["kinds"] = kinds
    if family in _QWEN35_TYPES:
        rope = records.record(text_config.get("rope_parameters") or {}, "rope_parameters")
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(type(value) is not int or value < 0 for value in sections)):
            raise ValueError("mrope_section must contain three nonnegative integer widths")
        kinds = _kinds_of(text_fields).copy()
        full = NativeFields(LayerKind, {
            **kinds.get("full_attention", {}),
            "mixer": {"name": "attention", "fields": {
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
                      model: nn.Module, gguf_path: Path | None = None) -> Processor | None:
    """Build the host preprocessing a source ships, or None where it ships none.

    Only a model with towers reads images or audio, and only through the
    processor its repo ships; a text decoder takes its tokenizer whatever
    processor files sit beside it (DiffusionGemma publishes a Gemma 4
    processor config beside a text-only model), and a tiny multimodal
    fixture without processor files tokenizes text only.
    """
    processor_files = any((directory / name).exists()
                          for name in ("processor_config.json", "preprocessor_config.json"))
    if isinstance(model, MultimodalTransformer) and processor_files:
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
            storage = auto_storage_dtype({}, diffusion.component_tensors(directory, denoiser))
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
    text = NativeFields(CausalTransformer, {**text_fields, **precision_fields(
        "causal_transformer", text_fields, dtype=dtype, attention_impl=attention_impl)})
    wrapper: decoders.WrapperFields = {**record, "text": text}
    language_model = from_record(CausalTransformer, wrapper["text"])
    model = _wrapper_model(config, record, language_model, dtype=dtype)
    parts = decoders.translate_wrapper_weights(tensors, record, param_dtype=param_dtype, lazy=lazy)
    variables = _native_variables({**parts, "language_model": decoders.with_constants(
        parts["language_model"], record["text"], directory)})
    layouts, retained = _wrapper_layouts(tensors, record, variables)
    return _Built(model, variables, record, wrapper, layouts, retained)


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
    built = with_precision("causal_transformer", record, dtype=dtype, attention_impl=attention_impl)
    model = from_record(CausalTransformer, built)
    variables = decoders.with_constants(decoders.translate_weights(
        tensors, record, family, param_dtype=param_dtype, lazy=lazy), record, directory)
    _check_tree(variables, model)
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
        param_dtype = auto_storage_dtype(config, tensors)
    tensors, quantized_tensors, scale_dtype, grid = _decoded(tensors, config, param_dtype)
    export_adapter = None
    if family == "diffusion_gemma":
        from dew.interop import diffusion_gemma

        model = diffusion_gemma.build(
            config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len
        )
        variables = diffusion_gemma.translate_weights(tensors, config, param_dtype=param_dtype)
        record, layouts, retained = config, (), {}
        built: Mapping[str, object] = {**config, "dtype": dtype, "attention_impl": attention_impl}
        export_adapter = diffusion_gemma.export_weights
    elif "text_config" in config and (family not in decoders.families() or decoders._bundles(config)):
        # A wrapper repo carries its decoder under text_config. Where its
        # model_type is a registered decoder family, its towers have no
        # counterpart and the text half, read from the nested config, is the
        # model, unless the family reads its media bundle whole
        # (`DecoderFamily.wrapper`); `translate_config` refuses the rest.
        model, variables, record, built, layouts, retained = _wrapper_source(
            config, tensors, directory, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len,
            param_dtype=param_dtype, lazy=streaming)
    else:
        model, variables, record, built, layouts, retained = _decoder_source(
            config, tensors, directory, verified, dtype=dtype, attention_impl=attention_impl,
            max_seq_len=max_seq_len, param_dtype=param_dtype, lazy=streaming)
    model = _input_quantization(model, layouts, config, grid)
    processor = _source_processor(directory, config, record, model, gguf_path)
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

    if isinstance(model, DiffusionGemma):
        return bundle(PretrainedBlockDecoder)
    if isinstance(model, CausalTransformer) and not model.causal and model.mask_token_id is not None:
        return bundle(PretrainedMaskedDecoder)
    return bundle(PretrainedDecoder)
