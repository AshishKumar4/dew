"""Load native models and their host processors from a Hugging Face source.

`Pretrained.load` is the front door: it reads a source directory or repo,
translates its config and weights through `dew.interop.hf_decoders`, and
returns the kind of `Pretrained` the source is, holding the model, its
variables and its processor.
`Pretrained` also carries the source's own decoding controls and the layouts
that write every tensor back, so `Pretrained.save` restores what it read.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, NoReturn, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import unflatten_dict
from jax.typing import DTypeLike

from dew import _records
from dew._model_types import QWEN35_TYPES
from dew.batch import Variables
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import SourceSchedule
from dew.inference import BlockGeneration, MaskedGeneration, TextGeneration
from dew.inference.images import TextToImage
from dew.inference.loading import place
from dew.inference.tasks import Processor as TaskProcessor
from dew.inputs import InputSpec
from dew.interop import decoder_parts, gguf, hf_decoders as decoders, mamba2, sources, verify
from dew.interop.codecs import SourceQuantization, source_quantization
from dew.interop.composite import CompositeSource, NativeComponent
from dew.interop.config_records import NativeFields
from dew.interop.decoder_bindings import decoder_layouts, reading_layout
from dew.interop.generation_config import (
    audit_masked,
    eos_ids,
    generation_config_of,
    generation_limit,
    masked_generation_config,
    pad_id,
    read_generation_config,
    return_sequences,
    source_decoding,
)
from dew.interop.pipeline_assembly import (
    SourceTask,
    assemble_pipeline,
    component_present,
    native_variables,
    pipeline_denoiser,
    text_components,
)
from dew.interop.processors import Processor, hosts
from dew.interop.safetensors_io import MAX_SHARD_SIZE, LazyTensors, tensor_specs
from dew.interop.streaming import WeightLayout
from dew.interop.weights import AUTO, checkpoint_dtype, declared_dtype
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import layer_scalar_mode
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.multimodal import MultimodalTransformer
from dew.objectives.base import SavedTask
from dew.registry import dtype_name, from_record, projectors, record_fields, resolve_dtype, towers
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from dew.distributed.placement import Layout, MeshSpec
    from dew.inference.tasks import SavedRun
    from dew.interop.torchax_fallback import TorchAdmission
    from dew.lora import Adapter, LoRA
    from dew.nn.cross_attention import CrossAttention



def _in_language_model(path: tuple[str, ...]) -> tuple[str, ...]:
    """Where a media wrapper holds its decoder's leaf: under `language_model` in the leaf's collection."""
    return (path[0], "language_model", *path[1:])


def _wrapper_layouts(tensors, record, variables, language: Sequence[decoder_parts.Reading],
                     media: tuple[WeightLayout, ...]):
    """Return one `WeightLayout` per wrapper source tensor, and the tensors no layout reads.

    Each layout keeps the source's own tensor name and points at the leaf path
    the loader built, so an export writes the names the source shipped. The
    language model's tensors bind through the readings its load placed
    (`language`, by its local names), an alias of a local name through the
    same one; the media components' through the layouts their readers placed
    them by (`media`). A media tensor no component computes with (SigLIP's
    pooling head, auxiliary tensors the reference ignores) keeps its source
    bytes for export.
    """
    readings = {reading.name: reading for reading in language}
    # The decoder's own tree, each collection's `language_model` entry, which its readings index.
    language_tree = {collection: tree["language_model"] for collection, tree in variables.items()
                     if isinstance(tree, Mapping) and "language_model" in tree}
    bindings = list(media)
    placed = {layout.name for layout in media}
    retained = {}
    for name, tensor in tensors.items():
        group, local = decoders.wrapper_route(name, record)
        if group == "language_model":
            layout = reading_layout(name, readings[local]._replace(stored=tensor), record["text"],
                                    record["text_model_type"], language_tree, _in_language_model)
            if layout is None:
                retained[name] = tensor
            else:
                bindings.append(layout)
        elif name not in placed:
            retained[name] = tensor
    return tuple(bindings), retained


def _share_quantized_aliases(tensors: dict[str, np.ndarray], aliases: tuple[tuple[str, str], ...],
                             quantized: tuple[str, ...]) -> None:
    """Share only aliases verified on original values before codec narrowing.

    A component may mix quantized and unquantized copies, or several MTP
    copies of a tied head. Each group of spellings the checked pairs join
    (`alias_groups`) reuses its quantized member's decoded storage: no
    second cast or weight copy.
    """
    for group in decoders.alias_groups(aliases):
        representative = next((name for name in quantized if name in group), None)
        if representative is not None:
            value = tensors[representative]
            for name in group:
                tensors[name] = value


def on_default_device(variables: Variables) -> Variables:
    """`variables` with each host leaf on JAX's default device as a `jnp` array is, uncommitted, so a
    jitted call reads it where it is and a placement moves it, one leaf at a time: a leaf's transfer
    completes before the next starts, and its host copy is released (a memory-mapped one's pages
    evicted, `dew.distributed.host.evict`) once its device copy is written back in its place."""
    from dew.distributed.host import placed_leaves

    # A freshly loaded tree's nodes are its own plain dicts, written in place so that no host leaf
    # outlives its device copy.
    return placed_leaves(variables, lambda _, leaf: jax.device_put(leaf)
                         if isinstance(leaf, (np.ndarray, np.generic)) else leaf, in_place=True)


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
    """The source tensors no forward reads, kept as stored for `save` to write back.

    A pipeline keeps a text encoder's vision tower here, each tensor named
    under its component as its layouts are.
    """
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
    origin: str | None = None
    """The Hub id the load resolved to `revision`, which an adapter's PEFT config names as its base
    (`AdapterBase`); None for a local directory, a payload a selector chose or a model trained in
    Dew, which a reader of an id alone does not load (`_published`)."""
    adapter: Adapter | None = None
    """The low-rank adapter that `adapt` attached to the model, or None for the source as published.

    The variables hold its factors under `params`, next to the base weights under
    `frozen`.
    """
    tokenizer: str | decoders.ExportTokenizer | None = None
    """The vocabulary `save` writes next to the weights, by name or as an object.

    It is used by a bundle that has no source processor to write (`from_model`).
    """
    code: Path | None = None
    """The directory of the source's own Python modules a load ran (`trust_remote_code`), which `save`
    writes beside the weights: the source's snapshot, which a weights conversion may not be."""
    names_tokenizer: ClassVar[bool] = True
    """Whether generation_config.json may record the tokenizer's name.

    It is False for a family whose reader treats that file as a closed set of
    fields.
    """


    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """The end-of-sequence ids the source's generation stops at, from its generation config or
        its config (`eos_token_id`), none where neither names one."""
        return eos_ids(self.config, self.generation_config)
    def map_variables(self, transform: Callable[[Variables], Variables]) -> Self:
        """This bundle with its variable tree transformed, for placement or a shape-only load.
        Composite bundles apply the same transformation to each component."""
        return replace(self, variables=transform(self.variables))

    def pipeline(self) -> SavedTask:
        """The native task declared by this source kind."""
        raise TypeError(f"{type(self).__name__} has no native task")

    def packed(self) -> Self:
        """A serving view with packed projections, preserving this bundle's arrays for reuse."""
        from dew.inference.projections import inference_projections

        return replace(self, variables=inference_projections(self.model, self.variables))

    @property
    def text_processor(self) -> TaskProcessor | None:
        """Return what the bundle's tasks encode and decode text with.

        That is the source's processor, or a run processor over the vocabulary that
        `tokenizer` names (a run's export, including Dew's byte vocabulary).
        """
        if self.processor is not None or not isinstance(self.tokenizer, str):
            return self.processor
        from dew.data.text import tokenizer_for
        from dew.inference.loading import RunProcessor
        return RunProcessor(tokenizer_for(self.tokenizer, local_files_only=True))

    @classmethod
    def from_run(cls, directory: str | Path, *, step: int | str | None = None,
                 ema: bool | None = None, trust: Sequence[str] = ()) -> Self:
        """Return a trained run's selected checkpoint, its latest or `step`, rebuilt from the run's
        own inference record (`from_saved`). `trust` names the packages outside Dew the record may
        import (`TextGeneration.from_run`)."""
        from dew.inference.tasks import SavedRun

        return cls.from_saved(SavedRun.pinned(str(directory), step, trust), ema=ema)

    @classmethod
    def from_saved(cls, run: SavedRun, *, ema: bool | None = None) -> Self:
        """Return the pinned checkpoint `run` (`SavedRun.pinned`), rebuilt from the run's own inference
        record, in the layout of the architecture the record names."""
        from dew.inference.tasks import (
            BlockSettings,
            MaskedSettings,
            TaskSettings,
            TextSettings,
            saved_model,
            saved_settings,
        )
        from dew.registry import models

        model_config = saved_model(run.record, None)
        model = model_config.build()
        variables = run.variables(ema=ema)
        # The run's task as its loader reads it (`saved_settings`): its policy, budget, pad and
        # end-of-sequence ids and process, which the export writes in the format its readers read.
        held = None if run.objective.settings is None else saved_settings(run.record, run.objective.settings)
        task = held if isinstance(held, TaskSettings) else None
        budget = None if task is None else task.max_new_tokens
        tokenizer = None if task is None else task.tokenizer
        # A run over a source's own processor records its source, and the bundle writes its files.
        processor = None if task is None or task.processor is None else load_processor(task.processor, model)
        trained = models[model_config.name]
        if trained is CausalTransformer:
            decoder = from_record(CausalTransformer, model)
            generation = (masked_generation_config(budget, None, held.eos_token_ids)
                          if isinstance(held, MaskedSettings) else
                          generation_config_of(held.sampling, budget) if isinstance(held, TextSettings)
                          else None)
            bundle = replace(PretrainedDecoder.from_model(decoder, variables, tokenizer=tokenizer,
                                                          generation_config=generation), processor=processor)
            if isinstance(held, MaskedSettings):
                bundle = PretrainedMaskedDecoder(
                    decoder, bundle.variables, bundle.processor, bundle.config, bundle.source,
                    bundle.model_config, bundle.generation_config,
                    export_adapter=bundle.export_adapter, tokenizer=bundle.tokenizer)
        elif trained is models["diffusion_gemma"]:
            from dew.interop import diffusion_gemma
            if not isinstance(held, BlockSettings):
                raise TypeError(f"a {model_config.name} run of {run.objective.__name__} records no block "
                                "process to export")
            block = from_record(DiffusionGemma, model)
            published = diffusion_gemma.published_config(block)
            generation = diffusion_gemma.generation_config_of(held.process, max_new_tokens=budget,
                                                              pad_token_id=held.pad_token_id,
                                                              eos_token_ids=held.eos_token_ids)
            bundle = PretrainedBlockDecoder(
                block, variables, processor, published, None, model_config.fields, generation,
                export_adapter=diffusion_gemma.export_weights, tokenizer=tokenizer)
        else:
            raise TypeError(f"{type(model).__name__} has no maintained exported bundle layout; "
                            "load diffusion runs with TextToImage.from_run")
        if not isinstance(bundle, cls):
            raise TypeError(f"{run.directory} is a {type(bundle).__name__} source, not a {cls.__name__}; "
                            f"load it with {type(bundle).__name__}.from_run or Pretrained.from_run")
        return bundle.bound(model, variables)

    def bound(self, model: nn.Module, variables: Variables) -> Self:
        """This bundle carrying the LoRA `model`'s record binds over `variables`, as
        `Adapter.from_run` rebuilds it, so `export` and `save` merge its factors into
        their kernels and `adapter.save` writes them alone; the bundle itself where
        `model` carries none."""
        from dew.lora import Adapter, recorded_adapter
        from dew.registry import ModelConfig

        record = recorded_adapter(ModelConfig.from_model(model).wrappers)
        return self if record is None else replace(self, adapter=Adapter.recorded(model, variables, record))

    @classmethod
    def load(cls, name_or_dir: str | Path, *, dtype: DTypeLike = jnp.bfloat16,
             param_dtype: DTypeLike | Literal["auto"] = jnp.float32,
             attention_impl: str = "auto", max_seq_len: int | None = None,
             revision: str | None = None, subfolder: str | None = None, gguf_file: str | None = None,
             single_file: str | None = None, dduf_file: str | None = None,
             mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
             fallback: str | None = None, weights: bool = True, on_host: bool = False,
             modeling_overrides: Mapping[str, object] | None = None,
             trust_remote_code: bool = False, text_only: bool = False,
             prediction_depths: int | None = None) -> Self:
        """Load a source as a native Flax model with explicit parameter trees, as the kind of source it is.

        `name_or_dir` is a local HF directory or a Hub model ID. Decoder, tower and
        projector weights keep their usual internal paths, and a wrapper's variables
        join under its existing component names. Processor files are loaded only when
        the source has them.

        `dtype` sets the compute dtype, and `param_dtype` separately sets the storage
        dtype of floating parameters. It defaults to FP32 master weights, and 'auto'
        keeps each tensor in the dtype the checkpoint stores it in; a quantized
        source's packed weights decode to the dtype it declares (`declared_dtype`).
        Each is a dtype (`jnp.bfloat16`) or its name, and the model's record keeps
        the name. Frozen components (text encoders and the VAE) follow `param_dtype`
        too, while router, clipping, positional and safety state keep their own FP32
        or integer dtypes.

        A loaded source is ready to run: without `mesh` or `layout` its variables are
        built on the host and then placed on JAX's default device (`jax.default_device`)
        one leaf at a time, uncommitted, as a `jnp` array is, so a call or a placement
        over other devices still takes them. With either, they are placed on that mesh
        (the default `MeshSpec()` when only `layout` is given) under that layout.
        `on_host=True` keeps them as host arrays instead, for a caller that places them
        itself, as a training start or an export does; a host tree passed to a jitted
        call is copied to the device at each call. With a mesh or layout, a decoder's or
        a diffusion pipeline's leaves are read from the memory-mapped checkpoint one
        device shard at a time, and cast and transposed there (`dew.interop.streaming`),
        so the host never holds the whole translated model. Towers, projectors, a quantized source's
        dequantized tensors and a pipeline's convolution kernels are still built
        whole on the host first.

        `gguf_file` names a GGUF file in the repo or directory. Its metadata is the
        config, its block-quantized tensors are dequantized to float32
        (`dew.interop.gguf`), and its tokenizer is the processor when the repo ships
        no tokenizer.

        `single_file` selects an original diffusion checkpoint, converted through
        Diffusers' key maps (`dew.interop.single_file`); `dduf_file` selects a packed
        diffusion pipeline (`dew.interop.dduf`). Both load through the ordinary
        directory pipeline after conversion. `fallback="torchax"` selects the
        explicit transformers-forward route (`dew.interop.torchax_fallback`).
        `subfolder` selects a folder within the resolved snapshot; decision releases
        are assembled from their declared backbone and head components.

        `trust_remote_code=True` lets tier 2 run the repository's own modeling code (its
        `auto_map`) for a model_type transformers has no class for (`dew.interop.verify`), and `save`
        writes it beside the weights (`code`).

        Called on a specific kind (`PretrainedDecoder.load`), it refuses a source of
        another kind and names that kind.

        `text_only=True` loads a multimodal wrapper's language model alone, as its text family's own
        checkpoint states it (Qwen3.5's `Qwen3_5ForCausalLM`): no tower, projector or audio encoder is
        built, nor their tensors read, and the source saves as that text checkpoint. A source that is
        a decoder already loads as it is; a pipeline or a fallback has no language model to take.

        `prediction_depths` keeps the first that many of a decoder's multi-token prediction depths
        (DeepSeek V3's, GLM's and Qwen3.5's `mtp` blocks): None the ones the checkpoint holds, 0
        none, `n` the first `n`; a DSpark drafter, whose stages draft as one, is kept whole or at 0
        not at all. The load reads the source as a checkpoint of those
        alone (`_kept_depths`): the others are neither built nor their tensors read, and it saves as
        that checkpoint. A source that holds fewer keeps what it holds; one with none holds none.

        Without `weights`, the variables are their shapes and dtypes alone
        (`jax.ShapeDtypeStruct`), read from the source's headers through the same
        translation: what a run that resumes from its own checkpoint builds its state
        around (`TrainerConfig.resumes`), as MaxText loads its base parameters only
        for a run with no checkpoint.

        `modeling_overrides` is a record of the form a release's config states under
        that key, read through the same reader over the release's own, kind by kind:
        `{"attention": {"full_attention": "block_bidirectional"}}` lets that layer kind's
        tokens read their whole block (`AttentionMixer.bidirectional_blocks`, a call's
        `block_groups`) on any decoder or wrapper this loads natively, and "causal"
        turns a release's back off. A saved model states it in its own config.
        """
        selected = _selected_payload(fallback, gguf_file=gguf_file, single_file=single_file,
                                     dduf_file=dduf_file)
        if on_host and (mesh is not None or layout is not None):
            raise ValueError("on_host keeps the variables as host arrays, and mesh and layout place them; "
                             "pass one")
        streaming = mesh is not None or layout is not None
        if streaming and not weights:
            raise ValueError("a source loaded without its weights has nothing to place; the restore that "
                             "fills it places it, so pass no mesh or layout")

        def placed(variables: Variables) -> Variables:
            return place(variables, mesh, layout) if streaming else variables

        dtype = dtype_name(dtype)
        param_dtype = AUTO if param_dtype == AUTO else dtype_name(param_dtype)
        if prediction_depths is not None and (isinstance(prediction_depths, bool) or prediction_depths < 0):
            raise ValueError(f"prediction_depths keeps the first that many depths, 0 or more, not "
                             f"{prediction_depths!r}")
        # The reference refuses a folder outside the source before anything is fetched.
        directory, reference, composite = _source_metadata(
            sources.SourceReference(str(name_or_dir), revision, subfolder), selected, fallback,
            modeling_overrides)
        loaded, extracted = _loaded_kind(
            reference, directory, composite, fallback=fallback, gguf_file=gguf_file,
            single_file=single_file, dduf_file=dduf_file, placed=placed, streaming=streaming or not weights,
            weights=weights, dtype=dtype, param_dtype=param_dtype, attention_impl=attention_impl,
            max_seq_len=max_seq_len, overrides=modeling_overrides, trust_remote_code=trust_remote_code,
            text_only=text_only, prediction_depths=prediction_depths)
        if not isinstance(loaded, cls):
            raise TypeError(f"{name_or_dir} is a {type(loaded).__name__} source, not a {cls.__name__}; "
                            f"load it with {type(loaded).__name__}.load or Pretrained.load")
        published = _published(loaded, name_or_dir,
                               selected=bool(selected) or reference.subfolder is not None or extracted)
        return _resident(published, weights=weights, streaming=streaming, on_host=on_host)

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
        from dew.lora import AdapterBase

        adapter = lora.apply(self.model, self.variables, key=key, layouts=self._adaptable,
                             base=None if self.origin is None else AdapterBase(self.origin, self.revision))
        return replace(self, model=adapter.model, variables=adapter.variables, adapter=adapter)

    def with_cross_attention(self, cross: CrossAttention, *, key: int | jax.Array,
                             frozen: bool = True) -> Self:
        """Return this source with gated cross-attention before the layers `cross` names, each
        block drawn from `key` (`dew.nn.cross_attention`).

        The blocks' gates start at zero, so the model computes what the source
        does until they train. The variables hold the blocks under `params` and,
        where `frozen`, the source's weights under `FROZEN`, so an objective
        built over the result trains the blocks alone; an adapter's factors that
        already train keep training, and `adapt` keeps the blocks training, so
        the two compose in either order. With `frozen=False` the source's own
        split stands: an unsplit source's every weight trains beside the
        blocks, a full fine-tune. The source's own format holds no
        cross-attention, so `save` and `export` refuse the result; a run's
        checkpoint keeps it.
        """
        from dew.batch import freeze, merge, select, thaw
        from dew.nn.cross_attention import crossing
        from dew.nn.inputs import request_key
        from dew.nn.protocols import CrossAttending, Predicting

        if not isinstance(self.model, CrossAttending):
            raise TypeError(f"a {type(self.model).__name__} takes no cross-attention layers")
        if self.model.cross_attention is not None:
            raise ValueError("this bundle already reads a context; attach cross-attention to the source")
        model = self.model.with_cross_attention(cross)
        # A token more than the prediction depths a source carries, which each read one further on.
        length = 1 + (model.num_nextn_predict_layers if isinstance(model, Predicting) else 0)

        def drawn(draw: jax.Array) -> Variables:
            # Only the blocks leave the program, so the source's own draws are dead code to XLA.
            return select(thaw(model.init(draw, jnp.zeros((1, length), jnp.int32))),
                          lambda path: path[0] == "params" and crossing(path))

        variables = merge(self.variables, jax.jit(drawn)(request_key(key)))
        return replace(self, model=model, variables=freeze(variables, crossing) if frozen else variables)

    def export(self, variables: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
        """Return the tensors `save` writes, by their source names; `dew.inference.NCCLPush` sends these.

        An adapted bundle (`lora`) writes the source's own tensors with the factors
        merged into their kernels, as PEFT's `merge_and_unload` does, from its
        variables or a trainer's split of them; `adapter.save` writes the factors
        alone. A loaded task's tree, its projections packed for serving, is written
        as the source's own members (`dew.inference.projections.unpacked`).
        """
        from dew.inference.projections import unpacked
        from dew.nn.protocols import CrossAttending

        if isinstance(self.model, CrossAttending) and self.model.cross_attention is not None:
            raise ValueError("the source's format holds no cross-attention layers; a run's checkpoint "
                             "keeps this model")
        values = unpacked(self.model, self.variables if variables is None else variables)
        if self.adapter is not None:
            values = self.adapter.merge(values)
        quantization = self._quantization()
        if self.export_adapter is not None:
            tensors = self.export_adapter(self.model, values, self.config)
        elif self.weight_layouts:
            # Source names and geometry first; the packed format goes back over them.
            scalars = [layout.paths[0][1:] for layout in self.weight_layouts
                       if layout.paths[0][-1] == "layer_scalar"]
            scalar_mode = layer_scalar_mode(values, scalars[0]) if scalars else None
            layouts = {layout.name: layout for layout in self.weight_layouts}
            if quantization is None:
                return decoder_parts.layout_tensors(layouts, values, scalar_mode, self.retained_tensors)
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
        """Write `export`'s tensors, the source's own config.json as published, and its tokenizer files,
        and the source's own Python modules a load ran (`code`), so the directory loads as the source
        did.

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
        if self.code is not None:
            for module in sorted(self.code.rglob("*.py")):
                written = destination / module.relative_to(self.code)
                written.parent.mkdir(parents=True, exist_ok=True)
                written.write_bytes(module.read_bytes())
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

    def pipeline(self):
        return self.packed().text_generation()

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
        fields, so any other model is refused. A decoder with cross-attention takes the
        config of the decoder its blocks join and runs as a bundle, but `save` refuses it.

        `tokenizer` is the vocabulary the weights were trained with, as an object or
        by name. `save` writes its files next to the weights, so the directory that
        `Pretrained.load` reads back includes its processor. `generation_config` is
        what generation_config.json records; None uses the native defaults with
        the exported model's cache mode.
        """
        try:
            decoder = from_record(CausalTransformer, model)
        except ValueError as error:
            raise TypeError(f"{type(model).__name__} has no Hugging Face decoder layout") from error
        # Cross-attention blocks have no layout in any family: the config is the decoder's they join,
        # and `export` refuses the blocks themselves.
        config = decoders.export_config(decoder.with_cross_attention(None))
        built = record_fields(decoder, CausalTransformer)
        return cls(decoder, variables, None, config, None, built,
                   {**decoders.GENERATION_DEFAULTS, 'use_cache': config.get('use_cache', decoder.causal)}
                   if generation_config is None else generation_config,
                   export_adapter=decoders.export_decoder_weights, tokenizer=tokenizer)

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration:
        """Build the text generation task this source describes.

        Without an override, the task runs the source's policy (`task.sampling` holds
        every common control its config sets), any transform chain the rarer controls
        need, and the strategy its config names. An explicit `sampling` replaces the
        policy and drops that chain, because the chain was built for the policy the
        caller replaced; the source's EOS and pad ids fill the ones it leaves None,
        and `num_return_sequences` still comes from the source.

        The task shares the bundle's weights and packs its decoder's projections inside
        each generation's compiled call; `dew.pipeline`'s task packs them once, after
        placing them. An adapted model whose factors are not merged keeps its
        projection paths and is not packed.
        """
        return _text_generation(self, sampling)



@dataclass(frozen=True)
class PretrainedMaskedDecoder(Pretrained):
    """A masked-diffusion decoder (LLaDA, Dream).

    It is a non-causal decoder with a mask id that generates by unmasking a whole
    response.
    """

    model: CausalTransformer

    def pipeline(self):
        return self.packed().text_generation()

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
                                eos_token_ids=self.eos_token_ids,
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

    def pipeline(self):
        return self.packed().block_generation()

    def block_generation(self) -> BlockGeneration:
        """Build the DiffusionGemma canvas task, with the source's sampler config as its default."""
        from dew.interop import diffusion_gemma

        return BlockGeneration(
            self.model,
            self.variables,
            diffusion_gemma.generation_process(self.config, self.generation_config),
            self.text_processor,
            self.eos_token_ids,
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

    def pipeline(self):
        return self.text_to_image()

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
        """Write each component's tensors and config in the diffusers layout,
        its weights in shards of at most `max_shard_size` where they exceed it.

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
        diffusion.save_source(self, values, Path(directory), max_shard_size)

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

    It fine-tunes on next-token prediction, without Dew's kernels or sharding rules; train it like
    any source, `LMObjective(fallback, seq_len)`. One whose attention runs on Dew's generates through
    Dew's decode cache (`text_generation`, as a native decoder's, which `dew.pipeline` and the
    server run); one on transformers' own refuses a cache (`TorchCausalLM`).
    """

    def pipeline(self):
        return self.text_generation()

    def text_generation(self, *, sampling: Sampling | None = None) -> TextGeneration:
        """The text generation task this source describes, as a native decoder's
        (`PretrainedDecoder.text_generation`)."""
        return _text_generation(self, sampling)


def _text_generation(loaded: Pretrained, sampling: Sampling | None) -> TextGeneration:
    """The text generation task a causal source describes (`PretrainedDecoder.text_generation`)."""
    rows = return_sequences(loaded.config, loaded.generation_config)
    policy, logits, strategy = source_decoding(loaded.config, loaded.generation_config, loaded.model, rows,
                                               sampling)
    return TextGeneration(
        loaded.model,
        loaded.variables,
        loaded.text_processor,
        policy,
        max_new_tokens=generation_limit(loaded.config, loaded.generation_config, "max_new_tokens"),
        max_length=generation_limit(loaded.config, loaded.generation_config, "max_length"),
        n=rows,
        logits=logits,
        strategy=strategy,
    )


def _no_language_model(name: str, kind: str) -> NoReturn:
    """Refuse `text_only` on a source of `kind`, which has no wrapper's language model to take."""
    raise ValueError(f"{name} is a {kind}, which has no wrapper's language model for text_only to take")


def load_diffusion_source(checkpoint: str, *, dtype: str = "bfloat16", param_dtype: str = "float32",
                          revision: str | None = None, attention_impl: str = "auto",
                          size: tuple[int, ...] | None = None, variables: Variables | None = None,
                          mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
                          text: bool = True, text_only: bool = False,
                          prediction_depths: int | None = None,
                          modeling_overrides: Mapping[str, object] | None = None) -> PretrainedPipeline:
    """A published diffusion pipeline, to train from its own weights.

    `size` is the (height, width) in pixels the pipeline runs at instead of
    its own, or a video pipeline's (frames, height, width): the geometry its
    conditioning, its training shift and its sampling grid are bound to.
    Supplied `variables` are a saved tree of the same pipeline, as a run
    that fine-tuned it wrote them: the modules are built from the
    directory's metadata and bind those variables, and no weight downloads.
    Their layouts come from the source's weight headers, so `save` writes them
    back, with the tensors no leaf holds read from the source as it writes.
    `mesh` and `layout` place the weights it reads as `Pretrained.load`'s
    do, streamed one leaf at a time.

    `text=False` reads and downloads every component's weights but the text
    encoder's, which then need not share the host or the device with the
    denoiser: the conditioner is built from metadata and holds none, a call
    takes the prompts its conditioner encoded alone
    (`TextToImage.prepare(conditions=...)`), and a prompt, training or
    `save`, which read the text encoder, are refused.

    `text_only`, `prediction_depths` and `modeling_overrides` read as
    `Pretrained.load`'s do: a pipeline has no wrapper's language model for
    `text_only` to take, its components hold no prediction depths, so any
    count keeps them all, and none reads overrides, which are refused.
    """
    _unread_overrides(modeling_overrides, "a diffusion pipeline")
    if text_only:
        _no_language_model(checkpoint, "diffusion pipeline")
    directory = sources.snapshot(checkpoint, revision, weights=False)
    if variables is not None and not text:
        raise ValueError("supplied variables bind every component; text=False skips reading one")
    if size is not None and len(size) not in (2, 3):
        raise ValueError(f"size is (height, width) or (frames, height, width), not {size}")
    streaming = variables is None and (mesh is not None or layout is not None)
    loaded = _pipeline_source(
        sources.SourceReference(checkpoint, None if os.path.isdir(checkpoint) else directory.name),
        directory, None, None,
        lambda values: place(values, mesh, layout) if streaming else values, dtype=dtype,
        attention_impl=attention_impl, param_dtype=param_dtype, streaming=streaming, size=size,
        variables=variables, text=text)
    if loaded is None:
        raise ValueError(f"{checkpoint} is not a diffusion pipeline: it has no model_index.json, or its "
                         "config.json names the model it is")
    return _published(loaded, checkpoint, selected=False)


def _published[K: Pretrained](loaded: K, name_or_dir: str | Path, *, selected: bool) -> K:
    """`loaded` naming the Hub id it resolved (`origin`) where a reader on any machine loads the
    same weights from that id at `revision`: a Hub load of the repo's own weights. A local
    directory resolves on no other machine, and a payload a selector chose (`gguf_file`,
    `single_file`, `dduf_file`, a wrapper's language model `text_only` took) is not what a reader
    of the id alone loads, so neither names one."""
    return loaded if loaded.revision is None or selected else replace(loaded, origin=str(name_or_dir))


def _pipeline_weights(source: sources.SourceReference, root: Path, index: Mapping[str, object],
                      skipped: tuple[str, ...] = ()) -> Path:
    """The snapshot with the weights of every component `index` declares but
    `skipped`, fetched at the commit the metadata fetch returned as
    `directory`, so the weights come from that commit even if the requested
    branch moves."""
    return source.directory(source.weights(root, tuple(
        name for name in index if component_present(index, name) and name not in skipped)))


def _assembled_pipeline(directory: Path, index: Mapping[str, object], commit: str | None,
                        placed: Callable[[Variables], Variables], *, streaming: bool, dtype: str,
                        attention_impl: str, param_dtype: str, variables: Variables | None = None,
                        text: bool = True, stored: sources.StoredWeights | None = None) -> PretrainedPipeline:
    """The pipeline `index` describes, assembled from the directory its
    source resolved to (a snapshot, a single file's conversion or a DDUF
    file's unpacking), its variables `placed` and `commit` its revision.
    `streaming` reads the weights as `SourceLeaf`s for `placed` to place one
    leaf at a time; `variables` and `text` are as `assemble_pipeline` takes
    them."""
    loaded = PretrainedPipeline(processor=None, **assemble_pipeline(
        directory, index, dtype=dtype, attention_impl=attention_impl, param_dtype=param_dtype,
        variables=variables, lazy=streaming, text=text, stored=stored))
    return replace(loaded, variables=placed(loaded.variables), revision=commit)


def _wrapper_attention(config: decoder_parts.DecoderFields, names: Sequence[str],
                        fields: Mapping[str, object]) -> None:
    """Compose a wrapper's attention metadata with each kind's source-declared mixer fields."""
    kinds = decoder_parts.kinds_of(config).copy()
    for name in names:
        kind = kinds.get(name, {})
        mixer = _records.record(kind.get("mixer") or config.get("mixer")
                               or {"class": "attention", "fields": {}}, "mixer")
        held = _records.record(mixer.get("fields", {}), "mixer.fields")
        kinds[name] = NativeFields(LayerKind, {
            **kind, "mixer": {**mixer, "fields": {**held, **fields}}})
    config["kinds"] = kinds


def _layer_kinds(config: decoder_parts.DecoderFields) -> tuple[str, ...]:
    """Each layer kind a decoder's `layer_types` names, once, in order."""
    return tuple(dict.fromkeys(_records.strings(config["layer_types"], "layer_types")))


def _wrapper_text_fields(config: Mapping[str, object], record: decoder_parts.WrapperFields,
                         max_seq_len: int | None) -> decoder_parts.DecoderFields:
    """Read the decoder fields a wrapper's text_config states, with the corrections
    its own family makes to them.

    Gemma 3 projects its logits without the causal-LM class's final tanh cap
    and attends its image spans both ways; Gemma 4 does the same on its
    sliding layers when the config asks for it; Gemma 4 Unified reads each
    vision block whole on every attention layer; Qwen3.5 splits its rotary
    into the three mrope sections its wrapper positions rows with.
    """
    family = config.get("model_type")
    text_config = _records.record(config["text_config"], "text_config")
    text_fields = record["text"].copy()
    if max_seq_len is not None:
        text_fields["max_seq_len"] = max_seq_len
    if family == "gemma3":
        text_fields["final_logit_softcap"] = None
        _wrapper_attention(text_fields, _layer_kinds(text_fields), {"bidirectional_images": True})
    if family == "gemma4" and text_config.get("use_bidirectional_attention") == "vision":
        _wrapper_attention(text_fields, ("sliding_attention",), {"bidirectional_images": True})
    # Unified states vision unless its config says otherwise, and reads each vision block whole on
    # every layer kind, past the sliding window.
    if family == "gemma4_unified" and text_config.get("use_bidirectional_attention", "vision") == "vision":
        _wrapper_attention(text_fields, _layer_kinds(text_fields), {"bidirectional_blocks": True})
    if family in QWEN35_TYPES:
        rope = _records.record(text_config.get("rope_parameters") or {}, "rope_parameters")
        sections = rope.get("mrope_section", [11, 11, 10])
        if (not isinstance(sections, (list, tuple)) or len(sections) != 3
                or any(type(value) is not int or value < 0 for value in sections)):
            raise ValueError("mrope_section must contain three nonnegative integer widths")
        _wrapper_attention(text_fields, ("full_attention",), {"mrope_section": list(sections)})
    return text_fields


def _wrapper_model(config: Mapping[str, object], record: decoder_parts.WrapperFields,
                   language_model: CausalTransformer, *, dtype: str) -> MultimodalTransformer:
    """Build the wrapper its record describes: the decoder above, the towers and
    projectors it names, and the placeholder ids its prompts carry."""
    family = _records.text(config["model_type"], "model_type")
    text_config = _records.record(config["text_config"], "text_config")
    audio_record = record["audio"]
    audio_projector = record["audio_projector"]
    return MultimodalTransformer(
        language_model, None if record["tower"] is None else towers.from_record(record["tower"]),
        None if record["projector"] is None else projectors.from_record(record["projector"]), family,
        record["image_token_id"], dtype=resolve_dtype(dtype),
        # A text_config that states a null pad id states none, and the
        # wrapper's own field defaults to 0 for exactly that.
        pad_token_id=_records.integer(text_config.get("pad_token_id") or 0, "pad_token_id"),
        extra_placeholder_ids=record["extra_placeholder_ids"],
        audio=None if audio_record is None else towers.from_record(audio_record),
        audio_projection=(None if audio_projector is None
                          else projectors.from_record(audio_projector)),
        audio_soft_tokens=record["audio_soft_tokens"])


def source_processor(directory: Path, config: Mapping[str, object], record: Mapping[str, object],
                      model: nn.Module, gguf_path: Path | None = None, *, source: sources.SourceReference
                      ) -> Processor | None:
    """Build the host preprocessing a source ships, or None where it ships none.

    Only a media source, a wrapper whose config names towers beside its
    decoder (`_media_wrapper`), reads images or audio, and only through the
    processor its repo ships; a text decoder takes its tokenizer whatever
    processor files sit beside it (DiffusionGemma publishes a Gemma 4
    processor config beside a text-only model), and a tiny multimodal
    fixture without processor files tokenizes text only.

    `source` records the directory or Hub repository, commit and selected folder
    the loader resolved, the identity a run records to reload the processor
    (`load_processor`). A GGUF file's text processor keeps none, since the
    source alone does not name the file it read. Only the media branch reads media
    files from its directory and keeps the wrapper's translated `record`.
    """
    processor_files = any((directory / name).exists()
                          for name in ("processor_config.json", "preprocessor_config.json"))
    media = _media_wrapper(config) and processor_files
    if media or (directory / "tokenizer_config.json").exists() or gguf_path is not None:
        from dew.data.text import load_tokenizer
        # A GGUF repo ships none; the file carries its tokenizer.
        tokenizer = (gguf.tokenizer(gguf_path) if gguf_path is not None
                     and not (directory / "tokenizer_config.json").exists()
                     else load_tokenizer(str(directory), local_files_only=True))
        if not hosts(tokenizer):
            raise TypeError(f"the tokenizer in {directory} lacks a host processor operation")
        return Processor(tokenizer, config, record if media else {}, model.vocab_size,
                         source=source if media or gguf_path is None else None,
                         directory=directory if media else None)
    return None


def load_processor(source: sources.SourceReference, model: nn.Module) -> Processor | None:
    """Reload the recorded processor from its source, beside `model`.

    A saved task keeps the source's directory/repository, commit and subfolder
    as a typed record, not its translated fields. Reading that source's config
    and translating its wrapper derives a media processor's geometry again
    without loading the model's weights. A composite's metadata codec selects
    the native component's config, as it does for the initial load.
    """
    directory = source.metadata()
    description = _composite_description(directory)
    component = None if description is None else description.backbone
    if component is None:
        config, _ = _source_config(source.name, directory, source.revision, None)
    else:
        config = component.config
        directory = directory / component.config_folder
    record = decoders.translate_wrapper_config(config) if _media_wrapper(config) else {}
    return source_processor(directory, config, record, model, source=source)


def _pipeline_source(source: sources.SourceReference, root: Path, single_file: str | None,
                     dduf_file: str | None, placed: Callable[[Variables], Variables], *, dtype: str,
                     attention_impl: str, param_dtype: str, streaming: bool,
                     size: tuple[int, ...] | None = None, variables: Variables | None = None,
                     text: bool = True) -> PretrainedPipeline | None:
    """The source as a latent diffusion pipeline, or None when it is a decoder.

    A single file converts into the pipeline it describes, and a DDUF file
    unpacks into the one it packs; a directory with a model_index.json is
    one unless its own config.json names a model (`_names_a_model`): a
    decoder that also ships a pipeline index for its sampler (DiffusionGemma)
    is loaded as the decoder its config names. `placed` puts the pipeline's variables on
    the mesh, and `streaming` has them read one leaf at a time as it does; a
    single file's conversion is kept only once that succeeds too. `size`,
    supplied `variables` and `text` are a training run's (`load_diffusion_source`):
    the geometry the pipeline runs at, a saved tree a directory's components
    bind instead of downloading their weights, and whether its text encoder
    is read.
    """

    directory, commit = source.directory(root), source.revision

    def pipeline(directory: Path, stored: sources.StoredWeights | None = None) -> PretrainedPipeline:
        index = json.loads((directory / "model_index.json").read_text())
        if size is not None:
            index = {**index, "dew_height": size[-2], "dew_width": size[-1],
                     **({"dew_frames": size[0]} if len(size) == 3 else {})}
        return _assembled_pipeline(directory, index, commit, placed, streaming=streaming, dtype=dtype,
                                   attention_impl=attention_impl, param_dtype=param_dtype,
                                   variables=variables, text=text, stored=stored)

    if dduf_file is not None:
        from dew.interop import dduf
        return pipeline(dduf.unpacked(source.file(root, dduf_file)))
    if single_file is not None:
        from dew.interop import single_file as original
        # A repo or directory that describes the pipeline (model_index.json
        # and its component configs) is the configs, as from_single_file(config=)
        # takes; a Hub repo's are its metadata snapshot, so the weights of a
        # component the file lacks come from the repo at that commit.
        configs = directory if (directory / "model_index.json").is_file() else None
        hub = None if configs is None or commit is None else source
        with original.unpacked(source.file(root, single_file), configs, hub) as (
                converted, published):
            return replace(pipeline(converted), source=published)
    if (directory / "model_index.json").is_file() and not _names_a_model(directory):
        if variables is not None:
            return pipeline(directory, sources.StoredWeights(source.name, root, source.subfolder or ""))
        index = json.loads((directory / "model_index.json").read_text())
        skipped = () if text else text_components(
            pipeline_denoiser(directory, dtype=dtype, attention_impl=attention_impl).text, index)
        return pipeline(_pipeline_weights(source, root, index, skipped))
    return None


def _names_a_model(directory: Path) -> bool:
    """Whether `directory`'s own config.json names a model by its `model_type`."""
    path = directory / "config.json"
    return path.is_file() and "model_type" in _records.record(json.loads(path.read_text()), "config.json")


def _loaded_kind(reference: sources.SourceReference, directory: Path, composite: CompositeSource | None, *,
                 fallback: str | None, gguf_file: str | None, single_file: str | None,
                 dduf_file: str | None, placed: Callable[[Variables], Variables], streaming: bool,
                 weights: bool, dtype: str, param_dtype: str, attention_impl: str,
                 max_seq_len: int | None, overrides: Mapping[str, object] | None,
                 trust_remote_code: bool, text_only: bool,
                 prediction_depths: int | None = None) -> tuple[Pretrained, bool]:
    """The source `Pretrained.load` resolved, loaded as its kind loads: a composite release, the
    transformers forward it asks for, a diffusion pipeline, or a natively translated decoder or
    wrapper (a wrapper's language model alone where `text_only` asks, the first `prediction_depths`
    of its prediction depths where that is given); and whether it is that language model, taken out
    of its wrapper."""
    folder = reference.directory(directory)
    pipeline = (single_file is not None or dduf_file is not None
                or (gguf_file is None and (folder / "model_index.json").is_file()
                    and not _names_a_model(folder)))
    if text_only and fallback is None and (pipeline or composite is not None):
        _no_language_model(reference.name, "diffusion pipeline" if pipeline else "composite release")
    def code() -> Path | None:
        """The snapshot holding the source's own modules, which tier 2 reads where the load opted into
        running them, fetched for a loader that reads them alone: a native decoder's at its folder, a
        composite release's backbone's at its component's."""
        return sources.remote_code(reference.name, directory) if trust_remote_code else None

    if composite is not None:
        loaded = _load_composite_source(reference, directory, composite,
                                        placed=placed, streaming=streaming, dtype=dtype,
                                        param_dtype=param_dtype, attention_impl=attention_impl,
                                        max_seq_len=max_seq_len, weights=weights,
                                        overrides=overrides,
                                        code=None if composite.backbone is None else code(),
                                        prediction_depths=prediction_depths)
    elif fallback is not None:
        from dew.interop import torchax_fallback

        _unread_overrides(overrides, "the transformers forward")
        loaded = torchax_fallback.load(reference, directory, dtype=dtype, param_dtype=param_dtype,
                                       attention_impl=attention_impl, max_seq_len=max_seq_len,
                                       weights=weights)
        loaded = replace(loaded, variables=placed(loaded.variables))
    elif gguf_file is None:
        # A selected GGUF file is a decoder, whatever pipeline index the repo also ships.
        loaded = _pipeline_source(reference, directory, single_file, dduf_file, placed,
                                  dtype=dtype, attention_impl=attention_impl, param_dtype=param_dtype,
                                  streaming=streaming)
        if loaded is not None:
            _unread_overrides(overrides, "a diffusion pipeline")
    else:
        loaded = None
    if loaded is None:
        # Without its weights, a decoder's leaves are translated as streaming translates them, each
        # a view of its stored tensor that nothing reads.
        remote = code()
        return _load_native_source(reference, directory, gguf_file=gguf_file, placed=placed,
                                   streaming=streaming, dtype=dtype, param_dtype=param_dtype,
                                   attention_impl=attention_impl, max_seq_len=max_seq_len, weights=weights,
                                   overrides=overrides,
                                   remote=None if remote is None else reference.directory(remote),
                                   pinned=reference, text_only=text_only, prediction_depths=prediction_depths)
    return loaded, False


def with_overrides(config: Mapping[str, object],
                   overrides: Mapping[str, object] | None) -> Mapping[str, object]:
    """`config` with `overrides`, a `modeling_overrides` record, over the source's own: each attention
    kind it names takes its mode, the others keep the source's. The decoder translation reads and
    checks the result (`decoder_parts.modeling_overrides`), and a composite's description what it
    derives from it (`SourceKind.describe`)."""
    if overrides is None:
        return config
    own = _records.record(config.get("modeling_overrides", {}), "modeling_overrides")
    given = _records.record(overrides, "modeling_overrides")
    merged = {**own, **given}
    if "attention" in own and "attention" in given:
        merged["attention"] = {**_records.record(own["attention"], "modeling_overrides.attention"),
                               **_records.record(given["attention"], "modeling_overrides.attention")}
    return {**config, "modeling_overrides": merged}


def _unread_overrides(overrides: Mapping[str, object] | None, route: str) -> None:
    """Refuse `overrides` on a route whose model no override reaches, rather than load without them."""
    if overrides is not None:
        raise ValueError(f"modeling_overrides reach a model Dew translates; {route} reads none")


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
        config = _records.record(json.load(handle), "config.json")
    return config, None


def _decoded(tensors: Mapping[str, np.ndarray], config: Mapping[str, object], param_dtype: str, *, media: bool
             ) -> tuple[Mapping[str, np.ndarray], tuple[str, ...], str | None, dict[str, np.ndarray],
                        decoders.Checked]:
    """The tensors with a quantized source's weights decoded, and what `save`
    needs to write them back: the quantized names, the scales' dtype and the
    integer formats' grid; and the spellings checked equal before a narrowing
    cast, which the readers then take as equal (`decoders.Checked`). `media`
    is whether the loader reads the source as a media wrapper (`_media_wrapper`)."""
    quantization = source_quantization(config)
    if quantization is None:
        return tensors, (), None, {}, decoders.NOTHING_CHECKED
    quantized_tensors, scale_dtype = quantization.names(tensors), quantization.scale_dtype(tensors)
    grid = {part: tensors[part] for name in quantized_tensors for part in quantization.grid(name)
            if part in tensors}
    aliases: tuple[tuple[str, str], ...] = ()
    if param_dtype != "float32":
        aliases = decoders.validate_source_aliases(
            quantization.tensor_names(tensors), partial(quantization.read, tensors, set(quantized_tensors)),
            config, media=media)
    tensors = quantization.dequantize(tensors, param_dtype=param_dtype)
    _share_quantized_aliases(tensors, aliases, quantized_tensors)
    return tensors, quantized_tensors, scale_dtype, grid, decoders.checked_groups(aliases)


def _language_model_tensors(tensors: Mapping[str, np.ndarray], record: decoder_parts.WrapperFields
                            ) -> Mapping[str, np.ndarray]:
    """A wrapper's language model's tensors under its text family's names
    (`decoders.language_model_names`), each read only when the translation reads it; the media
    towers' are neither among them nor read."""
    names = decoders.language_model_names(tensors, tensors.__getitem__, record)
    specs = tensor_specs(tensors)
    return LazyTensors({local: specs[name] for local, name in names.items()},
                       lambda local: tensors[names[local]])


def _input_quantization(model: nn.Module, variables: Variables, layouts: tuple[WeightLayout, ...],
                        config: Mapping[str, object], grid: Mapping[str, np.ndarray]
                        ) -> tuple[nn.Module, Variables]:
    """Write each stored input scale into the `QUANTIZATION` variables of every Linear its source
    weight binds, and return the model computing under them.

    A source weight split along its outputs (Phi-3's fused `qkv_proj`, `gate_up_proj`) binds one
    Linear a member, each reading the fused projection's one input, so each takes its scale. An
    expert stack's members read their own tokens, and a split along the inputs reads parts of the
    input, which one stored scale does not quantize; those are refused before returning a model
    with the wrong activation forward. Every stored scale must bind a Linear.
    """
    from dew.quantization import INPUT_SCALES, QUANTIZATION, InputKind, InputQuantization, InputQuantizer

    codec = source_quantization(config)
    if codec is None or codec.input_scale_dtype is None:
        return model, variables
    sites: dict[tuple[str, ...], tuple[InputKind, jax.Array]] = {}
    consumed: set[str] = set()
    for layout in layouts:
        part = layout.name.removesuffix('.weight') + codec.input_suffix
        if part not in grid:
            continue
        # A kernel leaf is [inputs, outputs]: a split on its last axis splits the outputs.
        if (any(path[-1] != 'kernel' for path in layout.paths) or layout.expert_index is not None
                or layout.concatenate not in (None, -1)):
            raise ValueError(f"{layout.name} needs its input QDQ per expert or input part, "
                             "which one stored input scale does not quantize")
        consumed.add(part)
        scale = np.asarray(grid[part])
        if scale.dtype != np.float32 or scale.size != 1:
            raise ValueError(f"{part} must be one float32 stored global scale, "
                             f"got {scale.dtype} {scale.shape}")
        value = float(scale.reshape(()))
        quantizer = (InputQuantizer.fp8(value) if codec.input_kind(layout.name) == 'fp8' else
                     InputQuantizer.nvfp4(value, codec.input_scale_dtype == 'float8_e4m3fn',
                                          format=codec.input_format))
        for path in layout.paths:
            site = path[1:-1]
            # A fused tensor and an equal member stored beside it bind one Linear twice: they must
            # quantize its input alike, or one calibration would be silently dropped.
            held_kind, held_scale = sites.setdefault(site, (quantizer.kind, quantizer.scale))
            agrees = held_kind == quantizer.kind and np.array_equal(np.asarray(held_scale),
                                                                    np.asarray(quantizer.scale))
            if not agrees:
                raise ValueError(f"{layout.name}'s input quantizer and another stored for "
                                 f"{'/'.join(site)} disagree; one Linear takes one")
    unbound = {name for name in grid if name.endswith(codec.input_suffix)} - consumed
    if unbound:
        raise ValueError(f"input scales {sorted(unbound)} bind no Linear; an unbound scale would "
                         "drop its QDQ")
    scheme = InputQuantization(codec.input_scale_dtype == 'float8_e4m3fn', codec.input_format,
                               tuple(sorted(('/'.join(path), kind) for path, (kind, _) in sites.items())))
    held = unflatten_dict({(*path, INPUT_SCALES[kind]): scale for path, (kind, scale) in sites.items()})
    return scheme.apply(model), {**variables, QUANTIZATION: held}


class _Built(NamedTuple):
    """A decoder source built and bound: the model, its variables, the
    record it was translated to, what it was built from, the source
    tensors' bindings and the ones none binds, and the encoder `export`
    writes through where the bindings do not (`Pretrained.export_adapter`)."""

    model: nn.Module
    variables: Variables
    record: Mapping[str, object]
    built: Mapping[str, object]
    layouts: tuple[WeightLayout, ...]
    retained: dict[str, np.ndarray]
    export_adapter: Callable[..., Mapping[str, np.ndarray]] | None = None


def _wrapper_source(config: Mapping[str, object], record: decoder_parts.WrapperFields,
                    tensors: Mapping[str, np.ndarray], directory: Path, *, dtype: str, attention_impl: str,
                    max_seq_len: int | None, param_dtype: str, lazy: bool,
                    checked: decoders.Checked) -> _Built:
    """A multimodal wrapper, `record` its translated config: the decoder under its text_config and the
    towers beside it; `checked` the spellings the load checked equal before decoding."""
    text_fields = _wrapper_text_fields(config, record, max_seq_len)
    parts, language, media = decoders.read_wrapper(tensors, record, param_dtype=param_dtype, lazy=lazy,
                                                   checked=checked)
    for fields in (text_fields, record['text']):
        decoders.held_depths(fields, parts["language_model"]["params"])
    text = NativeFields(CausalTransformer, {**text_fields, "dtype": dtype, "attention_impl": attention_impl})
    wrapper: decoder_parts.WrapperFields = {**record, "text": text}
    language_model = from_record(CausalTransformer, wrapper["text"])
    model = _wrapper_model(config, record, language_model, dtype=dtype)
    variables = native_variables({**parts, "language_model": decoders.with_constants(
        parts["language_model"], record["text"], directory)})
    # Every component the record builds holds every leaf it computes with: a tower, projector or
    # prediction-depth tensor the checkpoint lacks is refused here, as a decoder's is.
    decoder_parts.check_decoder_tree(variables, model)
    layouts, retained = _wrapper_layouts(tensors, record, variables, language, media)
    return _Built(model, variables, record, wrapper, layouts, retained)


@dataclasses.dataclass(frozen=True)
class DecoderRoute:
    """A registered decoder family's source, `record` its translated config."""

    record: decoder_parts.DecoderFields


@dataclasses.dataclass(frozen=True)
class WrapperRoute:
    """A media wrapper's source, `wrapper` its translated config."""

    wrapper: decoder_parts.WrapperFields


@dataclasses.dataclass(frozen=True)
class DiffusionGemmaRoute:
    """DiffusionGemma's source, read by `dew.interop.diffusion_gemma`."""


@dataclasses.dataclass(frozen=True)
class VerifiedRoute:
    """An unregistered decoder read through a registered family's convention, `verified` the mapping
    transformers' class checked (tier 2, `dew.interop.verify`)."""

    verified: verify.VerifiedMapping


@dataclasses.dataclass(frozen=True)
class TorchRoute:
    """An unregistered causal LM no registered convention computes, run as transformers' own forward
    by torchax (tier 3, `dew.interop.torchax_fallback`), `admitted` the decision that it runs."""

    admitted: TorchAdmission


type SourceRoute = DecoderRoute | WrapperRoute | DiffusionGemmaRoute | VerifiedRoute | TorchRoute
"""How `Pretrained.load` reads a source, decided on its config alone before any weight downloads
(`source_route`): each case carries what its reader reads."""


def source_route(config: Mapping[str, object], *,
                 verified: Callable[..., verify.VerifiedMapping | verify.Unmapped] = verify.verify_mapping,
                 admitted: Callable[[verify.Unmapped], TorchAdmission] | None = None,
                 remote: Path | None = None, pinned: sources.SourceReference | None = None) -> SourceRoute:
    """The route `Pretrained.load` reads `config`'s source by, or the refusal it raises for it, on the
    config alone: a storage format the codecs cannot decode, a registered family's or wrapper's own
    refusal of the config, a speculative drafter of any model_type (`decoder_parts.DrafterRefused`),
    or an unregistered model_type no registered family's convention computes and tier 3 does not run
    (`torchax_fallback.admitted`; tier 3 reads float weights only).
    `verified` checks an unregistered decoder against transformers (`verify.verify_mapping`), and
    `admitted` decides whether tier 3 runs one it leaves unmapped (`torchax_fallback.admitted` when
    None); a census over many configs passes a memo of each. `remote` is the repository whose own
    modeling code the load opted into running (`trust_remote_code`), and `pinned` the repository, Hub
    commit and folder it was fetched at, which a verification of that code is recorded under."""
    source_quantization(config)
    decoder_parts.refuse_drafter(config)
    if (route := _native_route(config)) is not None:
        return route
    match verified(config) if remote is None else verified(config, remote=remote, pinned=pinned):
        case verify.VerifiedMapping() as mapping:
            return VerifiedRoute(mapping)
        case verify.Unmapped() as unmapped:
            from dew.interop import torchax_fallback

            torchax_fallback.refuse_quantized(config)
            return TorchRoute((admitted or torchax_fallback.admitted)(unmapped))


def _native_route(config: Mapping[str, object]) -> DecoderRoute | WrapperRoute | DiffusionGemmaRoute | None:
    """The route of a source Dew translates itself, on its config alone: DiffusionGemma, a media wrapper
    or a registered decoder family; None for an unregistered model_type (tiers 2 and 3)."""
    family = config.get("model_type")
    if family == "diffusion_gemma":
        return DiffusionGemmaRoute()
    if _media_wrapper(config):
        return WrapperRoute(decoders.translate_wrapper_config(config))
    if isinstance(family, str) and family not in decoders.families():
        return None
    return DecoderRoute(decoders.translate_config(config))


def _lazy_route(config: Mapping[str, object]) -> DecoderRoute | WrapperRoute:
    """The route of a source read one tensor at a time (`native_backbone`): an unquantized registered
    decoder or its media wrapper, of a family whose preparation leaves its tensors as they are stored, or
    splits a packed one into views of it (`Packed.viewed`).
    Any other source is refused on its config, before any weight is read or any verification runs."""
    if source_quantization(config) is not None:
        raise ValueError("a lazy read takes unquantized safetensors; a whole-model codec is not bounded")
    decoder_parts.refuse_drafter(config)
    match _native_route(config):
        case DecoderRoute() | WrapperRoute() as route:
            text = _records.text(route.wrapper["text_model_type"] if isinstance(route, WrapperRoute)
                                 else config["model_type"], "model_type")
            entry = decoders.family_entry(text)
            # A packed tensor split into views of itself stays unread until a row is placed; a preparation,
            # or parts interleaved in runs, copies it at translation.
            if (entry.prepare is not decoder_parts.DecoderFamily.prepare
                    or not all(packing.viewed for packing in entry.packed)):
                raise ValueError(f"a lazy read takes a family whose tensors translate as they are stored or "
                                 f"split into views of themselves; {text}'s preparation or interleaved "
                                 f"packing copies checkpoint weights")
            return route
        case _:
            raise ValueError(f"a lazy read takes a registered native decoder or its media wrapper, not "
                             f"{config.get('model_type')!r}")


def _media_wrapper(config: Mapping[str, object]) -> bool:
    """Whether a source is a media wrapper, its decoder under text_config.

    Where its model_type is a registered decoder family, the text half is
    the model unless the family reads its media bundle whole
    (`DecoderFamily.wrapper`). DiffusionGemma nests its text config too.
    """
    family = config.get("model_type")
    return (family != "diffusion_gemma" and "text_config" in config
            and (family not in decoders.families() or decoders.bundles(config)))


def _spelled_source_weights(spelling: Mapping[str, str], model: CausalTransformer,
                            variables: Mapping[str, object], config: Mapping[str, object]
                            ) -> Mapping[str, np.ndarray]:
    """`decoders.export_decoder_weights` under the source's own names: `spelling` maps each name the
    family's encoder writes to the one the source stored it under (`DecoderFamily.spelling`)."""
    written = decoders.export_decoder_weights(model, variables, config)
    return {spelling.get(name, name): tensor for name, tensor in written.items()}


def _decoder_source(config: Mapping[str, object], tensors: Mapping[str, np.ndarray], directory: Path,
                    route: DecoderRoute | VerifiedRoute, *, dtype: str, attention_impl: str,
                    max_seq_len: int | None, param_dtype: str, lazy: bool, quantized: bool,
                    checked: decoders.Checked) -> _Built:
    """A decoder of a registered family, or of one it was verified as (tier 2), as `route` read it.
    `quantized` says the source ships weights in a quantization format; `checked` names the
    spellings the load checked equal before decoding."""
    match route:
        case DecoderRoute(record=translated):
            verified = None
            # The route's record is its own; this load edits a copy of it.
            record = NativeFields(CausalTransformer, translated)
            # translate_config refused every model_type but a registered family's name.
            family = _records.text(config.get("model_type"), "model_type")
        case VerifiedRoute(verified=verified):
            family = verified.entry(config)
            record = verified.translate(config, tensors, family)
    if max_seq_len is not None:
        record["max_seq_len"] = max_seq_len
    placed, readings = decoders.read_decoder(tensors, record, family, param_dtype=param_dtype, lazy=lazy,
                                             checked=checked)
    variables = decoders.with_constants(placed, record, directory)
    decoders.held_depths(record, variables["params"])
    built = {**record, "dtype": dtype, "attention_impl": attention_impl}
    model = from_record(CausalTransformer, built)
    decoder_parts.check_decoder_tree(variables, model)
    # The bindings are what an adapter loader resolves source names through
    # and what a quantized source is written back through, so a
    # derived-export family binds too; `save` picks its writer by
    # preserve_source_layout and quantization, not by whether bindings exist.
    # A family whose tensors are rewritten before the path map reads them
    # (GPT-2's and GPT-NeoX's prepare) has no raw-name bindings.
    entry = decoders.family_entry(family)
    layouts, retained = ((), {})
    if entry.preserve_source_layout or entry.prepare is decoder_parts.DecoderFamily.prepare:
        layouts, retained = decoder_layouts(readings, record, family, variables)
    export_adapter = None
    if verified is None and not entry.preserve_source_layout:
        # The family's own encoder, the inverse `PretrainedDecoder.from_model` writes
        # through too. A quantized source keeps its packed format by going back
        # over its source names (`weight_layouts`) where it binds them; a family
        # whose `prepare` rewrites its tensors first (GPT-2, GPT-J, GPT-NeoX,
        # ModernBERT) binds none, and its encoder undoes that rewrite, under the
        # source's own names, before the format encodes the weights again (`export`).
        if not quantized:
            export_adapter = decoders.export_decoder_weights
        elif not layouts:
            export_adapter = partial(_spelled_source_weights, {
                entry.spelling(name): name for name in tensors if entry.spelling(name) != name})
    return _Built(model, variables, record, built, layouts, retained, export_adapter)


def _resident[Loaded: Pretrained](loaded: Loaded, *, weights: bool, streaming: bool, on_host: bool) -> Loaded:
    """`loaded`'s variables as the load asked for them: their shapes and dtypes alone without its
    weights, host arrays where it keeps them on the host, else on JAX's default device unless the
    load placed them as it streamed them. A composite applies each to every component."""
    if not weights:
        return loaded.map_variables(lambda tree: jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), tree))
    if on_host:
        return loaded.map_variables(lambda tree: jax.tree.map(np.asarray, tree))
    return loaded if streaming else loaded.map_variables(on_default_device)


def _selected_payload(fallback: str | None, **files: str | None) -> list[str]:
    """The payload `files` `Pretrained.load` was given by name (`gguf_file`, `single_file`,
    `dduf_file`), refusing more than one, a fallback beside any, and a fallback no loader is."""
    if fallback not in (None, "torchax"):
        raise ValueError(f"fallback={fallback!r} names no loader; the one fallback is 'torchax', "
                         "tier 3 through transformers' PyTorch forward")
    selected = [name for name, given in files.items() if given is not None]
    if len(selected) > 1:
        raise ValueError(f"{' and '.join(selected)} each name the whole payload to load; pass one")
    if fallback is not None and selected:
        raise ValueError(f"fallback='torchax' builds the transformers model the repo's config.json "
                         f"and weights describe, and {' and '.join(selected)} selects another payload; "
                         "load it without fallback")
    return selected



def _source_metadata(requested: sources.SourceReference, selected: list, fallback: str | None,
                     overrides: Mapping[str, object] | None
                     ) -> tuple[Path, sources.SourceReference, CompositeSource | None]:
    """Resolve one metadata snapshot of `requested` before any component's weights are read: the
    snapshot, the reference pinned to its commit (none for a local directory), and the composite
    release its selected folder describes under the load's `overrides`. Explicit payload/fallback
    selections take precedence over composite metadata."""
    directory = sources.snapshot(requested.name, requested.revision, weights=False)
    commit = None if os.path.isdir(requested.name) else directory.name
    reference = replace(requested, revision=commit, subfolder=requested.subfolder or None)
    composite = (None if selected or fallback is not None
                 else _composite_description(reference.directory(directory), overrides))
    return directory, reference, composite


def _composite_description(directory: Path,
                           overrides: Mapping[str, object] | None = None) -> CompositeSource | None:
    """A registered composite's metadata under the load's `overrides`, without reading or placing its
    weights."""
    from dew.registry import source_kinds

    for provider in source_kinds.values():
        description = provider.describe(directory, overrides)
        if description is not None:
            return description
    return None


def _load_composite_source(source: sources.SourceReference, root: Path,
                           description: CompositeSource, *, placed: Callable[[Variables], Variables],
                           streaming: bool, dtype: str, param_dtype: str, attention_impl: str,
                           max_seq_len: int | None, weights: bool,
                           overrides: Mapping[str, object] | None = None,
                           code: Path | None = None, prediction_depths: int | None = None) -> Pretrained:
    """Materialize the described components through the native backbone and tensor-map owners.
    All Hub paths remain relative to the resolved snapshot root, including a selected subfolder. `code`
    is the snapshot holding the source's own modules where the load runs them (`trust_remote_code`),
    which the backbone's route and load read at its component's folder, as a decoder alone reads its."""
    from dew.interop.streaming import materialize
    from dew.interop.weights import record_layouts

    component = description.backbone
    head_source = description.head
    directory = source.directory(root)
    backbone = None
    shared: Mapping[str, np.ndarray] = {}
    if component is not None:
        # The route of the config the backbone translates (`_native_config`), decided once, before its
        # weights are read, and carried to its load, so an unregistered backbone is verified once. The
        # transformers forward reads its backbone's weights itself, and none without them; the shared
        # file is read for it only where the head is stored in it, but its weights' snapshot is resolved
        # and pinned here whenever it reads weights.
        pinned = source.component(component.config_folder)
        remote = None if code is None else pinned.directory(code)
        routed = _routed(_native_config(component.config, overrides), remote=remote, pinned=pinned,
                         text_only=False, overrides=overrides, prediction_depths=prediction_depths)
        forwarded = isinstance(routed.route, TorchRoute)
        reads_shared = not forwarded or bool(head_source.weight_file is None and component.tensor_prefix)
        if weights or reads_shared:
            weight_source = source.component(component.weights_folder)
            root = weight_source.weights(root)
            directory = source.directory(root)
            if source.revision is not None:
                source = replace(source, revision=root.name)
            if reads_shared:
                shared = sources.load_shards(weight_source.directory(root))
        tensors = None if forwarded else component.own_tensors(shared)
        backbone, _ = _load_native_source(
            source, root,
            gguf_file=None, component=component, tensors=tensors,
            placed=placed, streaming=streaming, dtype=dtype, param_dtype=param_dtype,
            attention_impl=attention_impl, max_seq_len=max_seq_len, weights=weights, overrides=overrides,
            remote=remote, routed=routed, prediction_depths=prediction_depths)
    else:
        _unread_overrides(overrides, "a release without a native backbone")
    missing = False
    if head_source.weight_file is None:
        head_tensors = {} if component is None else component.other_tensors(shared)
    else:
        path = (source.file(root, head_source.weight_file) if head_source.initialize is None
                else source.optional_file(root, head_source.weight_file))
        missing = path is None
        head_tensors = {} if path is None else sources.tensor_file(path)
    head = head_source.build(backbone, dtype, attention_impl,
                             shapes=None if missing else {name: tensor.shape
                                                          for name, tensor in head_tensors.items()})
    if missing:
        if backbone is None or head_source.initialize is None:
            raise ValueError("initializing a missing readout requires its backbone and initializer")
        initialize = head_source.initialize

        def drawn(variables):
            initialized = initialize(head, replace(backbone, variables=variables))
            return jax.tree_util.tree_map_with_path(
                lambda path, leaf: leaf.astype(checkpoint_dtype(
                    np.dtype(leaf.dtype), param_dtype,
                    path=tuple(jax.tree_util.keystr((key,), simple=True) for key in path))), initialized)

        if weights:
            variables = drawn(backbone.variables)
        else:
            shapes = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                                   backbone.variables)
            variables = jax.eval_shape(drawn, shapes)
    else:
        params, _ = record_layouts("", decoder_parts.split_packed(head_tensors, head_source.packed),
                                   head_source.path, (), param_dtype=param_dtype, lazy=True)
        if head_source.reshape is not None:
            params = head_source.reshape(head, params)
        if not streaming:
            params = materialize(params)
        variables = {"params": params}
    if backbone is None:
        return Pretrained(head, placed(variables), None, {}, directory,
                          record_fields(head, type(head)), revision=source.revision)
    return description.bundle(backbone, head, placed(variables), directory,
                              description.tokenizer_folder, **description.fields)


def _native_config(config: Mapping[str, object],
                   overrides: Mapping[str, object] | None) -> Mapping[str, object]:
    """The config a native load builds from a source's own: a mamba_ssm one's normalized to transformers'
    Mamba-2 fields, with the load's `overrides` over it (`with_overrides`)."""
    if mamba2.is_mamba_ssm(config):
        config = mamba2.config_from_mamba_ssm(config)
    return with_overrides(config, overrides)


class _Routed(NamedTuple):
    """A native source's config, its route and its wrapper fields where a wrapper's language model is
    taken alone (`_routed`)."""

    config: Mapping[str, object]
    route: SourceRoute
    language: decoder_parts.WrapperFields | None


def _routed(config: Mapping[str, object], *, remote: Path | None, pinned: sources.SourceReference | None,
            text_only: bool, overrides: Mapping[str, object] | None,
            prediction_depths: int | None = None) -> _Routed:
    """A native source's config, its route and, where `text_only` takes a wrapper's language model
    alone, that model's wrapper fields, decided before any weight downloads, on the config alone: a
    format the codec cannot read, a family's refusal, an unregistered decoder checked against
    transformers (tiers 2 and 3). The language model reads as its text family's checkpoint
    (`decoders.language_model_config`); a route that reads no `overrides` refuses them. `remote` and
    `pinned` are `source_route`'s. Where `prediction_depths` is given, the config is the checkpoint's
    of that many depths (`_kept_depths`), which the load reads and saves."""
    if prediction_depths is not None:
        config = _kept_depths(config, prediction_depths)
    route = source_route(config, remote=remote, pinned=pinned)
    language = route.wrapper if text_only and isinstance(route, WrapperRoute) else None
    if language is not None:
        if decoders.bundles(config):
            raise ValueError(f"{config.get('model_type')} reads its media bundle whole as one decoder "
                             "family; text_only takes a wrapper whose language model is its own text "
                             "family's checkpoint")
        if source_quantization(config) is not None:
            raise ValueError("text_only takes an unquantized wrapper: a quantized one's module selection "
                             "names the wrapper's modules, which its text checkpoint renames")
        config = decoders.language_model_config(config)
        route = source_route(config, remote=remote, pinned=pinned)
    elif text_only and isinstance(route, DiffusionGemmaRoute):
        raise ValueError("text_only takes a media wrapper's language model; DiffusionGemma's decoder is "
                         "its whole model")
    if isinstance(route, TorchRoute | DiffusionGemmaRoute):
        _unread_overrides(overrides, "the transformers forward" if isinstance(route, TorchRoute)
                          else "a diffusion Gemma's text tower")
    return _Routed(config, route, language)


class _Translated(NamedTuple):
    """A native source translated (`_translated`): its model bound, and what `save` needs to write a
    quantized source back (`_decoded`)."""

    built: _Built
    quantized_tensors: tuple[str, ...]
    scale_dtype: str | None
    grid: dict[str, np.ndarray]


def _translated(config: Mapping[str, object],
                route: DecoderRoute | WrapperRoute | DiffusionGemmaRoute | VerifiedRoute,
                tensors: Mapping[str, np.ndarray], directory: Path, *, dtype: str, param_dtype: str,
                attention_impl: str, max_seq_len: int | None, streaming: bool) -> _Translated:
    """`config`'s native model and variables over `tensors`, as `route` reads them: a quantized source's
    weights decoded, the decoder, media wrapper or DiffusionGemma translated and checked against its
    tree, and any stored input scales bound. No processor, generation config or bundle is built."""
    tensors, quantized_tensors, scale_dtype, grid, checked = _decoded(
        tensors, config, declared_dtype(config, tensors) if param_dtype == AUTO else param_dtype,
        media=isinstance(route, WrapperRoute))
    match route:
        case DiffusionGemmaRoute():
            from dew.interop import diffusion_gemma

            model = diffusion_gemma.build(
                config, dtype=dtype, attention_impl=attention_impl, max_seq_len=max_seq_len
            )
            # The vision layouts are the export's alone: the bundle's own are derived from its tree, as
            # every adapter (LoRA's projections) reads them.
            variables, vision = diffusion_gemma.translate_weights(tensors, config, param_dtype=param_dtype,
                                                                  lazy=streaming,
                                                                  checked=checked)
            built = _Built(model, variables, config,
                           {**config, "dtype": dtype, "attention_impl": attention_impl}, (), {},
                           partial(diffusion_gemma.export_weights, vision=vision))
        case WrapperRoute(wrapper=wrapper):
            built = _wrapper_source(
                config, wrapper, tensors, directory, dtype=dtype, attention_impl=attention_impl,
                max_seq_len=max_seq_len, param_dtype=param_dtype, lazy=streaming, checked=checked)
        case DecoderRoute() | VerifiedRoute():
            built = _decoder_source(
                config, tensors, directory, route, dtype=dtype, attention_impl=attention_impl,
                max_seq_len=max_seq_len, param_dtype=param_dtype, lazy=streaming,
                quantized=source_quantization(config) is not None, checked=checked)
    model, variables = _input_quantization(built.model, built.variables, built.layouts, config, grid)
    return _Translated(built._replace(model=model, variables=variables), quantized_tensors, scale_dtype, grid)


def native_backbone(directory: str | Path, *, param_dtype: str) -> tuple[Mapping[str, object], Variables]:
    """The native backbone a local source holds, its config and variables, every leaf a `SourceLeaf`
    that reads its tensor from the read-only memory-mapped safetensors only when it is read: a decoder,
    a media wrapper around one, or the backbone a release with a separately stored readout describes,
    translated and checked as `Pretrained.load` translates it, with no processor, generation config or
    bundle. Its route is a lazy one (`_lazy_route`); a source without safetensors weights is refused
    before any translation. `param_dtype` is a dtype name or 'auto'."""
    from dew.interop.safetensors_io import read_weights

    folder = Path(directory)
    description = _composite_description(folder)
    component = (NativeComponent(_source_config(str(folder), folder, None, None)[0]) if description is None
                 else description.backbone)
    if component is None:
        raise ValueError(f"{folder} holds a standalone readout and no native backbone")
    translated = _translated(component.config, _lazy_route(component.config),
                             component.own_tensors(read_weights(folder / component.weights_folder)), folder,
                             dtype="float32", param_dtype=param_dtype, attention_impl="reference",
                             max_seq_len=None, streaming=True)
    return component.config, translated.built.variables


DEPTH_COUNTS = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", "n_mtp_layers")
"""The fields a source's config counts its multi-token prediction depths by, family by family."""


def _scopes(config: dict) -> list[dict]:
    """The scopes a decoder's fields are stated in: `config`'s top, then its `text_config` if any."""
    text = config.get("text_config")
    return [config, *([text] if isinstance(text, dict) else [])]


def _kept_depths(config: Mapping[str, object], kept: int) -> dict:
    """`config` as the config of a checkpoint holding its first `kept` prediction depths alone: each
    depth count it states (`DEPTH_COUNTS`, at its top or in its `text_config`) at most `kept`. A DSpark
    drafter's stages, which its `compress_ratios` list past the trunk's layers, draft as one, its heads
    on its last stage (`dew.interop.families.deepseek`), so it is kept whole or, at 0, not at all;
    a count between is refused. A load reads, builds and saves the source this config states
    (`Pretrained.load(prediction_depths=)`)."""
    written = copy.deepcopy(dict(config))
    for scope in _scopes(written):
        for counted in DEPTH_COUNTS:
            if isinstance(scope.get(counted), int) and scope[counted] > kept:
                scope[counted] = kept
        ratios, layers = scope.get("compress_ratios"), scope.get("num_hidden_layers")
        if scope.get("dspark_block_size") and isinstance(ratios, list) and isinstance(layers, int):
            stages = len(ratios) - layers
            if 0 < kept < stages:
                raise ValueError(f"a DSpark drafter's {stages} stages draft as one, its heads on the last: "
                                 f"prediction_depths keeps all of them or none (0), not {kept}")
            if not kept:
                scope["compress_ratios"], scope["dspark_block_size"] = ratios[:layers], 0
    return written


def _kept_tensors(tensors: Mapping[str, np.ndarray], config: Mapping[str, object], route: SourceRoute,
                  kept: int) -> Mapping[str, np.ndarray]:
    """`tensors` without those of the prediction depths past the first `kept`, none read: the depth of
    each as its family places it (`decoders.source_depth`), a decoder's by its route's record, a
    wrapper's language model's through the wrapper's names."""
    match route:
        case DecoderRoute(record=record):
            depth = decoders.source_depths(record, _records.text(config.get("model_type"), "model_type"))
        case WrapperRoute(wrapper=wrapper):
            depth = decoders.wrapper_source_depths(wrapper)
        case VerifiedRoute(verified=verified):
            family = verified.entry(config)
            depth = decoders.source_depths(verified.translate(config, tuple(tensors), family), family)
        case _:
            return tensors
    specs = tensor_specs(tensors)
    kept_specs = {name: spec for name, spec in specs.items() if (held := depth(name)) is None or held < kept}
    return LazyTensors(kept_specs, tensors.__getitem__)


def _load_native_source(source: sources.SourceReference, root: Path, *,
                        gguf_file: str | None, placed: Callable[[Variables], Variables], streaming: bool,
                        dtype: str, param_dtype: str, attention_impl: str,
                        max_seq_len: int | None, weights: bool = True,
                        component: NativeComponent | None = None,
                        tensors: Mapping[str, np.ndarray] | None = None,
                        overrides: Mapping[str, object] | None = None,
                        remote: Path | None = None, pinned: sources.SourceReference | None = None,
                        routed: _Routed | None = None,
                        text_only: bool = False, prediction_depths: int | None = None
                        ) -> tuple[Pretrained, bool]:
    """Decode native source weights and bind their processors and export layout, and whether the
    bundle is a wrapper's language model taken alone (`text_only`). `remote` is the source's own code
    where the load runs it, and `pinned` where it was fetched (`source_route`). `routed` is the route a
    caller decided for this config already (`_routed`), a composite's for its backbone before reading
    its weights, which the load takes rather than route the config, and verify its code, again.

    Diffusers pipelines and the explicit torchax fallback have already taken
    their own routes; this boundary handles native decoders and media wrappers,
    and hands an unregistered causal LM tier 3 runs (`TorchRoute`) to it.
    `overrides` (`Pretrained.load`'s `modeling_overrides`) join the source's own.
    """
    name_or_dir, commit = source.name, source.revision
    directory = source.directory(root)
    gguf_path = None if gguf_file is None else source.file(root, gguf_file)
    if component is None:
        config, tensors = _source_config(name_or_dir, directory, commit, gguf_path)
    else:
        config = component.config
        directory = source.component(component.config_folder).directory(root)
    mamba_ssm = mamba2.is_mamba_ssm(config)
    config = _native_config(config, overrides)
    family = config.get("model_type")
    if routed is None:
        routed = _routed(config, remote=remote, pinned=pinned, text_only=text_only, overrides=overrides,
                         prediction_depths=prediction_depths)
    config, route, language = routed
    if isinstance(route, TorchRoute):
        from dew.interop import torchax_fallback
        selected = source if component is None else source.component(component.config_folder)
        loaded = torchax_fallback.load(selected, root, dtype=dtype, param_dtype=param_dtype,
                                       attention_impl=attention_impl, max_seq_len=max_seq_len,
                                       admission=route.admitted, weights=weights)
        return replace(loaded, variables=placed(loaded.variables)), False
    if tensors is None and not weights:
        tensors = sources.stored_tensors(source, root)
    if tensors is None:
        root = source.weights(root)
        directory = source.directory(root)
        tensors = sources.load_shards(directory)
    if mamba_ssm:
        tensors = mamba2.tensors_from_mamba_ssm(tensors)
    if prediction_depths is not None:
        # Before a wrapper's language model is taken out of it, by the wrapper's own names: a discarded
        # depth's spellings are neither read nor compared (`_language_model_tensors`).
        held = route if language is None else WrapperRoute(language)
        tensors = _kept_tensors(tensors, config, held, prediction_depths)
    if language is not None:
        tensors = _language_model_tensors(tensors, language)
    translated = _translated(config, route, tensors, directory, dtype=dtype, param_dtype=param_dtype,
                             attention_impl=attention_impl, max_seq_len=max_seq_len, streaming=streaming)
    model, variables, record, built, layouts, retained, export_adapter = translated.built
    quantized_tensors, scale_dtype, grid = translated[1:]
    resolved_commit = commit
    if component is None and commit is not None:
        resolved_commit = root.name
    source = replace(source, revision=resolved_commit)
    processor = source_processor(directory, config, record, model, gguf_path, source=source)
    generation_config = read_generation_config(directory)

    # Dew's byte vocabulary has no files: an export names it in
    # generation_config.json, and the name is the whole record of it.
    named = generation_config.get("tokenizer_name") if processor is None else None

    def bundle[K: Pretrained](kind: type[K]) -> K:
        return kind(model, placed(variables), processor, config, directory, built, generation_config,
                    layouts, retained, export_adapter, quantized_tensors=quantized_tensors,
                    quantized_scale_dtype=scale_dtype, quantization_grid=grid,
                    # The weights' commit: a pickle repo's may be its conversion's.
                    revision=resolved_commit,
                    tokenizer="byte" if named == "byte" else None,
                    # The modules a verification from parts ran, from the source's own snapshot.
                    code=remote if isinstance(route, VerifiedRoute) and route.verified.source is not None
                    else None)

    if family == "diffusion_gemma":
        return bundle(PretrainedBlockDecoder), False
    # The masked-diffusion families' records attend both ways and reserve a
    # mask id to denoise; a wrapper's record nests its decoder's.
    if record.get("causal") is False and record.get("mask_token_id") is not None:
        return bundle(PretrainedMaskedDecoder), language is not None
    return bundle(PretrainedDecoder), language is not None
