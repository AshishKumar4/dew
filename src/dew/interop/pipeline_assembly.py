"""Assemble a published diffusion pipeline's parts as Dew values.

A diffusers directory names its components (`model_index.json`): the
denoiser is built from its own config and weights, the text towers into the
conditioner the denoiser reads, the VAE into the autoencoder, and the
scheduler file into the source's training process and its sampling policy
(`SourceTask`). `assemble_pipeline` returns those parts, which
`Pretrained.load` and `load_diffusion_source` hold as a `PretrainedPipeline`;
`load_diffusion_conditioner` builds one conditioner alone.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, TypedDict

import jax
import numpy as np
from flax import linen as nn

from dew import _records
from dew.batch import Variables
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import Origin, SourceGrid, SourceSchedule
from dew.inference.loading import place
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.inputs.diffusion import (
    Composition,
    DiffusionConditioner,
    HiddenStatesConditioner,
    QwenImageConditioner,
    T5Segment,
    WanConditioner,
)
from dew.interop import decoder_parts, hf_decoders as decoders, sources
from dew.interop.components import bind_component
from dew.interop.decoder_bindings import decoder_layouts
from dew.interop.safetensors_io import LazyTensors
from dew.interop.sources import StoredWeights
from dew.interop.streaming import LazyTree, WeightLayout
from dew.interop.weights import AUTO, ParamTree, source_alias
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import dtype_name, from_record, resolve_dtype
from dew.sampling.guidance import CFG

if TYPE_CHECKING:
    from jax.sharding import Mesh

    from dew.distributed.placement import Layout, MeshSpec


def native_variables(parts: Mapping[str, Mapping[str, ParamTree]]) -> dict[str, dict[str, ParamTree]]:
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
    if published == "Flux2KleinPipeline" and _records.boolean(
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
    grid: SourceGrid


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
                     if name is not None and component_present(index, name))

    def build(self, directory: Path, index: Mapping[str, object], denoiser: _Denoiser,
              policy: _Call, compute, size: tuple[int, int], *, param_dtype: str,
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False,
              stored: StoredWeights | None = None
              ) -> tuple[DiffusionConditioner, tuple[WeightLayout, ...], Mapping[str, np.ndarray],
                         dict[str, Mapping[str, object]]]:
        """Construct the published text composition from metadata and either weight source."""
        names = tuple(name for name in self.towers if component_present(index, name))
        if not names:
            raise ValueError("A latent diffusion source needs at least one text encoder")
        towers, tokenizers, text_params, layouts, pins = _clip_towers(
            directory, names, compute, param_dtype=param_dtype, params=params, lazy=lazy, stored=stored)
        components: dict[str, Mapping[str, object]] = {
            name: _component_config(directory, name) for name in names}
        t5 = None
        if self.t5_tower is not None and component_present(index, self.t5_tower):
            t5, t5_params, t5_layouts, components[self.t5_tower] = _t5_tower(
                directory, compute, self.t5_tower, policy.sequence, param_dtype=param_dtype,
                params=None if params is None else params[self.t5_tower], lazy=lazy, stored=stored)
            if params is None:
                text_params = {**text_params, self.t5_tower: t5_params}
            layouts += t5_layouts
        height, width = _geometry(index, size)
        encoder = DiffusionConditioner(
            towers, tokenizers, names, text_params, str(directory), height, width,
            denoiser.context_width, composition=self.composition, t5=t5,
            guidance=policy.guidance if self.embeds_guidance and not policy.guided else None,
            aesthetics=bool(index.get("requires_aesthetics_score", False)), param_dtype=param_dtype,
            tokenizer_files=pins)
        return encoder, layouts, {}, components

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
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False,
              stored: StoredWeights | None = None
              ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...], Mapping[str, np.ndarray],
                         dict[str, Mapping[str, object]]]:
        """Construct `QwenImageConditioner` at the pipeline's prompt budget."""
        return _qwen_image_conditioning(directory, index, compute, tokens=policy.sequence,
                                        param_dtype=param_dtype, attention_impl=attention_impl,
                                        params=params, lazy=lazy, stored=stored)

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
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False,
              stored: StoredWeights | None = None
              ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...], Mapping[str, np.ndarray],
                         dict[str, Mapping[str, object]]]:
        """Construct `HiddenStatesConditioner` at the pipeline's prompt budget."""
        return _hidden_states_conditioning(
            directory, index, compute, pipeline=self.pipeline, tokens=policy.sequence,
            guidance=policy.guidance if self.embeds_guidance else None, param_dtype=param_dtype,
            attention_impl=attention_impl, params=params, lazy=lazy, stored=stored)

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
              attention_impl: str = "auto", params: Variables | None = None, lazy: bool = False,
              stored: StoredWeights | None = None
              ) -> tuple[WanConditioner, tuple[WeightLayout, ...], Mapping[str, np.ndarray],
                         dict[str, Mapping[str, object]]]:
        """Construct `WanConditioner` at the pipeline's prompt budget."""
        encoder, layouts, components = _wan_conditioning(
            directory, compute, tokens=policy.sequence, param_dtype=param_dtype, params=params, lazy=lazy,
            stored=stored)
        return encoder, layouts, {}, components

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
    plan: Callable[[Mapping[str, tuple[int, ...]]], tuple[WeightLayout, ...]]
    """The layouts `weights` returns, from the stored tensors' shapes alone: what a
    pipeline bound from saved variables exports through (`assemble_pipeline`)."""
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


def assemble_pipeline(directory: Path, index: Mapping[str, object], *, dtype: str,
                           attention_impl: str, param_dtype: str = "float32",
                           variables: Variables | None = None, lazy: bool = False,
                           text: bool = True, stored: StoredWeights | None = None) -> PipelineParts:
    """Read a published latent diffusion directory into native modules and variables.

    The directory's own denoiser component selects the family (`pipeline_denoiser`),
    and everything the families share - the autoencoder, the text towers,
    the geometry, the conditioning, the safety head a file declares, the
    schedule and the call policy - is read once here.

    Supplied `variables` are a saved tree in this layout, bound as they are:
    every module is built from the directory's metadata and no weight file is
    read, so the directory needs only its configs and tokenizers. Their
    layouts come from the `stored` source's headers (`StoredWeights`), which
    the export writes them back by, and a text encoder's unread tensors are
    read from it when the export does; without it they have none.

    With `lazy` every component's leaves are `SourceLeaf` recipes over the
    mapped files, for a placement to read one device shard at a time
    (`dew.inference.loading.place`), but for the convolution kernels and a
    UNet's per-head attention kernels, which are read whole
    (`dew.interop.weights.record_layouts`). `text=False` binds the text
    encoder to no weights and leaves it out of the variables.
    """
    compute = resolve_dtype(dtype)
    denoiser = pipeline_denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    if variables is None:
        denoiser_variables, denoiser_layouts = denoiser.weights(param_dtype, lazy)
    else:
        denoiser_variables = {name: value for name, value in variables.items()
                              if name not in ("encoders", "autoencoder")}
        shapes = _shapes(stored, denoiser.component)
        denoiser_layouts = () if shapes is None else denoiser.plan(shapes)
    held = {} if variables is None else variables["encoders"]
    if not text:
        held = {"conditioning": {name: {} for name in text_components(denoiser.text, index)}}
    policy = _call_policy(index, denoiser)
    autoencoder, vae_params, vae_layouts, vae_config = _diffusion_vae(
        directory, compute, param_dtype=param_dtype,
        params=None if variables is None else variables["autoencoder"], lazy=lazy, stored=stored)
    rows, columns = denoiser.sample_size
    size = (rows * autoencoder.downscale_factor, columns * autoencoder.downscale_factor)
    encoder, text_layouts, text_retained, components = denoiser.text.build(
        directory, index, denoiser, policy, compute, size, param_dtype=param_dtype,
        attention_impl=attention_impl, params=held.get("conditioning"), lazy=lazy, stored=stored)
    components.update({denoiser.component: denoiser.config, "vae": vae_config})
    height, width = _geometry(index, size)
    # AutoencoderKLWan declares no channel count; it reads RGB.
    channels = _records.integer(vae_config.get("in_channels", 3), "in_channels")
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
    if component_present(index, "safety_checker"):
        finish, encoders["safety"], safety_layouts, safety_configs = _image_safety(
            directory, compute, param_dtype=param_dtype, params=held.get("safety"), lazy=lazy,
            stored=stored)
        components.update(safety_configs)
    schedule = SourceSchedule.from_config(_component_config(directory, "scheduler"))
    components["scheduler"] = dict(schedule.config)
    patch = denoiser.patch * autoencoder.downscale_factor
    tokens = (height // patch) * (width // patch)
    task = SourceTask(min(policy.steps, schedule.train_steps),
                      CFG(policy.guidance) if policy.guided and policy.guidance > 1 else None,
                      SourceGrid(dict(schedule.config), tokens, denoiser.origin))
    variables = {**denoiser_variables, "encoders": encoders, "autoencoder": vae_params}
    geometry = {"dew_height": height, "dew_width": width}
    if denoiser.frames is not None:
        geometry["dew_frames"] = sample.shape[0]
    config = {"model_index": {**index, **geometry}, **components}
    return PipelineParts(model=denoiser.model, variables=variables, config=config,
                              source=directory, model_config=denoiser.built,
                              weight_layouts=denoiser_layouts + vae_layouts + text_layouts + safety_layouts,
                              retained_tensors=text_retained,
                              process=schedule.training_process(tokens), inputs=inputs,
                              autoencoder=autoencoder, schedule=schedule, finish=finish, task=task)


def load_diffusion_conditioner[C: ConditionEncoder](
        checkpoint: str, kind: type[C], *, dtype: str | None = "bfloat16", param_dtype: str = "float32",
        revision: str | None = None, attention_impl: str = "auto", tokens: int | None = None,
        params: Variables | None = None, mesh: MeshSpec | Mesh | None = None, layout: Layout | None = None,
        size: tuple[int, int] | None = None) -> C:
    """Load the text conditioning a pipeline's denoiser reads, which must be a
    `kind`, or bind supplied parameters using metadata only. `tokens` replaces
    the pipeline's own prompt budget, and `size` the (height, width) it
    renders. `mesh` and `layout` place the weights it
    reads as `Pretrained.load`'s do, streamed one leaf at a time, so a
    pipeline's prompts can be encoded with its text encoder alone."""
    from dew.nn.autoencoders import AutoencoderKL

    compute = resolve_dtype(dtype)
    if param_dtype != AUTO:
        resolve_dtype(param_dtype)
    directory = sources.snapshot(checkpoint, revision, weights=False)
    with open(directory / "model_index.json") as handle:
        index = json.load(handle)
    if size is not None:
        index = {**index, "dew_height": size[0], "dew_width": size[1]}
    denoiser = pipeline_denoiser(directory, dtype=dtype, attention_impl=attention_impl)
    text = denoiser.text
    if text.conditioner is not kind:
        raise ValueError(f"{checkpoint} conditions through a {text.conditioner.__name__}; "
                         f"build it with {text.conditioner.__name__}.from_pretrained")
    towers = isinstance(text, _TextTowers)
    if params is None:
        # snapshot_download returns a commit directory. Keep both fetches on
        # that commit even when the requested Hub branch moves between them.
        directory = sources.snapshot(checkpoint, directory.name, weights=text_components(text, index))
    policy = _call_policy(index, denoiser)
    # The CLIP families' geometry is their VAE's; the language-model encoders
    # are bound at 16 pixels per latent position.
    scale = (AutoencoderKL(channels=tuple(_component_config(directory, "vae")["block_out_channels"]))
             .downscale_factor if towers else 16)
    rows, columns = denoiser.sample_size
    streaming = params is None and (mesh is not None or layout is not None)
    encoder, _, _, _ = text.build(
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
    weights = _component_weights(directory, "unet", functools.partial(diffusion.translate_unet_weights,
                                                                      model=model))

    pooled = model.additional_time_features > 0
    built = {"class": "unet_2d_condition",
             "fields": {**fields, "dtype": dtype,
                        "stages": [asdict(stage) for stage in model.stages]}}
    return _Denoiser(
        component="unet",
        model=model,
        weights=weights,
        plan=functools.partial(diffusion.unet_layouts, model=model),
        built=built,
        config=config,
        text=_TextTowers("clip_pooled" if pooled else "clip", ("text_encoder", "text_encoder_2")),
        patch=1,
        latent_input=model.in_channels,
        sample_size=_square(config),
        context_width=_records.integer(config.get("cross_attention_dim", 1280), "cross_attention_dim"),
        pipeline="StableDiffusionXLPipeline" if pooled else "StableDiffusionPipeline",
    )


def pipeline_denoiser(directory: Path, *, dtype: str | None, attention_impl: str) -> _Denoiser:
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


def _transformer_source(directory: Path, path_of: Callable[[str, int], tuple[str, ...] | None]
                        ) -> tuple[Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]],
                                   Callable[[Mapping[str, tuple[int, ...]]], tuple[WeightLayout, ...]]]:
    """Read a transformer denoiser's parameters, and their layouts, from its
    component, each where its family's `path_of(name, rank)` places it;
    `lazy` leaves them `SourceLeaf`s for a placement to read. Beside it, the
    same layouts from the stored shapes alone (`_Denoiser.plan`)."""
    from dew.interop import diffusion

    weights = _component_weights(directory, "transformer", functools.partial(diffusion.read_transformer,
                                                                             path_of=path_of))
    return weights, functools.partial(diffusion.transformer_layouts, path_of=path_of)


def _component_weights(directory: Path, component: str,
                       read: Callable[..., tuple[LazyTree, tuple[WeightLayout, ...]]]
                       ) -> Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]]:
    """A denoiser's `weights(param_dtype, lazy)`: its `component`'s tensors read by `read`, the
    parameters under `params`, beside their layouts."""
    from dew.interop import diffusion

    def weights(param_dtype: str, lazy: bool) -> tuple[Variables, tuple[WeightLayout, ...]]:
        params, layouts = read(diffusion.component_tensors(directory, component), param_dtype=param_dtype,
                               lazy=lazy)
        return {"params": params}, layouts
    return weights


def _transformer_denoiser(name: str, model: nn.Module, fields: Mapping[str, object], config: dict,
                         dtype: str | None,
                         weights: Callable[[str, bool], tuple[Variables, tuple[WeightLayout, ...]]],
                         plan: Callable[[Mapping[str, tuple[int, ...]]], tuple[WeightLayout, ...]],
                         text: _TextTowers | _QwenImageText | _HiddenStatesText | _WanText, *,
                         patch: int, latent_input: int, sample_size: tuple[int, int], context_width: int,
                         pipeline: str, origin: Origin = "scheduler", frames: int | None = None) -> _Denoiser:
    """Build the native record with the pipeline's already resolved conditioning and geometry."""
    built = {"class": name, "fields": {**{key: list(value) if isinstance(value, tuple) else value
                                         for key, value in fields.items()}, "dtype": dtype}}
    return _Denoiser(
        component="transformer", model=model, weights=weights, plan=plan, built=built, config=config,
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
        "sd3_transformer", model, fields, config, dtype, weights, diffusion.sd3_layouts,
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
    weights, plan = _transformer_source(directory, diffusion.flux_path)
    return _transformer_denoiser(
        "flux_transformer", model, fields, config, dtype, weights, plan,
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
        *_transformer_source(directory, diffusion.qwen_image_path), _QwenImageText(),
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
    weights, plan = _transformer_source(directory, diffusion.flux2_path)
    return _transformer_denoiser(
        "flux2_transformer", model, fields, config, dtype, weights, plan,
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
    weights, plan = _transformer_source(directory, diffusion.z_image_path)
    return _transformer_denoiser(
        "z_image_transformer", model, fields, config, dtype, weights, plan,
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
    weights, plan = _transformer_source(directory, diffusion.wan_path)
    return _transformer_denoiser(
        "wan_transformer", model, fields, config, dtype, weights, plan,
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


def text_components(text: _TextTowers | _QwenImageText | _HiddenStatesText | _WanText,
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
    side = _records.integer(config["sample_size"] if default is None else config.get("sample_size", default),
                           "sample_size")
    return side, side


def _geometry(index: Mapping[str, object], size: tuple[int, int]) -> tuple[int, int]:
    """The (height, width) a pipeline is bound to: the index's own, or `size`."""
    height, width = index.get("dew_height", size[0]), index.get("dew_width", size[1])
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("Image geometry must contain positive integer dimensions")
    return height, width


def component_present(index: Mapping[str, object], name: str) -> bool:
    """Return whether the index declares a component rather than declaring it absent."""
    entry = index.get(name)
    return isinstance(entry, list) and entry[0] is not None


def _shapes(stored: StoredWeights | None, folder: str) -> dict[str, tuple[int, ...]] | None:
    """The shapes of `folder`'s stored tensors, by their headers, where a pipeline
    bound from saved variables writes its export through its source's layouts
    (`StoredWeights`); None for a pipeline read from its weights, or where the
    headers cannot be read offline."""
    held = None if stored is None else stored.specs(folder)
    return None if held is None else {name: spec.shape for name, (_, spec) in held.items()}


def _diffusion_vae(directory: Path, compute, *, param_dtype: str = "float32",
                   params: Variables | None = None, lazy: bool = False, stored: StoredWeights | None = None
                   ) -> tuple[AutoEncoder, Variables, tuple[WeightLayout, ...], dict]:
    """Build the published autoencoder `vae/` holds, as `LOADERS` names its
    class and an AutoencoderKL otherwise, with its parameters and their source
    layouts; supplied `params` are bound without a weight read, and `lazy`
    leaves read ones `SourceLeaf`s."""
    from dew.nn.autoencoders.pretrained import LOADERS, load_kl_vae

    load = LOADERS.get(_component_config(directory, "vae").get("_class_name", ""), load_kl_vae)
    return load(directory, compute, subfolder="vae", param_dtype=param_dtype, params=params, lazy=lazy,
                shapes=_shapes(stored, "vae"))


def _clip_towers(directory: Path, names: tuple[str, ...], compute, *, param_dtype: str = "float32",
                 params: Variables | None = None, lazy: bool = False, stored: StoredWeights | None = None):
    """Build the published CLIP text towers, their tokenizers, their parameters, the layouts those
    parameters came from and the tokenizers' files, each pinned as it is loaded."""
    from transformers import CLIPTokenizer

    from dew.data.text import pin_tokenizer
    from dew.nn.text_encoders import translate_config

    towers, tokenizers, pins, layouts = [], [], [], ()
    bound = {} if params is None else params
    for name in names:
        config = _component_config(directory, name)
        model = translate_config(config).value.clone(dtype=compute)
        tower, tree, recorded, _ = bind_component(
            directory / name, name, config, model, lambda name, rank: _text_head_path(name),
            lambda bound, model=model: model, prefix=("encoders", "conditioning", name),
            params=params, param_dtype=param_dtype, lazy=lazy, shapes=_shapes(stored, name))
        towers.append(tower)
        if params is None:
            bound = {**bound, name: tree}
        layouts += recorded
        pins.append(pin_tokenizer(str(directory / ("tokenizer" + name.removeprefix("text_encoder")))))
        tokenizers.append(CLIPTokenizer.from_pretrained(pins[-1].directory))
    return tuple(towers), tuple(tokenizers), bound, layouts, tuple(pins)


def _t5_tower(directory: Path, compute, component: str, tokens: int, *, param_dtype: str = "float32",
              params: Variables | None = None, lazy: bool = False, stored: StoredWeights | None = None):
    """Build the published T5 encoder as the conditioner's segment, with its
    parameters, their layouts and its config.

    `component` is where the family keeps it: an SD3 directory's third text
    encoder, a Flux directory's second one; `tokens` is the sequence budget
    the pipeline pads to.
    """
    from dew.data.text import pin_tokenizer
    from dew.nn.text_encoders import _t5_path, t5_embedding, translate_t5_config

    config = _component_config(directory, component)
    tower = translate_t5_config(config).value.clone(dtype=compute)
    pinned = pin_tokenizer(str(directory / ("tokenizer" + component.removeprefix("text_encoder"))))
    return bind_component(
        directory / component, component, config, tower, lambda name, rank: _t5_path(name),
        lambda bound: T5Segment(tower, pinned.transformers(), component, tokens, files=pinned),
        prefix=("encoders", "conditioning", component), params=params,
        param_dtype=param_dtype, lazy=lazy, validate=t5_embedding, shapes=_shapes(stored, component))


def _wan_conditioning(directory: Path, compute, *, tokens: int, param_dtype: str,
                      params: Variables | None = None, lazy: bool = False, stored: StoredWeights | None = None
                      ) -> tuple[WanConditioner, tuple[WeightLayout, ...], dict[str, Mapping[str, object]]]:
    """Build Wan's conditioner: the UMT5 encoder, its tokenizer, the
    parameters and their layouts."""
    from dew.data.text import pin_tokenizer
    from dew.nn.text_encoders import _t5_path, t5_embedding, translate_t5_config

    config = _component_config(directory, "text_encoder")
    if config.get("model_type") != "umt5":
        raise ValueError(f"Wan's text encoder is a umt5 model, not {config.get('model_type')!r}")
    tower = translate_t5_config(config).value.clone(dtype=compute)
    pinned = pin_tokenizer(str(directory / "tokenizer"))
    encoder, _, layouts, _ = bind_component(
        directory / "text_encoder", "text_encoder", config, tower, lambda name, rank: _t5_path(name),
        lambda bound: WanConditioner(tower, pinned.transformers(),
                                     bound if params is not None else {"text_encoder": bound},
                                     str(directory), tokens=tokens, param_dtype=param_dtype,
                                     tokenizer_files=pinned),
        prefix=("encoders", "conditioning", "text_encoder"), params=params,
        param_dtype=param_dtype, lazy=lazy, validate=t5_embedding, shapes=_shapes(stored, "text_encoder"))
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
    text = dict(_records.record(config["text_config"], "text_config"))
    if text.get("model_type") != "qwen3_vl_text":
        raise ValueError(f"A qwen3_vl text_config is qwen3_vl_text, not {text.get('model_type')!r}")
    half = _records.integer(text["head_dim"], "head_dim") // 2
    for key in ("rope_parameters", "rope_scaling"):
        entry = text.get(key)
        if isinstance(entry, Mapping):
            entry = dict(entry)
            section = _records.integers(entry.pop("mrope_section"), "mrope_section")
            _records.boolean(entry.pop("mrope_interleaved", False), "mrope_interleaved")
            if sum(section) != half:
                raise ValueError(f"mrope_section {section} must cover the {half} channel pairs")
            text[key] = entry
    return {**text, "model_type": "qwen3",
            "tie_word_embeddings": _records.boolean(config.get("tie_word_embeddings", False),
                                                   "tie_word_embeddings")}


_TEXT_ENCODER = ("encoders", "conditioning", "text_encoder")
"""Where a pipeline holds its text encoder's variables."""


def _bind_language_model[C: ConditionEncoder](
        directory: Path, record: decoder_parts.DecoderFields, family: str, rename: Callable[[str], str],
        conditioner: Callable[[Variables], C], *, vision: tuple[str, ...], param_dtype: str,
        params: Variables | None, lazy: bool, stored: StoredWeights | None = None
        ) -> tuple[C, tuple[WeightLayout, ...], Mapping[str, np.ndarray]]:
    """Bind a language-model text encoder into the conditioner its pipeline runs:
    supplied `params` as they are, or the `text_encoder` component's tensors
    read as their decoder `family` reads a decoder's checkpoint
    (`hf_decoders.read_decoder`), `rename` giving each tensor its name there,
    with their layouts in the pipeline under every name they are stored by
    (`decoder_bindings.decoder_layouts`), an equal alias's too. Supplied
    `params` take their layouts from the same reading of the stored tensors'
    headers (`StoredWeights`), over arrays that hold no bytes, where the
    pipeline has its source; none without it.

    The tensors under a `vision` prefix are a vision tower no text prompt
    reads, so they stay out of the variables and come back as stored, under
    their export names, for a source export to write back
    (`Pretrained.retained_tensors`); with supplied `params`, read from the
    source when the export reads them.
    """
    from dew.interop import diffusion

    def read(tensors: Mapping[str, np.ndarray], *, lazy: bool
             ) -> tuple[Variables, tuple[WeightLayout, ...], list[str]]:
        """The decoder's variables, the bindings of `tensors` under every name they
        are stored by, and the names no leaf takes, the vision tower's among them."""
        # A checkpoint may store one tensor under two names (a Mistral-3 under both
        # its spellings): equal copies read once, and each name keeps its binding.
        aliases: dict[str, list[str]] = {}
        owners: dict[tuple[str, ...], str] = {}
        for name in tensors:
            if not name.startswith(vision):
                local = rename(name)
                source_alias(tensors, owners, (local,), name)
                aliases.setdefault(local, []).append(name)
        variables, readings = decoders.read_decoder(
            {local: tensors[names[0]] for local, names in aliases.items()}, record, family,
            param_dtype=param_dtype, lazy=lazy)
        layouts, kept = decoder_layouts(readings, record, family, variables,
                                        place=lambda path: (*_TEXT_ENCODER, *path))
        bound = tuple(replace(layout, name=f"text_encoder/{name}")
                      for layout in layouts for name in aliases[layout.name])
        return variables, bound, [name for local in kept for name in aliases[local]] + [
            name for name in tensors if name.startswith(vision)]

    if params is None:
        tensors = diffusion.component_tensors(directory, "text_encoder")
        variables, bound, unread = read(tensors, lazy=lazy)
        return (conditioner({"text_encoder": variables}), bound,
                {f"text_encoder/{name}": tensors[name] for name in unread})
    if stored is None or (held := stored.specs("text_encoder")) is None:
        return conditioner(params), (), {}
    _, bound, unread = read({name: np.broadcast_to(np.zeros((), spec.dtype), spec.shape)
                             for name, (_, spec) in held.items()}, lazy=True)
    source = stored.tensors({name: held[name] for name in unread})
    return conditioner(params), bound, LazyTensors(
        {f"text_encoder/{name}": source.specs[name] for name in unread},
        lambda name: source[name.removeprefix("text_encoder/")])


def _qwen_text_name(name: str) -> str:
    """A Qwen3-VL checkpoint's language-model tensor under the Qwen3 decoder's own name, its head too."""
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name == "lm_head.weight":
        return name
    raise ValueError(f"unknown tensor name {name!r}")


def _qwen_image_conditioning(directory: Path, index: Mapping[str, object], compute, *,
                             tokens: int, param_dtype: str, attention_impl: str,
                             params: Variables | None = None, lazy: bool = False,
                             stored: StoredWeights | None = None
                             ) -> tuple[QwenImageConditioner, tuple[WeightLayout, ...],
                                        Mapping[str, np.ndarray], dict[str, Mapping[str, object]]]:
    """Build Qwen-Image's conditioner: the Qwen3-VL language model, its
    processor's tokenizer and chat template, the parameters, their layouts
    and the vision tower as stored, which the text-to-image prompt never
    reads and an export writes back."""
    from dew.data.text import pin_tokenizer

    config = _component_config(directory, "text_encoder")
    record = decoders.translate_config(_qwen_vl_text_config(config))
    pinned = pin_tokenizer(str(directory / "processor"))
    named = dtype_name(compute)
    if named is None:
        raise ValueError("Qwen-Image's Qwen3-VL encoder computes in a named dtype; pass dtype")
    decoder = from_record(CausalTransformer, {**record, "dtype": named, "attention_impl": attention_impl})
    encoder, layouts, retained = _bind_language_model(
        directory, record, "qwen3", _qwen_text_name,
        lambda bound: QwenImageConditioner(decoder, pinned.transformers(), bound, str(directory),
                                           tokens=tokens, param_dtype=param_dtype, tokenizer_files=pinned),
        vision=("model.visual.",), param_dtype=param_dtype, params=params, lazy=lazy, stored=stored)
    return encoder, layouts, retained, {"text_encoder": config}


_FLUX2_TEXT: Mapping[str, tuple[Literal["qwen3", "mistral3"], tuple[int, ...]]] = MappingProxyType({
    "qwen3": ("qwen3", (9, 18, 27)), "mistral3": ("mistral3", (10, 20, 30))})
"""Each FLUX.2 text encoder's `model_type`, the template its pipeline
formats a prompt with, and the `hidden_states` it stacks."""

_MISTRAL3_VISION = ("vision_tower.", "multi_modal_projector.", "model.vision_tower.",
                    "model.multi_modal_projector.")
"""Where a Mistral-3 checkpoint keeps its vision tower and projector: at its
root beside `language_model.model.`, or under `model.` beside
`model.language_model.` (`_hidden_states_name`)."""


def _hidden_states_name(multimodal: bool) -> Callable[[str], str]:
    """Name a hidden-states text encoder's tensor as its decoder's checkpoint
    does: a Qwen3's own names, or a Mistral-3's language model as the
    Mistral decoder's. A Mistral-3 checkpoint names its language model
    `language_model.model.` and its head `language_model.lm_head` (transformers
    4.50 and 5 write these), or `model.language_model.` and `lm_head` (4.52 to
    4.57)."""

    def name_of(name: str) -> str:
        if not multimodal or name == "lm_head.weight":
            return name
        if name == "language_model.lm_head.weight":
            return "lm_head.weight"
        for prefix in ("model.language_model.", "language_model.model."):
            if name.startswith(prefix):
                return "model." + name.removeprefix(prefix)
        raise ValueError(f"unknown tensor name {name!r}")
    return name_of


def _hidden_states_conditioning(directory: Path, index: Mapping[str, object], compute, *,
                                pipeline: Literal["flux2", "z_image"], tokens: int, guidance: float | None,
                                param_dtype: str, attention_impl: str, params: Variables | None = None,
                                lazy: bool = False, stored: StoredWeights | None = None
                                ) -> tuple[HiddenStatesConditioner, tuple[WeightLayout, ...],
                                           Mapping[str, np.ndarray], dict[str, Mapping[str, object]]]:
    """Build FLUX.2's or Z-Image's conditioner: the text encoder's language
    model, the tokenizer and chat template, the parameters, their layouts and
    a Mistral-3's vision tower and projector as stored, which a text prompt
    never reads and an export writes back. Z-Image reads the output of the
    encoder's second-to-last layer with the template's thinking on."""
    from dew.data.text import pin_tokenizer

    config = _component_config(directory, "text_encoder")
    kind = _records.text(config.get("model_type"), "model_type")
    pinned = pin_tokenizer(str(directory / "tokenizer"))
    known = _FLUX2_TEXT if pipeline == "flux2" else {"qwen3": _FLUX2_TEXT["qwen3"]}
    if kind not in known:
        raise ValueError(f"The {pipeline} text encoder is one of {sorted(known)}, not {kind!r}")
    template, layers = known[kind]
    multimodal = kind == "mistral3"
    text = dict(_records.record(config["text_config"], "text_config")) if multimodal else config
    if multimodal:
        text["tie_word_embeddings"] = _records.boolean(config.get("tie_word_embeddings", False),
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
    encoder, layouts, retained = _bind_language_model(
        directory, record, _records.text(text["model_type"], "model_type"), _hidden_states_name(multimodal),
        lambda bound: HiddenStatesConditioner(
            decoder, pinned.transformers(), bound, str(directory),
            template=template, layers=layers, thinking=pipeline == "z_image", tokens=tokens,
            guidance=guidance, param_dtype=param_dtype, tokenizer_files=pinned),
        vision=_MISTRAL3_VISION if multimodal else (), param_dtype=param_dtype, params=params, lazy=lazy,
        stored=stored)
    return encoder, layouts, retained, {"text_encoder": config}


def _image_safety(directory: Path, compute, *, param_dtype: str = "float32",
                  params: Variables | None = None, lazy: bool = False, stored: StoredWeights | None = None):
    """Build the safety head a file declares: the finish, its parameters, their
    layouts and the two configs it ships; supplied `params` are bound without
    a weight read."""
    from dew.inputs.diffusion import CLIPImageTransform, CLIPSafetyHead, ImageSafety
    from dew.nn.text_encoders import translate_vision_config

    config = _component_config(directory, "safety_checker")
    with open(directory / "feature_extractor" / "preprocessor_config.json") as handle:
        transform = json.load(handle)
    head = CLIPSafetyHead(translate_vision_config(config).value.clone(dtype=compute),
                          int(config["projection_dim"]), dtype=compute)
    finish, params, layouts, _ = bind_component(
        directory / "safety_checker", "safety_checker", config, head, lambda name, rank: _safety_path(name),
        lambda bound: ImageSafety(head, CLIPImageTransform.from_config(transform)),
        prefix=("encoders", "safety"), params=params, param_dtype=param_dtype, lazy=lazy,
        # The root scoring vectors and thresholds, the head's only one-name paths.
        state=lambda name: (path := _safety_path(name)) is not None and len(path) == 1,
        shapes=_shapes(stored, "safety_checker"))
    return finish, params, layouts, {"safety_checker": config, "feature_extractor": transform}


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


class PipelineParts(TypedDict):
    """What `assemble_pipeline` builds, the fields of a `PretrainedPipeline` but its processor."""

    model: nn.Module
    variables: Variables
    config: Mapping[str, object]
    source: Path | None
    model_config: Mapping[str, object]
    weight_layouts: tuple[WeightLayout, ...]
    retained_tensors: Mapping[str, np.ndarray]
    process: Process
    inputs: InputSpec
    autoencoder: AutoEncoder | None
    schedule: SourceSchedule
    finish: Callable[[Mapping[str, object], jax.Array], jax.Array] | None
    task: SourceTask
