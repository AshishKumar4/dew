"""Load a published latent diffusion pipeline as native modules and variables.

`load_diffusion_source` reads a diffusers directory or repo into a
`PretrainedPipeline`, and `load_diffusion_conditioner` reads its text
conditioning alone. The call policy each published pipeline carries is kept
here; the components a pipeline is built from come from
`dew.interop.diffusion_components`.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax

from dew import records
from dew.diffusion.process import Process
from dew.diffusion.schedules.source import SourceSchedule
from dew.inference.pipeline import place
from dew.inputs import Condition, Field, InputSpec
from dew.inputs.diffusion import (
    DiffusionConditioner,
    HiddenStatesConditioner,
    QwenImageConditioner,
    WanConditioner,
)
from dew.interop import sources
from dew.interop.diffusion_components import (
    _component_config,
    _Denoiser,
    _denoiser,
    _diffusion_vae,
    _geometry,
    _image_safety,
    _present,
    _text_components,
    _TextTowers,
)
from dew.objectives.base import Variables
from dew.registry import resolve_dtype
from dew.sampling.guidance import CFG

if TYPE_CHECKING:
    from dew.interop.pretrained import PretrainedPipeline
    from dew.training.distributed import Layout, MeshSpec


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
    from dew.interop.pretrained import PretrainedPipeline

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


def load_diffusion_conditioner[C: (DiffusionConditioner, QwenImageConditioner, HiddenStatesConditioner,
                                   WanConditioner)](
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
