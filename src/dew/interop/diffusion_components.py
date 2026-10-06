"""Build a published diffusion pipeline's components from its directory.

Each family's denoiser, the text conditioning it reads, its autoencoder and
the safety head a file declares are built here, from metadata alone or bound
to their weights; `dew.interop.diffusion_pipelines` assembles them into a
pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal

from flax import linen as nn

from dew import records
from dew.diffusion.schedules.source import Origin
from dew.inputs.diffusion import (
    Composition,
    DiffusionConditioner,
    HiddenStatesConditioner,
    QwenImageConditioner,
    T5Segment,
    WanConditioner,
)
from dew.interop import hf_decoders as decoders, weights as checkpoint_weights
from dew.interop.components import bind_component
from dew.interop.streaming import LazyTree, WeightLayout
from dew.nn.autoencoders import AutoEncoder
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Variables
from dew.registry import dtype_name, from_record, with_precision

if TYPE_CHECKING:
    from dew.interop.diffusion_pipelines import _Call


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
    built = {"name": "unet_2d_condition",
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
    built = {"name": name, "fields": {**{key: list(value) if isinstance(value, tuple) else value
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
    built = with_precision("causal_transformer", record, dtype=named, attention_impl=attention_impl)
    decoder = from_record(CausalTransformer, built)
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
    decoder = from_record(CausalTransformer, with_precision("causal_transformer", record, dtype=named,
                                                            attention_impl=attention_impl))
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
