"""The KL autoencoder as a release's diffusers config states it (`AutoencoderKL`, `FlaxAutoencoderKL`):
the model `dew.nn.autoencoders.kl` computes, built from the config's fields, a field asking for anything
else refused by name."""

from __future__ import annotations

from collections.abc import Mapping

from flax.typing import Dtype

from dew import _records
from dew.nn.autoencoders.kl import AutoencoderKL


def autoencoder_kl_from_diffusers(config: Mapping[str, object], dtype: Dtype) -> AutoencoderKL:
    """The model a diffusers `AutoencoderKL` config describes, computing in `dtype`. The stack
    computes SiLU, attention in the bottleneck, a decoder writing the image's channels and the
    plain encoder and decoder levels; a config asking for anything else is refused by name."""
    # The diffusers field reader, whose module reads the diffusion backbones: a run configuration that
    # names an autoencoder reads none (tests/test_imports.py's FOREIGN).
    from dew.interop.diffusion import flag

    channels = _records.integers(config["block_out_channels"], "block_out_channels")
    refused = [name for name, holds in (
        ("act_fn", config.get("act_fn", "silu") == "silu"),
        ("mid_block_add_attention", flag(config, "mid_block_add_attention", default=True)),
        ("out_channels", config.get("out_channels", 3) == config["in_channels"]),
        *((field, _records.strings(config.get(field) or [block] * len(channels), field)
           == (block,) * len(channels))
          for field, block in (("down_block_types", "DownEncoderBlock2D"),
                               ("up_block_types", "UpDecoderBlock2D"))),
    ) if not holds]
    if refused:
        named = ", ".join(f"{name}={config.get(name)!r}" for name in refused)
        raise ValueError(f"the KL autoencoder does not compute {named}")
    decoder = config.get("decoder_block_out_channels")
    return AutoencoderKL(channels=channels,
                         latent_channels=_records.integer(config["latent_channels"], "latent_channels"),
                         image_channels=_records.integer(config["in_channels"], "in_channels"),
                         blocks_per_level=_records.integer(config["layers_per_block"], "layers_per_block"),
                         norm_groups=_records.integer(config["norm_num_groups"], "norm_num_groups"),
                         quantize=flag(config, "use_quant_conv", default=True),
                         post_quantize=flag(config, "use_post_quant_conv", default=True),
                         decoder_channels=(None if not decoder
                                           else _records.integers(decoder, "decoder_block_out_channels")),
                         dtype=dtype)


def published_kl(config: Mapping[str, object], dtype: Dtype) -> tuple[AutoencoderKL, float, float]:
    """An AutoencoderKL config's model, computing in `dtype`, and its latent shift and scale: its
    `shift_factor` (0 where absent or null) and `scaling_factor` (SD1's 0.18215 where absent or null), built
    alike by a standalone load and a pipeline's (`load_kl_vae`)."""
    shift, scale = config.get("shift_factor") or 0.0, config.get("scaling_factor")
    return (autoencoder_kl_from_diffusers(config, dtype), _records.number(shift, "shift_factor"),
            0.18215 if scale is None else _records.number(scale, "scaling_factor"))
