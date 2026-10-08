"""Build the convolutional encoder and decoder of a diffusion VAE.

The modules are independent linen ports of huggingface/diffusers v0.29.2
src/diffusers/models/vae_flax.py (Apache-2.0). The loader below reads source
configuration and checkpoint files; no external model implementation runs.
"""

import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from dew.interop.weights import ParamTree, translate_parameters
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.conv import Conv


def _conv(features: int, dtype, name: str) -> Conv:
    """A 3x3 convolution padded to keep the size."""
    return Conv(features, (3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), dtype=dtype, name=name)


def _group_norm(groups: int, dtype, name: str) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=groups, epsilon=1e-6, dtype=dtype, name=name)


class FlaxUpsample2D(nn.Module):
    """Double each spatial axis by nearest-neighbour resize, then a 3x3 conv."""

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        return _conv(self.in_channels, self.dtype, "conv")(hidden_states)


class FlaxDownsample2D(nn.Module):
    """Halve each spatial axis with a stride-2 3x3 conv, padded on the far side."""

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        return Conv(self.in_channels, kernel_size=(3, 3), strides=(2, 2), padding="VALID", dtype=self.dtype,
                    name="conv")(jnp.pad(hidden_states, pad_width=pad))


class FlaxResnetBlock2D(nn.Module):
    """Run two group-normed 3x3 convolutions and add the input back, through
    a 1x1 convolution when the block changes the channel count."""

    in_channels: int
    out_channels: int
    groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = nn.swish(_group_norm(self.groups, self.dtype, "norm1")(hidden_states))
        hidden_states = _conv(self.out_channels, self.dtype, "conv1")(hidden_states)
        hidden_states = nn.swish(_group_norm(self.groups, self.dtype, "norm2")(hidden_states))
        hidden_states = _conv(self.out_channels, self.dtype, "conv2")(hidden_states)
        if self.in_channels != self.out_channels:
            residual = Conv(self.out_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID",
                            dtype=self.dtype, name="conv_shortcut")(residual)
        return hidden_states + residual


class FlaxAttentionBlock(nn.Module):
    """Attend over an image's pixels as a sequence, with a residual.

    One head as wide as the channels, which is what every AutoencoderKL
    mid-block runs, through the shared attention core
    (`dew.nn.attention.scaled_dot_product_attention`), which picks the kernel
    and runs the softmax in fp32. Its logits are scaled by 1/sqrt(channels),
    the scale diffusers splits over the query and the key: the same function,
    rounded in another order.
    """

    channels: int
    num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape
        hidden_states = _group_norm(self.num_groups, self.dtype, "group_norm")(hidden_states)
        hidden_states = hidden_states.reshape((batch, height * width, channels))
        query, key, value = (nn.Dense(self.channels, dtype=self.dtype, name=name)(hidden_states)[:, :, None]
                             for name in ("query", "key", "value"))
        hidden_states = scaled_dot_product_attention(query, key, value)[:, :, 0]

        hidden_states = nn.Dense(self.channels, dtype=self.dtype, name="proj_attn")(hidden_states)
        return hidden_states.reshape((batch, height, width, channels)) + residual


class _Level(nn.Module):
    """One encoder or decoder level: `num_layers` resnets to `out_channels`,
    only the first changing the channel count, then the level's resampling,
    `downsamplers_0` or `upsamplers_0`, which the last level does without."""

    in_channels: int
    out_channels: int
    num_layers: int
    groups: int
    resample: Literal["down", "up"] | None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        for index in range(self.num_layers):
            hidden_states = FlaxResnetBlock2D(self.in_channels if index == 0 else self.out_channels,
                                              self.out_channels, self.groups, self.dtype,
                                              name=f"resnets_{index}")(hidden_states)
        if self.resample == "down":
            return FlaxDownsample2D(self.out_channels, dtype=self.dtype, name="downsamplers_0")(hidden_states)
        if self.resample == "up":
            return FlaxUpsample2D(self.out_channels, dtype=self.dtype, name="upsamplers_0")(hidden_states)
        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    """A resnet, an attention block and a second resnet at the bottleneck
    resolution. The channel count does not change."""

    in_channels: int
    resnet_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        resnet = partial(FlaxResnetBlock2D, self.in_channels, self.in_channels, self.resnet_groups,
                         self.dtype)
        hidden_states = resnet(name="resnets_0")(hidden_states)
        hidden_states = FlaxAttentionBlock(self.in_channels, self.resnet_groups, self.dtype,
                                           name="attentions_0")(hidden_states)
        return resnet(name="resnets_1")(hidden_states)


class FlaxEncoder(nn.Module):
    """Encode images to latent moments with a conv stack that halves each axis.

    `block_out_channels` gives one level per entry, each `layers_per_block`
    resnets; the last level does not downsample. The output carries twice
    `out_channels`, which the caller splits into mean and log-variance.
    """

    out_channels: int = 3
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, sample):
        channels = self.block_out_channels
        sample = _conv(channels[0], self.dtype, "conv_in")(sample)
        for index, width in enumerate(channels):
            sample = _Level(channels[max(index - 1, 0)], width, self.layers_per_block, self.norm_num_groups,
                            "down" if index != len(channels) - 1 else None, self.dtype,
                            name=f"down_blocks_{index}")(sample)
        sample = FlaxUNetMidBlock2D(channels[-1], self.norm_num_groups, self.dtype, name="mid_block")(sample)
        sample = nn.swish(_group_norm(self.norm_num_groups, self.dtype, "conv_norm_out")(sample))
        return _conv(2 * self.out_channels, self.dtype, "conv_out")(sample)


class FlaxDecoder(nn.Module):
    """Decode latents to images with a conv stack that doubles each axis.

    `block_out_channels` is read in reverse, one level per entry, each
    `layers_per_block + 1` resnets; the last level does not upsample.
    """

    out_channels: int = 3
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, sample):
        channels = list(reversed(self.block_out_channels))
        sample = _conv(channels[0], self.dtype, "conv_in")(sample)
        sample = FlaxUNetMidBlock2D(channels[0], self.norm_num_groups, self.dtype, name="mid_block")(sample)
        for index, width in enumerate(channels):
            sample = _Level(channels[max(index - 1, 0)], width, self.layers_per_block + 1,
                            self.norm_num_groups, "up" if index != len(channels) - 1 else None, self.dtype,
                            name=f"up_blocks_{index}")(sample)
        sample = nn.swish(_group_norm(self.norm_num_groups, self.dtype, "conv_norm_out")(sample))
        return _conv(self.out_channels, self.dtype, "conv_out")(sample)


_VAE_ATTENTION = {"to_q": "query", "to_k": "key", "to_v": "value"}


def _vae_path(torch_name: str, rank: int) -> tuple[str, ...]:
    """Map one diffusers AutoencoderKL tensor name to its path in this tree.

    Three names differ. diffusers numbers repeated children with a dot
    (`down_blocks.0`) where linen uses an underscore (`down_blocks_0`). It
    calls the mid-block attention's projections `to_q/to_k/to_v/to_out.0`
    where `FlaxAttentionBlock` calls them `query/key/value/proj_attn`. It
    stores every gain as `weight`, where linen has a norm's `scale` and a
    convolution's or a dense's `kernel`, which `rank` tells apart.
    """
    parts = torch_name.split(".")
    leaf = "bias" if parts[-1] == "bias" else ("scale" if rank == 1 else "kernel")
    parts = parts[:-1]
    path = []
    index = 0
    while index < len(parts):
        name = parts[index]
        following = parts[index + 1] if index + 1 < len(parts) else None
        if name == "to_out":
            # to_out.0 is the projection; there is no second entry.
            path.append("proj_attn")
            index += 2
            continue
        if name in _VAE_ATTENTION:
            path.append(_VAE_ATTENTION[name])
            index += 1
            continue
        if following is not None and following.isdigit():
            path.append(f"{name}_{following}")
            index += 2
            continue
        path.append(name)
        index += 1
    return (*path, leaf)


def translate_vae_weights(torch_tensors: ParamTree) -> ParamTree:
    """Convert diffusers AutoencoderKL tensors into this module's param tree.

    Convolution kernels transpose from torch's [out, in, kh, kw] to linen's
    [kh, kw, in, out], and dense kernels from [out, in] to [in, out]; norms
    and biases keep their layout. Every tensor maps, so an unknown name
    raises rather than loading half an autoencoder.
    """
    def stored(name: str) -> np.ndarray:
        tensor = torch_tensors[name]
        if isinstance(tensor, dict):
            raise ValueError(f"{name} is a subtree; a diffusers VAE table is flat")
        return np.asarray(tensor, dtype=np.float32)

    tensors = {name: stored(name) for name in torch_tensors}
    return translate_parameters(tensors, lambda name: _vae_path(name, tensors[name].ndim))


# The SD1-era flax weights live on their own branches, so these name a layout,
# not a pinned revision.
FLAX_REVISIONS = ("bf16", "flax")


FLAX_WEIGHTS = "diffusion_flax_model.msgpack"
TORCH_WEIGHTS = "diffusion_pytorch_model.safetensors"

Candidate = tuple[str, str | None, str | None]
"""A place a VAE may live on the Hub: its weight file, revision and subfolder."""


def _candidates(revision: str) -> list[Candidate]:
    """Every place a load looks, in the order it looks.

    First the SD1-era flax msgpack: the revision the caller named and the
    `flax` branch, each with and without the `vae` subfolder. Then the torch
    safetensors: a revision outside `FLAX_REVISIONS` is a pin, so only that
    revision, and otherwise the repo's default branch.
    """
    torch = ([(revision, "vae"), (revision, None)] if revision not in FLAX_REVISIONS
             else [(None, "vae"), (None, None)])
    return [
        (FLAX_WEIGHTS, *place)
        for place in [(revision, "vae"), ("flax", "vae"), (revision, None), (None, None)]
    ] + [(TORCH_WEIGHTS, *place) for place in torch]


def _check_weights(modelname: str, candidates: list[Candidate], index: int) -> None:
    """Raise EntryNotFoundError when `candidates[index]` ships no weights, reading no weight bytes.

    Supplied params make the file's presence the only question, which a
    dry run answers online. With the Hub offline (HF_HUB_OFFLINE) the cache
    answers it: a cached file, or the Hub's cached record that the file is
    missing, settles it. When the cache knows neither, the candidate is
    taken only if no later candidate, in either layout, has a different
    cached config that could be the one an online load chose; otherwise
    which config applies cannot be told offline, and a FileNotFoundError
    says so rather than guessing.
    """
    from huggingface_hub import constants, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import EntryNotFoundError

    weights, revision, subfolder = candidates[index]
    if not constants.HF_HUB_OFFLINE:
        hf_hub_download(modelname, weights, revision=revision, subfolder=subfolder, dry_run=True)
        return

    def cached(name: str, place: Candidate):
        """The cached path (str), the Hub's cached miss (another object) or unknown (None)."""
        _, place_revision, place_subfolder = place
        return try_to_load_from_cache(
            modelname,
            name if place_subfolder is None else f"{place_subfolder}/{name}",
            revision=place_revision,
        )

    def recorded_missing(place: Candidate) -> bool:
        found = cached(place[0], place)
        return found is not None and not isinstance(found, str)

    candidate = candidates[index]
    if isinstance(cached(weights, candidate), str):
        return
    if recorded_missing(candidate):
        raise EntryNotFoundError(f"{modelname} has no {weights} at {candidate[1:]}, as the cache records")
    # A later candidate naming the same cached config (None is the default branch) is no rival.
    here = cached("config.json", candidate)
    rivals = [place[1:] for place in candidates[index + 1:]
              if isinstance(cached("config.json", place), str)
              and cached("config.json", place) != here and not recorded_missing(place)]
    if rivals:
        raise FileNotFoundError(
            f"offline, the cache cannot tell which VAE config of {modelname} applies: {weights} at "
            f"{candidate[1:]} is neither cached nor recorded missing, and {rivals} have cached configs too; "
            "load once with the Hub online, or put the weight file in the cache")


def _load_from_hub(modelname: str, revision: str, params, errors: list) -> dict | None:
    """The first candidate with a config and weights, as `load_pretrained_vae`
    reads it, or None. Every miss is appended to `errors`, so the caller can
    raise the last one."""
    from flax.serialization import msgpack_restore
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

    candidates = _candidates(revision)
    for index, (weights, candidate_revision, subfolder) in enumerate(candidates):
        try:
            config_path = hf_hub_download(modelname, "config.json",
                                          revision=candidate_revision, subfolder=subfolder)
            # Candidate eligibility must match a source load even when a
            # config exists in a revision that carries no matching weights.
            if params is not None:
                _check_weights(modelname, candidates, index)
            weights_path = None if params is not None else hf_hub_download(
                modelname, weights, revision=candidate_revision, subfolder=subfolder)
        except (EntryNotFoundError, RevisionNotFoundError) as e:
            errors.append(e)
            continue
        with open(config_path) as handle:
            config = json.load(handle)
        if weights_path is None:
            return {"config": config, "params": params}
        if weights == FLAX_WEIGHTS:
            with open(weights_path, "rb") as handle:
                return {"config": config, "params": msgpack_restore(handle.read())}
        return {"config": config, "params": _read_vae_weights(Path(config_path).parent)}
    return None


def load_pretrained_vae(modelname: str, revision: str = "bf16", *, params=None) -> dict:
    """Read a pretrained AutoencoderKL's config and params, local or from the Hub.

    Two weight layouts reach the same tree. The SD1-era repos ship flax
    `diffusion_flax_model.msgpack`, sometimes under a `vae` subfolder on a
    `flax`/`bf16` revision. Every 16-channel VAE (SD3.5, Flux) ships torch
    `diffusion_pytorch_model.safetensors` only, which `translate_vae_weights`
    reads. Supplied `params` are authoritative: only the configuration and
    the repository's file metadata are read then, never weight bytes.

    `revision` names the flax layout when it is one of `FLAX_REVISIONS`, and
    the torch path then reads the repo's default branch. Any other revision
    is a pin: the torch path reads only that revision, and raises
    FileNotFoundError when the repo has no weights at it.
    """
    if os.path.isdir(modelname):
        directory = Path(modelname)
        with open(directory / "config.json") as handle:
            config = json.load(handle)
        return {"config": config, "params": _read_vae_weights(directory) if params is None else params}

    errors: list[Exception] = []
    loaded = _load_from_hub(modelname, revision, params, errors)
    if loaded is not None:
        return loaded
    last_error = errors[-1] if errors else None
    pinned = revision not in FLAX_REVISIONS
    raise FileNotFoundError(
        f"no VAE weights in {modelname}"
        + (f" at revision {revision!r}" if pinned else "")
        + ": neither flax diffusion_flax_model.msgpack nor torch "
        "diffusion_pytorch_model.safetensors"
    ) from last_error


def _read_vae_weights(directory: Path) -> dict:
    """Read the params in `directory`, whichever of the two layouts it holds."""
    from flax.serialization import msgpack_restore

    msgpack = directory / "diffusion_flax_model.msgpack"
    if msgpack.exists():
        with open(msgpack, "rb") as handle:
            restored = msgpack_restore(handle.read())
        if not isinstance(restored, dict):
            raise ValueError(f"{msgpack} does not hold a param tree")
        return restored
    from dew.interop.safetensors_io import read_weights

    try:
        tensors: ParamTree = dict(read_weights(directory))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"no VAE weights in {directory}: neither diffusion_flax_model.msgpack "
            "nor diffusion_pytorch_model.safetensors (or its index)") from error
    # A diffusers name has no '/', so the flat table is the one translated.
    return translate_vae_weights(tensors)
