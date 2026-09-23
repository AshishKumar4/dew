"""Build the convolutional encoder and decoder of a diffusion VAE.

The modules are independent linen ports of huggingface/diffusers v0.29.2
src/diffusers/models/vae_flax.py (Apache-2.0). The loader below reads source
configuration and checkpoint files; no external model implementation runs.
"""

import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.text_encoders import ParamTree


class FlaxUpsample2D(nn.Module):
    """Double each spatial axis by nearest-neighbour resize, then a 3x3 conv."""

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        return self.conv(hidden_states)


class FlaxDownsample2D(nn.Module):
    """Halve each spatial axis with a stride-2 3x3 conv, padded on the far side."""

    in_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        hidden_states = jnp.pad(hidden_states, pad_width=pad)
        return self.conv(hidden_states)


class FlaxResnetBlock2D(nn.Module):
    """Run two group-normed 3x3 convolutions and add the input back.

    `use_nin_shortcut` puts a 1x1 convolution on the residual; None takes it
    whenever the block changes the channel count.
    """

    in_channels: int
    out_channels: int | None = None
    dropout: float = 0.0
    groups: int = 32
    use_nin_shortcut: bool | None = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels

        self.norm1 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, dtype=self.dtype)
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.norm2 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


class FlaxAttentionBlock(nn.Module):
    """Attend over an image's pixels as a sequence, with a residual.

    `num_head_channels` is the width of one head, so the head count is
    `channels // num_head_channels`; None is a single head. The scale is
    split over the query and the key, as diffusers writes it, which is the
    arithmetic the published weights were trained under.
    """

    channels: int
    num_head_channels: int | None = None
    num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_heads = self.channels // self.num_head_channels if self.num_head_channels is not None else 1

        dense = partial(nn.Dense, self.channels, dtype=self.dtype)

        self.group_norm = nn.GroupNorm(num_groups=self.num_groups, epsilon=1e-6, dtype=self.dtype)
        self.query, self.key, self.value = dense(), dense(), dense()
        self.proj_attn = dense()

    def transpose_for_scores(self, projection):
        new_projection_shape = (*projection.shape[:-1], self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D)
        new_projection = projection.reshape(new_projection_shape)
        # (B, T, H, D) -> (B, H, T, D)
        return jnp.transpose(new_projection, (0, 2, 1, 3))

    def __call__(self, hidden_states):
        residual = hidden_states
        batch, height, width, channels = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape((batch, height * width, channels))

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # transpose
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # compute attentions
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attn_weights = jnp.einsum("...qc,...kc->...qk", query * scale, key * scale)
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # attend to values
        hidden_states = jnp.einsum("...kc,...qk->...qc", value, attn_weights)

        hidden_states = jnp.transpose(hidden_states, (0, 2, 1, 3))
        new_hidden_states_shape = (*hidden_states.shape[:-2], self.channels)
        hidden_states = hidden_states.reshape(new_hidden_states_shape)

        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.reshape((batch, height, width, channels))
        return hidden_states + residual


class FlaxDownEncoderBlock2D(nn.Module):
    """Run `num_layers` resnets to `out_channels`, then halve the spatial axes.

    Only the first resnet changes the channel count. `add_downsample` is
    False on the last level, which keeps its resolution.
    """

    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)

        return hidden_states


class FlaxUpDecoderBlock2D(nn.Module):
    """Run `num_layers` resnets to `out_channels`, then double the spatial axes.

    Only the first resnet changes the channel count. `add_upsample` is
    False on the last level, which keeps its resolution.
    """

    in_channels: int
    out_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels
            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                groups=self.resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    """Alternate attention and resnet blocks at the bottleneck resolution.

    There is always a leading resnet, then `num_layers` attention/resnet
    pairs. The channel count does not change.
    """

    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    resnet_groups: int = 32
    num_attention_heads: int | None = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnet_groups = self.resnet_groups if self.resnet_groups is not None else min(self.in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                groups=resnet_groups,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = FlaxAttentionBlock(
                channels=self.in_channels,
                num_head_channels=self.num_attention_heads,
                num_groups=resnet_groups,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                groups=resnet_groups,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, deterministic=deterministic)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        return hidden_states


class FlaxEncoder(nn.Module):
    """Encode images to latent moments with a conv stack that halves each axis.

    `block_out_channels` gives one level per entry, each `layers_per_block`
    resnets; the last level does not downsample. `double_z` doubles the
    output channels so the caller can split them into mean and log-variance.
    """

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Sequence[str] = ("DownEncoderBlock2D",)
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    double_z: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels
        # in
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # downsampling
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, _ in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = FlaxDownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block,
                resnet_groups=self.norm_num_groups,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # middle
        self.mid_block = FlaxUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            num_attention_heads=None,
            dtype=self.dtype,
        )

        # end
        conv_out_channels = 2 * self.out_channels if self.double_z else self.out_channels
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6, dtype=self.dtype)
        self.conv_out = nn.Conv(
            conv_out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # in
        sample = self.conv_in(sample)

        # downsampling
        for block in self.down_blocks:
            sample = block(sample, deterministic=deterministic)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # end
        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        return self.conv_out(sample)


class FlaxDecoder(nn.Module):
    """Decode latents to images with a conv stack that doubles each axis.

    `block_out_channels` is read in reverse, one level per entry, each
    `layers_per_block + 1` resnets; the last level does not upsample.
    """

    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Sequence[str] = ("UpDecoderBlock2D",)
    block_out_channels: Sequence[int] = (64,)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        block_out_channels = self.block_out_channels

        # z to block_in
        self.conv_in = nn.Conv(
            block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # middle
        self.mid_block = FlaxUNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_groups=self.norm_num_groups,
            num_attention_heads=None,
            dtype=self.dtype,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_blocks = []
        for i, _ in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = FlaxUpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                num_layers=self.layers_per_block + 1,
                resnet_groups=self.norm_num_groups,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )
            up_blocks.append(up_block)

        self.up_blocks = up_blocks

        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6, dtype=self.dtype)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, sample, deterministic: bool = True):
        # z to block_in
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, deterministic=deterministic)

        # upsampling
        for block in self.up_blocks:
            sample = block(sample, deterministic=deterministic)

        sample = self.conv_norm_out(sample)
        sample = nn.swish(sample)
        return self.conv_out(sample)


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


def translate_vae_weights(torch_tensors: ParamTree) -> dict:
    """Convert diffusers AutoencoderKL tensors into this module's param tree.

    Convolution kernels transpose from torch's [out, in, kh, kw] to linen's
    [kh, kw, in, out], and dense kernels from [out, in] to [in, out]; norms
    and biases keep their layout. Every tensor maps, so an unknown name
    raises rather than loading half an autoencoder.
    """
    params: dict = {}
    for name, tensor in torch_tensors.items():
        if isinstance(tensor, dict):
            raise ValueError(f"{name} is a subtree; a diffusers VAE table is flat")
        leaf = np.asarray(tensor, dtype=np.float32)
        path = _vae_path(name, leaf.ndim)
        if path[-1] == "kernel":
            leaf = leaf.transpose(2, 3, 1, 0) if leaf.ndim == 4 else leaf.T
        node = params
        for entry in path[:-1]:
            node = node.setdefault(entry, {})
        node[path[-1]] = leaf
    return params


# The SD1-era flax weights live on their own branches, so these name a layout,
# not a pinned revision.
FLAX_REVISIONS = ("bf16", "flax")


def _flax_layout(modelname: str, revision: str, params, errors: list) -> dict | None:
    """Read the SD1-era flax msgpack, or None when the repo ships none.

    The four candidates are the revision the caller named and the `flax`
    branch, each with and without the `vae` subfolder. Every miss is
    appended to `errors`, so the caller can raise the last one.
    """
    from flax.serialization import msgpack_restore
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

    candidates = [
        (revision, "vae"),
        ("flax", "vae"),
        (revision, None),
        (None, None),
    ]
    for candidate_revision, subfolder in candidates:
        try:
            config_path = hf_hub_download(modelname, "config.json",
                                          revision=candidate_revision, subfolder=subfolder)
            with open(config_path) as f:
                config = json.load(f)
            # Candidate eligibility must match a source load even when a
            # config exists in a revision that carries no matching weights.
            weights_path = hf_hub_download(modelname, "diffusion_flax_model.msgpack",
                                           revision=candidate_revision, subfolder=subfolder,
                                           dry_run=params is not None)
            if params is not None:
                return {"config": config, "params": params}
            assert isinstance(weights_path, str)
            with open(weights_path, "rb") as f:
                return {"config": config, "params": msgpack_restore(f.read())}
        except (EntryNotFoundError, RevisionNotFoundError) as e:
            errors.append(e)
    return None


def _torch_layout(modelname: str, revision: str, params, errors: list) -> dict | None:
    """Read the torch safetensors and translate them, or None when absent.

    A revision outside `FLAX_REVISIONS` is a pin, so only that revision is
    tried; otherwise the repo's default branch is. Every miss is appended
    to `errors`.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RevisionNotFoundError

    pinned = revision not in FLAX_REVISIONS
    candidates = ([(revision, "vae"), (revision, None)]
                  if pinned else [(None, "vae"), (None, None)])
    for candidate_revision, subfolder in candidates:
        try:
            config_path = hf_hub_download(modelname, "config.json",
                                          revision=candidate_revision, subfolder=subfolder)
            hf_hub_download(modelname, "diffusion_pytorch_model.safetensors",
                            revision=candidate_revision, subfolder=subfolder, dry_run=params is not None)
        except (EntryNotFoundError, RevisionNotFoundError) as e:
            errors.append(e)
            continue
        directory = Path(config_path).parent
        with open(config_path) as handle:
            config = json.load(handle)
        return {"config": config, "params": _read_vae_weights(directory) if params is None else params}
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
    loaded = _flax_layout(modelname, revision, params, errors)
    if loaded is None:
        loaded = _torch_layout(modelname, revision, params, errors)
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
