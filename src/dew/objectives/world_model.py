"""Dreamer 4's world model at a small scale (arXiv 2509.24527, docs/research/dreamer4.md): a causal
tokenizer of video frames, and an interactive dynamics transformer over its latents trained by shortcut
forcing (Eqs. 4, 6-8) or, as the baseline the paper improves on, diffusion forcing.

The tokenizer turns each frame's patches and `latents` learned tokens into `latents` x `bottleneck`
values through a tanh, masking each patch with probability drawn from U(0, 0.9) per frame (MAE); it
reads frames one at a time, so it is causal in time. The dynamics model reads each timestep's
action token, shortcut token (its signal level's and step size's indices) and the latents regrouped
into `spatial` tokens, attending block-causally (a timestep's tokens to each other and to the past),
and predicts each frame's clean latents (x-prediction).

Choices the paper leaves open (docs/research/dreamer4.md, last section) are this module's: the
widths and depths, `K_max` = 64, one attention over space and time in every layer, learned position
embeddings, the ramp weight multiplying each frame's loss, no LPIPS term in the tokenizer's loss,
and context frames at signal level 1 - `tau_ctx` when generating.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import Batch, Variables, unboxed
from dew.inputs import unit_range
from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock
from dew.nn.inputs import AttentionMetadata
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention
from dew.nn.mlp import GatedMLP
from dew.nn.scan_orders import patchify, unpatchify
from dew.objectives.base import Aux, Objective, Ratio, Shown, Step

RMS_EPSILON = 1e-6
"""The world model's RMS norms' epsilon, every one of them (Flax's `nn.RMSNorm` default, which its blocks
were drawn with)."""


def _block(width: int, heads: int, tokens: int, name: str) -> DecoderBlock:
    """A pre-RMSNorm `DecoderBlock` of `width` features over `tokens` tokens: unrotated attention of `heads`
    biased heads (`nope`), each head's queries and keys RMS-normed, under the call's pairwise mask
    (`AttentionMetadata.pairwise_mask`) in the reference implementation, whose logits are at least float32;
    then a biased SwiGLU of four times the width (`GatedMLP`)."""
    attention = functools.partial(CausalSelfAttention, nope=True, ctx=MixerContext(
        emb_features=width, num_heads=heads, num_kv_heads=heads, head_dim=width // heads, max_seq_len=tokens,
        causal=False, qk_norm=True, norm_eps=RMS_EPSILON, attention_bias=True, attention_impl="reference"))
    mlp = functools.partial(GatedMLP, hidden_features=4 * width, out_features=width, use_bias=True)
    return DecoderBlock(attention, mlp, width, BlockWiring(), norm_eps=RMS_EPSILON, name=name)


def _attended(blocks: Sequence[DecoderBlock], tokens: jax.Array, mask: jax.Array) -> jax.Array:
    """`tokens` `[..., S, D]` through `blocks`, each token reading the tokens `mask` `[S, S]` lets it, any
    leading axes of the tokens batch axes it is shared across, flattened into one for the blocks and
    restored."""
    *lead, count, width = tokens.shape
    flat = tokens.reshape(-1, count, width)
    metadata = AttentionMetadata(pairwise_mask=jnp.broadcast_to(mask, (flat.shape[0], count, count)))
    for block in blocks:
        flat = block(flat, attention_metadata=metadata)
    return flat.reshape(*lead, count, width)


class CausalTokenizer(nn.Module):
    """A frame `[..., H, W, C]` in [-1, 1] to latents `[..., latents, bottleneck]` and back, one frame
    at a time (Section 3.1): patches of `patch` pixels and `latents` learned tokens through `depth`
    blocks, the latents to `bottleneck` channels through a tanh; the decoder's readout tokens, one a
    patch, attend to themselves and the latents, the latents only to latents."""

    frame: tuple[int, int, int]
    patch: int = 4
    width: int = 64
    depth: int = 2
    heads: int = 4
    latents: int = 16
    bottleneck: int = 8

    @property
    def patches(self) -> int:
        height, width, _ = self.frame
        return (height // self.patch) * (width // self.patch)

    def _patchify(self, frames: jax.Array) -> jax.Array:
        """Frames `[..., H, W, C]` as `nn.scan_orders.patchify`'s row-major patches, its batch the frames'
        leading axes."""
        *lead, height, width, channels = frames.shape
        patches = patchify(frames.reshape(-1, height, width, channels), self.patch)
        return patches.reshape(*lead, *patches.shape[1:])

    def _unpatchify(self, patches: jax.Array) -> jax.Array:
        """`_patchify`'s inverse (`nn.scan_orders.unpatchify`)."""
        *lead, count, size = patches.shape
        frames = unpatchify(patches.reshape(-1, count, size), self.patch, *self.frame)
        return frames.reshape(*lead, *self.frame)

    def setup(self) -> None:
        self.embed = nn.Dense(self.width)
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02), (self.width,))
        self.latent_tokens = self.param("latent_tokens", nn.initializers.normal(0.02),
                                        (self.latents, self.width))
        self.patch_positions = self.param("patch_positions", nn.initializers.normal(0.02),
                                          (self.patches, self.width))
        tokens = self.patches + self.latents
        self.encoder = [_block(self.width, self.heads, tokens, f"encoder_{index}")
                        for index in range(self.depth)]
        self.squeeze = nn.Dense(self.bottleneck)
        self.unsqueeze = nn.Dense(self.width)
        self.readout_tokens = self.param("readout_tokens", nn.initializers.normal(0.02),
                                         (self.patches, self.width))
        self.decoder = [_block(self.width, self.heads, tokens, f"decoder_{index}")
                        for index in range(self.depth)]
        self.pixels = nn.Dense(self.patch * self.patch * self.frame[2])

    def __call__(self, frames: jax.Array) -> jax.Array:
        return self.decode(self.encode(frames))

    def encode(self, frames: jax.Array, mask_rate: jax.Array | None = None, key: jax.Array | None = None,
               *, masked: jax.Array | None = None) -> jax.Array:
        """Each frame's latents; each patch `masked` `[..., patches]` replaced by the mask token, or with
        `mask_rate` `[...]` each with that frame's probability, drawn from `key`."""
        x = self.embed(self._patchify(frames)) + self.patch_positions
        if mask_rate is not None:
            if key is None:
                raise ValueError("masking a tokenizer's patches draws from a key")
            masked = jax.random.uniform(key, x.shape[:-1]) < mask_rate[..., None]
        if masked is not None:
            x = jnp.where(masked[..., None], self.mask_token, x)
        lead = x.shape[:-2]
        latent = jnp.broadcast_to(self.latent_tokens, (*lead, *self.latent_tokens.shape))
        tokens = jnp.concatenate([x, latent], -2)
        count = self.patches
        index = jnp.arange(tokens.shape[-2])
        # Latents read everything; patches read only patches.
        tokens = _attended(self.encoder, tokens, (index[:, None] >= count) | (index[None, :] < count))
        return jnp.tanh(self.squeeze(tokens[..., count:, :]))

    def decode(self, latents: jax.Array) -> jax.Array:
        lead = latents.shape[:-2]
        readout = jnp.broadcast_to(self.readout_tokens, (*lead, *self.readout_tokens.shape))
        tokens = jnp.concatenate([self.unsqueeze(latents), readout], -2)
        index = jnp.arange(tokens.shape[-2])
        count = self.latents
        # Latents read only latents; readout tokens read themselves and the latents.
        tokens = _attended(self.decoder, tokens, (index[None, :] < count) | (index[:, None] >= count))
        return self._unpatchify(self.pixels(tokens[..., count:, :]))


def unit_frames(video: jax.typing.ArrayLike) -> jax.Array:
    """A batch's `video` as the objectives read it, floats in [-1, 1]: uint8 pixels, as the data workers write
    clips (`presets.clips`), are scaled to it (`dew.inputs.unit_range`), and floats are taken as the caller
    made them, already in the range."""
    video = jnp.asarray(video)
    return unit_range(video) if video.dtype == jnp.uint8 else video.astype(jnp.float32)


class TokenizerObjective(Objective[Ratio]):
    """The tokenizer's MAE reconstruction (Eq. 5 without LPIPS) of a batch's `video` `[B, T, H, W, C]`,
    each frame's mask rate drawn from U(0, 0.9)."""

    shown: Mapping[str, Shown] = {"loss": Shown()}

    def __init__(self, model: CausalTokenizer, *, max_mask: float = 0.9):
        self.model = model
        self.max_mask = max_mask

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, *self.model.frame)))

    @property
    def validation_loss_refusal(self) -> str | None:
        return None

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        video = unit_frames(batch["video"])
        rows, frames = video.shape[:2]
        masked = None
        if step.training:
            # Each clip's own draws (`Step.sample`): a frame's mask rate, and each patch's uniform.
            rate = step.sample("mask rate", functools.partial(jax.random.uniform, maxval=self.max_mask),
                               (frames,), rows=rows)
            uniform = step.sample("patch draw", jax.random.uniform, (frames, self.model.patches), rows=rows)
            masked = step.boundary("patch mask", uniform < rate[..., None])

        tokenizer = self.model.bind(variables)
        stats = self.row_mean(jnp.square(tokenizer.decode(tokenizer.encode(video, masked=masked)) - video),
                              batch)
        return stats, Aux({"mse": stats.mean()[0]})


class Dynamics(nn.Module):
    """The interactive dynamics transformer (Section 3.2) over frames of `latents` x `bottleneck`
    tokenizer latents regrouped into `spatial` tokens, with one action token a step (a learned
    embedding plus a projection of `action_size` actions, none for unlabeled video) and one shortcut
    token, the embeddings of its signal level's and step size's grid indices at `levels` levels
    (`K_max` = 2 ** (levels - 1)), predicting each frame's clean latents."""

    latents: int = 16
    bottleneck: int = 8
    spatial: int = 8
    action_size: int = 0
    width: int = 128
    depth: int = 4
    heads: int = 4
    levels: int = 7
    frames: int = 32

    @property
    def k_max(self) -> int:
        return 2 ** (self.levels - 1)

    @nn.compact
    def __call__(self, noisy: jax.Array, tau: jax.Array, d: jax.Array,
                 actions: jax.Array | None = None) -> jax.Array:
        """`noisy` `[B, T, latents, bottleneck]` at signal levels `tau` with step sizes `d` `[B, T]`."""
        batch, frames = noisy.shape[:2]
        grouped = noisy.reshape(batch, frames, self.spatial, -1)
        spatial = nn.Dense(self.width)(grouped)
        step_index = jnp.round(-jnp.log2(d)).astype(jnp.int32)
        level_index = jnp.round(tau * self.k_max).astype(jnp.int32)
        shortcut = jnp.concatenate([nn.Embed(self.levels, self.width // 2)(step_index),
                                    nn.Embed(self.k_max + 1, self.width // 2)(level_index)], -1)
        action = self.param("action_token", nn.initializers.normal(0.02), (self.width,))
        action = jnp.broadcast_to(action, (batch, frames, self.width))
        if actions is not None:
            # Actions projected and summed into the learned token; unlabeled video has the token alone.
            if not self.action_size or actions.shape[-1] != self.action_size:
                raise ValueError(f"a dynamics model of {self.action_size} actions a step read actions "
                                 f"{actions.shape}")
            action = action + nn.Dense(self.width, name="action_projection")(actions)
        tokens = jnp.concatenate([action[:, :, None], shortcut[:, :, None], spatial], 2)
        per_frame = tokens.shape[2]
        tokens = tokens + self.param("slots", nn.initializers.normal(0.02), (per_frame, self.width))
        times = self.param("times", nn.initializers.normal(0.02), (self.frames, self.width))
        tokens = tokens + times[:frames, None]
        tokens = tokens.reshape(batch, frames * per_frame, self.width)
        time = jnp.repeat(jnp.arange(frames), per_frame)
        blocks = [_block(self.width, self.heads, self.frames * per_frame, f"block_{index}")
                  for index in range(self.depth)]
        tokens = _attended(blocks, tokens, time[:, None] >= time[None, :])
        spatial_out = tokens.reshape(batch, frames, per_frame, -1)[:, :, 2:]
        out = nn.Dense(grouped.shape[-1])(nn.RMSNorm(epsilon=RMS_EPSILON)(spatial_out))
        return out.reshape(noisy.shape)


def sampled_steps(key: jax.Array, shape: tuple[int, ...], k_max: int) -> tuple[jax.Array, jax.Array]:
    """Eq. 4: each frame's step size d, the reciprocal of a power of two up to `k_max` drawn uniformly,
    and its signal level tau drawn uniformly from {0, d, ..., 1 - d} (`_gridded`)."""
    size_key, level_key = jax.random.split(key)
    return _gridded(jax.random.randint(size_key, shape, 0, int(math.log2(k_max)) + 1),
                   jax.random.uniform(level_key, shape))


def _gridded(exponent: jax.Array, uniform: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Eq. 4 from its draws: the signal level tau and the step size d = 2^-`exponent`, tau `uniform` in
    [0, 1) floored to d's grid."""
    d = 1.0 / 2.0 ** exponent.astype(jnp.float32)
    return jnp.floor(uniform / d) * d, d


def ramp(tau: jax.Array) -> jax.Array:
    """Eq. 8's weight, 0.9 tau + 0.1."""
    return 0.9 * tau + 0.1


type Predictor = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
"""`f(noisy, tau, d) -> clean latents`, every frame at once."""


def shortcut_forcing_loss(f: Predictor, clean: jax.Array, noise: jax.Array, tau: jax.Array, d: jax.Array,
                          d_min: float) -> jax.Array:
    """Eqs. 6-8: each frame's loss `[B, T]`, its x-prediction's squared error at the smallest step and
    else its velocity's against the average of its own two half steps, held still and scaled back to
    x-space by (1 - tau)^2, every frame weighted by the ramp."""
    level = tau.reshape(*tau.shape, *(1,) * (clean.ndim - tau.ndim))
    step = d.reshape(level.shape)
    noisy = (1 - level) * noise + level * clean
    predicted = f(noisy, tau, d)
    half = jnp.maximum(d / 2, d_min)
    first = (f(noisy, tau, half) - noisy) / (1 - level)
    between = noisy + first * step / 2
    later = tau + d / 2
    second = (f(between, later, half) - between) / (1 - later.reshape(level.shape))
    target = jax.lax.stop_gradient((first + second) / 2)
    axes = tuple(range(tau.ndim, clean.ndim))
    smallest = jnp.sum(jnp.square(predicted - clean), axes)
    bootstrap = jnp.sum(jnp.square(1 - level) * jnp.square((predicted - noisy) / (1 - level) - target), axes)
    return ramp(tau) * jnp.where(d <= d_min, smallest, bootstrap)


def diffusion_forcing_loss(f: Predictor, clean: jax.Array, noise: jax.Array, tau: jax.Array,
                           d_min: float) -> jax.Array:
    """The baseline: each frame at its own signal level, its x-prediction regressed at the smallest
    step size alone, the ramp weighting it as shortcut forcing's."""
    level = tau.reshape(*tau.shape, *(1,) * (clean.ndim - tau.ndim))
    noisy = (1 - level) * noise + level * clean
    predicted = f(noisy, tau, jnp.full(tau.shape, d_min))
    return ramp(tau) * jnp.sum(jnp.square(predicted - clean), tuple(range(tau.ndim, clean.ndim)))


TOKENIZER = "tokenizer"
"""The collection of a world model's tree holding its frozen tokenizer's weights (`WorldModelObjective`)."""


class WorldModelObjective(Objective[Ratio]):
    """Trains `Dynamics` over a frozen tokenizer's latents of a batch's `video` (and `actions`, where
    the model reads them) by shortcut forcing, or with `shortcut` False by diffusion forcing. The
    tokenizer's weights are state, under `TOKENIZER` beside the dynamics' (`held_variables`), as a
    diffusion objective's frozen towers are: they reach the compiled step as arguments, placed and
    checkpointed with the rest, and the optimizer never sees them."""

    shown: Mapping[str, Shown] = {"loss": Shown()}

    # The held tree is the starting tree or the tokenizer a draw goes beside.
    _held_parts = True

    def __init__(self, model: Dynamics, tokenizer: CausalTokenizer, tokenizer_variables: Variables, *,
                 shortcut: bool = True):
        self.model = model
        # A held tree enters plain: a draw's boxes (`nn.with_logical_partitioning`) are its table's.
        self.tokenizer, self._tokenizer = tokenizer, unboxed(tokenizer_variables)
        self.shortcut = shortcut

    def held_variables(self) -> Variables:
        """The starting tree, or none, with the tokenizer's weights it starts from under `TOKENIZER`."""
        return {TOKENIZER: self._tokenizer, **(self.variables or {})}

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        held = self.held_variables() if held is None else held
        frames = jnp.zeros((1, 2, self.model.latents, self.model.bottleneck))
        # A labeled model's action projection drawn with the rest, so labeled and unlabeled batches
        # read one tree.
        actions = jnp.zeros((1, 2, self.model.action_size)) if self.model.action_size else None
        drawn = self.model.init(key, frames, jnp.ones((1, 2)), jnp.ones((1, 2)), actions)
        return {**drawn, TOKENIZER: held[TOKENIZER]}

    @property
    def validation_loss_refusal(self) -> str | None:
        return None

    def latents(self, variables: Variables, video: jax.Array) -> jax.Array:
        """`video`'s latents, each frame encoded by the tokenizer `variables` hold (`TOKENIZER`)."""
        return jax.lax.stop_gradient(self.tokenizer.bind(variables[TOKENIZER]).encode(video))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        clean = self.latents(variables, unit_frames(batch["video"]))
        actions = batch.get("actions")
        rows, frames = clean.shape[:2]
        d_min = 1.0 / self.model.k_max
        # Each clip's own draws (`Step.sample`): Eq. 4's step size and signal level a frame, and its noise.
        exponents = functools.partial(jax.random.randint, minval=0, maxval=self.model.levels)
        exponent = step.sample("step exponent", exponents, (frames,), rows=rows)
        tau, d = _gridded(exponent, step.sample("level draw", jax.random.uniform, (frames,), rows=rows))
        tau, d = step.boundary("signal level", tau), step.boundary("step size", d)
        noise = step.sample("noise", jax.random.normal, clean.shape[1:], rows=rows)

        dynamics = self.model.bind(_dynamics(variables))

        def f(noisy: jax.Array, tau: jax.Array, d: jax.Array) -> jax.Array:
            return dynamics(noisy, tau, d, actions)

        if self.shortcut:
            per_frame = shortcut_forcing_loss(f, clean, noise, tau, d, d_min)
        else:
            per_frame = diffusion_forcing_loss(f, clean, noise, jnp.floor(tau / d_min) * d_min, d_min)
        stats = self.row_mean(per_frame, batch)
        return stats, Aux({"frame_loss": stats.mean()[0]})

    def generate(self, variables: Variables, context: jax.Array, frames: int, key: jax.Array, *,
                 steps: int = 4, tau_ctx: float = 0.1, actions: jax.Array | None = None) -> jax.Array:
        """`frames` latent frames after the `context` latents `[B, C, ...]`, one at a time, each from
        noise in `steps` steps of d = 1 / `steps` (a shortcut model's step size; the diffusion-forcing
        baseline is told the smallest), the context at signal level 1 - `tau_ctx`."""
        d_model = 1.0 / steps if self.shortcut else 1.0 / self.model.k_max
        produced = context
        for _ in range(frames):
            key, noise_key, context_key = jax.random.split(key, 3)
            held = (1 - tau_ctx) * produced + tau_ctx * jax.random.normal(context_key, produced.shape)
            current = jax.random.normal(noise_key, context.shape[:1] + context.shape[2:])
            for k in range(steps):
                level = k / steps
                sequence = jnp.concatenate([held, current[:, None]], 1)
                tau = jnp.concatenate([jnp.full(held.shape[:2], 1 - tau_ctx),
                                       jnp.full((context.shape[0], 1), level)], 1)
                d = jnp.full(tau.shape, d_model)
                given = None if actions is None else actions[:, :sequence.shape[1]]
                predicted = self.model.bind(_dynamics(variables))(sequence, tau, d, given)[:, -1]
                current = current + (predicted - current) / (1 - level) / steps
            produced = jnp.concatenate([produced, current[:, None]], 1)
        return produced[:, context.shape[1]:]


def _dynamics(variables: Variables) -> Variables:
    """A world model's tree without its tokenizer: the dynamics model's own collections."""
    return {name: tree for name, tree in variables.items() if name != TOKENIZER}


__all__ = ["TOKENIZER", "CausalTokenizer", "Dynamics", "TokenizerObjective", "WorldModelObjective",
           "diffusion_forcing_loss", "ramp", "sampled_steps", "shortcut_forcing_loss", "unit_frames"]
