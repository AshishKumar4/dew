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

import math
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.batch import Batch, Variables
from dew.objectives.base import Aux, Objective, Ratio, Shown, Step


class _Block(nn.Module):
    """A pre-RMSNorm transformer block: QK-normed attention under `mask`, then a SwiGLU MLP."""

    width: int
    heads: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        head = self.width // self.heads
        h = nn.RMSNorm()(x)
        q, k, v = (nn.DenseGeneral((self.heads, head))(h) for _ in range(3))
        q, k = nn.RMSNorm()(q), nn.RMSNorm()(k)
        logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / math.sqrt(head)
        logits = jnp.where(mask, logits, -1e9)
        attended = jnp.einsum("...hqk,...khd->...qhd", jax.nn.softmax(logits, -1), v)
        x = x + nn.DenseGeneral(self.width, axis=(-2, -1))(attended)
        h = nn.RMSNorm()(x)
        gate, up = nn.Dense(4 * self.width)(h), nn.Dense(4 * self.width)(h)
        return x + nn.Dense(self.width)(nn.silu(gate) * up)


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
        *lead, height, width, channels = frames.shape
        p = self.patch
        x = frames.reshape(*lead, height // p, p, width // p, p, channels)
        x = jnp.moveaxis(x, -4, -3)
        return x.reshape(*lead, (height // p) * (width // p), p * p * channels)

    def _unpatchify(self, patches: jax.Array) -> jax.Array:
        *lead, _, _ = patches.shape
        height, width, channels = self.frame
        p = self.patch
        x = patches.reshape(*lead, height // p, width // p, p, p, channels)
        x = jnp.moveaxis(x, -3, -4)
        return x.reshape(*lead, height, width, channels)

    def setup(self) -> None:
        self.embed = nn.Dense(self.width)
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02), (self.width,))
        self.latent_tokens = self.param("latent_tokens", nn.initializers.normal(0.02),
                                        (self.latents, self.width))
        self.patch_positions = self.param("patch_positions", nn.initializers.normal(0.02),
                                          (self.patches, self.width))
        self.encoder = [_Block(self.width, self.heads) for _ in range(self.depth)]
        self.squeeze = nn.Dense(self.bottleneck)
        self.unsqueeze = nn.Dense(self.width)
        self.readout_tokens = self.param("readout_tokens", nn.initializers.normal(0.02),
                                         (self.patches, self.width))
        self.decoder = [_Block(self.width, self.heads) for _ in range(self.depth)]
        self.pixels = nn.Dense(self.patch * self.patch * self.frame[2])

    def __call__(self, frames: jax.Array) -> jax.Array:
        return self.decode(self.encode(frames))

    def encode(self, frames: jax.Array, mask_rate: jax.Array | None = None,
               key: jax.Array | None = None) -> jax.Array:
        """Each frame's latents; with `mask_rate` `[...]`, each patch replaced by the mask token with
        that frame's probability, drawn from `key`."""
        x = self.embed(self._patchify(frames)) + self.patch_positions
        if mask_rate is not None:
            masked = jax.random.uniform(key, x.shape[:-1]) < mask_rate[..., None]
            x = jnp.where(masked[..., None], self.mask_token, x)
        lead = x.shape[:-2]
        latent = jnp.broadcast_to(self.latent_tokens, (*lead, *self.latent_tokens.shape))
        tokens = jnp.concatenate([x, latent], -2)
        count = self.patches
        index = jnp.arange(tokens.shape[-2])
        # Latents read everything; patches read only patches.
        mask = (index[:, None] >= count) | (index[None, :] < count)
        for block in self.encoder:
            tokens = block(tokens, mask)
        return jnp.tanh(self.squeeze(tokens[..., count:, :]))

    def decode(self, latents: jax.Array) -> jax.Array:
        lead = latents.shape[:-2]
        readout = jnp.broadcast_to(self.readout_tokens, (*lead, *self.readout_tokens.shape))
        tokens = jnp.concatenate([self.unsqueeze(latents), readout], -2)
        index = jnp.arange(tokens.shape[-2])
        count = self.latents
        # Latents read only latents; readout tokens read themselves and the latents.
        mask = (index[None, :] < count) | (index[:, None] >= count)
        for block in self.decoder:
            tokens = block(tokens, mask)
        return self._unpatchify(self.pixels(tokens[..., count:, :]))


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
        video = jnp.asarray(batch["video"], jnp.float32)
        rate_key, mask_key = jax.random.split(step.key)
        rate = jax.random.uniform(rate_key, video.shape[:2], maxval=self.max_mask) if step.training else None

        def reconstructed(module: CausalTokenizer) -> jax.Array:
            return module.decode(module.encode(video, rate, mask_key))

        error = jnp.mean(jnp.square(self.model.apply(variables, method=reconstructed) - video))
        return Ratio(error, jnp.ones(())), Aux({"mse": error})


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
        if actions is not None and self.action_size:
            action = action + nn.Dense(self.width)(actions)
        tokens = jnp.concatenate([action[:, :, None], shortcut[:, :, None], spatial], 2)
        per_frame = tokens.shape[2]
        tokens = tokens + self.param("slots", nn.initializers.normal(0.02), (per_frame, self.width))
        times = self.param("times", nn.initializers.normal(0.02), (self.frames, self.width))
        tokens = tokens + times[:frames, None]
        tokens = tokens.reshape(batch, frames * per_frame, self.width)
        time = jnp.repeat(jnp.arange(frames), per_frame)
        mask = time[:, None] >= time[None, :]
        for _ in range(self.depth):
            tokens = _Block(self.width, self.heads)(tokens, mask)
        spatial_out = tokens.reshape(batch, frames, per_frame, -1)[:, :, 2:]
        out = nn.Dense(grouped.shape[-1])(nn.RMSNorm()(spatial_out))
        return out.reshape(noisy.shape)


def sampled_steps(key: jax.Array, shape: tuple[int, ...], k_max: int) -> tuple[jax.Array, jax.Array]:
    """Eq. 4: each frame's step size d, the reciprocal of a power of two up to `k_max` drawn uniformly,
    and its signal level tau drawn uniformly from {0, d, ..., 1 - d}."""
    size_key, level_key = jax.random.split(key)
    levels = int(math.log2(k_max)) + 1
    d = 1.0 / 2.0 ** jax.random.randint(size_key, shape, 0, levels).astype(jnp.float32)
    tau = jnp.floor(jax.random.uniform(level_key, shape) / d) * d
    return tau, d


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


class WorldModelObjective(Objective[Ratio]):
    """Trains `Dynamics` over a frozen tokenizer's latents of a batch's `video` (and `actions`, where
    the model reads them) by shortcut forcing, or with `shortcut` False by diffusion forcing."""

    shown: Mapping[str, Shown] = {"loss": Shown()}

    def __init__(self, model: Dynamics, tokenizer: CausalTokenizer, tokenizer_variables: Variables, *,
                 shortcut: bool = True):
        self.model = model
        self.tokenizer, self.tokenizer_variables = tokenizer, tokenizer_variables
        self.shortcut = shortcut

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        frames = jnp.zeros((1, 2, self.model.latents, self.model.bottleneck))
        return self.model.init(key, frames, jnp.ones((1, 2)), jnp.ones((1, 2)))

    @property
    def validation_loss_refusal(self) -> str | None:
        return None

    def latents(self, video: jax.Array) -> jax.Array:
        return jax.lax.stop_gradient(self.tokenizer.apply(self.tokenizer_variables, video,
                                                          method=CausalTokenizer.encode))

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        clean = self.latents(jnp.asarray(batch["video"], jnp.float32))
        actions = batch.get("actions")
        steps_key, noise_key = jax.random.split(step.key)
        d_min = 1.0 / self.model.k_max
        tau, d = sampled_steps(steps_key, clean.shape[:2], self.model.k_max)
        noise = jax.random.normal(noise_key, clean.shape)

        def f(noisy, tau, d):
            return self.model.apply(variables, noisy, tau, d, actions)

        if self.shortcut:
            per_frame = shortcut_forcing_loss(f, clean, noise, tau, d, d_min)
        else:
            per_frame = diffusion_forcing_loss(f, clean, noise, jnp.floor(tau / d_min) * d_min, d_min)
        stats = Ratio(per_frame.sum(), jnp.asarray(per_frame.size, per_frame.dtype))
        return stats, Aux({"frame_loss": per_frame.mean()})

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
                predicted = self.model.apply(variables, sequence, tau, d, given)[:, -1]
                current = current + (predicted - current) / (1 - level) / steps
            produced = jnp.concatenate([produced, current[:, None]], 1)
        return produced[:, context.shape[1]:]


__all__ = ["CausalTokenizer", "Dynamics", "TokenizerObjective", "WorldModelObjective",
           "diffusion_forcing_loss", "ramp", "sampled_steps", "shortcut_forcing_loss"]
