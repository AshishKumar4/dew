"""DreamerV3 (Hafner et al., arXiv 2301.04104 v2 and Nature 2025) for vector observations, as its code
computes it (danijar/dreamerv3 at e01491fad6434b2245a3b8ca201dd7faedcc458c, cited by file:line).

The world model is an RSSM: a block-diagonal GRU's deterministic state `deter` (rssm.py:135-158),
a posterior over `stoch` categoricals of `classes` classes from it and the encoded observation, a
prior from it alone (1% unimix, straight-through draws, rssm.py:161-176), and heads off the two
concatenated: the symlog-MSE decoder, the two-hot reward, the Bernoulli continue. It is trained
on replayed windows by the decoder's, reward's and continue's losses and the dynamics and
representation KLs with one free nat (rssm.py:120-133, agent.py:156-182).

The world model is an `Env` whose state is the latent and whose step is the prior (`LatentEnv`), so
the actor and critic learn in imagination through the same `env_rollout` acting in a simulator
runs: from every replayed posterior state, 15 imagined steps (agent.py:189-201). The critic is a
two-hot head trained toward the imagination's lambda-returns and toward its own slow EMA
(agent.py:382-447), and toward the replay's lambda-returns at 0.3 (agent.py:449-480); the actor by
REINFORCE on the returns less the values, over the 5th-to-95th percentile range of the returns
floored at 1 (utils.py:16-72), with an entropy bonus of 3e-4.

Two differences from the code: a window starts from a zero latent at its first step, which the
code's replay context instead resumes from a latent stored at acting time; and the computation runs
in float32 where the code runs bfloat16.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
from flax.typing import Dtype

from dew.batch import Batch, Variables, merge, under
from dew.nn.categorical import kl_divergence
from dew.nn.kernels.options import program
from dew.nn.precision import at_least_fp32, divided
from dew.objectives.base import Aux, EMASpec, Objective, Ratio, Shown, Step, traced_update_flops
from dew.objectives.rl.tssm import TSSM, Cache
from dew.rl.aggregation import LABEL, Teacher, labelled, paired
from dew.rl.distributions import (
    onehot_sample,
    symlog,
    twohot_bins,
    twohot_loss,
    twohot_mean,
    unimix_log_probs,
)
from dew.rl.env import (
    Carry,
    Driven,
    Env,
    Memory,
    Observation,
    Params,
    Policy,
    TimeStep,
    Transition,
    Tree,
    env_rollout,
    start,
    view,
    views,
)
from dew.rl.evaluation import episodes
from dew.rl.replay import ReplayBuffer, Steps
from dew.rl.simulator import DrivenSimulator, Simulator, simulator_rollout, simulator_start

if TYPE_CHECKING:
    from dew.state import TrainState


def _init(scale: float = 1.0) -> nn.initializers.Initializer:
    """The code's `trunc_normal_in`: a standard normal truncated to two deviations, uncorrected, times
    1.1368 sqrt(1 / fan in), times `scale` (nets.py:169-175), the fan in every axis but the output's
    (`compute_fans`, nets.py:187-197): a dense kernel's inputs, a convolution's window of input
    channels, a block-diagonal kernel's inputs over all its blocks."""
    if scale == 0:
        return nn.initializers.zeros

    def init(key: jax.Array, shape: tuple[int, ...], dtype: Dtype = jnp.float32) -> jax.Array:
        deviation = 1.1368 * scale / math.sqrt(math.prod(shape[:-1]))
        return (deviation * jax.random.truncated_normal(key, -2, 2, shape)).astype(dtype)

    return init


class _Layers(nn.Module):
    """`count` layers of a linear map to `units`, an RMS norm and SiLU (nets.py:565)."""

    units: int
    count: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        held = at_least_fp32(self.dtype)
        for _ in range(self.count):
            x = nn.Dense(self.units, kernel_init=_init(), dtype=self.dtype, param_dtype=held)(x)
            x = nn.silu(nn.RMSNorm(epsilon=1e-4, dtype=self.dtype, param_dtype=held)(x))
        return x


class _BlockLinear(nn.Module):
    """A block-diagonal linear map of `blocks` blocks to `features` (nets.py:254)."""

    features: int
    blocks: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        grouped = x.reshape(*x.shape[:-1], self.blocks, x.shape[-1] // self.blocks)
        fan = grouped.shape[-1]
        held = at_least_fp32(self.dtype)
        kernel = self.param("kernel", _init(), (self.blocks, fan, self.features // self.blocks), held)
        bias = self.param("bias", nn.initializers.zeros, (self.features,), held)
        grouped, kernel, bias = (value.astype(self.dtype) for value in (grouped, kernel, bias))
        return jnp.einsum("...gi,gio->...go", grouped, kernel).reshape(*x.shape[:-1], self.features) + bias


class ConvEncoder(nn.Module):
    """DreamerV3's image encoder (rssm.py:179-248, the `simple` encoder): pixels `[..., H, W, C]` in
    [0, 255] scaled to [-0.5, 0.5], then a stage for each of `mults`: a `kernel`-square same-padded
    convolution to `depth` times the multiplier channels, a 2x2 max pool, an RMS norm and a SiLU; the
    last stage's map flattened. The defaults are configs.yaml's `enc.simple` at its 200M size."""

    depth: int = 64
    mults: tuple[int, ...] = (2, 3, 4, 4)
    kernel: int = 5
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image: jax.Array) -> jax.Array:
        lead = image.shape[:-3]
        x = divided(image.reshape(-1, *image.shape[-3:]).astype(self.dtype), 255) - 0.5
        for stage, mult in enumerate(self.mults):
            x = nn.Conv(self.depth * mult, (self.kernel, self.kernel), padding="SAME", kernel_init=_init(),
                        dtype=self.dtype, param_dtype=at_least_fp32(self.dtype), name=f"cnn{stage}")(x)
            rows, height, width, channels = x.shape
            x = x.reshape(rows, height // 2, 2, width // 2, 2, channels).max((2, 4))
            norm = nn.RMSNorm(epsilon=1e-4, dtype=self.dtype, param_dtype=at_least_fp32(self.dtype),
                              name=f"cnn{stage}norm")
            x = nn.silu(norm(x))
        return x.reshape(*lead, -1)


class ConvDecoder(nn.Module):
    """DreamerV3's image decoder (rssm.py:253-355, the `simple` decoder with `bspace` 8): from the
    deterministic state through `blocks` block-diagonal maps and from the categoricals through two
    dense layers to the smallest map, its side the image's over 2 to the stages, summed, normed and
    a SiLU; then a stage for each but the last of `mults`, deepest first: a 2x nearest upsampling, a
    convolution, an RMS norm and a SiLU; a last upsampling and convolution to the image's channels;
    a sigmoid, the pixels in [0, 1]."""

    image: tuple[int, int, int]
    depth: int = 64
    units: int = 1024
    mults: tuple[int, ...] = (2, 3, 4, 4)
    kernel: int = 5
    blocks: int = 8
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array) -> jax.Array:
        lead = deter.shape[:-1]
        depths = [self.depth * mult for mult in self.mults]
        height, width = (side // 2 ** len(depths) for side in self.image[:2])
        shape = (height, width, depths[-1])
        size = math.prod(shape)
        dense = functools.partial(nn.Dense, kernel_init=_init(), dtype=self.dtype,
                                  param_dtype=at_least_fp32(self.dtype))
        conv = functools.partial(nn.Conv, kernel_size=(self.kernel, self.kernel), padding="SAME",
                                 dtype=self.dtype, param_dtype=at_least_fp32(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-4, dtype=self.dtype,
                                 param_dtype=at_least_fp32(self.dtype))
        x0 = _BlockLinear(size, self.blocks, self.dtype, name="sp0")(deter.reshape(-1, deter.shape[-1]))
        # Each block's outputs its own channels of every position: '(g h w c) -> h w (g c)'.
        x0 = x0.reshape(-1, self.blocks, height, width, depths[-1] // self.blocks)
        x0 = x0.transpose(0, 2, 3, 1, 4).reshape(-1, *shape)
        x1 = dense(2 * self.units, name="sp1")(stoch.reshape(math.prod(lead), -1))
        x1 = dense(size, name="sp2")(nn.silu(norm(name="sp1norm")(x1)))
        x = nn.silu(norm(name="spnorm")(x0 + x1.reshape(-1, *shape)))
        # The stages' names are the code's, `conv{i}` the stage before the i-th encoder stage's map.
        for stage in reversed(range(len(depths) - 1)):
            x = conv(depths[stage], kernel_init=_init(), name=f"conv{stage}")(x.repeat(2, -2).repeat(2, -3))
            x = nn.silu(norm(name=f"conv{stage}norm")(x))
        x = conv(self.image[-1], kernel_init=_init(), name="imgout")(x.repeat(2, -2).repeat(2, -3))
        return jax.nn.sigmoid(x).reshape(*lead, *self.image)


class WorldModel(nn.Module):
    """DreamerV3's networks over observations of `observation_size` and continuous actions of
    `action_size` in [-1, 1]. The defaults are the code's 1M model (configs.yaml `size1m`, which its
    DM Control proprio preset trains): a 512-wide deterministic state in 8 blocks, 32 categoricals of
    4 classes, 64 hidden and 64 units a layer."""

    observation_size: int
    action_size: int
    image: tuple[int, int, int] | None = None
    """An image observation's shape `(H, W, C)`, uint8 pixels, its sides divisible by 16: encoded and
    decoded by convolutions (`ConvEncoder`, `ConvDecoder`, `conv_depth` channels); None for a vector of
    `observation_size`."""
    conv_depth: int = 64
    reads: str | None = None
    """Where observations are dicts, the entry the encoder reads and the decoder rebuilds (FlyDreamer's
    `image`); the others are privileged, read by the `auxiliary` heads alone."""
    proprio: tuple[tuple[str, int], ...] = ()
    """Vector entries the encoder reads beside the image, each `(entry, width)`, symlogged, concatenated and
    through an MLP beside the convolutions (DreamerV3's encoder over `mlp_keys`, nets.py; SkyDreamer's body
    rates, motor speeds and flight plan, arXiv 2510.14783 Fig 3): sensed, so the policy reads them; decoding
    them is the auxiliary heads'."""
    auxiliary: tuple[tuple[str, int, int], ...] = ()
    """Privileged entries the latent learns to predict, each `(entry, width, units)`: two layers of
    `units` then a linear map to the entry's `width`, its squared error summed (FlyDreamer's
    `aux_state_head` and `aux_goal_head`, dreamerv3/agent.py:72-78 and :207-217, configs.yaml
    `{layers: 2, output: mse}`)."""
    dynamics: TSSM | None = None
    """TransDreamer's transformer dynamics (`dew.objectives.rl.tssm.TSSM`) in place of the block GRU,
    its own posterior and prior; None for the GRU's (RSSM)."""
    deter: int = 512
    hidden: int = 64
    stoch: int = 32
    classes: int = 4
    blocks: int = 8
    units: int = 64
    bins: int = 255
    unimix: float = 0.01
    min_std: float = 0.1
    max_std: float = 1.0
    dtype: Dtype = jnp.float32
    """The networks' computation dtype, bfloat16 for speed (DreamerV3's COMPUTE_DTYPE): parameters stay at
    least float32, and norms, distributions, heads' outputs and losses at least float32. A TSSM's
    dynamics compute in its own dtype."""

    def setup(self) -> None:
        dense = functools.partial(nn.Dense, kernel_init=_init(), dtype=self.dtype,
                                  param_dtype=at_least_fp32(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-4, dtype=self.dtype,
                                 param_dtype=at_least_fp32(self.dtype))
        layers = functools.partial(_Layers, dtype=self.dtype)
        if self.image is None:
            self.encoder = layers(self.units, 3)
            self.vector_decoder = layers(self.units, 3)
            self.decoded = dense(self.observation_size)
        else:
            self.encoder = ConvEncoder(self.conv_depth, dtype=self.dtype)
            self.image_decoder = ConvDecoder(self.image, self.conv_depth, self.units, blocks=self.blocks,
                                             dtype=self.dtype)
            if self.proprio:
                self.proprio_encoder = layers(self.units, 3)
        self.dynamics_in = [dense(self.hidden) for _ in range(3)]
        self.dynamics_norms = [norm() for _ in range(3)]
        self.dynamics_hidden = _BlockLinear(self.deter, self.blocks, self.dtype)
        self.dynamics_hidden_norm = norm()
        self.gru = _BlockLinear(3 * self.deter, self.blocks, self.dtype)
        self.posterior_layers = layers(self.hidden, 1)
        self.posterior_logits = dense(self.stoch * self.classes)
        self.prior_layers = layers(self.hidden, 2)
        self.prior_logits = dense(self.stoch * self.classes)
        self.reward_layers = layers(self.units, 1)
        self.reward_logits = dense(self.bins, kernel_init=_init(0.0))
        self.continue_layers = layers(self.units, 1)
        self.continue_logit = dense(1)
        self.policy_layers = layers(self.units, 3)
        self.policy_mean = dense(self.action_size, kernel_init=_init(0.01))
        self.policy_std = dense(self.action_size, kernel_init=_init(0.01))
        self.value = Critic(self.units, self.bins, self.dtype)
        self.auxiliary_heads = {name: (layers(units, 2), dense(width))
                                for name, width, units in self.auxiliary}

    def __call__(self, observation: jax.Array | dict[str, jax.Array], action: jax.Array) -> jax.Array:
        """Every network once, for initialization: a step from a zero latent."""
        tokens = self.embed(observation)
        deter, stoch = self.initial(action.shape[:-1])
        if self.dynamics is None:
            deter = self.core(deter, stoch, action)
            stoch = self.categorical(self.posterior(deter, tokens))
            prior = self.prior(deter)
        else:
            prior = self.dynamics(tokens[:, None], action[:, None], jnp.ones((action.shape[0], 1), bool))
        feature = self.feature(deter, stoch)
        mean, _ = self.policy(feature)
        predicted = sum(self.predicted(name, feature).sum() for name, _, _ in self.auxiliary)
        return (self.decode(deter, stoch).sum() + self.reward(feature).sum() + self.continue_(feature).sum()
                + mean.sum() + self.value(feature).sum() + prior.sum() + predicted)

    def example(self, batch: int) -> jax.Array | dict[str, jax.Array]:
        """`batch` zero observations as the encoder reads them (`seen`): a vector, an image, or an image
        beside its proprioceptive entries."""
        if self.image is None:
            return jnp.zeros((batch, self.observation_size))
        image = jnp.zeros((batch, *self.image), jnp.uint8)
        if not self.proprio:
            return image
        return {"image": image, "proprio": jnp.zeros((batch, sum(width for _, width in self.proprio)))}

    def initial(self, batch: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
        if self.dynamics is not None:
            held = self.dynamics
            return jnp.zeros((*batch, held.width)), jnp.zeros((*batch, held.stoch, held.classes))
        return jnp.zeros((*batch, self.deter)), jnp.zeros((*batch, self.stoch, self.classes))

    def seen(self, observation: Observation) -> jax.Array | dict[str, jax.Array]:
        """What the encoder reads of an observation: its `reads` view, or the observation whole; beside an
        image, its `proprio` entries concatenated (`example`'s structure)."""
        read = view(observation, self.reads)
        if not self.proprio:
            return read
        widths = [jnp.asarray(view(observation, name), jnp.float32) for name, _ in self.proprio]
        return {"image": read, "proprio": jnp.concatenate(widths, -1)}

    def predicted(self, name: str, feature: jax.Array) -> jax.Array:
        """The auxiliary head `name`'s prediction of its privileged entry."""
        hidden, out = self.auxiliary_heads[name]
        return out(hidden(feature)).astype(jnp.float32)

    def embed(self, observation: Observation) -> jax.Array:
        """The encoder's tokens of an observation (`seen`): a vector's symlogged first (rssm.py:218), an
        image's pixels through the convolutions, its proprioceptive entries symlogged through an MLP and
        joined to them."""
        if self.proprio:
            image, proprio = view(observation, "image"), view(observation, "proprio")
            return jnp.concatenate([self.encoder(image), self.proprio_encoder(symlog(proprio))], -1)
        read = view(observation)
        return self.encoder(symlog(read) if self.image is None else read)

    def core(self, deter: jax.Array, stoch: jax.Array, action: jax.Array) -> jax.Array:
        """The block GRU's next deterministic state (rssm.py:135-158)."""
        action = action / jax.lax.stop_gradient(jnp.maximum(1, jnp.abs(action)))
        parts = [nn.silu(norm(dense(x))) for dense, norm, x in
                 zip(self.dynamics_in, self.dynamics_norms,
                     (deter, stoch.reshape(*stoch.shape[:-2], -1), action), strict=True)]
        joined = jnp.concatenate(parts, -1)[..., None, :].repeat(self.blocks, -2)
        grouped = deter.reshape(*deter.shape[:-1], self.blocks, -1)
        x = jnp.concatenate([grouped, joined], -1).reshape(*deter.shape[:-1], -1)
        x = nn.silu(self.dynamics_hidden_norm(self.dynamics_hidden(x)))
        gates = self.gru(x).reshape(*deter.shape[:-1], self.blocks, -1)
        reset, candidate, update = (part.reshape(*deter.shape[:-1], -1) for part in jnp.split(gates, 3, -1))
        candidate = jnp.tanh(jax.nn.sigmoid(reset) * candidate)
        update = jax.nn.sigmoid(update - 1)
        # The state carried at its own dtype, whatever the networks compute in.
        return (update * candidate + (1 - update) * deter).astype(deter.dtype)

    def posterior(self, deter: jax.Array | None, tokens: jax.Array) -> jax.Array:
        """The posterior's log-probabilities `[..., stoch, classes]` (rssm.py:75-92); the TSSM's reads
        the tokens alone, the block GRU's the deterministic state too."""
        if self.dynamics is not None:
            return self.dynamics.posterior(tokens)
        if deter is None:
            raise ValueError("the block GRU's posterior reads the deterministic state")
        logits = self.posterior_logits(self.posterior_layers(jnp.concatenate([deter, tokens], -1)))
        return unimix_log_probs(logits.reshape(*logits.shape[:-1], self.stoch, self.classes), self.unimix)

    def prior(self, deter: jax.Array) -> jax.Array:
        if self.dynamics is not None:
            return self.dynamics.prior(deter)
        logits = self.prior_logits(self.prior_layers(deter))
        return unimix_log_probs(logits.reshape(*logits.shape[:-1], self.stoch, self.classes), self.unimix)

    def categorical(self, log_probs: jax.Array) -> jax.Array:
        return onehot_sample(self.make_rng("latent"), log_probs)

    @staticmethod
    def feature(deter: jax.Array, stoch: jax.Array) -> jax.Array:
        return jnp.concatenate([deter, stoch.reshape(*stoch.shape[:-2], -1)], -1)

    def decode(self, deter: jax.Array, stoch: jax.Array) -> jax.Array:
        """The observation decoded from a latent: a vector's symlogged values, an image's pixels in
        [0, 1]."""
        if self.image is None:
            return self.decoded(self.vector_decoder(self.feature(deter, stoch)))
        return self.image_decoder(deter, stoch)

    def reconstruction(self, deter: jax.Array, stoch: jax.Array, observation: Observation) -> jax.Array:
        """The decoder's squared error, summed over the observation: against a vector's symlog, or an
        image's pixels over 255 (rssm.py:345-355, outs.MSE summed over the image's axes); the
        proprioceptive entries are the auxiliary heads' to decode."""
        read = view(observation, "image" if self.proprio else None)
        decoded = self.decode(deter, stoch).astype(jnp.float32)
        if self.image is None:
            return jnp.sum(jnp.square(decoded - symlog(read)), -1)
        return jnp.sum(jnp.square(decoded - divided(read.astype(decoded.dtype), 255)), (-3, -2, -1))

    def reward(self, feature: jax.Array) -> jax.Array:
        return self.reward_logits(self.reward_layers(feature)).astype(jnp.float32)

    def continue_(self, feature: jax.Array) -> jax.Array:
        return self.continue_logit(self.continue_layers(feature))[..., 0].astype(jnp.float32)

    def policy(self, feature: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The bounded normal's mean, tanh of its head, and its deviation between the floor and the
        ceiling (heads.py:146-155)."""
        x = self.policy_layers(feature)
        mean, std = (head(x).astype(jnp.float32) for head in (self.policy_mean, self.policy_std))
        std = (self.max_std - self.min_std) * jax.nn.sigmoid(std + 2.0) + self.min_std
        return jnp.tanh(mean), std


class Critic(nn.Module):
    """The two-hot value head, 3 layers then zero-initialized logits (configs.yaml `value`)."""

    units: int
    bins: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, feature: jax.Array) -> jax.Array:
        logits = nn.Dense(self.bins, kernel_init=_init(0.0), dtype=self.dtype,
                          param_dtype=at_least_fp32(self.dtype))(_Layers(self.units, 3, self.dtype)(feature))
        return logits.astype(jnp.float32)


def _shifted(stoch: jax.Array) -> jax.Array:
    """Each step's previous latent along a window's step axis `[B, T, ...]`, zero before the first."""
    return jnp.concatenate([jnp.zeros_like(stoch[:, :1]), stoch[:, :-1]], 1)


def normal_log_prob(action: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return jnp.sum(-0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * math.log(2 * math.pi), -1)


def normal_entropy(std: jax.Array) -> jax.Array:
    return jnp.sum(0.5 + 0.5 * math.log(2 * math.pi) + jnp.log(std), -1)


def lambda_return(last: jax.Array, term: jax.Array, reward: jax.Array, value: jax.Array, boot: jax.Array,
                  discount: float, lam: float) -> jax.Array:
    """The code's lambda-returns `[B, T - 1]` of sequences `[B, T]` (agent.py:482-490): from the last
    bootstrap backward, each step's reward plus the discounted blend of its bootstrap and the next
    return, cut where the episode ended (`last`) or terminated (`term`)."""
    live = (1 - term)[:, 1:] * discount
    cont = (1 - last)[:, 1:] * lam
    between = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]

    def back(following, inputs):
        step, alive, kept = inputs
        current = step + alive * kept * following
        return current, current

    _, returns = jax.lax.scan(back, boot[:, -1], (between.T, live.T, cont.T), reverse=True)
    return returns.T


@struct.dataclass
class _Latent:
    deter: jax.Array
    stoch: jax.Array
    key: jax.Array
    cache: Cache | None = None


@struct.dataclass
class AgentMemory:
    """An agent's memory between its steps in an environment: the block GRU's deterministic state or the
    TSSM's cache, `held`; the last posterior draw, `stoch`; and the action it took, `previous`."""

    held: jax.Array | Cache
    stoch: jax.Array
    previous: jax.Array


@struct.dataclass
class LatentEnv:
    """The world model as an `Env`: its state the latent and a key, its step the GRU and a draw from
    the prior, its observation the latent's feature, its reward the reward head's two-hot mean and
    its `terminated` never set: the continue head's probability weighs an imagined step instead
    (agent.py:396-405). It starts where it is given (`at`), never by `reset`."""

    model: WorldModel = struct.field(pytree_node=False)
    variables: Variables
    bins: jax.Array

    @property
    def action_size(self) -> int:
        return self.model.action_size

    def at(self, deter: jax.Array, stoch: jax.Array, key: jax.Array,
           cache: Cache | None = None) -> tuple[_Latent, jax.Array]:
        """The latent imagination starts from, with the cache a TSSM's next step attends to."""
        return _Latent(deter, stoch, key, cache), WorldModel.feature(deter, stoch)

    def reset(self, key: jax.Array) -> tuple[_Latent, TimeStep]:
        raise TypeError("an imagination starts from replayed latents (LatentEnv.at), never a reset")

    def step(self, state: _Latent, action: jax.Array) -> tuple[_Latent, TimeStep]:
        key, latent = jax.random.split(state.key)
        model = self.model.bind(self.variables, rngs={"latent": latent})
        cache, dynamics = state.cache, model.dynamics
        if dynamics is None:
            deter = model.core(state.deter, state.stoch, action)
        else:
            if cache is None:
                raise ValueError("a TSSM imagines from a window's cache (LatentEnv.at)")
            never = jnp.zeros(action.shape[:-1], bool)
            deter, cache = dynamics.step(dynamics.inputs(state.stoch, action, never), never, cache)
        stoch = model.categorical(model.prior(deter))
        feature = WorldModel.feature(deter, stoch)
        reward = twohot_mean(model.reward(feature), self.bins)
        never = jnp.zeros(reward.shape, bool)
        return _Latent(deter, stoch, key, cache), TimeStep(feature, reward, never, never)


AUXILIARY_SCALE = 0.5
"""An auxiliary head's loss weight, FlyDreamer's `aux_state` and `aux_goal` (configs.yaml `loss_scales`)."""


class DreamerV3Objective(Objective[Ratio]):
    """DreamerV3 over a `WorldModel` (the module docstring). A batch holds replayed windows `[B, T]`
    of DreamerV3's steps (`Driven`): `observation`, the `action` taken at it, the `reward` that came
    with it, and whether it was its episode's `first`, `last` or `terminal`. The scales weigh the
    losses (configs.yaml `loss_scales`); `horizon` sets the continue target's discount, 1 - 1/333;
    `imagination` is the imagined steps from each replayed state, `lam` the lambda-returns' lambda,
    `entropy` the actor's bonus, `slow_decay` the slow critic's EMA and `return_rate` the rate of
    the returns' percentile EMA. The slow critic is the EMA slot over the critic alone; with
    `slow_targets` the imagined returns bootstrap from it and the advantage is taken over it (configs.yaml
    `slowtar`, agent.py's `tarval`, as SkyDreamer sets it). A `decoder` scale of 0 rebuilds no observation:
    an informed world model's auxiliary heads decode privileged entries alone (SkyDreamer, Sec II-B).
    `smoothness` weighs the squared change of the policy's mean between imagined steps, added to the
    policy's loss (SkyDreamer, arXiv 2510.14783 Eq 11, its 0.002: on the mean alone, so the entropy bonus
    is untouched)."""

    shown: Mapping[str, Shown] = {"loss": Shown(), "episode_return": Shown(better="higher")}
    resumable = ("entropy",)
    """The entropy bonus weighs how the policy explores, which SkyDreamer lowers for its last phase; the
    horizon, lambda and the rest say what the critic and world model estimate."""
    scales: Mapping[str, float] = {"decoder": 1.0, "reward": 1.0, "continue": 1.0, "dynamics": 1.0,
                                   "representation": 0.1, "policy": 1.0, "value": 1.0, "replay_value": 0.3}

    def __init__(self, model: WorldModel, *, horizon: float = 333.0, imagination: int = 15,
                 lam: float = 0.95, entropy: float = 3e-4, free_nats: float = 1.0, slow_decay: float = 0.98,
                 return_rate: float = 0.01, variables: Variables | None = None,
                 scales: Mapping[str, float] | None = None, imitation: float = 0.0,
                 imitates: str = "action", slow_targets: bool = False, smoothness: float = 0.0):
        """`scales` overrides the losses' weights by name, each auxiliary head's its entry's name
        (`AUXILIARY_SCALE` unless set; pretraining from demonstrations sets `policy`, `value` and
        `replay_value` to 0, and imagination is then skipped); `imitation` weighs
        behaviour cloning, the actor's negative log-likelihood of each replayed step's `imitates` field at
        its posterior state, which reaches the actor alone: the `action` taken there, or a teacher's label
        of the step (`dew.rl.aggregation.LABEL`), while the world model reads the action taken. An auxiliary
        head named as one of the objective's own losses is refused: its loss and weight would take that
        loss's place."""
        if imagination < 1 or not 0 <= lam <= 1 or horizon <= 1:
            raise ValueError("imagination must be positive, lam in [0, 1] and the horizon past one step")
        if smoothness < 0:
            raise ValueError("smoothness weighs at least 0")
        if imitates != "action" and not imitation:
            raise ValueError(f"imitates names the field imitation reads ({imitates!r}), and imitation "
                             "weighs 0")
        clashing = sorted({name for name, _, _ in model.auxiliary} & {*type(self).scales, "imitation"})
        if clashing:
            raise ValueError(f"auxiliary heads named as the objective's losses {clashing}: name them apart")
        known = {**type(self).scales, **{name: AUXILIARY_SCALE for name, _, _ in model.auxiliary}}
        unknown = set(scales or {}) - set(known)
        if unknown or imitation < 0:
            raise ValueError(f"scales name the losses {sorted(known)}, not {sorted(unknown)}, "
                             "and imitation weighs at least 0")
        imitating = {"imitation": imitation} if imitation else {}
        self.scales = {**known, **(scales or {}), **imitating}
        self.model = model
        self.horizon, self.imagination, self.lam = horizon, imagination, lam
        self.entropy, self.free_nats, self.return_rate = entropy, free_nats, return_rate
        self.variables, self.imitates = variables, imitates
        self.slow_targets, self.smoothness = slow_targets, smoothness
        self.ema = EMASpec(optax.constant_schedule(slow_decay), under("params", "value"))
        self.bins = twohot_bins(model.bins)

    def held_variables(self) -> Variables | None:
        return self.variables

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        params_key, latent = jax.random.split(key)
        inputs = self.model.example(1), jnp.zeros((1, self.model.action_size))
        tree = self.model.init({"params": params_key, "latent": latent}, *inputs)
        # The returns' 5th and 95th percentiles, an EMA from zero (utils.py:16-72, debias off).
        return {**tree, "returns": {"low": jnp.zeros(()), "high": jnp.zeros(())}}

    @property
    def validation_loss_refusal(self) -> str | None:
        return "DreamerV3 scores its agent by its episodes' returns, not a validation loss"

    def observe(self, model: WorldModel, deter: jax.Array, stoch: jax.Array, observation: Observation,
                previous: jax.Array, first: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One posterior step: the latent and action masked where an episode starts, the GRU, then
        the posterior's draw; the new latent, beside the posterior's log-probabilities."""
        keep = (~first)[..., None]
        deter, previous = deter * keep, previous * keep
        stoch = stoch * keep[..., None]
        deter = model.core(deter, stoch, previous)
        posterior = model.posterior(deter, model.embed(observation))
        return deter, model.categorical(posterior), posterior

    def observed(self, variables: Variables, observation: Observation, previous: jax.Array, first: jax.Array,
                 key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The posterior latents over windows `[B, T, ...]` from a zero latent: each step's
        deterministic state, drawn categoricals and posterior log-probabilities, each step drawing from
        its own of `T` keys split from `key`, as the code's scan seeds each step's sample
        (rssm.py:63-90)."""
        rows, length = first.shape
        model = self.model.bind(variables, rngs={"latent": key})
        dynamics = model.dynamics
        if dynamics is not None:
            # TransDreamer: the window's posteriors at once, then the transformer over the latents and
            # actions before each step.
            posterior = model.posterior(None, model.embed(observation))
            stoch = model.categorical(posterior)
            deter = dynamics.states(dynamics.inputs(_shifted(stoch), previous, first), first)
            return deter, stoch, posterior

        def step(latent, inputs):
            deter, stoch = latent
            observation, previous, first, key = inputs
            at = self.model.bind(variables, rngs={"latent": key})
            deter, stoch, posterior = self.observe(at, deter, stoch, observation, previous, first)
            return (deter, stoch), (deter, stoch, posterior)

        steps = (*jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), (observation, previous, first)),
                 jax.random.split(key, length))
        _, found = jax.lax.scan(step, self.model.initial((rows,)), steps)
        deter, stoch, posterior = (jnp.swapaxes(x, 0, 1) for x in found)
        return deter, stoch, posterior

    def act(self, variables: Params, observed: Observation, key: jax.Array,
            memory: Memory) -> tuple[jax.Array, Tree, AgentMemory]:
        """The agent's step in an environment (`dew.rl.env.Policy` over `Driven`): the posterior
        given the step's observation, then an action drawn from the policy, clipped to [-1, 1]."""
        return self._acted(variables, observed, key, memory, drawn=True)

    def greedy(self, variables: Params, observed: Observation, key: jax.Array,
               memory: Memory) -> tuple[jax.Array, Tree, AgentMemory]:
        """`act`'s step with the policy's mean in place of its draw, as a control objective's `greedy`
        scores a policy: the latent is still the posterior's draw, as the world model reads it."""
        return self._acted(variables, observed, key, memory, drawn=False)

    def _acted(self, variables: Params, observed: Observation, key: jax.Array, memory: Memory, *,
               drawn: bool) -> tuple[jax.Array, Tree, AgentMemory]:
        if not isinstance(memory, AgentMemory):
            raise TypeError("an agent acts from its own memory (DreamerV3Objective.memory)")
        latent, draw = jax.random.split(key)
        model = self.model.bind(variables, rngs={"latent": latent})
        step = views(observed)
        seen, first = model.seen(step["observation"]), view(step, "first")
        held, dynamics = memory.held, model.dynamics
        if dynamics is None:
            if isinstance(held, Cache):
                raise TypeError("a block GRU's agent remembers its deterministic state, not a cache")
            deter, stoch, _ = self.observe(model, held, memory.stoch, seen, memory.previous, first)
            held = deter
        else:
            if not isinstance(held, Cache):
                raise TypeError("a TSSM's agent remembers its cache")
            deter, held = dynamics.step(dynamics.inputs(memory.stoch, memory.previous, first), first, held)
            stoch = model.categorical(model.posterior(None, model.embed(seen)))
        mean, std = model.policy(WorldModel.feature(deter, stoch))
        action = jnp.clip(mean + std * jax.random.normal(draw, mean.shape) if drawn else mean, -1, 1)
        return action, {}, AgentMemory(held, stoch, action)

    def memory(self, count: int) -> AgentMemory:
        """The agent's memory at the start for each of `count` environments: a zero latent, or a TSSM's
        empty cache of its `context` steps, and a zero previous latent and action."""
        deter, stoch = self.model.initial((count,))
        dynamics = self.model.dynamics
        held = deter if dynamics is None else dynamics.empty_cache(count, dynamics.context)
        return AgentMemory(held, stoch, jnp.zeros((count, self.model.action_size)))

    def model_flops(self, model, batch: Batch) -> float:
        """One update's forward work on `batch`, its loss's traced program's (`traced_update_flops`)."""
        return traced_update_flops(self, batch)[0]

    def training_flops(self, model, batch: Batch) -> float:
        """One update's work on `batch`, its loss's gradient's traced program's (`traced_update_flops`)."""
        return traced_update_flops(self, batch)[1]

    def loss(self, variables: Variables, batch: Batch, step: Step) -> tuple[Ratio, Aux]:
        world_key, imagine_key = jax.random.split(step.key)
        model = self.model.bind(variables, rngs={"latent": world_key})
        observation, action = model.seen(batch["observation"]), batch["action"]
        reward = jnp.asarray(batch["reward"], jnp.float32)
        # A window starts its episode's latent afresh at its first step.
        first = jnp.asarray(batch["first"], bool).at[:, 0].set(True)
        last, terminal = jnp.asarray(batch["last"], jnp.float32), jnp.asarray(batch["terminal"], jnp.float32)
        previous = jnp.concatenate([jnp.zeros_like(action[:, :1]), action[:, :-1]], 1)
        rows, length = reward.shape

        deter, stoch, posterior = self.observed(variables, observation, previous, first, world_key)
        prior = model.prior(deter)
        feature = WorldModel.feature(deter, stoch)
        # `kl_divergence(logits, reference)` is KL(reference || logits): the code's dynamics loss
        # KL(sg(post) || prior) and representation loss KL(post || sg(prior)) (rssm.py:125-126).
        losses = {
            "dynamics": jnp.maximum(kl_divergence(prior, jax.lax.stop_gradient(posterior)).sum(-1),
                                    self.free_nats),
            "representation": jnp.maximum(kl_divergence(jax.lax.stop_gradient(prior), posterior).sum(-1),
                                          self.free_nats),
            "reward": twohot_loss(model.reward(feature), reward, self.bins),
            # An informed world model decodes privileged entries in the observation's place (scale 0).
            **({"decoder": model.reconstruction(deter, stoch, observation)}
               if self.scales["decoder"] else {}),
        }
        alive = (1 - terminal) * (1 - 1 / self.horizon)
        logit = model.continue_(feature)
        losses["continue"] = optax.losses.sigmoid_binary_cross_entropy(logit, alive)
        for name, _, _ in self.model.auxiliary:
            target = jnp.asarray(batch["observation"][name], jnp.float32)
            losses[name] = jnp.sum(jnp.square(model.predicted(name, feature) - target), -1)
        if "imitation" in self.scales:
            # Each replayed step's imitated action at the state it was taken at; a last step's has no effect.
            mean, std = model.policy(jax.lax.stop_gradient(feature))
            imitated = jnp.asarray(batch[self.imitates], jnp.float32)
            losses["imitation"] = (1 - last) * -normal_log_prob(imitated, mean, std)
        if not any(self.scales[name] for name in ("policy", "value", "replay_value")):
            total = sum((self.scales[name] * jnp.mean(value) for name, value in losses.items()),
                        jnp.zeros(()))
            metrics = {f"loss_{name}": jnp.mean(value) for name, value in losses.items()}
            return Ratio(total, jnp.ones(())), Aux(metrics, variables={"returns": variables["returns"]})

        # Imagination from every replayed posterior state, its inputs held still (agent.py:189-196).
        env = LatentEnv(self.model, jax.lax.stop_gradient(variables), self.bins)
        start_deter, start_stoch = (jax.lax.stop_gradient(x.reshape(rows * length, *x.shape[2:]))
                                    for x in (deter, stoch))
        latent_key, act_key, last_key = jax.random.split(imagine_key, 3)
        cache, dynamics = None, model.dynamics
        if dynamics is not None:
            # Each start's cache, the window's keys and values of its episode up to it.
            inputs = dynamics.inputs(_shifted(stoch), previous, first)
            cache = jax.lax.stop_gradient(dynamics.window_cache(inputs, first, self.imagination))
        state, begin = env.at(start_deter, start_stoch, latent_key, cache)

        def policy(params: Params, feature: Observation, key: jax.Array, memory: Memory) -> tuple[
                jax.Array, Tree, Memory]:
            mean, std = self.model.bind(params).policy(view(feature))
            return mean + std * jax.random.normal(key, mean.shape), {}, None

        held = jax.lax.stop_gradient(variables)
        _, imagined = env_rollout(env, policy, held, Carry(state, begin, act_key), self.imagination,
                                  reset=False)
        # The start's feature then each imagined one, and the action taken at each [B*T, H + 1].
        imagined_features = jnp.swapaxes(view(imagined.following.observation), 0, 1)
        features = jnp.concatenate([begin[:, None], imagined_features], 1)
        last_action, _, _ = policy(held, features[:, -1], last_key, None)
        actions = jnp.concatenate([jnp.swapaxes(imagined.action, 0, 1), last_action[:, None]], 1)
        still = self.model.bind(held)
        rewards = twohot_mean(still.reward(features), self.bins)
        cont = jax.nn.sigmoid(still.continue_(features))
        values = twohot_mean(model.value(features), self.bins)
        slow = self.model.bind(merge(variables, jax.lax.stop_gradient(step.ema)) if step.ema is not None
                               else held)
        slow_values = twohot_mean(slow.value(features), self.bins)
        weight = jax.lax.stop_gradient(jnp.cumprod(cont, 1))
        boot = slow_values if self.slow_targets else values
        returns = jax.lax.stop_gradient(lambda_return(jnp.zeros_like(cont), 1 - cont, rewards, boot, boot,
                                                      1.0, self.lam))
        rate, held_returns = self.return_rate, variables["returns"]
        low = (1 - rate) * held_returns["low"] + rate * jnp.percentile(returns, 5)
        high = (1 - rate) * held_returns["high"] + rate * jnp.percentile(returns, 95)
        scale = jnp.maximum(1.0, high - low)
        advantage = jax.lax.stop_gradient((returns - boot[:, :-1]) / scale)
        imagined_policy = self._imagined_policy(model.policy(features), actions, advantage, weight)
        value_logits = model.value(features)
        targets = jnp.concatenate([returns, jnp.zeros_like(returns[:, -1:])], 1)
        imagined_value = weight[:, :-1] * (twohot_loss(value_logits, targets, self.bins)
                                           + twohot_loss(value_logits, slow_values, self.bins))[:, :-1]
        losses["policy"] = imagined_policy.mean(1).reshape(rows, length)
        losses["value"] = imagined_value.mean(1).reshape(rows, length)

        # The critic toward the replay's own lambda-returns, bootstrapped by imagination's (agent.py:449-480).
        replay_logits = model.value(feature)
        replay_values = twohot_mean(replay_logits, self.bins)
        boot = returns[:, 0].reshape(rows, length)
        replay_returns = jax.lax.stop_gradient(lambda_return(last, terminal, reward, replay_values, boot,
                                                             1 - 1 / self.horizon, self.lam))
        replay_slow = twohot_mean(slow.value(feature), self.bins)
        replay_targets = jnp.concatenate([replay_returns, jnp.zeros_like(replay_returns[:, -1:])], 1)
        replay_value = (twohot_loss(replay_logits, replay_targets, self.bins)
                        + twohot_loss(replay_logits, replay_slow, self.bins))
        losses["replay_value"] = (1 - last)[:, :-1] * replay_value[:, :-1]

        total = sum((self.scales[name] * jnp.mean(value) for name, value in losses.items()), jnp.zeros(()))
        metrics = {f"loss_{name}": jnp.mean(value) for name, value in losses.items()}
        metrics.update(return_scale=scale, imagined_return=jnp.mean(returns))
        stats = {"returns": {"low": jax.lax.stop_gradient(low), "high": jax.lax.stop_gradient(high)}}
        return Ratio(total, jnp.ones(())), Aux(metrics, variables=stats)

    def _imagined_policy(self, policy: tuple[jax.Array, jax.Array], actions: jax.Array, advantage: jax.Array,
                         weight: jax.Array) -> jax.Array:
        """The actor's loss at each imagined step `[B*T, H]`: REINFORCE on the advantage with the entropy
        bonus (agent.py:420-440), and where `smoothness` weighs it each step's mean against the one before
        it (SkyDreamer Eq 11, on the mean alone)."""
        mean, std = policy
        log_pi = normal_log_prob(jax.lax.stop_gradient(actions), mean, std)[:, :-1]
        loss = weight[:, :-1] * -(log_pi * advantage + self.entropy * normal_entropy(std)[:, :-1])
        if self.smoothness:
            loss = loss + self.smoothness * jnp.sum(jnp.square(mean[:, 1:] - mean[:, :-1]), -1)
        return loss


class Saved(TypedDict):
    """What a `DreamerRollout` carries between calls, as a checkpoint holds it (`DreamerRollout.saved`):
    the environment steps it took and those of them its prefills took (each two 32-bit words, `_words`),
    where it owes its updates from (`paced`: the updates and the steps owed for them, an update's replayed
    steps and the train ratio's float64 bits, each two words) and with a checkpointed replay its
    environments, replay and returns."""

    taken: np.ndarray
    prefilled: np.ndarray
    paced: NotRequired[np.ndarray]
    carry: NotRequired[Carry]
    replay: NotRequired[ReplayBuffer]
    returns: NotRequired[jax.Array]


def _kept(replay: ReplayBuffer, returns: jax.Array,
          recorded: Transition) -> tuple[ReplayBuffer, jax.Array, jax.Array, jax.Array]:
    """The replay with a rollout's steps written, each environment's running return, and the sum and
    count of the returns of the episodes that ended in it. A step is written with its teacher's label
    where its policy gave one (`dew.rl.aggregation.labelled`)."""
    observed = views(recorded.observation)
    extras = recorded.extras
    label = {LABEL: extras[LABEL]} if isinstance(extras, Mapping) and LABEL in extras else {}
    replay = replay.add({**observed, "action": recorded.action, **label})

    def tally(held, observed):
        total = jnp.where(observed["first"], 0, held) + observed["reward"]
        return total, (total, observed["last"])

    returns, (totals, ended) = jax.lax.scan(tally, returns, observed)
    return replay, returns, jnp.sum(jnp.where(ended, totals, 0)), jnp.sum(ended)


@dataclass
class DreamerRollout:
    """The trainer's rollout of a `DreamerV3Objective` (`dew.training.rollout.Rollout`): its agent acts
    in `env`, a batched `Env` driven as `Driven` drives it or a `Simulator` driven alike
    (`DrivenSimulator`), writing each step to a device replay of `capacity` steps an environment; each
    update trains on `batch` windows of `length` replayed steps while the agent takes environment steps
    so that each replays `train_ratio` steps (the code's `run.train_ratio`), after `prefill` steps of
    each environment taken uniformly at random. A `replay` given (demonstrations recorded in the
    agent's convention from as many environments, `[capacity, envs, ...]`) is the replay it starts from
    and writes on, in place of the prefill. `metrics` reports the mean return of the episodes that ended
    since the last update. Given a privileged `teacher` (`dew.rl.aggregation.Teacher`), a policy of the
    environment's observation, every step the rollout takes, the prefill's too, is written with the
    teacher's action at it, its label (`dew.rl.aggregation.labelled`), which an objective that `imitates`
    it learns: DAgger, the replay aggregating the student's steps beside the teacher's own flights it
    was given, labelled alike. Given a `demonstrator` (a `Teacher` too, a policy of the environment's
    observation, such as a teacher flying its policy's draws, DART), it flies the prefill in place of the
    uniform actions, so the replay starts from its demonstrations, labelled where there is a teacher; the
    agent's memory passes through its steps untouched, as through the uniform ones. The prefill is flown
    `length` steps at a time, so a long prefill of images holds one window's flight beside the replay, not
    its whole.

    The run's checkpoints hold the environment steps it has taken (`saved`), so a resumed run takes the
    steps it still owes; its environments begin again and its replay is made again, from `replay` or the
    prefill. A rollout owes each update from the step it goes on from at its own pace, so a run resumed
    with another batch, length or train ratio (a later phase's longer windows) takes the steps of its
    updates from there at that pace alone. Resumed from a step that holds none of its state (one an
    earlier Dew saved), it owes the steps of the updates after that step alone. A cooldown branch runs on
    copies of its environments and replay (`branch`), a `Simulator` keeping its own
    (`dew.rl.simulator.branched`). With `checkpointed_replay`, its checkpoints also hold its replay, its
    environments mid episode and their running returns, and a resumed run samples what the uninterrupted
    run did, at the replay's size a checkpoint (a JAX `Env`'s alone: a `Simulator`'s environments live
    outside JAX)."""

    objective: DreamerV3Objective
    env: Env | Simulator
    capacity: int = 100_000
    batch: int = 16
    length: int = 64
    train_ratio: float = 1024.0
    prefill: int = 64
    replay: ReplayBuffer | None = None
    checkpointed_replay: bool = False
    teacher: Teacher | None = None
    demonstrator: Teacher | None = None
    _live: _Live | None = field(default=None, init=False, repr=False)
    _taken: int = field(default=0, init=False, repr=False)
    _prefilled: int = field(default=0, init=False, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)
    _paced: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _recorded: tuple[int, int, int, float] | None = field(default=None, init=False, repr=False)
    _ended: tuple[jax.Array | float, jax.Array | int] = field(default=(0.0, 0), init=False, repr=False)
    _driven: DrivenSimulator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.replay is None and self.prefill < self.length:
            raise ValueError("the prefill must hold at least one window")
        if self.checkpointed_replay and isinstance(self.env, Simulator):
            raise ValueError("a checkpointed replay resumes with its environments mid episode, and a "
                             "Simulator's live outside JAX: its run's checkpoints hold the steps it took "
                             "alone")
        if self.replay is not None and self.demonstrator is not None:
            raise ValueError("a demonstrator flies the prefill, which a replay given to the rollout replaces")
        held = None if self.replay is None else self.replay.held
        if held is not None and (isinstance(held, Mapping) and LABEL in held) != (self.teacher is not None):
            raise ValueError("a replay given to a rollout holds the steps it writes: each with its teacher's "
                             f"label ({LABEL!r}) where it has a teacher, none where it has none")
        env = self.env
        if isinstance(env, Simulator):
            driven = self._driven = DrivenSimulator(env)
            self._begin = functools.partial(simulator_start, driven)
            self._roll = functools.partial(self._simulated, driven)
        else:
            stepped = Driven(env)
            self._begin = functools.partial(start, stepped)
            self._roll = functools.partial(self._flown, stepped)

    @property
    def metrics(self) -> Mapping[str, float]:
        total, count = float(self._ended[0]), int(self._ended[1])
        return {"episode_return": total / count if count else float("nan")}

    @property
    def environment_steps(self) -> int:
        """The steps every environment has taken so far, summed."""
        return self._taken

    def saved(self) -> Saved | None:
        """What the run's checkpoints hold of the rollout (`Rollout`'s `saved`): the environment steps it
        has taken and those its prefills took, and with `checkpointed_replay` its replay, its environments
        mid episode and their running returns; None before its first call."""
        live = self._live
        if live is None:
            return None
        counts = Saved(taken=_words(self._taken), prefilled=_words(self._prefilled))
        # Where it owes from, or where the checkpoint it was restored from did, before its first call.
        paced = self._recorded if self._paced is None else (*self._paced, self.batch * self.length,
                                                             self.train_ratio)
        if paced is not None:
            since, owed, windows, ratio = paced
            counts["paced"] = np.stack([_words(since), _words(owed), _words(windows), _words(_bits(ratio))])
        if self.checkpointed_replay:
            counts.update(carry=live.carry, replay=live.replay, returns=live.returns)
        return counts

    def saved_shapes(self) -> Mapping[str, object]:
        """`saved`'s leaves' shapes and dtypes, before any call."""
        counted = jax.ShapeDtypeStruct((2,), np.uint32)
        taken = {"taken": counted, "prefilled": counted, "paced": jax.ShapeDtypeStruct((4, 2), np.uint32)}
        env = self.env
        if not self.checkpointed_replay or isinstance(env, Simulator):
            return taken
        key = jax.random.key(0)
        count = env.reset(key)[1].reward.shape[0]
        carry = jax.eval_shape(lambda key: self._begin(key, self.objective.memory(count)), key)
        example = self._step_example(carry.observation, count)
        # The replay the rollout was given, or the one its prefill fills.
        replay = (jax.eval_shape(lambda: self.replay) if self.replay is not None
                  else jax.eval_shape(lambda example: ReplayBuffer.empty(example, self.capacity), example))
        returns = jax.ShapeDtypeStruct((count,), jnp.float32)
        return {**taken, "carry": carry, "replay": replay, "returns": returns}

    def restored(self, tree: Saved) -> None:
        """The rollout carrying on from a `saved` tree: owing the steps after those it had taken, and with
        `checkpointed_replay` from its replay and its environments as they stood."""
        self._taken, self._prefilled, self._restored = _count(tree["taken"]), _count(tree["prefilled"]), True
        self._paced, self._recorded = None, None
        if "paced" in tree:
            since, owed, windows, bits = (_count(words) for words in np.asarray(tree["paced"]))
            self._recorded = (since, owed, windows, _ratio(bits))
        if "replay" in tree and "carry" in tree and "returns" in tree:
            self._live = _Live(tree["carry"], tree["replay"], tree["returns"])

    def _step_example(self, observation: Observation, count: int) -> Steps:
        """A step as the replay holds it over `count` environments: the observation's views, the action
        taken, and the teacher's label of it where there is a teacher."""
        action = jnp.zeros((count, self.env.action_size), jnp.float32)
        return {**views(observation), "action": action, **({} if self.teacher is None else {LABEL: action})}

    @property
    def held_replay(self) -> ReplayBuffer | None:
        """The replay the rollout writes and samples, None before its first call."""
        return None if self._live is None else self._live.replay

    def _uniform(self, params: Params, observed: Observation, key: jax.Array,
                 memory: Memory) -> tuple[jax.Array, Tree, Memory]:
        """The prefill's policy: actions uniform in [-1, 1]."""
        shape = (view(observed, "reward").shape[0], self.env.action_size)
        return jax.random.uniform(key, shape, minval=-1, maxval=1), {}, memory

    def _policy(self, random: bool) -> Policy:
        """The collection's policy: for the prefill the demonstrator's where there is one, else uniform; else
        the agent's draws; each step labelled by the teacher where there is one, the policy then reading
        what acts beside the teacher's variables (`_acting`)."""
        acting = self._uniform if random else self.objective.act
        demonstrator = self.demonstrator
        if random and demonstrator is not None:
            def demonstrated(params: Params, observed: Observation, key: jax.Array, memory: Memory):
                # The demonstrator acts in the environment, from the observation the driven step holds; the
                # agent's memory passes through, as the uniform prefill passes it.
                action, _, _ = demonstrator.policy(params, views(observed)["observation"], key, None)
                return action, {}, memory

            acting = demonstrated
        if self.teacher is None:
            return acting
        teacher = self.teacher.policy

        def taught(params: Params, observed: Observation, key: jax.Array, memory: Memory):
            # The teacher acts in the environment: it reads the observation the driven step holds.
            return teacher(params, views(observed)["observation"], key, memory)

        return labelled(acting, taught)

    def _acting(self, variables: Variables, random: bool) -> Variables:
        """What the collection's policy reads (`_policy`): the agent's variables, or the demonstrator's
        for the prefill it flies, beside the teacher's where there is one (`dew.rl.aggregation.paired`)."""
        acting = self.demonstrator.variables if random and self.demonstrator is not None else variables
        return acting if self.teacher is None else paired(acting, self.teacher.variables)

    def _stepped(self, env: Env, variables: Variables, live: _Live, *, steps: int,
                 random: bool) -> tuple[_Live, jax.Array, jax.Array]:
        """`live` after `steps` steps of every environment of `env`, one scan, and the ended episodes'
        returns' sum and count."""
        carry, recorded = env_rollout(env, self._policy(random), variables, live.carry, steps, reset=False)
        replay, returns, total, count = _kept(live.replay, live.returns, recorded)
        return _Live(carry, replay, returns), total, count

    def _simulated(self, simulator: DrivenSimulator, variables: Variables, live: _Live, *, steps: int,
                   random: bool) -> tuple[_Live, jax.Array, jax.Array]:
        """`live` after `steps` steps of every environment of `simulator`, stepped from the host, and the
        ended episodes' returns' sum and count. The replay is written in place: its buffer is donated to
        the program that writes it."""
        carry, steps_taken = simulator_rollout(simulator, self._policy(random), variables, live.carry, steps,
                                               reset=False, models=self.objective.towers())
        replay, returns, total, count = program(_kept, donate_argnames=("replay",))(
            live.replay, live.returns, steps_taken)
        return _Live(carry, replay, returns), total, count

    def _flown(self, stepped: Driven, variables: Variables, live: _Live, *, steps: int,
               random: bool) -> tuple[_Live, jax.Array, jax.Array]:
        """`_stepped` over `stepped` as a program of the objective's (`Objective.program`), as it runs the
        policy's modules, the live state donated to it so the replay is written in place."""
        rolled = self.objective.program(
            "rollout", functools.partial(self._stepped, stepped), holds=(stepped,),
            static_argnames=("steps", "random"), donate_argnames=("live",))
        return rolled(variables, live, steps=steps, random=random)

    def _started(self, variables: Variables, key: jax.Array) -> _Live:
        """Every environment's first episode begun, and the replay: the one given, or a fresh one the
        prefill fills."""
        count = self.env.count if isinstance(self.env, Simulator) else self.env.reset(key)[1].reward.shape[0]
        # The live state is donated to the steps that advance it, each buffer once, so each leaf gets a buffer
        # of its own: an environment's state may hold one array in two of its fields (Playground's do). Each
        # holds its dtype strongly, as a restored state's leaves do: a weakly typed one (a Python number a
        # reset folds in, the flight race's `ramp`) lowers the rollout to another program than a resumed
        # run's, which compiles again (ledger 3451).
        carry = jax.tree.map(lambda leaf: jnp.array(leaf, dtype=leaf.dtype),
                             self._begin(key, self.objective.memory(count)))
        returns = jnp.zeros(count)
        if self.replay is not None:
            return _Live(carry, self.replay, returns)
        # No update owes the prefill's steps; a resumed run that fills its replay again takes them again.
        self._prefilled += self.prefill * count
        example = self._step_example(carry.observation, count)
        live = _Live(carry, ReplayBuffer.empty(example, self.capacity), returns)
        for flown in range(0, self.prefill, self.length):
            live = self._advanced(variables, live, min(self.length, self.prefill - flown), random=True)
        return live

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        act_key, sample_key = jax.random.split(key)
        variables = state.variables
        updates = int(state.updates)
        if self._paced is None and not self._restored:
            # A run this rollout begins mid way (resumed from a step that holds no state of it) owes the
            # steps of the updates after that step alone: those before it were taken by the run it resumes.
            self._taken = self._paced_steps(updates)
        live = self._started(variables, act_key) if self._live is None else self._live
        envs = live.returns.shape[0]
        if self._paced is None:
            self._paced = self._anchored(updates, envs)
        due = max(self._owed(updates + 1, envs) - (self._taken - self._prefilled), 0)
        if due:
            live = self._advanced(variables, live, -(-due // envs), random=False)
        self._live = live
        # The update's windows stay on the device the replay is held on, which the trainer places from: the
        # host does not wait to fetch them (a FlyDreamer update's 56 MB of images went there and back).
        return self._drawn(sample_key)

    def _anchored(self, updates: int, envs: int) -> tuple[int, int]:
        """Where the rollout owes its updates from (`_owed`), at its first call with `updates` committed: the
        updates and the steps owed for them. Restored, its checkpoint's anchor where it goes on at the pace
        that saved it, so the steps it took for an update not yet committed (in an accumulation window, or
        one the trainer rejected) stand paid; at another pace (a later phase's longer windows), the steps
        that pace owed for `updates`, from which this one owes the updates after them. A checkpoint that
        holds no anchor (an earlier Dew's, its run at one pace) has its pace's steps for `updates` within
        one update of those taken, else the nearest. Begun afresh, or mid way without its state, it owes
        from the steps it has."""
        taken = self._taken - self._prefilled
        windows = self.batch * self.length
        if self._recorded is not None:
            since, owed, saved_windows, ratio = self._recorded
            if (saved_windows, ratio) == (windows, self.train_ratio):
                return since, owed
            return updates, owed + _whole(updates, saved_windows, ratio, envs) - _whole(
                since, saved_windows, ratio, envs)
        if not self._restored:
            return updates, taken
        owed = _whole(updates, windows, self.train_ratio, envs)
        step = _whole(updates + 1, windows, self.train_ratio, envs) - owed
        return updates, min(max(owed, taken - step), taken)

    def _owed(self, updates: int, envs: int) -> int:
        """The environment steps owed for `updates` updates over `envs` environments stepped together: those
        owed where the rollout owes from (`_anchored`), and from there the steps of its updates at its
        pace (`_whole`)."""
        assert self._paced is not None
        since, owed = self._paced
        windows = self.batch * self.length
        return owed + _whole(updates, windows, self.train_ratio, envs) - _whole(
            since, windows, self.train_ratio, envs)

    def _paced_steps(self, updates: int) -> int:
        """The environment steps `updates` updates train on at the rollout's batch, length and train ratio,
        rounded up."""
        return _whole(updates, self.batch * self.length, self.train_ratio, 1)

    @contextlib.contextmanager
    def branch(self, name: str) -> Iterator[None]:
        """A cooldown branch's calls (`Rollout`'s `branch`), on copies of the rollout's environments and
        replay, after which the rollout stands as the run left it: its live state is donated to the steps
        that advance it, so the run's is copied aside first, at the replay's size again for the branch; a
        begun `Simulator` keeps its own environments (`DrivenSimulator.branch`)."""
        kept = (None if self._live is None else jax.tree.map(jnp.copy, self._live), self._taken,
                self._prefilled, self._restored, self._paced, self._recorded, self._ended)
        driven = self._driven if self._live is not None else None
        try:
            with contextlib.nullcontext() if driven is None else driven.branch():
                yield
        finally:
            (self._live, self._taken, self._prefilled, self._restored, self._paced, self._recorded,
             self._ended) = kept

    def sample(self, key: jax.Array) -> Batch:
        """`batch` windows of `length` steps drawn from the replay as an update draws them, fetched to the
        host."""
        return jax.device_get(self._drawn(key))

    def _drawn(self, key: jax.Array) -> Batch:
        """`sample`'s windows on the device the replay is held on."""
        replay = self.held_replay
        if replay is None:
            raise ValueError("a rollout samples its replay once it has begun (its first call)")
        key = jax.device_put(key, jax.tree.leaves(replay.held)[0].sharding)
        return program(ReplayBuffer.sample, static_argnames=("batch", "length"))(
            replay, key, batch=self.batch, length=self.length)

    def _advanced(self, variables: Variables, live: _Live, steps: int, *, random: bool) -> _Live:
        live, total, count = self._roll(self._acting(variables, random), live, steps=steps, random=random)
        self._taken += steps * live.returns.shape[0]
        # Read when the metrics are, so the host does not wait for each advance to finish.
        self._ended = (total, count)
        return live


def _words(count: int) -> np.ndarray:
    """A count of steps as two 32-bit words, high then low, which a checkpoint keeps exactly whether or
    not JAX runs in 64 bits (`_count`)."""
    return np.asarray([count >> 32, count & 0xFFFFFFFF], np.uint32)


def _count(words) -> int:
    """The count `_words` wrote."""
    high, low = (int(word) for word in np.asarray(words, np.uint64))
    return high << 32 | low


def _bits(ratio: float) -> int:
    """A train ratio's float64 bits, which `_words` keeps exactly (`_ratio`)."""
    return int(np.asarray(ratio, np.float64).view(np.uint64))


def _ratio(bits: int) -> float:
    """The train ratio `_bits` wrote."""
    return float(np.asarray(bits, np.uint64).view(np.float64))


def _whole(updates: int, windows: int, ratio: float, envs: int) -> int:
    """The environment steps `updates` updates of `windows` replayed steps each train on at the train
    `ratio`, rounded up, then to whole steps of every one of `envs` environments stepped together."""
    return envs * -(-math.ceil(updates * windows / ratio) // envs)


@struct.dataclass
class _Live:
    """A rollout's state between collections: every environment's carry, the replay, and each
    environment's running return."""

    carry: Carry
    replay: ReplayBuffer
    returns: jax.Array


def demonstration_replay(episodes: Sequence[Mapping[str, np.ndarray]]) -> ReplayBuffer:
    """Demonstrations as a replay to pretrain from, one stream of every episode in order, in the
    convention the agent records its own steps (`Driven`): an episode's row t holds the observation t,
    the reward that came with it (none at its first), and the action taken at it, after it was seen
    (FlyDreamer's recorder stored each action one row later, with the observation it produced, and so
    shifted every action-conditioned prediction). An episode is `observation`, `action` and `reward`
    along its steps (an observation an array or a dict of them), and `terminal` where it ended in a
    terminal state (only its last step can)."""
    rows = []
    for episode in episodes:
        length = len(episode["reward"])
        terminal = np.asarray(episode.get("terminal", np.zeros(length, bool)), bool)
        leaves = jax.tree.leaves((episode["observation"], episode["action"]))
        if terminal[:-1].any() or any(len(leaf) != length for leaf in leaves):
            raise ValueError("an episode's fields hold one row a step, and only its last step is terminal")
        steps = np.arange(length)
        rows.append({"observation": jax.tree.map(np.asarray, episode["observation"]),
                     "action": np.asarray(episode["action"]),
                     "reward": np.asarray(episode["reward"], np.float32), "first": steps == 0,
                     "last": steps == length - 1, "terminal": terminal})
    stream = jax.tree.map(lambda *parts: np.concatenate(parts)[:, None], *rows)
    example = jax.tree.map(lambda leaf: leaf[0], stream)
    return ReplayBuffer.empty(example, len(stream["reward"])).add(stream)


def episodes_return(objective: DreamerV3Objective, env: Env | Simulator, variables: Variables, key: jax.Array,
                    steps: int) -> float:
    """The mean undiscounted return of the first episode of each of `env`'s environments, an `Env` or a
    `Simulator`, over `steps` steps of the agent's draws from fresh episodes: the whole episode where
    `steps` is its length (`dew.rl.evaluation.episodes`, the environment driven for the agent)."""
    driven = DrivenSimulator(env) if isinstance(env, Simulator) else Driven(env)
    return episodes(objective.act, driven, steps, memory=objective.memory)(variables, key)["return"]


__all__ = ["AUXILIARY_SCALE", "AgentMemory", "ConvDecoder", "ConvEncoder", "Critic", "DreamerRollout",
           "DreamerV3Objective", "LatentEnv", "WorldModel", "demonstration_replay", "episodes_return",
           "lambda_return", "normal_entropy", "normal_log_prob"]
