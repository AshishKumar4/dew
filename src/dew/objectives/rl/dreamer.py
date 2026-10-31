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

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct

from dew.batch import Batch, Variables, merge, under
from dew.objectives.base import Aux, EMASpec, Objective, Ratio, Shown, Step
from dew.objectives.rl.tssm import TSSM, Cache, parameters_of
from dew.rl.distributions import (
    categorical_kl,
    onehot_sample,
    symlog,
    twohot_bins,
    twohot_loss,
    twohot_mean,
    unimix_log_probs,
)
from dew.rl.env import Carry, Driven, Env, Observation, TimeStep, Transition, env_rollout, start
from dew.rl.replay import ReplayBuffer
from dew.rl.simulator import DrivenSimulator, Simulator, device_of, simulator_rollout, simulator_start

if TYPE_CHECKING:
    from dew.state import TrainState


def _init(scale: float = 1.0) -> nn.initializers.Initializer:
    """The code's `trunc_normal_in`: a standard normal truncated to two deviations, uncorrected, times
    1.1368 sqrt(1 / fan in), times `scale` (nets.py:169-175), the fan in every axis but the output's
    (`compute_fans`, nets.py:187-197): a dense kernel's inputs, a convolution's window of input
    channels, a block-diagonal kernel's inputs over all its blocks."""
    if scale == 0:
        return nn.initializers.zeros

    def init(key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> jax.Array:
        deviation = 1.1368 * scale / math.sqrt(math.prod(shape[:-1]))
        return (deviation * jax.random.truncated_normal(key, -2, 2, shape)).astype(dtype)

    return init


class _Layers(nn.Module):
    """`count` layers of a linear map to `units`, an RMS norm and SiLU (nets.py:565)."""

    units: int
    count: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        held = parameters_of(self.dtype)
        for _ in range(self.count):
            x = nn.Dense(self.units, kernel_init=_init(), dtype=self.dtype, param_dtype=held)(x)
            x = nn.silu(nn.RMSNorm(epsilon=1e-4, dtype=self.dtype, param_dtype=held)(x))
        return x


class _BlockLinear(nn.Module):
    """A block-diagonal linear map of `blocks` blocks to `features` (nets.py:254)."""

    features: int
    blocks: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        grouped = x.reshape(*x.shape[:-1], self.blocks, x.shape[-1] // self.blocks)
        fan = grouped.shape[-1]
        held = parameters_of(self.dtype)
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
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, image: jax.Array) -> jax.Array:
        lead = image.shape[:-3]
        x = image.reshape(-1, *image.shape[-3:]).astype(self.dtype) / 255 - 0.5
        for stage, mult in enumerate(self.mults):
            x = nn.Conv(self.depth * mult, (self.kernel, self.kernel), padding="SAME", kernel_init=_init(),
                        dtype=self.dtype, param_dtype=parameters_of(self.dtype), name=f"cnn{stage}")(x)
            rows, height, width, channels = x.shape
            x = x.reshape(rows, height // 2, 2, width // 2, 2, channels).max((2, 4))
            norm = nn.RMSNorm(epsilon=1e-4, dtype=self.dtype, param_dtype=parameters_of(self.dtype),
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
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, deter: jax.Array, stoch: jax.Array) -> jax.Array:
        lead = deter.shape[:-1]
        depths = [self.depth * mult for mult in self.mults]
        height, width = (side // 2 ** len(depths) for side in self.image[:2])
        shape = (height, width, depths[-1])
        size = math.prod(shape)
        dense = functools.partial(nn.Dense, kernel_init=_init(), dtype=self.dtype,
                                  param_dtype=parameters_of(self.dtype))
        conv = functools.partial(nn.Conv, kernel_size=(self.kernel, self.kernel), padding="SAME",
                                 dtype=self.dtype, param_dtype=parameters_of(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-4, dtype=self.dtype,
                                 param_dtype=parameters_of(self.dtype))
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
    dtype: jnp.dtype = jnp.float32
    """The networks' computation dtype, bfloat16 for speed (DreamerV3's COMPUTE_DTYPE): parameters stay at
    least float32, and norms, distributions, heads' outputs and losses at least float32. A TSSM's
    dynamics compute in its own dtype."""

    def setup(self) -> None:
        dense = functools.partial(nn.Dense, kernel_init=_init(), dtype=self.dtype,
                                  param_dtype=parameters_of(self.dtype))
        norm = functools.partial(nn.RMSNorm, epsilon=1e-4, dtype=self.dtype,
                                 param_dtype=parameters_of(self.dtype))
        layers = functools.partial(_Layers, dtype=self.dtype)
        if self.image is None:
            self.encoder = layers(self.units, 3)
            self.decoder = layers(self.units, 3)
            self.decoded = dense(self.observation_size)
        else:
            self.encoder = ConvEncoder(self.conv_depth, dtype=self.dtype)
            self.decoder = ConvDecoder(self.image, self.conv_depth, self.units, blocks=self.blocks,
                                       dtype=self.dtype)
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

    def __call__(self, observation: jax.Array, action: jax.Array) -> jax.Array:
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

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.observation_size,) if self.image is None else self.image

    def initial(self, batch: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
        if self.dynamics is not None:
            held = self.dynamics
            return jnp.zeros((*batch, held.width)), jnp.zeros((*batch, held.stoch, held.classes))
        return jnp.zeros((*batch, self.deter)), jnp.zeros((*batch, self.stoch, self.classes))

    def seen(self, observation: Observation) -> jax.Array:
        """What the encoder reads of an observation: its `reads` entry, or the observation whole."""
        return observation if self.reads is None else observation[self.reads]

    def predicted(self, name: str, feature: jax.Array) -> jax.Array:
        """The auxiliary head `name`'s prediction of its privileged entry."""
        hidden, out = self.auxiliary_heads[name]
        return out(hidden(feature)).astype(jnp.float32)

    def embed(self, observation: jax.Array) -> jax.Array:
        """The encoder's tokens of an observation: a vector's symlogged first (rssm.py:218), an image's
        pixels through the convolutions."""
        return self.encoder(symlog(observation) if self.image is None else observation)

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

    def posterior(self, deter: jax.Array, tokens: jax.Array) -> jax.Array:
        """The posterior's log-probabilities `[..., stoch, classes]` (rssm.py:75-92); the TSSM's reads
        the tokens alone."""
        if self.dynamics is not None:
            return self.dynamics.posterior(tokens)
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
            return self.decoded(self.decoder(self.feature(deter, stoch)))
        return self.decoder(deter, stoch)

    def reconstruction(self, deter: jax.Array, stoch: jax.Array, observation: jax.Array) -> jax.Array:
        """The decoder's squared error, summed over the observation: against a vector's symlog, or an
        image's pixels over 255 (rssm.py:345-355, outs.MSE summed over the image's axes)."""
        decoded = self.decode(deter, stoch).astype(jnp.float32)
        if self.image is None:
            return jnp.sum(jnp.square(decoded - symlog(observation)), -1)
        return jnp.sum(jnp.square(decoded - observation.astype(decoded.dtype) / 255), (-3, -2, -1))

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
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, feature: jax.Array) -> jax.Array:
        logits = nn.Dense(self.bins, kernel_init=_init(0.0), dtype=self.dtype,
                          param_dtype=parameters_of(self.dtype))(_Layers(self.units, 3, self.dtype)(feature))
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
        cache = state.cache
        if model.dynamics is None:
            deter = model.core(state.deter, state.stoch, action)
        else:
            never = jnp.zeros(action.shape[:-1], bool)
            inputs = model.dynamics.inputs(state.stoch, action, never)
            deter, cache = model.dynamics.step(inputs, never, cache)
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
    the returns' percentile EMA. The slow critic is the EMA slot over the critic alone."""

    shown: Mapping[str, Shown] = {"loss": Shown(), "episode_return": Shown(better="higher")}
    scales: Mapping[str, float] = {"decoder": 1.0, "reward": 1.0, "continue": 1.0, "dynamics": 1.0,
                                   "representation": 0.1, "policy": 1.0, "value": 1.0, "replay_value": 0.3}

    def __init__(self, model: WorldModel, *, horizon: float = 333.0, imagination: int = 15,
                 lam: float = 0.95, entropy: float = 3e-4, free_nats: float = 1.0, slow_decay: float = 0.98,
                 return_rate: float = 0.01, variables: Variables | None = None,
                 scales: Mapping[str, float] | None = None, imitation: float = 0.0):
        """`scales` overrides the losses' weights by name, each auxiliary head's its entry's name
        (`AUXILIARY_SCALE` unless set; pretraining from demonstrations sets `policy`, `value` and
        `replay_value` to 0, and imagination is then skipped); `imitation` weighs
        behaviour cloning, the actor's negative log-likelihood of each replayed action at its posterior
        state, which reaches the actor alone."""
        if imagination < 1 or not 0 <= lam <= 1 or horizon <= 1:
            raise ValueError("imagination must be positive, lam in [0, 1] and the horizon past one step")
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
        self.variables = variables
        self.ema = EMASpec(optax.constant_schedule(slow_decay), under("params", "value"))
        self.bins = twohot_bins(model.bins)

    def held_variables(self) -> Variables | None:
        return self.variables

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        params_key, latent = jax.random.split(key)
        observation = jnp.zeros((1, *self.model.observation_shape),
                                jnp.float32 if self.model.image is None else jnp.uint8)
        inputs = observation, jnp.zeros((1, self.model.action_size))
        tree = self.model.init({"params": params_key, "latent": latent}, *inputs)
        # The returns' 5th and 95th percentiles, an EMA from zero (utils.py:16-72, debias off).
        return {**tree, "returns": {"low": jnp.zeros(()), "high": jnp.zeros(())}}

    @property
    def validation_loss_refusal(self) -> str | None:
        return "DreamerV3 scores its agent by its episodes' returns, not a validation loss"

    def observe(self, model: WorldModel, deter: jax.Array, stoch: jax.Array, observation: jax.Array,
                previous: jax.Array, first: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One posterior step: the latent and action masked where an episode starts, the GRU, then
        the posterior's draw; the new latent, beside the posterior's log-probabilities."""
        keep = (~first)[..., None]
        deter, previous = deter * keep, previous * keep
        stoch = stoch * keep[..., None]
        deter = model.core(deter, stoch, previous)
        posterior = model.posterior(deter, model.embed(observation))
        return deter, model.categorical(posterior), posterior

    def observed(self, variables: Variables, observation: jax.Array, previous: jax.Array, first: jax.Array,
                 key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The posterior latents over windows `[B, T, ...]` from a zero latent: each step's
        deterministic state, drawn categoricals and posterior log-probabilities, each step drawing from
        its own of `T` keys split from `key`, as the code's scan seeds each step's sample
        (rssm.py:63-90)."""
        rows, length = first.shape
        if self.model.dynamics is not None:
            # TransDreamer: the window's posteriors at once, then the transformer over the latents and
            # actions before each step.
            model = self.model.bind(variables, rngs={"latent": key})
            posterior = model.posterior(None, model.embed(observation))
            stoch = model.categorical(posterior)
            deter = model.dynamics.states(model.dynamics.inputs(_shifted(stoch), previous, first), first)
            return deter, stoch, posterior

        def step(latent, inputs):
            deter, stoch = latent
            observation, previous, first, key = inputs
            at = self.model.bind(variables, rngs={"latent": key})
            deter, stoch, posterior = self.observe(at, deter, stoch, observation, previous, first)
            return (deter, stoch), (deter, stoch, posterior)

        steps = (*(jnp.swapaxes(x, 0, 1) for x in (observation, previous, first)),
                 jax.random.split(key, length))
        _, found = jax.lax.scan(step, self.model.initial((rows,)), steps)
        deter, stoch, posterior = (jnp.swapaxes(x, 0, 1) for x in found)
        return deter, stoch, posterior

    def act(self, variables: Variables, observed: Mapping[str, jax.Array], key: jax.Array,
            memory: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, dict, tuple]:
        """The agent's step in an environment (`dew.rl.env.Policy` over `Driven`): the posterior
        given the step's observation, then an action drawn from the policy, clipped to [-1, 1]."""
        latent, draw = jax.random.split(key)
        model = self.model.bind(variables, rngs={"latent": latent})
        held, stoch, previous = memory
        seen = model.seen(observed["observation"])
        if model.dynamics is None:
            deter, stoch, _ = self.observe(model, held, stoch, seen, previous, observed["first"])
        else:
            first = observed["first"]
            deter, held = model.dynamics.step(model.dynamics.inputs(stoch, previous, first), first, held)
            stoch = model.categorical(model.posterior(None, model.embed(seen)))
        mean, std = model.policy(WorldModel.feature(deter, stoch))
        action = jnp.clip(mean + std * jax.random.normal(draw, mean.shape), -1, 1)
        return action, {}, (deter if model.dynamics is None else held, stoch, action)

    def memory(self, count: int) -> tuple[jax.Array | Cache, jax.Array, jax.Array]:
        """The agent's memory at the start for each of `count` environments: a zero latent, or a TSSM's
        empty cache of its `context` steps, and a zero previous latent and action."""
        deter, stoch = self.model.initial((count,))
        dynamics = self.model.dynamics
        held = deter if dynamics is None else dynamics.empty_cache(count, dynamics.context)
        return held, stoch, jnp.zeros((count, self.model.action_size))

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
        losses = {
            "dynamics": jnp.maximum(categorical_kl(jax.lax.stop_gradient(posterior), prior).sum(-1),
                                    self.free_nats),
            "representation": jnp.maximum(categorical_kl(posterior, jax.lax.stop_gradient(prior)).sum(-1),
                                          self.free_nats),
            "decoder": model.reconstruction(deter, stoch, observation),
            "reward": twohot_loss(model.reward(feature), reward, self.bins),
        }
        alive = (1 - terminal) * (1 - 1 / self.horizon)
        logit = model.continue_(feature)
        losses["continue"] = -(alive * jax.nn.log_sigmoid(logit) + (1 - alive) * jax.nn.log_sigmoid(-logit))
        for name, _, _ in self.model.auxiliary:
            target = jnp.asarray(batch["observation"][name], jnp.float32)
            losses[name] = jnp.sum(jnp.square(model.predicted(name, feature) - target), -1)
        if "imitation" in self.scales:
            # Each replayed action at the state it was taken at; a last step's action takes no effect.
            mean, std = model.policy(jax.lax.stop_gradient(feature))
            losses["imitation"] = (1 - last) * -normal_log_prob(action, mean, std)
        if not any(self.scales[name] for name in ("policy", "value", "replay_value")):
            total = sum(self.scales[name] * jnp.mean(value) for name, value in losses.items())
            metrics = {f"loss_{name}": jnp.mean(value) for name, value in losses.items()}
            return Ratio(total, jnp.ones(())), Aux(metrics, variables={"returns": variables["returns"]})

        # Imagination from every replayed posterior state, its inputs held still (agent.py:189-196).
        env = LatentEnv(self.model, jax.lax.stop_gradient(variables), self.bins)
        starts = jax.tree.map(lambda x: jax.lax.stop_gradient(x.reshape(rows * length, *x.shape[2:])),
                              (deter, stoch))
        latent_key, act_key, last_key = jax.random.split(imagine_key, 3)
        cache = None
        if model.dynamics is not None:
            # Each start's cache, the window's keys and values of its episode up to it.
            inputs = model.dynamics.inputs(_shifted(stoch), previous, first)
            cache = jax.lax.stop_gradient(model.dynamics.window_cache(inputs, first, self.imagination))
        state, begin = env.at(*starts, latent_key, cache)

        def policy(params, feature, key, memory):
            mean, std = self.model.apply(params, feature, method=WorldModel.policy)
            return mean + std * jax.random.normal(key, mean.shape), {}, None

        held = jax.lax.stop_gradient(variables)
        _, imagined = env_rollout(env, policy, held, Carry(state, begin, act_key), self.imagination,
                                  reset=False)
        # The start's feature then each imagined one, and the action taken at each [B*T, H + 1].
        features = jnp.concatenate([begin[:, None], jnp.swapaxes(imagined.following.observation, 0, 1)], 1)
        last_action, _, _ = policy(held, features[:, -1], last_key, None)
        actions = jnp.concatenate([jnp.swapaxes(imagined.action, 0, 1), last_action[:, None]], 1)
        rewards = twohot_mean(self.model.apply(held, features, method=WorldModel.reward), self.bins)
        cont = jax.nn.sigmoid(self.model.apply(held, features, method=WorldModel.continue_))
        values = twohot_mean(model.value(features), self.bins)
        slow = merge(variables, jax.lax.stop_gradient(step.ema)) if step.ema is not None else held
        slow_values = twohot_mean(self.model.apply(slow, features, method=lambda m, f: m.value(f)), self.bins)
        weight = jax.lax.stop_gradient(jnp.cumprod(cont, 1))
        returns = jax.lax.stop_gradient(lambda_return(jnp.zeros_like(cont), 1 - cont, rewards, values, values,
                                                      1.0, self.lam))
        rate, held_returns = self.return_rate, variables["returns"]
        low = (1 - rate) * held_returns["low"] + rate * jnp.percentile(returns, 5)
        high = (1 - rate) * held_returns["high"] + rate * jnp.percentile(returns, 95)
        scale = jnp.maximum(1.0, high - low)
        advantage = jax.lax.stop_gradient((returns - values[:, :-1]) / scale)
        mean, std = model.policy(features)
        log_pi = normal_log_prob(jax.lax.stop_gradient(actions), mean, std)[:, :-1]
        imagined_policy = weight[:, :-1] * -(log_pi * advantage + self.entropy * normal_entropy(std)[:, :-1])
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
        replay_slow = twohot_mean(self.model.apply(slow, feature, method=lambda m, f: m.value(f)), self.bins)
        replay_targets = jnp.concatenate([replay_returns, jnp.zeros_like(replay_returns[:, -1:])], 1)
        replay_value = (twohot_loss(replay_logits, replay_targets, self.bins)
                        + twohot_loss(replay_logits, replay_slow, self.bins))
        losses["replay_value"] = (1 - last)[:, :-1] * replay_value[:, :-1]

        total = sum(self.scales[name] * jnp.mean(value) for name, value in losses.items())
        metrics = {f"loss_{name}": jnp.mean(value) for name, value in losses.items()}
        metrics.update(return_scale=scale, imagined_return=jnp.mean(returns))
        stats = {"returns": {"low": jax.lax.stop_gradient(low), "high": jax.lax.stop_gradient(high)}}
        return Ratio(total, jnp.ones(())), Aux(metrics, variables=stats)


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
    since the last update."""

    objective: DreamerV3Objective
    env: Env | Simulator
    capacity: int = 100_000
    batch: int = 16
    length: int = 64
    train_ratio: float = 1024.0
    prefill: int = 64
    replay: ReplayBuffer | None = None
    _driven: Env | DrivenSimulator = field(init=False, repr=False)
    _carry: Carry | None = field(default=None, init=False, repr=False)
    _replay: ReplayBuffer | None = field(default=None, init=False, repr=False)
    _returns: jax.Array | None = field(default=None, init=False, repr=False)
    _taken: int = field(default=0, init=False, repr=False)
    _ended: tuple[float, int] = field(default=(0.0, 0), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.replay is None and self.prefill < self.length:
            raise ValueError("the prefill must hold at least one window")
        self._driven = DrivenSimulator(self.env) if isinstance(self.env, Simulator) else Driven(self.env)
        # The replay is written in place, its buffer donated to the step that writes it.
        self._acted = jax.jit(self._steps, static_argnames=("steps", "random"), donate_argnames=("replay",))
        self._recorded = jax.jit(self._kept, donate_argnames=("replay",))
        self._sample = jax.jit(lambda replay, key: replay.sample(key, self.batch, self.length))

    @property
    def metrics(self) -> Mapping[str, float]:
        total, count = self._ended
        return {"episode_return": total / count if count else float("nan")}

    @property
    def environment_steps(self) -> int:
        """The steps every environment has taken so far, summed."""
        return self._taken

    def _uniform(self, params: Variables, observed: Mapping[str, jax.Array], key: jax.Array,
                 memory: tuple) -> tuple[jax.Array, dict, tuple]:
        """The prefill's policy: actions uniform in [-1, 1]."""
        shape = (observed["reward"].shape[0], self.env.action_size)
        return jax.random.uniform(key, shape, minval=-1, maxval=1), {}, memory

    def _policy(self, random: bool) -> Callable:
        return self._uniform if random else self.objective.act

    def _kept(self, replay: ReplayBuffer, returns: jax.Array,
              recorded: Transition) -> tuple[ReplayBuffer, jax.Array, jax.Array, jax.Array]:
        """The replay with a rollout's steps written, each environment's running return, and the sum
        and count of the returns of the episodes that ended in it."""
        observed = recorded.observation
        replay = replay.add({**observed, "action": recorded.action})

        def tally(held, observed):
            total = jnp.where(observed["first"], 0, held) + observed["reward"]
            return total, (total, observed["last"])

        returns, (totals, ended) = jax.lax.scan(tally, returns, observed)
        return replay, returns, jnp.sum(jnp.where(ended, totals, 0)), jnp.sum(ended)

    def _steps(self, variables: Variables, carry: Carry, replay: ReplayBuffer, returns: jax.Array, *,
               steps: int, random: bool) -> tuple[Carry, ReplayBuffer, jax.Array, jax.Array, jax.Array]:
        policy = self._policy(random)
        carry, recorded = env_rollout(self._driven, policy, variables, carry, steps, reset=False)
        return carry, *self._kept(replay, returns, recorded)

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> Batch:
        act_key, sample_key = jax.random.split(key)
        variables = state.variables
        if self._carry is None:
            if isinstance(self.env, Simulator):
                self._carry = simulator_start(self._driven, act_key, self.objective.memory(self.env.count))
            else:
                count = self.env.reset(act_key)[1].reward.shape[0]
                self._carry = start(self._driven, act_key, self.objective.memory(count))
            count = len(self._carry.observation["reward"])
            self._returns = jnp.zeros(count)
            if self.replay is not None:
                self._replay = self.replay
            else:
                example = {**self._carry.observation, "action": jnp.zeros((count, self.env.action_size))}
                self._replay = ReplayBuffer.empty(example, self.capacity)
                self._advance(variables, self.prefill, random=True)
        owed = math.ceil((int(state.updates) + 1) * self.batch * self.length / self.train_ratio)
        envs = self._returns.shape[0]
        prefilled = 0 if self.replay is not None else self.prefill * envs
        due = max(owed - (self._taken - prefilled), 0)
        if due:
            self._advance(variables, -(-due // envs), random=False)
        return self.sample(sample_key)

    def sample(self, key: jax.Array) -> Batch:
        """`batch` windows of `length` steps drawn from the replay as an update draws them; a simulator's
        replay is on its device, where the key is put."""
        if isinstance(self.env, Simulator):
            key = jax.device_put(key, device_of(self._replay.data))
        return jax.device_get(self._sample(self._replay, key))

    def _advance(self, variables: Variables, steps: int, *, random: bool) -> None:
        if isinstance(self.env, Simulator):
            self._carry, recorded = simulator_rollout(self._driven, self._policy(random), variables,
                                                      self._carry, steps, reset=False)
            self._replay, self._returns, total, count = self._recorded(self._replay, self._returns, recorded)
        else:
            self._carry, self._replay, self._returns, total, count = self._acted(
                variables, self._carry, self._replay, self._returns, steps=steps, random=random)
        self._taken += steps * self._returns.shape[0]
        self._ended = (float(total), int(count))


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
    """The mean return of `env`'s environments, an `Env` or a `Simulator`, over `steps` steps of the agent
    from fresh episodes, each episode summed from its first step: one episode each where `steps` is its
    length."""
    if isinstance(env, Simulator):
        driven = DrivenSimulator(env)
        carry = simulator_start(driven, key, objective.memory(env.count))
        _, recorded = simulator_rollout(driven, objective.act, variables, carry, steps, reset=False)
        return float(jnp.mean(jnp.sum(recorded.observation["reward"], axis=0)))
    driven = Driven(env)
    count = env.reset(key)[1].reward.shape[0]

    @jax.jit
    def run(variables: Variables, key: jax.Array) -> jax.Array:
        carry = start(driven, key, objective.memory(count))
        _, recorded = env_rollout(driven, objective.act, variables, carry, steps, reset=False)
        return jnp.mean(jnp.sum(recorded.observation["reward"], axis=0))

    return float(run(variables, key))


__all__ = ["AUXILIARY_SCALE", "ConvDecoder", "ConvEncoder", "Critic", "DreamerRollout", "DreamerV3Objective",
           "LatentEnv", "WorldModel", "demonstration_replay", "episodes_return", "lambda_return",
           "normal_entropy", "normal_log_prob"]
