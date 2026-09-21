"""Masked (absorbing-state) discrete diffusion, shaped like the Gaussian one.

The forward process replaces each token by a mask id independently, with a
probability that grows along t in [0, 1]; `MaskingSchedule.alpha(t)` is the
fraction of tokens still visible. Training is the continuous-time negative
ELBO of MDLM (Sahoo et al. 2024, "Simple and Effective Masked Diffusion
Language Models"): the cross entropy of the model's prediction at the masked
positions, weighted by -alpha'(t) / (1 - alpha(t)). Sampling reverses the
process one interval at a time: a masked token is revealed with probability
(alpha(s) - alpha(t)) / (1 - alpha(t)) and, when revealed, drawn from the
model's categorical, which is MDLM's `_ddpm_update`.

`DiscreteProcess` has the surface `dew.sampling.sample` walks: a time grid,
an initial state, and a denoiser whose two outputs are the model's argmax
fill of the masked positions and the log-probabilities the solver draws
from, in the slots a Gaussian denoiser puts x_0 and epsilon.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.artifacts import agree_process_phase
from dew.diffusion.block import CanvasGeneration
from dew.nn.inputs import (
    ModelInputs,
    RowPlan,
    agreed_validity,
    continuation_keys,
    generation_signature,
    local_rows,
    mesh_of,
    request_key,
)
from dew.objectives.base import Variables
from dew.registry import presets, samplers

MDLM_STEPS = 64


class MaskingSchedule(ABC):
    """alpha(t) in (0, 1]: the fraction of tokens left unmasked at t, with
    alpha(0) = 1."""

    @abstractmethod
    def alpha(self, t) -> jax.Array: ...

    @abstractmethod
    def alpha_prime(self, t) -> jax.Array:
        """d alpha / dt, negative."""


@dataclass(frozen=True)
class LogLinear(MaskingSchedule):
    """MDLM's log-linear schedule: alpha(t) = 1 - (1 - eps) t, so the masking
    rate -log alpha is linear in log space and the NELBO weight is 1 / t."""

    eps: float = 1e-3

    def alpha(self, t):
        return 1 - (1 - self.eps) * jnp.asarray(t, jnp.float32)

    def alpha_prime(self, t):
        return jnp.full_like(jnp.asarray(t, jnp.float32), -(1 - self.eps))


@dataclass(frozen=True)
class DiscreteProcess:
    """The masking process over a vocabulary whose mask token is `mask_id`."""

    schedule: MaskingSchedule
    mask_id: int
    T = 1.0

    def sample_t(self, key, n: int) -> jax.Array:
        """`n` times stratified over [0, 1), MDLM's antithetic draw. One
        uniform offset is shared by the batch, so the weights 1 / t of one
        batch cover the trajectory."""
        offset = jax.random.uniform(key, (), minval=0.0, maxval=1.0)
        return (jnp.arange(n, dtype=jnp.float32) + offset) / n

    def corrupt(self, key, tokens, t) -> tuple[jax.Array, jax.Array]:
        """`(masked tokens, is_masked)` at `t`, one t per row."""
        move_chance = 1 - self.schedule.alpha(t)
        is_masked = jax.random.uniform(key, tokens.shape) < move_chance[:, None]
        return jnp.where(is_masked, self.mask_id, tokens), is_masked

    def weight(self, t) -> jax.Array:
        """The NELBO weight -alpha'(t) / (1 - alpha(t)) on the masked cross
        entropy, and exactly zero at t = 0. Nothing is masked there, so no
        token contributes, and the quotient itself is undefined."""
        t = jnp.asarray(t, jnp.float32)
        return jnp.where(t > 0, -self.schedule.alpha_prime(t) / (1 - self.schedule.alpha(t)), 0.0)

    def times(self, steps: int) -> jax.Array:
        return jnp.linspace(self.T, 0.0, steps, dtype=jnp.float32)

    def noise(self, key, shape) -> jax.Array:
        """x_T: every position masked. `key` is unused, the fully masked state
        is one point."""
        return jnp.full(shape, self.mask_id, jnp.int32)

    def denoiser(self, model: nn.Module, params: Variables,
                 conditions: dict[str, Any] | None = None,
                 unconditional: dict[str, Any] | None = None, *,
                 inputs: ModelInputs | None = None, mutable_mask: jax.Array | None = None) -> DiscreteDenoiser:
        if conditions or unconditional is not None:
            raise ValueError("the masked diffusion LM takes no conditions")
        return DiscreteDenoiser(self, model, params, inputs, mutable_mask)

    def generate(self, model: nn.Module, variables: Variables, inputs: ModelInputs | jax.typing.ArrayLike,
                 max_new_tokens: int, *, key: jax.Array | None = None, seed: int | None = None,
                 n: int = 1, steps: int = MDLM_STEPS, sampler: Unmask | None = None,
                 eos_token_ids: tuple[int, ...] = (), pad_token_id: int = 0) -> CanvasGeneration:
        """Native MDLM over one full response span, not source-specific remasking.

        Prompt tokens are immutable, including literal mask IDs. EOS trims the
        completed response; it does not stop bidirectional refinement early.
        """
        prepared = request = None
        error = None
        solver = Unmask() if sampler is None else sampler
        try:
            request = request_key(key, seed)
            canonical = ModelInputs.from_value(inputs)
            prepared = jax.tree.map(lambda leaf: local_rows(leaf, host=False), canonical)
            _validate_request(model, self, prepared, max_new_tokens, steps, n, eos_token_ids, pad_token_id)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="masked generation setup")
        assert prepared is not None and request is not None
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            controls = (max_new_tokens, steps, n, self, solver, eos_token_ids, pad_token_id, model)
            prepared = agreed_validity(prepared, jax.process_count(), controls=controls, phase="masked input")
            multihost_utils.assert_equal(generation_signature(prepared, controls),
                                        "masked input schemas and generation policy must agree")
        plan = RowPlan.over(mesh_of(variables), prepared.tokens.shape[0])
        padded = plan.pad(prepared)
        if plan.count != plan.rows:
            valid = padded.token_fields.get("attention_mask", jnp.ones(padded.tokens.shape, bool))
            padded = replace(padded, token_fields={**padded.token_fields,
                "attention_mask": jnp.asarray(valid, bool) & ~plan.padding[:, None]})
        result = _compiled(plan.sharding)(model, variables, plan.place(padded), plan.keys(request),
            self, solver, max_new_tokens, steps, n, eos_token_ids, pad_token_id)
        return replace(result, rows=plan.rows * n, prompt_width=prepared.tokens.shape[1])


@dataclass(frozen=True)
class DiscreteDenoiser:
    """`(x_t, t) -> (argmax fill, log-probabilities)` for `model` under `params`.

    The model's own logits at an unmasked position are irrelevant. The
    position keeps its token (MDLM's carry-over parameterization).
    The mask token itself carries no mass. It marks corruption, so the
    categorical a reveal draws from never offers it, however the model
    scores it.
    """

    process: DiscreteProcess
    model: nn.Module
    params: Variables
    inputs: ModelInputs | None = None
    mutable_mask: jax.Array | None = None

    def masked(self, tokens):
        mask = tokens == self.process.mask_id
        return mask if self.mutable_mask is None else mask & self.mutable_mask

    def __call__(self, x_t, t):
        fields = {} if self.inputs is None else self.inputs.kwargs()
        logits = self.model.apply(self.params, x_t, **fields, rngs=None, mutable=False,
                                  capture_intermediates=False, method=None)
        assert not isinstance(logits, tuple)  # no mutable collections were asked for
        logits = logits.at[..., self.process.mask_id].set(-jnp.inf)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        masked = self.masked(x_t)
        filled = jnp.where(masked, jnp.argmax(log_probs, axis=-1), x_t)
        return filled, log_probs


@samplers("unmask")
@dataclass(frozen=True)
class Unmask:
    """MDLM's reverse step from t to s < t: each masked position is revealed
    with probability (alpha(s) - alpha(t)) / (1 - alpha(t)), with a token drawn
    from the model's categorical; the rest stay masked. Integrates a
    `DiscreteProcess`."""

    def init(self, x, times, process, *, key) -> tuple:
        return ()

    def step(self, x, t, t_next, denoised, log_probs, state, key, process, denoise):
        if not isinstance(process, DiscreteProcess):
            raise ValueError(
                f"Unmask reveals masked tokens, so it needs a DiscreteProcess and not "
                f"{type(process).__name__}")
        alpha_t = process.schedule.alpha(t)
        alpha_s = process.schedule.alpha(t_next)
        reveal_chance = (alpha_s - alpha_t) / (1 - alpha_t)
        reveal_key, token_key = jax.random.split(key)
        masked = denoise.masked(x)
        reveal = jax.random.uniform(reveal_key, x.shape) < reveal_chance[:, None]
        drawn = jax.random.categorical(token_key, log_probs, axis=-1)
        return jnp.where(masked & reveal, drawn, x), state


@presets("mdlm")
@dataclass(frozen=True)
class MDLM:
    """Sahoo et al. 2024 with the log-linear schedule."""

    mask_id: int
    eps: float = 1e-3

    def __call__(self) -> DiscreteProcess:
        return DiscreteProcess(schedule=LogLinear(eps=self.eps), mask_id=self.mask_id)


def _validate_request(model: nn.Module, process: DiscreteProcess, inputs: ModelInputs,
                      budget: int, steps: int, n: int, eos_ids: tuple[int, ...], pad_id: int) -> None:
    if getattr(model, "causal", True):
        raise ValueError("masked generation requires a bidirectional model")
    for name, value, minimum in (("max_new_tokens", budget, 0), ("steps", steps, 1), ("n", n, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    vocab = model.vocab_size
    if type(process.mask_id) is not int or not 0 <= process.mask_id < vocab:
        raise ValueError("mask_id is outside the vocabulary")
    if type(pad_id) is not int or not 0 <= pad_id < vocab:
        raise ValueError("pad_token_id is outside the vocabulary")
    if any(type(token) is not int or not 0 <= token < vocab for token in eos_ids):
        raise ValueError("eos_token_ids contain an id outside the vocabulary")
    if min(inputs.tokens.shape) < 1:
        raise ValueError("masked generation needs nonempty token rows")
    if inputs.tokens.shape[1] + budget > model.max_seq_len:
        raise ValueError("prompt plus max_new_tokens exceeds max_seq_len")
    if inputs.conditioning:
        raise ValueError("native MDLM is text-only; conditioning payloads are not supported")
    unknown = inputs.token_fields.keys() - {"attention_mask", "positions", "rotary_positions", "segment_ids"}
    if unknown:
        raise ValueError(f"masked generation cannot extend token fields {sorted(unknown)}")
    positions = inputs.token_fields.get("positions")
    if positions is not None and positions.ndim != 2:
        raise ValueError("positions must be [B, S] logical token positions")


def _response_inputs(inputs: ModelInputs, width: int, mask_id: int) -> tuple[ModelInputs, jax.Array]:
    batch, prompt = inputs.tokens.shape
    valid = jnp.asarray(inputs.token_fields.get("attention_mask", jnp.ones((batch, prompt), bool)), bool)
    active = valid.any(axis=1)
    positions = inputs.token_fields.get("positions", jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0))
    last_slot = jnp.max(jnp.where(valid, jnp.arange(prompt)[None, :], 0), axis=1)
    fields = {}
    for name, value in {**inputs.token_fields, "positions": positions, "attention_mask": valid}.items():
        if name == "positions":
            tail = value[jnp.arange(batch), last_slot, None] + jnp.arange(1, width + 1)[None]
        elif name == "rotary_positions":
            # Multi-axis RoPE continues one beyond the largest real coordinate
            # on every axis, unlike resettable logical document positions.
            keep = valid.reshape(valid.shape + (1,) * (value.ndim - 2))
            last = jnp.max(jnp.where(keep, value, -1), axis=tuple(range(1, value.ndim)))
            coordinates = last[:, None] + jnp.arange(1, width + 1)[None]
            tail = jnp.broadcast_to(coordinates.reshape((batch, width) + (1,) * (value.ndim - 2)),
                                    (batch, width, *value.shape[2:]))
        elif name == "attention_mask":
            tail = jnp.broadcast_to(active[:, None], (batch, width))
        elif name == "segment_ids":
            tail = jnp.broadcast_to(value[jnp.arange(batch), last_slot, None], (batch, width))
        else:
            tail = jnp.full((batch, width, *value.shape[2:]), -1, value.dtype)
        fields[name] = jnp.concatenate([value, tail.astype(value.dtype)], axis=1)
    tokens = jnp.concatenate([inputs.tokens, jnp.full((batch, width), mask_id, jnp.int32)], axis=1)
    mutable = jnp.concatenate([jnp.zeros((batch, prompt), bool),
                               jnp.broadcast_to(active[:, None], (batch, width))], axis=1)
    return ModelInputs(tokens, fields, inputs.conditioning), mutable


def _generate(model: nn.Module, variables: Variables, inputs: ModelInputs, keys: jax.Array,
              process: DiscreteProcess, sampler: Unmask, budget: int, steps: int, n: int,
              eos_ids: tuple[int, ...], pad_id: int) -> CanvasGeneration:
    from dew.sampling.sample import sample

    batch, prompt = inputs.tokens.shape
    if budget == 0:
        zeros = jnp.zeros((batch * n,), jnp.int32)
        return CanvasGeneration(jnp.repeat(inputs.tokens, n, axis=0), zeros, zeros.astype(bool), zeros)
    full, mutable = _response_inputs(inputs, budget, process.mask_id)

    def row(prepared, changeable, key):
        prepared = jax.tree.map(lambda leaf: leaf[None], prepared)
        denoise = process.denoiser(model, variables, inputs=prepared, mutable_mask=changeable[None])
        def continued(draw_key):
            tokens = sample(denoise, prepared.tokens, steps, solver=sampler, key=draw_key)[0]
            response = tokens[prompt:]
            is_eos = jnp.isin(response, jnp.asarray(eos_ids, jnp.int32))
            first = jnp.min(jnp.where(is_eos, jnp.arange(budget), budget))
            active = jnp.any(changeable)
            length = jnp.where(active, jnp.minimum(first + 1, budget), 0)
            response = jnp.where(jnp.arange(budget) < length, response, pad_id)
            return CanvasGeneration(jnp.concatenate([tokens[:prompt], response]), length,
                                    active & is_eos.any(), jnp.where(active, steps, 0))
        return jax.lax.map(continued, continuation_keys(key, n))

    result = jax.vmap(row)(full, mutable, keys)
    return jax.tree.map(lambda leaf: leaf.reshape((batch * n, *leaf.shape[2:])), result)


@functools.cache
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(_generate, static_argnames=("model", "process", "sampler", "budget", "steps", "n", "eos_ids", "pad_id"),
                   in_shardings=(None, rows, rows), out_shardings=rows)

