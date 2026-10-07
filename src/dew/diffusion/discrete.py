"""Masked (absorbing-state) discrete diffusion, with the interface of the Gaussian one.

The forward process replaces each token with a mask id independently, with
a probability that grows with t in [0, 1]. `MaskingSchedule.alpha(t)` is
the fraction of tokens still visible at t. Training minimizes the
continuous-time negative ELBO of MDLM (Sahoo et al. 2024, "Simple and
Effective Masked Diffusion Language Models"), which is the cross entropy of
the model's prediction at the masked positions, weighted by
-alpha'(t) / (1 - alpha(t)).

Sampling reverses the process one interval at a time. A masked token is
revealed with probability (alpha(s) - alpha(t)) / (1 - alpha(t)), and a
revealed token is drawn from the model's categorical distribution. This is
MDLM's `_ddpm_update`.

`DiscreteProcess` provides what `dew.sampling.sample` needs from a process:
a time grid, an initial state and a denoiser. The denoiser's two outputs
sit where a Gaussian denoiser returns x_0 and epsilon. They are the model's
argmax fill of the masked positions and the log-probabilities the solver
draws from.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.diffusion.block import CanvasGeneration, through_eos
from dew.diffusion.process import Conditioning
from dew.nn.inputs import RESPONSE_FIELDS, ModelInputs, Request, continuation_keys, local_rows, mesh_of
from dew.nn.protocols import Logits, TokenModel
from dew.objectives.base import Variables
from dew.registry import presets, solvers

MDLM_STEPS = 64
"""Reverse steps `generate` takes by default, the count MDLM samples with."""

SAMPLING_EPS = 1e-3
"""The least training time, MDLM's `training.sampling_eps` (configs/config.yaml)."""


class MaskingSchedule(ABC):
    """Sets how fast tokens are masked as t grows.

    `alpha(t)` in (0, 1] is the fraction of tokens left unmasked at t, with
    alpha(0) = 1.
    """

    @abstractmethod
    def alpha(self, t) -> jax.Array: ...

    @abstractmethod
    def alpha_prime(self, t) -> jax.Array:
        """Return d alpha / dt, which is negative."""


@dataclass(frozen=True)
class LogLinear(MaskingSchedule):
    """MDLM's log-linear schedule, alpha(t) = 1 - (1 - eps) t.

    The masking rate -log alpha is then linear in log space and the NELBO
    weight is 1 / t.
    """

    eps: float = 1e-3

    def alpha(self, t):
        return 1 - (1 - self.eps) * jnp.asarray(t, jnp.float32)

    def alpha_prime(self, t):
        return jnp.full_like(jnp.asarray(t, jnp.float32), -(1 - self.eps))


@dataclass(frozen=True)
class DiscreteProcess:
    """Masks tokens of a vocabulary whose mask token is `mask_id`."""

    schedule: MaskingSchedule
    mask_id: int
    T = 1.0
    """The fully masked end of the time domain, named `T` as in a Gaussian `Process`."""

    def to_json(self) -> dict:
        if not isinstance(self.schedule, LogLinear):
            raise TypeError("custom masking schedules need an explicit record declaration")
        return {'mask_id': self.mask_id, 'eps': self.schedule.eps}

    @classmethod
    def from_json(cls, record: Mapping) -> DiscreteProcess:
        return cls(LogLinear(eps=record['eps']), record['mask_id'])

    def sample_t(self, key, n: int) -> jax.Array:
        """Return `n` training times stratified over [SAMPLING_EPS, 1), MDLM's antithetic draw.

        Row i draws its own point in the i-th of n strata, so the weights
        1 / t of one batch cover the whole trajectory. The floor at
        SAMPLING_EPS keeps every weight at most 1 / SAMPLING_EPS.
        """
        # MDLM's `_sample_t`, kuleshov-group/mdlm@c112c52, diffusion.py:800-808.
        stratified = (jax.random.uniform(key, (n,)) + jnp.arange(n, dtype=jnp.float32)) / n
        return (1 - SAMPLING_EPS) * stratified + SAMPLING_EPS

    def corrupt(self, key, tokens, t) -> tuple[jax.Array, jax.Array]:
        """Return `(masked tokens, is_masked)` at `t`, with one t per row."""
        move_chance = 1 - self.schedule.alpha(t)
        is_masked = jax.random.uniform(key, tokens.shape) < move_chance[:, None]
        return jnp.where(is_masked, self.mask_id, tokens), is_masked

    def weight(self, t) -> jax.Array:
        """Return the NELBO weight -alpha'(t) / (1 - alpha(t)) on the masked cross entropy.

        The weight is exactly zero at t = 0. Nothing is masked there, so no
        token contributes, and the quotient itself is undefined.
        """
        t = jnp.asarray(t, jnp.float32)
        return jnp.where(t > 0, -self.schedule.alpha_prime(t) / (1 - self.schedule.alpha(t)), 0.0)

    def times(self, steps: int) -> jax.Array:
        return jnp.linspace(self.T, 0.0, steps, dtype=jnp.float32)

    def noise(self, key, shape) -> jax.Array:
        """Return x_T, with every position masked.

        `key` is not used, because the fully masked state is a single point
        and needs no random draw.
        """
        return jnp.full(shape, self.mask_id, jnp.int32)

    def denoiser(
        self,
        model: nn.Module,
        params: Variables,
        conditions: Mapping[str, Conditioning] | None = None,
        unconditional: Mapping[str, Conditioning] | None = None,
        *,
        inputs: ModelInputs | None = None,
        mutable_mask: jax.Array | None = None,
    ) -> DiscreteDenoiser:
        if conditions or unconditional is not None:
            raise ValueError("the masked diffusion LM takes no conditions")
        return DiscreteDenoiser(self, model, params, inputs, mutable_mask)

    def generate(self, model: nn.Module, variables: Variables, inputs: ModelInputs | jax.typing.ArrayLike,
                 max_new_tokens: int, *, key: int | jax.Array | None = None,
                 n: int = 1, steps: int = MDLM_STEPS, solver: Unmask | None = None,
                 eos_token_ids: tuple[int, ...] = (), pad_token_id: int = 0) -> CanvasGeneration:
        """Generate `max_new_tokens` tokens after each prompt with native MDLM.

        The whole response span starts masked, and `solver` (`Unmask` by
        default) unmasks it over a grid of `steps` times. Each prompt row
        gets `n` continuations in the returned `CanvasGeneration`. Prompt
        tokens never change, including literal mask ids. An EOS token trims
        the finished response, and the tokens after it become
        `pad_token_id`, but it does not end the bidirectional refinement
        early.

        The response continues the prompt's token fields
        (`ModelInputs.extended`): its slots are text, at the positions after
        the prompt's, and the conditioning is the prompt's own.

        Raises `ValueError` when the model does not read the whole row
        (`refuse_causal`), when the prompt plus `max_new_tokens` exceeds the
        `max_seq_len` it declares, or when the inputs include a token field
        a response has no rule to continue.
        """
        solver = Unmask() if solver is None else solver

        def check(canonical: ModelInputs, pooled: bool) -> tuple[ModelInputs, tuple, None]:
            prepared = jax.tree.map(lambda leaf: local_rows(leaf, host=False), canonical)
            _validate_request(model, self, prepared, max_new_tokens, steps, n, eos_token_ids, pad_token_id)
            controls = (max_new_tokens, steps, n, self, solver, eos_token_ids, pad_token_id, model)
            return prepared, controls, None

        request, _ = Request.prepare(inputs, key, mesh_of(variables), check, phase="masked generation")
        plan = request.plan
        generated = _compiled(plan.sharding)(model, variables, plan.place(request.padded()),
                                             plan.keys(request.key), self, solver, max_new_tokens, steps, n,
                                             eos_token_ids, pad_token_id)
        return replace(generated, rows=plan.rows * n, prompt_width=request.inputs.tokens.shape[1])


@dataclass(frozen=True)
class DiscreteDenoiser:
    """Maps `(x_t, t)` to `(argmax fill, log-probabilities)` for `model` under `params`.

    The denoiser ignores `t`, because the masked model is conditioned on the
    corruption it sees and not on the time. `Unmask.step` reads the time
    from the process. `t` stays in the signature because `sample` calls
    every denoiser as `(x_t, t)`. A model with full-sequence logits
    (`Logits`) is read through them, and any other through its call.

    The model's own logits at an unmasked position do not matter, since the
    position keeps its token (MDLM's carry-over parameterization). The mask
    token gets zero probability, whatever score the model gives it, because
    it marks corruption and a revealed position must never draw it. When
    `mutable_mask` is given, only masked positions inside it can change.
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
                                  capture_intermediates=False,
                                  method="logits" if isinstance(self.model, Logits) else None)
        assert not isinstance(logits, tuple)  # no mutable collections were asked for
        # Normalized in fp32, as `token_log_probs` does: the reveal draws its
        # categorical from these, and a bf16 log partition would quantize it.
        logits = logits.astype(jnp.float32).at[..., self.process.mask_id].set(-jnp.inf)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        masked = self.masked(x_t)
        filled = jnp.where(masked, jnp.argmax(log_probs, axis=-1), x_t)
        return filled, log_probs


@solvers("unmask")
@dataclass(frozen=True)
class Unmask:
    """Integrates a `DiscreteProcess` with MDLM's reverse step from t to s < t.

    Each masked position is revealed with probability
    (alpha(s) - alpha(t)) / (1 - alpha(t)), and a revealed position takes a
    token drawn from the model's categorical distribution. The rest stay
    masked. Stepping any other kind of process raises `ValueError`.
    """

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
    """Builds MDLM's masked diffusion process on the log-linear schedule.

    MDLM is Sahoo et al. 2024.
    """

    mask_id: int
    eps: float = 1e-3

    def __call__(self) -> DiscreteProcess:
        return DiscreteProcess(schedule=LogLinear(eps=self.eps), mask_id=self.mask_id)


def refuse_causal(model: nn.Module) -> None:
    """Raise ValueError unless `model` can fill a masked row: MDLM draws each
    masked slot from the row's full-sequence logits (`Logits`), so every
    position must read the ones after it (`TokenModel` with `causal` False)."""
    name = type(model).__name__
    if not isinstance(model, Logits):
        raise ValueError(f"masked generation requires a bidirectional model, and this {name} gives no "
                         f"full-sequence logits (Logits)")
    if not isinstance(model, TokenModel):
        raise ValueError(f"masked generation requires a bidirectional model, and this {name} does not say "
                         f"whether a position reads the ones after it (TokenModel)")
    if model.causal:
        raise ValueError(f"masked generation requires a bidirectional model, and this {name} is causal: "
                         f"no position reads the masked ones after it")


def _validate_request(model: nn.Module, process: DiscreteProcess, inputs: ModelInputs,
                      budget: int, steps: int, n: int, eos_ids: tuple[int, ...], pad_id: int) -> None:
    from dew.sampling.text import Bounded

    refuse_causal(model)
    for name, value, minimum in (("max_new_tokens", budget, 0), ("steps", steps, 1), ("n", n, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    # A model that declares no vocabulary or capacity is checked for neither, as text generation does.
    bounded = isinstance(model, Bounded)
    vocab = model.vocab_size if bounded else None
    if vocab is not None:
        if type(process.mask_id) is not int or not 0 <= process.mask_id < vocab:
            raise ValueError("mask_id is outside the vocabulary")
        if type(pad_id) is not int or not 0 <= pad_id < vocab:
            raise ValueError("pad_token_id is outside the vocabulary")
        if any(type(token) is not int or not 0 <= token < vocab for token in eos_ids):
            raise ValueError("eos_token_ids contain an id outside the vocabulary")
    if min(inputs.tokens.shape) < 1:
        raise ValueError("masked generation needs nonempty token rows")
    capacity = model.max_seq_len if bounded else None
    if capacity is not None and inputs.tokens.shape[1] + budget > capacity:
        raise ValueError("prompt plus max_new_tokens exceeds max_seq_len")
    unruled = sorted(inputs.token_fields.keys() - RESPONSE_FIELDS.keys())
    if unruled:
        raise ValueError(f"masked generation cannot extend token fields {unruled}: a response has no "
                         f"rule for their slots (dew.nn.inputs.RESPONSE_FIELDS)")
    positions = inputs.token_fields.get("positions")
    if positions is not None and positions.ndim != 2:
        raise ValueError("positions must be [B, S] logical token positions")


def _response_inputs(inputs: ModelInputs, width: int, mask_id: int) -> tuple[ModelInputs, jax.Array]:
    """`inputs` extended by `width` masked slots, and which slots may change.

    The prompt's validity and logical positions are written out first, so
    the response's continue from them (`ModelInputs.extended`): its slots
    are real in the rows that hold a prompt, the only rows whose slots may
    change, and count on from the prompt's last real token.
    """
    batch, prompt = inputs.tokens.shape
    valid = jnp.asarray(inputs.token_fields.get("attention_mask", jnp.ones((batch, prompt), bool)), bool)
    positions = inputs.token_fields.get("positions", jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0))
    written = replace(inputs, token_fields={**inputs.token_fields, "positions": positions,
                                            "attention_mask": valid})
    extended = written.extended(jnp.full((batch, width), mask_id, jnp.int32))
    return extended, extended.token_fields["attention_mask"] & (jnp.arange(prompt + width) >= prompt)


def _generate(model: nn.Module, variables: Variables, inputs: ModelInputs, keys: jax.Array,
              process: DiscreteProcess, solver: Unmask, budget: int, steps: int, n: int,
              eos_ids: tuple[int, ...], pad_id: int) -> CanvasGeneration:
    """The compiled body: one masked walk per row, `n` continuations each.

    Each row is sampled on its own so the response span is the only mutable
    part of it. A continuation keeps the tokens up to its first EOS, pads the
    rest, and reports the steps it took.
    """
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
            tokens = sample(denoise, prepared.tokens, steps, solver=solver, key=draw_key)[0]
            active = jnp.any(changeable)
            response, length, ended = through_eos(tokens[prompt:], jnp.where(active, budget, 0),
                                                  eos_ids, pad_id)
            return CanvasGeneration(jnp.concatenate([tokens[:prompt], response]), length,
                                    ended, jnp.where(active, steps, 0))
        return jax.lax.map(continued, continuation_keys(key, n))

    generated = jax.vmap(row)(full, mutable, keys)
    return jax.tree.map(lambda leaf: leaf.reshape((batch * n, *leaf.shape[2:])), generated)


@functools.cache
def _compiled(rows: jax.sharding.NamedSharding | None):
    return jax.jit(
        _generate,
        static_argnames=("model", "process", "solver", "budget", "steps", "n", "eos_ids", "pad_id"),
        in_shardings=(None, rows, rows),
        out_shardings=rows,
    )


__all__ = ["MDLM", "DiscreteDenoiser", "DiscreteProcess", "LogLinear", "MaskingSchedule", "Unmask"]
