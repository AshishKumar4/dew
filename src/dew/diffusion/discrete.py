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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew._distributions import counted, draw, row_draw, wellformed
from dew.diffusion.block import CanvasGeneration, through_eos
from dew.diffusion.process import Conditioning
from dew.diffusion.schedules import common
from dew.interop.weights import decoding_storage
from dew.nn import categorical
from dew.nn.inputs import (
    RESPONSE_FIELDS,
    Coordinates,
    ModelInputs,
    Request,
    continuation_keys,
    coordinates,
    local_rows,
    mesh_of,
    prompt_major,
    real_slots,
    token_id,
    token_ids,
    token_layout,
)
from dew.nn.kernels.options import program
from dew.nn.protocols import Logits, ShiftsLogits, TokenModel
from dew.nn.sharding import explicit_spec
from dew.objectives.base import Variables

MDLM_STEPS = 128
"""Reveal steps `generate` takes by default, MDLM's `sampling.steps` (configs/config.yaml)."""

SAMPLING_EPS = 1e-3
"""The least training time, MDLM's `training.sampling_eps` (configs/config.yaml)."""

NOISE_REMOVAL_TIME = 1e-5
"""The last time MDLM's sampler reveals at, where it removes the noise left
(`_sample(eps=1e-5)`, kuleshov-group/mdlm@c112c52, diffusion.py:658)."""


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
        return 1 - (1 - self.eps) * common.times(t)

    def alpha_prime(self, t):
        return jnp.full_like(common.times(t), -(1 - self.eps))


@dataclass(frozen=True)
class DiscreteProcess:
    """Masks tokens of a vocabulary whose mask token is `mask_id`."""

    schedule: MaskingSchedule
    mask_id: int
    T = 1.0
    """The fully masked end of the time domain, named `T` as in a Gaussian `Process`."""

    def sample_t(self, key, n: int) -> jax.Array:
        """Return `n` training times stratified over [SAMPLING_EPS, 1), MDLM's antithetic draw.

        Row i draws its own point in the i-th of n strata, so the weights
        1 / t of one batch cover the whole trajectory. The floor at
        SAMPLING_EPS keeps every weight at most 1 / SAMPLING_EPS.
        """
        # MDLM's `_sample_t`, kuleshov-group/mdlm@c112c52, diffusion.py:800-808.
        return self.stratified(jax.random.uniform(key, (n,), jnp.float32), jnp.arange(n), n)

    @staticmethod
    def stratified(uniforms, strata, n: int) -> jax.Array:
        """`sample_t` from a uniform for each row and its stratum of `n`."""
        stratified = (uniforms + strata.astype(jnp.float32)) / n
        return (1 - SAMPLING_EPS) * stratified + SAMPLING_EPS

    def corrupt(self, key, tokens, t) -> tuple[jax.Array, jax.Array]:
        """Return `(masked tokens, is_masked)` at `t`, with one t per row."""
        draws = jax.random.uniform(key, tokens.shape, jnp.float32, out_sharding=explicit_spec(tokens))
        return self.masked(draws, tokens, t)

    def masked(self, draws, tokens, t) -> tuple[jax.Array, jax.Array]:
        """`corrupt` from a uniform draw for each of the tokens: those under `1 - alpha(t)` masked."""
        is_masked = draws < (1 - self.schedule.alpha(t))[:, None]
        return jnp.where(is_masked, self.mask_id, tokens), is_masked

    def weight(self, t) -> jax.Array:
        """Return the NELBO weight -alpha'(t) / (1 - alpha(t)) on the masked cross entropy.

        The weight is exactly zero at t = 0. Nothing is masked there, so no
        token contributes, and the quotient itself is undefined.
        """
        t = common.times(t)
        return jnp.where(t > 0, -self.schedule.alpha_prime(t) / (1 - self.schedule.alpha(t)), 0.0)

    def times(self, steps: int) -> jax.Array:
        """MDLM's sampling grid for `steps` reveal steps: `steps + 1` times from
        T to `NOISE_REMOVAL_TIME`, at which `sample`'s closing argmax fills what
        is still masked, as MDLM's noise removal does."""
        return jnp.linspace(self.T, NOISE_REMOVAL_TIME, steps + 1, dtype=jnp.float32)

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
                 max_new_tokens: int, *, key: int | jax.Array,
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
        generation = program(
            _generate, model,
            static_argnames=("model", "process", "solver", "budget", "steps", "n", "eos_ids", "pad_id"),
            in_shardings=(None, plan.sharding, plan.sharding), out_shardings=(plan.sharding, None))
        generated, undefined = generation(model, decoding_storage(model, variables),
                                          plan.place(request.padded()), plan.keys(request.key), self,
                                          solver, max_new_tokens, steps, n, eos_token_ids, pad_token_id)
        if bool(jax.device_get(undefined)):
            raise ValueError(REVEALED_UNDEFINED)
        return replace(generated, rows=plan.rows * n, prompt_width=request.inputs.tokens.shape[1])


REVEALED_UNDEFINED = "the model left a position it reveals without a distribution to draw its token from"
"""What a masked walk raises when a revealed position's scores hold no distribution."""


def aligned(model: nn.Module, scores: jax.Array,
            fields: Mapping[str, jax.Array | np.ndarray | Coordinates] | None = None) -> jax.Array:
    """`scores` `[B, S, ...]` per position as the model's predictions of each position's token: its
    own for a model whose logits line up with its inputs (LLaDA's, MDLM's), and for one that declares
    `shifted_logits` (Dream-v0's) the position before's, as its reference shifts them. A position
    that opens its text, the row's first, the first real one after padding and each packed
    document's first in the token `fields`' layout (`token_layout`), reads its own, as the
    reference repeats a row's first: no document reads another's."""
    if not (isinstance(model, ShiftsLogits) and model.shifted_logits):
        return scores
    before = jnp.concatenate([scores[:, :1], scores[:, :-1]], axis=1)
    valid, documents = token_layout(fields or {})
    opens = jnp.zeros(scores.shape[:2], bool).at[:, 0].set(True)
    if valid is not None:
        opens |= valid & ~jnp.concatenate([jnp.zeros_like(valid[:, :1]), valid[:, :-1]], axis=1)
    if documents is not None:
        changes = documents[:, 1:] != documents[:, :-1]
        opens |= jnp.concatenate([jnp.ones_like(documents[:, :1], bool), changes], axis=1)
    return jnp.where(opens.reshape(opens.shape + (1,) * (scores.ndim - 2)), scores, before)


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
        # Normalized in at least fp32 (`dew.nn.categorical`) over the logits each position reads
        # (`aligned`): the reveal draws its categorical from these, and a bf16 log partition would
        # quantize it. The mask token takes no probability.
        logits = aligned(self.model, logits, None if self.inputs is None else self.inputs.token_fields)
        log_probs = categorical.log_probs(logits, jnp.arange(logits.shape[-1]) != self.process.mask_id)
        # A masked position whose scores hold no distribution stays masked: its argmax is no token.
        masked = self.masked(x_t) & wellformed(log_probs)
        filled = jnp.where(masked, jnp.argmax(log_probs, axis=-1), x_t)
        return filled, log_probs


@dataclass(frozen=True)
class Unmask:
    """Integrates a `DiscreteProcess` with MDLM's reverse step from t to s < t.

    Each masked position is revealed with probability
    (alpha(s) - alpha(t)) / (1 - alpha(t)), and a revealed position takes a
    token drawn from the model's categorical distribution. The rest stay
    masked, as does a position whose scores hold no distribution (a NaN), so
    a walk never commits an arbitrary token: generation refuses a response
    left masked. Stepping any other kind of process raises `ValueError`.
    """

    def init(self, x, times, process, *, key) -> tuple:
        return ()

    def step(self, x, t, t_next, denoised, log_probs, state, key, process, denoise):
        chance = self._reveal_chance(process, t, t_next)
        reveal_key, token_key = jax.random.split(key)
        reveal = jax.random.uniform(reveal_key, x.shape) < chance[:, None]
        revealed = denoise.masked(x) & reveal
        drawn = draw(token_key, log_probs, revealed, REVEALED_UNDEFINED)
        # A position whose scores hold no distribution stays masked rather than take an arbitrary token.
        return jnp.where(revealed & wellformed(log_probs), drawn, x), state

    def step_rows(self, x, t, t_next, denoised, log_probs, state, keys, process, denoise):
        """`step`, row `i` drawing its reveals and tokens from `keys[i]` alone (`RowSolver`), as the
        row sampled by itself draws them."""
        chance = self._reveal_chance(process, t, t_next)
        split = jax.vmap(jax.random.split)(counted(keys))
        reveal = jax.vmap(lambda key: jax.random.uniform(key, x.shape[1:]))(split[:, 0]) < chance[:, None]
        revealed = denoise.masked(x) & reveal
        drawn = row_draw(split[:, 1], log_probs, revealed, REVEALED_UNDEFINED)
        return jnp.where(revealed & wellformed(log_probs), drawn, x), state

    @staticmethod
    def _reveal_chance(process, t, t_next):
        """Each row's chance of revealing a masked position from `t` to `t_next`."""
        if not isinstance(process, DiscreteProcess):
            raise ValueError(
                f"Unmask reveals masked tokens, so it needs a DiscreteProcess and not "
                f"{type(process).__name__}")
        alpha_t = process.schedule.alpha(t)
        return (process.schedule.alpha(t_next) - alpha_t) / (1 - alpha_t)


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
    from dew.nn.protocols import Bounded

    refuse_causal(model)
    for name, value, minimum in (("max_new_tokens", budget, 0), ("steps", steps, 1), ("n", n, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    # A model that declares no vocabulary or capacity is checked for neither, as text generation does.
    bounded = isinstance(model, Bounded)
    vocab = model.vocab_size if bounded else None
    # Ids by the one law every parser of them reads; the vocabulary bounds them here.
    ids = {"mask_id": [token_id(process.mask_id, where="mask_id")],
           "pad_token_id": [token_id(pad_id, where="pad_token_id")],
           "eos_token_ids": token_ids(eos_ids, where="eos_token_ids") if eos_ids else []}
    for name, held in ids.items():
        if vocab is not None and any(token >= vocab for token in held):
            raise ValueError(f"{name} holds an id outside the vocabulary")
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

    The prompt's real slots (`real_slots`) and logical positions are written
    out first, so the response's continue from them (`ModelInputs.extended`):
    its slots are real in the rows that hold a prompt, the only rows whose
    slots may change, and count on from the prompt's last real token. The
    positions are the ones the model reads (`Coordinates.places`): the given
    ones, else each real token's place in its packed document, else its rank
    among its row's real tokens.
    """
    batch, prompt = inputs.tokens.shape
    fields = inputs.token_fields
    valid = real_slots(fields, (batch, prompt))
    positions = coordinates((batch, prompt), segments=fields.get("segment_ids"), valid=valid,
                            positions=fields.get("positions")).places()
    written = replace(inputs, token_fields={**fields, "positions": positions, "attention_mask": valid})
    extended = written.extended(jnp.full((batch, width), mask_id, jnp.int32))
    return extended, extended.token_fields["attention_mask"] & (jnp.arange(prompt + width) >= prompt)


def _generate(model: nn.Module, variables: Variables, inputs: ModelInputs, keys: jax.Array,
              process: DiscreteProcess, solver: Unmask, budget: int, steps: int, n: int,
              eos_ids: tuple[int, ...], pad_id: int) -> tuple[CanvasGeneration, jax.Array]:
    """The compiled body: one masked walk per row, `n` continuations each, and whether a response
    position stayed masked, its scores holding no distribution at every step it would have revealed.

    One model call denoises the batch at each step; only the random draws
    map over row keys. Mapping the whole model would batch its kernels too,
    which ragged_dot cannot do with an unmapped expert matrix. A continuation
    keeps tokens up to its first EOS and reports the steps it took.
    """
    from dew.sampling.sample import sample
    from dew.sampling.solvers import RowKeys

    batch, prompt = inputs.tokens.shape
    if budget == 0:
        zeros = jnp.zeros((batch * n,), jnp.int32)
        return (CanvasGeneration(jnp.repeat(inputs.tokens, n, axis=0), zeros, zeros.astype(bool), zeros),
                jnp.zeros((), bool))
    full, mutable = _response_inputs(inputs, budget, process.mask_id)

    denoise = process.denoiser(model, variables, inputs=full, mutable_mask=mutable)

    def continued(draw_keys):
        tokens = sample(denoise, full.tokens, steps, solver=solver, key=RowKeys(draw_keys))
        active = jnp.any(mutable, axis=-1)
        undefined = jnp.any(mutable & (tokens == process.mask_id))
        response, length, ended = jax.vmap(
            lambda row, count: through_eos(row, count, eos_ids, pad_id))(
                tokens[:, prompt:], jnp.where(active, budget, 0))
        return CanvasGeneration(jnp.concatenate([tokens[:, :prompt], response], axis=-1), length,
                                ended, jnp.where(active, steps, 0)), undefined

    generated, undefined = jax.lax.map(continued, continuation_keys(keys, n))
    return prompt_major(generated), jnp.any(undefined)




__all__ = ["MDLM", "DiscreteDenoiser", "DiscreteProcess", "LogLinear", "MaskingSchedule", "Unmask", "aligned"]
