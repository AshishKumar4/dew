"""Masked diffusion language modelling (MDLM, Sahoo et al. 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time; the model, a `CausalTransformer` with
`causal=False`, reads the whole corrupted row and predicts the original
tokens; the loss is the cross entropy at the masked positions weighted by the
process's NELBO weight, averaged over every position of the batch, which is
the continuous-time negative ELBO the paper trains. The cross entropy is the
LM objective's chunked one, so the full logits tensor is never held.

Evaluation unmasks a few rows from the fully masked state with the averaged
weights, which is the text a reader can judge.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import TextSamples
from dew.diffusion.discrete import DiscreteProcess, Unmask
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, EMASpec, Objective, Step
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.sampling.sample import sample

TEXT_KEY = "text"


class MaskedDiffusionObjective(Objective):
    """The MDLM negative ELBO over `[B, seq_len]` rows of `batch["text"]`."""

    artifact = TextSamples

    def __init__(
        self,
        model,
        process: DiscreteProcess,
        seq_len: int,
        *,
        head_chunks: int = 4,
        ema_decay: float = 0.999,
        sampler=Unmask(),
        steps: int = 64,
        samples: int = 4,
        decode: Optional[Callable[[Sequence[int]], str]] = None,
    ):
        """`seq_len` is the width of a batch row; `sampler`, `steps` and
        `samples` are how evaluation unmasks; `decode` turns a row of ids into
        the text the artifact shows, and None shows the ids alone."""
        if model.causal:
            raise ValueError(
                "a masked diffusion model reads the whole corrupted row, so it needs "
                "CausalTransformer(causal=False)")
        self.model = model
        self.process = process
        self.seq_len = seq_len
        self.head_chunks = head_chunks
        self.sampler = sampler
        self.steps = steps
        self.samples = samples
        self.decode = decode
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len,)))
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay))
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def init(self, key):
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def loss(self, params, batch, step: Step):
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        if tokens.shape[-1] != self.seq_len:
            raise ValueError(
                f"the objective was built for {self.seq_len}-token rows, got {tokens.shape[-1]}")
        time_key, mask_key, dropout_key = jax.random.split(step.key, 3)
        t = self.process.sample_t(time_key, tokens.shape[0])
        masked, is_masked = self.process.corrupt(mask_key, tokens, t)

        hidden = self.model.apply(params, masked, train=True, rngs={"dropout": dropout_key},
                                  method=type(self.model).hidden_states)
        head = self.model.apply(params, params["params"], method=type(self.model).head_weight)
        losses, predicted = chunked_cross_entropy(
            hidden, head, tokens, self.head_chunks,
            softcap=self.model.final_logit_softcap, precision=self.model.precision)

        counted = is_masked.astype(losses.dtype)
        weights = counted * self.process.weight(t)[:, None]
        nelbo = jnp.sum(losses * weights) / tokens.size
        correct = (predicted == tokens).astype(losses.dtype)
        return nelbo, Aux(metrics={
            "masked_accuracy": jnp.sum(correct * counted) / jnp.maximum(jnp.sum(counted), 1.0),
            "masked_fraction": jnp.mean(counted),
        })

    def _sample_impl(self, params, key, *, count: int):
        denoise = self.process.denoiser(self.model, params)
        x_T = self.process.noise(key, (count, self.seq_len))
        return sample(denoise, x_T, self.steps, solver=self.sampler, key=key)

    def evaluate(self, params, batch, step: Step) -> TextSamples:
        params = params if step.ema is None else step.ema
        tokens = self._sample(params, step.key, count=self.samples)
        texts = () if self.decode is None else tuple(
            self.decode(row.tolist()) for row in np.asarray(tokens))
        return TextSamples(tokens=tokens, texts=texts)
