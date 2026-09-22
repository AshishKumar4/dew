"""Masked diffusion language modelling (MDLM, Sahoo et al. 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time; the model, a `CausalTransformer` with
`causal=False`, reads the whole corrupted row and predicts the original
tokens; the loss is the cross entropy at the masked positions weighted by the
process's NELBO weight, averaged over every position of the batch, which is
the continuous-time negative ELBO the paper trains. The cross entropy is the
LM objective's chunked one, which holds one vocabulary slice of logits at a time.

Evaluation generates one token row per input row for custom text metrics.
The separate preview hook generates and decodes the configured display count.

`pretrained` continues from a released masked-diffusion checkpoint (LLaDA,
Dream) instead of a fresh init. It reaches the trainer's state JIT as data
through `held_variables`, so the loaded tree is an argument of that
compilation rather than a constant embedded in the executable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import TextSamples, TokenScores, agree_process_phase, collective_host
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import objectives
from dew.sampling.sample import sample

if TYPE_CHECKING:
    from dew.inference.tasks import MaskedGeneration, Processor
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.training.state import TrainState

TEXT_KEY = "text"


@objectives("masked_diffusion")
class MaskedDiffusionObjective(Objective[Mean]):
    """The MDLM negative ELBO over `[B, seq_len]` rows of `batch["text"]`."""

    artifact = TextSamples

    def __init__(
        self,
        model: CausalTransformer,
        process: DiscreteProcess,
        seq_len: int,
        *,
        head_chunks: int = 4,
        ema_decay: float | None = 0.999,
        sampler: Unmask = Unmask(),
        steps: int = MDLM_STEPS,
        samples: int = 4,
        decode: Callable[[Sequence[int]], str] | None = None,
        pretrained: Variables | None = None,
    ):
        """`seq_len` is the width of a batch row; `sampler`, `steps` and
        `samples` are how evaluation unmasks; `decode` turns a row of ids into
        the text the artifact shows, and None shows the ids alone.

        `pretrained` is a released masked-diffusion checkpoint's variables as
        `load_pretrained` returns them, so a run continues from LLaDA's or
        Dream's weights instead of a fresh init; None draws the init."""
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
        self.pretrained = pretrained
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len,)))
        self.ema = None if ema_decay is None else EMASpec(decay=optax.constant_schedule(ema_decay))
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def pipeline(self, state: TrainState, *, ema: bool = True, processor: Processor | None = None) -> MaskedGeneration:
        """The published weights as a native full-response MDLM task."""
        from dew.inference.tasks import MaskedGeneration

        return MaskedGeneration(self.model, self._pipeline_weights(state, ema), self.process,
                                processor, sampler=self.sampler, steps=self.steps)

    def held_variables(self) -> Variables | None:
        """The checkpoint this run continues from, or None for a fresh init."""
        return self.pretrained

    def init(self, key, variables: Variables | None = None):
        pretrained = self.pretrained if variables is None else variables
        if pretrained is None:
            return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))
        if "params" not in pretrained:
            raise ValueError(
                "pretrained is the variables dict ({'params': ...}) that "
                "load_pretrained and model.init return")
        return pretrained

    def loss(self, params, batch, step: Step):
        tokens, losses, weights, counted, predicted = self._token_losses(params, batch, step.key, train=True)
        nelbo = Mean(jnp.sum(losses * weights), jnp.asarray(tokens.size, jnp.float32))
        correct = (predicted == tokens).astype(losses.dtype)
        return nelbo, Aux(metrics={
            "masked_accuracy": jnp.sum(correct * counted) / jnp.maximum(jnp.sum(counted), 1.0),
            "masked_fraction": jnp.mean(counted),
        })

    def evaluate(self, params, batch, step: Step) -> TokenScores:
        """The negative ELBO of every token in the batch.

        One noise level and one masking are drawn from the pass's key, as
        training draws them, with dropout off and the averaged weights when
        the run keeps them. Every token counts and carries its weighted masked
        cross entropy, zero where it was left visible, so `perplexity` over a
        validation pass is exp of the ELBO bound per token, the number MDLM
        reports."""
        params = params if step.ema is None else step.ema
        _, losses, weights, _, _ = self._token_losses(params, batch, step.key, train=False)
        return TokenScores(losses=losses * weights, weights=jnp.ones_like(losses))

    def _token_losses(self, params, batch, key, *, train: bool):
        """The rows, their per-token cross entropies under one corruption, the
        time weight of each masked token, the mask itself, and the argmax."""
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        if tokens.shape[-1] != self.seq_len:
            raise ValueError(
                f"the objective was built for {self.seq_len}-token rows, got {tokens.shape[-1]}")
        time_key, mask_key, dropout_key = jax.random.split(key, 3)
        t = self.process.sample_t(time_key, tokens.shape[0])
        masked, is_masked = self.process.corrupt(mask_key, tokens, t)

        hidden = self.model.apply(params, masked, train=train, rngs={"dropout": dropout_key},
                                  method=type(self.model).hidden_states)
        head = self.model.apply(params, params["params"], method=type(self.model).head_weight)
        losses, predicted, _ = chunked_cross_entropy(
            hidden, head, tokens, self.head_chunks,
            softcap=self.model.final_logit_softcap, precision=self.model.precision)
        counted = is_masked.astype(losses.dtype)
        return tokens, losses, counted * self.process.weight(t)[:, None], counted, predicted

    def _sample_impl(self, params, key, *, count: int):
        denoise = self.process.denoiser(self.model, params)
        x_T = self.process.noise(key, (count, self.seq_len))
        return sample(denoise, x_T, self.steps, solver=self.sampler, key=key)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Generate the configured display count, then decode on process zero."""
        error = None
        prepared = None
        try:
            params = params if step.ema is None else step.ema
            prepared = (self._sample, self.samples)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="masked diffusion preview setup")
        error = None
        tokens = None
        try:
            assert prepared is not None
            sample, count = prepared
            tokens = sample(params, step.key, count=count)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="masked diffusion preview generation")
        tokens = collective_host(tokens, phase="masked diffusion preview")
        if jax.process_index() != 0:
            return None
        assert tokens is not None
        texts = () if self.decode is None else tuple(
            self.decode(row.tolist()) for row in np.asarray(tokens))
        return TextSamples(tokens=tokens, texts=texts)
