"""Masked diffusion language modelling (MDLM, Sahoo et al. 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time. The model, a `CausalTransformer` with
`causal=False`, reads the whole corrupted row and predicts the original
tokens. The loss is the cross entropy at the masked positions, weighted by
the process's NELBO weight and averaged over every position of the batch.
That average is the continuous-time negative ELBO the paper trains. The
cross entropy is the LM objective's chunked one, which holds one vocabulary
slice of logits at a time.

Evaluation generates one token row per input row for custom text metrics.
The separate preview hook generates and decodes the configured display count.

`pretrained` continues from a released masked-diffusion checkpoint (LLaDA,
Dream) instead of a fresh init. It reaches the trainer's state JIT as data
through `held_variables`, so the loaded tree is an argument of that
compilation rather than a constant embedded in the executable.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import TextSamples, TokenScores, agreed, collective_host
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask
from dew.inputs import Field, InputSpec
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.objectives.lm.objective import _batch_text
from dew.registry import objectives
from dew.sampling.sample import sample

if TYPE_CHECKING:
    from dew.inference.tasks import MaskedGeneration, Processor
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.training.state import TrainState

TEXT_KEY = "text"


@objectives("masked_diffusion")
class MaskedDiffusionObjective(Objective[Mean]):
    """Train a masked diffusion model on the MDLM negative ELBO.

    The rows are `[B, seq_len]` token ids under `batch["text"]`; packed
    windows carry `text_segment_ids` and `text_positions` beside them.
    """

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
        """Build an MDLM objective over `model` for `seq_len`-token rows.

        `sampler`, `steps` and `samples` are how evaluation unmasks.
        `decode` turns a row of ids into the text the artifact shows, and
        None shows the ids alone.

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
        """Publish the state's weights as a native full-response MDLM task."""
        from dew.inference.tasks import MaskedGeneration

        return MaskedGeneration(self.model, self._pipeline_weights(state, ema), self.process,
                                processor, sampler=self.sampler, steps=self.steps)

    def held_variables(self) -> Variables | None:
        """Return the checkpoint this run continues from, or None for a fresh init."""
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
        tokens, losses, weights, counted, predicted, real = self._token_losses(
            params, batch, step.key, train=True)
        nelbo = Mean(jnp.sum(losses * weights), jnp.sum(real, dtype=jnp.float32))
        correct = (predicted == tokens).astype(losses.dtype)
        return nelbo, Aux(metrics={
            "masked_accuracy": jnp.sum(correct * counted) / jnp.maximum(jnp.sum(counted), 1.0),
            "masked_fraction": jnp.sum(counted) / jnp.maximum(jnp.sum(real, dtype=losses.dtype), 1.0),
        })

    def evaluate(self, params, batch, step: Step) -> TokenScores:
        """Score the negative ELBO of every token in the batch.

        One noise level and one masking are drawn from the pass's key, as
        training draws them, with dropout off and the averaged weights when
        the run keeps them. Every real token counts and carries its weighted
        masked cross entropy, zero where it was left visible, and a packed
        window's padding weighs nothing, so `perplexity` over a validation
        pass is exp of the ELBO bound per token, the number MDLM reports."""
        params = params if step.ema is None else step.ema
        losses, weights = self._scored(params, batch, step.key)
        return TokenScores(losses=losses, weights=weights)

    @functools.cached_property
    def _scored(self):
        """Compile the evaluation's corruption and scores once per objective.
        Run op by op, the model's forward would dispatch every operation of
        every validation batch from the host, and jax's eager shard_map
        refuses the chunked head's map over the data axis alone."""
        def scored(params, batch, key):
            _, losses, weights, _, _, real = self._token_losses(params, batch, key, train=False)
            return losses * weights, real.astype(losses.dtype)

        return jax.jit(scored)

    def _token_losses(self, params, batch, key, *, train: bool):
        """Corrupt the batch once and score it.

        Returns the rows, their per-token cross entropies under that
        corruption, the time weight of each masked token, the mask itself,
        the argmax prediction, and which slots hold real tokens.

        A packed batch (data:packed-tokens) names each window's documents in
        `text_segment_ids`, 0 for the padded tail, and their positions in
        `text_positions`. Each document then attends to itself alone, in both
        directions, with its own positions, and the tail is neither masked
        nor scored: a packed window scores as its documents would one by one.
        """
        prepared = _batch_text(batch)
        unread = sorted(set(prepared.token_fields) - {"positions", "segment_ids"})
        if unread or prepared.conditioning:
            raise ValueError(
                f"masked diffusion reads token ids and their packing; this batch also carries "
                f"{unread + sorted(prepared.conditioning)}")
        tokens = prepared.tokens
        if tokens.shape[-1] != self.seq_len:
            raise ValueError(
                f"the objective was built for {self.seq_len}-token rows, got {tokens.shape[-1]}")
        segment_ids = prepared.token_fields.get("segment_ids")
        real = jnp.ones(tokens.shape, bool) if segment_ids is None else segment_ids != 0
        time_key, mask_key, dropout_key = jax.random.split(key, 3)
        t = self.process.sample_t(time_key, tokens.shape[0])
        masked, is_masked = self.process.corrupt(mask_key, tokens, t)
        is_masked = is_masked & real
        masked = jnp.where(is_masked, masked, tokens)

        hidden = self.model.apply(params, masked, train=train, positions=prepared.token_fields.get("positions"),
                                  segment_ids=segment_ids, rngs={"dropout": dropout_key},
                                  method=type(self.model).hidden_states)
        head = self.model.apply(params, params["params"], method=type(self.model).head_weight)
        losses, predicted, _ = chunked_cross_entropy(
            hidden, head, tokens, self.head_chunks,
            softcap=self.model.final_logit_softcap, precision=self.model.precision)
        counted = is_masked.astype(losses.dtype)
        return tokens, losses, counted * self.process.weight(t)[:, None], counted, predicted, real

    def _sample_impl(self, params, key, *, count: int):
        denoise = self.process.denoiser(self.model, params)
        x_T = self.process.noise(key, (count, self.seq_len))
        return sample(denoise, x_T, self.steps, solver=self.sampler, key=key)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Generate the configured display count, then decode on process zero."""
        def setup():
            return params if step.ema is None else step.ema, self.samples

        weights, count = agreed("masked diffusion preview setup", setup)
        tokens = agreed("masked diffusion preview generation",
                        lambda: self._sample(weights, step.key, count=count))
        tokens = collective_host(tokens, phase="masked diffusion preview")
        if jax.process_index() != 0:
            return None
        texts = () if self.decode is None else tuple(
            self.decode(row.tolist()) for row in np.asarray(tokens))
        return TextSamples(tokens=tokens, texts=texts)
