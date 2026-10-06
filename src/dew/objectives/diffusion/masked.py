"""Masked diffusion language modelling (MDLM, by Sahoo and coauthors, 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time. The model reads the whole corrupted row, its
states attending both ways (`dew.nn.protocols.Ordered` with `causal` False),
and predicts the original tokens through its head (`AffineHead`). The loss
is the cross entropy at the masked positions, under a distribution that
gives the mask token no mass (MDLM's SUBS parameterization), weighted by
the process's NELBO weight and averaged over every position of the batch.
That average is the continuous-time negative ELBO the paper trains. The
cross entropy is the LM objective's chunked one, which holds one vocabulary
slice of logits at a time.

Evaluation scores every row's corruption loss as `TokenScores`, so
perplexity-style metrics read the negative ELBO; the preview hook alone
generates, and decodes the configured display count.

`pretrained` continues from a released masked-diffusion checkpoint (LLaDA,
Dream) instead of a fresh init. It reaches the trainer's state JIT as data
through `held_variables`, so the loaded tree is an argument of that
compilation rather than a constant embedded in the executable.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.artifacts import TextSamples, TokenScores
from dew.coordination import agreed, collective_host
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask
from dew.inference.tasks import MaskedGeneration
from dew.inputs import Field, InputSpec
from dew.nn.protocols import AffineHead, HiddenStates, Ordered
from dew.objectives.base import (
    FROZEN,
    OMITTED,
    Aux,
    EMASpec,
    Objective,
    Omitted,
    Ratio,
    Shown,
    Source,
    Step,
    Variables,
    thaw,
)
from dew.objectives.lm.chunked import affine_head, chunked_cross_entropy
from dew.objectives.lm.objective import _batch_text
from dew.records import JSON
from dew.registry import objectives
from dew.sampling.sample import sample

if TYPE_CHECKING:
    from dew.inference.tasks import Processor

TEXT_KEY = "text"


_DEFAULT_SOLVER = Unmask()


@objectives("masked_diffusion")
class MaskedDiffusionObjective(Objective[Ratio]):
    """Trains a masked diffusion model on the MDLM negative ELBO.

    A batch holds `[B, seq_len]` token ids under `batch["text"]`, and packed
    windows also include `text_segment_ids` and `text_positions`.
    """

    artifact = TextSamples
    saved_task = MaskedGeneration
    shown: Mapping[str, Shown] = {
        "masked_accuracy": Shown(better="higher", percent=True), "masked_fraction": Shown(percent=True)}

    def __init__(
        self,
        model: nn.Module | Source,
        process: DiscreteProcess,
        seq_len: int,
        *,
        head_chunks: int = 4,
        ema_decay: float | None = 0.999,
        solver: Unmask = _DEFAULT_SOLVER,
        steps: int = MDLM_STEPS,
        samples: int = 4,
        decode: Callable[[Sequence[int]], str] | None = None,
        variables: Variables | None | Omitted = OMITTED,
        processor: Processor | None | Omitted = OMITTED,
    ):
        """Build an MDLM objective over `model` for `seq_len`-token rows.

        `model` gives its final states and its head (`HiddenStates`,
        `AffineHead`), and its states attend both ways (`Ordered` with
        `causal` False), as `CausalTransformer(causal=False)`'s do. `solver`
        and `steps` set how generation unmasks, in the preview and in the
        task `pipeline` returns, and `samples` is how many rows the preview
        draws. `decode` turns a row of ids into the text the artifact shows;
        with None, the artifact shows the ids alone.

        `variables` is the tree training starts from: a released
        masked-diffusion checkpoint as `Pretrained.load` returns it, so a run
        continues from LLaDA's or Dream's weights, or an adapter's split of
        one, kept as given. None starts from a fresh init. `model` may be the
        loaded source itself, which supplies its model, variables and
        processor.

        `processor` is what `pipeline` uses to turn text into ids and decode
        them, unless it is given another one. A run records its tokenizer."""
        model = self.bind_model(model, variables=variables, processor=processor)
        lacking = [read.__name__ for read in (HiddenStates, AffineHead) if not isinstance(model, read)]
        if lacking:
            raise TypeError(
                f"masked diffusion scores a model's final states through its head, and a "
                f"{type(model).__name__} gives no {' or '.join(lacking)}")
        if not isinstance(model, Ordered) or model.causal:
            raise ValueError(
                "a masked diffusion model reads the whole corrupted row, so its states attend "
                "both ways (Ordered with causal False), as CausalTransformer(causal=False)'s do")
        self.model = model
        self.process = process
        self.seq_len = seq_len
        self.head_chunks = head_chunks
        self.solver = solver
        self.steps = steps
        self.samples = samples
        self.decode = decode
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len,)))
        # The EMA follows what moves; the frozen collection never does.
        self.ema = None if ema_decay is None else EMASpec(
            decay=optax.constant_schedule(ema_decay), select=lambda path: path[0] != FROZEN)
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def task_record(self) -> Mapping[str, JSON]:
        """The row length, tokenizer, process, solver and sampling steps."""
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record
        return {'seq_len': self.seq_len, 'sample_tokens': self.seq_len,
                'tokenizer': recorded_tokenizer(self.processor),
                'process': self.process.to_json(), 'solver': to_record(self.solver, type(self.solver)),
                'sampling_steps': self.steps}

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> MaskedGeneration:
        """Return the model over `variables` as a full-response MDLM task, with this
        objective's solver and steps."""
        from dew.inference.tasks import MaskedGeneration

        return MaskedGeneration(self.model, variables, self.process,
                                self.processor if processor is OMITTED else processor,
                                solver=self.solver, steps=self.steps)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def loss(self, variables, batch, step: Step):
        tokens, losses, weights, counted, predicted, real = self._token_losses(
            variables, batch, step.key, train=True)
        nelbo = Ratio(self.row_mean(losses * weights, batch).total,
                      self.row_mean(real.astype(jnp.float32), batch).total)
        correct = (predicted == tokens).astype(losses.dtype)
        accuracy, _ = self.accuracy(correct, batch, counted).mean()
        return nelbo, Aux(metrics={
            "masked_accuracy": accuracy,
            "masked_fraction": jnp.sum(counted) / jnp.maximum(jnp.sum(real, dtype=losses.dtype), 1.0),
        })

    def evaluate(self, params, batch, step: Step) -> TokenScores:
        """Return the negative ELBO of every token in the batch.

        One noise level and one masking are drawn from the pass's key, as in
        training. Dropout is off, and the averaged weights are used when the
        run keeps them. Every real token counts: a masked token scores its
        weighted cross entropy, a visible one scores zero, and a packed
        window's padding has no weight. So `perplexity` over a validation
        pass is the exponential of the ELBO bound per token, the number MDLM
        reports."""
        params = self.evaluation_variables(params, step)
        losses, weights, correct = self._scored(params, batch, step.key)
        return TokenScores(losses=losses, weights=weights, correct=correct)

    @functools.cached_property
    def _scored(self):
        """Compile the evaluation's corruption and scores once per objective.
        Run op by op, the model's forward would dispatch every operation of
        every validation batch from the host, and jax's eager shard_map
        refuses the chunked head's map over the data axis alone."""
        def scored(params, batch, key):
            tokens, losses, weights, _, predicted, real = self._token_losses(params, batch, key, train=False)
            return losses * weights, real.astype(losses.dtype), predicted == tokens

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
        params = thaw(params)
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

        hidden = self.model.apply(
            params,
            masked,
            train=train,
            positions=prepared.token_fields.get("positions"),
            segment_ids=segment_ids,
            rngs={"dropout": dropout_key},
            method="hidden_states",
        )
        head = affine_head(self.model, params)
        # MDLM's SUBS parameterization gives the mask token no mass: it is
        # never a target, so the partition and the prediction leave it out.
        losses, predicted, _ = chunked_cross_entropy(
            hidden, head.matrix, tokens, self.head_chunks, vocab_major=head.vocab_major,
            softcap=head.softcap, precision=head.precision, excluded=self.process.mask_id,
            bias=head.bias)
        counted = is_masked.astype(losses.dtype)
        return tokens, losses, counted * self.process.weight(t)[:, None], counted, predicted, real

    def _sample_impl(self, params, key, *, count: int):
        denoise = self.process.denoiser(self.model, thaw(params))
        x_T = self.process.noise(key, (count, self.seq_len))
        return sample(denoise, x_T, self.steps, solver=self.solver, key=key)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Generate `samples` rows on every process, then decode them on process zero.

        The other processes return None. Without `decode`, the artifact holds
        the ids alone.
        """
        def setup():
            return self.evaluation_variables(params, step), self.samples

        weights, count = agreed("masked diffusion preview setup", setup)
        tokens = agreed("masked diffusion preview generation",
                        lambda: self._sample(weights, step.key, count=count))
        tokens = collective_host(tokens, phase="masked diffusion preview")
        if jax.process_index() != 0:
            return None
        texts = () if self.decode is None else tuple(
            self.decode(row.tolist()) for row in np.asarray(tokens))
        return TextSamples(tokens=tokens, texts=texts)
