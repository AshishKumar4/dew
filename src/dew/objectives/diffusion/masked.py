"""Masked diffusion language modelling (MDLM, by Sahoo and coauthors, 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time. The model reads the whole corrupted row, its
states attending both ways (`dew.nn.protocols.TokenModel` with `causal` False),
and predicts the original tokens through its head. The loss
is the cross entropy at the masked positions, under a distribution that
gives the mask token no mass (MDLM's SUBS parameterization), weighted by
the process's NELBO weight and averaged over every position of the batch.
That average is the continuous-time negative ELBO the paper trains. The
cross entropy is the LM objective's chunked one, which holds one vocabulary
slice of logits at a time where the head is a matrix.

Evaluation scores every row's corruption loss as `TokenScores`, so
perplexity-style metrics read the negative ELBO; the preview hook alone
generates, and decodes the configured display count.

`pretrained` continues from a released masked-diffusion checkpoint (LLaDA,
Dream) instead of a fresh init. It reaches the trainer's state JIT as data
through `held_variables`, so the loaded tree is an argument of that
compilation rather than a constant embedded in the executable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.artifacts import TextSamples, TokenScores
from dew.diffusion.discrete import MDLM_STEPS, DiscreteProcess, Unmask
from dew.inference.tasks import MaskedGeneration
from dew.inputs import Field, InputSpec
from dew.nn.protocols import Logits, TokenModel
from dew.objectives.base import (
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
    training_rngs,
)
from dew.objectives.lm.chunked import head_cross_entropy, logits_cross_entropy, reads_states
from dew.objectives.lm.objective import TEXT_KEY, batch_text, text_preview
from dew.records import JSON
from dew.sampling.sample import sample
from dew.sampling.solvers import Solver

if TYPE_CHECKING:
    from dew.inference.tasks import Processor

_DEFAULT_SOLVER = Unmask()


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

        `model` gives its logits, from its final states through its head or
        from its tokens (`chunked.reads_states`, `Logits`), and its states
        attend both ways (`TokenModel` with `causal` False), as
        `CausalTransformer(causal=False)`'s do. `solver`
        and `steps` set how generation unmasks, in the preview and in the
        task `pipeline` returns, and `samples` is how many rows the preview
        draws. `decode` turns a row of ids into the text the artifact shows;
        with None, the artifact shows the ids alone.

        `model` may be a loaded source, such as a released LLaDA or Dream
        checkpoint, and `variables` and `processor` override its own
        (`Objective.bind_model`)."""
        model = self.bind_model(model, variables=variables, processor=processor)
        if not isinstance(model, TokenModel) or model.causal:
            raise ValueError(
                "a masked diffusion model reads the whole corrupted row, so its states attend "
                "both ways (TokenModel with causal False), as CausalTransformer(causal=False)'s do")
        if not reads_states(model) and not isinstance(model, Logits):
            raise TypeError(
                f"masked diffusion scores a model's logits, and a {type(model).__name__} gives none: no "
                f"head over its final states (HiddenStates with AffineHead or LogitsFromHidden) and "
                f"no Logits")
        self.model = model
        self.process = process
        self.seq_len = seq_len
        self.head_chunks = head_chunks
        self.solver = solver
        self.steps = steps
        self.samples = samples
        self.decode = decode
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len,)))
        self.ema = EMASpec.constant(ema_decay)
        self._sample = jax.jit(self._sample_impl, static_argnames=("count",))

    def task_record(self) -> Mapping[str, JSON]:
        """The row length, tokenizer, process, solver and sampling steps."""
        from dew.inference.tasks import recorded_tokenizer
        from dew.registry import to_record
        return {'seq_len': self.seq_len, 'max_new_tokens': self.seq_len,
                'tokenizer': recorded_tokenizer(self.processor),
                'process': to_record(self.process, DiscreteProcess), 'solver': to_record(self.solver, Solver),
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

    def _evaluation_scores(self, params, batch, key) -> TokenScores:
        """Return the negative ELBO of every token in the batch.

        One noise level and one masking are drawn from the pass's key, as in
        training, with dropout off. Every real token counts: a masked token
        scores its weighted cross entropy, a visible one scores zero, and a
        packed window's padding has no weight. So `perplexity` over a
        validation pass is the exponential of the ELBO bound per token, the
        number MDLM reports."""
        tokens, losses, weights, _, predicted, real = self._token_losses(params, batch, key, train=False)
        return TokenScores(losses * weights, real.astype(losses.dtype), predicted == tokens)

    def _token_losses(self, params, batch, key, *, train: bool):
        """Corrupt the batch once and score it.

        Returns the rows, their per-token cross entropies under that
        corruption, the time weight of each masked token, the mask itself,
        the argmax prediction, and which slots hold text.

        The model reads every field the batch's `ModelInputs` carries, and a
        model that takes no such field refuses it. A packed batch
        (`TokenWindows(pack=True)`) names each window's documents in
        `text_segment_ids`, 0 for the padded tail, and their positions in
        `text_positions`. Each document then attends to itself alone, in both
        directions, with its own positions, and the tail is neither masked
        nor scored: a packed window scores as its documents would one by one.
        Padding under `attention_mask` is not scored either, and a media
        placeholder (`image_indices` or `audio_indices` at or above zero) is
        the media's slot, which the model reads and nothing masks or predicts.
        """
        params = thaw(params)
        prepared = batch_text(batch)
        tokens = prepared.tokens
        if tokens.shape[-1] != self.seq_len:
            raise ValueError(
                f"the objective was built for {self.seq_len}-token rows, got {tokens.shape[-1]}")
        fields = prepared.token_fields
        real = jnp.ones(tokens.shape, bool)
        if "segment_ids" in fields:
            real &= fields["segment_ids"] != 0
        if "attention_mask" in fields:
            real &= fields["attention_mask"].astype(bool)
        for media in ("image_indices", "audio_indices"):
            if media in fields:
                real &= fields[media] < 0
        time_key, mask_key, dropout_key = jax.random.split(key, 3)
        t = self.process.sample_t(time_key, tokens.shape[0])
        masked, is_masked = self.process.corrupt(mask_key, tokens, t)
        is_masked = is_masked & real
        masked = jnp.where(is_masked, masked, tokens)

        over_states = reads_states(self.model)
        read = self.model.apply(params, masked, train=train, rngs=training_rngs(dropout_key),
                                method="hidden_states" if over_states else "logits", mutable=False,
                                capture_intermediates=False, **prepared.kwargs())
        # MDLM's SUBS parameterization gives the mask token no mass: it is
        # never a target, so the partition and the prediction leave it out.
        mask_id = self.process.mask_id
        losses, predicted, _ = (
            head_cross_entropy(self.model, params, read, tokens, self.head_chunks, excluded=mask_id)
            if over_states else logits_cross_entropy(read, tokens, excluded=mask_id))
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
        def generate(prepared):
            weights, count = prepared
            return self._sample(weights, step.key, count=count), None

        return text_preview("masked diffusion preview",
                             lambda: (self.evaluation_variables(params, step), self.samples), generate,
                             self.decode)
