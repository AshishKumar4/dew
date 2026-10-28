"""Masked diffusion language modelling (MDLM, by Sahoo and coauthors, 2024).

A row of token ids is corrupted by masking each position with the process's
probability at a drawn time. The model reads the whole corrupted row, its
states attending both ways (`dew.nn.protocols.TokenModel` with `causal` False),
and predicts the original tokens through its head. The loss
is the cross entropy at the masked positions, under a distribution that
gives the mask token no mass (MDLM's SUBS parameterization), weighted by
the process's NELBO weight and averaged over every position of the batch.
That average is the continuous-time negative ELBO the paper trains. The
cross entropy runs the head by the one policy every token objective takes
(`dew.objectives.lm.head`): its whole logits kept where they fit, tiled where
the fit ladder moves it.

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
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from dew.artifacts import TokenScores
from dew.batch import TEXT_KEY, Batch, Variables, real_slots, thaw
from dew.diffusion.discrete import MDLM_STEPS, REVEALED_UNDEFINED, DiscreteProcess, Unmask, aligned
from dew.inference.tasks import MaskedGeneration, MaskedSettings, recorded_processor, recorded_tokenizer
from dew.inputs import Field, InputSpec
from dew.nn.moe import ROUTER_COLLECTION
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import Lacks, TokenModel
from dew.objectives.base import OMITTED, Aux, EMASpec, Omitted, Ratio, Shown, Source, Step, ended, model_rngs
from dew.objectives.lm.chunked import final_read, refuse_without_logits
from dew.objectives.lm.head import HeadedObjective
from dew.objectives.lm.objective import batch_text, text_preview
from dew.objectives.routing import ExpertBalance, RoutedStatistics, apply_routed, reduce_routed
from dew.sampling.sample import sample

if TYPE_CHECKING:
    from dew.inference.tasks import Processor

_DEFAULT_SOLVER = Unmask()


class MaskedDiffusionObjective(HeadedObjective[Ratio | RoutedStatistics, Variables]):
    """Trains a masked diffusion model on the MDLM negative ELBO.

    A batch holds `[B, seq_len]` token ids under `batch["text"]`, and packed
    windows also their `text_coordinates` (`dew.batch.Coordinates`).
    `head_chunks` and `head_tile` set how the head is scored, as an LM's
    (`LMObjective`). `balance` trains the routers of a mixture as an LM's does
    (`ExpertBalance`, `dew.objectives.routing`).
    """

    saved_task = MaskedGeneration
    settings = MaskedSettings
    shown: Mapping[str, Shown] = {
        "masked_accuracy": Shown(better="higher", percent=True), "masked_fraction": Shown(percent=True)}

    def __init__(
        self,
        model: nn.Module | Source,
        process: DiscreteProcess,
        seq_len: int,
        *,
        head_chunks: int = 4,
        head_tile: tuple[int, int] | Literal['whole', 'tiled'] | None = None,
        balance: ExpertBalance | None = None,
        ema_decay: float | None = 0.999,
        solver: Unmask = _DEFAULT_SOLVER,
        steps: int = MDLM_STEPS,
        samples: int = 4,
        decode: Callable[[Sequence[int]], str] | None = None,
        variables: Variables | None | Omitted = OMITTED,
        processor: Processor | None | Omitted = OMITTED,
        eos_token_ids: Sequence[int] | Omitted = OMITTED,
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
        (`Objective.bind_model`), as `eos_token_ids`, the ids its task's
        responses stop at, overrides the source's (`ended`)."""
        self.eos_token_ids = ended(model, eos_token_ids)
        model = self.bind_model(model, variables=variables, processor=processor)
        if not isinstance(model, TokenModel) or model.causal:
            raise Lacks("bidirectional",
                        "a masked diffusion model reads the whole corrupted row, so its states attend "
                        "both ways (TokenModel with causal False), as CausalTransformer(causal=False)'s do")
        refuse_without_logits(model, "masked diffusion")
        self.model = model
        self.process = process
        self.seq_len = seq_len
        self.balance = balance
        self.use_head(head_chunks, head_tile)
        self.solver = solver
        self.steps = steps
        self.samples = samples
        self.decode = decode
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len,)))
        self.ema = EMASpec.constant(ema_decay)

    @property
    def _sample(self):
        return self.program('sample', self._sample_impl, static_argnames=("count",))

    def data(self, tokenizer=None):
        from dew.data.pipeline import ObjectiveData
        from dew.data.steps.text import FixedTokens

        geometry = FixedTokens(self.seq_len)
        return ObjectiveData(train=(geometry,), evaluate=(geometry,))

    def task_settings(self) -> MaskedSettings:
        """The row length, response budget, process, solver, sampling steps, end-of-sequence ids, and
        tokenizer or source processor: the task's own."""
        return MaskedSettings(seq_len=self.seq_len, max_new_tokens=self.seq_len, process=self.process,
                              solver=self.solver, sampling_steps=self.steps, eos_token_ids=self.eos_token_ids,
                              tokenizer=recorded_tokenizer(self.processor),
                              processor=recorded_processor(self.processor))

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def row_tokens(self, batch: Batch) -> int | None:
        """The tokens of each row, every one of them a slot the corruption may mask."""
        return self.inputs.sample.held(batch)[-1] if self.inputs is not None else None

    def loss(self, variables, batch, step: Step):
        nelbo, aux, _ = self._scored(variables, batch, step, train=step.training)
        return nelbo, aux

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        """Return the negative ELBO of every token in the batch.

        One noise level and one masking are drawn by each example, as in
        training (`Step.sample`), with dropout off. Every real token counts: a masked token
        scores its weighted cross entropy, a visible one scores zero, and a
        packed window's padding has no weight. So `perplexity` over a
        validation pass is the exponential of the ELBO bound per token, the
        number MDLM reports."""
        return self._scored(params, batch, step, train=False)[2]

    def _validation_scores(self, params, batch, step: Step) -> tuple[Ratio | RoutedStatistics, TokenScores]:
        nelbo, _, scores = self._scored(params, batch, step, train=False)
        return nelbo, scores

    def reduce_loss(self, stats: Ratio | RoutedStatistics) -> tuple[jax.Array, jax.Array]:
        return reduce_routed(self.balance, stats)

    def apply_effects(self, variables: Variables, effects: Variables) -> Variables:
        return apply_routed(self.balance, variables, effects)

    def _scored(self, params, batch, step: Step, *, train: bool
                ) -> tuple[Ratio | RoutedStatistics, Aux[Variables], TokenScores]:
        """The negative ELBO, its reports and the per-token scores, from one corruption of the batch,
        with the attention logit maxima of its forward where the step asks for them (`Step.qk_stats`),
        and the routers' balance terms where the objective balances them (`balance`)."""
        tokens, losses, weights, counted, predicted, real, sown = self._token_losses(params, batch, step,
                                                                                   train=train)
        prediction = Ratio(self.row_mean(losses, batch, weights=weights).total,
                           self.row_mean(real.astype(at_least_fp32(losses.dtype)), batch).total)
        correct = predicted == tokens
        accuracy, _ = self.accuracy(correct.astype(losses.dtype), batch, counted).mean()
        reported = {
            "masked_accuracy": accuracy,
            "masked_fraction": jnp.sum(counted) / jnp.maximum(jnp.sum(real, dtype=losses.dtype), 1.0),
        }
        statistics: Ratio | RoutedStatistics = prediction
        effects = None
        if self.balance is not None:
            fields = batch_text(batch).token_fields
            rows = tokens.shape[0]
            queries = real_slots(fields, tokens.shape) & (self.row_weights(batch, rows) > 0)[:, None]
            routing = sown.get(ROUTER_COLLECTION, {})
            statistics, effects, metrics = self.balance.compose(
                params, prediction, routing, batch, queries, fields)
            reported.update(metrics)
        qk = sown.get("qk", {}) if step.qk_stats else None
        return statistics, Aux(metrics=reported, qk_stats=qk, effects=effects), TokenScores(
            losses * weights, real.astype(losses.dtype), correct)

    def _token_losses(self, params, batch, step: Step, *, train: bool):
        """Corrupt the batch once, each row's noise level and mask its example's own (`Step.sample`),
        and score it.

        Returns the rows, their per-token cross entropies under that
        corruption, the time weight of each masked token, the mask itself,
        the argmax prediction, which slots hold text, and the collections the forward sowed: `qk` where
        the step asks for it (`Step.qk_stats`) and the routers' where the objective balances them.

        The model reads every field the batch's `ModelInputs` carries, and a
        model that takes no such field refuses it. A packed batch
        (``presets.lm(..., pack="first-fit")``) lays each window's documents out in
        `text_coordinates`, segment 0 for the padded tail, with their
        positions. Each document then attends to itself alone, in both
        directions, with its own positions, and the tail is neither masked
        nor scored: a packed window scores as its documents would one by one.
        Padding that the validity marks (`attention_mask`, or the coordinates'
        `valid`) is not masked or scored either (`real_slots`), and a media
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
        real = real_slots(fields, tokens.shape)
        for media in ("image_indices", "audio_indices"):
            if media in fields:
                real &= fields[media] < 0
        rows = tokens.shape[0]
        # MDLM's antithetic times: each example a point in its own stratum of the update's examples.
        places, count = step.places(rows)
        uniforms = step.sample("time uniforms", jax.random.uniform, (), dtype=jnp.float32, rows=rows)
        t = step.boundary("times", self.process.stratified(uniforms, places, count))
        draws = step.sample("mask uniforms", jax.random.uniform, tokens.shape[1:], dtype=jnp.float32,
                            rows=rows)
        _, is_masked = self.process.masked(draws, tokens, t)
        is_masked = step.boundary("mask", is_masked & real)
        dropout_key = jax.random.fold_in(step.key, 2)
        masked = jnp.where(is_masked, self.process.mask_id, tokens)

        # Each position's prediction, as the checkpoint's logits line up with its inputs.
        opened = (["qk"] if step.qk_stats else []) + ([ROUTER_COLLECTION] if self.balance is not None else [])
        observed = final_read(self.model, params, masked, train=train,
                              rngs=model_rngs(dropout_key, training=train),
                              mutable=opened or False, capture_intermediates=False,
                              **prepared.kwargs())
        read, sown = observed if opened else (observed, {})
        read = aligned(self.model, read, fields)
        # MDLM's SUBS parameterization gives the mask token no mass: it is
        # never a target, so the partition and the prediction leave it out.
        mask_id = self.process.mask_id
        losses, predicted, _ = self.head_terms(self.model, params, read, tokens, excluded=mask_id)
        # A slot that holds no text scores nothing: its token may be the mask's, which the partition
        # leaves out, and its +inf times a zero weight would be NaN.
        losses = jnp.where(real, losses, 0.0)
        counted = is_masked.astype(losses.dtype)
        weights = step.boundary("loss weights", self.process.weight(t))
        return (tokens, losses, counted * weights[:, None], counted, predicted, real, sown)

    def _sample_impl(self, params, key, *, count: int):
        """`count` rows sampled from noise, and whether a position stayed masked, its scores holding
        no distribution (`Unmask`)."""
        denoise = self.process.denoiser(self.model, thaw(params))
        x_T = self.process.noise(key, (count, self.seq_len))
        sampled = sample(denoise, x_T, self.steps, solver=self.solver, key=key)
        return sampled, jnp.any(sampled == self.process.mask_id)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Generate `samples` rows on every process, then decode them on process zero.

        The other processes return None. Without `decode`, the artifact holds
        the ids alone.
        """
        def generate(prepared):
            weights, count = prepared
            sampled, undefined = self._sample(weights, step.key, count=count)
            if bool(jax.device_get(undefined)):
                raise ValueError(REVEALED_UNDEFINED)
            return sampled, None, None

        return text_preview("masked diffusion preview",
                             lambda: (self.evaluation_variables(params, step), self.samples), generate,
                             self.decode)
