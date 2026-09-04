"""The autoregressive language modelling objective.

Next-token prediction over packed token ids. A batch row holds `seq_len + 1`
ids, the model sees all but the last, and the targets are the same row shifted
by one; the causal mask lives in the backbone, so nothing here has to know how
the model keeps the future out of a prediction.

Cross entropy is computed in float32 even when the model runs in bfloat16: a
bf16 logsumexp over a large vocabulary loses enough precision to move the loss
and, through it, the gradient. It is also computed one vocabulary chunk at a
time, because the full `[tokens, vocab]` logits tensor is the largest thing in
a step and every pass over it costs bandwidth; `chunked` holds the arithmetic
and the reason. Padding is excluded only when the run names the pad id,
because packed token files have no padding and masking out a real id would
quietly drop those tokens from the average.

Evaluation returns the teacher-forced per-token scores, which `perplexity`
reduces over a whole pass with the token counts, and, when asked for, the text
the model writes from a fixed prompt, which is the only part of a training
curve a human can read.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import TextSamples, TokenScores
from dew.inputs import Field, InputSpec
from dew.nn.moe import calculate_load_balance_updates
from dew.objectives.base import Aux, EMASpec, Objective, Step, Variables
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import metrics

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""


def prompt_batch(prompt) -> jax.Array:
    """`[B, P]` int32 ids from one prompt, or several of the same length."""
    try:
        ids = np.asarray(prompt)
    except ValueError as ragged:
        raise ValueError("sample prompts have to be of equal length") from ragged
    if ids.ndim == 1:
        ids = ids[None]
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError(f"a sample prompt is non-empty [P] or [B, P] ids, got {ids.shape}")
    return jnp.asarray(ids, jnp.int32)


@dataclass(frozen=True)
class Samples:
    """The text an evaluation writes: `prompt` ids (one, or several of equal
    length), how many tokens to add, the sampling knobs, and the `decode`
    that turns ids back into a string."""
    prompt: Sequence[int] | Sequence[Sequence[int]]
    max_new_tokens: int
    temperature: float = 1.0
    top_k: Optional[int] = None
    decode: Callable[[list[int]], str] = lambda ids: str(ids)


def _packing(batch):
    """A packed batch's `segment_ids` and `positions`, None on a plain one."""
    segment_ids = batch.get("text_segment_ids")
    positions = batch.get("text_positions")
    return (None if segment_ids is None else jnp.asarray(segment_ids, jnp.int32),
            None if positions is None else jnp.asarray(positions, jnp.int32))


def balance(moe: Variables, routing: Variables, rate: float
            ) -> tuple[Variables, dict[str, jax.Array]]:
    """The `moe` collection with every router's bias moved against its load,
    and the load itself.

    `routing` is what the routers sowed under the `router` collection, the
    top-k expert indices at the module path the bias lives under. DeepSeek's
    update (arXiv 2408.15664) raises the bias of an expert below the average
    load and lowers it above, by `rate` a step. The load is reported as the
    busiest and the idlest expert's share of the routed tokens, averaged over
    the sparse layers; both read 1 / num_experts for an even router.
    """
    heaviest, lightest = [], []

    def update(path, bias):
        sown = routing
        for entry in path[:-1]:
            sown = sown[entry.key]
        (indices,) = sown["indices"]
        share = jnp.bincount(indices.ravel(), length=bias.shape[0]) / indices.size
        heaviest.append(share.max())
        lightest.append(share.min())
        return bias + calculate_load_balance_updates(indices, bias.shape[0], rate)

    balanced = jax.tree_util.tree_map_with_path(update, moe)
    return balanced, {"moe/max_load": jnp.mean(jnp.stack(heaviest)),
                      "moe/min_load": jnp.mean(jnp.stack(lightest))}


class LMObjective(Objective):
    """Shifted cross entropy; evaluation scores tokens and writes text."""

    artifact = TokenScores

    def __init__(
        self,
        model,
        seq_len: int,
        *,
        ema_decay: float = 0.999,
        pad_id: Optional[int] = None,
        head_chunks: int = 4,
        samples: Optional[Samples] = None,
        pretrained: Optional[Variables] = None,
        balance_rate: Optional[float] = None,
    ):
        """`head_chunks` is how many vocabulary slices the loss scores a batch
        in; four is the measured best on one RTX 4080 at vocabulary 50,304,
        see docs/research/lm-head.md.

        `pretrained` is a variables dict to start from instead of a fresh
        init, as dew.interop.hf_decoders.load_pretrained_decoder returns for a
        Hugging Face checkpoint. The trainer takes its whole initial state
        from `init`, so this is where continued pretraining begins.

        `balance_rate` moves each sparse layer's routing bias against its
        load by this much every step (DeepSeek's aux-loss-free balancing);
        the model has to keep that bias, `expert_bias=True` on the
        CausalTransformer. Unset leaves the bias where it is."""
        self.model = model
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.head_chunks = head_chunks
        self.samples = samples
        self.pretrained = pretrained
        self.balance_rate = balance_rate
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len + 1,)))
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay))
        if samples is not None:
            self._prompt = prompt_batch(samples.prompt)

    def init(self, key):
        if self.pretrained is not None:
            if "params" not in self.pretrained:
                raise ValueError(
                    "pretrained is the variables dict ({'params': ...}) that "
                    "load_pretrained_decoder and model.init return")
            return self.pretrained
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def token_scores(self, params, tokens, train: bool = False, rngs=None,
                     segment_ids=None, positions=None, routing: bool = False):
        """Per-token next-token cross entropy over a `[B, seq_len + 1]` batch.

        Returns the `[B, seq_len]` losses, the weight of each target (1 where
        it counts), whether each prediction was right, and what the routers
        sowed when `routing` asked for it.

        A packed batch carries `segment_ids` for the same rows: the last token
        of a document does not predict the first of the next one, so that
        transition is dropped from the loss and the accuracy, and the model
        reads the per-document `positions` for its rotary angles.
        """
        if tokens.shape[-1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row "
                f"so the targets can be the shifted input, got {tokens.shape[-1]}")
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        # Only a packed batch names these, and only a model that packs takes
        # them: an unpacked run calls the model exactly as it always did.
        packing = {}
        if positions is not None:
            packing["positions"] = positions[:, :-1]
        if segment_ids is not None:
            packing["segment_ids"] = segment_ids[:, :-1]
        hidden = self.model.apply(params, inputs, train=train, rngs=rngs,
                                  method=type(self.model).hidden_states,
                                  mutable=["router"] if routing else False, **packing)
        sown = None
        if routing:
            hidden, sown = hidden
            sown = sown.get("router", {})
        head = self.model.apply(params, params["params"],
                                method=type(self.model).head_weight)
        losses, predicted = chunked_cross_entropy(
            hidden, head, targets, self.head_chunks,
            softcap=self.model.final_logit_softcap,
            precision=self.model.precision)
        weights = (jnp.ones_like(losses) if self.pad_id is None
                   else (targets != self.pad_id).astype(losses.dtype))
        if segment_ids is not None:
            # A target counts only inside a document: the first token of the
            # next packed document, the padding after the last one (segment 0,
            # which the seg==seg comparison alone would keep), and every
            # cross-boundary transition drop out of loss and accuracy alike.
            weights = weights * (
                (segment_ids[:, 1:] == segment_ids[:, :-1])
                & (segment_ids[:, 1:] != 0)).astype(losses.dtype)
        correct = (predicted == targets).astype(losses.dtype)
        return losses, weights, correct, sown

    def loss(self, params, batch, step: Step):
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        segment_ids, positions = _packing(batch)
        balancing = self.balance_rate is not None
        losses, weights, correct, routing = self.token_scores(
            params, tokens, train=True, rngs={"dropout": step.key},
            segment_ids=segment_ids, positions=positions, routing=balancing)
        # A batch that is entirely padding would divide by zero and take the
        # whole run down with a nan.
        counted = jnp.maximum(jnp.sum(weights), 1.0)
        ce = jnp.sum(losses * weights) / counted
        reported = {"ce": ce, "perplexity": jnp.exp(ce),
                    "token_accuracy": jnp.sum(correct * weights) / counted}
        variables = None
        if balancing:
            if "moe" not in params:
                raise ValueError(
                    "balance_rate moves the routers' balancing bias, so the model "
                    "needs sparse layers with expert_bias=True")
            balanced, load = balance(params["moe"], routing, self.balance_rate)
            reported.update(load)
            variables = {"moe": balanced}
        return ce, Aux(reported, variables)

    def evaluate(self, params, batch, step: Step):
        """Teacher-forced token scores, plus the sampled text when asked for.

        Both read the EMA copy, so the perplexity and the samples describe the
        same weights.
        """
        params = params if step.ema is None else step.ema
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        segment_ids, positions = _packing(batch)
        losses, weights = self._scored(params, tokens, segment_ids, positions)
        scores = TokenScores(losses=losses, weights=weights)
        if self.samples is None:
            return scores
        # Deferred: a run that writes no text pulls in no sampler.
        from dew.sampling.text import generate

        generated = generate(
            self.model, params, self._prompt, self.samples.max_new_tokens,
            key=step.key, temperature=self.samples.temperature, top_k=self.samples.top_k)
        decode = self.samples.decode
        return scores, TextSamples(
            tokens=generated,
            prompt=decode(np.asarray(self._prompt)[0].tolist()),
            texts=tuple(decode(row.tolist()) for row in np.asarray(generated)))

    @functools.cached_property
    def _scored(self):
        """The teacher-forced scores, compiled once per objective."""
        def scored(params, tokens, segment_ids, positions):
            losses, weights, _, _ = self.token_scores(
                params, tokens, segment_ids=segment_ids, positions=positions)
            return losses, weights

        return jax.jit(scored)


@metrics("perplexity")
class Perplexity:
    """exp of the cross entropy per counted target over a whole pass.

    Every batch weighs by its own count of counted targets, so a packed or
    padded pass whose batches differ in size is scored per token, and a batch
    with no counted target contributes nothing rather than a zero.
    """

    name = "perplexity"
    reads = TokenScores

    def __call__(self, scores: TokenScores, batch) -> tuple[float, float]:
        weights = jnp.asarray(scores.weights, jnp.float32)
        return (float(jnp.sum(jnp.asarray(scores.losses, jnp.float32) * weights)),
                float(jnp.sum(weights)))

    def reduce(self, values: Sequence[tuple[float, float]]) -> float:
        total = sum(loss for loss, _ in values)
        count = sum(count for _, count in values)
        if count == 0:
            raise ValueError("no counted target in the validation pass")
        return float(np.exp(total / count))
