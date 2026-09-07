"""The autoregressive language modelling objective.

Next-token prediction over packed token ids. A batch row holds `seq_len + 1`
ids, the model sees all but the last, and the targets are the same row shifted
by one; the causal mask lives in the backbone, so nothing here has to know how
the model keeps the future out of a prediction.

Cross entropy is computed in float32 even when the model runs in bfloat16. A
bf16 logsumexp over a large vocabulary loses enough precision to move the loss
and, through it, the gradient. It is also computed one vocabulary chunk at a
time, because the full `[tokens, vocab]` logits tensor is the largest thing in
a step and every pass over it costs bandwidth; `chunked` holds the arithmetic
and the reason. Padding is excluded only when the run names the pad id.
Packed token files have no padding, and masking out a real id would drop
those tokens from the average.

Evaluation returns teacher-forced per-token scores for streaming perplexity.
The separate preview hook writes text from a fixed prompt once per event.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct
import numpy as np
import optax

from dew.artifacts import TextSamples, TokenScores, agree_process_phase, collective_host
from dew.data.chat import ROLES_KEY, Role
from dew.inputs import Field, InputSpec
from dew.nn.inputs import ModelInputs
from dew.nn.moe import (RouterMoments, global_router_loss, load_balance_update,
                        router_moments, sequence_router_losses)
from dew.objectives.base import Aux, EMASpec, Mean, Objective, Step, Variables, mean_loss
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import metrics, objectives
from dew.sampling.text import Sampling

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
    """Once-per-event text preview configuration.

    Prompts contain token IDs, with equal lengths for multiple prompts.
    This display count does not limit the teacher-forced scoring population.
    """
    prompt: Sequence[int] | Sequence[Sequence[int]]
    max_new_tokens: int
    sampling: Sampling = Sampling()
    decode: Callable[[list[int]], str] = lambda ids: str(ids)


def _shift_rows(values: jax.Array, shifts: jax.Array) -> jax.Array:
    """Shift each row left by its count, wrapping padding to the right."""
    indices = (jnp.arange(values.shape[1])[None, :] + shifts[:, None]) % values.shape[1]
    return jnp.take_along_axis(values, indices, axis=1)


def _packing(batch):
    """A packed batch's `segment_ids` and `positions`, None on a plain one."""
    segment_ids = batch.get("text_segment_ids")
    positions = batch.get("text_positions")
    return (None if segment_ids is None else jnp.asarray(segment_ids, jnp.int32),
            None if positions is None else jnp.asarray(positions, jnp.int32))


def router_counts(moe: Variables, routing: Variables) -> Variables:
    """Selected-slot counts at each active bias path."""
    def count(path, bias):
        sown = routing
        for entry in path[:-1]:
            sown = sown[entry.key]
        (indices,) = sown["indices"]
        return jnp.bincount(indices.ravel(), length=bias.shape[0])
    return jax.tree_util.tree_map_with_path(count, moe)

def _updated_bias(bias: jax.Array, counts: jax.Array, rate: float) -> jax.Array:
    dtype = jnp.result_type(bias.dtype, rate, jnp.float32)
    correction = load_balance_update(counts, jnp.asarray(rate, dtype))
    return (bias.astype(dtype) + correction).astype(bias.dtype)


def balance(moe: Variables, routing: Variables, rate: float
            ) -> tuple[Variables, dict[str, jax.Array]]:
    """Bias replacements and load telemetry from one routed batch."""
    counts = router_counts(moe, routing)
    shares = [count / jnp.sum(count) for count in jax.tree.leaves(counts)]
    balanced = jax.tree.map(lambda bias, count: _updated_bias(bias, count, rate),
                            moe, counts)
    return balanced, {"moe/max_load": jnp.mean(jnp.stack([x.max() for x in shares])),
                      "moe/min_load": jnp.mean(jnp.stack([x.min() for x in shares]))}


def _router_scores(routing: Variables) -> list[tuple[jax.Array, jax.Array]]:
    """Every router's sown (scores, indices), in the tree's key order."""
    found: list[tuple[jax.Array, jax.Array]] = []

    def visit(node) -> None:
        if not isinstance(node, Mapping):
            return
        if "scores" in node and "indices" in node:
            (scores,), (indices,) = node["scores"], node["indices"]
            found.append((scores, indices))
            return
        for key in sorted(node):
            visit(node[key])

    visit(routing)
    return found


def _global_qk_max(qk) -> jax.Array | None:
    """The largest per-head maximum anywhere in the sowed `qk` dict, None
    when no layer sowed one."""
    found: list[jax.Array] = []

    def visit(node) -> None:
        if not isinstance(node, Mapping):
            return
        if "max_logits" in node:
            logged = node["max_logits"]
            found.append(jnp.max(
                logged[0] if isinstance(logged, (tuple, list)) else logged))
            return
        for value in node.values():
            visit(value)

    visit(qk)
    return jnp.max(jnp.stack(found)) if found else None


class Scores(NamedTuple):
    """What `LMObjective.token_scores` computes over a `[B, seq_len + 1]` batch.

    `losses` and `weights` are `[B, seq_len]`: the next-token cross entropy
    and 1 where the target counts. `correct` is 1 where the argmax was the
    target. `routing` is what the routers sowed, `depths` the prediction
    depths' (losses, weights) pairs, `qk` the attention layers' per-head
    logit maxima; each is None or empty unless its flag asked for it.
    """

    losses: jax.Array
    weights: jax.Array
    correct: jax.Array
    routing: Optional[dict]
    depths: list
    qk: Optional[dict]


@struct.dataclass
class LMStatistics:
    """Prediction support and independently normalized router terms."""
    prediction: Mean
    sequence: tuple[Mean, ...]
    global_routers: tuple[RouterMoments, ...]


@objectives("lm")
class LMObjective(Objective[Mean | LMStatistics, Variables]):
    """Shifted cross entropy, teacher-forced scoring and optional text previews."""

    artifact = TokenScores

    def __init__(
        self,
        model,
        seq_len: int,
        *,
        ema_decay: float | None = 0.999,
        pad_id: Optional[int] = None,
        head_chunks: int = 4,
        samples: Optional[Samples] = None,
        pretrained: Optional[Variables] = None,
        balance_rate: Optional[float] = None,
        aux_loss_alpha: Optional[float] = None,
        seq_aux: bool = True,
        loss_role: Role | None = None,
        mtp_weight: Optional[float] = None,
        qk_stats: bool = False,
    ):
        """`head_chunks` is how many vocabulary slices the loss scores a batch
        in; the `[tokens, vocab]` logits are built one slice at a time. Four costs
        2.2% of the step for 1.2 GiB less peak memory at vocabulary 50,304
        on one RTX 4080 (docs/benchmarks.md); the saving grows with the
        vocabulary, and one is the full pass.

        `pretrained` is a variables dict to start from instead of a fresh
        init, as dew.interop.hf_decoders.load_pretrained_decoder returns for a
        Hugging Face checkpoint. The trainer takes its whole initial state
        from `init`, so continued pretraining starts here.

        `balance_rate` moves each sparse layer's routing bias against its
        load by this much every step (DeepSeek's aux-loss-free balancing);
        the model has to keep that bias, `bias=True` on the
        CausalTransformer's mixture. Unset leaves the bias where it is.

        `aux_loss_alpha` adds DeepSeek V2's expert-level balance loss
        (`dew.nn.moe.deepseek_v2_aux_loss`) of every sparse layer, scaled by
        this much, with `seq_aux` choosing its per-sequence form; the
        released V2 configs carry both under these names. Unset adds
        nothing.

        `loss_role` counts only the targets whose `text_roles` entry matches
        it, for SFT on the chat data path (`dew.data.chat.Role`); None counts
        every target the pad and segment weights keep. A batch without the
        `text_roles` column raises.

        `mtp_weight`
        (arXiv 2412.19437, eq. 24). The training loss adds that times the
        mean over the model's prediction depths of each depth's cross
        entropy, so the model needs `num_nextn_predict_layers` above zero.
        Unset leaves the term out and the depths untrained.

        `qk_stats` opens the `qk` collection the attention layers sow their
        per-head logit maxima under, and reports them for the optimizer's
        QK-Clip; the recipe sets it when the optimizer is `muonclip`. Unset
        leaves the collection closed, which costs no extra matmul."""
        if getattr(model, "causal", True) is False:
            raise ValueError("LMObjective requires a causal model for next-token likelihoods")
        self.model = model
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.head_chunks = head_chunks
        self.samples = samples
        self.pretrained = pretrained
        self.balance_rate = balance_rate
        if aux_loss_alpha is not None and aux_loss_alpha <= 0:
            raise ValueError(
                f"aux_loss_alpha scales the balance loss, so it is positive, "
                f"got {aux_loss_alpha}; None adds no balance loss")
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.loss_role = loss_role
        if mtp_weight is not None:
            depths = getattr(model, "num_nextn_predict_layers", 0)
            if depths < 1:
                raise ValueError(
                    "mtp_weight scales the prediction depths' cross entropy, so "
                    "the model needs num_nextn_predict_layers above zero")
            if mtp_weight <= 0:
                raise ValueError(
                    f"mtp_weight is a positive weight on the term, got {mtp_weight}; "
                    "None leaves the term out")
        self.mtp_weight = mtp_weight
        self.qk_stats = qk_stats
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len + 1,)))
        self.ema = None if ema_decay is None else EMASpec(decay=optax.constant_schedule(ema_decay))
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
                     segment_ids=None, positions=None, routing: bool = False,
                     depths: bool = False, roles=None, qk_stats: bool = False):
        """Per-token next-token cross entropy over a `[B, seq_len + 1]` batch.

        Returns `Scores`: the losses, the weight of each target, whether each
        prediction was right, and what `routing`, `depths` and `qk_stats`
        asked for.

        A packed batch carries `segment_ids` for the same rows. The last token
        of a document does not predict the first of the next one, so that
        transition is dropped from the loss and the accuracy, and the model
        reads the per-document `positions` for its rotary angles. A chat batch
        carries `roles` for the same rows; with `loss_role` set, only the
        targets whose role matches keep their weight.
        """
        prepared = tokens if isinstance(tokens, ModelInputs) else ModelInputs(jnp.asarray(tokens, jnp.int32))
        tokens = prepared.tokens
        if tokens.shape[-1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row "
                f"so the targets can be the shifted input, got {tokens.shape[-1]}")
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        # Only a packed batch names these, and only a model that packs takes
        # them. An unpacked run calls the model without them.
        packing = prepared.slice_tokens(stop=-1).kwargs()
        if positions is not None and "positions" in packing:
            raise ValueError("positions must come from either ModelInputs or the packing column")
        if segment_ids is not None and "segment_ids" in packing:
            raise ValueError("segment_ids must come from either ModelInputs or the packing column")
        if positions is not None:
            packing["positions"] = positions[:, :-1]
        if segment_ids is not None:
            packing["segment_ids"] = segment_ids[:, :-1]
        collections = (["router"] if routing else []) + (["qk"] if qk_stats else [])
        hidden = self.model.apply(params, inputs, train=train, rngs=rngs,
                                  method=type(self.model).hidden_states,
                                  mutable=collections or False, **packing)
        sown = None
        qk = None
        if collections:
            hidden, gathered = hidden
            if routing:
                sown = gathered.get("router", {})
            if qk_stats:
                qk = gathered.get("qk")
        head = self.model.apply(params, params["params"],
                                method=type(self.model).head_weight)
        losses, predicted = chunked_cross_entropy(
            hidden, head, targets, self.head_chunks,
            softcap=self.model.final_logit_softcap,
            precision=self.model.precision)
        packed_segments = segment_ids if segment_ids is not None else prepared.token_fields.get("segment_ids")
        weights = self._target_weights(targets, packed_segments, losses.dtype)
        valid = prepared.token_fields.get("attention_mask")
        if valid is not None:
            weights = weights * (valid[:, :-1] & valid[:, 1:]).astype(weights.dtype)
        if roles is not None:
            if roles.shape != tokens.shape:
                raise ValueError(
                    f"text_roles has shape {tuple(roles.shape)} for "
                    f"{tuple(tokens.shape)} ids; the roles align with the input "
                    "tokens, one per token")
            if self.loss_role is not None:
                weights = weights * (roles[:, 1:] == int(self.loss_role))
        correct = (predicted == targets).astype(losses.dtype)
        depth_scores = []
        if depths:
            # Depth d's state at p scores the target d further out, so its
            # targets are the shifted row's from d on, with the same weight
            # rule between the state's document and the target's.
            states = self.model.apply(
                params, hidden, inputs, train=train, rngs=rngs,
                method=type(self.model).mtp_hidden_states,
                mutable=collections or False, **packing)
            if collections:
                # A routed depth balances and counts like a trunk layer.
                states, depth_sown = states
                if routing:
                    sown = {**(sown or {}), **depth_sown.get("router", {})}
                if qk_stats:
                    qk = {**(qk or {}), **depth_sown.get("qk", {})}
            for depth, state in enumerate(states, start=1):
                depth_losses, _ = chunked_cross_entropy(
                    state, head, targets[:, depth:], self.head_chunks,
                    softcap=self.model.final_logit_softcap,
                    precision=self.model.precision)
                depth_weights = self._target_weights(
                    targets[:, depth:], segment_ids, losses.dtype, depth)
                if roles is not None and self.loss_role is not None:
                    depth_weights = depth_weights * (roles[:, depth + 1:] == int(self.loss_role))
                depth_scores.append((depth_losses, depth_weights))
        return Scores(losses, weights, correct, sown, depth_scores, qk)

    def per_token_log_probs(self, params: Variables, tokens: jax.Array | ModelInputs, *,
                            left_padding: jax.Array | None = None) -> jax.Array:
        """Raw policy likelihoods aligned to next-token targets.

        Explicit left-padding counts move real context to position zero before
        scoring. Returned slots whose input is padding are zero and unscored.
        """
        if left_padding is None:
            return -self.token_scores(params, tokens).losses
        prepared = tokens if isinstance(tokens, ModelInputs) else ModelInputs(jnp.asarray(tokens, jnp.int32))
        padding = jnp.asarray(left_padding, jnp.int32)
        if padding.shape != (prepared.tokens.shape[0],):
            raise ValueError("left_padding must have one count per token row")
        aligned = prepared.align_left(padding)
        losses = self.token_scores(params, aligned).losses
        restored = -_shift_rows(losses, -padding)
        return jnp.where(jnp.arange(restored.shape[1])[None, :] >= padding[:, None],
                         restored, 0.0)

    def _target_weights(self, targets, segment_ids, dtype, depth: int = 0):
        """1 where a target counts: not padding, and in a packed batch inside
        the document of the state that predicts it, which sits `depth + 1`
        positions before it."""
        weights = (jnp.ones_like(targets, dtype) if self.pad_id is None
                   else (targets != self.pad_id).astype(dtype))
        if segment_ids is not None:
            # A target counts only inside a document. The first token of the
            # next packed document, the padding after the last one (segment 0,
            # which the seg==seg comparison alone would keep), and every
            # cross-boundary transition drop out of loss and accuracy alike.
            span = depth + 1
            weights = weights * (
                (segment_ids[:, span:] == segment_ids[:, :-span])
                & (segment_ids[:, span:] != 0)).astype(dtype)
        return weights

    def _batch_roles(self, batch):
        """The batch's `text_roles` column, or None on a run that counts
        every target. A `loss_role` run on a batch without the column
        raises, naming it."""
        if self.loss_role is None:
            return None
        if ROLES_KEY not in batch:
            raise ValueError(
                f"loss_role={self.loss_role.name} counts only "
                f"{self.loss_role.name.lower()} targets, but the batch carries "
                f"no {ROLES_KEY} column; train this objective on the chat data path")
        return jnp.asarray(batch[ROLES_KEY])

    def loss(self, params, batch, step: Step):
        tokens = batch[TEXT_KEY]
        segment_ids, positions = _packing(batch)
        rate = self.balance_rate
        alpha = self.aux_loss_alpha
        scores = self.token_scores(
            params, tokens, train=True, rngs={"dropout": step.key},
            segment_ids=segment_ids, positions=positions,
            routing=rate is not None or alpha is not None,
            depths=self.mtp_weight is not None, roles=self._batch_roles(batch),
            qk_stats=self.qk_stats)
        losses, weights, correct, routing, depths, qk = scores
        mass = jax.lax.stop_gradient(jnp.sum(weights))
        prediction = Mean(jnp.sum(losses * weights), mass)
        ce, _ = mean_loss(prediction)
        reported = {"ce": ce, "perplexity": jnp.exp(ce),
                    "token_accuracy": jnp.sum(correct * weights) / jnp.where(mass > 0, mass, 1)}
        if self.mtp_weight is not None:
            # Depths retain the main target denominator and configured depth average.
            mtp_total = jnp.mean(jnp.stack([
                jnp.sum(depth_losses * depth_weights)
                for depth_losses, depth_weights in depths]))
            reported["mtp_ce"], _ = mean_loss(Mean(mtp_total, mass))
            prediction = Mean(prediction.total + self.mtp_weight * mtp_total, mass)
        statistics: Mean | LMStatistics = prediction
        if alpha is not None:
            if not routing:
                raise ValueError("aux_loss_alpha requires a model with a mixture")
            routers = _router_scores(routing)
            sequence = tuple(Mean(jnp.sum(sequence_router_losses(s, i, alpha)),
                                  jnp.asarray(s.shape[0], jnp.int32))
                             for s, i in routers) if self.seq_aux else ()
            global_routers = () if self.seq_aux else tuple(router_moments(s, i) for s, i in routers)
            statistics = LMStatistics(prediction, sequence, global_routers)
            combined, _ = self.reduce_loss(statistics)
            prediction_loss, _ = mean_loss(prediction)
            reported["aux_loss"] = combined - prediction_loss
        effects = None
        if rate is not None:
            if "moe" not in params or routing is None:
                raise ValueError("balance_rate requires a mixture with bias=True")
            ran = {name: bias for name, bias in params["moe"].items()
                   if self.mtp_weight is not None or not name.startswith("mtp_")}
            counts = router_counts(ran, routing)
            shares = [count / jnp.sum(count) for count in jax.tree.leaves(counts)]
            reported.update({"moe/max_load": jnp.mean(jnp.stack([x.max() for x in shares])),
                             "moe/min_load": jnp.mean(jnp.stack([x.min() for x in shares]))})
            effects = counts
        if self.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
        return statistics, Aux(reported, qk_stats=qk, effects=effects)

    def reduce_loss(self, stats: Mean | LMStatistics) -> tuple[jax.Array, jax.Array]:
        if isinstance(stats, Mean):
            return mean_loss(stats)
        value, active = mean_loss(stats.prediction)
        for term in stats.sequence:
            auxiliary, supported = mean_loss(term)
            value, active = value + auxiliary, active | supported
        for term in stats.global_routers:
            if self.aux_loss_alpha is None:
                raise ValueError("global router statistics require aux_loss_alpha")
            value = value + global_router_loss(term, self.aux_loss_alpha)
            active = active | (term.positions > 0)
        return value, active

    def apply_effects(self, variables: Variables, effects: Variables) -> Variables:
        rate = self.balance_rate
        if rate is None:
            raise ValueError("router count effects require balance_rate")
        moe = variables["moe"]
        active = {name: moe[name] for name in effects}
        balanced = jax.tree.map(
            lambda bias, count: _updated_bias(bias, count, rate),
            active, effects)
        return {"moe": {**moe, **balanced}}

    def evaluate(self, params, batch, step: Step):
        """Teacher-forced scores over the complete batch, using EMA when present."""
        params = params if step.ema is None else step.ema
        tokens = batch[TEXT_KEY]
        segment_ids, positions = _packing(batch)
        losses, weights = self._scored(params, tokens, segment_ids, positions,
                                       self._batch_roles(batch))
        return TokenScores(losses=losses, weights=weights)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Sample the configured prompt once, then decode only on process zero."""
        error = None
        settings = prepared = generate_text = prompt = generated = None
        try:
            settings = self.samples
            if settings is not None:
                from dew.sampling.text import generate as generate_text

                params = params if step.ema is None else step.ema
                prepared = (self.model, self._prompt, settings.max_new_tokens, settings.sampling)
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="LM preview setup")
        error = None
        try:
            if prepared is not None:
                model, prompt, max_new_tokens, sampling = prepared
                assert generate_text is not None
                generated = generate_text(model, params, prompt, max_new_tokens,
                                          key=step.key, sampling=sampling).tokens
        except BaseException as failure:
            error = failure
        agree_process_phase(error, phase="LM preview generation")
        generated, prompt = collective_host((generated, prompt), phase="LM preview")
        if settings is None or jax.process_index() != 0:
            return None
        assert generated is not None and prompt is not None
        decode = settings.decode
        return TextSamples(
            tokens=generated,
            prompt=decode(np.asarray(prompt)[0].tolist()),
            texts=tuple(decode(row.tolist()) for row in np.asarray(generated)))

    @functools.cached_property
    def _scored(self):
        """The teacher-forced scores, compiled once per objective."""
        def scored(params, tokens, segment_ids, positions, roles):
            scores = self.token_scores(params, tokens, segment_ids=segment_ids,
                                       positions=positions, roles=roles)
            return scores.losses, scores.weights

        return jax.jit(scored)


class Perplexity:
    """exp of the cross entropy per counted target over a whole pass.

    Every batch weighs by its own count of counted targets, so a packed or
    padded pass whose batches differ in size is scored per token, and a batch
    with no counted target contributes nothing.
    """

    name = "perplexity"
    reads = TokenScores

    def __call__(self, scores: TokenScores, batch) -> tuple[float, float]:
        weights = np.asarray(scores.weights, dtype=np.float64)
        losses = np.asarray(scores.losses, dtype=np.float64)
        return float(np.sum(losses * weights)), float(np.sum(weights))

    def merge(self, accumulated: tuple[float, float],
              contribution: tuple[float, float]) -> tuple[float, float]:
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated: tuple[float, float]) -> float:
        total, count = accumulated
        if count == 0:
            raise ValueError("no counted target in the validation pass")
        return float(np.exp(total / count))


@metrics("perplexity")
def perplexity() -> Perplexity:
    return Perplexity()
