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

Evaluation returns the teacher-forced per-token scores, which `perplexity`
reduces over a whole pass with the token counts, and, when asked for, the text
the model writes from a fixed prompt.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.artifacts import TextSamples, TokenScores
from dew.data.chat import ROLES_KEY, Role
from dew.inputs import Field, InputSpec
from dew.nn.moe import calculate_load_balance_updates, deepseek_v2_aux_loss
from dew.objectives.base import Aux, EMASpec, Objective, Step, Variables
from dew.objectives.lm.chunked import chunked_cross_entropy
from dew.registry import metrics, objectives

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

    `routing` carries what the routers sowed under the `router` collection,
    the top-k expert indices at the module path the bias lives under. DeepSeek's
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


@objectives("lm")
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
                     segment_ids=None, positions=None, routing: bool = False,
                     depths: bool = False, roles=None, qk_stats: bool = False):
        """Per-token next-token cross entropy over a `[B, seq_len + 1]` batch.

        Returns the `[B, seq_len]` losses, the weight of each target (1 where
        it counts), whether each prediction was right, what the routers
        sowed when `routing` asked for it, the prediction depths' losses
        and weights when `depths` asked for them, and the attention layers'
        per-head logit maxima when `qk_stats` asked for them.

        A packed batch carries `segment_ids` for the same rows. The last token
        of a document does not predict the first of the next one, so that
        transition is dropped from the loss and the accuracy, and the model
        reads the per-document `positions` for its rotary angles. A chat batch
        carries `roles` for the same rows; with `loss_role` set, only the
        targets whose role matches keep their weight.
        """
        if tokens.shape[-1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row "
                f"so the targets can be the shifted input, got {tokens.shape[-1]}")
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        # Only a packed batch names these, and only a model that packs takes
        # them. An unpacked run calls the model without them.
        packing = {}
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
        weights = self._target_weights(targets, segment_ids, losses.dtype)
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
                depth_scores.append((depth_losses, self._target_weights(
                    targets[:, depth:], segment_ids, losses.dtype, depth)))
        return losses, weights, correct, sown, depth_scores, qk

    def per_token_log_probs(self, params, tokens):
        """Next-token log-probabilities, negated cross entropies, over a
        `[B, seq_len + 1]` batch: the `[B, seq_len]` row per token the
        rollout reads back for `old_log_probs`."""
        return -self.token_scores(params, tokens)[0]

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
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        segment_ids, positions = _packing(batch)
        rate = self.balance_rate
        alpha = self.aux_loss_alpha
        losses, weights, correct, routing, depths, qk = self.token_scores(
            params, tokens, train=True, rngs={"dropout": step.key},
            segment_ids=segment_ids, positions=positions,
            routing=rate is not None or alpha is not None,
            depths=self.mtp_weight is not None, roles=self._batch_roles(batch),
            qk_stats=self.qk_stats)
        # A batch that is entirely padding would divide by zero and take the
        # whole run down with a nan.
        counted = jnp.maximum(jnp.sum(weights), 1.0)
        ce = jnp.sum(losses * weights) / counted
        reported = {"ce": ce, "perplexity": jnp.exp(ce),
                    "token_accuracy": jnp.sum(correct * weights) / counted}
        loss = ce
        if self.mtp_weight is not None:
            # Each depth's cross entropy over the same count as the main
            # term, which is the paper's 1/T with the depth's missing tail
            # positions contributing nothing, averaged over the depths.
            mtp = jnp.mean(jnp.stack([
                jnp.sum(depth_losses * depth_weights) / counted
                for depth_losses, depth_weights in depths]))
            reported["mtp_ce"] = mtp
            loss = ce + self.mtp_weight * mtp
        if alpha is not None:
            if not routing:
                raise ValueError(
                    "aux_loss_alpha scales the routers' balance loss, so the "
                    "model needs a mixture")
            balance_loss = jnp.sum(jnp.stack([
                deepseek_v2_aux_loss(scores, indices, alpha, self.seq_aux)
                for scores, indices in _router_scores(routing)]))
            reported["aux_loss"] = balance_loss
            loss = loss + balance_loss
        variables = None
        if rate is not None:
            if "moe" not in params or routing is None:
                raise ValueError(
                    "balance_rate moves the routers' balancing bias, so the model "
                    "needs a mixture with bias=True")
            moe = params["moe"]
            # A depth the step never ran observed no load, so its bias is left alone.
            ran = {name: bias for name, bias in moe.items()
                   if self.mtp_weight is not None or not name.startswith("mtp_")}
            balanced, load = balance(ran, routing, rate)
            reported.update(load)
            variables = {"moe": {**moe, **balanced}}
        stats = None
        if self.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
            stats = qk
        return loss, Aux(reported, variables, stats)

    def evaluate(self, params, batch, step: Step):
        """Teacher-forced token scores, plus the sampled text when asked for.

        Both read the EMA copy, so the perplexity and the samples describe the
        same weights.
        """
        params = params if step.ema is None else step.ema
        tokens = jnp.asarray(batch[TEXT_KEY], jnp.int32)
        segment_ids, positions = _packing(batch)
        losses, weights = self._scored(params, tokens, segment_ids, positions,
                                       self._batch_roles(batch))
        scores = TokenScores(losses=losses, weights=weights)
        if self.samples is None:
            return scores
        # Deferred, so a run that writes no text pulls in no sampler.
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
        def scored(params, tokens, segment_ids, positions, roles):
            losses, weights, _, _, _, _ = self.token_scores(
                params, tokens, segment_ids=segment_ids, positions=positions,
                roles=roles)
            return losses, weights

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
        weights = jnp.asarray(scores.weights, jnp.float32)
        return (float(jnp.sum(jnp.asarray(scores.losses, jnp.float32) * weights)),
                float(jnp.sum(weights)))

    def reduce(self, values: Sequence[tuple[float, float]]) -> float:
        total = sum(loss for loss, _ in values)
        count = sum(count for _, count in values)
        if count == 0:
            raise ValueError("no counted target in the validation pass")
        return float(np.exp(total / count))


@metrics("perplexity")
def perplexity() -> Perplexity:
    return Perplexity()
