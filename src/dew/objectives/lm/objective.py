"""The autoregressive language-modelling objective.

It trains next-token prediction over packed token ids. A batch row holds
`seq_len + 1` ids; the model sees all but the last, and the targets are the
same row shifted by one. The causal mask is in the backbone, so this module
does not need to know how the model hides the future from a prediction.

Cross entropy is computed in float32 even when the model runs in bfloat16,
because a bf16 logsumexp over a large vocabulary loses enough precision to
change the loss and, through it, the gradient. It is also computed one
vocabulary chunk at a time, because the full `[tokens, vocab]` logits tensor is
the largest array in a step and every pass over it costs memory bandwidth;
`chunked` holds the arithmetic and the reasoning. Padding is excluded only when
the run names the pad id. Packed token files have no padding, and masking a
real id would drop those tokens from the average.

The auxiliary terms use the same batch: PaLM's z-loss on the log partition,
DeepSeek's router balancing and balance loss, the multi-token prediction
depths, and V3.2's lightning indexer, whose KL the attention layers compute and
record when the objective opens their `indexer` collection (`IndexerTraining`
names the phase). `predict` returns the scores, the logits and named layers'
states for a distillation. Evaluation returns teacher-forced per-token scores
for streaming perplexity. The separate preview hook writes text from a fixed
prompt once per event.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

from dew.artifacts import Artifact, TextSamples, TokenScores
from dew.coordination import agreed, collective_host
from dew.data.chat import ROLES_KEY, Role
from dew.inference import TextGeneration
from dew.inference.tasks import Processor, recorded_tokenizer
from dew.inputs import Field, InputSpec
from dew.nn.backbones.causal_transformer import INTERMEDIATES, layer_output, layer_outputs
from dew.nn.inputs import ModelInputs
from dew.nn.mla import INDEXER, INDEXER_COLLECTION
from dew.nn.moe import RouterMoments, global_router_loss, load_balance_update, sequence_router_losses
from dew.nn.protocols import DecoderTraining, Logits, TokenModel
from dew.nn.sharding import LOGITS, constrain
from dew.objectives.base import (
    OMITTED,
    Aux,
    Batch,
    EMASpec,
    Objective,
    Omitted,
    PathFilter,
    Prediction,
    ProgramModule,
    Ratio,
    Shown,
    Step,
    Variables,
    freeze,
    merge,
    merge_totals,
    model_rngs,
    thaw,
)
from dew.objectives.lm.chunked import (
    chunked_tile,
    head_cross_entropy,
    head_table,
    logits_cross_entropy,
    logits_support_log_probs,
    model_logits,
    reads_states,
    support_log_probs,
)
from dew.records import JSON
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from dew.nn.backbones.decoder_stack import DecoderBank

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""


@dataclass(frozen=True)
class IndexerTraining:
    """One stage of training DeepSeek-V3.2's lightning indexer.

    These are the two stages of the continued pre-training (arXiv 2512.02556,
    section 2.1.1):

    - `warmup` trains a fresh indexer alone. The model runs dense attention with
      the indexer scoring next to it (an mla mixer with the indexer's heads and no
      top-k), every other weight is frozen, and the loss is the KL of the indexer's
      softmax from the dense attention distribution over every allowed key.
    - `sparse` trains everything. The model selects its top-k keys (a mixer with
      `index_topk`), the cross entropy trains the main weights, and the KL over the
      selected keys alone trains the indexer, whose inputs are detached so that
      neither loss reaches the other's weights.

    `weight` scales the KL term (MaxText's `indexer_loss_scaling_factor`). The
    reference sets the indexer's pace with its learning rate: 1e-3 for the 1000
    warm-up steps and 7.3e-6 for the sparse stage.
    """

    phase: Literal["warmup", "sparse"]
    weight: float = 1.0

    def __post_init__(self):
        if self.phase not in ("warmup", "sparse"):
            raise ValueError(
                f"the indexer trains in the 'warmup' or the 'sparse' phase, "
                f"not {self.phase!r}")
        if self.weight <= 0:
            raise ValueError(
                f"weight scales the indexer's KL, so it is positive, got "
                f"{self.weight}")


def _check_reads(model: nn.Module) -> None:
    """Refuse a model next-token scoring cannot read.

    A state may see only the tokens up to its own, so a model that says its
    states attend both ways (`TokenModel`) is refused. The loss scores the
    final states through the head (`chunked.reads_states`), or else the
    logits a model gives from its tokens (`Logits`).
    """
    if isinstance(model, TokenModel) and not model.causal:
        raise ValueError("LMObjective requires a causal model for next-token likelihoods")
    if not reads_states(model) and not isinstance(model, Logits):
        raise TypeError(
            f"LMObjective scores a model's logits, and a {type(model).__name__} gives none: no head over "
            f"its final states (HiddenStates with AffineHead or LogitsFromHidden) and no Logits")


def _streamed_depths(model: nn.Module) -> bool:
    """Say whether the prediction depths carry their own residual streams (`DecoderTraining`)."""
    return isinstance(model, DecoderTraining) and model.mtp_hyper_connections is not None


def _check_terms(model: nn.Module, *, aux_loss_alpha: float | None, mtp_weight: float | None,
                 z_loss: float, router_z_loss: float = 0.0) -> None:
    """Refuse a weight the term it scales cannot carry.

    The checks run in the order the constructor takes the arguments. The
    balance loss and the prediction depths' cross entropy are weighted
    positively or left out. The depths have to exist for a weight on them
    to move anything. The log partition's weight is finite and nonnegative.
    """
    if aux_loss_alpha is not None and aux_loss_alpha <= 0:
        raise ValueError(
            f"aux_loss_alpha scales the balance loss, so it is positive, "
            f"got {aux_loss_alpha}; None adds no balance loss")
    if mtp_weight is not None:
        depths = model.num_nextn_predict_layers if isinstance(model, DecoderTraining) else 0
        if depths < 1:
            raise ValueError(
                "mtp_weight scales the prediction depths' cross entropy, so "
                "the model needs num_nextn_predict_layers above zero")
        if mtp_weight <= 0:
            raise ValueError(
                f"mtp_weight is a positive weight on the term, got {mtp_weight}; "
                "None leaves the term out")
    if not (0 <= z_loss < float("inf")):
        raise ValueError(
            f"z_loss weights the squared log partition, so it is finite and "
            f"nonnegative, got {z_loss}; 0 adds nothing")
    if not (0 <= router_z_loss < float("inf")):
        raise ValueError(
            f"router_z_loss weights the routers' squared log partitions, so it is "
            f"finite and nonnegative, got {router_z_loss}; 0 adds nothing")


def _check_indexer(model: nn.Module, indexer: IndexerTraining,
                   lm_terms: Mapping[str, object]) -> None:
    """Refuse an indexer phase the model or the other terms cannot serve.

    Each phase names the attention it trains against: the warm-up learns
    the indexer beside dense attention, so its mixers carry no `index_topk`,
    and the sparse phase trains the model on the selection, so they do.
    The warm-up moves the indexer alone, so any term of the main loss asked
    for beside it would weight leaves the optimizer does not touch.
    """
    mixers = model.indexed_mixers if isinstance(model, DecoderTraining) else ()
    if not mixers:
        raise ValueError(
            "indexer training needs an mla mixer with the indexer's "
            "index_n_heads and index_head_dim")
    sparse = {mixer.sparse for mixer in mixers}
    if indexer.phase == "warmup" and sparse != {False}:
        raise ValueError(
            "the warm-up keeps dense attention while the indexer "
            "learns it, so its mixers name no index_topk")
    if indexer.phase == "sparse" and sparse != {True}:
        raise ValueError(
            "the sparse phase trains the model on its selection, so "
            "its mixers name an index_topk")
    if indexer.phase == "warmup":
        asked = sorted(name for name, value in lm_terms.items() if value is not None)
        if asked:
            raise ValueError(
                f"the warm-up trains the indexer alone, so {', '.join(asked)} "
                "would move nothing")


def _is_indexer(path: tuple[str, ...]) -> bool:
    return INDEXER in path


def _sown_nodes(tree, *markers: str) -> list[Mapping]:
    """Find every node of a sown collection that holds all of `markers`.

    A sown collection nests by module path, so what a layer sowed is the
    branch that names the keys it sowed under. The search stops at such a
    branch rather than descending into the arrays below it, and walks the
    children it does descend into in sorted key order.
    """
    found: list[Mapping] = []

    def visit(node) -> None:
        if not isinstance(node, Mapping):
            return
        if all(marker in node for marker in markers):
            found.append(node)
            return
        for key in sorted(node):
            visit(node[key])

    visit(tree)
    return found


def _indexer_kls(sown: Variables) -> list[jax.Array]:
    """Collect every attention layer's sown per-query indexer KL.

    Each is `[B, S]`, and they come back in the tree's key order.
    """
    kls: list[jax.Array] = []
    for node in _sown_nodes(sown, "kl"):
        (kl,) = node["kl"]
        kls.append(kl)
    return kls


def _leaf_paths(tree: Variables) -> set[tuple[str, ...]]:
    """Return the dict-key path of every leaf of a variables subtree."""
    return {tuple(entry.key for entry in path)
            for path, _ in jax.tree_util.tree_leaves_with_path(tree)}


def _prepared(tokens: ModelInputs | jax.Array, segment_ids: jax.Array | None = None,
              positions: jax.Array | None = None) -> ModelInputs:
    """The rows as `ModelInputs`, bare ids cast to int32, their packing among the token fields.

    A packed batch names each row's `segment_ids` and `positions` either in
    columns beside the ids or in its `ModelInputs`; a field named both ways
    is refused. Everything after reads the packing off what this returns.
    """
    prepared = tokens if isinstance(tokens, ModelInputs) else ModelInputs(jnp.asarray(tokens, jnp.int32))
    columns = {name: column for name, column in (("positions", positions), ("segment_ids", segment_ids))
               if column is not None}
    for name in columns:
        if name in prepared.token_fields:
            raise ValueError(f"{name} must come from either ModelInputs or the packing column")
    return (
        dataclasses.replace(prepared, token_fields={**prepared.token_fields, **columns})
        if columns
        else prepared
    )


def prompt_batch(prompt) -> jax.Array:
    """Build `[B, P]` int32 ids from one prompt, or several of the same length."""
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
    """Settings for the text preview drawn once per event.

    Prompts hold token IDs; several prompts must have equal lengths. This display
    count does not limit the teacher-forced scoring population. A budget of 0
    draws no preview but keeps `sampling` as the policy the run records and
    publishes.
    """
    prompt: Sequence[int] | Sequence[Sequence[int]]
    max_new_tokens: int
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    decode: Callable[[list[int]], str] = lambda ids: str(ids)


def _shift_rows(values: jax.Array, shifts: jax.Array) -> jax.Array:
    """Shift each row left by its count, wrapping padding to the right."""
    indices = (jnp.arange(values.shape[1])[None, :] + shifts[:, None]) % values.shape[1]
    return jnp.take_along_axis(values, indices, axis=1)


def _unpadded(values: jax.Array, padding: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Undo a left alignment on scored rows, and mark the slots that scored.

    `values` were scored over rows whose real tokens had been moved down to
    position zero by `padding` places. They come back to the columns they
    were scored for, beside a mask that is true where the row held a real
    token rather than padding.
    """
    restored = _shift_rows(values, -padding)
    valid = jnp.arange(restored.shape[1])[None, :] >= padding[:, None]
    return restored, valid


def batch_text(batch) -> ModelInputs:
    """The batch's `text` rows as `_prepared` gives them, a packed batch's
    `text_segment_ids` and `text_positions` columns folded in."""
    segment_ids = batch.get("text_segment_ids")
    positions = batch.get("text_positions")
    return _prepared(batch[TEXT_KEY],
                     None if segment_ids is None else jnp.asarray(segment_ids, jnp.int32),
                     None if positions is None else jnp.asarray(positions, jnp.int32))


def text_preview[S](phase: str, setup: Callable[[], S | None],
                     generate: Callable[[S], tuple[jax.Array, jax.Array | None]],
                     decode: Callable[[list[int]], str] | None) -> TextSamples | None:
    """A text preview, on every process: agree on `setup`, generate the rows
    and the prompt they continue (None for none) from what it returns, gather
    both to process zero's host (`collective_host`, held by the first), and
    decode them into `TextSamples` there. The other processes return None,
    as every process does when `setup` returns None, which draws no preview.
    Without `decode`, the artifact holds the ids alone."""
    prepared = agreed(f"{phase} setup", setup)
    drawn = agreed(f"{phase} generation", lambda: None if prepared is None else generate(prepared))
    drawn = collective_host(drawn, phase=phase, held_by="first")
    if drawn is None:
        return None
    tokens, prompt = drawn
    if decode is None:
        return TextSamples(tokens=tokens)
    return TextSamples(tokens=tokens, prompt="" if prompt is None else decode(np.asarray(prompt)[0].tolist()),
                       texts=tuple(decode(row.tolist()) for row in np.asarray(tokens)))


# The `moe` leaf a balanced router keeps. DeepSeek V4's hash router keeps
# the frozen token-to-expert table it selects by in the same collection
# (dew.nn.moe.Router), and no load-balance update moves that.
ROUTER_BIAS = "e_score_correction_bias"


def _balanced_biases(moe: Variables | jax.Array) -> Variables | jax.Array:
    """Select the balancing biases of a `moe` collection, at their own paths.

    The recursion hands itself the branches it walks, so the argument is a
    collection or one of its leaves.

    Router state that is not a bias, and any branch left holding none of
    them, is dropped, so what comes back is one leaf per router the load
    balancer moves.
    """
    if not isinstance(moe, Mapping):
        return moe
    kept = {}
    for key, value in moe.items():
        if not isinstance(value, Mapping):
            if key == ROUTER_BIAS:
                kept[key] = value
            continue
        branch = _balanced_biases(value)
        if branch:
            kept[key] = branch
    return kept


def router_counts(moe: Variables, routing: Variables) -> Variables:
    """Count the selected slots at each active bias path."""
    def count(path, bias):
        sown = routing
        for entry in path[:-1]:
            sown = sown[entry.key]
        (indices,) = sown["indices"]
        return jnp.bincount(indices.ravel(), length=bias.shape[0])
    return jax.tree_util.tree_map_with_path(count, _balanced_biases(moe))


def _updated_bias(bias: jax.Array, counts: jax.Array, rate: float) -> jax.Array:
    dtype = jnp.result_type(bias.dtype, rate, jnp.float32)
    correction = load_balance_update(counts, jnp.asarray(rate, dtype))
    return (bias.astype(dtype) + correction).astype(bias.dtype)


def router_z_terms(routing: Variables, weight: float, batch: Batch) -> tuple[Ratio, ...]:
    """Return ST-MoE's router z-loss, one `Ratio` per router.

    This is eq. 5 of ST-MoE (arXiv 2202.08906): `weight` times the squared log
    partition of the gate logits, summed over the positions the router saw and
    divided by their count. The count adds across micro-batches, so a step's term
    is the mean over its routed positions, and the routers' terms add up, as
    lm-engine's per-layer `(logsumexp(logits) ** 2).mean()` does. The positions
    are `batch`'s rows' (`Objective.row_mean`).
    """
    terms = []
    # lm-engine: moe/module.py at 45b6b57b, before its 0.1 and router_aux_loss_coef.
    for node in _sown_nodes(routing, "log_z"):
        (log_z,) = node["log_z"]
        work = Objective.row_mean(jnp.square(log_z.astype(jnp.promote_types(log_z.dtype, jnp.float32))),
                                  batch)
        terms.append(Ratio(weight * work.total, work.mass))
    return tuple(terms)


def router_moments(scores: jax.Array, indices: jax.Array, batch: Batch) -> RouterMoments:
    """Return one router's routed-position statistics over `batch`'s rows (`Objective.row_mean`).

    `scores` is `[rows, positions, experts]` and `indices` the experts each
    position chose, `[rows, positions, top_k]`.
    """
    experts = scores.shape[-1]
    dtype = jnp.promote_types(scores.dtype, jnp.float32)
    # These counts enter a floating loss, unlike the exact integer bias effects.
    summed = Objective.row_mean(scores.astype(dtype), batch, axis=(0, 1))
    chosen = jax.vmap(lambda row: jnp.bincount(row.ravel(), length=experts))(indices)
    counts = Objective.row_mean(chosen, batch, axis=0).total.astype(dtype)
    return RouterMoments(summed.total, counts, summed.mass, indices.shape[-1])


def _router_scores(routing: Variables) -> list[tuple[jax.Array, jax.Array]]:
    """Collect every router's sown (scores, indices), in the tree's key order."""
    found: list[tuple[jax.Array, jax.Array]] = []
    for node in _sown_nodes(routing, "scores", "indices"):
        (scores,), (indices,) = node["scores"], node["indices"]
        found.append((scores, indices))
    return found


def _global_qk_max(qk) -> jax.Array | None:
    """Return the largest per-head maximum anywhere in the sowed `qk` dict.

    None when no layer sowed one.
    """
    found: list[jax.Array] = []
    for node in _sown_nodes(qk, "max_logits"):
        logged = node["max_logits"]
        found.append(jnp.max(logged[0] if isinstance(logged, (tuple, list)) else logged))
    return jnp.max(jnp.stack(found)) if found else None


class Scores(NamedTuple):
    """The per-token results `LMObjective.token_scores` computes over a `[B, seq_len + 1]` batch.

    `losses`, `weights` and `log_z` are `[B, seq_len]`: the next-token cross
    entropy, 1 where the target counts, and the log partition of each prediction's
    distribution (what PaLM's z-loss squares). `correct` is 1 where the argmax was
    the target, and None unless the objective reports `token_accuracy`. `hidden` is
    the `[B, seq_len, D]` final states the head scored, or the `[B, seq_len, vocab]`
    logits of a model that gives them only from its tokens (`chunked.reads_states`),
    and `layers` the states of
    the layers `token_scores` was asked for, in that order. `routing` is what the
    routers recorded, `depths` the prediction depths' (losses, weights) pairs, `qk`
    the attention layers' per-head logit maxima, and `indexer` their per-query
    indexer KL; each is None or empty unless its flag asked for it.
    """

    losses: jax.Array
    weights: jax.Array
    log_z: jax.Array
    correct: jax.Array | None
    hidden: jax.Array
    layers: tuple[jax.Array, ...]
    routing: dict | None
    depths: list
    qk: dict | None
    indexer: dict | None


@struct.dataclass
class LMStatistics:
    """The prediction's statistics, next to the separately normalized router terms."""
    prediction: Ratio
    sequence: tuple[Ratio, ...]
    global_routers: tuple[RouterMoments, ...]
    router_z: tuple[Ratio, ...] = ()


def _phase_filter(model: nn.Module, indexer: IndexerTraining | None,
                  terms: Mapping[str, object]) -> PathFilter | None:
    """The leaves an indexer phase trains, once it is checked against the
    main loss's `terms`: the warm-up trains the indexer alone and refuses
    every term. None trains what the starting variables leave in `params`."""
    if indexer is None:
        return None
    _check_indexer(model, indexer, terms)
    return _is_indexer if indexer.phase == "warmup" else None


_DEFAULT_SAMPLING = Sampling()


class LMObjective(Objective[Ratio | LMStatistics, Variables]):
    """Trains a next-token model with shifted cross entropy, teacher-forced scoring and optional previews.

    `LMObjective(model, seq_len, ...)` scores `seq_len`-token rows. Every
    auxiliary term below is off until its argument is set; the other options
    trade memory for time.

    `head_tile` is the head's backward tile (`chunked_cross_entropy`'s
    `tile`), or 'whole' or 'tiled'. None, the default, is 'whole' on an
    objective that keeps the whole logits (`keeps_whole_logits`) and
    the generation's tile on one that does not. Whole logits are the
    fastest head where they fit: on one A100, a Qwen3-0.6B step at 4 x 1024
    tokens took 142 ms against 163 ms tiled, at the cost of 5.3 GiB more
    peak memory. A trainer whose compiled step does not fit the devices
    moves it to the generation's tile (`chunked.chunked_tile`) before it
    recomputes any block. A pass with no backward (evaluation, scoring)
    runs the tiled forward either way, so it never holds the logits
    whole.

    `head_chunks` is the number of vocabulary slices a tiled head scores in.
    Four slices cost 2.2% of the step and save 1.2 GiB of peak memory at
    vocabulary 50,304 on one RTX 4080 (docs/benchmarks.md), and one slice is
    the full pass. It also slices the forward pass of an evaluation or a
    scoring pass.

    `ema_decay` keeps an exponential moving average of the trained leaves
    at that decay, and evaluation and previews then read the average. None,
    the default, keeps no average, so there is no second copy of the weights
    and validation scores the weights that trained.

    The trainer takes its whole initial state from `init`, which starts
    from `variables` (described below).

    `balance_rate` moves each sparse layer's routing bias against its load
    by this much every step, which is DeepSeek's aux-loss-free balancing. The
    model has to have that bias (`bias=True` on the CausalTransformer's
    mixture). Unset, the bias does not change.

    `aux_loss_alpha` scales DeepSeek V2's expert-level balance loss
    (`dew.nn.moe.global_router_loss`), summed over every sparse
    layer, and `seq_aux` chooses its per-sequence form
    (`dew.nn.moe.sequence_router_losses`). The released V2 configs set both
    under these names. Unset, it adds nothing.

    `loss_role` counts only the targets whose `text_roles` entry matches
    it, for SFT on the chat data path (`dew.data.chat.Role`). None
    counts every target the pad and segment weights keep. A batch
    without the `text_roles` column raises.

    `mtp_weight` scales DeepSeek V3's multi-token prediction loss
    (arXiv 2412.19437, eq. 24). The loss adds this weight times the
    mean cross entropy over the model's prediction depths, so the
    model needs `num_nextn_predict_layers` above zero. Unset leaves
    the term out and the depths untrained.

    `qk_stats` opens the `qk` collection where the attention layers record
    their per-head logit maxima, and reports them for the optimizer's
    QK-Clip. The recipe sets it when the optimizer is `muonclip`. Unset, the
    collection stays closed and costs no extra matmul.

    `indexer` trains DeepSeek-V3.2's lightning indexer, one
    `IndexerTraining` phase at a time, on a model whose mla mixer has the
    indexer. The warm-up phase keeps only the indexer in
    the `params` collection and the rest of the model under `frozen`,
    which is what `init` returns and a checkpoint stores. A
    starting tree for that phase may omit the indexer's weights, as
    a dense checkpoint does, and the fresh init fills them. The sparse
    phase reads a whole tree, either layout. The warm-up trains nothing
    but the indexer, so the terms of the main loss (`balance_rate`,
    `aux_loss_alpha`, `mtp_weight`, `loss_role`, `z_loss`) are refused there.

    `z_loss` adds PaLM's auxiliary to the cross entropy: this
    coefficient times the squared log partition of every counted
    prediction (MaxText's `z_loss_multiplier`; PaLM used 1e-4). It
    keeps the logits from drifting away from normalised log
    probabilities. Zero adds nothing.

    `router_z_loss` is the routers' own z-loss (ST-MoE, arXiv
    2202.08906): this coefficient times the squared log partition of
    every router's gate logits, averaged over the positions each router saw
    in the step and summed over the routers (`router_z_terms`). It is added
    next to the balance loss. lm-engine's MoE adds 0.1 of it to its switch
    loss before `router_aux_loss_coef`, which corresponds to
    `router_z_loss = 0.1 * aux_loss_alpha` here. Zero adds nothing.

    `model` may be a loaded source (`LMObjective(qwen, seq_len=512)`), and
    `variables` and `processor` override its own (`Objective.bind_model`).

    `token_accuracy` reports the argmax accuracy; False skips the pass
    over every logit it costs (0.77 ms of the head's 8.0 on a TPU v6e).
    """

    artifact = TokenScores
    saved_task = TextGeneration
    shown: Mapping[str, Shown] = {"ce": Shown(better="lower"), "perplexity": Shown(better="lower"),
             "token_accuracy": Shown(better="higher", percent=True)}

    # tests/test_packed_grpo.py covers the tiled default.
    keeps_whole_logits: ClassVar[bool] = True
    """Whether the head's default (`head_tile` None) keeps the whole fp32 logits for the backward pass.

    A plain LM step does, and the trainer's fit ladder tiles it when the step does
    not fit (`recompute_more`). An objective whose devices also hold rollouts or a
    frozen reference, as GRPO's and DPO's do, keeps the tiled head as its default,
    so the loss's temporaries stay one tile in size whatever the vocabulary.
    """

    def __init__(
        self,
        model,
        seq_len: int,
        *,
        ema_decay: float | None = None,
        pad_id: int | None = None,
        head_chunks: int = 4,
        head_tile: tuple[int, int] | Literal['whole', 'tiled'] | None = None,
        samples: Samples | None = None,
        variables: Variables | None | Omitted = OMITTED,
        balance_rate: float | None = None,
        aux_loss_alpha: float | None = None,
        seq_aux: bool = True,
        loss_role: Role | None = None,
        mtp_weight: float | None = None,
        z_loss: float = 0.0,
        router_z_loss: float = 0.0,
        qk_stats: bool = False,
        indexer: IndexerTraining | None = None,
        token_accuracy: bool = True,
        processor: Processor | None | Omitted = OMITTED,
    ):
        """Build the objective; the class docstring describes each argument."""
        model = self.bind_model(model, variables=variables, processor=processor)
        _check_reads(model)
        self.model = model
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.head_chunks = head_chunks
        self.head_tile = _head_tile(head_tile, self.keeps_whole_logits)
        self.samples = samples
        self.balance_rate = balance_rate
        _check_terms(model, aux_loss_alpha=aux_loss_alpha, mtp_weight=mtp_weight, z_loss=z_loss,
                     router_z_loss=router_z_loss)
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.loss_role = loss_role
        self.mtp_weight = mtp_weight
        self.z_loss = z_loss
        self.router_z_loss = router_z_loss
        self.qk_stats = qk_stats
        self.token_accuracy = token_accuracy
        self.indexer = indexer
        self.phase = _phase_filter(model, indexer, {
            "balance_rate": balance_rate, "aux_loss_alpha": aux_loss_alpha,
            "mtp_weight": mtp_weight, "loss_role": loss_role, "z_loss": z_loss or None,
            "router_z_loss": router_z_loss or None})
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (seq_len + 1,)))
        self.ema = EMASpec.constant(ema_decay)
        if samples is not None and samples.max_new_tokens > 0:
            self._prompt = prompt_batch(samples.prompt)

    @property
    def _warmup(self) -> bool:
        return self.indexer is not None and self.indexer.phase == "warmup"

    @property
    def bank_sites(self) -> tuple[DecoderBank, ...]:
        return self.model.bank_sites

    def program_key(self) -> tuple[ProgramModule, ...]:
        return (ProgramModule(self.model, self.head_tile, trained=True),)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Split the tree as the indexer phase trains it.

        An indexer phase decides its own split, the warm-up the indexer alone
        and the sparse phase the whole tree, so it starts from the tree
        whole. The warm-up may start from a dense checkpoint that has no
        indexer yet, and a fresh init supplies exactly those weights.
        """
        if self.indexer is not None:
            tree = thaw(tree)
            drawn = jax.eval_shape(self.fresh_variables, key, None)
            missing = _leaf_paths(drawn["params"]) - _leaf_paths(tree["params"])
            if self._warmup and missing:
                outside = sorted("/".join(path) for path in missing if not _is_indexer(path))
                if outside:
                    raise ValueError(
                        "the warm-up initialises the indexer and nothing else, but the "
                        f"starting tree lacks {outside[:3]}{'...' if len(outside) > 3 else ''}")
                tree = merge(self.fresh_variables(key, None), tree)
        return tree if self.phase is None else freeze(tree, self.phase)


    def policy(self, params: Variables, sampling: Sampling = _DEFAULT_SAMPLING) -> TextGeneration:
        """Return the model over this training tree as a generation task.

        A rollout uses one snapshot of the policy and draws every completion from it;
        the result records the actual and raw-policy likelihoods that the objective's
        ratio needs.
        """
        return TextGeneration(self.model, params, sampling=sampling)

    def task_record(self) -> Mapping[str, JSON]:
        """The decoder's row length, preview budget, sampling policy and tokenizer."""
        from dew.registry import to_record
        samples = self.samples
        return {
            'seq_len': self.seq_len,
            'max_new_tokens': 0 if samples is None else samples.max_new_tokens,
            'sampling': to_record(Sampling() if samples is None else samples.sampling, Sampling),
            'tokenizer': recorded_tokenizer(self.processor),
        }

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> TextGeneration:
        """Return the decoder over `variables` as a generation task.

        It samples, and uses the same token budget, as this objective's previews do.
        """
        samples = self.samples
        return TextGeneration(self.model, variables, self.processor if processor is OMITTED else processor,
                              sampling=Sampling() if samples is None else samples.sampling,
                              max_new_tokens=None if samples is None or samples.max_new_tokens <= 0
                              else samples.max_new_tokens)

    def token_scores(self, params, tokens, train: bool = False, rngs=None,
                     segment_ids=None, positions=None, routing: bool = False,
                     depths: bool = False, roles=None, qk_stats: bool = False,
                     indexer: bool = False, layers: Sequence[int] = (),
                     routes: tuple[jax.Array, jax.Array | None] | None = None,
                     predict: bool | None = None):
        """Score per-token next-token cross entropy over a `[B, seq_len + 1]` batch.

        Returns `Scores`: the losses, the weight of each target, whether each
        prediction was right, the states behind them, and what `routing`,
        `depths`, `qk_stats`, `indexer` and `layers` asked for. `layers`
        names the model's layers whose output states to keep, `layers_N` in
        its tree, as a feature distillation reads them.

        A packed batch carries `segment_ids` for the same rows. The last token
        of a document does not predict the first of the next one, so that
        transition is dropped from the loss and the accuracy, and the model
        reads the per-document `positions` for its rotary angles. A chat batch
        carries `roles` for the same rows; with `loss_role` set, only the
        targets whose role matches keep their weight.

        `routes` replays a rollout engine's expert choices: `[B, seq_len + 1,
        layers, top_k]` ids aligned with `tokens`, and `[B, seq_len + 1]`
        booleans marking the ids the record covers (None for all). Every
        router selects those experts instead of its own top-k and still
        weights them from its scores (`dew.nn.moe.Routes`); the stack slices
        the record by layer however it runs.
        """
        prepared = _prepared(tokens, segment_ids, positions)
        inputs, targets = self._rows(prepared.tokens)
        packing = prepared.slice_tokens(stop=-1).kwargs()
        params = thaw(params)
        replay = {}
        if routes is not None:
            # The last id is never forwarded, so its row drops with it.
            routed, covered = routes
            replay["routed_experts"] = jnp.asarray(routed)[:, :-1]
            if covered is not None:
                replay["routed"] = jnp.asarray(covered, bool)[:, :-1]
        collections = ((["router"] if routing else []) + (["qk"] if qk_stats else [])
                       + ([INDEXER_COLLECTION] if indexer else []))
        stream_depth = depths and _streamed_depths(self.model)
        opened = [*collections, 'prediction_inputs'] if stream_depth else collections
        hidden, gathered = self._hidden_states(params, inputs, train, rngs, opened,
                                               {**packing, **replay}, layers)
        sown = gathered.get("router", {}) if routing else None
        qk = gathered.get("qk") if qk_stats else None
        kls = gathered.get(INDEXER_COLLECTION) if indexer else None
        kept = tuple(layer_output(gathered[INTERMEDIATES], index) for index in layers)
        predict = self.token_accuracy if predict is None else predict
        losses, predicted, log_z = (
            head_cross_entropy(self.model, params, hidden, targets, self.head_chunks, tile=self.head_tile,
                               predict=predict) if reads_states(self.model)
            else logits_cross_entropy(hidden, targets, predict=predict))
        weights = self._row_weights(prepared, targets, roles, losses.dtype)
        correct = None if predicted is None else (predicted == targets).astype(losses.dtype)
        depth_scores = []
        if depths:
            # Depth d's state at p scores the target d further out, so its
            # targets are the shifted row's from d on.
            states = self.model.apply(
                params, gathered['prediction_inputs']['states'] if stream_depth else hidden,
                inputs, train=train, rngs=rngs,
                method="mtp_hidden_states",
                mutable=collections or False, **packing)
            if collections:
                # A routed depth balances, counts and indexes like a trunk layer.
                states, depth_sown = states
                if routing:
                    sown = {**(sown or {}), **depth_sown.get("router", {})}
                if qk_stats:
                    qk = {**(qk or {}), **depth_sown.get("qk", {})}
                if indexer:
                    kls = {**(kls or {}), **depth_sown.get(INDEXER_COLLECTION, {})}
            for depth, state in enumerate(states, start=1):
                depth_losses, _, _ = head_cross_entropy(
                    self.model, params, state, targets[:, depth:], self.head_chunks,
                    tile=self.head_tile, predict=False)
                depth_scores.append((depth_losses, self._depth_weights(
                    prepared, targets, roles, losses.dtype, depth)))
        return Scores(losses, weights, log_z, correct, hidden, kept, sown, depth_scores, qk, kls)

    def _row_weights(self, prepared, targets, roles, dtype):
        """Weight the targets the row itself scores.

        A supplied validity mask drops the transitions across padding, and
        `loss_role` keeps one role's targets alone.
        """
        weights = self._target_weights(prepared, targets, dtype)
        valid = prepared.token_fields.get("attention_mask")
        if valid is not None:
            weights = weights * (valid[:, :-1] & valid[:, 1:]).astype(weights.dtype)
        if roles is not None:
            if roles.shape != prepared.tokens.shape:
                raise ValueError(
                    f"text_roles has shape {tuple(roles.shape)} for "
                    f"{tuple(prepared.tokens.shape)} ids; the roles align with the input "
                    "tokens, one per token")
            if self.loss_role is not None:
                weights = weights * (roles[:, 1:] == int(self.loss_role))
        return weights

    def _depth_weights(self, prepared, targets, roles, dtype, depth: int):
        """Weight the targets one prediction depth scores.

        Depth d's state at p scores the target d further out, so the same
        document rule applies between the state's document and the
        target's. A padded row admits the target only when every position
        from the state to it is real.
        """
        weights = self._target_weights(prepared, targets[:, depth:], dtype, depth)
        valid = prepared.token_fields.get("attention_mask")
        if valid is not None:
            span = targets.shape[1] - depth
            admitted = jnp.ones((targets.shape[0], span), dtype=bool)
            for offset in range(depth + 2):
                admitted = admitted & valid[:, offset:offset + span]
            weights = weights * admitted.astype(weights.dtype)
        if roles is not None and self.loss_role is not None:
            weights = weights * (roles[:, depth + 1:] == int(self.loss_role))
        return weights

    def tile_head(self, tile: tuple[int, int] | None = None) -> str | None:
        """Move a head that keeps its whole logits to `tile`, and report what it moved to.

        `tile` defaults to the generation's tile (`chunked.chunked_tile`).
        """
        if self.head_tile is not None:
            return None
        self.head_tile = chunked_tile() if tile is None else tile
        return f"the head tiled {self.head_tile}"

    def _hidden_states(self, params, inputs, train, rngs, collections: list[str],
                       packing: dict[str, jax.Array | Mapping[str, jax.Array]],
                       layers: Sequence[int] = ()):
        """Run the model over `inputs` and return its final states, or the
        logits of a model that gives them only from its tokens (`reads_states`).

        Beside them come whatever the open `collections` gathered, empty
        when none was opened, and under `intermediates` the outputs of
        every layer when `layers` asks for any.
        """
        opened = [*collections, INTERMEDIATES] if layers else collections
        hidden = self.model.apply(params, inputs, train=train, rngs=rngs,
                                  method="hidden_states" if reads_states(self.model) else "logits",
                                  mutable=opened or False,
                                  capture_intermediates=layer_outputs if layers else False,
                                  **packing)
        if not opened:
            return hidden, {}
        return hidden

    def per_token_log_probs(self, params: Variables, tokens: jax.Array | ModelInputs, *,
                            left_padding: jax.Array | None = None) -> jax.Array:
        """Score raw policy likelihoods aligned to next-token targets.

        Explicit left-padding counts move real context to position zero before
        scoring. Returned slots whose input is padding are zero and unscored.
        """
        if left_padding is None:
            return -self.token_scores(params, tokens).losses
        prepared = _prepared(tokens)
        padding = jnp.asarray(left_padding, jnp.int32)
        aligned = prepared.align_left(padding)
        losses = self.token_scores(params, aligned).losses
        restored, valid = _unpadded(losses, padding)
        return jnp.where(valid, -restored, 0.0)

    def sampled_log_probs(self, params: Variables, scores: Scores, tokens: jax.Array,
                          support: tuple[jax.Array, jax.Array] | None = None,
                          temperature: float = 1.0) -> jax.Array:
        """Return each next-token target's likelihood as the sampler that drew it saw it.

        `scores` is `token_scores` over `tokens`, `[B, S + 1]`, and the result is
        `[B, S]`. At unit temperature without `support`, that is the raw policy,
        `-scores.losses`. `temperature` divides the capped logits (`head_logits`).
        `support` is the per-row ragged `(ids, columns)` pair that `sessions.pack`
        builds, each `[B, C]`, where `columns` is the column in `tokens` of the id
        each kept id belongs to; a target with entries is renormalized over them
        (`support_log_probs`, or `logits_support_log_probs` over the whole
        logits of a head no matrix gives, as an adapter's on `lm_head`).
        """
        log_probs = -scores.losses
        if temperature == 1.0 and support is None:
            return log_probs
        params = thaw(params)
        targets = tokens[:, 1:]
        if temperature != 1.0:
            losses, _, _ = (
                head_cross_entropy(self.model, params, scores.hidden, targets, self.head_chunks,
                                   tile=self.head_tile, predict=False, temperature=temperature)
                if reads_states(self.model)
                else logits_cross_entropy(scores.hidden, targets, predict=False, temperature=temperature))
            log_probs = -losses
        if support is not None:
            ids, columns = (jnp.asarray(value, jnp.int32) for value in support)
            # The id at column c is the target the state at c - 1 predicts.
            owners = jnp.where(columns > 0, columns - 1, -1)
            head = head_table(self.model, params)
            if head is not None:
                filtered, present = support_log_probs(
                    scores.hidden, head.matrix, targets, ids, owners, temperature=temperature,
                    vocab_major=head.vocab_major, softcap=head.softcap, precision=head.precision,
                    bias=head.bias)
            else:
                logits = (model_logits(self.model, params, scores.hidden) if reads_states(self.model)
                          else scores.hidden)
                filtered, present = logits_support_log_probs(logits, targets, ids, owners,
                                                             temperature=temperature)
            log_probs = jnp.where(present, filtered, log_probs)
        return log_probs

    def _target_weights(self, prepared, targets, dtype, depth: int = 0):
        """Mark with 1 every target that counts.

        A target counts when it is not padding and, in a packed batch,
        when it sits inside the document of the state that predicts it,
        which is `depth + 1` positions before it.
        """
        weights = (jnp.ones_like(targets, dtype) if self.pad_id is None
                   else (targets != self.pad_id).astype(dtype))
        segment_ids = prepared.token_fields.get("segment_ids")
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
        """Read the batch's `text_roles` column, or None on a run that counts
        every target.

        A `loss_role` run on a batch without the column raises, naming it.
        """
        if self.loss_role is None:
            return None
        if ROLES_KEY not in batch:
            raise ValueError(
                f"loss_role={self.loss_role.name} counts only "
                f"{self.loss_role.name.lower()} targets, but the batch carries "
                f"no {ROLES_KEY} column; train this objective on the chat data path")
        return jnp.asarray(batch[ROLES_KEY])

    def _query_weights(self, prepared: ModelInputs, dtype):
        """Mark with 1 every query whose indexer KL counts.

        A query counts when its token is real, neither the pad id nor a slot
        the `ModelInputs` attention mask leaves out, and, in a packed batch,
        when it sits inside a document.
        """
        inputs = prepared.tokens[:, :-1]
        weights = (jnp.ones_like(inputs, dtype) if self.pad_id is None
                   else (inputs != self.pad_id).astype(dtype))
        documents = prepared.token_fields.get("segment_ids")
        if documents is not None:
            weights = weights * (documents[:, :-1] != 0).astype(dtype)
        valid = prepared.token_fields.get("attention_mask")
        if valid is not None:
            weights = weights * valid[:, :-1].astype(dtype)
        return weights

    def _indexer_term(self, sown, prepared: ModelInputs, batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Sum the batch's indexer KL over the layers that sowed one, with
        the number of queries it counted.

        A prediction depth's layer sows one query fewer per depth; its
        leading queries are the rows' leading positions, so the weights
        are cut to match."""
        kls = _indexer_kls(sown)
        if not kls:
            raise ValueError(
                "no attention layer sowed an indexer KL, though the model's mla "
                "mixer carries the indexer")
        weights = self._query_weights(prepared, jnp.float32)
        total = jnp.sum(jnp.stack([
            self.row_mean(kl.astype(jnp.float32) * weights[:, :kl.shape[1]], batch).total for kl in kls]))
        mass = jax.lax.stop_gradient(self.row_mean(weights, batch).total)
        return total / len(kls), mass

    def _warmup_loss(self, params, batch, step: Step) -> tuple[Ratio, Aux[Variables]]:
        """Score the dense warm-up: the indexer's KL alone.

        The main loss is never scored, since nothing it reaches moves.
        """
        assert self.indexer is not None
        prepared = batch_text(batch)
        inputs, _ = self._rows(prepared.tokens)
        collections = [INDEXER_COLLECTION] + (["qk"] if self.qk_stats else [])
        _, gathered = self._hidden_states(
            thaw(params), inputs, train=step.training, rngs=model_rngs(step.key, training=step.training),
            collections=collections, packing=prepared.slice_tokens(stop=-1).kwargs())
        total, mass = self._indexer_term(gathered[INDEXER_COLLECTION], prepared, batch)
        reported = {"indexer_kl": total / jnp.where(mass > 0, mass, 1)}
        qk = gathered.get("qk") if self.qk_stats else None
        if self.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
        return Ratio(self.indexer.weight * total, mass), Aux(reported, qk_stats=qk)

    def _rows(self, tokens) -> tuple[jax.Array, jax.Array]:
        """Split a `[B, seq_len + 1]` batch into its inputs and shifted targets."""
        if tokens.shape[-1] != self.seq_len + 1:
            raise ValueError(
                f"a {self.seq_len}-token context needs {self.seq_len + 1} ids per row "
                f"so the targets can be the shifted input, got {tokens.shape[-1]}")
        return tokens[:, :-1], tokens[:, 1:]

    def loss(self, variables, batch, step: Step) -> tuple[Ratio | LMStatistics, Aux[Variables]]:
        if self._warmup:
            return self._warmup_loss(variables, batch, step)
        statistics, aux, _ = self._scored_loss(variables, batch, step, train=step.training)
        return statistics, aux

    def predict(self, params, batch, step: Step, *, train: bool,
                layers: Sequence[int] = ()) -> tuple[Ratio, Aux[Variables], Prediction]:
        """Return the loss with the logits, target weights and `layers` outputs behind it.

        A teacher compares these (`Objective.predict`). The logits are the whole
        `[B, seq_len, vocab]` fp32 tensor that the chunked loss never holds, because a
        distillation's KL reads every column. `aux_loss_alpha`'s router terms have
        their own normalization, so they cannot share a distillation's token mass and
        are refused; `balance_rate` balances without a loss term. The indexer warm-up
        scores no tokens, so it has nothing to distill.
        """
        if self._warmup:
            raise ValueError("the indexer warm-up scores no token, so it has no prediction")
        if self.aux_loss_alpha is not None or self.router_z_loss:
            raise ValueError(
                "aux_loss_alpha's and router_z_loss's router terms normalise per router, and a "
                "distillation mixes terms over the counted tokens; balance the "
                "student's routers with balance_rate instead")
        statistics, aux, scores = self._scored_loss(params, batch, step, train=train, layers=layers)
        assert isinstance(statistics, Ratio)
        logits = constrain(model_logits(self.model, thaw(params), scores.hidden)
                           if reads_states(self.model) else scores.hidden, LOGITS)
        return statistics, aux, Prediction(logits, scores.losses, scores.weights, scores.layers)

    def _scored_loss(self, params, batch, step: Step, *, train: bool, layers: Sequence[int] = ()
                     ) -> tuple[Ratio | LMStatistics, Aux[Variables], Scores]:
        """Compute the loss's statistics and reports, with the scores they came from."""
        prepared = batch_text(batch)
        rate = self.balance_rate
        alpha = self.aux_loss_alpha
        scores = self.token_scores(
            params, prepared, train=train, rngs=model_rngs(step.key, training=train),
            routing=rate is not None or alpha is not None or bool(self.router_z_loss),
            depths=self.mtp_weight is not None, roles=self._batch_roles(batch),
            qk_stats=self.qk_stats, indexer=self.indexer is not None, layers=layers)
        losses, weights, log_z, correct, _, _, routing, depths, qk, kls = scores
        mass = jax.lax.stop_gradient(self.row_mean(weights, batch).total)
        prediction = Ratio(self.row_mean(losses * weights, batch).total, mass)
        ce, _ = prediction.mean()
        reported = {"ce": ce, "perplexity": jnp.exp(ce)}
        if correct is not None:
            reported["token_accuracy"], _ = self.accuracy(correct, batch, weights).mean()
        if self.z_loss:
            # PaLM's auxiliary over the same counted targets as the cross
            # entropy, so one Ratio carries both.
            z_total = self.z_loss * self.row_mean(jnp.square(log_z) * weights, batch).total
            reported["z_loss"] = z_total / jnp.where(mass > 0, mass, 1)
            prediction = Ratio(prediction.total + z_total, mass)
        if self.mtp_weight is not None:
            # Depths retain the main target denominator and configured depth average.
            mtp_total = jnp.mean(jnp.stack([
                self.row_mean(depth_losses * depth_weights, batch).total
                for depth_losses, depth_weights in depths]))
            reported["mtp_ce"], _ = Ratio(mtp_total, mass).mean()
            prediction = Ratio(prediction.total + self.mtp_weight * mtp_total, mass)
        if self.indexer is not None:
            # The KL shares the cross entropy's denominator, as the depths
            # do, so one Ratio carries the step: the queries counted differ
            # from the targets only by the documents' last tokens. The
            # report is the KL per counted query, the paper's quantity.
            total, queries = self._indexer_term(kls, prepared, batch)
            reported["indexer_kl"] = total / jnp.where(queries > 0, queries, 1)
            prediction = Ratio(prediction.total + self.indexer.weight * total, mass)
        statistics: Ratio | LMStatistics = prediction
        if alpha is not None:
            statistics, reported["aux_loss"] = self._router_statistics(prediction, routing, alpha, batch)
        if self.router_z_loss:
            router_z = router_z_terms(routing or {}, self.router_z_loss, batch)
            if not router_z:
                raise ValueError(
                    "router_z_loss needs routers that sow their gate's log partition "
                    "(dew.nn.moe.Router); this model's routers sow none")
            statistics = (LMStatistics(prediction, (), (), router_z) if isinstance(statistics, Ratio)
                          else dataclasses.replace(statistics, router_z=router_z))
            reported["router_z_loss"] = jnp.sum(jnp.stack([term.mean()[0] for term in router_z]))
        effects = None
        if rate is not None:
            effects, load = self._router_load(params, routing)
            reported.update(load)
        if self.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
        return statistics, Aux(reported, qk_stats=qk, effects=effects), scores

    def _router_statistics(self, prediction: Ratio, routing, alpha: float, batch: Batch
                           ) -> tuple[LMStatistics, jax.Array]:
        """Add the balance loss's own statistics beside the prediction's.

        Each router term normalises by its own count, not by the counted
        tokens, so the terms travel as separate statistics and only
        `reduce_loss` adds them up. The reported number is what they add.
        """
        if not routing:
            raise ValueError("aux_loss_alpha requires a model with a mixture")
        routers = _router_scores(routing)
        sequence = tuple(self.row_mean(sequence_router_losses(s, i, alpha), batch)
                         for s, i in routers) if self.seq_aux else ()
        global_routers = () if self.seq_aux else tuple(router_moments(s, i, batch) for s, i in routers)
        statistics = LMStatistics(prediction, sequence, global_routers)
        combined, _ = self.reduce_loss(statistics)
        prediction_loss, _ = prediction.mean()
        return statistics, combined - prediction_loss

    def _router_load(self, params: Variables, routing) -> tuple[Variables, dict[str, jax.Array]]:
        """Count what each balanced router selected, and report how evenly.

        The counts are the effect the optimizer applies once per commit;
        the report is each router's busiest and idlest share of the batch.
        """
        if "moe" not in params or routing is None:
            raise ValueError("balance_rate requires a mixture with bias=True")
        ran = {name: bias for name, bias in params["moe"].items()
               if self.mtp_weight is not None or not name.startswith("mtp_")}
        counts = router_counts(ran, routing)
        shares = [count / jnp.sum(count) for count in jax.tree.leaves(counts)]
        if not shares:
            raise ValueError(
                "balance_rate requires routers that keep a balancing bias, "
                "and these select by their hash table alone")
        return counts, {"moe/max_load": jnp.mean(jnp.stack([x.max() for x in shares])),
                        "moe/min_load": jnp.mean(jnp.stack([x.min() for x in shares]))}

    def reduce_loss(self, stats: Ratio | LMStatistics) -> tuple[jax.Array, jax.Array]:
        if isinstance(stats, Ratio):
            return stats.mean()
        value, active = stats.prediction.mean()
        for term in stats.sequence:
            auxiliary, supported = term.mean()
            value, active = value + auxiliary, active | supported
        for term in stats.global_routers:
            if self.aux_loss_alpha is None:
                raise ValueError("global router statistics require aux_loss_alpha")
            value = value + global_router_loss(term, self.aux_loss_alpha)
            active = active | (term.positions > 0)
        for term in stats.router_z:
            auxiliary, supported = term.mean()
            value, active = value + auxiliary, active | supported
        return value, active

    def apply_effects(self, variables: Variables, effects: Variables) -> Variables:
        rate = self.balance_rate
        if rate is None:
            raise ValueError("router count effects require balance_rate")
        moe = variables["moe"]
        active = _balanced_biases({name: moe[name] for name in effects})
        balanced = jax.tree.map(
            lambda bias, count: _updated_bias(bias, count, rate),
            active, effects)
        return {"moe": merge(moe, balanced)}

    def evaluate(self, params, batch, step: Step):
        """Score the complete batch teacher-forced, using EMA when present; the
        rows and roles are read outside the compiled scoring."""
        prepared = {TEXT_KEY: batch_text(batch), ROLES_KEY: self._batch_roles(batch)}
        return super().evaluate(params, prepared, step)

    def _evaluation_scores(self, params, batch, key) -> TokenScores:
        scores = self.token_scores(params, batch[TEXT_KEY], roles=batch[ROLES_KEY], predict=True)
        assert scores.correct is not None
        return TokenScores(scores.losses, scores.weights, scores.correct)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Sample the configured prompt once, then decode only on process zero.

        An objective whose EMA holds a frozen reference draws from the live
        policy instead, which is what `_ema_is_reference` says.
        """
        settings = self.samples

        def setup():
            if settings is None or settings.max_new_tokens <= 0:
                return None
            weights = self.evaluation_variables(params, step)
            return self.policy(weights, settings.sampling), self._prompt, settings.max_new_tokens

        def generate(prepared):
            policy, prompt, max_new_tokens = prepared
            return policy(prompt, max_new_tokens, key=step.key).host().tokens, prompt

        return text_preview("LM preview", setup, generate, None if settings is None else settings.decode)


class Perplexity:
    """Reports exp of the cross entropy per counted target over a whole pass.

    Each batch is weighted by its own count of counted targets, so a packed or
    padded pass whose batches differ in size is scored per token, and a batch with
    no counted target contributes nothing.
    """

    name = "perplexity"
    reads = TokenScores
    shown = Shown(better="lower")

    def __call__(self, artifact: Artifact, batch: Batch, /) -> tuple[float, float]:
        assert isinstance(artifact, TokenScores), "the trainer hands a metric the artifact it reads"
        weights = np.asarray(artifact.weights, dtype=np.float64)
        losses = np.asarray(artifact.losses, dtype=np.float64)
        return float(np.sum(losses * weights)), float(np.sum(weights))

    merge = staticmethod(merge_totals)

    def finalize(self, accumulated: tuple[float, float]) -> float:
        total, count = accumulated
        if count == 0:
            raise ValueError("no counted target in the validation pass")
        return float(np.exp(total / count))


def _head_tile(tile, keeps_whole_logits: bool) -> tuple[int, int] | None:
    """The head's tile: None keeps the whole logits; 'whole' and 'tiled' name
    the choice; None on an objective that does not keep them by default is
    the generation's tile (`chunked_tile`). A config's pair is a list, and
    the tile is a static argument, so it becomes a tuple."""
    if tile == 'whole' or (tile is None and keeps_whole_logits):
        return None
    if tile is None or tile == 'tiled':
        return chunked_tile()
    return (int(tile[0]), int(tile[1]))
