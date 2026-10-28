"""The autoregressive language-modelling objective.

It trains next-token prediction over packed token ids. A batch row holds
`seq_len + 1` ids, or as many as its chain lays out within the model's context;
the model sees all but the last, and the targets are the same row shifted by
one. The causal mask is in the backbone, so this module does not need to know
how the model hides the future from a prediction.

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
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import ArrayLike

from dew._coordination import agreed, collective_host
from dew.artifacts import Artifact, TextSamples, TokenScores
from dew.batch import (
    BLOCKS_FIELD,
    COORDINATES_FIELD,
    FROZEN,
    VALID_ROWS,
    Batch,
    Coordinates,
    PathFilter,
    Variables,
    freeze,
    leaves_by_path,
    merge,
    real_slots,
    scored_targets,
    shifted_span,
    sown_nodes,
    thaw,
    token_layout,
)
from dew.data.chat import ROLES_KEY, Role
from dew.inference import TextGeneration
from dew.inference.tasks import Processor, TextSettings, recorded_processor, recorded_tokenizer
from dew.inputs import Field, InputSpec
from dew.nn.cross_attention import crosses_alone
from dew.nn.flops import MissingFlopsCount, counted_flops, frozen_training_flops, parameter_flops
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS, ModelInputs, pad_token_rows, rotate_rows
from dew.nn.mla import INDEXER, INDEXER_COLLECTION
from dew.nn.moe import ROUTER_COLLECTION
from dew.nn.precision import at_least_fp32
from dew.nn.protocols import (
    Bounded,
    CountedLogits,
    CountedLogitsTraining,
    CountedPredictions,
    CountedPredictionTraining,
    CrossAttending,
    DecoderTraining,
    Lacks,
    LayerStates,
    PositionCapacity,
    SteppedInContext,
    TokenModel,
    head_table,
    refuse_block_reading,
    refuse_prefix,
)
from dew.nn.sharding import LOGITS, constrain
from dew.nn.visibility import prefix_groups
from dew.objectives.base import (
    OMITTED,
    Aux,
    EMASpec,
    Omitted,
    Prediction,
    Ratio,
    Shown,
    Step,
    ended,
    merge_totals,
    model_rngs,
)
from dew.objectives.lm.adaptation import ChunkAdaptation, adapted_scores
from dew.objectives.lm.chunked import (
    final_read,
    fused_head_loss,
    logits_cross_entropy,
    logits_support_log_probs,
    model_logits,
    refuse_without_logits,
    support_log_probs,
)
from dew.objectives.lm.head import HeadedObjective
from dew.objectives.routing import ExpertBalance, RoutedStatistics, apply_routed, reduce_routed
from dew.sampling.text import Sampling

if TYPE_CHECKING:
    from dew.nn.backbones.decoder_stack import DecoderBank

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""


BLOCKS_KEY = "text_block_groups"
"""Batch key `[B, seq_len + 1]` int32 block ids are read under, written by `dew.data.steps.blocks.Blocks`."""


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
        raise Lacks("causal", "LMObjective requires a causal model for next-token likelihoods")
    refuse_without_logits(model, "LMObjective")


def _streamed_depths(model: nn.Module) -> bool:
    """Say whether the prediction depths carry their own residual streams (`DecoderTraining`)."""
    return isinstance(model, DecoderTraining) and model.mtp_hyper_connections is not None


def _check_terms(model: nn.Module, *, mtp_weight: float | None, z_loss: float) -> None:
    """Refuse a weight the term it scales cannot carry.

    The checks run in the order the constructor takes the arguments. The
    prediction depths' cross entropy is weighted positively or left out. The
    depths have to exist for a weight on them to move anything. The log
    partition's weight is finite and nonnegative.
    """
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


def _indexer_kls(sown: Variables) -> list[jax.Array]:
    """Collect every attention layer's sown per-query indexer KL.

    Each is `[B, S]`, and they come back in the tree's key order.
    """
    kls: list[jax.Array] = []
    for _, node in sown_nodes(sown, "kl"):
        (kl,) = node["kl"]
        kls.append(kl)
    return kls


def _prepared(tokens: ModelInputs | jax.Array, segment_ids: jax.Array | None = None,
              positions: jax.Array | None = None, coordinates: Coordinates | None = None,
              block_groups: jax.Array | None = None) -> ModelInputs:
    """The rows as `ModelInputs`, bare ids cast to int32, their packing among the token fields.

    A packed batch from a data pipeline carries its windows' `coordinates`
    (`dew.batch.Coordinates`, the packer's); a rollout's names each row's
    `segment_ids` and `positions`. A batch's `block_groups` (a prefix run's
    blocks) come the same way. Either comes beside the ids or in the
    `ModelInputs`; a field named both ways is refused. Everything after reads
    the packing off what this returns (`dew.batch.token_layout`).
    """
    prepared = tokens if isinstance(tokens, ModelInputs) else ModelInputs(jnp.asarray(tokens, jnp.int32))
    columns = {name: column for name, column in (("positions", positions), ("segment_ids", segment_ids),
                                                 (COORDINATES_FIELD, coordinates),
                                                 (BLOCKS_FIELD, block_groups))
               if column is not None}
    for name in columns:
        if name in prepared.token_fields:
            raise ValueError(f"{name} must come from either ModelInputs or the packing column")
    return (
        dataclasses.replace(prepared, token_fields={**prepared.token_fields, **columns})
        if columns
        else prepared
    )


def prompt_batch(prompt: Sequence[int] | Sequence[Sequence[int]]) -> ModelInputs:
    """One prompt's ids, or several prompts of any lengths, as one left-padded
    request (`pad_token_rows`): its rows, and an attention mask where they differ."""
    rows = ([np.asarray(prompt).tolist()] if np.ndim(prompt[0]) == 0
            else [np.asarray(row).tolist() for row in prompt])
    tokens, fields = pad_token_rows(rows)
    return ModelInputs(jnp.asarray(tokens), {name: jnp.asarray(value) for name, value in fields.items()})


@dataclass(frozen=True)
class Samples:
    """Settings for the text preview drawn once per event.

    Prompts hold token IDs; several prompts may differ in length, and are
    left-padded together (`pad_token_rows`). This display
    count does not limit the teacher-forced scoring population. A budget of 0
    draws no preview but keeps `sampling` as the policy the run records and
    publishes.
    """
    prompt: Sequence[int] | Sequence[Sequence[int]]
    max_new_tokens: int
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    decode: Callable[[list[int]], str] = lambda ids: str(ids)


def _unpadded(values: jax.Array, padding: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Undo a left alignment on scored rows, and mark the slots that scored.

    `values` were scored over rows whose real tokens had been moved down to
    position zero by `padding` places. They come back to the columns they
    were scored for, beside a mask that is true where the row held a real
    token rather than padding.
    """
    restored = rotate_rows(values, -padding)
    valid = jnp.arange(restored.shape[1])[None, :] >= padding[:, None]
    return restored, valid


def batch_text(batch) -> ModelInputs:
    """The batch's `text` rows as `_prepared` gives them, a packed batch's `text_coordinates` folded in,
    and its `text_block_groups` beside them."""
    stale = sorted({"text_segment_ids", "text_positions"} & set(batch))
    if stale:
        raise ValueError(f"a packed batch lays its documents out in text_coordinates "
                         f"(dew.batch.coordinates), not {', '.join(stale)}")
    laid = batch.get("text_coordinates")
    blocks = batch.get(BLOCKS_KEY)
    return _prepared(batch[TEXT_KEY], coordinates=None if laid is None else jax.tree.map(jnp.asarray, laid),
                     block_groups=None if blocks is None else jnp.asarray(blocks, jnp.int32))


def prefixed_fields(fields: Mapping[str, jax.Array | Coordinates], shape: tuple[int, ...],
                    prefix: int | Literal["prompt"] | None, roles, *,
                    xp=jnp) -> Mapping[str, jax.Array | Coordinates]:
    """`fields` with the blocks a prefix run reads where they carry none: each document's first `prefix`
    places (`prefix_blocks`), or under "prompt" its tokens before its first assistant token, read from the
    rows' `roles` (`role_blocks`). A run's prefix is its loss's mask whatever built the batch, a precompiled
    loader or a raw stream included, so the blocks it trains on are the ones it serves; the data stage
    `Blocks` writes the same. `shape` is the rows' `[B, S]`, `xp` the array module (the host's counts read
    numpy, the loss jax.numpy)."""
    if prefix is None or BLOCKS_FIELD in fields:
        return fields
    _, documents = token_layout(fields, xp=xp)
    real = real_slots(fields, shape, xp=xp)
    if prefix == "prompt" and roles is None:
        raise ValueError(f"a prompt prefix run reads each row's roles ({ROLES_KEY}) to find its "
                         "prompt, and this batch has none")
    laid = fields.get(COORDINATES_FIELD)
    positions = laid.positions if isinstance(laid, Coordinates) else fields.get("positions")
    groups = prefix_groups(xp.ones(shape, xp.int32) if documents is None else documents, real, prefix,
                           positions=positions, roles=roles, assistant=Role.ASSISTANT, xp=xp)
    return {**fields, BLOCKS_FIELD: groups}


def text_preview[S](phase: str, setup: Callable[[], S | None],
                     generate: Callable[[S], tuple[ArrayLike, ArrayLike | None, ArrayLike | None]],
                     decode: Callable[[list[int]], str] | None) -> TextSamples | None:
    """A text preview, on every process: agree on `setup`, generate from what
    it returns the rows, which of their slots are real (None for all), and the
    ids of the prompt they continue (None for none), gather them to process
    zero's host (`collective_host`, held by the first), and decode each row's
    real ids into `TextSamples` there. The other processes return None, as
    every process does when `setup` returns None, which draws no preview.
    Without `decode`, the artifact holds the ids alone."""
    prepared = agreed(f"{phase} setup", setup)
    drawn = agreed(f"{phase} generation", lambda: None if prepared is None else generate(prepared))
    drawn = collective_host(drawn, phase=phase, held_by="first")
    if drawn is None:
        return None
    tokens, real, prompt = drawn
    rows = np.asarray(tokens)
    if decode is None:
        return TextSamples(tokens=rows)
    kept = np.ones(rows.shape, bool) if real is None else np.asarray(real, bool)
    return TextSamples(tokens=rows, prompt="" if prompt is None else decode(np.asarray(prompt).tolist()),
                       texts=tuple(decode(row[keep].tolist()) for row, keep in zip(rows, kept, strict=True)))


def _global_qk_max(qk) -> jax.Array | None:
    """Return the largest per-head maximum anywhere in the sowed `qk` dict.

    None when no layer sowed one.
    """
    found: list[jax.Array] = []
    for _, node in sown_nodes(qk, "max_logits"):
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
    routers recorded, `depths` the prediction depths' (losses, weights, total)
    triples, `qk` the attention layers' per-head logit maxima, and `indexer`
    their per-query indexer KL; each is None or empty unless its flag asked for
    it. `total` is the weighted sum of the losses and the z-loss
    (`chunked.fused_head_loss`), and a depth's total its losses' weighted sum,
    where `token_scores` was asked for the sums alone (`z_loss`): then only
    the totals carry a gradient. Otherwise they are None. `logits` is the
    `[B, seq_len, vocab]` logits the losses were scored from, where
    `token_scores` was asked for them, else None.
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
    total: jax.Array | None = None
    logits: jax.Array | None = None


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


class TokenScorer(HeadedObjective[Ratio | RoutedStatistics, Variables]):
    """Scores a causal decoder's tokens: the machinery the LM, preference and
    policy losses share, without a loss of its own.

    It holds the decoder (`model`, a loaded source among them, with
    `variables` and `processor` overriding its own through
    `Objective.bind_model`), the row length `seq_len`, which a batch's rows may exceed up to the
    model's own context (`context`), and how the head is scored: `head_chunks` and `head_tile`, the
    head policy every token objective takes (`dew.objectives.lm.head`; `LMObjective` describes each).
    `samples` draws previews. `balance` trains the routers of a mixture as
    `ExpertBalance` describes (`dew.objectives.routing`), which every scorer
    composes. A loss built on it keeps no average unless it sets `ema`;
    `LMObjective` adds the language-model terms and loss, and the options
    only they read, `pad_id` and `token_accuracy`.
    """

    saved_task = TextGeneration
    settings = TextSettings
    shown: Mapping[str, Shown] = {"ce": Shown(better="lower"), "perplexity": Shown(better="lower"),
             "token_accuracy": Shown(better="higher", percent=True)}

    counts_tokens = True

    loss_role: Role | None = None
    """The role whose targets alone count (`LMObjective`); None counts every target."""

    prefix: int | Literal["prompt"] | None = None
    """The prefix run's blocks (`LMObjective`): a document's first tokens, or its prompt; None is causal."""

    mtp_weight: float | None = None
    """The multi-token prediction depths' weight (`LMObjective`); None trains none."""

    pad_id: int | None = None
    """The id whose targets never count (`LMObjective`); None counts every id."""

    token_accuracy: bool = False
    """Whether `token_scores` marks each prediction by default (`LMObjective`'s token accuracy).
    A policy or preference loss reports none, and asks for the marks where it scores them."""

    adaptation: ChunkAdaptation | None = None
    """How the model's continuum level steps in context by the loss it scores (`LMObjective`);
    None reads the level as it is trained."""

    scored_fields: ClassVar[tuple[str, ...] | None] = None
    """The batch fields `_evaluation_scores` reads, which alone enter its compiled program (`evaluate`),
    so a batch's other fields (rollouts, rewards) neither enter it nor key it; None passes the batch whole."""

    def evaluate(self, params, batch, step: Step):
        """The base's scores (`Objective.evaluate`) over the `scored_fields` of `batch`."""
        if self.scored_fields is not None:
            batch = {name: batch[name] for name in self.scored_fields}
        return super().evaluate(params, batch, step)

    def __init__(self, model, seq_len: int, *, head_chunks: int = 4,
                 head_tile: tuple[int, int] | Literal['whole', 'tiled'] | None = None,
                 samples: Samples | None = None, variables: Variables | None | Omitted = OMITTED,
                 balance: ExpertBalance | None = None,
                 processor: Processor | None | Omitted = OMITTED):
        """Bind the decoder and set how its tokens are scored; the class docstring describes each argument."""
        self.eos_token_ids = ended(model, OMITTED)
        model = self.bind_model(model, variables=variables, processor=processor)
        _check_reads(model)
        self.model = model
        self.seq_len = seq_len
        # A row longer than its positions cover would read past a learned table, which clamps silently.
        capacity = model.position_capacity if isinstance(model, PositionCapacity) else None
        if capacity is not None and seq_len > capacity:
            raise ValueError(f"{seq_len}-token rows read past the {capacity} places {type(model).__name__}'s "
                             "positions cover")
        # A schedule's later stage may read longer rows, up to what the model declares it holds.
        declared = model.max_seq_len if isinstance(model, Bounded) else None
        self.context = seq_len if declared is None else max(seq_len, declared)
        self.use_head(head_chunks, head_tile)
        self.samples = samples
        self.balance = balance
        self.inputs = InputSpec(sample=Field(TEXT_KEY, (self.context + 1,), variable_axes=(0,)))
        if samples is not None and samples.max_new_tokens > 0:
            self._prompt = prompt_batch(samples.prompt)

    def reduce_loss(self, stats: Ratio | RoutedStatistics) -> tuple[jax.Array, jax.Array]:
        """The objective value of the statistics, with the router terms `balance` adds (`reduce_routed`)."""
        return reduce_routed(self.balance, stats)

    def apply_effects(self, variables: Variables, effects: Variables) -> Variables:
        """`variables` with the router counts `balance` took applied, once per committed update."""
        return apply_routed(self.balance, variables, effects)

    def _missing_flops(self, model) -> str | None:
        """What the forward and training counts both lack, or None."""
        if type(self) is not LMObjective:
            return f"{type(self).__name__} needs counts for its auxiliary model forwards"
        if not isinstance(model, CountedLogits):
            return f"{type(model).__name__} does not declare logits_flops for the LM head"
        if self.mtp_weight and not isinstance(model, CountedPredictions):
            return f"{type(model).__name__} does not declare prediction_flops for its MTP depths"
        if self.adaptation is not None:
            return "a continuum level stepped in context has no count of its windows' gradients"
        return None

    def _token_geometry(self, batch: Batch) -> tuple[np.ndarray, Mapping[str, jax.Array | Coordinates]]:
        """The host's tokens and their fields, a packed batch's coordinates and blocks among them: what the
        loss reads."""
        raw = batch[TEXT_KEY]
        prepared = raw if isinstance(raw, ModelInputs) else ModelInputs(raw)
        extra = {name: batch[key] for name, key in ((COORDINATES_FIELD, "text_coordinates"),
                                                    (BLOCKS_FIELD, BLOCKS_KEY)) if key in batch}
        return np.asarray(prepared.tokens), {**prepared.token_fields, **extra}

    def _counted_tokens(self, batch: Batch):
        """Host input geometry: each real token, and the documents."""
        tokens, fields = self._token_geometry(batch)
        valid = real_slots(fields, tokens.shape, xp=np)
        if self.pad_id is not None:
            valid = valid & (tokens != self.pad_id)
        return valid, token_layout(fields, xp=np)[1]

    def scored_tokens(self, batch: Batch) -> np.ndarray:
        """Each row's targets the loss scores (`scored_targets`), its `loss_role`'s alone."""
        return self._prediction_lengths(batch)

    def row_tokens(self, batch: Batch) -> int | None:
        """Each row's targets, of the ids under the field the scorer reads (`inputs`): every sequence a
        row holds (a preference pair's two) has one fewer than its ids."""
        if self.inputs is None:
            return None
        *sequences, ids = self.inputs.sample.held(batch)
        return math.prod(sequences) * (ids - 1)

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """Real input lengths, each packed document its own attention square.

        The shifted LM model sees all but the last input column. Repeated
        segment ids on different rows remain separate documents.
        """
        valid, documents = self._counted_tokens(batch)
        return self._row_lengths(valid[:, :-1], None if documents is None else documents[:, :-1])

    @staticmethod
    def _row_lengths(valid, documents) -> np.ndarray:
        if documents is None:
            return np.sum(valid, axis=-1, dtype=np.int64)
        lengths = []
        for row, visible in zip(documents, valid, strict=True):
            _, counts = np.unique(row[visible], return_counts=True)
            lengths.extend(counts)
        return np.asarray(lengths, np.int64)

    def model_flops(self, model, batch: Batch) -> float:
        """Hidden states plus predictions within real documents; never padded logits."""
        model = self.model if model is None else model
        missing = self._missing_flops(model)
        if missing is not None:
            raise MissingFlopsCount(missing)
        crossed = self._context_flops(model, batch)
        lengths = self._prediction_lengths(batch)
        count = counted_flops(model, self.real_lengths(batch)) + model.logits_flops(lengths) + crossed
        return count + (model.prediction_flops(self.real_lengths(batch),
                                               prediction_lengths=self._prediction_rows(model, batch))
                        if self.mtp_weight else 0)

    def training_flops(self, model, batch: Batch) -> float:
        """The adapted trunk and prediction head on actual token geometry."""
        model = self.model if model is None else model
        missing = self._missing_flops(model)
        if missing is not None:
            raise MissingFlopsCount(missing)
        crossed = self._context_flops(model, batch)
        targets = self._prediction_lengths(batch)
        lengths = self.real_lengths(batch)
        if self._trains_cross_attention_alone(model):
            if self.mtp_weight:
                raise MissingFlopsCount("a frozen trunk's prediction depths have no frozen training count")
            # Every trunk and head matrix frozen: its forward and its input's gradient, no weight's.
            trunk, head = counted_flops(model, lengths), model.logits_flops(targets)
            return (frozen_training_flops(trunk, parameter_flops(model, lengths))
                    + frozen_training_flops(head, head) + 3 * crossed)
        if not isinstance(model, CountedLogitsTraining):
            return super().training_flops(model, batch)
        # Every cross-attention matrix trains: its forward, its input's gradient and its own.
        count = model.training_flops(lengths) + model.logits_training_flops(targets) + 3 * crossed
        if self.mtp_weight:
            if not isinstance(model, CountedPredictionTraining):
                raise MissingFlopsCount(f"{type(model).__name__} has no prediction_training_flops "
                                        "for adapted depths")
            count += model.prediction_training_flops(
                lengths, prediction_lengths=self._prediction_rows(model, batch))
        return count

    def _prediction_rows(self, model: CountedPredictions,
                          batch: Batch) -> list[tuple[np.ndarray, np.ndarray]]:
        """A depth's real fusion inputs and scored targets differ at document boundaries.

        Every intervening input must be real; the targets are the ones the
        loss scores (`scored_targets`). Counting totals and subtracting a depth
        would join across pad holes.
        """
        valid, documents = self._counted_tokens(batch)
        tokens, fields = self._token_geometry(batch)
        depth_rows = []
        for depth in range(1, model.num_nextn_predict_layers + 1):
            span = valid.shape[1] - 1 - depth
            admitted = shifted_span(valid.shape, depth, valid=valid, segments=documents, xp=np)[:, :-1]
            segments = None if documents is None else documents[:, :span]
            scored = self._scored(np, tokens, fields, self._batch_roles(batch, np), depth)
            depth_rows.append((self._row_lengths(admitted, segments), self._row_lengths(scored, segments)))
        return depth_rows

    def _trains_cross_attention_alone(self, model) -> bool:
        """Whether this objective trains a decoder's cross-attention blocks and nothing else: its
        variables split (`FROZEN`) with every trained leaf a block's (`dew.nn.cross_attention`), as
        `Pretrained.with_cross_attention` splits a source."""
        held = self.variables
        if held is None or FROZEN not in held or not isinstance(model, CrossAttending) or (
                model.cross_attention is None):
            return False
        return crosses_alone(held["params"])

    def _context_flops(self, model, batch: Batch) -> float:
        """The forward work of the cross-attention blocks over the batch's context
        (`CrossAttending.cross_flops`), each row's real tokens reading its context's;
        none without conditioning. Media needs its own count, which raises
        `MissingFlopsCount`."""
        raw = batch[TEXT_KEY]
        conditioning = raw.conditioning if isinstance(raw, ModelInputs) else {}
        if not conditioning:
            return 0.0
        if (set(conditioning) - {CONTEXT, CONTEXT_LENGTHS} or not isinstance(model, CrossAttending)
                or model.cross_attention is None):
            raise MissingFlopsCount("LMObjective needs its conditioning tower's multi-sequence geometry")
        valid, _ = self._counted_tokens(batch)
        context = np.asarray(conditioning[CONTEXT])
        lengths = conditioning.get(CONTEXT_LENGTHS)
        lengths = (np.full(context.shape[:1], context.shape[1]) if lengths is None
                   else np.asarray(lengths)).astype(np.int64)
        return float(model.cross_flops(np.sum(valid[:, :-1], axis=-1, dtype=np.int64), lengths))

    def _prediction_lengths(self, batch: Batch, depth: int = 0) -> np.ndarray:
        """Each row's targets the loss scores at `depth` (`scored_targets`)."""
        tokens, fields = self._token_geometry(batch)
        scored = self._scored(np, tokens, fields, self._batch_roles(batch, np), depth)
        return np.sum(scored, axis=-1, dtype=np.int64)

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        return self.model.bank_sites(store)

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        return self.model.init(key, jnp.zeros((1, self.seq_len), jnp.int32))

    def sampled(self, sampling: Sampling) -> Sampling:
        """`sampling` with the end-of-sequence ids of the source this objective was built over where it
        names none (`ended`), so a policy drawn from a loaded source stops where the source's own task
        does; a bare model's objective names none to add."""
        if sampling.eos_token_ids is not None or not self.eos_token_ids:
            return sampling
        return dataclasses.replace(sampling, eos_token_ids=self.eos_token_ids)

    def policy(self, params: Variables, sampling: Sampling = _DEFAULT_SAMPLING) -> TextGeneration:
        """Return the model over this training tree as a generation task, under `sampling` (`sampled`).

        A rollout uses one snapshot of the policy and draws every completion from it;
        the result records the actual and raw-policy likelihoods that the objective's
        ratio needs. An objective that steps its level in context (`adaptation`) has no
        generation that steps it, so it refuses (ValueError) rather than draw from the
        level it starts with.
        """
        if self.adaptation is not None:
            raise ValueError("an objective that steps its continuum level in context scores what no "
                             "generation draws yet: a generation reads the level it starts with")
        return TextGeneration(self.model, params, sampling=self.sampled(sampling), prefix=self.prefix)

    def task_settings(self) -> TextSettings | None:
        """The decoder's row length, preview budget, sampling policy (`sampled`), and tokenizer or
        source processor. None for an objective that steps its level in context (`adaptation`), which
        no generation task serves yet: its saved runs restore their trained state, and build no task."""
        if self.adaptation is not None:
            return None
        samples = self.samples
        return TextSettings(seq_len=self.seq_len,
                            max_new_tokens=0 if samples is None else samples.max_new_tokens,
                            sampling=self.sampled(Sampling() if samples is None else samples.sampling),
                            tokenizer=recorded_tokenizer(self.processor),
                            processor=recorded_processor(self.processor), prefix=self.prefix)

    def _refuse_blocks(self, prepared: ModelInputs, depths: bool) -> None:
        """Refuse a batch's blocks where no trunk layer reads them, where prediction depths would read
        them beyond their source, which `scored_targets` does not account for, or where a layer's chunk could
        cut one (`refuse_block_reading`): a batch's blocks are runs of any place in a row, a prefix run's
        among them, since its batch may carry blocks of its own. The check is static, since a trace cannot
        read the runs' places."""
        if BLOCKS_FIELD not in prepared.token_fields:
            return
        if depths:
            raise ValueError("a batch's blocks train the trunk's targets alone: a prediction depth reads "
                             "blocks beyond its own source, so depths are refused with blocks")
        refuse_block_reading(self.model, "a batch's blocks")

    def token_scores(self, params, tokens, train: bool = False, rngs=None,
                     segment_ids=None, positions=None, routing: bool = False,
                     depths: bool = False, roles=None, qk_stats: bool = False,
                     indexer: bool = False, layers: Sequence[int] = (),
                     routes: tuple[jax.Array, jax.Array | None] | None = None,
                     predict: bool | None = None, z_loss: float | None = None, logits: bool = False):
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
        weights them from its scores (`dew.nn.protocols.Routes`); the stack slices
        the record by layer however it runs.

        `z_loss` asks for the weighted sums alone (`Scores.total`, PaLM's
        z-loss at that weight included, and each depth's), the caller's loss
        being linear in the per-token cross entropies: a head the model holds
        as a table and tiles (`head_tile`, where its whole logits do not fit)
        then takes them with their gradient in its forward
        (`chunked.fused_head_loss`, while `head_fused`), a tile's tokens at
        most a chunk, and the per-token results carry none. A chunk scores
        the whole vocabulary, which the tiled head bounds to a tile, so the
        fit ladder moves a fused head that does not fit to the tiled head
        (`LMObjective.unfuse_head`) before it recomputes more: at an 8B's
        head on an A100 (4096 wide, vocabulary 128256, 4096 tokens in one
        chunk) the fused head took 58.4 ms against the tiled head's 64.9.
        Where the whole logits fit, they are kept: three products either way,
        and on an H100 the whole head ran Qwen3.5-0.8B's 4 x 3072 step at
        36.1% MFU against the fused head's 34.6%.

        `logits` scores the losses from the whole `[B, seq_len, vocab]`
        logits and returns them (`Scores.logits`), for a caller that reads
        every column, so the head is applied once.
        """
        prepared = _prepared(tokens, segment_ids, positions)
        self._refuse_blocks(prepared, depths)
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
        collections = (([ROUTER_COLLECTION] if routing else []) + (["qk"] if qk_stats else [])
                       + ([INDEXER_COLLECTION] if indexer else []))
        stream_depth = depths and _streamed_depths(self.model)
        opened = [*collections, 'prediction_inputs'] if stream_depth else collections
        hidden, gathered = self._hidden_states(params, inputs, train, rngs, opened,
                                               {**packing, **replay}, layers)
        sown = gathered.get(ROUTER_COLLECTION, {}) if routing else None
        qk = gathered.get("qk", {}) if qk_stats else None
        kls = gathered.get(INDEXER_COLLECTION) if indexer else None
        kept = tuple(self._layer_states().layer_state(gathered, index) for index in layers)
        predict = self.token_accuracy if predict is None else predict
        table = self.fused_head(self.model, params) if z_loss is not None else None
        dtype = at_least_fp32(hidden.dtype)
        weights = self._row_weights(prepared, roles, dtype)
        total, (losses, predicted, log_z), whole = self._head_scores(
            params, hidden, targets, weights, prepared, table, predict=predict, z_loss=z_loss, logits=logits)
        weights = weights.astype(losses.dtype)
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
                    sown = {**(sown or {}), **depth_sown.get(ROUTER_COLLECTION, {})}
                if qk_stats:
                    qk = {**(qk or {}), **depth_sown.get("qk", {})}
                if indexer:
                    kls = {**(kls or {}), **depth_sown.get(INDEXER_COLLECTION, {})}
            for depth, state in enumerate(states, start=1):
                depth_weights = self._depth_weights(prepared, roles, losses.dtype, depth)
                depth_total = None
                if table is not None:
                    depth_total, (depth_losses, _, _) = fused_head_loss(
                        state, table, targets[:, depth:], depth_weights, tile=self.head_tile, predict=False)
                else:
                    depth_losses, _, _ = self.head_terms(self.model, params, state, targets[:, depth:],
                                                         predict=False)
                depth_scores.append((depth_losses, depth_weights, depth_total))
        return Scores(losses, weights, log_z, correct, hidden, kept, sown, depth_scores, qk, kls, total,
                      whole)

    def _head_scores(self, params, hidden, targets, weights, prepared, table, *, predict: bool,
                     z_loss: float | None, logits: bool):
        """The head's scores of `targets` from final states `hidden`: the weighted sum where a fused
        head took it (`table`), each target's loss, top-1 column and log partition, and the whole
        logits where asked; through the continuum level stepped in context (`adaptation`), the
        whole logits, the fused head or the tiled head."""
        if self.adaptation is not None:
            # The level steps window by window over these targets, weighted as the loss weighs them.
            adapted = adapted_scores(self.model, params, hidden, targets, weights,
                                     prepared.slice_tokens(stop=-1).token_fields, self.adaptation,
                                     predict=predict, logits=logits)
            return None, (adapted.losses, adapted.predicted, adapted.log_z), adapted.logits
        if logits:
            whole = model_logits(self.model, params, hidden)
            return None, logits_cross_entropy(whole, targets, predict=predict), whole
        if table is not None:
            total, scores = fused_head_loss(hidden, table, targets, weights, z_loss=float(z_loss or 0.0),
                                            tile=self.head_tile, predict=predict)
            return total, scores, None
        return None, self.head_terms(self.model, params, hidden, targets, predict=predict), None

    def _row_weights(self, prepared, roles, dtype):
        """Weight the targets the row itself scores (`scored_targets`), and
        with `loss_role` one role's targets alone."""
        if roles is not None and roles.shape != prepared.tokens.shape:
            raise ValueError(
                f"text_roles has shape {tuple(roles.shape)} for "
                f"{tuple(prepared.tokens.shape)} ids; the roles align with the input "
                "tokens, one per token")
        return self._scored(jnp, prepared.tokens, prepared.token_fields, roles).astype(dtype)

    def _depth_weights(self, prepared, roles, dtype, depth: int):
        """Weight the targets one prediction depth scores: depth d's state at
        p scores the target d further out (`scored_targets`)."""
        return self._scored(jnp, prepared.tokens, prepared.token_fields, roles, depth).astype(dtype)

    def _scored(self, xp, tokens, fields, roles, depth: int = 0):
        """The targets this objective scores at `depth` (`scored_targets`), its `loss_role`'s alone,
        in `xp`: the loss's weights on the device and the FLOP count's on the host read one rule."""
        mask, documents = token_layout(fields, xp=xp)
        return scored_targets(xp, tokens, self.pad_id, mask, documents, depth, roles=roles,
                              role=None if self.loss_role is None else int(self.loss_role),
                              blocks=fields.get(BLOCKS_FIELD))

    def _layer_states(self) -> LayerStates:
        """The model as the reader of its layers' states that a request for `layers` needs."""
        if not isinstance(self.model, LayerStates):
            raise TypeError(f"{type(self.model).__name__} keeps no layer states "
                            "(dew.nn.protocols.LayerStates), which reading named layers needs")
        return self.model

    def _hidden_states(self, params, inputs, train, rngs, collections: list[str],
                       packing: dict[str, jax.Array | Mapping[str, jax.Array]],
                       layers: Sequence[int] = ()):
        """Run the model over `inputs` and return what a loss scores (`chunked.final_read`).

        Beside them come whatever the open `collections` gathered, empty
        when none was opened, and the model's layers' states (`LayerStates`)
        when `layers` asks for any.
        """
        opened = [*collections, *self._layer_states().layer_collections] if layers else collections
        capture = self._layer_states().capture_layers if layers else False
        hidden = final_read(self.model, params, inputs, train=train, rngs=rngs, mutable=opened or False,
                            capture_intermediates=capture, **packing)
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
        `-scores.losses`. `temperature` divides the capped logits (`OutputTable.logits`).
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
            losses, _, _ = self.head_terms(self.model, params, scores.hidden, targets, predict=False,
                                           temperature=temperature)
            log_probs = -losses
        if support is not None:
            ids, columns = (jnp.asarray(value, jnp.int32) for value in support)
            # The id at column c is the target the state at c - 1 predicts.
            owners = jnp.where(columns > 0, columns - 1, -1)
            head = head_table(self.model, params)
            if head is not None:
                filtered, present = support_log_probs(scores.hidden, head, targets, ids, owners,
                                                      temperature=temperature)
            else:
                logits = model_logits(self.model, params, scores.hidden)
                filtered, present = logits_support_log_probs(logits, targets, ids, owners,
                                                             temperature=temperature)
            log_probs = jnp.where(present, filtered, log_probs)
        return log_probs

    def _batch_roles(self, batch, xp=jnp):
        """Read the batch's `text_roles` column as an `xp` array, or None on a
        run that counts every target.

        A `loss_role` run on a batch without the column raises, naming it.
        """
        if self.loss_role is None:
            return None
        if ROLES_KEY not in batch:
            raise ValueError(
                f"loss_role={self.loss_role.name} counts only "
                f"{self.loss_role.name.lower()} targets, but the batch carries "
                f"no {ROLES_KEY} column; train this objective on the chat data path")
        return xp.asarray(batch[ROLES_KEY])

    def _query_weights(self, prepared: ModelInputs, dtype):
        """Mark every real query whose router statistics and indexer KL count.

        A query counts when its token is real (`real_slots`), and not the pad id.
        """
        real = real_slots(prepared.token_fields, prepared.tokens.shape)[:, :-1]
        if self.pad_id is not None:
            real = real & (prepared.tokens[:, :-1] != self.pad_id)
        return real.astype(dtype)

    def _rows(self, tokens) -> tuple[jax.Array, jax.Array]:
        """Split a `[B, n + 1]` batch into its `n` inputs and their shifted targets, each batch at its
        own row length, as a schedule's stages lay theirs out, up to the objective's `context`."""
        if tokens.shape[-1] > self.context + 1:
            raise ValueError(
                f"rows of {tokens.shape[-1]} ids hold {tokens.shape[-1] - 1} inputs, more than the "
                f"{self.context}-token context of the objective's seq_len and the model's max_seq_len")
        return tokens[:, :-1], tokens[:, 1:]

    def preview(self, params, batch, step: Step, *, scored=None):
        """Sample the configured prompt once, then decode only on process zero.

        An objective whose EMA holds a frozen reference draws from the live
        policy instead, which is what `ema_is_reference` says.
        """
        settings = self.samples

        def setup():
            if settings is None or settings.max_new_tokens <= 0:
                return None
            weights = self.evaluation_variables(params, step)
            return self.policy(weights, settings.sampling), self._prompt, settings.max_new_tokens

        def generate(prepared: tuple[TextGeneration, ModelInputs, int]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            policy, prompt, max_new_tokens = prepared
            tokens = np.asarray(policy(prompt, max_new_tokens, key=step.key).host().tokens)
            ids = np.asarray(prompt.tokens)
            asked = real_slots(prompt.token_fields, ids.shape, xp=np)
            # The prompt's padding is no text; every drawn column is shown, as it was drawn.
            drawn = np.ones((ids.shape[0], tokens.shape[1] - ids.shape[1]), bool)
            return tokens, np.concatenate([asked, drawn], axis=1), ids[0][asked[0]]

        return text_preview("LM preview", setup, generate, None if settings is None else settings.decode)


def _steps_in_context(model) -> bool:
    """Whether `model` holds a level an objective steps in context (`SteppedInContext`)."""
    return isinstance(model, SteppedInContext) and model.in_context


class LMObjective(TokenScorer):
    """Trains a next-token model with shifted cross entropy, teacher-forced scoring and optional previews.

    `LMObjective(model, seq_len, ...)` scores `seq_len`-token rows, and the rows a chain lays out
    itself at another length up to the model's `max_seq_len`, as the stages of a context-extension
    schedule do (8k-token rows, then 64k), each length compiled once. Every
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

    `balance` trains the routers of a mixture as `ExpertBalance` describes (`dew.objectives.routing`): the
    aux-loss-free balancing bias, DeepSeek V2's balance loss and ST-MoE's router z-loss, each optional. The
    bias needs a mixture with `bias=True`. None, the default, adds no router term and leaves the routers'
    bias where it is.

    `loss_role` counts only the targets whose `text_roles` entry matches
    it, for SFT on the chat data path (`dew.data.chat.Role`). None
    counts every target the pad and segment weights keep. A batch
    without the `text_roles` column raises.

    `mtp_weight` scales DeepSeek V3's multi-token prediction loss
    (arXiv 2412.19437, eq. 24). The loss adds this weight times the
    mean cross entropy over the model's prediction depths, so the
    model needs `num_nextn_predict_layers` above zero. Unset leaves
    the term out and the depths untrained.

    A step whose optimizer clips by the attention logit maxima (`Step.qk_stats`,
    a QK-Clip such as `muonclip`'s) opens the `qk` collection where the
    attention layers record their per-head maxima, reports the largest as
    `qk/max_logit` and returns them for the clip. Otherwise the collection
    stays closed and costs no extra matmul.

    `indexer` trains DeepSeek-V3.2's lightning indexer, one
    `IndexerTraining` phase at a time, on a model whose mla mixer has the
    indexer. The warm-up phase keeps only the indexer in
    the `params` collection and the rest of the model under `frozen`,
    which is what `init` returns and a checkpoint stores. A
    starting tree for that phase may omit the indexer's weights, as
    a dense checkpoint does, and the fresh init fills them. The sparse
    phase reads a whole tree, either layout. The warm-up trains nothing
    but the indexer, so the terms of the main loss (`balance`, `mtp_weight`, `loss_role`, `z_loss`) are
    refused there.

    `z_loss` adds PaLM's auxiliary to the cross entropy: this
    coefficient times the squared log partition of every counted
    prediction (MaxText's `z_loss_multiplier`; PaLM used 1e-4). It
    keeps the logits from drifting away from normalised log
    probabilities. Zero adds nothing.

    `prefix` trains a prefix run: the blocks a layer that reads blocks sees whole
    (`AttentionMixer.bidirectional_blocks`). Each document's first `prefix` tokens, or with "prompt" its
    tokens before the first assistant token (a chat row's roles), form one block, and each token of a block
    sees the block's later tokens too. The loss scores no target in its source's block, which the state
    would read off; every other target counts as usual. Serving reads the same blocks (`TextGeneration`'s
    `prefix`): a prompt's first `prefix` tokens, or the whole prompt under "prompt", are one block. None, the
    default, trains causally. A model none of whose trunk layers reads blocks is refused, as is an int above
    `seq_len`, and `mtp_weight`, whose prediction depths read blocks beyond their source. A batch carrying
    `text_block_groups` (the data stage `dew.data.steps.blocks.Blocks`) trains the same way, whatever
    `prefix` says, so a run's blocks may be any runs of a row: a layer's attention chunk (`LayerKind.chunk`)
    would cut one, and the loss would drop a target its source never saw. A run reads blocks only where a
    trunk layer that reads them has no chunk, and a prefix run on any other model is refused where it is
    built.

    `model` may be a loaded source (`LMObjective(qwen, seq_len=512)`), and
    `variables` and `processor` override its own (`Objective.bind_model`).

    `token_accuracy` reports the argmax accuracy; False skips the pass
    over every logit it costs (0.77 ms of the head's 8.0 on a TPU v6e).

    `adaptation` steps the model's continuum level (`CausalTransformer.continuum`) in context by
    this loss's own next-token targets, Nested Learning's test-time training over the task loss
    (`dew.objectives.lm.adaptation.ChunkAdaptation`): every `window` tokens of a document, the level
    steps by `rate` times the summed gradient of their weighted losses, and the next window reads
    it; training learns the level's start through every step. Evaluation and validation score the
    same way. A preview, a rollout policy and a saved run's task would draw from the level's start,
    which no generation steps yet, so an objective with `adaptation` takes no `samples`, refuses
    `policy` and records no task.
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
        balance: ExpertBalance | None = None,
        loss_role: Role | None = None,
        prefix: int | Literal["prompt"] | None = None,
        mtp_weight: float | None = None,
        z_loss: float = 0.0,
        indexer: IndexerTraining | None = None,
        token_accuracy: bool = True,
        processor: Processor | None | Omitted = OMITTED,
        adaptation: ChunkAdaptation | None = None,
    ):
        """Build the objective; the class docstring describes each argument."""
        super().__init__(model, seq_len, head_chunks=head_chunks, head_tile=head_tile, samples=samples,
                         variables=variables, balance=balance, processor=processor)
        self.pad_id = pad_id
        self.token_accuracy = token_accuracy
        refuse_prefix(self.model, prefix, longest=seq_len)
        if prefix is not None and mtp_weight is not None:
            raise ValueError("a prefix run trains the trunk's targets alone: a prediction depth reads "
                             "blocks beyond its own source, so mtp_weight is refused with prefix")
        self.prefix = prefix
        _check_terms(self.model, mtp_weight=mtp_weight, z_loss=z_loss)
        if adaptation is not None:
            if not _steps_in_context(self.model):
                raise ValueError("adaptation steps a continuum level, which this model does not have "
                                 "(CausalTransformer.continuum, `SteppedInContext`)")
            if samples is not None:
                raise ValueError("adaptation scores through a level stepped in context, which a preview's "
                                 "generation does not step; train it without samples")
        self.adaptation = adaptation
        self.loss_role = loss_role
        self.mtp_weight = mtp_weight
        self.z_loss = z_loss
        self.indexer = indexer
        self.phase = _phase_filter(self.model, indexer, {
            "balance": balance, "mtp_weight": mtp_weight, "loss_role": loss_role,
            "z_loss": z_loss or None})
        self.ema = EMASpec.constant(ema_decay)

    @property
    def _warmup(self) -> bool:
        return self.indexer is not None and self.indexer.phase == "warmup"

    def scored_tokens(self, batch: Batch) -> np.ndarray:
        """Each row's targets the loss scores, or in an indexer's warm-up, which scores its KL alone,
        each row's real queries (`_query_weights`)."""
        if not self._warmup:
            return super().scored_tokens(batch)
        valid, _ = self._counted_tokens(batch)
        return np.sum(valid[:, :-1], axis=-1, dtype=np.int64)

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
            missing = leaves_by_path(drawn["params"]).keys() - leaves_by_path(tree["params"]).keys()
            if self._warmup and missing:
                outside = sorted("/".join(path) for path in missing if not _is_indexer(path))
                if outside:
                    raise ValueError(
                        "the warm-up initialises the indexer and nothing else, but the "
                        f"starting tree lacks {outside[:3]}{'...' if len(outside) > 3 else ''}")
                tree = merge(self.fresh_variables(key, None), tree)
        return tree if self.phase is None else freeze(tree, self.phase)

    def _text(self, batch) -> ModelInputs:
        """The batch's rows (`batch_text`) with the prefix's blocks where the batch carries none
        (`prefixed_fields`)."""
        prepared = batch_text(batch)
        fields = prefixed_fields(prepared.token_fields, jnp.shape(prepared.tokens), self.prefix,
                                 batch.get(ROLES_KEY))
        return (prepared if fields is prepared.token_fields
                else dataclasses.replace(prepared, token_fields=fields))

    def _token_geometry(self, batch: Batch) -> tuple[np.ndarray, Mapping[str, jax.Array | Coordinates]]:
        """`TokenScorer._token_geometry`, with the prefix's blocks where the batch carries none, so the
        host counts the targets the loss scores (`prefixed_fields`)."""
        tokens, fields = super()._token_geometry(batch)
        roles = batch.get(ROLES_KEY)
        return tokens, prefixed_fields(fields, tokens.shape, self.prefix,
                                       None if roles is None else np.asarray(roles), xp=np)

    def data(self, tokenizer=None):
        from dew.data.pipeline import ObjectiveData
        from dew.data.steps.blocks import Blocks
        from dew.data.steps.text import CountTargets, FixedTokens

        geometry = FixedTokens(self.seq_len, extra=1)
        blocks: tuple[Blocks, ...] = ()
        if self.prefix == "prompt":
            blocks = (Blocks(prompt=True, fills=True),)
        elif self.prefix is not None:
            blocks = (Blocks(prefix=self.prefix, fills=True),)
        if self.loss_role is None and self.pad_id is None and self.prefix is None:
            return ObjectiveData(geometry=geometry)
        counted = CountTargets(
            role=None if self.loss_role is None else int(self.loss_role), pad_id=self.pad_id
        )
        return ObjectiveData(train=(*blocks, counted), evaluate=(*blocks, counted), geometry=geometry)

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
        # The KL's own width, at least fp32: a float64 model's KL is summed in float64.
        width = at_least_fp32(jnp.result_type(*kls))
        weights = self._query_weights(prepared, width)
        total = jnp.sum(jnp.stack([
            self.row_mean(kl.astype(width), batch, weights=weights[:, :kl.shape[1]]).total for kl in kls]))
        mass = jax.lax.stop_gradient(self.row_mean(weights, batch).total)
        return total / len(kls), mass

    def _warmup_loss(self, params, batch, step: Step) -> tuple[Ratio, Aux[Variables]]:
        """Score the dense warm-up: the indexer's KL alone.

        The main loss is never scored, since nothing it reaches moves.
        """
        assert self.indexer is not None
        prepared = self._text(batch)
        inputs, _ = self._rows(prepared.tokens)
        collections = [INDEXER_COLLECTION] + (["qk"] if step.qk_stats else [])
        _, gathered = self._hidden_states(
            thaw(params), inputs, train=step.training, rngs=model_rngs(step.key, training=step.training),
            collections=collections, packing=prepared.slice_tokens(stop=-1).kwargs())
        statistics, kl = self._warmup_statistics(gathered[INDEXER_COLLECTION], prepared, batch)
        reported = {"indexer_kl": kl}
        qk = gathered.get("qk", {}) if step.qk_stats else None
        if step.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
        return statistics, Aux(reported, qk_stats=qk)

    def _warmup_statistics(self, sown, prepared: ModelInputs, batch: Batch) -> tuple[Ratio, jax.Array]:
        """The warm-up's loss, the indexer's weighted KL over the queries it
        counted, and the KL per query it reports."""
        assert self.indexer is not None
        total, mass = self._indexer_term(sown, prepared, batch)
        return Ratio(self.indexer.weight * total, mass), total / jnp.where(mass > 0, mass, 1)

    def fuses_head(self) -> bool:
        """The training loss takes a tiled head's gradient in its forward (`token_scores`' `z_loss`),
        outside an indexer's warm-up, which scores no head."""
        return not self._warmup

    def loss(self, variables, batch, step: Step) -> tuple[Ratio | RoutedStatistics, Aux[Variables]]:
        if self._warmup:
            return self._warmup_loss(variables, batch, step)
        statistics, aux, _ = self._scored_loss(variables, batch, step, train=step.training,
                                               summed=step.training)
        return statistics, aux

    def predict(self, params, batch, step: Step, *, train: bool,
                layers: Sequence[int] = ()) -> tuple[Ratio, Aux[Variables], Prediction]:
        """Return the loss with the logits, target weights and `layers` outputs behind it.

        A teacher compares these (`Objective.predict`). The logits are the whole
        `[B, seq_len, vocab]` fp32 tensor that the chunked loss never holds, because a
        distillation's KL reads every column. The balance loss's and the router z-loss's terms
        normalize per router, so they cannot share a distillation's token mass and are refused;
        a `balance_rate` alone balances without a loss term. The indexer warm-up
        scores no tokens, so it has nothing to distill.
        """
        if self._warmup:
            raise ValueError("the indexer warm-up scores no token, so it has no prediction")
        balance = self.balance
        if balance is not None and (balance.aux_loss_alpha is not None or balance.router_z_loss):
            raise ValueError(
                "the balance loss's and the router z-loss's router terms normalize per router, and a "
                "distillation mixes terms over the counted tokens; balance the "
                "student's routers with balance_rate instead")
        statistics, aux, scores = self._scored_loss(params, batch, step, train=train, layers=layers,
                                                    logits=True)
        assert isinstance(statistics, Ratio) and scores.logits is not None
        logits = constrain(scores.logits, LOGITS)
        return statistics, aux, Prediction(logits, scores.losses, scores.weights, scores.layers)

    def _scored_loss(self, params, batch, step: Step, *, train: bool, layers: Sequence[int] = (),
                     summed: bool = False, logits: bool = False, predict: bool | None = None
                     ) -> tuple[Ratio | RoutedStatistics, Aux[Variables], Scores]:
        """Compute the loss's statistics and reports, with the scores they came from.

        `summed` takes the cross entropies, the z-loss and the depths' as
        their weighted sums alone (`token_scores`'s `z_loss`), which a
        training step's loss is: a sum of a batch's rows, each counted once.
        `logits` scores them from the whole logits and keeps those
        (`token_scores`'s `logits`). `predict` is `token_scores`'.
        """
        prepared = self._text(batch)
        balance = self.balance
        summed = summed and VALID_ROWS not in batch
        scores = self.token_scores(
            params, prepared, train=train, rngs=model_rngs(step.key, training=train),
            routing=balance is not None,
            depths=self.mtp_weight is not None, roles=self._batch_roles(batch),
            qk_stats=step.qk_stats, indexer=self.indexer is not None, layers=layers,
            z_loss=self.z_loss if summed else None, logits=logits, predict=predict)
        losses, weights, log_z, correct, _, _, routing, depths, qk, kls, sums, _ = scores
        mass = jax.lax.stop_gradient(self.row_mean(weights, batch).total)
        prediction = Ratio(self.row_mean(losses, batch, weights=weights).total, mass)
        ce, _ = prediction.mean()
        reported = {"ce": ce, "perplexity": jnp.exp(ce)}
        if correct is not None:
            reported["token_accuracy"], _ = self.accuracy(correct, batch, weights).mean()
        if self.z_loss:
            # PaLM's auxiliary over the same counted targets as the cross
            # entropy, so one Ratio carries both.
            z_total = self.z_loss * self.row_mean(jnp.square(log_z), batch, weights=weights).total
            reported["z_loss"] = z_total / jnp.where(mass > 0, mass, 1)
            prediction = Ratio(prediction.total + z_total, mass)
        if sums is not None:
            # The head's sum carries the gradient; the per-token terms above report.
            prediction = Ratio(sums, mass)
        if self.mtp_weight is not None:
            # Depths retain the main target denominator and configured depth average.
            mtp_total = jnp.mean(jnp.stack([
                self.row_mean(depth_losses, batch, weights=depth_weights).total if depth_sum is None
                else depth_sum
                for depth_losses, depth_weights, depth_sum in depths]))
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
        statistics: Ratio | RoutedStatistics = prediction
        effects = None
        if balance is not None:
            queries = (self._query_weights(prepared, bool)
                       & (self.row_weights(batch, weights.shape[0]) > 0)[:, None])
            statistics, effects, metrics = balance.compose(
                params, prediction, routing or {}, batch, queries, prepared.token_fields)
            reported.update(metrics)
        if step.qk_stats:
            peak = _global_qk_max(qk)
            if peak is not None:
                reported["qk/max_logit"] = peak
        return statistics, Aux(reported, qk_stats=qk, effects=effects), scores

    def evaluate(self, params, batch, step: Step):
        """Score the complete batch teacher-forced, using EMA when present; the
        rows and roles are read outside the compiled scoring."""
        prepared = {TEXT_KEY: self._text(batch), ROLES_KEY: self._batch_roles(batch)}
        return super().evaluate(params, prepared, step)

    def _evaluation_scores(self, params, batch, step) -> TokenScores:
        scores = self.token_scores(params, batch[TEXT_KEY], roles=batch[ROLES_KEY], predict=True)
        assert scores.correct is not None
        return TokenScores(scores.losses, scores.weights, scores.correct)

    def _validation_scores(self, params, batch, step: Step) -> tuple[Ratio | RoutedStatistics, TokenScores]:
        """The validation loss and the batch's scores, from one scoring of it:
        the loss's, or in the warm-up, whose loss is the indexer's KL alone
        (`_warmup_loss`), the tokens' with the KL the trunk sows."""
        if self._warmup:
            prepared = self._text(batch)
            scores = self.token_scores(params, prepared, roles=self._batch_roles(batch), indexer=True,
                                       predict=True)
            statistics, _ = self._warmup_statistics(scores.indexer, prepared, batch)
        else:
            statistics, _, scores = self._scored_loss(params, batch, step, train=False, predict=True)
        assert scores.correct is not None
        return statistics, TokenScores(scores.losses, scores.weights, scores.correct)


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
        # A target of no weight is selected away before its product, so a padded target's infinite
        # loss adds nothing (`Objective.row_mean`'s `weights`).
        return float(np.sum(np.where(weights != 0, losses, 0.0) * weights)), float(np.sum(weights))

    merge = staticmethod(merge_totals)

    def finalize(self, accumulated: tuple[float, float]) -> float:
        total, count = accumulated
        if count == 0:
            raise ValueError("no counted target in the validation pass")
        return float(np.exp(total / count))
