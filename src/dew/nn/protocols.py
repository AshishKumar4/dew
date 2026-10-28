"""What a model can be asked for, as structural protocols.

An objective, a task or a server reads a model through these capabilities
instead of asking which class it is. A model has a capability by defining
the method; there is no Dew base class to inherit and no wrapper around a
model. A caller checks one with `isinstance(model, Logits)` and calls it
through Flax's own `apply`, with `rngs` and `mutable` passed to `apply`:

    logits = model.apply(variables, tokens, method='logits')
    states = model.apply(variables, tokens, train=True, method='hidden_states',
                         rngs={'dropout': key})
    table = model.apply(variables, method='output_table')

`Logits` and `HiddenStates` are token models' full-sequence reads, with no
cache: a call writes no `cache` collection, and `TokenModel` says whether
those states attend causally and names the id masked diffusion corrupts
tokens to. `DenoisingModel` is the call every image and video
denoiser already has, the raw network that `dew.diffusion.process.Denoiser`
wraps into a prediction a solver steps with. `AffineHead` and
`LogitsFromHidden` let a loss score the vocabulary without a second trunk
pass: `output_table` gives the matrix a tiled loss contracts in place of the
logits, and `logits_from_hidden` is the exact head for a head no matrix alone
gives.
`DenoisingModel` is an annotation only: every Flax module has a `__call__`, so
an `isinstance` check on it would hold for any model. `RequiresText`,
`IntervalModel` and `TimeScaled` are what a denoiser declares about the
text and time it reads.

The serving and training hooks are read off the model itself, not through
`apply`, from the layers it declares; a wrapper answers for its decoder.

The keywords a model takes past its tokens are `ModelKwarg` values:
`ModelInputs.kwargs()` gives a request's token fields and conditioning, and
callers add flags and absent values.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import PrecisionLike, VariableDict
from jax.typing import DTypeLike

from dew.nn.inputs import ModelKwarg
from dew.nn.precision import capped_logits, head_product, scaled_logits

if TYPE_CHECKING:
    from dew._records import JSON
    from dew.diffusion.process import DenoisingCondition
    from dew.inputs.encoders import ConditionWork
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.cross_attention import CrossAttention
    from dew.nn.dit import TextContext
    from dew.nn.dspark import DSpark
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.kv_cache import KVCache
    from dew.nn.mla import MLAMixer
    from dew.quantization import InputQuantizer

__all__ = [
    "AffineHead",
    "BlockDenoiser",
    "BlockDrafting",
    "Bounded",
    "CacheCapacity",
    "Capability",
    "Counted",
    "CountedCondition",
    "CountedConditionLayout",
    "CountedConditionedSpatial",
    "CountedDecisions",
    "CountedImages",
    "CountedLogits",
    "CountedLogitsTraining",
    "CountedMatrixUsage",
    "CountedMultimodal",
    "CountedParameters",
    "CountedPredictionParameters",
    "CountedPredictionTraining",
    "CountedPredictions",
    "CountedSequences",
    "CountedSpatial",
    "CountedTraining",
    "CountedVideo",
    "DecoderTraining",
    "Decoding",
    "DenoisingModel",
    "Exposing",
    "Generating",
    "HiddenStates",
    "IntervalModel",
    "Lacks",
    "LayerScalars",
    "LayerStates",
    "Layered",
    "Logits",
    "LogitsFromHidden",
    "Looping",
    "ModelKwarg",
    "OutputTable",
    "Predicting",
    "ProjectionGroup",
    "ProjectsParameters",
    "QuantizesInputs",
    "ReadsTrain",
    "Recomputing",
    "RematChoosing",
    "RequiresText",
    "Selective",
    "Serving",
    "TimeScaled",
    "TokenModel",
    "TritonGemm",
]


type Capability = Literal["logits", "causal", "bidirectional", "packing", "cache", "interval", "guided",
                          "features", "canvas"]
"""What a consumer can need of a model past its protocols' presence: its logits (`Logits`, or states
through a head), causal or bidirectional attention (`TokenModel.causal`), packed rows apart, a token
by token decode cache (`Decoding`), an interval's duration (`IntervalModel`), a guidance input, block
outputs a discriminator reads, or a canvas against a committed cache (`BlockDenoiser`)."""


class Lacks(TypeError, ValueError):
    """A consumer's refusal of a model for a capability it lacks, raised where the consumer first
    reads it: as it admits the model, or in the step that reads it. `capability` names what it
    lacks, so a caller tells the refusal from any other error by that and not by the message. A
    TypeError, as a model of the wrong kind, and a ValueError, as one configured otherwise, as numpy's
    `AxisError` is both of its errors."""

    def __init__(self, capability: Capability, message: str):
        super().__init__(message)
        self.capability = capability

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.capability, self.args[0])


@struct.dataclass
class OutputTable:
    """A vocabulary head as a matrix a loss contracts: `matrix` as the tree
    stores it, `[vocab, features]` when `vocab_major` and `[features, vocab]`
    otherwise, an optional fp32 `[vocab]` `scale` multiplying the product
    (nGPT's per-token logit gain) and `bias` added after it, the `softcap`
    applied last, and the `precision` the product runs at. `compute`, the
    model's compute dtype, decides the product's operands
    (`dew.nn.precision.head_product`; the states' dtype where None), and an
    `input_quantizer` rounds the states first as a checkpoint's head input
    quantization does.

    Contracting the model's final states with it (`logits`) reproduces the
    model's own logits, scale, bias, softcap and any adapter on the head
    included. Where no matrix does, `output_table` returns None, and the
    logits come from `logits_from_hidden` or `logits` instead.
    """

    matrix: jax.Array
    vocab_major: bool = struct.field(pytree_node=False)
    bias: jax.Array | None = None
    softcap: float | None = struct.field(pytree_node=False, default=None)
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)
    compute: DTypeLike | None = struct.field(pytree_node=False, default=None)
    input_quantizer: InputQuantizer | None = None
    scale: jax.Array | None = None

    def states(self, hidden: jax.Array) -> jax.Array:
        """`hidden` as the product reads it: through the input quantizer's QDQ."""
        return hidden if self.input_quantizer is None else self.input_quantizer.qdq(hidden)

    def logits(self, hidden: jax.Array, *, temperature: float = 1.0) -> jax.Array:
        """`hidden`'s `[..., vocab]` logits, at least fp32: the states through
        the input quantizer, the product (`head_product`), the scale and the
        bias (`scaled_logits`), the softcap, then `temperature`, which a
        sampler applies to the head's logits."""
        logits = head_product('...d,vd->...v' if self.vocab_major else '...d,dv->...v',
                              self.states(hidden), self.matrix, self.precision, compute=self.compute)
        return capped_logits(scaled_logits(logits, self.scale, self.bias), self.softcap, temperature)

    def columns(self, hidden: jax.Array, ids: Sequence[int] | jax.Array, *, temperature: float = 1.0,
                exact: bool = False) -> jax.Array:
        """`hidden`'s logits at the vocabulary columns `ids` alone, `[..., len(ids)]`: `logits`
        restricted to them by construction, the same states (the input quantizer's QDQ), product
        (`head_product`) at the table's precision and compute, scale and bias at `ids`
        (`scaled_logits`) and softcap, so a fixed set of a few columns costs no full-vocabulary
        product. `exact` computes the product in fp32 or wider at fp32 precision, whatever the
        table's compute, as a readout that reads a label's logit exactly does. (`dew.objectives.lm
        .chunked`'s support log probabilities restrict each position to its own ids, another
        contraction.) Static `ids` outside the vocabulary are refused (IndexError); traced ids are
        the caller's to bound."""
        if isinstance(ids, jax.Array):
            index: jax.Array | np.ndarray = ids
        else:
            index = np.asarray(ids, np.int64)
            vocab = self.matrix.shape[0 if self.vocab_major else 1]
            if index.size and (index.min() < 0 or index.max() >= vocab):
                raise IndexError(f"columns {index.tolist()} reach past a vocabulary of {vocab}")
        matrix = jnp.take(self.matrix, index, axis=0 if self.vocab_major else 1)
        states = self.states(hidden)
        subscripts = '...d,vd->...v' if self.vocab_major else '...d,dv->...v'
        if exact:
            wide = jnp.promote_types(states.dtype, jnp.float32)
            logits = head_product(subscripts, states.astype(wide),
                                  matrix.astype(jnp.promote_types(matrix.dtype, jnp.float32)),
                                  jax.lax.Precision.HIGHEST, compute=wide)
        else:
            logits = head_product(subscripts, states, matrix, self.precision, compute=self.compute)
        scale = None if self.scale is None else jnp.take(self.scale, index)
        bias = None if self.bias is None else jnp.take(self.bias, index)
        return capped_logits(scaled_logits(logits, scale, bias), self.softcap, temperature)

    def rows(self) -> jax.Array:
        """Each vocabulary token's `[vocab, features]` row: the matrix's, times
        the token's scale, whose dot product with a state is the token's logit
        before its bias and softcap, in the wider of the two dtypes. A reader
        of tokens' vectors (a decision head's lexical prior) takes these."""
        rows = self.matrix if self.vocab_major else self.matrix.T
        return rows if self.scale is None else rows * self.scale[:, None]


@runtime_checkable
class ReadsTrain(Protocol):
    """A module whose call takes `train`, for a stochastic layer of its own (a token mixer's or a
    feed-forward's dropout, a vision tower's drop path): its owner passes `train` where `reads_train`
    says the call reads it, whatever the module's family."""

    @property
    def reads_train(self) -> bool: ...


@runtime_checkable
class SteppedInContext(Protocol):
    """A model whose head reads parameters an objective steps in context, a window at a time
    (`dew.nn.continuum`, `LMObjective(adaptation=...)`): the level's parameters within a tree of its
    variables, None without a level; the tree with others in their place; and the two halves of the
    head the steps run between, the level on final states (`continuum_states`) and the head past it
    (`head_logits`). Its owner keeps the level's place in its tree, so an objective never reads it.
    `in_context` says whether the model holds a level, and `affine_head`, read from the bound model,
    whether `head_logits` is affine in its states, which a damped step's curvature bound needs
    (`ChunkAdaptation.damped`)."""

    @property
    def in_context(self) -> bool: ...

    def affine_head(self) -> bool: ...

    def in_context_level(self, variables: VariableDict) -> VariableDict | None: ...

    def with_in_context_level(self, variables: VariableDict, level: VariableDict) -> VariableDict: ...

    def continuum_states(self, hidden: jax.Array) -> jax.Array: ...

    def head_logits(self, states: jax.Array) -> jax.Array: ...


@runtime_checkable
class Logits(Protocol):
    """A token model's `[B, S, vocab]` fp32 logits over the whole sequence."""

    def logits(self, tokens: jax.Array, *, train: bool = False, **fields: ModelKwarg) -> jax.Array: ...


@runtime_checkable
class Counted(Protocol):
    """Analytical forward FLOPs summed over real row lengths, on the host.

    A token backbone counts its hidden states, excluding the vocabulary
    head, which an LM objective adds through `CountedLogits`; a decision
    objective instead adds its decision head's work. Lookup embeddings,
    rematerialization and padded tokens are excluded. Geometry beyond one
    sequence is supplied by the objective through a family's named count.
    A family that cannot count it raises `MissingFlopsCount`, never HLO.
    """

    def model_flops(self, lengths: np.ndarray) -> float: ...


@runtime_checkable
class CountedParameters(Protocol):
    """Forward work of learned matrix maps, separately from state/attention MACs.

    An adapter freezes these base matrices: their training work is twice
    forward work, while its trainable factors cost three times their own
    forward work. Nonparameter attention/state contractions remain separate.
    """

    def parameter_flops(self, lengths: np.ndarray) -> float: ...


@runtime_checkable
class CountedMatrixUsage(Protocol):
    """A model's declared call geometry for an adapted native matrix path."""

    def matrix_usage(self, path: tuple[str, ...], lengths: np.ndarray, *,
                      logits: bool = False) -> np.ndarray: ...


@runtime_checkable
class CountedTraining(Protocol):
    """An adapted backbone's training work, including its factors, not 3*forward."""

    def training_flops(self, lengths: np.ndarray) -> float: ...


@runtime_checkable
class CountedLogitsTraining(CountedTraining, Protocol):
    """An adapted LM head's training work, separately from its backbone."""

    def logits_training_flops(self, lengths: np.ndarray) -> float: ...


@runtime_checkable
class CountedLogits(Counted, Protocol):
    """A counted token backbone with separate vocabulary-head work.

    `lengths` counts predicted real tokens. A plain head contributes
    2*width*vocabulary per token, whether its matrix is tied or tiled.
    """

    def logits_flops(self, lengths: np.ndarray) -> float: ...


@runtime_checkable
class CountedCondition(Protocol):
    """A condition encoder owns its work and the geometry its denoiser reads."""

    def conditioning_flops(self, tokens: Mapping[str, np.ndarray]) -> ConditionWork: ...


@runtime_checkable
class CountedConditionLayout(Protocol):
    """A denoiser selects the valid keys or physical positions its forward reads."""

    def condition_lengths(self, lengths: np.ndarray, spans: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class CountedPredictions(Protocol):
    """The extra decoder depths and heads trained beside a language model's trunk."""

    num_nextn_predict_layers: int

    def prediction_flops(self, lengths: np.ndarray, *,
                          prediction_lengths: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
                          ) -> float: ...


@runtime_checkable
class CountedPredictionParameters(CountedPredictions, Protocol):
    """Prediction depths declare their learned maps and adapted scopes separately."""

    def prediction_parameter_flops(self, lengths: np.ndarray, *,
                                    prediction_lengths: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
                                    ) -> float: ...

    def prediction_matrix_usage(self, path: tuple[str, ...], lengths: np.ndarray, *,
                                  prediction_lengths: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
                                  ) -> np.ndarray: ...


@runtime_checkable
class CountedPredictionTraining(CountedPredictions, Protocol):
    """An adapted prediction stack's frozen-base and trainable-factor training work."""

    def prediction_training_flops(self, lengths: np.ndarray, *,
                                   prediction_lengths: Sequence[tuple[np.ndarray, np.ndarray]] | None = None
                                   ) -> float: ...


@runtime_checkable
class CountedImages(Protocol):
    """A patch-token model's forward work with channel/conditioning geometry.

    Models whose patch projection infers its channels at init cannot derive
    that map's input width from token lengths alone. The objective supplies
    `input_channels` and the pooled text width (None for no text).
    """

    def image_flops(self, lengths: np.ndarray, input_channels: int,
                    text_features: int | None = None) -> float: ...

    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray: ...


@runtime_checkable
class CountedSequences(Protocol):
    """A joint/cross-attention backbone with separate image and text geometry.

    The objective supplies independent lengths and the actual image, context
    and pooled widths, because Dense input axes are inferred from the arrays.
    """

    def multisequence_flops(self, lengths: np.ndarray, context_lengths: np.ndarray, *,
                            input_channels: int, text_features: int,
                            pooled_features: int | None = None) -> float: ...

    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray: ...


@runtime_checkable
class CountedMultimodal(Protocol):
    """A joint-stream model whose input maps infer their widths at init."""

    def multimodal_flops(self, lengths: np.ndarray, context_lengths: np.ndarray,
                         input_channels: int, text_features: int) -> float: ...

    def image_lengths(self, sample_shape: tuple[int, ...], rows: int) -> np.ndarray: ...


@runtime_checkable
class CountedSpatial(Protocol):
    """A convolutional denoiser with a resolution-specific stage walk."""

    def spatial_flops(self, sample_shape: tuple[int, ...], rows: int,
                       context_lengths: np.ndarray | None = None,
                       text_features: int | None = None) -> float: ...


@runtime_checkable
class CountedConditionedSpatial(Protocol):
    """A spatial model also consumes the encoder's pooled and time-id geometry."""

    def conditioned_spatial_flops(self, sample_shape: tuple[int, ...], rows: int,
                                   condition: ConditionWork) -> float: ...


@runtime_checkable
class CountedVideo(Protocol):
    """A factorized video model with independent frame and spatial axes."""

    def video_lengths(self, sample_shape: tuple[int, ...], rows: int) -> tuple[np.ndarray, np.ndarray]: ...

    def spatiotemporal_flops(self, frames: np.ndarray, patches: np.ndarray, input_channels: int,
                             text_features: int | None = None) -> float: ...


@runtime_checkable
class CountedDecisions(Protocol):
    """A decision head counts the layout's real questions and options with its rows."""

    def decision_flops(self, lengths: np.ndarray, questions: np.ndarray, options: np.ndarray, *,
                       pooled: np.ndarray | None = None) -> float:
        """The head's work over rows of `lengths` tokens, their real questions and options, its spans
        pooled over `pooled` tokens a row where they are not the row's (a context's)."""
        ...


@runtime_checkable
class HiddenStates(Protocol):
    """A token model's `[B, S, features]` final states, the head's input."""

    def hidden_states(self, tokens: jax.Array, *, train: bool = False,
                      **fields: ModelKwarg) -> jax.Array: ...


@runtime_checkable
class TokenModel(Protocol):
    """Whether a token model's `hidden_states` attend causally (each position
    to itself and the ones before it) or to the whole sequence, and the
    vocabulary id masked diffusion corrupts tokens to, None without one."""

    @property
    def causal(self) -> bool: ...

    @property
    def mask_token_id(self) -> int | None: ...


class DenoisingModel(Protocol):
    """A diffusion network's output for `sample` at `time`, given its conditions:
    arrays, encoded text (`TextContext`) or a process's `DenoisingCondition`."""

    def __call__(self, sample: jax.Array, time: jax.Array, *, train: bool = False,
                 **conditions: ModelKwarg | TextContext | DenoisingCondition) -> jax.Array: ...


@runtime_checkable
class AffineHead(Protocol):
    """A model whose logits may be one matrix product of its final states.

    It reads the variables `apply` binds, so an adapter's frozen base weights
    are there too, and `apply` copies nothing: under a gradient the matrix is
    the caller's parameter, which a loss keeps for its backward pass.
    """

    def output_table(self) -> OutputTable | None: ...


def head_table(model: nn.Module, variables) -> OutputTable | None:
    """`model`'s head over `variables` as one matrix (`AffineHead.output_table`),
    or None for a model whose head no matrix alone is, or that names none: the
    one checked read of a model's output table, which a loss and a readout share."""
    table = model.apply(variables, method="output_table") if isinstance(model, AffineHead) else None
    if table is not None and not isinstance(table, OutputTable):
        raise TypeError(f"{type(model).__name__}.output_table gives a {type(table).__name__}, "
                        f"not an OutputTable")
    return table


def label_logits(model: nn.Module, variables, answers: jax.Array, ids: Sequence[Sequence[int]]) -> jax.Array:
    """`model`'s own logits `[B, N, K, V]` at each of the `[K, V]` label `ids` of each answer state of
    `answers` `[B, N, features]`, a class read as its label's tokens' logits (a verbalizer): its output
    table at the labels' columns alone, computed in fp32 (`OutputTable.columns`), where one matrix is
    its head (`head_table`); else its exact head over the answer states, a few rows
    (`LogitsFromHidden`), the labels' columns taken. No product is rebuilt from the vocabulary's rows,
    which carry no bias and no softcap."""
    held = np.asarray(ids, np.int64)
    table = head_table(model, variables)
    if table is not None:
        return table.columns(answers, held.reshape(-1).tolist(), exact=True).reshape(
            *answers.shape[:2], *held.shape)
    if not isinstance(model, LogitsFromHidden):
        raise TypeError(f"a readout of labels reads its backbone's exact head, which a "
                        f"{type(model).__name__} does not give (AffineHead, LogitsFromHidden)")
    full = jnp.asarray(model.apply(variables, answers, method="logits_from_hidden"))
    picked = jnp.take(full, jnp.asarray(held.reshape(-1)), axis=-1)
    return picked.astype(jnp.promote_types(picked.dtype, jnp.float32)).reshape(
        *answers.shape[:2], *held.shape)


@runtime_checkable
class VocabularyRows(Protocol):
    """A model whose head scores its final states against one row per vocabulary token: each
    token's `[vocab, features]` row as `OutputTable.rows` gives it, read from the bound variables,
    where no matrix alone is the head too (an `AffineHead` returning None): past a nonlinear head
    transform the matrix's rows, with an adapter on the head its rows as the adapter updates them.
    A reader of tokens' vectors, a decision head's lexical prior, takes these."""

    def vocabulary_rows(self) -> jax.Array: ...


@runtime_checkable
class QuantizesInputs(Protocol):
    """A model computing under a checkpoint's input quantization
    (`dew.quantization.InputQuantization`): the quantizer a bound
    module's input takes, which a head read as a table (`OutputTable`) applies
    as the module's own call does."""

    def input_quantizer(self, module: nn.Module) -> InputQuantizer | None: ...


@runtime_checkable
class LogitsFromHidden(Protocol):
    """The exact head from final states, for scoring states a loss already holds."""

    def logits_from_hidden(self, hidden: jax.Array) -> jax.Array: ...


@runtime_checkable
class CacheCapacity(Protocol):
    """A decoder at another decode-cache capacity: the same variables, and
    the same draws for a request that fits."""

    def with_cache_capacity(self, capacity: int) -> Self: ...


@runtime_checkable
class Looping(Protocol):
    """A model whose layer stack can run a block of it several times over its own output,
    and the same model at `steps` passes of its block (None keeps its count), or of the block
    `layers`, [first, end), which gives a model with no loop one; None where they cannot.
    `backprop_steps` trains through the final passes and `remat` checkpoints each pass;
    None keeps an existing loop's value, or a fresh loop's default."""

    def with_passes(self, steps: int | None = None, layers: tuple[int, int] | None = None, *,
                    backprop_steps: int | None = None, remat: bool | None = None) -> Self | None: ...


@runtime_checkable
class RematChoosing(Protocol):
    """A decoder whose layers start at a chosen `RematPolicy`, or None for no remat.

    CausalTransformer and the multimodal and DiffusionGemma models that wrap
    it answer this capability. Image diffusion models choose a `RematChoice`
    instead of named decoder residuals, so they do not. A policy outside
    `DECODER_REMAT` is valid, but `Recomputing.recompute_more` cannot climb
    from it.
    """

    def with_remat(self, policy: RematPolicy | None) -> Self: ...


@runtime_checkable
class CrossAttending(Protocol):
    """A decoder whose layers can read a conditioning sequence through gated
    cross-attention (`dew.nn.cross_attention`): CausalTransformer answers this."""

    cross_attention: CrossAttention | None

    @property
    def context_width(self) -> int:
        """The width of the tokens of a context its cross-attention layers read."""
        ...

    @property
    def context_dtype(self) -> jnp.dtype:
        """The dtype a context's tokens enter its cross-attention layers in."""
        ...

    def cross_flops(self, lengths, context_lengths) -> float:
        """The cross-attention layers' forward work over rows of `lengths` tokens reading contexts
        of `context_lengths` tokens."""
        ...

    def token_embeddings(self, tokens: jax.Array) -> jax.Array:
        """Each token's row of the model's own table, unscaled and without positions: a context's
        tokens as a cross-attention reads them by the table (`dew.nn.cross_attention.encode_context`)."""
        ...

    def with_cross_attention(self, cross: CrossAttention | None) -> Self: ...


@runtime_checkable
class ProjectsParameters(Protocol):
    """A model whose training keeps a constraint on the parameters it stores.

    `projected` returns the stored parameters a training step reads and
    updates. The trainer calls it at each optimizer commit: first with
    `updated=False` on the parameters the step's update applies to, which
    the optimizer sees (Gemma 3n's AltUp clamps its prediction coefficients
    there, as the reference clamps the stored weight in place before its
    forward); then with `updated=True` on the updated parameters, which the
    step stores (a normalized transformer puts its matrices back on the unit
    sphere there). The tree a run starts from is projected once with
    `updated=True`, drawn or held, so a projection leaves a tree that already
    satisfies the constraint as it is. `params` is the trainable tree,
    which may hold a part of the model's parameters (an adapter's frozen
    base is absent): a model projects the leaves present and keeps the rest.
    """

    def projected(self, params: VariableDict, *, updated: bool) -> VariableDict: ...


@runtime_checkable
class Recomputing(Protocol):
    """A model one remat rung up (None at the top), its rung as a checkpoint
    records it, and the model at a record's rung where that is above its
    own: a resumed run never compiles a lighter rung than it trained on.
    `recompute_half` is the model one rung up where that rung recomputes only
    the few values it names, a half rung the planner tries at each split
    (`dew.training.planning.FitSearch`), else None."""

    def recompute_record(self) -> JSON: ...

    def recompute_more(self) -> Self | None: ...

    def recompute_half(self) -> Self | None: ...

    def restore_recompute(self, record: JSON) -> Self: ...


@dataclasses.dataclass(frozen=True)
class ProjectionGroup:
    """Projections under `path` in `params` that read one input: `members`,
    `widths` wide, concatenate on the output axis into `packed`, which their
    module reads in their place, one product for a decode step."""

    path: tuple[str, ...]
    packed: str
    members: tuple[str, ...]
    widths: tuple[int, ...]

    @classmethod
    def of(cls, path: tuple[str, ...], packed: str, members: tuple[str, ...],
           projections: Mapping[str, tuple[int, int]]) -> Self:
        """The group of `members`, their widths read from the module's own
        `projections()`, `{name: (inputs, outputs)}`, the geometry its setup builds."""
        return cls(path, packed, members, tuple(projections[name][1] for name in members))

    def split(self, out: jax.Array) -> tuple[jax.Array, ...]:
        """The members' outputs from the packed projection's `out`, in member order."""
        ends = np.cumsum(self.widths)[:-1]
        return tuple(jnp.split(out, [int(end) for end in ends], axis=-1))

    def held(self, params: Mapping) -> bool:
        """Whether `params`, the parameters at `path`, hold the group packed or every member."""
        return self.packed in params or all(member in params for member in self.members)


type Routes = tuple[jax.Array, jax.Array | None]
"""A routing replay for one layer: `[..., top_k]` expert ids the rollout
engine used, and `[...]` booleans marking the tokens its record covers (None
for all of them).

Routing replay (R3, arXiv 2510.11370; verl 12ebe0c `router_replay_patch.py`)
trains a mixture on the engine's experts, so a top-k flip between the
engine's numerics and the trainer's cannot move a token to experts that
never produced its sample. Only the selection is replayed; the gate weights
come from this forward's scores, so the router keeps its gradient. An
uncovered token keeps the router's choice, as verl's replay mask does, and
the sown `indices` are the replayed ones, so a balancing bias counts where
the tokens went. The stack hands each layer its slice of a `[B, S, layers,
top_k]` record (`CausalTransformer.hidden_states`)."""


@runtime_checkable
class RoutesBlocks(Protocol):
    """A routed feed-forward that chooses experts block by block (LLaDA2.2's), each block of
    `routing_block` places sharing a shortlist when that is set: its caller hands it each token's
    block (`dew.nn.inputs.routing_blocks`)."""

    @property
    def routing_block(self) -> int | None: ...


@runtime_checkable
class RoutesExperts(Protocol):
    """A module whose call takes `routes`, a layer's replayed experts, because
    it routes over experts: a routed feed-forward, or a token mixer holding one
    (an MLP in the mixer's slot)."""

    @property
    def routes_experts(self) -> bool: ...

    def __call__(self, x: jax.Array, *, routes: Routes | None = None) -> jax.Array: ...


@runtime_checkable
class ProjectionSites(Protocol):
    """A bound module's packed groups, its own and its children's."""

    def projection_groups(self) -> tuple[ProjectionGroup, ...]: ...


def held_groups(inner: Serving, name: str, variables: Mapping[str, Mapping]) -> tuple[ProjectionGroup, ...]:
    """The groups `inner`, held under `name` in a wrapper's variables, declares
    (`Serving.inference_projection_groups`), at their paths in the wrapper's tree: what a module
    around a decoder answers as its own groups, a multimodal model's around its language model or
    a user's around Dew's decoder."""
    held = {collection: tree[name] for collection, tree in variables.items() if name in tree}
    return tuple(dataclasses.replace(group, path=(name, *group.path))
                 for group in inner.inference_projection_groups(held))


def declared_groups(*modules: nn.Module) -> tuple[ProjectionGroup, ...]:
    """The groups the bound `modules` declare, in order; a module declaring none adds none."""
    return tuple(group for module in modules if isinstance(module, ProjectionSites)
                 for group in module.projection_groups())


@runtime_checkable
class Serving(Protocol):
    """What a server reads off a decoder. `inference_projection_groups` names
    the packed groups `variables` hold, packed or as every member (the packer
    checks the members concatenate; a model that does not decode
    autoregressively names none). It reads which parameters the tree holds and
    their shapes, never their values or placement, and the model is an unbound,
    hashable module, so its answer is kept per model and tree shape
    (`dew.inference.projections`) and is called over `jax.ShapeDtypeStruct`
    leaves. `cache_rebuild_position` is where cached
    keys go stale and the prefix is recomputed (LongRoPE's long factors,
    Phi-3), or None. `mixed_admission_refusal` says why its layers would not
    run a mixed call (`dew.nn.inputs.Admitted`) as separate decode and
    prefill calls would, or None. `split_prefill_refusal` says why it
    prefills each prompt whole, a prompt taken in pieces or after a prefix
    another request cached computing something its whole prefill does not,
    and what serves those modes instead, or None."""

    def inference_projection_groups(self, variables: Mapping[str, Mapping]
                                    ) -> tuple[ProjectionGroup, ...]: ...

    @property
    def cache_rebuild_position(self) -> int | None: ...

    def mixed_admission_refusal(self) -> str | None: ...

    def split_prefill_refusal(self) -> str | None: ...


@runtime_checkable
class BlockReading(Protocol):
    """A decoder whose trunk layers may read a block whole (`AttentionMixer.bidirectional_blocks`: a block's
    tokens see each other both ways). `block_chunks` lists each trunk layer that reads blocks by its attention
    chunk, None where it has none. A prefix run needs one: its blocks reach a model call as `block_groups`,
    and a layer that reads none computes them causally."""

    def block_chunks(self) -> tuple[int | None, ...]: ...


def block_chunks_of(model: nn.Module) -> tuple[int | None, ...]:
    """The attention chunk of each trunk layer that reads blocks whole, None where a layer has none; empty for
    a model that says nothing, or whose layers read none (`BlockReading`)."""
    return model.block_chunks() if isinstance(model, BlockReading) else ()


def whole_block(chunks: tuple[int | None, ...]) -> bool:
    """Whether a layer that reads blocks reads every block whole: one with no attention chunk. A chunk is
    a run of places counted from 0 (`chunk_mask`), and a block's places start wherever its tokens sit in
    their document, a window that continues a document or a prompt after cached tokens included, so a
    chunk of any length cuts a block that crosses one of its multiples. The loss drops every target inside
    a block, so a layer must read the block's every token for its targets to be ones its source saw."""
    return any(chunk is None for chunk in chunks)


def refuse_block_reading(model: nn.Module, reading: str) -> None:
    """Refuse `reading` blocks (a prefix run's, a batch's) where `model` cannot read them whole: it needs a
    trunk layer that reads blocks (`block_chunks_of`) with no attention chunk (`whole_block`). The loss drops
    the targets inside a block, which the trunk sees only where a layer reads the whole block. A prefix run
    and a batch that carries its own blocks are both admitted by this one rule."""
    chunks = block_chunks_of(model)
    if not chunks:
        raise ValueError(f"{reading} needs a trunk layer that reads blocks "
                         f"(AttentionMixer.bidirectional_blocks), and {type(model).__name__} has none")
    if not whole_block(chunks):
        raise ValueError(f"{reading} is read whole only by a layer with no attention chunk, and every layer "
                         f"of {type(model).__name__} that reads blocks has one ({list(chunks)}), which cuts "
                         "a block that crosses its multiples")


def refuse_prefix(model: nn.Module, prefix: int | Literal["prompt"] | None, *, longest: int | None) -> None:
    """Refuse a prefix run the model or its values cannot take. A prefix is "prompt" or a positive number of
    tokens, at most `longest` where a longest row is declared, and its blocks are read as a batch's are
    (`refuse_block_reading`)."""
    if prefix is None:
        return
    if prefix != "prompt" and (type(prefix) is not int or prefix < 1):
        raise ValueError(f"prefix is 'prompt' or a positive number of tokens, got {prefix!r}")
    if isinstance(prefix, int) and longest is not None and prefix > longest:
        raise ValueError(f"prefix {prefix} is more tokens than the longest row, {longest}")
    refuse_block_reading(model, "a prefix's blocks")


@runtime_checkable
class ShiftsLogits(Protocol):
    """A masked-diffusion model that declares how its logits line up with its inputs: with
    `shifted_logits`, position i's prediction is position i-1's logits (Dream-v0's). A model that
    declares nothing predicts each position's own."""

    @property
    def shifted_logits(self) -> bool: ...


@runtime_checkable
class PositionCapacity(Protocol):
    """A model that declares the most places a row may hold that its positions cover, which a row longer
    than that would read past (a learned table's rows), None where its positions cover any (rotary,
    ALiBi): what a training row's length is checked against, apart from its decode cache's capacity
    (`Bounded.max_seq_len`). A model that declares none is checked for none."""

    @property
    def position_capacity(self) -> int | None: ...


@runtime_checkable
class Bounded(Protocol):
    """A model that declares its vocabulary and cache capacity, which the host
    checks read to refuse an id outside the vocabulary or a request that would
    overflow the cache; a model that declares neither is checked for neither."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int | None: ...


@runtime_checkable
class Decoding(Protocol):
    """A decoder that continues its rows token by token: `init_cache` opens the
    cache its decode calls (`decode=True`) write, a position a call. A caller
    that appends with those calls itself, past `generate`'s and the server's
    host checks, reads `dew.nn.kv_cache.overflowed` after them."""

    def init_cache(self, batch_size: int) -> None: ...


@runtime_checkable
class Layered(Protocol):
    """A decoder that declares its decode cache's layout, as `CausalTransformer` does, and gives
    the same model over another (`with_kv_cache`): a wrapper answers for the decoder that holds
    the cache."""

    @property
    def kv_cache(self) -> KVCache: ...

    def with_kv_cache(self, layout: KVCache) -> Self: ...


@runtime_checkable
class Exposing(Protocol):
    """A decoder that hands back its hidden states beside its logits, which a
    drafting strategy seeds from; without them a model decodes but cannot draft."""

    def states_and_logits(self, tokens: jax.Array, **kwargs: jax.Array | bool | None
                          ) -> tuple[jax.Array, jax.Array]: ...


@runtime_checkable
class Selective(Protocol):
    """A decoder that scores one position per row instead of them all, so its
    prefill runs the head only on the slot the first draw reads."""

    def states_and_logits_at(self, tokens: jax.Array, slots: jax.Array, **kwargs: jax.Array | bool | None
                             ) -> tuple[jax.Array, jax.Array]: ...


@runtime_checkable
class Predicting(Protocol):
    """A decoder that declares how many multi-token prediction depths it carries."""

    @property
    def num_nextn_predict_layers(self) -> int: ...


@runtime_checkable
class BlockDrafting(Protocol):
    """A decoder that may carry a block drafter: DeepSeek's DSpark (V4.1, V4-Flash-0731),
    which drafts a whole block per pass from the context its target layers
    record instead of chaining prediction depths."""

    @property
    def dspark(self) -> DSpark | None: ...


@runtime_checkable
class Generating(Protocol):
    """A decoder that says, after examining the layers it builds, why a
    generation loop cannot continue its rows token by token, or None; a
    sampler refuses with that reason without knowing where a family keeps
    the layers that cause it."""

    def generation_refusal(self) -> str | None: ...


@runtime_checkable
class DecoderTraining(Predicting, Protocol):
    """What a decoder trains beside its next token: its multi-token prediction
    depths (arXiv 2412.19437, section 2.2; `apply`'s `method='mtp_hidden_states'`
    and `'mtp_logits'`, `Predicting`), which with `mtp_hyper_connections` set
    read the residual streams a forward sows under `prediction_inputs/states`,
    and the MLA mixers it declares with DeepSeek's lightning indexer."""

    @property
    def mtp_hyper_connections(self) -> HyperConnections | None: ...

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]: ...


@runtime_checkable
class LayerStates(Protocol):
    """A model whose forward keeps its layers' residual states, each layer's output before the
    final norm, for a reader of a middle layer's features: applied with
    `capture_intermediates=model.capture_layers` and `model.layer_collections` mutable, the
    collections it returns hold them, and `model.layer_state(collections, index)` reads layer
    `index`'s (0-based; -1 is the last). A layer a loop runs more than once is read at its last
    pass, the state the layers after it read."""

    layer_collections: tuple[str, ...]

    def capture_layers(self, module: nn.Module, method: str) -> bool: ...

    def layer_state(self, collections: Mapping[str, Mapping], index: int) -> jax.Array: ...


@runtime_checkable
class TritonGemm(Protocol):
    """Whether a mixer of a model keeps XLA's Triton GEMM fusions (`MixerBase.keeps_triton_gemm`)."""

    @property
    def keeps_triton_gemm(self) -> bool: ...


@runtime_checkable
class BlockDenoiser(Protocol):
    """A diffusion language model that generates a canvas at a time, as
    DiffusionGemma does. `init_cache` allocates `batch_size` rows of a
    prefix cache, and `encode` appends clean tokens to it: a prompt, with its
    `ModelInputs` fields, or a committed canvas. Calling the model refines a
    `canvas_length` canvas against that frozen cache and returns its `[B,
    canvas_length, vocab_size]` logits, self-conditioned on the previous step's
    (`self_conditioning_logits`, a zero signal in rows whose
    `self_conditioning_mask` is false); `max_seq_len` bounds the prompt and
    its canvases. A canvas is refined and committed at the logical `positions`
    that continue its prompt's, None where the prompt gives none and the
    cache's cursor places it. A causal language model scores tokens too, but
    runs none of these, so it is not one."""

    @property
    def canvas_length(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int: ...

    def init_cache(self, batch_size: int) -> None: ...

    def encode(self, tokens: jax.Array, *, positions: jax.Array | None = None) -> jax.Array: ...

    def __call__(self, tokens: jax.Array, *, self_conditioning_logits: jax.Array | None = None,
                 self_conditioning_mask: jax.Array | None = None,
                 positions: jax.Array | None = None) -> jax.Array: ...


@runtime_checkable
class LayerScalars(Protocol):
    """A model with per-layer output scalars that a fine-tune trains, as
    DiffusionGemma's published SFT does: `with_trainable_layer_scalars` is
    the model that trains them, and `trainable_variables` the source's tree
    as that model reads it."""

    def with_trainable_layer_scalars(self) -> Self: ...

    def trainable_variables(self, variables: VariableDict) -> VariableDict: ...


@runtime_checkable
class RequiresText(Protocol):
    """A denoising model that cannot run without text, which reaches it under
    the keyword `text_keyword` on every call. Any other runs unconditionally."""

    @property
    def text_keyword(self) -> str: ...


@runtime_checkable
class IntervalModel(Protocol):
    """A denoising model that, with its `interval` field set
    (`model.clone(interval=True)`), embeds the `duration` an interval process
    hands it beside its time, a missing one as zero; unset, it refuses one."""

    @property
    def interval(self) -> bool: ...


@runtime_checkable
class TimeScaled(Protocol):
    """A denoising model whose `time_scale` field is its time's unit: at scale
    s on time t (and duration d) it computes what it does at s / k on k t (and
    k d). A loss that differentiates it in time needs the scale small
    (`dew.objectives.diffusion.few_step.SMOOTH_TIME_SCALE`). A table of time
    features in the variables, as `FourierEmbedding`'s, is drawn at `init`'s
    scale and loaded variables keep theirs."""

    @property
    def time_scale(self) -> float: ...
