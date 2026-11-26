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
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import Dtype, PrecisionLike, VariableDict
from jax.typing import DTypeLike

from dew.batch import Coordinates
from dew.nn.inputs import ModelKwarg
from dew.nn.precision import EXACT_FLOAT32, capped_logits, head_product, scaled_logits

if TYPE_CHECKING:
    from flax.linen.module import Interceptor

    from dew._records import JSON
    from dew.lora import _Bound
    from dew.nn.attention_residuals import AttentionResiduals
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.backbones.decoder_stack import DecoderBank, Loop, StackView
    from dew.nn.continuum import HeadTerms, LevelTerms
    from dew.nn.cross_attention import CrossBlocks
    from dew.nn.cross_pass import CrossPass
    from dew.nn.dex import DifferentialExtension
    from dew.nn.dit import TextContext
    from dew.nn.flops import ConditionWork
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.inputs import DenoisingCondition
    from dew.nn.kv_cache import KVCache
    from dew.nn.norms import NormBounds
    from dew.quantization import InputQuantizer

__all__ = [
    "Adapted",
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
    "DrawsAlone",
    "Exposing",
    "Generating",
    "HiddenStates",
    "HookedDecoder",
    "IntervalModel",
    "Lacks",
    "LayerScalars",
    "LayerStates",
    "Layered",
    "Logits",
    "LogitsFromHidden",
    "Looping",
    "MediaWrapper",
    "ModelKwarg",
    "OutputTable",
    "Predicting",
    "ProjectionGroup",
    "ProjectsParameters",
    "QuantizesInputs",
    "ReadsClips",
    "ReadsTrain",
    "Recomputing",
    "RematChoosing",
    "RequiresText",
    "Selective",
    "Serving",
    "StagedStacks",
    "TimeScaled",
    "TokenModel",
    "TritonGemm",
    "VocabularyRows",
    "label_logits",
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
class Projection:
    """A bias-free linear map the final states take before a vocabulary matrix meets them
    (`OutputTable.projection`): an embedding of its own width's `project_out` (OPT-350m's), its
    `kernel` `[features, width]` as the layer multiplies by it (an adapter's update included), after a
    checkpoint's quantizer of the layer's input, its operands promoted to `compute` as the layer
    promotes them and multiplied at `precision` by the layer's own plain product (`dense.plain_product`:
    the GEMM template's route, the fp32 sum taken in float64 on a CPU under Dew's tests)."""

    kernel: jax.Array
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)
    compute: DTypeLike | None = struct.field(pytree_node=False, default=None)
    input_quantizer: InputQuantizer | None = None
    path: tuple | None = struct.field(pytree_node=False, default=None)
    """The variables path of the layer's kernel (`projections.kernel_path`), by which a mesh splitting the
    rows places it per shard (`dense.plain_product`)."""

    def __call__(self, hidden: jax.Array) -> jax.Array:
        """`hidden` `[..., features]` projected to `[..., width]`."""
        from dew.nn.dense import plain_product

        hidden = hidden if self.input_quantizer is None else self.input_quantizer.qdq(hidden)
        dtype = jnp.result_type(hidden, self.kernel) if self.compute is None else self.compute
        return plain_product(self.path)(hidden.astype(dtype), self.kernel.astype(dtype),
                               (((jnp.ndim(hidden) - 1,), (0,)), ((), ())), precision=self.precision)


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
    quantization does. Where the matrix is an embedding's own width, the states
    reach it through a `projection` first (`Projection`), so every reader of
    the states the product takes (`states`) reads them at the matrix's width.

    Contracting the model's final states with it (`logits`) reproduces the
    model's own logits, scale, bias, softcap and any adapter on the head
    included. Where no matrix does, `output_table` returns None, and the
    logits come from `logits_from_hidden` or `logits` instead. A head scoring
    the first `scored` of the matrix's rows alone (`CausalTransformer.output_vocab`)
    keeps the matrix whole, the stored parameter itself, and contracts those rows
    alone (`scored_matrix`); its `scale` and `bias` are theirs.
    """

    matrix: jax.Array
    vocab_major: bool = struct.field(pytree_node=False)
    bias: jax.Array | None = None
    softcap: float | None = struct.field(pytree_node=False, default=None)
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)
    compute: DTypeLike | None = struct.field(pytree_node=False, default=None)
    input_quantizer: InputQuantizer | None = None
    scale: jax.Array | None = None
    projection: Projection | None = None
    scored: int | None = struct.field(pytree_node=False, default=None)

    def states(self, hidden: jax.Array) -> jax.Array:
        """`hidden` as the product reads it: through the `projection` where the head has one, then the
        input quantizer's QDQ."""
        if self.projection is not None:
            hidden = self.projection(hidden)
        return hidden if self.input_quantizer is None else self.input_quantizer.qdq(hidden)

    def logits(self, hidden: jax.Array, *, temperature: float = 1.0) -> jax.Array:
        """`hidden`'s `[..., vocab]` logits, at least fp32: the states through
        the input quantizer, the product (`head_product`), the scale and the
        bias (`scaled_logits`), the softcap, then `temperature`, which a
        sampler applies to the head's logits."""
        logits = head_product('...d,vd->...v' if self.vocab_major else '...d,dv->...v',
                              self.states(hidden), self.scored_matrix(), self.precision, compute=self.compute)
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
            vocab = self.vocab
            if index.size and (index.min() < 0 or index.max() >= vocab):
                raise IndexError(f"columns {index.tolist()} reach past a vocabulary of {vocab}")
        matrix = jnp.take(self.scored_matrix(), index, axis=0 if self.vocab_major else 1)
        states = self.states(hidden)
        subscripts = '...d,vd->...v' if self.vocab_major else '...d,dv->...v'
        if exact:
            wide = jnp.promote_types(states.dtype, jnp.float32)
            logits = head_product(subscripts, states.astype(wide),
                                  matrix.astype(jnp.promote_types(matrix.dtype, jnp.float32)),
                                  EXACT_FLOAT32, compute=wide)
        else:
            logits = head_product(subscripts, states, matrix, self.precision, compute=self.compute)
        scale = None if self.scale is None else jnp.take(self.scale, index)
        bias = None if self.bias is None else jnp.take(self.bias, index)
        return capped_logits(scaled_logits(logits, scale, bias), self.softcap, temperature)

    def rows(self) -> jax.Array:
        """Each vocabulary token's `[vocab, features]` row: the matrix's, times
        the token's scale, whose dot product with a state as the product reads
        it (`states`) is the token's logit before its bias and softcap, in the
        wider of the two dtypes. A reader
        of tokens' vectors (a decision head's lexical prior) takes these."""
        rows = self.scored_matrix() if self.vocab_major else self.scored_matrix().T
        return rows if self.scale is None else rows * self.scale[:, None]

    def scored_matrix(self) -> jax.Array:
        """The matrix over the tokens it scores (`vocab`), in its stored layout: the stored matrix itself
        where it scores every row, its first rows (columns, `vocab_major` False) otherwise, which every
        product takes, so no unscored row enters a product or its pullback."""
        if self.scored is None:
            return self.matrix
        return self.matrix[:self.scored] if self.vocab_major else self.matrix[:, :self.scored]

    @property
    def vocab(self) -> int:
        """The tokens it scores: its `scored` rows, else every row of its matrix."""
        return self.matrix.shape[0 if self.vocab_major else 1] if self.scored is None else self.scored


@runtime_checkable
class ComputesIn(Protocol):
    """A model that states the dtype it computes in (`dtype`), a field of its own or its decoder's
    (DiffusionGemma's); None computes in its parameters' dtype (`computes_in`)."""

    @property
    def dtype(self) -> Dtype | None: ...


def computes_in(stated: DTypeLike | None, params) -> jnp.dtype:
    """The dtype a module computes in: `stated` where given (its `dtype`, `ComputesIn`), else what Flax
    promotes its float parameters (`params`, arrays or their shapes) to, as its products with an input of
    theirs give, float32 where it holds none. A quantized store's codes (floats of one byte or less: fp8,
    fp6, fp4) are no dtype a product computes in, their scales the store's: the one rule a cache's dtype
    (`dew.nn.kv_cache.compute_dtype`), a streamed step's price (`dew.training.streaming.compute_dtype`) and
    a release's (`dew.interop.release.computed_in`) read."""
    if stated is not None:
        return jnp.dtype(stated)
    floats = [leaf.dtype for leaf in jax.tree.leaves(params)
              if jnp.issubdtype(leaf.dtype, jnp.floating) and jnp.dtype(leaf.dtype).itemsize > 1]
    return jnp.dtype(jnp.result_type(*floats)) if floats else jnp.dtype(jnp.float32)


@runtime_checkable
class InputSites(Protocol):
    """A token mixer whose projections read its input at more than one site (attention's query, key and
    value), each of which a looped pass's message merges into apart (`dew.nn.cross_pass`): its
    `input_sites`, the first the input its call reads as `x`, the rest the call's `site_inputs`
    (`MixerCall`). A mixer that is no `InputSites` reads one site."""

    @property
    def input_sites(self) -> tuple[str, ...]: ...


@runtime_checkable
class ReadsTrain(Protocol):
    """A module whose call takes `train`, for a stochastic layer of its own (a token mixer's or a
    feed-forward's dropout, a vision tower's drop path): its owner passes `train` where `reads_train`
    says the call reads it, whatever the module's family."""

    @property
    def reads_train(self) -> bool: ...


@runtime_checkable
class ReadsFormed(Protocol):
    """A linear layer whose call takes `formed` (`dew.nn.kernels.projections.FormedInput`: its input formed
    entrywise from other values, a gated MLP's hidden activation from its gate and up projections), which its
    product multiplies holding those values for reverse mode in place of the input: its owner hands them where
    `reads_formed` says the call reads them, whatever the layer's class."""

    @property
    def reads_formed(self) -> bool: ...


@runtime_checkable
class ReadsClips(Protocol):
    """An encoder that says whether it reads a clip's frames, (B, T, H, W, C), or an image: a run over
    clips takes one that does, and a run over images one that doesn't."""

    @property
    def reads_clips(self) -> bool: ...


@runtime_checkable
class SteppedInContext(Protocol):
    """A model whose head reads parameters an objective steps in context, a window at a time
    (`dew.nn.continuum`, `LMObjective(adaptation=...)`): the level's parameters within a tree of its
    variables, None without a level (a chain's levels' trees in order, `dew.nn.continuum`); the tree
    with others in their place; and the two halves of the head the steps run between, the levels on
    final states (`continuum_states`, each level alone by its index) and the head past them
    (`head_logits`). Its owner keeps the levels' place in its tree, so an objective never reads it;
    `in_context_levels` counts them.
    `in_context` says whether the model holds a level, and what a damped step's rate reads
    (`ChunkAdaptation.damped`): `head_bounds`, read from the bound model at the states `head_logits`
    reads, what bounds the curvature and slope of a slot's loss in them (`HeadTerms`); `level_terms`, for
    the parts that step, a level's terms at the states it reads (`LevelTerms`), a level after a stepped
    one `passed` through."""

    @property
    def in_context(self) -> bool: ...

    @property
    def in_context_levels(self) -> int: ...

    def head_bounds(self, states: jax.Array) -> HeadTerms: ...

    def level_terms(self, states: jax.Array, parts: tuple[str, ...], index: int = 0,
                    passed: bool = False) -> LevelTerms: ...

    def in_context_level(self, variables: VariableDict) -> VariableDict | None: ...

    def with_in_context_level(self, variables: VariableDict, level: VariableDict) -> VariableDict: ...

    def continuum_states(self, hidden: jax.Array, index: int | None = None) -> jax.Array: ...

    def head_logits(self, states: jax.Array) -> jax.Array: ...


@runtime_checkable
class SteppedFeedForward(Protocol):
    """A decoder whose own feed-forward blocks an objective steps in context, chunk by chunk through its
    decode cache (`dew.objectives.lm.stacking`, Nested Learning's ad-hoc level stacking): how a tree of its
    variables stores its layer stack (`stored_view`), where the blocks of some layers are read and written
    (`dew.nn.continuum.feedforward_levels`), its cache (`init_cache`) and a cached call's states and
    logits (`states_and_logits`)."""

    def stored_view(self, store: VariableDict) -> StackView: ...

    def init_cache(self, batch_size: int): ...

    def states_and_logits(self, tokens: jax.Array, **kwargs): ...


@runtime_checkable
class BoundedNorm(Protocol):
    """A norm whose derivatives in its input a curvature bound through it reads (a damped in-context
    step's under a head transform, `CausalTransformer.head_bounds`): `spread(x)`, per row the root mean
    square its statistic reads (of `x` centered for a layer norm, the least of its groups' for a grouped
    RMS norm), and `derivative_bounds(width)`, its bounds at a floor of that spread (`NormBounds`). A norm
    that bounds none of its derivatives declares neither, and a bound through it is refused."""

    def spread(self, x: jax.Array) -> jax.Array: ...

    def derivative_bounds(self, width: int) -> NormBounds: ...


@runtime_checkable
class Logits(Protocol):
    """A token model's `[B, S, vocab]` fp32 logits over the whole sequence."""

    def logits(self, tokens: jax.Array, *, train: bool = False, **fields: ModelKwarg) -> jax.Array: ...


_FORWARDLESS = frozenset(nn.Module.__mro__)
"""Flax's module base and its own bases, whose methods are no model's forward (`forward_method`)."""


def forward_method(model: nn.Module, output: str) -> str | None:
    """The method a caller reads `model`'s `output` by (`logits`, `hidden_states`): the model's own method
    of that name where the model has one, its class's or one it exposes (a bridge's `__getattr__` over
    the module it wraps), else its own forward (`__call__`), so a native Flax module whose call returns
    that output is read as it is, with no method that only forwards to it (row 171); None where it has
    neither. Flax's own module classes declare no forward of a model's."""
    if callable(getattr(model, output, None)):
        return output
    declared = [vars(klass) for klass in type(model).__mro__ if klass not in _FORWARDLESS]
    return "__call__" if any(callable(names.get("__call__")) for names in declared) else None


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
class MediaWrapper(Protocol):
    """A multimodal wrapper: media towers and their projectors around the language model it holds
    (`language_model`), its release a wrapper of its `family`'s (`dew.interop.wrappers`)."""

    @property
    def language_model(self) -> nn.Module: ...

    @property
    def family(self) -> str: ...


@runtime_checkable
class ReplacesTokens(Protocol):
    """A model that replaces some of its token ids before its embedding table reads them, as it states
    (`dew.nn.multimodal.MultimodalTransformer`, `dew.nn.diffusion_gemma.DiffusionGemma`): the ids it
    replaces wherever they occur (`replaced_ids`), and whether it replaces the id at each slot a call's
    media fill (`replaces_media`, `dew.nn.inputs.media_slots`), whose embedding its towers give; every
    other id reads its table."""

    @property
    def replaced_ids(self) -> frozenset[int]: ...

    @property
    def replaces_media(self) -> bool: ...


@runtime_checkable
class WrapsDecoder(Protocol):
    """A module around one decoder, which it holds in the field `wraps` names
    (`dew.nn.wrapping.DecoderWrapper`)."""

    wraps: ClassVar[str]


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
    """A diffusion network's output for `sample` at `time`, given its conditions: arrays, encoded text
    (`TextContext`) or a denoiser's conditioning inputs (`dew.nn.inputs.DenoisingCondition`)."""

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
class BankedStacks(Protocol):
    """A model whose stacks of like layers a store may hold banked, each run's layers one array under the
    run's name (`layers_3_7`, `dew.inference.banks`): where in `store` each stack lies and the runs it is
    banked by (`DecoderBank`)."""

    def bank_sites(self, store: VariableDict) -> tuple[DecoderBank, ...]: ...


@runtime_checkable
class StagedStacks(Protocol):
    """A model whose stacks of like layers a pipeline over the stage axis runs: where in `store`, the
    variables a stage-local state is made from, each stack's run lies and how the pipeline over
    `stages` stages stacks it (`DecoderBank`), so a trainer keeps that run stacked by stage on the
    stage's own devices (`dew.training.stages`). A stack that runs whole on every stage has none."""

    def stage_sites(self, store: VariableDict, stages: int) -> tuple[DecoderBank, ...]: ...


EXITS = "exits"
"""The collection a looped decoder gathers its passes' exits in, where a caller opens it (`Looped`)."""

EXIT_PASSES = "exit_passes"
"""The collection a looped decoder under an exit threshold (`Loop.exit_threshold`) writes each token's
1-based exit pass in, `[B, L]` int32, where a caller opens it: the pass the threshold chose, the one whose
state the forward hands on (row 11714). A caller that opens `EXITS` reads every pass's exit instead, the
threshold's choice made by none, and so finds nothing here."""

class ExitPasses(NamedTuple):
    """What a looped decoder's forward gathered of its exits (`Looped.exit_passes`): the states the head
    reads of each pass before the last, every pass's exit-gate logit (none without a gate), and whether
    each exit's pass ran (`ran`, one each, beside the states): `True` each where the forward ran all its
    loop's passes, known as it traces; a traced bool scalar each where a count of passes given per call
    may have skipped some
    (`dew.nn.backbones.looping.run_loop`'s `passes`), whose exit is a zero state's, which a reader of the
    exits leaves out."""

    states: tuple[jax.Array, ...]
    logits: tuple[jax.Array, ...]
    ran: tuple[jax.Array | bool, ...]


@runtime_checkable
class Looped(Protocol):
    """A decoder whose layer stack may run a block of its layers over its own output (`Loop`), None
    where it runs it once, and, where every pass of its loop exits (`Loop.exiting`), what a forward with
    the `exits` collection open gathered of them: the states the head reads of each pass before the
    last, and every pass's exit-gate logit under a gate, as its own scope holds them in `collections`."""

    @property
    def loop(self) -> Loop | None: ...

    def exit_passes(self, collections: Mapping[str, Mapping]) -> ExitPasses: ...


@runtime_checkable
class MixesDepth(Protocol):
    """A decoder whose residual may be Kimi K3's depth mixture of its blocks (`AttentionResiduals`), None
    where it is the plain stream."""

    @property
    def attention_residuals(self) -> AttentionResiduals | None: ...


@runtime_checkable
class Looping(Protocol):
    """A model whose layer stack can run a block of it several times over its own output,
    and the same model at `steps` passes of its block (None keeps its count), or of the block
    `layers`, [first, end), which gives a model with no loop one; None where they cannot.
    `backprop_steps` trains through the final passes, `remat` checkpoints each pass, `implicit`
    takes each later pass by the θ-method, `exit_gate` weighs each pass's exit, `exit_threshold`
    exits each token where they reach it, `exits` gives every pass its exit, `theta_passes` gives an
    implicit loop a θ row for each later pass of that many (`Loop.theta_passes`) and `messages` couples
    the passes through their block summaries (`Loop.messages`); None keeps an existing loop's value, or a
    fresh loop's default."""

    def with_passes(self, steps: int | None = None, layers: tuple[int, int] | None = None, *,
                    backprop_steps: int | None = None, remat: bool | None = None,
                    implicit: bool | None = None, exit_gate: bool | None = None,
                    exit_threshold: float | None = None, exits: bool | None = None,
                    theta_passes: int | None = None, messages: CrossPass | None = None) -> Self | None: ...

    def with_loop(self, loop: Loop) -> Self:
        """The same model running `loop` in place of its own, whole (`dew.config.DecoderRunConfig.loop`)."""
        ...


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

    cross_attention: CrossBlocks | None

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

    def held_cross_flops(self, lengths, context_lengths) -> float:
        """The part of `cross_flops` done by the decoder's own matrices, which train as its trunk does."""
        ...

    def stream_embeddings(self, tokens: jax.Array) -> jax.Array:
        """Each token's row of the model's own table at its stream's width, unscaled and without
        positions: a context's tokens as a cross-attention reads them by the table
        (`dew.nn.cross_attention.encode_context`)."""
        ...

    def with_cross_attention(self, cross: CrossBlocks | None) -> Self: ...


@runtime_checkable
class Differential(Protocol):
    """A decoder whose softmax layers can take DEX's differential on chosen heads (`dew.nn.dex`):
    CausalTransformer, and a wrapper over one."""

    @property
    def dex(self) -> DifferentialExtension | None: ...

    def with_dex(self, extension: DifferentialExtension | None) -> Self: ...

    def dex_sites(self) -> tuple[tuple[str, ...], frozenset[int]]:
        """Where the decoder's parameters sit under `params` (`()` for its own, a wrapper's deeper), and
        its softmax attention layers, by index: what DEX extends and its recipe trains."""
        ...

    def dex_exported(self, variables: Mapping[str, Mapping]) -> dict[str, Mapping]:
        """`variables` as an export writes them, its DEX folded into its layers' output projections
        (`dew.nn.dex.folded`); `ValueError` where a layer's does not fold."""
        ...


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
    own: a resumed run never compiles a lighter rung than it trained on."""

    def recompute_record(self) -> JSON: ...

    def recompute_more(self) -> Self | None: ...

    def restore_recompute(self, record: JSON) -> Self: ...


RUNG_SCOPE = "dew_rung"
"""The name scope each decoder block's forward runs under at every rung, none included (`DecoderBlock`): the
values a remat rung's checkpoint keeps or recomputes by its policy's names, which a compile's live set tells
by the `op_name` of the op that made each."""

FIXED_SCOPE = "dew_fixed_remat"
"""The name scope of the body of a checkpoint whose policy no rung sets (`fixed_remat`: a scanned layer read
from host memory and an in-backward run's layer, `dew.nn.backbones.decoder_stack`; a looped pass,
`dew.nn.backbones.looping`): values made within it no rung keeps or drops by its names, whatever blocks they
sit in."""


def fixed_remat[**Given, Value](function: Callable[Given, Value]) -> Callable[Given, Value]:
    """`function` run under `FIXED_SCOPE`: the body of a checkpoint whose policy no rung sets."""
    @functools.wraps(function)
    def marked(*args: Given.args, **kwargs: Given.kwargs) -> Value:
        with jax.named_scope(FIXED_SCOPE):
            return function(*args, **kwargs)
    return marked


@dataclasses.dataclass(frozen=True)
class RecomputeStatement:
    """What a model's remat rung recomputes in place of keeping it, as a compile's live set names it
    (`jax.stages.Compiled.peak_live`): within its blocks (`scope`, a part of the `op_name` of each value its
    blocks' forward makes, `RUNG_SCOPE`), but under a checkpoint no rung sets (`FIXED_SCOPE`), every value
    `save` names none of where its rung keeps names (`save`, a checkpoint policy keeping those alone: the
    unnamed ones too), else only the values it names `recompute` (a rung that recomputes those alone and
    keeps the rest); no remat keeps all."""

    scope: str
    save: frozenset[str] | None
    recompute: frozenset[str] = frozenset()

    def drops(self, op_name: str, names: tuple[str, ...]) -> bool:
        """Whether this rung keeps no value made by `op_name` that carries `names` for its backward."""
        if self.scope not in op_name or FIXED_SCOPE in op_name:
            return False
        if self.save is None:
            return bool(self.recompute.intersection(names))
        return not self.save.intersection(names)


@runtime_checkable
class StatesRecompute(Protocol):
    """A model that states what its remat rung recomputes (`RecomputeStatement`), which a fit search reads
    a miss's live set by, to ask first the rung that recomputes what the miss was short of."""

    def recompute_statement(self) -> RecomputeStatement: ...


@runtime_checkable
class OffloadsResiduals(Protocol):
    """A model at the top of its remat ladder with a sibling that keeps the residuals a lighter rung saves
    in pinned host memory in place of recomputing them (`recompute_offloaded`), else None: no fit rung, as
    it keeps as many device bytes as the top, but another speed, which only the GPUs' timing chooses
    (`dew.training.ahead.compile_ahead`'s frontier)."""

    def recompute_offloaded(self) -> Self | None: ...


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
        """The members' outputs from the packed projection's `out`, in member order, each its slice."""
        ends = np.cumsum(self.widths)
        return tuple(out[..., int(end - width):int(end)]
                     for end, width in zip(ends, self.widths, strict=True))

    def held(self, params: Mapping) -> bool:
        """Whether `params`, the parameters at `path`, hold the group packed or every member."""
        return self.packed in params or all(member in params for member in self.members)

    def apart(self, projection: Callable[[jax.Array], jax.Array],
              inputs: Sequence[jax.Array]) -> tuple[jax.Array, ...]:
        """Each member's output of its own input, `inputs` in member order, through the packed
        `projection`: the projection run once on each distinct input, keeping that member's columns of it
        (`split`). Exact under whatever wraps the projection (a bias, an adapter, a quantization), as one
        product's columns are the members' own products, where slicing its kernel would read the wrapper's
        parameters by hand."""
        outputs: list[tuple[jax.Array, ...]] = []
        for index, each in enumerate(inputs):
            # An input an earlier member reads too is projected once.
            same = next((at for at, before in enumerate(inputs[:index]) if before is each), None)
            outputs.append(self.split(projection(each)) if same is None else outputs[same])
        return tuple(output[index] for index, output in enumerate(outputs))


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
class RoutesBySite(Protocol):
    """A token mixer composed of routed sites (a parallel layer's branches, `dew.nn.mixers.parallel`):
    `routed_sites` are their paths below it, `/`-joined branch names, and its call takes `routes` as each
    site's own record by its path (`dew.nn.inputs.LayerInputs.experts`) beside the coverage."""

    @property
    def routed_sites(self) -> frozenset[str]: ...


def routed_children(children: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Those of a composite's named `children` that route over experts (`RoutesExperts`), by name, each with
    its routed sites' paths below the composite: its name, or a composed child's (`RoutesBySite`) sites
    below its name."""
    return {name: (tuple(f'{name}/{site}' for site in sorted(child.routed_sites))
                   if isinstance(child, RoutesBySite) else (name,))
            for name, child in children.items() if isinstance(child, RoutesExperts) and child.routes_experts}


def partitioned_routes(routed: Mapping[str, tuple[str, ...]], records, covered: jax.Array | None
                       ) -> dict[str, tuple]:
    """A composite's routing replay, `records` beside their `covered` tokens, as each routed child's own
    (`routed_children`), by its name: a leaf site's record, or a composed child's records by their paths
    below it, each beside the coverage as it is. `records` is a layer's one record, which replays its
    one routed site, or records keyed by site path (`dew.nn.inputs.LayerInputs.experts`); a record for
    several sites, or one that names not every routed site, raises `ValueError`. The one partition a
    decoder's layer (its mixer and its feed-forward) and a parallel mixer (its branches) make."""
    if records is None:
        return {}
    sites = [site for paths in routed.values() for site in paths]
    if not isinstance(records, Mapping):
        if len(sites) != 1:
            raise ValueError(f"a layer's one routing record replays one routed site, and sites "
                             f"{sorted(sites)} route; key the record by site path")
        records = {sites[0]: records}
    if missing := sorted(set(sites) - set(records)):
        raise ValueError(f"the routing record names {sorted(records)}, not the routed sites {missing}")
    return {name: (records[name] if paths == (name,)
                   else {site[len(name) + 1:]: records[site] for site in paths}, covered)
            for name, paths in routed.items()}


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
    (`dew.nn.projections`) and is called over `jax.ShapeDtypeStruct`
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
                         f"(MixerBase.reads_blocks), and {type(model).__name__} has none")
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
class StatesVocabulary(Protocol):
    """A model that declares its vocabulary, the rows of the table its token ids index, which the host
    checks read to refuse an id outside it (`dew.nn.blocks.token_table`), whether or not it declares a
    cache's capacity (`Bounded`)."""

    @property
    def vocab_size(self) -> int: ...


@runtime_checkable
class LogitsWidth(Protocol):
    """A model whose head scores fewer tokens than its table holds rows: its logits' width (Inkling 2's
    200058 of 201024 rows, `CausalTransformer.output_vocab`)."""

    @property
    def logits_width(self) -> int: ...


def logits_width(model: StatesVocabulary) -> int:
    """The width of `model`'s logits: its own where it states one (`LogitsWidth`), else its vocabulary's."""
    return model.logits_width if isinstance(model, LogitsWidth) else model.vocab_size


@runtime_checkable
class Bounded(StatesVocabulary, Protocol):
    """A model that declares its vocabulary and cache capacity, which the host
    checks read to refuse an id outside the vocabulary or a request that would
    overflow the cache; a model that declares neither is checked for neither."""

    @property
    def max_seq_len(self) -> int | None: ...


@runtime_checkable
class Decoding(Protocol):
    """A decoder that continues its rows token by token: `init_cache` opens the
    cache its decode calls (`decode=True`) write, a position a call, as its
    holders state it (`dew.nn.kv_cache.HoldsCache`), tracing no forward; a
    tier-3 model's external graph states its cache only through its step's
    trace (`dew.interop.torchax_fallback`). A caller that appends with those
    calls itself, past `generate`'s and the server's host checks, reads
    `dew.nn.kv_cache.overflowed` after them."""

    def init_cache(self, batch_size: int) -> None: ...


@runtime_checkable
class Layered(Protocol):
    """A decoder that declares its decode cache's layout, as `CausalTransformer` does, and gives
    the same model over another (`with_kv_cache`): a wrapper answers for the decoder that holds
    the cache."""

    @property
    def kv_cache(self) -> KVCache: ...

    def with_kv_cache(self, layout: KVCache) -> Self: ...


class HookedDecoder(Bounded, CacheCapacity, Layered, Protocol):
    """A decoder a module around it forwards its hooks to (`dew.nn.wrapping.DecoderHooks`): its vocabulary
    and bounds, its cache's layout and the same decoder at another layout or capacity. The rest the module
    reads through the protocol the decoder states, where it states one."""


def block_chunks_of(model: nn.Module | HookedDecoder) -> tuple[int | None, ...]:
    """The attention chunk of each trunk layer that reads blocks whole, None where a layer has none; empty for
    a model that says nothing, or whose layers read none (`BlockReading`)."""
    return model.block_chunks() if isinstance(model, BlockReading) else ()


@runtime_checkable
class Exposing(Protocol):
    """A decoder that hands back its hidden states beside its logits, which a
    drafting strategy seeds from; without them a model decodes but cannot draft."""

    def states_and_logits(self, tokens: jax.Array, **kwargs: ModelKwarg) -> tuple[jax.Array, jax.Array]: ...


@runtime_checkable
class Selective(Protocol):
    """A decoder that scores one position per row instead of them all, so its
    prefill runs the head only on the slot the first draw reads; its prediction
    states beside the logits, or None for a decoder that keeps none, whose decode
    steps carry none (`dew.sampling.strategies.DecoderState.hidden`)."""

    def states_and_logits_at(self, tokens: jax.Array, slots: jax.Array, **kwargs: ModelKwarg
                             ) -> tuple[jax.Array | None, jax.Array]: ...


@runtime_checkable
class Predicting(Protocol):
    """A decoder that declares how many multi-token prediction depths it carries."""

    @property
    def num_nextn_predict_layers(self) -> int: ...


@runtime_checkable
class BlockDrafting(Protocol):
    """A decoder that may carry a block drafter: DeepSeek's DSpark (V4.1, V4-Flash-0731) or a Markov
    block drafter (Inkling 2's, `dew.nn.block_draft`), which drafts a whole block per pass from the context
    its target layers record instead of chaining prediction depths."""

    @property
    def drafts_blocks(self) -> bool: ...


@runtime_checkable
class Generating(Protocol):
    """A decoder that says, after examining the layers it builds, why a
    generation loop cannot continue its rows token by token, or None; a
    sampler refuses with that reason without knowing where a family keeps
    the layers that cause it."""

    def generation_refusal(self) -> str | None: ...


@runtime_checkable
class ForkedCalls(Protocol):
    """A decoder that says, after examining the layers it builds, why a call over forks
    (`dew.batch.Forks`: documents that continue another, a shared prefix computed once) would not compute
    each document as its own row of its ancestors and itself does, or None (`fork_refusal`; with `train`,
    as a training step runs it, its dropout drawing at random), such a call refused; and why rows that
    begin alike should not be packed over their shared prefixes (`prefix_sharing_refusal`): that reason,
    or a layer that computes a call over forks at more work than the rows laid whole (an in-context memory
    off its shared recurrence). A caller that packs rows over their shared prefixes
    (`dew.data.forked_rows.ForkedRows`) lays them whole where the second says why; a model that does not
    state it is laid whole too."""

    def fork_refusal(self, train: bool = False) -> str | None: ...

    def prefix_sharing_refusal(self, train: bool = False) -> str | None: ...


@runtime_checkable
class DecoderTraining(Predicting, Protocol):
    """What a decoder trains beside its next token: its multi-token prediction
    depths (arXiv 2412.19437, section 2.2; `apply`'s `method='mtp_hidden_states'`
    and `'mtp_logits'`, `Predicting`), which with `mtp_hyper_connections` set
    read the residual streams a forward sows under `prediction_inputs/states`,
    and the mixers it declares with an indexer an objective trains (`IndexedMixer`)."""

    @property
    def mtp_hyper_connections(self) -> HyperConnections | None: ...

    @property
    def indexed_mixers(self) -> tuple[IndexedMixer, ...]: ...


class IndexedMixer(Protocol):
    """A mixer kind carrying an indexer an objective trains (`MixerBase.indexed_mixers`): DeepSeek's lightning
    indexer (`MLAMixer`), MSA's index branch (`BlockSparseAttentionMixer`); `sparse` where its selection
    picks the keys each query attends, its indexer trained beside dense attention where not."""

    @property
    def sparse(self) -> bool: ...


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
    its canvases. A canvas is refined at the logical `positions` that continue
    its prompt's, None where the prompt gives none and the cache's cursor
    places it, and committed at them laid out as its `coordinates`, the one
    layout a method past `__call__` reads (ledger 1122). A causal language
    model scores tokens too, but runs none of these, so it is not one."""

    @property
    def canvas_length(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int: ...

    def init_cache(self, batch_size: int) -> None: ...

    def encode(self, tokens: jax.Array, *, coordinates: Coordinates | None = None) -> jax.Array: ...

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
class DrawsAlone(Protocol):
    """A decoder whose decode state moves with every token it reads, past what a cache's cursor rewinds
    (an in-context level's chain, `dew.objectives.lm.adaptation.InContextDecoder`): it draws one token a
    row at a time (`dew.sampling.strategies.Sample`), and a strategy that reads tokens it then rejects or
    reorders (a speculative block's, a beam's) is refused, naming `draws_alone`'s reason."""

    @property
    def draws_alone(self) -> str: ...


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


@runtime_checkable
class Adapted(Protocol):
    """Marks a module class an adapter already wrapped, at any depth of its wrappers (`dew.lora`): the
    interceptor its `apply` and `init` run under and the names it binds.

    The wrapper subclass declares the interceptor its `apply` and `init` run
    under, which is the whole record that a class was adapted: a second
    adapter over it would run one branch inside the other. Declared here, below
    the adapters, so a reader of a model's tree asks it without importing them.
    """

    _dew_lora_interceptor: ClassVar[Interceptor]
    _dew_lora_bound: ClassVar[_Bound]
