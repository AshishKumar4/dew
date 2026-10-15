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
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.typing import PrecisionLike, VariableDict
from jax.typing import DTypeLike

from dew.nn.inputs import ModelKwarg
from dew.nn.precision import biased_logits, capped_logits, head_product

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition
    from dew.inputs.encoders import ConditionWork
    from dew.nn.backbones.decoder_block import RematPolicy
    from dew.nn.dit import TextContext
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.mla import MLAMixer
    from dew.records import JSON
    from dew.training.quantization import InputQuantizer

__all__ = [
    "AffineHead",
    "BlockDenoiser",
    "CacheCapacity",
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
    "DenoisingModel",
    "HiddenStates",
    "IntervalModel",
    "LayerScalars",
    "Logits",
    "LogitsFromHidden",
    "Looping",
    "ModelKwarg",
    "OutputTable",
    "ProjectionGroup",
    "QuantizesInputs",
    "Recomputing",
    "RematChoosing",
    "RequiresText",
    "Serving",
    "TimeScaled",
    "TokenModel",
    "TritonGemm",
]


@struct.dataclass
class OutputTable:
    """A vocabulary head as a matrix a loss contracts: `matrix` as the tree
    stores it, `[vocab, features]` when `vocab_major` and `[features, vocab]`
    otherwise, an optional fp32 `[vocab]` `bias` added after the product,
    the `softcap` applied last, and the `precision` the product runs at.
    `compute`, the model's compute dtype, decides the product's operands
    (`dew.nn.precision.head_product`; the states' dtype where None), and an
    `input_quantizer` rounds the states first as a checkpoint's head input
    quantization does.

    Contracting the model's final states with it (`logits`) reproduces the
    model's own logits, bias, softcap and any adapter on the head included.
    Where no matrix does, `output_table` returns None, and the logits come
    from `logits_from_hidden` or `logits` instead.
    """

    matrix: jax.Array
    vocab_major: bool = struct.field(pytree_node=False)
    bias: jax.Array | None = None
    softcap: float | None = struct.field(pytree_node=False, default=None)
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)
    compute: DTypeLike | None = struct.field(pytree_node=False, default=None)
    input_quantizer: InputQuantizer | None = None

    def states(self, hidden: jax.Array) -> jax.Array:
        """`hidden` as the product reads it: through the input quantizer's QDQ."""
        return hidden if self.input_quantizer is None else self.input_quantizer.qdq(hidden)

    def logits(self, hidden: jax.Array, *, temperature: float = 1.0) -> jax.Array:
        """`hidden`'s `[..., vocab]` logits, at least fp32: the states through
        the input quantizer, the product (`head_product`), the bias, the
        softcap, then `temperature`, which a sampler applies to the head's
        logits."""
        logits = head_product('...d,vd->...v' if self.vocab_major else '...d,dv->...v',
                              self.states(hidden), self.matrix, self.precision, compute=self.compute)
        return capped_logits(biased_logits(logits, self.bias), self.softcap, temperature)


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

    def decision_flops(self, lengths: np.ndarray, questions: np.ndarray, options: np.ndarray) -> float: ...


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


@runtime_checkable
class QuantizesInputs(Protocol):
    """A model computing under a checkpoint's input quantization
    (`dew.training.quantization.InputQuantization`): the quantizer a bound
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
class Recomputing(Protocol):
    """A model one remat rung up (None at the top), its rung as a checkpoint
    records it, and the model at a record's rung where that is above its
    own: a resumed run never compiles a lighter rung than it trained on.
    `recompute_half` is the model one rung up where that rung recomputes only
    the few values it names, a half rung the planner tries at each split
    (`dew.training.planning.plan_step`), else None."""

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


def declared_groups(*modules: nn.Module) -> tuple[ProjectionGroup, ...]:
    """The groups the bound `modules` declare, in order; a module declaring none adds none."""
    return tuple(group for module in modules if isinstance(module, ProjectionSites)
                 for group in module.projection_groups())


@runtime_checkable
class Serving(Protocol):
    """What a server reads off a decoder. `inference_projection_groups` names
    the packed groups `variables` hold, packed or as every member (the packer
    checks the members concatenate; a model that does not decode
    autoregressively names none). `cache_rebuild_position` is where cached
    keys go stale and the prefix is recomputed (LongRoPE's long factors,
    Phi-3), or None. `mixed_admission_refusal` says why its layers would not
    run a mixed call (`dew.nn.inputs.Admitted`) as separate decode and
    prefill calls would, or None."""

    def inference_projection_groups(self, variables: Mapping[str, Mapping]
                                    ) -> tuple[ProjectionGroup, ...]: ...

    @property
    def cache_rebuild_position(self) -> int | None: ...

    def mixed_admission_refusal(self) -> str | None: ...


@runtime_checkable
class DecoderTraining(Protocol):
    """What a decoder trains beside its next token: its multi-token prediction
    depths (arXiv 2412.19437, section 2.2; `apply`'s `method='mtp_hidden_states'`
    and `'mtp_logits'`), which with `mtp_hyper_connections` set read the
    residual streams a forward sows under `prediction_inputs/states`, and the
    MLA mixers it declares with DeepSeek's lightning indexer."""

    @property
    def num_nextn_predict_layers(self) -> int: ...

    @property
    def mtp_hyper_connections(self) -> HyperConnections | None: ...

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]: ...


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
    its canvases. A causal language model scores tokens too, but runs none of
    these, so it is not one."""

    @property
    def canvas_length(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def max_seq_len(self) -> int: ...

    def init_cache(self, batch_size: int) -> None: ...

    def encode(self, tokens: jax.Array) -> jax.Array: ...

    def __call__(self, tokens: jax.Array, *, self_conditioning_logits: jax.Array | None = None,
                 self_conditioning_mask: jax.Array | None = None) -> jax.Array: ...


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
