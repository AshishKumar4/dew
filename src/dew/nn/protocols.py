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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import jax
from flax import linen as nn, struct
from flax.typing import PrecisionLike, VariableDict

from dew.nn.inputs import ModelKwarg

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition
    from dew.nn.dit import TextContext
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.mla import MLAMixer
    from dew.records import JSON

__all__ = ["AffineHead", "BlockDenoiser", "CacheCapacity", "DecoderTraining", "DenoisingModel",
           "HiddenStates", "IntervalModel", "LayerScalars", "Logits", "LogitsFromHidden", "ModelKwarg",
           "OutputTable", "ProjectionGroup", "Recomputing", "RequiresText", "Serving", "TimeScaled",
           "TokenModel", "TritonGemm"]


@struct.dataclass
class OutputTable:
    """A vocabulary head as a matrix a loss contracts: `matrix` as the tree
    stores it, `[vocab, features]` when `vocab_major` and `[features, vocab]`
    otherwise, an optional fp32 `[vocab]` `bias` added after the product,
    the `softcap` applied last, and the `precision` the product runs at.

    Contracting the model's final states with it reproduces the model's own
    logits, bias, softcap and any adapter on the head included. Where no
    matrix does, `output_table` returns None, and the logits come from
    `logits_from_hidden` or `logits` instead.
    """

    matrix: jax.Array
    vocab_major: bool = struct.field(pytree_node=False)
    bias: jax.Array | None = None
    softcap: float | None = struct.field(pytree_node=False, default=None)
    precision: PrecisionLike = struct.field(pytree_node=False, default=None)


@runtime_checkable
class Logits(Protocol):
    """A token model's `[B, S, vocab]` fp32 logits over the whole sequence."""

    def logits(self, tokens: jax.Array, *, train: bool = False, **fields: ModelKwarg) -> jax.Array: ...


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
class LogitsFromHidden(Protocol):
    """The exact head from final states, for scoring states a loss already holds."""

    def logits_from_hidden(self, hidden: jax.Array) -> jax.Array: ...


@runtime_checkable
class CacheCapacity(Protocol):
    """A decoder at another decode-cache capacity: the same variables, and
    the same draws for a request that fits."""

    def with_cache_capacity(self, capacity: int) -> Self: ...


@runtime_checkable
class Recomputing(Protocol):
    """A model one remat rung up (None at the top), its rung as a checkpoint
    records it, and the model at a record's rung where that is above its
    own: a resumed run never compiles a lighter rung than it trained on."""

    def recompute_record(self) -> JSON: ...

    def recompute_more(self) -> Self | None: ...

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

    def held(self, params: Mapping) -> bool:
        """Whether `params`, the parameters at `path`, hold the group packed or every member."""
        return self.packed in params or all(member in params for member in self.members)


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
