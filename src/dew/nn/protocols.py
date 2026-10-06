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
cache: a call writes no `cache` collection, `Ordered` says whether those
states attend causally, and `MaskToken` names the id masked diffusion
corrupts tokens to. `DenoisingModel` is the call every image and video
denoiser already has, the raw network that `dew.diffusion.process.Denoiser`
wraps into a prediction a solver steps with. `AffineHead` and
`LogitsFromHidden` let a loss score the vocabulary without a second trunk
pass: `output_table` gives the matrix a tiled loss contracts in place of the
logits, and `logits_from_hidden` is the exact head for a head no matrix alone
gives. `HardVocabularyEmbedder` is a media embedder's: the range of the text
vocabulary it embeds itself.
`DenoisingModel` is an annotation only: every Flax module has a `__call__`, so
an `isinstance` check on it would hold for any model.

The rest are read off the model itself, not through `apply`: what a server
or a trainer needs to know about it, from the layers it declares.
`CacheCapacity` and `Recomputing` give another model of the same variables
(a cache size, a remat rung); `PackedProjections`, `MixedAdmission` and
`CacheRebuilding` say how its decode step may be served; `Indexed`,
`StreamedPrediction` and `TritonGemm` describe its mixers and prediction
depths. A wrapper answers for the decoder it holds.

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
from flax.typing import PrecisionLike

from dew.nn.inputs import ModelKwarg

if TYPE_CHECKING:
    from dew.diffusion.process import DenoisingCondition
    from dew.nn.dit import TextContext
    from dew.nn.hyper_connections import HyperConnections
    from dew.nn.mla import MLAMixer
    from dew.records import JSON

__all__ = ["AffineHead", "CacheCapacity", "CacheRebuilding", "DenoisingModel", "HardVocabularyEmbedder",
           "HiddenStates", "Indexed", "Logits", "LogitsFromHidden", "MaskToken", "MixedAdmission",
           "ModelKwarg", "Ordered", "OutputTable", "PackedProjections", "Predicting", "ProjectionGroup",
           "ProjectionSites", "ReadsTrain", "Recomputing", "StreamedPrediction", "TritonGemm",
           "declared_groups"]


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
class Ordered(Protocol):
    """A token model that says whether its `hidden_states` attend causally,
    each position to itself and the ones before it, or to the whole sequence."""

    @property
    def causal(self) -> bool: ...


@runtime_checkable
class MaskToken(Protocol):
    """A token model that names the vocabulary id a masked-diffusion objective
    corrupts tokens to, or None for one trained without it."""

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
class HardVocabularyEmbedder(Protocol):
    """A media embedder that also embeds a range of the text vocabulary, the
    hard tokens the decoder's own table does not hold, as Gemma 3n's vision
    and audio embedders do (modeling_gemma3n.py, Gemma3nMultimodalEmbedder)."""

    def embed_hard(self, ids: jax.Array) -> jax.Array: ...

    def merge_hard_embeddings(self, token_embeddings: jax.Array, ids: jax.Array) -> jax.Array: ...


@runtime_checkable
class LogitsFromHidden(Protocol):
    """The exact head from final states, for scoring states a loss already holds."""

    def logits_from_hidden(self, hidden: jax.Array) -> jax.Array: ...


@runtime_checkable
class CacheCapacity(Protocol):
    """A decoder whose decode cache holds a capacity its owner chooses.

    `with_cache_capacity` is the same model with `capacity` cache slots per
    row, the only change being the cache it allocates: it reads the same
    variables and draws the same tokens for a request that fits. A server
    keeps one per capacity, since the model is a static argument of the
    compiled step.
    """

    def with_cache_capacity(self, capacity: int) -> Self: ...


@runtime_checkable
class Recomputing(Protocol):
    """A model whose backward pass recomputes more of its forward, one rung at a time.

    `recompute_more` is the model one rung up its own ladder, slower and
    holding less for the backward pass, or None at the top or off the
    ladder. `recompute_record` is the rung as a checkpoint records it, and
    `restore_recompute` climbs to a record's rung where that is above this
    model's own, and otherwise returns this model: a resumed run compiles
    the rung its checkpoint trained on, never a lighter one. The variables
    are the same at every rung.
    """

    def recompute_record(self) -> JSON: ...

    def recompute_more(self) -> Self | None: ...

    def restore_recompute(self, record: JSON) -> Self: ...


@dataclasses.dataclass(frozen=True)
class ProjectionGroup:
    """Dense projections that read one input, which a server packs into one kernel.

    `members` are the projections under `path` in `params`, in the order
    their outputs concatenate into the projection named `packed`, each
    `widths` wide on the output axis. A module whose variables hold `packed`
    reads it in their place, so a decode step multiplies once where it
    would `len(members)` times; the training layout keeps the members.
    """

    path: tuple[str, ...]
    packed: str
    members: tuple[str, ...]
    widths: tuple[int, ...]

    def held(self, params: Mapping) -> bool:
        """Whether `params`, the parameters at `path`, hold the group packed or every member."""
        return self.packed in params or all(member in params for member in self.members)


@runtime_checkable
class ProjectionSites(Protocol):
    """A bound module that declares the groups it reads packed (`ProjectionGroup`),
    its own and its children's, at their paths in the variables it is bound to."""

    def projection_groups(self) -> tuple[ProjectionGroup, ...]: ...


def declared_groups(*modules: nn.Module) -> tuple[ProjectionGroup, ...]:
    """The groups the bound `modules` declare (`ProjectionSites`), in order;
    a module that declares none, such as a mixer with its own projections,
    adds nothing."""
    return tuple(group for module in modules if isinstance(module, ProjectionSites)
                 for group in module.projection_groups())


@runtime_checkable
class PackedProjections(Protocol):
    """A decoder that serves its decode step from packed projections.

    `inference_projection_groups` names every group its modules read packed
    whose packed form or members `variables` hold, from the layers it
    declares. Whether the stored members can be concatenated is the
    packer's to check; a model that does not decode autoregressively names
    none.
    """

    def inference_projection_groups(self, variables: Mapping[str, Mapping]
                                    ) -> tuple[ProjectionGroup, ...]: ...


@runtime_checkable
class MixedAdmission(Protocol):
    """A decoder that may run a server's admitting step as one mixed forward
    (`dew.nn.inputs.Admitted`): the fed rows' draws and the admitted prompt
    pieces in one row.

    `mixed_admission_refusal` is why its layers would not run that call as
    the separate decode and prefill calls would, or None when they do. A
    model without it keeps the two forwards.
    """

    def mixed_admission_refusal(self) -> str | None: ...


@runtime_checkable
class CacheRebuilding(Protocol):
    """A decoder whose cached keys go stale when a row crosses a position.

    `cache_rebuild_position` is that position, past which the whole prefix
    is recomputed (LongRoPE's switch from its short to its long factors,
    Phi-3), or None for a cache that stays valid at every length.
    """

    @property
    def cache_rebuild_position(self) -> int | None: ...


@runtime_checkable
class Indexed(Protocol):
    """A decoder whose MLA mixers may carry DeepSeek's lightning indexer.

    `indexed_mixers` are the mixer values with the indexer among those the
    model declares, the model's own and each layer kind's; an indexer
    objective trains them, dense (`sparse` False) or on their selection.
    """

    @property
    def indexed_mixers(self) -> tuple[MLAMixer, ...]: ...


@runtime_checkable
class Predicting(Protocol):
    """A decoder that declares how many multi-token prediction depths it carries.

    Depth d pairs the state at position p with the embedding of the token at
    p + d and scores the token after it (arXiv 2412.19437, section 2.2).
    Through `apply`, `method='mtp_hidden_states'` gives one final-normed
    state array per depth from the trunk's states and the tokens, and
    `method='mtp_logits'` their logits under the shared head.
    """

    @property
    def num_nextn_predict_layers(self) -> int: ...


@runtime_checkable
class StreamedPrediction(Protocol):
    """A decoder whose prediction depths may carry their own residual streams.

    With `mtp_hyper_connections` set, a forward that collects
    `prediction_inputs` sows the depths' input streams there under `states`,
    and the depths read those rather than the final hidden states.
    """

    @property
    def mtp_hyper_connections(self) -> HyperConnections | None: ...


@runtime_checkable
class TritonGemm(Protocol):
    """A model that says whether a mixer of it keeps XLA's Triton GEMM fusions
    (`MixerBase.keeps_triton_gemm`), which a trainer otherwise turns off on
    the GPU generations `dew.telemetry.devices.TRITON_GEMM_OFF_GENERATIONS`
    names."""

    @property
    def keeps_triton_gemm(self) -> bool: ...


@runtime_checkable
class ReadsTrain(Protocol):
    """A token mixer whose call takes `train` when `reads_train` holds: one
    that runs a dropout of its own while training. A block passes the flag to
    such a mixer alone, since another mixer's call does not take it."""

    @property
    def reads_train(self) -> bool: ...
