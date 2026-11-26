"""Tier 3: a transformers PyTorch causal LM, lowered to JAX by torchax.

`Pretrained.load` takes this tier for an unregistered causal model_type no
registered convention computes (`verify.Unmapped`) once `admitted` decides
transformers' own forward runs it, and `Pretrained.load(source,
fallback="torchax")` takes it for any source. It builds the model with
transformers on the host and hands its forward to torchax, which runs every
torch op as a JAX op, so the model jits, differentiates and shards like any
other JAX function. Dew reads no architecture of it: it computes what the
model asks transformers' interfaces for, its attention and decode cache
(`AttentionInterface`) and its routed experts (`ExpertsInterface`, `EXPERTS`),
and the experts transformers loops over in Python (`functional_parts`), with
its own kernels; it places each tensor by its torch name under the logical axes
transformers' tensor-parallel plan states (`planned_axes`, `TorchLayout`); and
it wraps the torch forward in a Flax module, its parameters split from its
buffers.

The split is `JittableModule`'s. Its `params` are the trainable
`nn.Parameter`s and go under the `params` collection, the one the optimizer
moves; every other tensor (rotary `inv_freq`, causal masks) goes under
`buffers`, which the trainer carries as state and never updates. A tied
output head is one parameter under its first name, as `JittableModule`
deduplicates it.

This module imports neither torch nor torchax until a load asks for them.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import importlib.metadata
import inspect
import json
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core.scope import Variable
from flax.typing import Dtype, PrecisionLike
from jax.extend.core import Literal, primitives

from dew import _records
from dew._records import JSON
from dew.batch import COORDINATES_FIELD, Coordinates, Variables, real_slots, token_coordinates
from dew.distributed.placement import Layout
from dew.interop import sources, verify
from dew.interop.decoder_parts import context_lengths
from dew.interop.generation_config import read_generation_config
from dew.interop.pickles import host_dtype, host_view
from dew.interop.pretrained import Placing, PretrainedFallback, source_processor
from dew.interop.sequence_causality import (
    CAUSAL_ATTENTION,
    STATED_ATTENTION,
    TOKENWISE,
    attested,
    causality,
    causality_refusal,
)
from dew.interop.streaming import SourceLeaf
from dew.interop.weights import AUTO, checkpoint_dtype
from dew.mesh import LogicalAxes
from dew.nn.attention import AttentionImpl, chosen_attention
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.inputs import document_reach
from dew.nn.kernels import generation
from dew.nn.kv_cache import CURSOR, cache_positions, cursor, filled_slots, put_opened
from dew.nn.precision import at_least_fp32, broadcast_summed, configured_precision, exact_float32, scaled
from dew.nn.programs import purely_called
from dew.nn.protocols import ExitPasses, Lacks, OutputTable
from dew.nn.sharding import parameter_path
from dew.registry import resolve_dtype

if TYPE_CHECKING:
    import torch
    from jax.extend.core import Jaxpr
    from torchax.interop import JittableModule
    from transformers import PreTrainedModel

    from dew.nn.backbones.decoder_stack import DecoderBank

INSTALL = "pip install 'dewml[torchax]'"


class TransformersTierWarning(UserWarning):
    """A source loaded as tier 3: transformers' own PyTorch forward, run by torchax, in place of a
    registered family's."""


EXPERTS = "dew"
"""The routed-experts implementation tier 3 builds transformers' models with, registered in transformers'
`ExpertsInterface` (`_registered`): Dew's grouped expert dispatch (`_dew_experts`) for experts its gated
product computes, transformers' batched products over each token's chosen experts otherwise
(`batched_mm`, whose shapes the token count fixes, where `grouped_mm` has no torchax lowering and the
eager loop selects tokens by `nonzero`). A model whose experts do not dispatch through the interface runs
its own, which `admitted` observes."""


def _summary(reasons: Mapping[str, str], shown: int = 3) -> str:
    """The first `shown` of `reasons`, each convention's, and how many more there are."""
    listed = "; ".join(f"{convention}: {reason}" for convention, reason in list(reasons.items())[:shown])
    rest = len(reasons) - shown
    return listed + (f"; and {rest} more conventions" if rest > 0 else "")


@dataclass(frozen=True)
class TorchAdmission:
    """An unregistered causal LM no registered convention computes that tier 3 runs (`admitted`):
    its model_type, why each convention did not compute it (`verify.Unmapped.reasons`), and whether
    its every decoder layer attends through transformers' attention interface with visibility Dew's
    attention computes (`attends`, `_Probe`), which runs them on Dew's attention, packing included;
    `why` says what kept it otherwise."""

    model_type: str
    reasons: Mapping[str, str]
    attends: bool = False
    why: str = ""
    assumes: str = ""
    """What its causality rests on past its values' finiteness (`sequence_causality.Causality.assumption`),
    which its load's warning names; empty where nothing."""


def admitted(unmapped: verify.Unmapped) -> TorchAdmission:
    """Tier 3's decision on an unregistered model_type no registered convention computes, on the
    probe's shrunken config with random weights (`verify.scatter_weights`), before any weight
    downloads: transformers builds the type as a causal LM (`AutoModelForCausalLM`); its forward asks
    the attention interface what Dew's attention computes in every decoder layer, or runs its own
    (`attends`, `_Probe`); and the model tier 3 runs on that attention (`load`), its inference and
    training forwards traced as tier 3 traces them (`TorchGraph.traced`), is causal by its programs
    (`sequence_causality.causality_refusal`: each output position reads no later input through every
    operation, on any backend, from the program alone), the training one drawing nothing on the host. A
    decoder configured bidirectionally (BERT's with is_decoder False) is not. An operation whose output's
    shape or value the traced values decide (a `nonzero`, an `item()`) fails that trace with JAX's own
    error, naming it. On Dew's attention, each program the model then runs is proven before it runs
    (`TorchGraph.certify`): a trace at one length says nothing of another's. Each refusal names its
    cause."""
    import torch
    from torchax.interop import JittableModule, extract_all_buffers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    model_type = unmapped.model_type
    detail = _summary(unmapped.reasons)
    if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        raise ValueError(f"model_type {model_type!r} is no registered family, no registered family's "
                         f"convention computes it ({detail}), and transformers builds no causal LM for it, "
                         "which tier 3 runs")
    fields = {key: value for key, value in _records.decoded_floats(unmapped.fields).items()
              if key != "model_type"}
    config = AutoConfig.for_model(model_type, **fields, experts_implementation=_experts())
    _registered()
    # Built as `load` builds it (its config's own attention, which transformers refuses where it has none
    # of the kind), then on the probe's.
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=torch.float32).eval()
    model.set_attn_implementation(_PROBE)
    verify.scatter_weights(model)
    ids = verify.probe_ids(int(config.vocab_size)).astype(np.int32)
    probe = _Probe()
    with torch.no_grad(), probe.recording():
        model(input_ids=torch.from_numpy(ids.astype(np.int64)), use_cache=False)
    why = probe.kept_from_dew(int(config.num_hidden_layers))
    # The model tier 3 runs, built as `load` builds it (`_built`, its config's own attention), its parts
    # functional, on Dew's attention where every layer asks for what it computes; its experts at static
    # shapes (`functional_parts`), so one token's route rounds no other's products (ledger 3125).
    ran = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=torch.float32).eval()
    verify.scatter_weights(ran)
    functional_parts(ran)
    if not why:
        ran.set_attn_implementation(_ATTENTION)
    jittable = JittableModule(ran)
    params = {name: host_view(leaf, name) for name, leaf in jittable.params.items()}
    buffers = {name: host_view(leaf, name) for name, leaf in extract_all_buffers(ran)[1].items()}
    graph = TorchGraph(jittable, (), dew_attention=not why)
    # Dew's attention as the reference path computes it, whatever the run names: a program read, not run.
    with generation.chosen(attention="reference"):
        inference, _, _ = graph.traced(params, buffers, ids, None, train=False)
        training, _, draws = graph.traced(params, buffers, ids, jax.random.key(0), train=True)
    assumed = set()
    for program, forward in ((inference, "forward"), (training, "training forward")):
        verdict = causality(program, len(params) + len(buffers), ids.shape[1], (0, 1))
        if verdict.refusal is not None:
            raise ValueError(f"model_type {model_type!r} as configured is not causal by its {forward}'s "
                             f"program ({verdict.refusal}), and tier 3 runs causal LMs")
        assumed.add(verdict.assumption())
    if draws.host:
        raise ValueError(f"model_type {model_type!r}'s training forward draws on the host ({draws.host} "
                         "draws, BioGPT's layer drop among them), which a traced step would fix for every "
                         "step, and tier 3 trains the models it loads")
    return TorchAdmission(model_type, dict(unmapped.reasons), attends=not why, why=why,
                          assumes="; ".join(sorted(filter(None, assumed))))


type Traceable = jax.typing.ArrayLike | jax.ShapeDtypeStruct
"""A value `TorchGraph.traced` traces the forward over: an array, or its shape and dtype alone, as an
initialization and an unprobed source's qualification (`qualified`) hold them."""

QUALIFYING = 13
"""The most tokens a row `qualified` traces an unprobed source's forward over at its load, the largest
prime up to it its context holds (one token for a context of one): no other size of its program is
taken for the sequence's. Each program it then runs is proven on its own (`TorchGraph.certify`)."""


def qualified(config: Mapping[str, object], unavailable: verify.ProbeUnavailable) -> TorchAdmission:
    """Tier 3's decision on an unregistered causal LM whose shrunken probe cannot run
    (`verify.ProbeUnavailable`), on its full config and no weights (ledger 1595, the 1493 design's H7):
    transformers builds the type as a causal LM on torch's meta device; its forward, traced as tier 3
    traces it over abstract weights (`TorchGraph.traced`), runs on Dew's attention in every decoder
    layer, each passing only keywords Dew computes with a mask its causal visibility reproduces
    (`_Probe`, read off the mask, `_masked`); an operation whose value the traced values decide (BLT's
    `item()`) fails that trace with JAX's own error, naming it; the trace on Dew's attention at
    `QUALIFYING` tokens is causal by its program (`sequence_causality.causality_refusal`: each output
    position reading no later input through every operation); and its training forward draws nothing on
    the host. Each refusal names its cause. torchax holds no symbolic length (its tensors' torch sizes
    are integers), so no one trace proves every length: each program the model then runs is proven before it
    runs (`TorchGraph.certify`)."""
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    model_type = _records.text(config.get("model_type"), "model_type")
    if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        raise ValueError(f"model_type {model_type!r}'s probe does not run ({unavailable}), and transformers "
                         "builds no causal LM for it, which tier 3 runs")
    fields = {key: value for key, value in _records.decoded_floats(config).items() if key != "model_type"}
    source = AutoConfig.for_model(model_type, **fields, experts_implementation=_experts())
    _registered()
    context = fields.get("max_position_embeddings")
    stated = None if context is None else _records.integer(context, "max_position_embeddings")
    length = max(prime for prime in (1, 2, 3, 5, 7, 11, QUALIFYING) if stated is None or prime <= stated)
    probe = _Probe(abstract=True)
    try:
        # Its masks evaluated as the trace runs, so the probe reads their values (`_masked`).
        with probe.recording(), jax.ensure_compile_time_eval():
            _abstract(source, _PROBE, length, train=False)
        why = probe.kept_from_dew(int(source.num_hidden_layers))
        if why:
            raise ValueError(f"model_type {model_type!r}'s probe does not run ({unavailable}), and tier 3 "
                             "holds an unprobed source causal by Dew's attention, which it cannot run on: "
                             f"{why}")
        closed, tokens, _ = _abstract(source, _ATTENTION, length, train=False)
        trained = _abstract(source, _ATTENTION, length, train=True)[2]
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
            jax.errors.TracerBoolConversionError, jax.errors.TracerIntegerConversionError) as error:
        raise ValueError(f"model_type {model_type!r}'s probe does not run ({unavailable}), and its forward "
                         f"does not trace at its full config: {type(error).__name__}: {error}") from error
    refusal = causality_refusal(closed, tokens, length, (0, 1))
    if refusal is not None:
        raise ValueError(f"model_type {model_type!r}'s probe does not run ({unavailable}), and its forward "
                         f"is not causal by its program: {refusal}")
    if trained.host:
        raise ValueError(f"model_type {model_type!r}'s training forward draws on the host ({trained.host} "
                         "draws), which a traced step would fix for every step, and tier 3 trains the "
                         "models it loads")
    return TorchAdmission(model_type, {"probe": str(unavailable)}, attends=True)


def _abstract(source, attention: str, length: int, *, train: bool):
    """`source`'s causal LM built on torch's meta device as `load` builds it (its config's own attention,
    which transformers refuses where it has none of the kind), then on `attention`, its forward over a row
    of `length` tokens traced as tier 3 traces it (`TorchGraph.traced`) over abstract weights: the closed
    program, the index of the tokens among its inputs (past the parameters' and the buffers' leaves, as
    `traced` flattens them), and the draws it made."""
    import torch
    from torchax.interop import JittableModule, extract_all_buffers
    from torchax.ops.mappings import t2j_dtype
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(source, trust_remote_code=False, dtype=torch.float32).eval()
    model.set_attn_implementation(attention)
    functional_parts(model)
    jittable = JittableModule(model)
    params, buffers = ({name: jax.ShapeDtypeStruct(tuple(leaf.shape), t2j_dtype(leaf.dtype))
                        for name, leaf in held.items()}
                       for held in (jittable.params, extract_all_buffers(model)[1]))
    graph = TorchGraph(jittable, (), dew_attention=attention == _ATTENTION)
    tokens = jax.ShapeDtypeStruct((1, length), jnp.int32)
    key = jax.random.key(0) if train else None
    with generation.chosen(attention="reference"):
        closed, _, draws = graph.traced(params, buffers, tokens, key, train=train)
    return closed, len(params) + len(buffers), draws


_ATTENTION = "dew"
"""The name Dew's attention is registered under in transformers' attention interface (`_registered`):
a model built with it computes each layer's attention with `dew.nn.attention`, given no mask, as
transformers gives an attention it has no mask function for none."""

_PROBE = "dew_probe"
"""The name of the attention `admitted` builds its probe with: transformers' SDPA over its own SDPA
mask, recording what each call asks (`_Probe`)."""

_PLUMBING = frozenset({"position_ids", "cache_position", "position_embeddings", "use_cache",
                       "output_attentions", "past_key_values", "past_key_value", "layer_idx", "is_causal"})
"""Keywords transformers' attention modules pass beside the computation's own, which name nothing the
attention computes: positions are already in the rotated queries and keys, and a decoding call's cache is
Dew's (`_Cached`), not transformers'."""

def _asked(kwargs: Mapping[str, object]) -> set[str]:
    """The keywords an attention call passes that ask its attention a computation Dew's does not make:
    past `_PLUMBING` and `_COMPUTED`, those that hold a value (a BERT decoder's `encoder_hidden_states`
    of None asks no cross-attention)."""
    return {name for name, value in kwargs.items() if value is not None} - _PLUMBING - _COMPUTED


_COMPUTED = frozenset({"sliding_window", "softcap", "s_aux"})
"""The keywords Dew's attention computes (`_dew_attention`): a window over the most recent keys, a
tanh cap on the logits, and one value-free logit a head (sinks)."""

_PROBING: contextvars.ContextVar[_Probe | None] = contextvars.ContextVar("dew_probe", default=None)


class _Probe:
    """What each attention call of a probed forward asks (`recording`): the module, the keywords it
    passes past `_PLUMBING`, and whether the mask transformers built for it is the visibility Dew's
    attention computes from the module's causality and those keywords (`_visible`)."""

    def __init__(self, *, abstract: bool = False):
        self.modules: set[int] = set()
        self.unknown: set[str] = set()
        self.unmatched = 0
        self.abstract = abstract
        """Whether the probed forward is traced over abstract weights (`qualified`), so the keys each
        query sees are read off its mask (`_masked`), not its attention's weights."""

    @contextlib.contextmanager
    def recording(self):
        token = _PROBING.set(self)
        try:
            yield self
        finally:
            _PROBING.reset(token)

    def called(self, module, query, key, attention_mask, scaling, kwargs) -> None:
        self.modules.add(id(module))
        self.unknown |= _asked(kwargs)
        # A probe traced over abstract weights (`qualified`) reads the keys each query sees off the mask
        # alone, whose values its trace holds; a mask it does not hold is no visibility it can verify.
        seen = (_masked(module, query, key, attention_mask, scaling, kwargs) if self.abstract
                else _seen(module, query.shape, key.shape, query.dtype, attention_mask, scaling, kwargs))
        if seen is None or not _visible(seen, kwargs.get("sliding_window")):
            self.unmatched += 1

    def kept_from_dew(self, layers: int) -> str:
        """Why the probed model cannot run on Dew's attention, or "" where it can: every one of its
        `layers` decoder layers attended through the interface, passing only keywords Dew computes,
        with a mask Dew's visibility reproduces."""
        if len(self.modules) != layers:
            return (f"{len(self.modules)} of its {layers} decoder layers attend through transformers' "
                    "attention interface")
        if self.unknown:
            return f"its attention takes {sorted(self.unknown)}, which Dew's attention does not compute"
        if self.unmatched:
            return (f"{self.unmatched} of its attention calls mask other keys than causality and its "
                    "window name")
        return ""


def _seen(module, queries, keys, dtype, attention_mask, scaling, kwargs) -> np.ndarray:
    """Which keys each query of a call of `queries` and `keys` (their shapes) reads in transformers'
    SDPA attention, `[B, Q, H, K]`: its attention weights over one-hot values, nonzero where a key is
    read. Read from the attention itself, so a mask transformers builds or one it leaves to the
    attention's causal flag (None) alike, over zero queries and keys of `dtype`, whose logits are all
    zero: every key the mask keeps takes an equal weight. The call's own logits would not do: a probe's
    random weights put a row's logits 120 apart, where the softmax weight of a key the mask keeps
    rounds to zero in float32."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    batch, heads, length, _ = keys
    identity = torch.eye(length, dtype=dtype).expand(batch, heads, length, length)
    query, key = (torch.zeros(tuple(shape), dtype=dtype) for shape in (queries, keys))
    weights, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, identity, attention_mask,
                                               scaling=scaling, dropout=0.0, **kwargs)
    return weights.detach().cpu().numpy() > 0


def _masked(module, query, key, attention_mask, scaling, kwargs) -> np.ndarray | None:
    """Which keys each query of a call traced over abstract weights reads (`_seen`'s), `[B, Q, H, K]`:
    its mask's values held as the trace evaluated them, and its other arguments the call's own that are
    no tensors; None where the mask's values are traced, not held."""
    import torch
    from jax.core import Tracer
    from torchax.interop import jax_view

    mask = None
    if attention_mask is not None:
        held = jax_view(attention_mask)
        if isinstance(held, Tracer):
            return None
        mask = torch.from_numpy(np.array(held))
    plain = {name: value for name, value in kwargs.items() if not isinstance(value, torch.Tensor)}
    return _seen(module, query.shape, key.shape, torch.float32, mask, scaling, plain)


def _visible(seen: np.ndarray, window: int | None) -> bool:
    """Whether the keys a call's queries read (`_seen`, `[B, Q, H, K]`) are the causal visibility Dew's
    attention computes with `window` (`dew.nn.visibility.visible`), the last query at the last key: a
    causal LM's (`admitted`)."""
    from dew.nn.visibility import visible

    queries, keys = seen.shape[1], seen.shape[3]
    expected = np.asarray(visible(np.arange(queries)[:, None] + (keys - queries), np.arange(keys)[None, :],
                                  causal=True, window=window))
    return bool(np.array_equal(seen, np.broadcast_to(expected[None, :, None, :], seen.shape)))


def _probe_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers' SDPA attention over the eager mask the probe's model is built with, which every
    attention reads, its own code's too: each call recorded by the probe running (`_Probe.called`)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    probe = _PROBING.get()
    if probe is not None:
        probe.called(module, query, key, attention_mask, scaling, kwargs)
    return ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, attention_mask, scaling=scaling,
                                         dropout=dropout, **kwargs)


@dataclass(frozen=True)
class _Attending:
    """What Dew's attention reads of the call it runs in (`TorchGraph.traced`): the kernel it names,
    each token's document (`[B, S]`, 0 for padding, None for one document a row), the real tokens
    (`[B, S]`, None for all) and the most documents a row holds, whether the call trains, and its
    dropout key (None for none). `visible` `[B, S, S]` is the call's own visibility where it states
    one (a decision row's options, `attention_pairwise_mask`), which then replaces the documents and
    causality; a window then keeps each query to the keys within it of its position (`places`, each
    token's index where None) only where the call states its keys' positions (`key_places`), as Dew's
    own attention reads a stated visibility (`AttentionMetadata.key_positions`)."""

    implementation: AttentionImpl
    documents: jax.Array | None
    valid: jax.Array | None
    max_segments: int
    train: bool
    key: jax.Array | None
    visible: jax.Array | None = None
    places: jax.Array | None = None
    key_places: jax.Array | None = None
    cache: _Cached | None = None
    """A decoding call's cache (`_Cached`): each layer writes its keys and values at the call's slots
    and attends over its row's whole cache."""


@dataclass(frozen=True)
class _Cached:
    """A decoding call's per-layer key and value caches as Dew's attention reads and writes them:
    `held` by `layer_idx` (`cached_key`, `cached_value`, `[rows, capacity, kv_heads, dim]`), the slot
    each token writes `[B, S]` (-1 for a token that writes none) and the filled slots after this call
    `[B, capacity]`; `written` collects each layer's caches after its write, which the traced call
    returns (`TorchGraph`)."""

    held: Mapping[int, Mapping[str, jax.Array]]
    slots: jax.Array
    filled: jax.Array
    written: dict[int, dict[str, jax.Array]] = field(default_factory=dict)


_ATTENDING: contextvars.ContextVar[_Attending | None] = contextvars.ContextVar("dew_attending", default=None)


@contextlib.contextmanager
def _attending(call: _Attending):
    """`call` as what Dew's attention reads (`_dew_attention`) while the forward traces."""
    token = _ATTENDING.set(call)
    try:
        yield call
    finally:
        _ATTENDING.reset(token)


def _documents(laid: Coordinates | None) -> tuple[jax.Array | None, int]:
    """Each token's document `[B, S]` as Dew's attention keeps them apart, and the most a row holds:
    the packed documents, numbered from 1, and every slot the caller's validity drops 0, padding
    that attends as a document of its own; a row of real tokens and padding is one document and its
    padding; None for one document a row."""
    if laid is None or (laid.segments is None and laid.valid is None):
        return None, 1
    if laid.segments is None:
        return jnp.asarray(laid.valid, jnp.int32), 1
    segments = jnp.asarray(laid.segments, jnp.int32)
    if laid.valid is not None:
        segments = jnp.where(jnp.asarray(laid.valid, bool), segments, 0)
    return segments, laid.max_segments


def _dew_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """One layer's attention, as transformers' attention interface calls it, computed by Dew's
    (`dew.nn.attention.scaled_dot_product_attention`): `[B, H, S, D]` queries and `[B, H_kv, S, D]`
    keys and values in, `[B, S, H, D]` out, causal as an admitted model's mask is (`_Probe`),
    inside the call's documents (`_Attending`). A row with padding among its tokens attends over its
    real tokens moved to its front (`dew.nn.local_attention.compacted`), so each document is one run
    of slots and a window counts its real tokens, whichever kernel runs, dropout's included; the
    query carries the ratio of `scaling` to the kernels' own 1/sqrt(D), as Dew's attention layers
    carry a checkpoint's scale (`dew.nn.precision.scaled`); attention dropout draws from the call's
    key folded by the layer, and a training call that drops with no key is refused."""
    from torchax.interop import jax_view, torch_view

    from dew.nn.attention import scaled_dot_product_attention
    from dew.nn.attention_masks import causal_attention_mask
    from dew.nn.kv_cache import write_cache
    from dew.nn.local_attention import compacted
    from dew.nn.visibility import windowed

    call = _ATTENDING.get()
    if call is None:
        raise ValueError("Dew's attention runs inside a tier-3 model's traced call")
    unknown = sorted(_asked(kwargs))
    if unknown:
        # `admitted` runs a model on Dew's attention only where its probe passed no such keyword.
        raise ValueError(f"Dew's attention does not compute {unknown}, which this call passes")
    q, k, v = (jnp.swapaxes(jax_view(tensor), 1, 2) for tensor in (query, key, value))
    if scaling is not None:
        q = scaled(q, scaling * math.sqrt(q.shape[-1]))
    sinks = kwargs.get("s_aux")
    sinks = None if sinks is None else jax_view(sinks)
    window, softcap = kwargs.get("sliding_window"), kwargs.get("softcap")
    max_segments = call.max_segments if call.documents is not None else None
    drawn = bool(dropout) and call.train
    if drawn and call.key is None:
        raise ValueError("this model's training forward drops attention probabilities (dropout) and was "
                         "given no 'dropout' rng")

    rng = jax.random.fold_in(call.key, module.layer_idx) if drawn and call.key is not None else None
    if rng is not None and (step := _PASS.get()):
        # A tier-3 loop's later pass drops its own (`_layer_seam`).
        rng = jax.random.fold_in(rng, step)
    dropping = {"dropout_rate": float(dropout) if drawn else 0.0, "deterministic": not drawn,
                "dropout_rng": rng}

    # Each path runs as a program of its own, attested under the rule a walk of the program's causality
    # reads it by (`sequence_causality.attested`): the cached and the plain or packed paths causal over
    # their keys by construction (`CAUSAL_ATTENTION`), a stated visibility the caller's
    # (`STATED_ATTENTION`). Every array each reads is its argument, none a value of the enclosing trace.
    def attend(q, k, v, *, places, segment_ids, valid, sinks, rng):
        del places, valid  # the documents hold the padding apart, as 0
        return scaled_dot_product_attention(
            q, k, v, implementation=call.implementation, causal=True, sliding_window=window,
            softcap=softcap, sinks=sinks, segment_ids=segment_ids, max_segments=max_segments,
            **{**dropping, "dropout_rng": rng})

    def causal(q, k, v, segment_ids, valid, sinks, rng):
        if valid is None:
            return attend(q, k, v, places=None, segment_ids=segment_ids, valid=None, sinks=sinks, rng=rng)
        return compacted(functools.partial(attend, sinks=sinks, rng=rng), q, k, v, None, segment_ids, valid)

    def cached(q, k, v, held_key, held_value, slots, filled, sinks, rng):
        # The call's tokens join the row's cache at their slots, and each attends over the whole row:
        # what it has filled, causal by slot and within the window, as Dew's decoders attend a cache.
        keys, values = write_cache(held_key, k, slots), write_cache(held_value, v, slots)
        read_keys, read_values = keys.astype(q.dtype), values.astype(q.dtype)
        if q.shape[1] == 1 and not drawn:
            # One token a row reads its row's filled slots before its own, from its window's first: a
            # key count and start a row, which the decode kernels take as Dew's decoders give them
            # (`CausalSelfAttention._decode_attention`); a padding token reads one key, its output unused.
            slot = slots[:, 0]
            reads = jnp.maximum(slot + 1, 1).astype(jnp.int32)
            starts = None if window is None else jnp.maximum(slot + 1 - window, 0).astype(jnp.int32)
            out = scaled_dot_product_attention(q, read_keys, read_values, implementation=call.implementation,
                                               causal=False, softcap=softcap, sinks=sinks,
                                               key_value_seq_lengths=reads, key_value_seq_starts=starts)
        else:
            mask = causal_attention_mask(slots, keys.shape[1], window, key_valid=filled)
            out = scaled_dot_product_attention(q, read_keys, read_values, implementation=call.implementation,
                                               mask=mask, softcap=softcap, sinks=sinks,
                                               **{**dropping, "dropout_rng": rng})
        return out, keys, values

    def stated(q, k, v, visible, sinks, rng):
        return scaled_dot_product_attention(q, k, v, implementation=call.implementation,
                                            mask=visible[:, None], softcap=softcap, sinks=sinks,
                                            **{**dropping, "dropout_rng": rng})

    if call.cache is not None:
        held = call.cache.held.get(module.layer_idx)
        if held is None:
            # A cache being opened: its layer's keys and values give its shape, and nothing is held.
            held = {"cached_key": jnp.zeros_like(k), "cached_value": jnp.zeros_like(v)}
        out, keys, values = attested(cached, CAUSAL_ATTENTION)(q, k, v, held["cached_key"],
                                                               held["cached_value"], call.cache.slots,
                                                               call.cache.filled, sinks, rng)
        call.cache.written[module.layer_idx] = {"cached_key": keys, "cached_value": values}
    elif call.visible is not None:
        # The call's own visibility is the whole of it; a window counts the stated keys' positions.
        visible = jnp.asarray(call.visible, bool)
        if window is not None and call.key_places is not None:
            batch, length = visible.shape[:2]
            places = (jnp.broadcast_to(jnp.arange(length), (batch, length)) if call.places is None
                      else call.places)
            reach = windowed(places[:, :, None], call.key_places[:, None, :], window, causal=False)
            visible = visible & reach
        out = attested(stated, STATED_ATTENTION)(q, k, v, visible, sinks, rng)
    else:
        out = attested(causal, CAUSAL_ATTENTION)(q, k, v, call.documents, call.valid, sinks, rng)
    return torch_view(out.astype(q.dtype)), None


def _experts() -> str:
    """`EXPERTS`, once Dew's implementation is registered under it (`_registered`): transformers checks a
    model's experts implementation against its interface's names as it builds the model."""
    _registered()
    return EXPERTS


@functools.cache
def _registered() -> None:
    """Dew's attention (`_dew_attention`) and the probe's (`_probe_attention`, over transformers' eager
    mask, additive and always built, which a model computing its own attention adds to its scores) in
    transformers' attention interfaces, and Dew's experts (`_dew_experts`) in its experts interface, once a
    process."""
    from transformers import AttentionInterface
    from transformers.integrations.moe import ExpertsInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    AttentionInterface.register(_ATTENTION, _dew_attention)
    AttentionInterface.register(_PROBE, _probe_attention)
    AttentionMaskInterface.register(_PROBE, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    ExpertsInterface.register(EXPERTS, _dew_experts)


_MOVING = ("dynamic", "longrope")
"""The rope types whose tables transformers recomputes from each call's positions (`dynamic_rope_update`)."""


@functools.cache
def _moving_rotary_code():
    """The code of transformers' `dynamic_rope_update` wrapper, which every rotary whose forward it
    decorates shares."""
    from transformers.modeling_rope_utils import dynamic_rope_update

    wrapper = dynamic_rope_update(lambda module, x, position_ids: None)
    if not isinstance(wrapper, FunctionType):
        raise TypeError("transformers' dynamic_rope_update wraps a forward in no Python function")
    return wrapper.__code__


def _table_forward(module: torch.nn.Module) -> FunctionType | None:
    """The forward `dynamic_rope_update` wraps in `module`'s class forward, under any decorator around it
    (`torch.no_grad()`'s): the one that turns the rotary's current table into cos and sin; None where
    the class forward is not so wrapped."""
    function = type(module).forward
    while isinstance(function, FunctionType):
        wrapped = vars(function).get("__wrapped__")
        if function.__code__ is _moving_rotary_code():
            return wrapped if isinstance(wrapped, FunctionType) else None
        function = wrapped
    return None


def functional_rotaries(model: torch.nn.Module) -> int | None:
    """Replace each rotary of `model` whose tables transformers recomputes from each call's positions
    (a 'dynamic' or 'longrope' rope_type under `dynamic_rope_update`, which mutates its buffers from the
    call's largest position and so cannot run in a traced program) with `_functional_rotary`, which
    computes the same tables from each token's reach (`_reaching`) inside the trace. Returns the position
    past which a cache's keys go stale (LongRoPE's original context, which `Serving.cache_rebuild_position`
    names), or None. A rotary that recomputes per layer kind is refused by name."""
    rebuild = None
    for name, module in list(model.named_modules()):
        forward = _table_forward(module)
        if forward is None:
            continue
        rope_type = vars(module)["rope_type"]
        if not isinstance(rope_type, str):
            if moving := sorted(kind for kind, value in rope_type.items() if value in _MOVING):
                raise ValueError(f"{type(module).__name__} recomputes its {moving} layers' tables per call, "
                                 "which tier 3 computes for one rotary a model, not per layer kind")
            continue
        if rope_type not in _MOVING:
            continue
        parent, _, attribute = name.rpartition(".")
        replacement = _functional_rotary()(module, forward)
        setattr(model.get_submodule(parent), attribute, replacement)
        if rope_type == "longrope":
            rebuild = replacement.original
    return rebuild


@functools.cache
def _functional_rotary() -> type:
    """The torch module `functional_rotaries` substitutes (built on first use, so importing this module
    imports no torch)."""
    import torch
    from transformers import PretrainedConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    class FunctionalRotary(torch.nn.Module):
        """A 'dynamic' or 'longrope' rotary computed inside a traced program: each token's tables are the
        ones transformers' rotary computes for a call reaching that token's reach (`_reaching`: its
        document's furthest place, or its row's), with transformers' own initializer (`ROPE_INIT_FUNCTIONS`)
        and amplitude, as `dynamic_rope_update` updates them at batch one. LongRoPE's short and long
        tables are the rotary's own at its original context and one past it; dynamic NTK's is its
        initializer's at the reach, which a traced length broadcasts per token. Each angle lands where
        the replaced class's own forward puts its pair (`layout`: Llama's halves, Cohere's interleave)."""

        def __init__(self, replaced: torch.nn.Module, table_forward: FunctionType):
            super().__init__()
            held = vars(replaced)
            self.config: PretrainedConfig = held["config"]
            self.rope_type: str = held["rope_type"]
            self.attention_scaling: float = held["attention_scaling"]
            self.original = 0
            short = replaced.get_buffer("original_inv_freq").clone()
            if self.rope_type == "longrope":
                self.original = int(self.config.rope_parameters["original_max_position_embeddings"])
                long, _ = ROPE_INIT_FUNCTIONS["longrope"](self.config, None, seq_len=self.original + 1)
                self.register_buffer("short_inv_freq", short, persistent=False)
                self.register_buffer("long_inv_freq", long, persistent=False)
            layout, self.held_dtype = _rotary_layout(replaced, table_forward, short.shape[0])
            self.register_buffer("layout", layout, persistent=False)

        def forward(self, x, position_ids, layer_type=None):
            reach = _reach_of(position_ids)[..., None]
            if self.rope_type == "longrope":
                inv_freq = torch.where(reach + 1 > self.original, self.get_buffer("long_inv_freq"),
                                       self.get_buffer("short_inv_freq"))
            else:
                inv_freq, _ = ROPE_INIT_FUNCTIONS[self.rope_type](self.config, x.device, seq_len=reach + 1)
            freqs = position_ids[..., None].float() * inv_freq.float()
            emb = freqs[..., self.get_buffer("layout")]
            dtype = x.dtype if self.held_dtype is None else self.held_dtype
            return ((emb.cos() * self.attention_scaling).to(dtype=dtype),
                    (emb.sin() * self.attention_scaling).to(dtype=dtype))

    def _rotary_layout(replaced: torch.nn.Module, table_forward: FunctionType, pairs: int
                       ) -> tuple[torch.Tensor, torch.dtype | None]:
        """Which of its `pairs` frequencies each of the replaced rotary's output dimensions carries, read
        off its own `table_forward` at amplitude one over a table of distinct frequencies in (0, 1) at
        position 1, where cos is one to one; and the dtype its tables keep, None where they take the
        input's (Llama's), else the one they hold whatever the input (Ernie 4.5's float32). A rotary
        whose table forward is not that, its cos and sin over the table it holds laid out by frequency
        (a complex `freqs_cis`, a table recomputed from its own initializer, another amplitude), is
        refused by name."""
        probe = copy.deepcopy(replaced)
        distinct = torch.arange(1, pairs + 1, dtype=torch.float32) / (pairs + 1)
        probe.register_buffer("inv_freq", distinct, persistent=False)
        vars(probe)["attention_scaling"] = 1.0
        with torch.no_grad():
            computed = table_forward(probe, torch.zeros(1, dtype=torch.float32),
                                     torch.ones(1, 1, dtype=torch.long))
        refused = ValueError(f"{type(replaced).__name__}'s table forward does not lay its cos and sin out by "
                             "frequency from the table it holds, which tier 3 computes in the trace")
        if not (isinstance(computed, tuple) and len(computed) == 2 and all(
                isinstance(table, torch.Tensor) and not table.is_complex() for table in computed)):
            raise refused
        cos, sin = computed
        layout = (torch.arccos(cos[0, 0].float().clamp(-1, 1)) * (pairs + 1)).round().long() - 1
        if not (bool(((layout >= 0) & (layout < pairs)).all())
                and torch.equal(torch.cos(distinct[layout]).to(cos.dtype), cos[0, 0])
                and torch.equal(torch.sin(distinct[layout]).to(sin.dtype), sin[0, 0])):
            raise refused
        # The tables' dtype for a narrow input: the input's, or one the forward holds whatever it.
        with torch.no_grad():
            narrow, _ = table_forward(probe, torch.zeros(1, dtype=torch.bfloat16),
                                      torch.ones(1, 1, dtype=torch.long))
        return layout, None if narrow.dtype == torch.bfloat16 else narrow.dtype

    return FunctionalRotary


_LOOPED_EXPERTS = frozenset({"transformers.models.longcat_flash.modeling_longcat_flash.LongcatFlashExperts"})
"""Experts transformers runs as a Python loop over the experts a call routes to (`nonzero`, `item()`:
shapes the traced values decide) and does not dispatch through its experts interface
(`use_experts_implementation`): routed experts, a gated product each, beside identity ('zero') experts
numbered after them, as `_functional_experts` computes them."""


def functional_experts(model: torch.nn.Module) -> None:
    """Replace each of `model`'s `_LOOPED_EXPERTS` modules with `_functional_experts`, which computes every
    routed slot at static shapes inside the trace, keeping its parameters under their names."""
    for name, module in list(model.named_modules()):
        if f"{type(module).__module__}.{type(module).__qualname__}" in _LOOPED_EXPERTS:
            parent, _, attribute = name.rpartition(".")
            setattr(model.get_submodule(parent), attribute, _functional_experts()(module))


def _routed_slots(x, index, weights, kernels, *, routed: int, activation: str):
    """Each token's experts, weighted and summed in at least fp32, its slots routed (an index below `routed`,
    the expert's gated product through Dew's grouped dispatch, `dew.nn.moe.expert_dispatch`, over the rows
    sorted by expert: the gate's and up projection's `kernels[:2]` one paired product, `activation`'s gate,
    the down projection `kernels[2]`, each `[experts, in, out]`) or identity (one past them, which the
    dispatch's sentinel holds at zero: the token itself); none routed where `kernels` is empty."""
    from dew.nn.activations import gated_product
    from dew.nn.moe import (
        EXPERT_AXES,
        expert_compute_dtype,
        expert_dispatch,
        expert_projection,
        paired_projections,
    )

    identity = index >= routed
    slots = jnp.zeros((*index.shape, x.shape[-1]), x.dtype)
    if kernels:
        compute = expert_compute_dtype(x, *kernels, dtype=None)

        def project(tokens, sizes, _ids, held):
            gate, up = paired_projections(tokens, held[0], held[1], sizes, compute, "xla", None)
            gated = gated_product(activation)(gate, up)
            # No history: a fallback's experts take no delayed-scaling recipe's state.
            return jnp.asarray(expert_projection(gated, held[2], sizes, compute, "xla", None)), ()

        slots, _ = expert_dispatch(project, x, jnp.where(identity, routed, index), kernels,
                                   tuple(EXPERT_AXES[name] for name in ("gate_proj", "up_proj", "down_proj")),
                                   num_experts=routed, dispatch="global", output_dtype=compute)
    held = jnp.where(identity[..., None], x[:, None, :], slots)
    wide = at_least_fp32(x.dtype)
    return jnp.sum(held.astype(wide) * weights[..., None].astype(wide), axis=1).astype(x.dtype)


def _dew_experts(module, hidden_states, top_k_index, top_k_weights):
    """transformers' experts interface's forward for tier 3 (`EXPERTS`): a gated experts module's routed
    slots through Dew's grouped dispatch (`_routed_slots`), its concatenated gate and up projection
    (`gate_up_proj` `[E, 2I, D]`, or `[E, D, 2I]` transposed) and down projection read as Dew's `[E, in,
    out]` kernels; an index past its experts (an expert-parallel sentinel, whose weight transformers zeroes)
    no slot. A module whose experts Dew's gated product does not compute (biased, ungated, interleaved, a
    gate of its own, an activation Dew gates by none), or one run in torch outside the trace, runs
    transformers' batched products. A program of its
    own, attested token by token (`sequence_causality.TOKENWISE`): its dispatch sorts the tokens by expert,
    and each token's output reads its own input alone."""
    import torchax
    from torchax.interop import jax_view, torch_view
    from transformers.integrations.moe import _default_apply_gate, batched_mm_experts_forward

    from dew.interop.parts import Ambiguous, gated_activation

    gate = inspect.getattr_static(type(module), "_apply_gate", None)
    # Torch's own tensors (a load's probe runs the model in torch) take transformers' products.
    plain = (isinstance(hidden_states, torchax.tensor.Tensor) and module.has_gate and not module.has_bias
             and module.is_concatenated and gate is _default_apply_gate)
    try:
        activation = gated_activation(module.act_fn) if plain else None
    except Ambiguous:
        activation = None
    if activation is None:
        return batched_mm_experts_forward(module, hidden_states, top_k_index, top_k_weights)
    gate_up, down = jax_view(module.gate_up_proj), jax_view(module.down_proj)
    if module.is_transposed:
        width = gate_up.shape[2] // 2
        kernels = (gate_up[..., :width], gate_up[..., width:], down)
    else:
        width = gate_up.shape[1] // 2
        kernels = (jnp.swapaxes(gate_up[:, :width], 1, 2), jnp.swapaxes(gate_up[:, width:], 1, 2),
                   jnp.swapaxes(down, 1, 2))
    x, index, weights = (jax_view(tensor) for tensor in (hidden_states, top_k_index, top_k_weights))
    slots = functools.partial(_routed_slots, routed=int(module.num_experts), activation=activation)
    return torch_view(attested(slots, TOKENWISE)(x, index, weights, kernels))


@functools.cache
def _functional_experts() -> type:
    """The torch module `functional_experts` substitutes (built on first use)."""
    import torch

    from dew.interop.parts import gated_activation

    class FunctionalExperts(torch.nn.Module):
        """Routed experts beside identity experts at static shapes, through Dew's grouped expert
        dispatch (`_routed_slots`), as transformers' loop computes them expert by expert. Its parameters
        are the replaced module's; one of identity experts alone holds none."""

        def __init__(self, replaced: torch.nn.Module):
            super().__init__()
            self.routed: int = vars(replaced)["num_routed_experts"]
            self.activation = gated_activation(replaced.get_submodule("act_fn"))
            if self.routed:
                self.gate_up_proj: torch.Tensor = replaced.get_parameter("gate_up_proj")
                self.down_proj: torch.Tensor = replaced.get_parameter("down_proj")

        def forward(self, hidden_states, top_k_index, top_k_weights):
            from torchax.interop import jax_view, torch_view

            x, index, weights = (jax_view(tensor) for tensor in (hidden_states, top_k_index, top_k_weights))
            kernels = ()
            if self.routed:
                # Under the trace these are its views of the parameters, not torch's parameters.
                gate_up = jax_view(self.gate_up_proj)[:self.routed]
                width = gate_up.shape[1] // 2
                kernels = (jnp.swapaxes(gate_up[:, :width], 1, 2), jnp.swapaxes(gate_up[:, width:], 1, 2),
                           jnp.swapaxes(jax_view(self.down_proj), 1, 2))
            # A program of its own, attested token by token (`_dew_experts`).
            slots = functools.partial(_routed_slots, routed=self.routed, activation=self.activation)
            return torch_view(attested(slots, TOKENWISE)(x, index, weights, kernels))

    return FunctionalExperts


def functional_parts(model: torch.nn.Module) -> int | None:
    """Replace `model`'s parts a traced program cannot run as transformers writes them with ones computing
    the same inside the trace (`functional_rotaries`, `functional_experts`). Returns the cache's rebuild
    position (`functional_rotaries`)."""
    functional_experts(model)
    return functional_rotaries(model)


_REACH: contextvars.ContextVar[jax.Array | None] = contextvars.ContextVar("dew_reach", default=None)
"""Each token's reach `[B, S]` in the traced call (`_reaching`), which a functional rotary reads."""


@contextlib.contextmanager
def _reaching(reach: jax.Array | None):
    """Run a traced call whose tokens reach `reach` `[B, S]`: each token's document's furthest place
    (`dew.nn.inputs.document_reach`) in a laid call, each decoding row's furthest token in a decoding
    one; None for a plain call, whose rows each reach their furthest position."""
    token = _REACH.set(reach)
    try:
        yield
    finally:
        _REACH.reset(token)


def _reach_of(position_ids):
    """Each token's reach, a torch view of `_REACH`, or its row's largest position."""
    from torchax.interop import torch_view

    reach = _REACH.get()
    if reach is None:
        return position_ids.max(dim=-1, keepdim=True).values.expand_as(position_ids)
    return torch_view(reach)


def refuse_quantized(config: Mapping[str, object]) -> None:
    """Refuse a source whose config declares a quantization: tier 3 reads float weights only."""
    quantization = config.get("quantization_config")
    if quantization is not None:
        method = quantization.get("quant_method") if isinstance(quantization, Mapping) else None
        raise ValueError(
            f"tier 3 reads float weights only, and this source declares a "
            f"{method or 'quantization'} quantization_config; load the model's BF16 or FP32 "
            f"release, or drop fallback for a registered family whose codec Dew reads")


def planned_axes(model: PreTrainedModel) -> tuple[tuple[str, LogicalAxes], ...]:
    """The logical axes of `model`'s parameters by their torch names, from its tensor-parallel plan
    (`model.tp_plan`: its config's `base_model_tp_plan` beside its head's), each parameter's style found
    by its module's name with its layer indices read as `*` and named by transformers' registry
    (`ALL_PARALLEL_STYLES`, which owns the styles' names): a column-split projection's
    (`ColwiseParallel`, `PackedColwiseParallel`: an `nn.Linear` weight `[out, in]` split on its first
    dimension, its bias with it) output width `mlp` over its input `embed`, a row-split one's
    (`RowwiseParallel`, `PackedRowwiseParallel`: its second, its bias whole) the other way, and the input
    and output tables `vocab` over `embed`. Dew's rules then place them as a native model's (`mlp` on the
    tensor axis, `embed` on fsdp). A parameter no style splits so (the plan's expert, router and
    replicated styles, a norm) takes `Layout`'s shape heuristic."""
    import torch
    from transformers.distributed.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        ColwiseParallel,
        PackedColwiseParallel,
        PackedRowwiseParallel,
        RowwiseParallel,
    )

    def split(style: str | None) -> str | None:
        layer = None if style is None or style not in ALL_PARALLEL_STYLES else ALL_PARALLEL_STYLES[style]
        if isinstance(layer, ColwiseParallel | PackedColwiseParallel):
            return "column"
        return "row" if isinstance(layer, RowwiseParallel | PackedRowwiseParallel) else None

    plan = model.tp_plan
    tables = [module.weight for module in (model.get_input_embeddings(), model.get_output_embeddings())
              if isinstance(module, torch.nn.Module) and isinstance(module.weight, torch.Tensor)]
    axes: list[tuple[str, LogicalAxes]] = []
    for name, parameter in model.named_parameters():
        module, _, kind = name.rpartition(".")
        # A layer's index, a name of digits alone, reads as the plan's `*` (`dense_h_to_4h` keeps its 4).
        style = split(plan.get(".".join("*" if part.isdigit() else part for part in module.split("."))))
        if parameter.ndim == 2 and any(parameter is table for table in tables):
            axes.append((name, ("vocab", "embed")))
        elif style == "column" and parameter.ndim in (1, 2):
            axes.append((name, ("mlp", "embed")[:parameter.ndim]))
        elif style == "row" and kind == "weight" and parameter.ndim == 2:
            axes.append((name, ("embed", "mlp")))
    return tuple(axes)


@dataclass(frozen=True)
class TorchLayout(Layout):
    """A `Layout` for a tree keyed by torch parameter names, which a tier-3 model makes of the run's
    (`TorchCausalLM.placement`).

    `names` gives a tensor's logical axes by its torch name, the model's
    tensor-parallel plan's (`planned_axes`); `rules` then map those onto the
    mesh exactly as they do for a native model, so the tensor axis, fsdp,
    `min_shard` and `check` mean what they mean there. A tensor it names none
    for takes `Layout`'s shape heuristic. Optimizer moments and EMA copies end
    in their parameter's name, so they take its placement.

    The placement decides where GSPMD keeps each tensor, not what the forward
    computes: a spec that suits the model badly costs collectives, never
    correctness.
    """

    names: tuple[tuple[str, LogicalAxes], ...] = ()

    def axes(self, name: str, ndim: int) -> LogicalAxes | None:
        """The logical axes `names` gives `name` at rank `ndim`, or None."""
        return next((axes for held, axes in self.names if held == name and len(axes) == ndim), None)

    def _axes(self, path, ndim: int) -> LogicalAxes | None:
        """The axes the name table gives the leaf's torch name, else `Layout`'s."""
        names = parameter_path(path)
        axes = self.axes(names[-1], ndim) if names else None
        return super()._axes(path, ndim) if axes is None else axes


@dataclass(frozen=True, eq=False)
class TorchGraph:
    """The transformers module torchax runs, as a function of its tensors.

    `persistent` names the buffers the source checkpoint stores. A training
    forward runs the module's own training forward, its dropout and other
    draws taking `key` as torchax's random key; one that draws on the host
    instead (OPT's layer drop, `torch.rand([])`) is decided once at trace
    time, and is refused. `dew_attention` says the module is built with Dew's attention (`_ATTENTION`),
    which runs the kernel the run's context names (`chosen_attention`); without it the module runs
    transformers' own.
    """

    module: JittableModule
    persistent: tuple[str, ...]
    dew_attention: bool = False
    shapes: Mapping[str, Mapping[str, jax.ShapeDtypeStruct]] = field(default_factory=dict)
    """The source's variables, `params` and `buffers`, as shapes and dtypes: what an initialization,
    which holds none of them, traces the forward over (`TorchCausalLM._run`)."""
    axes: tuple[tuple[str, LogicalAxes], ...] = ()
    """Its parameters' logical axes by their torch names, its tensor-parallel plan's (`planned_axes`),
    which its placement reads (`TorchCausalLM.placement`)."""
    certified: dict[tuple, str | None] | None = None
    """For a source tier 3 runs on Dew's attention (`TorchAdmission.attends`): each program
    specialization it has run, by the refusal its walk gave (None where causal); None for one on
    transformers' own attention, whose causality its admission's walk decided at the probe's length
    (`admitted`), or one loaded by `fallback="torchax"`, whose causality its caller vouches for
    (`certify`)."""

    def __call__(self, params: Mapping[str, jax.Array], buffers: Mapping[str, jax.Array],
                 tokens: jax.Array, *, train: bool, key: jax.Array | None = None,
                 precision: jax.lax.PrecisionLike = None, laid: Coordinates | None = None,
                 visible: jax.Array | None = None, key_places: jax.Array | None = None,
                 head_input: jax.Array | None = None, recomputed: bool = False,
                 loop: Loop | None = None) -> tuple[jax.Array, jax.Array]:
        """The logits and the output head's input states for `tokens`, in Dew's torchax
        environment (`_environment`), every product at `precision` (`configured_precision`):
        torch's operators state none. Each broadcast of a value narrower than fp32 sums
        its gradient in fp32 (`_accumulating`), as torch's reduction of a broadcast's
        gradient does. `laid` places the tokens in their documents, which Dew's attention
        keeps apart and whose positions transformers' rotary reads; `visible` `[B, S, S]`, where
        given, is what each token sees in their place, and `key_places` `[B, S]` the keys' positions a
        window counts (`_Attending`). `head_input`, where
        given, is the output head's input in place of the states the forward computes, so the
        logits are the head's over it, every step the forward takes after the head included. `recomputed`
        recomputes each decoder layer in the backward (`TorchCausalLM.remat`); `loop` runs a block of the
        layers over its own output (`TorchCausalLM.loop`)."""
        documents, max_segments = _documents(laid)
        positions = None if laid is None else laid.places()
        valid = None if laid is None or laid.valid is None else jnp.asarray(laid.valid, bool)
        arguments = (dict(params), dict(buffers), tokens, key, positions, documents, valid, visible,
                     key_places, head_input, None, None)
        (logits, hidden), _ = self._run(arguments, train=train, precision=precision,
                                        max_segments=max_segments, recomputed=recomputed, loop=loop)
        return logits, hidden

    def step(self, params: Mapping[str, jax.Array], buffers: Mapping[str, jax.Array], tokens: jax.Array, *,
             cache: Mapping[int, Mapping[str, jax.Array]], slots: jax.Array, filled: jax.Array,
             positions: jax.Array, reach: jax.Array | None = None, precision: jax.lax.PrecisionLike = None
             ) -> tuple[jax.Array, dict[int, dict[str, jax.Array]]]:
        """A decoding call: the logits for `tokens` at their `positions` `[B, S]` (which transformers'
        rotary reads), each layer writing its keys and values into `cache` at `slots` and attending
        over its row's `filled` slots (`_Cached`), and every layer's cache after the call. `reach` is
        each token's reach (`_reaching`), its row's furthest position where None."""
        (logits, _), written = self._run(
            (dict(params), dict(buffers), tokens, None, positions, None, None, None, None, None,
             (dict(cache), slots, filled), reach), train=False, precision=precision, max_segments=1)
        return logits, written

    def _run(self, arguments: tuple, *, train: bool, precision: jax.lax.PrecisionLike, max_segments: int,
             recomputed: bool = False, loop: Loop | None = None):
        layers: list = []
        traced, shapes, draws = self.traced(*arguments, train=train, precision=precision,
                                            max_segments=max_segments, layers=layers if recomputed else None,
                                            loop=loop)
        self.certify(traced, arguments, train=train, max_segments=max_segments, loop=loop)
        outputs = jax.tree.unflatten(jax.tree.structure(shapes),
                                     _accumulating(traced, *jax.tree.leaves(arguments), recomputed=layers))
        key = arguments[3]
        if draws.host:
            raise ValueError(
                f"tier 3 traces this model's training forward once, and it draws on the host "
                f"({draws.host} draws, e.g. a layer drop), which tracing would fix for every step; set "
                f"that rate to 0 in its config, or train a family Dew loads natively")
        if (draws.device or draws.keyed) and key is None:
            # Unseeded, its draws would read the environment's one key, fixed by the trace for every step.
            raise ValueError("this model's training forward draws (dropout) and was given no 'dropout' rng")
        return outputs

    def certify(self, traced, arguments: tuple, *, train: bool, max_segments: int,
                loop: Loop | None = None) -> None:
        """For a source tier 3 runs on Dew's attention (`certified`), prove the program about to run,
        `traced` over `arguments`, causal before it runs: its logits and states (its first two outputs)
        read at a position no later token (`sequence_causality.causality_refusal`), once a specialization
        (its arguments' structure, shapes and dtypes, train or not, its documents' bound). Refused where an
        output position reads a later one."""
        if self.certified is None:
            return
        leaves, structure = jax.tree.flatten(arguments)
        signature = (structure, tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves), train,
                     loop, max_segments)
        if signature not in self.certified:
            tokens = len(jax.tree.leaves(arguments[0])) + len(jax.tree.leaves(arguments[1]))
            self.certified[signature] = causality_refusal(traced, tokens, int(arguments[2].shape[1]), (0, 1))
        refusal = self.certified[signature]
        if refusal is not None:
            raise ValueError(f"tier 3 runs this source causal in each program its walk proves, and this one "
                             f"is not: {refusal}")

    def traced(self, params: Mapping[str, Traceable], buffers: Mapping[str, Traceable], tokens: Traceable,
               key: jax.Array | None,
               positions: jax.typing.ArrayLike | None = None, documents: jax.typing.ArrayLike | None = None,
               valid: jax.typing.ArrayLike | None = None, visible: jax.typing.ArrayLike | None = None,
               key_places: jax.typing.ArrayLike | None = None, head_input: jax.typing.ArrayLike | None = None,
               cache: tuple | None = None, reach: jax.typing.ArrayLike | None = None, *, train: bool,
               precision: jax.lax.PrecisionLike = None, max_segments: int = 1, layers: list | None = None,
               loop: Loop | None = None):
        """The forward over these arguments traced once (`jax.make_jaxpr`), its output shapes, and the
        draws it made (`_Draws`): on torchax's device, a key's, or on the host, fixed by the trace. Its
        outputs are the forward's and, for a decoding call (`cache`, the held caches, slots and filled
        slots of `_Cached`), every layer's cache after it. Where `layers` is given, each decoder layer
        traces as a program of its own, which `layers` takes; `loop` runs a block of the layers over its own
        output (`_layer_seam`)."""
        env = _environment()
        draws = _Draws()

        def forward(params, buffers, tokens, key, positions, documents, valid, visible, key_places,
                    head_input, cache, reach):
            arguments = env.j2t_iso((dict(params), dict(buffers), tokens, positions, head_input))
            if reach is None and documents is not None:
                reach = document_reach(documents, valid, positions)
            cached = None if cache is None else _Cached(*cache)
            attending = (contextlib.nullcontext() if not self.dew_attention else _attending(
                _Attending(chosen_attention(), documents, valid, max_segments, train, key, visible,
                           positions, key_places, cached)))
            self.module.train(train)
            seam = (contextlib.nullcontext() if layers is None and loop is None
                    else _layer_seam(self.module._model, env, draws, layers=layers, loop=loop))
            try:
                # The runtime property scoped to this trace whatever it is handed, so a draw inside (a refused
                # nested layer's) leaves no key of its trace in the environment past it.
                with (env, env.override_property(prng=env.prng_key if key is None else key),
                      draws, attending, _reaching(reach), seam):
                    held = env.prng_key
                    outputs = env.t2j_iso(self.module.functional_call(_forward, *arguments))
                    draws.keyed = env.prng_key is not held
            finally:
                self.module.eval()
            return outputs, ({} if cached is None else cached.written)

        with configured_precision(precision):
            closed, shapes = jax.make_jaxpr(forward, return_shape=True)(
                dict(params), dict(buffers), tokens, key, positions, documents, valid, visible, key_places,
                head_input, cache, reach)
        return closed, shapes, draws

    def export(self, model: nn.Module, variables: Mapping[str, object],
               config: Mapping[str, object]) -> dict[str, np.ndarray]:
        """The source's tensors under its own names: the parameters and the
        buffers its checkpoint stores. `Pretrained.save` writes them."""
        params = variables["params"]
        buffers = variables.get("buffers", {})
        if not isinstance(params, Mapping) or not isinstance(buffers, Mapping):
            raise TypeError("a torchax model's variables are {'params': ..., 'buffers': ...} "
                            "keyed by torch names, as Pretrained.load returns them")
        tensors = {name: np.asarray(leaf) for name, leaf in params.items()}
        tensors.update({name: np.asarray(buffers[name]) for name in self.persistent})
        return tensors


def _Draws():
    """A torch function mode counting the calls of torch's random functions,
    on torchax's device (`device`) and on the host (`host`)."""
    import torch
    import torchax
    from torch.overrides import TorchFunctionMode

    drawing = {torch.rand, torch.rand_like, torch.randn, torch.randn_like, torch.randint, torch.randint_like,
               torch.randperm, torch.bernoulli, torch.multinomial, torch.normal, torch.Tensor.uniform_,
               torch.Tensor.normal_, torch.Tensor.bernoulli_, torch.Tensor.random_, torch.Tensor.exponential_,
               torch.Tensor.cauchy_, torch.Tensor.geometric_, torch.Tensor.log_normal_}

    class Draws(TorchFunctionMode):
        device = host = 0
        keyed = False
        """Whether torchax's key moved over the forward (`TorchGraph.traced`): a draw on its device by
        whatever dispatch reached it, ATen's dropout among them, which no function above names."""

        def __torch_function__(self, func, types, args=(), kwargs=None):
            tensor = func(*args, **(kwargs or {}))
            if func in drawing:
                if isinstance(tensor, torchax.tensor.Tensor):
                    self.device += 1
                else:
                    self.host += 1
            return tensor

    return Draws()


_OPMATH = ("gelu", "silu", "native_layer_norm", "native_group_norm", "_softmax", "_log_softmax")
"""The aten ops torchax lowers into several roundings of a narrow float (jax.nn's gelu, silu,
softmax and log_softmax, and its layer and group norms' mean, variance and normalization),
where torch's kernels compute a bf16 or fp16 operand in fp32 and round each output once. Which ops
torch computes at its opmath is a fact of ATen's compiled kernels (`at::opmath_type`), which no
schema or Python registry of the installed torch carries; the argument that keeps an op's output
fp32 (`half_to_float`) its schema states (`_keeps_fp32`). Left to torchax, a GPT-NeoX's LayerNorm
took the variance of its bf16-rounded deviations, which scaled its logits by 1 - 8.9e-4 and put its
bf16 loss 7.7 times torch's distance from float64."""

_KEEPS_FP32 = "half_to_float"
"""The aten argument that keeps a softmax's output fp32 for a narrow input."""


def _keeps_fp32(op) -> tuple[int, str] | None:
    """The position and name of `op`'s argument that keeps its output fp32 (`_KEEPS_FP32`), as its schema
    states them, or None where it has none."""
    names = [argument.name for argument in op._schema.arguments]
    return (names.index(_KEEPS_FP32), _KEEPS_FP32) if _KEEPS_FP32 in names else None


@functools.cache
def _environment():
    """The torchax environment Dew's fallback runs transformers' forward in: torchax's own
    lowerings, but the `_OPMATH` ops computed at torch's opmath (`_at_opmath`). Its own, so
    the process's default environment keeps torchax's."""
    import torch
    from torchax.ops import ops_registry
    from torchax.tensor import Environment

    env = Environment()
    for name in _OPMATH:
        packet = getattr(torch.ops.aten, name)
        keeps_fp32 = _keeps_fp32(packet.default)
        # Every key torchax registers the op under: its packet, or one of the packet's overloads.
        registered = ops_registry.all_aten_ops
        for op in (packet, *(getattr(packet, overload) for overload in packet.overloads())):
            if op in registered:
                env.override_op_definition(op, _at_opmath(env, registered[op].func, keeps_fp32))
    return env


def _at_opmath(env, lowering, keeps_fp32: tuple[int, str] | None):
    """`lowering`, torchax's jax function for one aten op, run as torch's kernel runs a narrow
    float: its floating operands narrower than fp32 widened to fp32, its floating outputs
    rounded once to their dtype, unless the argument at `keeps_fp32` asks for fp32 outputs."""
    def computed(*args, **kwargs):
        args, kwargs = env.t2j_iso((args, kwargs))
        narrow = [leaf.dtype for leaf in jax.tree.leaves((args, kwargs))
                  if isinstance(leaf, jax.Array) and _narrow_float(leaf)]
        if not narrow:
            return env.j2t_iso(lowering(*args, **kwargs))
        wide_args, wide_kwargs = jax.tree.map(
            lambda leaf: leaf.astype(jnp.float32) if isinstance(leaf, jax.Array) and _narrow_float(leaf)
            else leaf, (args, kwargs))
        outputs = lowering(*wide_args, **wide_kwargs)
        position, name = keeps_fp32 or (len(args), "")
        kept = bool(args[position] if position < len(args) else kwargs.get(name, False))
        if not kept:
            outputs = jax.tree.map(lambda leaf: leaf.astype(narrow[0]) if isinstance(leaf, jax.Array)
                                   and leaf.dtype == jnp.float32 else leaf, outputs)
        return env.j2t_iso(outputs)

    return computed


def _accumulating(closed: Jaxpr, *arguments: jax.Array, recomputed: Sequence[Jaxpr] = ()) -> list[jax.Array]:
    """`closed` evaluated on `arguments` with each broadcast of a floating value
    narrower than fp32 summing its gradient in fp32 (`dew.nn.precision.broadcast_summed`): an explicit
    `broadcast_in_dim`, and the size-1 dimensions an elementwise operation
    broadcasts implicitly (`_BROADCASTING`), each made explicit first; a jit's body
    is inlined so its broadcasts are found too. jax transposes a broadcast into a
    sum in the broadcast value's own dtype (jax-ml/jax#30310), where torch's
    autograd reduces a bf16 gradient in fp32 and rounds the sum once, so a norm
    weight's or a bias's gradient summed in bf16 sat 2-5 times further from
    float64 than torch's."""

    def evaluate(jaxpr, consts, values):
        held = dict(zip(jaxpr.constvars, consts, strict=True)) | dict(zip(jaxpr.invars, values, strict=True))
        # Each value is dropped after the last equation that reads it, as XLA would free it.
        last = {var: index for index, equation in enumerate(jaxpr.eqns)
                for var in equation.invars if not isinstance(var, Literal)}
        kept = {var for var in jaxpr.outvars if not isinstance(var, Literal)}

        def read(var):
            return var.val if isinstance(var, Literal) else held[var]

        for index, equation in enumerate(jaxpr.eqns):
            inputs = [read(var) for var in equation.invars]
            for var in equation.invars:
                if not isinstance(var, Literal) and last[var] == index and var not in kept:
                    held.pop(var, None)
            if (inner := purely_called(equation)) is not None and any(inner is layer for layer in recomputed):
                # A decoder layer a step recomputes in its backward (`TorchCausalLM.remat`).
                outputs = jax.checkpoint(functools.partial(_evaluated, evaluate, inner))(*inputs)
            elif inner is not None:
                outputs = evaluate(inner.jaxpr, inner.consts, inputs)
            elif equation.primitive is primitives.broadcast_in_dim_p and _narrow_float(inputs[0]):
                outputs = [broadcast_summed(inputs[0], tuple(equation.params["shape"]),
                                            tuple(equation.params["broadcast_dimensions"]))]
            else:
                if equation.primitive in _BROADCASTING:
                    shape = equation.outvars[0].aval.shape
                    rank = len(shape)
                    # An operand broadcasts on its trailing dimensions: a scalar on none.
                    inputs = [broadcast_summed(value, shape, tuple(range(rank - jnp.ndim(value), rank)))
                              if _narrow_float(value) and jnp.shape(value) != shape else value
                              for value in inputs]
                # As `jax.core.eval_jaxpr` binds an equation.
                params = equation.primitive.get_bind_params(equation.params)
                with equation.ctx.manager:
                    outputs = equation.primitive.bind(*inputs, **params)
                outputs = outputs if equation.primitive.multiple_results else [outputs]
            held.update(zip(equation.outvars, outputs, strict=True))
        return [read(var) for var in jaxpr.outvars]

    return evaluate(closed.jaxpr, closed.consts, arguments)


def _evaluated(evaluate, inner: Jaxpr, *inputs: jax.Array) -> list[jax.Array]:
    """`inner` evaluated over `inputs` by `evaluate` (`_accumulating`'s), as a function `jax.checkpoint`
    recomputes."""
    return evaluate(inner.jaxpr, inner.consts, inputs)


def _layer_list(model):
    """`model`'s decoder layers (its decoder's `layers`, a `ModuleList`), None where it holds none so."""
    import torch

    stack = vars(model.get_decoder()).get('_modules', {}).get('layers')
    return stack if isinstance(stack, torch.nn.ModuleList) else None


_PASS: contextvars.ContextVar[int] = contextvars.ContextVar("dew_tier3_pass", default=0)
"""The pass of a tier-3 loop a replayed layer runs in (`_layer_seam`), which Dew's attention folds into its
dropout's key past the first (`_dew_attention`), so each pass drops its own."""


def runnable_loop(loop: Loop) -> Loop:
    """`loop` where a tier-3 source can run it: its steps, block and step norm alone (`TorchCausalLM.loop`);
    a gate, a gradient cut, messages, θ rows, exits or a shared cache refused."""
    stated, plain = vars(loop), vars(Loop(steps=loop.steps))
    unrun = sorted(name for name in stated
                   if name not in ('steps', 'layers', 'step_norm') and stated[name] != plain[name])
    if unrun:
        raise ValueError(f"a tier-3 source loops its layers alone, and this loop states {unrun}")
    return loop


@contextlib.contextmanager
def _layer_seam(model, env, draws, *, layers: list | None, loop: Loop | None) -> Iterator[None]:
    """Each of `model`'s decoder layers called through one seam while it traces, its own forward restored
    after. Under `layers`, each call traced as a program of its own over its tensor arguments
    (`dew.nn.nested.nested`), each program `layers` takes: what `_accumulating` recomputes in a step's
    backward (`TorchCausalLM.remat`); a layer that draws (its dropout) is refused, as its program would hold
    the draw's key, which torchax carries past it (its key moved, whatever dispatch drew). Under `loop`,
    the block `loop.layers` runs `loop.steps`
    times: past its last layer's first call, its later passes, each layer called with the arguments its own
    first pass took (the call's masks and rotary tables, which differ by layer kind), its hidden input the
    pass before's output, the decoder's `norm` between passes under `step_norm`, each pass's attention
    dropout its own (`_PASS`)."""
    import torch
    from torch.utils import _pytree

    from dew.nn.nested import nested

    held, stack = model.get_decoder(), _layer_list(model)
    if stack is None:
        raise ValueError(f"tier 3 runs a source's decoder layers apart, and {type(held).__name__} holds no "
                         "`layers` list")
    first, end = (0, len(stack)) if loop is None or loop.layers is None else loop.layers
    if loop is not None and not 0 <= first < end <= len(stack):
        raise ValueError(f"a loop's block of layers lies in the decoder's {len(stack)}, not [{first}, {end})")
    norm = vars(held).get('_modules', {}).get('norm')
    if loop is not None and loop.step_norm and norm is None:
        raise ValueError(f"a loop's step norm is the decoder's final norm, and {type(held).__name__} holds "
                         "no `norm`")
    captured: dict[int, tuple] = {}
    replaying: list[bool] = []

    def state_of(output):
        return output if isinstance(output, torch.Tensor) else output[0]

    def traced_apart(forward, args, kwargs, programs: list):
        leaves, structure = _pytree.tree_flatten((args, kwargs))
        tensors = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]

        def run(*values):
            given = list(leaves)
            for index, value in zip(tensors, values, strict=True):
                given[index] = env.j2t_iso(value)
            args, kwargs = _pytree.tree_unflatten(given, structure)
            return env.t2j_iso(forward(*args, **kwargs))
        # A draw inside advances torchax's key (`env.param.prng`), which would carry the program's key past
        # it, through whatever torch dispatch reached the draw.
        key = env.prng_key
        out = nested(run, traced=programs.append)(*(env.t2j_iso(leaves[index]) for index in tensors))
        if env.prng_key is not key:
            raise ValueError("tier 3 recomputes each decoder layer from its inputs, and this source's layers "
                             "draw (dropout): set its rates to 0, or train it unrecomputed")
        return env.j2t_iso(out)

    def replayed(output, loop: Loop):
        replaying.append(True)
        try:
            for step in range(1, loop.steps):
                state = norm(state_of(output)) if loop.step_norm else state_of(output)
                token = _PASS.set(step)
                try:
                    for index in range(first, end):
                        args, kwargs = captured[index]
                        if args:
                            output = stack[index](state, *args[1:], **kwargs)
                        else:
                            output = stack[index](**{**kwargs, 'hidden_states': state})
                        state = state_of(output)
                finally:
                    _PASS.reset(token)
            return output
        finally:
            replaying.clear()

    def seam(index, forward):
        def call(*args, **kwargs):
            looping = loop is not None and first <= index < end and not replaying
            if looping:
                captured[index] = (args, kwargs)
            if layers is None:
                output = forward(*args, **kwargs)
            else:
                output = traced_apart(forward, args, kwargs, layers)
            if loop is not None and looping and index == end - 1:
                return replayed(output, loop)
            return output
        return call

    own = {index: vars(layer).get('forward') for index, layer in enumerate(stack)}
    for index, layer in enumerate(stack):
        layer.forward = seam(index, layer.forward)
    try:
        yield
    finally:
        for index, layer in enumerate(stack):
            if own[index] is None:
                del layer.forward
            else:
                layer.forward = own[index]


_BROADCASTING = frozenset((primitives.add_p, primitives.sub_p, primitives.mul_p, primitives.div_p,
                           primitives.max_p, primitives.min_p, primitives.pow_p, primitives.atan2_p,
                           primitives.rem_p, primitives.nextafter_p, primitives.select_n_p,
                           primitives.clamp_p))
"""The elementwise primitives that broadcast an operand's size-1 dimensions implicitly."""


def _narrow_float(value) -> bool:
    """Whether `value` is a floating array narrower than fp32."""
    dtype = jnp.result_type(value)
    return jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits < 32


def _forward(model: PreTrainedModel, tokens: torch.Tensor, positions: torch.Tensor | None = None,
             head_input: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `tokens` at `positions` (each token's index where None), keeping what its
    output head reads, or, given `head_input`, reading that into its head in place of its states."""
    import torch

    states: list[torch.Tensor] = []
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Module):
        raise ValueError(f"{type(model).__name__} has no output embeddings for a causal LM's logits")

    def read(_, inputs):
        states.append(inputs[0])
        return None if head_input is None else (head_input, *inputs[1:])

    hook = head.register_forward_pre_hook(read)
    try:
        logits = model(input_ids=tokens, position_ids=positions, use_cache=False).logits
    finally:
        hook.remove()
    return logits, states[0]


class TorchCausalLM(nn.Module):
    """A transformers causal LM as a Flax module, run by torchax.

    `apply(variables, tokens)` returns logits in at least float32, a float64
    model's in float64 (`at_least_fp32`). The variables are the
    source's own tensors under their torch names, `params` and `buffers`;
    `init` has no initializer to draw from, so the module starts from the
    loaded variables only. Floating parameters are cast to `dtype` for the
    forward, which keeps them in their stored dtype as masters; buffers keep
    theirs. A training forward runs the module's own, its dropout drawing
    from the 'dropout' rng (`TorchGraph`).

    A loss scores its whole logits (`Logits`). A model whose logits are the
    plain head product loads as `TorchTableLM` instead, which `LMObjective`
    scores through its head input and head weight. A model whose attention
    runs on Dew's (`TorchGraph.dew_attention`) reads packed and padded rows:
    their `Coordinates`, or the `segment_ids`, `positions` and validity they
    are built from, keep each document's attention to itself and give
    transformers' rotary each token's place in its document (`_laid`); one on
    transformers' own attention reads plain token rows, refusing a packing
    field by name.

    A model on Dew's attention decodes (`init_cache`, `__call__(..., decode=True)`) as Dew's decoders
    do: each attention layer's keys and values join a cache of `max_seq_len` slots a row, under the
    names Dew's caches use (`cache_index`, the row's filled slots; `cached_key`, `cached_value`), at
    the slots its valid tokens fill in order, transformers' rotary reading the tokens' `positions`
    (their slots where none are given), and each token attends over its row's filled slots up to its
    own. One on transformers' own attention has no cache Dew's attention writes, and refuses one.

    `hidden_states` are the states the output head reads, the final norm's
    output, which a reader of a backbone's states takes (a decision readout,
    a probe), and `logits_from_hidden` the head over given states. Its
    initialization declares no variables, the source's being its own, and
    returns the forward's shape and dtype over them (`TorchGraph.shapes`) as
    zeros: what a reader sizing itself to the states traces (`DecisionModel.head_size`).
    """

    graph: TorchGraph
    vocab_size: int
    dtype: Dtype | None = None
    final_logit_softcap: float | None = None
    precision: PrecisionLike = None
    max_seq_len: int = 2048
    """The slots a decode cache holds a row, the tokens a row may reach (`Bounded`): `load` sets the
    source's own context, `max_position_embeddings`, or the load's `max_seq_len`."""
    rebuild: int | None = None
    """The position past which cached keys go stale, LongRoPE's original context, where the source's
    rotary switches tables there (`functional_rotaries`), or None (`Serving.cache_rebuild_position`)."""
    remat: bool = False
    """Each of the source's decoder layers (its decoder's `layers`) recomputed in a training step's
    backward from its inputs alone (`jax.checkpoint` over the layer's own program, `_layers_apart`),
    keeping nothing inside it: `REMAT_POLICIES['full']`. torch's code marks none of the residuals a
    named policy keeps, so this is the one rung a tier-3 model climbs (`recompute_more`)."""

    loop: Loop | None = None
    """A block of the source's decoder layers run `loop.steps` times over its own output (`Loop`), the
    layers before and after it once, every pass reading the call's positions and masks, the decoder's
    final norm (its `norm`) between passes under `step_norm` (`_layer_seam`). A tier-3 source holds no
    exit gate, cross-pass messages or θ rows Dew would draw, so a loop stating any is refused as it runs
    (`runnable_loop`), as is a cached call (its cache keeps one entry a layer); a run sets it through
    `Looping` (`with_loop`, `with_passes`), which declares no parameter the source lacks."""

    @property
    def exiting(self) -> bool:
        return False

    @nn.nowrap
    def exit_passes(self, collections: Mapping[str, Mapping]) -> ExitPasses:
        """None of its passes exits (`loop` states no gate): no states, no logits."""
        return ExitPasses((), (), ())

    @nn.nowrap
    def with_passes(self, steps: int | None = None, layers: tuple[int, int] | None = None, *,
                    backprop_steps: int | None = None, remat: bool | None = None,
                    implicit: bool | None = None, exit_gate: bool | None = None,
                    exit_threshold: float | None = None, exits: bool | None = None,
                    theta_passes: int | None = None, messages=None) -> Self | None:
        """The source at `steps` passes of its block `layers` (`Looping.with_passes`): its loop's counts
        restated, or a new loop of both, without a step norm (a count alone over no loop is None, as a
        native decoder's); None where it asks what a tier-3 source cannot run (`runnable_loop`). Its
        parameters are the source's own: a run's loop (`DecoderRunConfig.looped`) keeps them, drawing none
        (`PretrainedFallback.with_model`)."""
        if any(value not in (None, False) for value in (backprop_steps, remat, implicit, exit_gate,
                                                         exit_threshold, exits, theta_passes, messages)):
            return None
        if self.loop is None:
            # A new loop names its passes and its block, and starts without a step norm, as a native
            # decoder's does (`CausalTransformer.with_passes`).
            if steps is None or layers is None:
                return None
            return self.with_loop(Loop(steps=steps, layers=layers, step_norm=False))
        return self.with_loop(replace(self.loop, steps=self.loop.steps if steps is None else steps,
                                      layers=self.loop.layers if layers is None else layers))

    @nn.nowrap
    def with_loop(self, loop: Loop) -> Self:
        """The source running `loop` whole (`runnable_loop`'s refusals)."""
        return self.clone(loop=runnable_loop(loop))

    @nn.nowrap
    def recompute_record(self) -> JSON:
        """None unrecomputed, as the trainer reads a model at no rung; its rung's record recomputed."""
        return {'remat': True} if self.remat else None

    @nn.nowrap
    def recompute_more(self) -> Self | None:
        """Its layers recomputed (`remat`) where they are not and its decoder holds a list of them
        (`_layer_list`), else None: the ladder's one rung."""
        if self.remat or _layer_list(self.graph.module._model) is None:
            return None
        return self.clone(remat=True)

    @nn.nowrap
    def restore_recompute(self, record: JSON) -> Self:
        """Its layers recomputed where `record` (`recompute_record`'s, None for none) says the run's were."""
        if record is None or self.remat:
            return self
        if _records.record(record, 'recompute') != {'remat': True}:
            raise ValueError(f"a tier-3 model's recompute record is None or {{'remat': True}}, got "
                             f"{record!r}")
        return self.clone(remat=True)

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        return ()

    @nn.nowrap
    def placement(self, layout: Layout) -> Layout:
        """The run's `layout` for this model's torch-named tree (`dew.distributed.placement.RefinesLayout`):
        its rules, fsdp and every other choice it makes, its tensors named by the model's tensor-parallel
        plan (`TorchLayout`) beside those another tier-3 model of the run named (a distillation's
        teacher), a name two models give different axes refused."""
        names = dict(layout.names if isinstance(layout, TorchLayout) else ())
        for name, axes in self.graph.axes:
            if names.setdefault(name, axes) != axes:
                raise ValueError(f"two tier-3 models of the run name {name} with different axes "
                                 f"({names[name]} and {axes}), which one layout cannot place both by")
        return TorchLayout(**{held.name: getattr(layout, held.name) for held in fields(Layout) if held.init},
                           names=tuple(names.items()))

    @property
    def causal(self) -> bool:
        """True: `load` builds the model with `AutoModelForCausalLM`, whose
        architectures all predict each token from the ones before it."""
        return True

    @property
    def mask_token_id(self) -> int | None:
        """None: a causal LM is not trained to fill masked tokens."""
        return None

    @nn.compact
    def __call__(self, tokens, train: bool = False, decode: bool = False, attention_mask=None,
                 positions=None, segment_ids=None, coordinates: Coordinates | None = None,
                 **packing) -> jax.Array:
        """The logits of `tokens`, in at least float32: of whole rows laid as `logits` lays them, or
        with `decode` of a decoding call into the cache `init_cache` opened, `attention_mask` `[B, S]`
        the tokens that write and `positions` `[B, S]` their places in their rows (their slots where
        None), or the rows' `coordinates` stating both, converted once (`token_coordinates`); a
        decoding row holds one document."""
        laid = token_coordinates(jnp.shape(tokens)[:2], coordinates, segments=segment_ids,
                                 valid=attention_mask, positions=positions)
        given = {} if laid is None else {COORDINATES_FIELD: laid}
        if not decode:
            return self.logits(tokens, train=train, **given, **packing)
        self._refuse_cache()
        laid = self._laid({**given, **packing})
        if laid is not None and laid.max_segments > 1:
            raise ValueError("a decoding row continues one document, and these rows hold several")
        # The tokens that write: those the validity marks less the padding a row numbers 0 in its documents.
        real = real_slots(given, jnp.shape(tokens)[:2])
        return self._decoded(tokens, real, None if laid is None else laid.positions)

    @property
    def cache_rebuild_position(self) -> int | None:
        return self.rebuild

    def inference_projection_groups(self, variables: Mapping[str, Mapping]) -> tuple:
        """None: a tier-3 model's projections are transformers' modules, which a server does not pack."""
        del variables
        return ()

    def mixed_admission_refusal(self) -> str:
        return ("a tier-3 model runs transformers' forward over Dew's cache, whose mixed step (one row of "
                "every slot's tokens) no check holds against its separate decode and prefill calls")

    def split_prefill_refusal(self) -> str:
        return ("a tier-3 model runs transformers' forward over Dew's cache, whose prefill in pieces no "
                "check holds against its whole prefill; serve it with whole-prompt admission")

    def init_cache(self, batch_size: int) -> None:
        """Open the decode cache of `batch_size` rows (`Decoding`), every slot unfilled. Its leaves are an
        external graph's, which only a step's trace states (`_layer_caches`): the one decoder whose cache's
        opening traces its step, for its shapes alone, where Dew's own decoders' holders state theirs."""
        self._refuse_cache()
        for layer, zeros in self._layer_caches(batch_size).items():
            self.put_variable("cache", f"layer_{layer}", zeros())
        put_opened(self, cursor(batch_size))

    def with_cache_capacity(self, capacity: int) -> Self:
        """The model with a decode cache of `capacity` slots a row (`CacheCapacity`)."""
        return self.clone(max_seq_len=capacity)

    def _cached(self, rows: int) -> dict[int, Variable]:
        """Each attention layer's keys and values by its `layer_idx`, opened at `rows` rows where they are
        not yet (`_layer_caches`). The rows' cursor is the model's own (`cache_positions`), one for every
        layer."""
        return {layer: self.variable("cache", f"layer_{layer}", zeros)
                for layer, zeros in self._layer_caches(rows).items()}

    def _layer_caches(self, rows: int) -> dict[int, Callable[[], dict[str, jax.Array]]]:
        """Each attention layer's keys and values by its `layer_idx` at `rows` rows of `max_seq_len`
        slots, every slot zero: their shapes and dtypes are those one decoding token of each row writes
        (`TorchGraph.step` over no held cache, traced for its shapes alone)."""
        tokens = jnp.zeros((rows, 1), jnp.int32)
        slots = jnp.zeros((rows, 1), jnp.int32)
        filled = jnp.ones((rows, 1), bool)
        _, shapes = jax.eval_shape(lambda held: self._step(held, tokens, {}, slots, filled, slots),
                                   self.graph.shapes)

        def opened(written: Mapping[str, jax.ShapeDtypeStruct]) -> Callable[[], dict[str, jax.Array]]:
            def zeros():
                return {name: jnp.zeros((rows, self.max_seq_len, *shape.shape[2:]), shape.dtype)
                        for name, shape in written.items()}
            return zeros

        return {layer: opened(written) for layer, written in sorted(shapes.items())}

    def _decoded(self, tokens, real, positions) -> jax.Array:
        tokens = jnp.asarray(tokens)
        rows, width = tokens.shape[:2]
        cache = self._cached(rows)
        # Each real token fills its row's next slot, the rest none (-1), as Dew's caches fill.
        slots = cache_positions(self, rows, width, real)
        filled = filled_slots(self.get_variable("cache", CURSOR), self.max_seq_len)
        places = jnp.maximum(slots, 0) if positions is None else jnp.asarray(positions, jnp.int32)
        # A decoding call's tokens reach their row's furthest real one, as a rotary's tables read it.
        furthest = jnp.max(jnp.where(real, places, -1), axis=1, keepdims=True)
        held = {layer: dict(variable.value) for layer, variable in cache.items()}
        logits, written = self._step(self.variables, tokens, held, slots, filled, places,
                                     jnp.broadcast_to(furthest, places.shape))
        for layer, variable in cache.items():
            variable.value = written[layer]
        return logits.astype(at_least_fp32(logits.dtype))

    def _step(self, variables, tokens, held, slots, filled, places, reach=None):
        params, buffers, compute = self._operands(variables)
        precision = self._products(compute)
        return self.graph.step(params, buffers, tokens, cache=held, slots=slots, filled=filled,
                               positions=places, reach=reach, precision=precision)

    def logits(self, tokens, *, train: bool = False, **packing) -> jax.Array:
        """The logits of the whole rows, as `__call__` returns them: in at least float32
        (`at_least_fp32`), so a float64 model's are float64, as its output table's are."""
        logits, _ = self._run(tokens, train, self._laid(packing))
        return logits.astype(at_least_fp32(logits.dtype))

    def hidden_states(self, tokens, train: bool = False, *, conditioning: Mapping[str, object] | None = None,
                      attention_pairwise_mask=None, attention_key_runs=None, attention_key_positions=None,
                      **packing) -> jax.Array:
        """The states the output head reads, `[B, S, features]`, for rows laid as `logits` lays them
        (`_laid`). `attention_pairwise_mask` `[B, S, S]`, or `attention_key_runs` `[B, S, R, 2]` (the keys
        each token sees as runs, `AttentionMetadata.key_runs`), states what each token sees, which then
        replaces documents and causality, and `attention_key_positions` `[B, S]` the keys' positions a
        window counts (on Dew's attention only), as Dew's own decoders read them; the tokens' rotary
        positions are the packing's (`positions`, `coordinates`).
        A tier-3 model reads no context: `conditioning` stays empty."""
        if conditioning:
            raise TypeError("a tier-3 model has no cross-attention to read a context with")
        laid = self._laid(packing)
        if attention_key_runs is not None:
            from dew.nn.attention_masks import visibility_mask

            # The runs as the mask they state, the stated visibility this path reads.
            length = jnp.shape(tokens)[1]
            attention_pairwise_mask = visibility_mask(length, length, causal=False, sliding_window=None,
                                                      key_runs=attention_key_runs)[:, 0]
        if attention_pairwise_mask is None:
            return self._run(tokens, train, laid)[1]
        if not self.graph.dew_attention:
            # Transformers' own attention reads its own causal mask; Dew's reads the stated one.
            raise ValueError("a stated visibility (attention_pairwise_mask) runs on Dew's attention, and "
                             "this model runs transformers' own")
        keys = None if attention_key_positions is None else jnp.asarray(attention_key_positions)
        return self._run(tokens, train, laid, visible=jnp.asarray(attention_pairwise_mask, bool),
                         key_places=keys)[1]

    def logits_from_hidden(self, hidden: jax.Array) -> jax.Array:
        """The output head over `hidden` `[B, S, features]`, every step the forward takes after the
        head included (a logit scale, a softcap), in at least float32 as `logits` returns them."""
        tokens = jnp.zeros(jnp.shape(hidden)[:2], jnp.int32)
        logits, _ = self._run(tokens, train=False, head_input=hidden)
        return logits.astype(at_least_fp32(logits.dtype))

    def _refuse_cache(self) -> None:
        """Refuse a decode cache to a model on transformers' own attention, which keeps none Dew's
        attention writes, or one that loops (`loop`), as the capability it lacks."""
        if not self.graph.dew_attention:
            raise Lacks("cache", "this model runs transformers' own attention, and a decode cache is the one "
                                 "Dew's attention writes")
        if self.loop is not None:
            raise Lacks("cache", "a tier-3 loop runs its block's layers once a pass, and a decode cache "
                                 "keeps one entry a layer")

    def _laid(self, packing: Mapping[str, object]) -> Coordinates | None:
        """The rows' `Coordinates`, their layout past `__call__` (ledger 1122), None for none: a model on
        Dew's attention reads them, one on transformers' own refuses them (`_refuse_packing`); any
        other field is refused."""
        if not self.graph.dew_attention:
            _refuse_packing(packing)
            return None
        laid = packing.get(COORDINATES_FIELD)
        unknown = sorted(set(packing) - {COORDINATES_FIELD})
        if unknown or (laid is not None and not isinstance(laid, Coordinates)):
            raise Lacks("packing", f"a tier-3 model reads its rows' layout as their coordinates, not "
                                   f"{unknown or [COORDINATES_FIELD]}")
        return laid

    def _run(self, tokens, train: bool, laid: Coordinates | None = None, *, visible=None, key_places=None,
             head_input=None) -> tuple[jax.Array, jax.Array]:
        if self.is_initializing():
            # The source's variables are its own: the forward's shapes over them, as zeros.
            def over(held):
                return self._over(held, tokens, laid, visible, key_places, head_input, train=False)

            shapes = jax.eval_shape(over, self.graph.shapes)
            return jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)
        return self._over(self.variables, tokens, laid, visible, key_places, head_input, train=train)

    def _over(self, held, tokens, laid, visible, key_places, head_input, *,
              train: bool) -> tuple[jax.Array, jax.Array]:
        params, buffers, compute = self._operands(held)
        key = self.make_rng("dropout") if train and self.has_rng("dropout") else None
        loop = None if self.loop is None else runnable_loop(self.loop)
        return self.graph(params, buffers, jnp.asarray(tokens), train=train, key=key,
                          precision=self._products(compute), laid=laid, visible=visible,
                          key_places=key_places, head_input=head_input, recomputed=self.remat and train,
                          loop=loop)

    def _operands(self, held) -> tuple[dict[str, jax.Array], dict[str, jax.Array], Dtype]:
        """The forward's parameters at `dtype`, its buffers, and the dtype its products take: `dtype`, or
        the dtype its floating parameters hold."""
        params = {name: self._compute(leaf) for name, leaf in held["params"].items()}
        buffers = {name: jnp.asarray(leaf) for name, leaf in held.get("buffers", {}).items()}
        compute = self.dtype or jnp.result_type(*[leaf for leaf in params.values()
                                                  if jnp.issubdtype(leaf.dtype, jnp.floating)])
        return params, buffers, compute

    def _products(self, compute: Dtype) -> jax.lax.PrecisionLike:
        """The precision the forward's products take: the one stated, else, for float32 operands, IEEE
        float32 (`exact_float32`), since this forward is transformers' torch code, whose float32 matmuls
        torch multiplies in IEEE float32 (TF32 off by default), and its output table's head alike; other
        operands' at JAX's configured precision."""
        return exact_float32(self.precision) if jnp.dtype(compute) == jnp.float32 else self.precision

    def _compute(self, leaf) -> jax.Array:
        value = jnp.asarray(leaf)
        if self.dtype is None or not jnp.issubdtype(value.dtype, jnp.floating):
            return value
        return value.astype(self.dtype)


class TorchTableLM(TorchCausalLM):
    """A `TorchCausalLM` whose logits are by construction its head input
    times the weight `head` names, which the loader establishes from the
    model's structure (`_plain_head`): the pair `LMObjective` scores through,
    so the vocabulary-sized logits are never built whole."""

    head: str = field(kw_only=True)

    def output_table(self) -> OutputTable:
        """The output head's `[vocab, features]` torch weight as the bound params
        hold it, the parameter a loss keeps, with the forward's compute dtype,
        which casts the product's operands as the forward casts them whatever
        dtype the states arrive in (`head_product`)."""
        return OutputTable(self.variables["params"][self.head], vocab_major=True,
                           precision=self.precision, compute=self.dtype)

    def vocabulary_rows(self) -> jax.Array:
        """The output head's rows (`OutputTable.rows`)."""
        return self.output_table().rows()


def _refuse_packing(packing: Mapping[str, object]) -> None:
    """Refuse every field past the tokens: transformers' forward takes whole
    plain rows, with no packing columns and no cache."""
    if packing:
        raise Lacks("packing",
                    f"a torchax model runs transformers' forward over whole plain token rows, with no cache, "
                    f"which takes no {sorted(packing)}; train it on unpacked rows without packing columns")


def load(source: sources.SourceReference, root: Path, *, dtype: str,
         param_dtype: str, max_seq_len: int | None,
         admission: TorchAdmission | None = None, weights: bool = True,
         placed: Placing | None = None) -> PretrainedFallback:
    """Load the causal LM `source` selects through transformers and torchax.

    `root` is the metadata snapshot `Pretrained.load` resolved at `source`'s
    revision, and `source`'s folder selects the model below it; the weights
    come from the same commit and folder (`SourceReference.weights`), so the
    file selection is the native loader's. `admission` is the decision that
    sent an unregistered type here (`admitted`), None for
    `fallback="torchax"`. Without `weights` the model is built from its
    config on torch's meta device, nothing but metadata fetched or read, and
    its variables are their shapes and dtypes alone (`jax.ShapeDtypeStruct`).
    `placed` places them as the load asks, by the layout as the model refines it for its torch names
    (`TorchCausalLM.placement`); they stay on the host where it is None.
    """
    name_or_dir, revision = source.name, source.revision
    directory = source.directory(root)
    attends = admission is not None and admission.attends
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(
            f"fallback='torchax' builds the model a transformers config.json describes, and "
            f"{name_or_dir} has none; load a repo that ships config.json and its weights")
    config = json.loads((directory / "config.json").read_text())
    refuse_quantized(config)
    try:
        import torch
        from torchax.interop import JittableModule, extract_all_buffers
    except ImportError as error:
        raise ImportError(f"fallback='torchax' needs torch, torchax and transformers: {INSTALL}") from error
    if weights:
        directory = source.directory(source.weights(root))
    # Before transformers, whose reader fails on a malformed file without naming it.
    generation_config = read_generation_config(directory)
    model, by_name = _built(directory, param_dtype, weights)
    model.eval()
    jittable = JittableModule(model)
    head = model.get_output_embeddings()
    if not isinstance(head, torch.nn.Module):
        raise ValueError(f"{type(model).__name__} has no output embeddings for a causal LM's logits")
    head_weight = head.weight
    if not isinstance(head_weight, torch.Tensor):
        raise ValueError(f"{type(model).__name__}'s output head holds no weight tensor")
    head_name = next((name for name, value in jittable.params.items() if value is head_weight), None)
    # The head's structure, which a forward shows: the loaded model's, or without its weights the
    # same class at the probe's sizes (`verify.shrink`), since a forward on the meta device reads no
    # values, which some forwards do.
    if head_name is not None and not (_plain_head(model, head) if weights and by_name is None
                                      else _shrunk_plain_head(config)):
        head_name = None
    # After the head's eager check: the functional parts and Dew's attention run only inside the traced
    # forward. The parts keep their parameters, which the jittable module holds already.
    rebuild = functional_parts(model)
    if attends:
        _registered()
        model.set_attn_implementation(_ATTENTION)
    # Host views of torch's storage: nothing reaches a device before the
    # caller's placement. Without the weights, their shapes and dtypes.
    def held(leaf, name):
        if not weights:
            return jax.ShapeDtypeStruct(tuple(leaf.shape), host_dtype(leaf, name))
        if by_name is not None and name in by_name:
            # Read a device shard at a time, cast to its storage as the placement reads it
            # (`dew.interop.streaming.SourceLeaf`), as a native family's checkpoint streams.
            tensor = by_name[name]
            return SourceLeaf((tensor,), checkpoint_dtype(np.dtype(tensor.dtype), param_dtype))
        return host_view(leaf, name)

    if by_name is not None and (unread := sorted(set(jittable.params) - set(by_name))):
        raise ValueError(_unread(name_or_dir, unread))

    params = {name: held(leaf, name) for name, leaf in jittable.params.items()}
    buffers = {name: held(leaf, name) for name, leaf in extract_all_buffers(model)[1].items()}
    stored = model.state_dict()
    persistent = tuple(name for name in buffers if name in stored)
    vocab = int(head_weight.shape[0])
    axes = planned_axes(model)
    # Torch's copies are no longer read: the forward runs on the arrays above.
    torch.nn.Module.to(model, "meta")
    graph = TorchGraph(jittable, persistent, dew_attention=attends,
                       shapes={"params": _shapes(params), "buffers": _shapes(buffers)}, axes=axes,
                       certified={} if admission is not None and admission.attends else None)
    # The decode cache's slots: the load's, else the source's own context as its config class reads it
    # (an alias, CTRL's `n_positions`, included).
    named = type(model.config).attribute_map.get("max_position_embeddings", "max_position_embeddings")
    context = model.config.to_dict().get(named)
    capacity = (max_seq_len if max_seq_len is not None
                else context_lengths({"max_position_embeddings": context})[0])
    built = {"dtype": resolve_dtype(dtype), "max_seq_len": capacity, "rebuild": rebuild}
    module = (TorchCausalLM(graph, vocab, **built) if head_name is None
              else TorchTableLM(graph, vocab, head=head_name, **built))
    processor = source_processor(directory, config, config, module, source=source)
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "torchax", "torch")}
    why = ("fallback='torchax'" if admission is None else
           f"model_type {admission.model_type!r} is no registered family and no registered family's "
           f"convention computes it: {_summary(admission.reasons)}")
    warnings.warn(
        f"{name_or_dir} loads as tier 3 ({why}): transformers {versions['transformers']}'s "
        f"PyTorch forward lowered by torchax {versions['torchax']}, pinned to torch "
        f"{versions['torch']}; "
        + ("its attention on Dew's, packed rows and Dew's decode cache included; " if attends
           else "its attention transformers' own, no packing or decode cache; ")
        + (f"{admission.assumes}; " if admission is not None and admission.assumes else "")
        + f"placed by its tensor-parallel plan (TorchLayout); {_host_cost(weights, by_name)}",
        TransformersTierWarning, stacklevel=3)
    built = {**config, "dtype": dtype, "fallback": "torchax", **versions}
    variables = {"params": params, "buffers": buffers}
    landed = variables if placed is None else placed(variables, module)
    # What no placement read (a load onto no mesh), read whole in its storage; torch's names are flat.
    kept = {collection: {name: leaf.read() if isinstance(leaf, SourceLeaf) else leaf
                         for name, leaf in tree.items()} for collection, tree in landed.items()}
    return PretrainedFallback(module, kept, processor, config, directory, built, generation_config,
                              export_adapter=graph.export, revision=revision)


def _unread(source: str | Path, names: Sequence[str]) -> str:
    """The refusal of a source whose checkpoint lacks parameters its model's class names, which a load
    would draw: the source is its checkpoint."""
    return (f"{source}'s checkpoint holds no {', '.join(names[:4])}{' and more' if len(names) > 4 else ''}, "
            "which its model's class names: a load would draw them, and the source is its checkpoint")


def _shapes(leaves: Mapping[str, np.ndarray | SourceLeaf | jax.ShapeDtypeStruct]
            ) -> dict[str, jax.ShapeDtypeStruct]:
    """Each leaf's shape and dtype."""
    return {name: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for name, leaf in leaves.items()}


def _shrunk_plain_head(config: Mapping[str, object]) -> bool:
    """`_plain_head` of the class `config` builds, at the probe's sizes and depth (`verify.shrink`,
    `verify.shallow`) with transformers' own initialization: the head's structure, which no size
    changes."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type = str(config["model_type"])
    probed = verify.shallow(verify.ModelSource(model_type), verify.shrink(config))
    fields = {key: value for key, value in _records.decoded_floats(probed).items() if key != "model_type"}
    stated = AutoConfig.for_model(model_type, **fields)
    model = AutoModelForCausalLM.from_config(stated, dtype=torch.float32, experts_implementation=_experts())
    return _plain_head(model.eval(), model.get_output_embeddings())


def _host_cost(weights: bool, by_name: Mapping[str, np.ndarray] | None) -> str:
    """What a tier-3 load held on the host (`_built`): only transformers' own load, which converts its
    checkpoint, holds the torch model (ledger 2781)."""
    if not weights:
        return "its weights unread"
    if by_name is not None:
        return "its tensors read from the checkpoint by name, no torch copy of the model held"
    return "the load holds the torch model on the host, at least 2x the checkpoint"


def _built(directory: Path, param_dtype: str, weights: bool
           ) -> tuple[PreTrainedModel, dict[str, np.ndarray] | None]:
    """The causal LM `directory`'s config describes, and with `weights` its tensors by name
    (`_stored_tensors`): its parameters on torch's meta device, its buffers computed as its modules
    compute them, each tensor read from the checkpoint memory-mapped, which the load's placement streams
    onto the mesh a leaf at a time, no copy of the model held on the host. It is built in its config's
    dtype, or under `AUTO` with none stated in its checkpoint's first floating tensor's, as transformers'
    own load infers it, so its computed buffers are the ones that load computes. A checkpoint a tensor
    operation of transformers' converts (a fused expert's merge) loads through transformers' own load,
    which alone applies them, and its tensors are the model's (None), refused where the checkpoint
    lacks a parameter its class names. Without `weights`, the model on the meta device alone."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        from accelerate import init_empty_weights
    except ImportError as error:
        raise ImportError(f"fallback='torchax' builds its model's parameters on the meta device with "
                          f"accelerate: {INSTALL}") from error
    storage = "auto" if param_dtype == AUTO else getattr(torch, param_dtype)
    stated = AutoConfig.from_pretrained(str(directory), local_files_only=True)
    built_as = stated.dtype if storage == "auto" else storage
    shards = sources.load_shards(directory) if weights else {}
    if built_as is None:
        floating = (tensor.dtype for tensor in shards.values() if jnp.issubdtype(tensor.dtype, jnp.floating))
        built_as = next((getattr(torch, dtype.name) for dtype in floating), None)
    if not weights:
        with torch.device("meta"):
            built = AutoModelForCausalLM.from_config(stated, dtype=built_as,
                                                     experts_implementation=_experts())
        return built, None
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(stated, dtype=built_as, experts_implementation=_experts())
    # Its tied parameters one, as transformers' load ties them, which `_stored_tensors` reads by each name.
    model.tie_weights()
    stored = _stored_tensors(model, shards)
    if stored is not None:
        return model, stored
    loaded = AutoModelForCausalLM.from_pretrained(
        str(directory), dtype=storage, local_files_only=True, experts_implementation=_experts(),
        output_loading_info=True)
    # The model and what its load read and drew (`output_loading_info`).
    assert isinstance(loaded, tuple)
    converted, loading = loaded
    if loading["missing_keys"]:
        raise ValueError(_unread(directory, sorted(loading["missing_keys"])))
    return converted, None


def _stored_tensors(model: PreTrainedModel, shards: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    """A checkpoint's `shards` (`sources.load_shards`, memory-mapped) by the names `model` holds them
    under, as stored, which the load casts to their storage leaf by leaf where it places them
    (`_in_storage`): each key named as transformers' load names it (`sources.load_names`), a key the model
    holds nothing under left out, and a tied parameter (an output table that is its input table) under
    each of its names from the one the checkpoint stores it under, a key renamed by whether another is
    stored read over the shards' names. None where a key takes one of the mapping's tensor conversions
    (a fused expert's merge), which transformers' own load applies."""
    names = sources.load_names(model, shards.keys())
    if names is None:
        return None
    held = model.state_dict()
    named: dict[str, np.ndarray] = {}
    for key, tensor in shards.items():
        read = names(key)
        if read.converter is not None:
            return None
        if read.name in held:
            named[read.name] = tensor
    tied: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        tied.setdefault(id(parameter), []).append(name)
    for names in tied.values():
        for stored in [name for name in names if name in named][:1]:
            for name in names:
                named.setdefault(name, named[stored])
    return named


def _plain_head(model: PreTrainedModel, head: torch.nn.Module) -> bool:
    """Whether `model`'s logits are by construction its head input times its
    head weight: the head a plain bias-free torch Linear, and the logits the
    model returns the very tensor that Linear made, so nothing ran after it.

    Values cannot show this, nor gradients: a zero head weight makes a
    softcapped model's logits and their first derivatives the plain
    product's, and bf16 rounding can hide a small term. Structure can.
    """
    import torch

    if type(head) is not torch.nn.Linear or head.bias is not None:
        return False
    made: list[object] = []
    hook = head.register_forward_hook(lambda _, __, output: made.append(output))
    tokens = torch.arange(min(8, int(head.out_features)))[None]
    try:
        with torch.no_grad():
            logits = model(input_ids=tokens, use_cache=False).logits
    finally:
        hook.remove()
    return len(made) == 1 and logits is made[0]
