"""Tier 3: a transformers PyTorch causal LM, lowered to JAX by torchax.

`Pretrained.load` takes this tier for an unregistered causal model_type no
registered convention computes (`verify.Unmapped`) once `admitted` decides
transformers' own forward runs it, and `Pretrained.load(source,
fallback="torchax")` takes it for any source. It builds the model with
transformers on the host and hands its forward to torchax, which runs every
torch op as a JAX op, so the model jits, differentiates and shards like any
other JAX function. Dew knows nothing about the architecture: no Dew
kernels but its attention's, no logical sharding declarations. What Dew
supplies is the seam: a Flax module over the torch forward, the parameters
split from the buffers, Dew's attention and decode cache where the model asks
transformers' attention interface for its attention, and a `Layout` that
places tensors by their torch names.

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
import fnmatch
import functools
import importlib.metadata
import json
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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

from dew.batch import COORDINATES_FIELD, VALIDITY_FIELD, Coordinates, Variables, coordinates, real_slots
from dew.distributed.placement import Layout
from dew.interop import sources, verify
from dew.interop.decoder_parts import context_lengths
from dew.interop.generation_config import read_generation_config
from dew.interop.pickles import host_dtype, host_view
from dew.interop.pretrained import PretrainedFallback, source_processor
from dew.interop.weights import AUTO
from dew.mesh import LogicalAxes
from dew.nn.inputs import document_reach
from dew.nn.kv_cache import cache_positions, filled_slots
from dew.nn.precision import at_least_fp32, configured_precision, float32_products, scaled
from dew.nn.protocols import Lacks, OutputTable
from dew.nn.sharding import Suffix, parameter_path
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


EXPERTS = "batched_mm"
"""The routed-experts implementation tier 3 builds transformers' models with: its batched products
over each token's chosen experts, whose shapes the token count fixes, where the default
(`grouped_mm`) has no torchax lowering and an eager loop selects tokens by `nonzero`. A model whose
experts do not dispatch through transformers' `ExpertsInterface` runs its own, which `admitted`
observes."""


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


def admitted(unmapped: verify.Unmapped) -> TorchAdmission:
    """Tier 3's decision on an unregistered model_type no registered convention computes, on the
    probe's shrunken config with random weights (`verify.scatter_weights`), before any weight
    downloads: transformers builds the type as a causal LM (`AutoModelForCausalLM`); its forward is
    causal, so changing a row's last token leaves every earlier position's logits bit for bit (a
    decoder configured bidirectionally, BERT's with is_decoder False, is not); and its inference and
    training forwards trace as tier 3 traces them (`TorchGraph.traced`), the training one drawing
    nothing on the host. An operation whose output's shape or value the traced values decide (a
    `nonzero`, an `item()`) fails that trace with JAX's own error, naming it. Each refusal names its
    cause."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    model_type = unmapped.model_type
    detail = _summary(unmapped.reasons)
    if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        raise ValueError(f"model_type {model_type!r} is no registered family, no registered family's "
                         f"convention computes it ({detail}), and transformers builds no causal LM for it, "
                         "which tier 3 runs")
    fields = {key: value for key, value in unmapped.fields.items() if key != "model_type"}
    config = AutoConfig.for_model(model_type, **fields, experts_implementation=EXPERTS)
    _registered()
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=torch.float32,
                                             attn_implementation=_PROBE).eval()
    verify.scatter_weights(model)
    ids = torch.from_numpy(verify.probe_ids(int(config.vocab_size)).astype(np.int64))
    probe = _Probe()
    with torch.no_grad(), probe.recording():
        logits = model(input_ids=ids, use_cache=False).logits
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % int(config.vocab_size)
    with torch.no_grad():
        moved = model(input_ids=changed, use_cache=False).logits
    if not torch.equal(logits[:, :-1], moved[:, :-1]):
        raise ValueError(f"model_type {model_type!r} as configured attends to later tokens (its earlier "
                         "positions' logits move with a row's last token), and tier 3 runs causal LMs")
    # Its inference and training forwards traced as tier 3 traces them; the training one's dropout and
    # any layer drop run only there, and a draw on the host would be fixed for every step.
    from torchax.interop import JittableModule, extract_all_buffers

    functional_parts(model)
    jittable = JittableModule(model)
    params = {name: host_view(leaf, name) for name, leaf in jittable.params.items()}
    buffers = {name: host_view(leaf, name) for name, leaf in extract_all_buffers(model)[1].items()}
    graph = TorchGraph(jittable, ())
    graph.traced(params, buffers, ids.numpy().astype(np.int32), None, train=False)
    _, _, draws = graph.traced(params, buffers, ids.numpy().astype(np.int32), jax.random.key(0), train=True)
    if draws.host:
        raise ValueError(f"model_type {model_type!r}'s training forward draws on the host ({draws.host} "
                         "draws, BioGPT's layer drop among them), which a traced step would fix for every "
                         "step, and tier 3 trains the models it loads")
    why = probe.kept_from_dew(int(config.num_hidden_layers))
    return TorchAdmission(model_type, dict(unmapped.reasons), attends=not why, why=why)


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

_COMPUTED = frozenset({"sliding_window", "softcap", "s_aux"})
"""The keywords Dew's attention computes (`_dew_attention`): a window over the most recent keys, a
tanh cap on the logits, and one value-free logit a head (sinks)."""

_PROBING: contextvars.ContextVar[_Probe | None] = contextvars.ContextVar("dew_probe", default=None)


class _Probe:
    """What each attention call of a probed forward asks (`recording`): the module, the keywords it
    passes past `_PLUMBING`, and whether the mask transformers built for it is the visibility Dew's
    attention computes from the module's causality and those keywords (`_visible`)."""

    def __init__(self):
        self.modules: set[int] = set()
        self.unknown: set[str] = set()
        self.unmatched = 0

    @contextlib.contextmanager
    def recording(self):
        token = _PROBING.set(self)
        try:
            yield self
        finally:
            _PROBING.reset(token)

    def called(self, module, query, key, attention_mask, scaling, kwargs) -> None:
        self.modules.add(id(module))
        self.unknown |= set(kwargs) - _PLUMBING - _COMPUTED
        seen = _seen(module, query, key, attention_mask, scaling, kwargs)
        if not _visible(seen, kwargs.get("sliding_window")):
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


def _seen(module, query, key, attention_mask, scaling, kwargs) -> np.ndarray:
    """Which keys each query of a call reads in transformers' SDPA attention as the probe runs it,
    `[B, Q, H, K]`: its attention weights over one-hot values, nonzero where a key is read. Read from
    the attention itself, so a mask transformers builds or one it leaves to the attention's causal
    flag (None) alike."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    batch, heads, keys, _ = key.shape
    identity = torch.eye(keys, dtype=query.dtype).expand(batch, heads, keys, keys)
    weights, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, identity, attention_mask,
                                               scaling=scaling, dropout=0.0, **kwargs)
    return weights.detach().cpu().numpy() > 0


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

    implementation: str
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
    unknown = sorted(set(kwargs) - _PLUMBING - _COMPUTED)
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
    dropping = {"dropout_rate": float(dropout) if drawn else 0.0, "deterministic": not drawn,
                "dropout_rng": rng}

    def attend(q, k, v, *, places, segment_ids, valid):
        del places, valid  # the documents hold the padding apart, as 0
        return scaled_dot_product_attention(
            q, k, v, implementation=call.implementation, causal=True, sliding_window=window,
            softcap=softcap, sinks=sinks, segment_ids=segment_ids, max_segments=max_segments, **dropping)

    if call.cache is not None:
        # The call's tokens join the row's cache at their slots, and each attends over the whole row:
        # what it has filled, causal by slot and within the window, as Dew's decoders attend a cache.
        held = call.cache.held.get(module.layer_idx)
        if held is None:
            # A cache being opened: its layer's keys and values give its shape, and nothing is held.
            held = {"cached_key": jnp.zeros_like(k), "cached_value": jnp.zeros_like(v)}
        keys, values = (write_cache(held[name], incoming, call.cache.slots)
                        for name, incoming in (("cached_key", k), ("cached_value", v)))
        call.cache.written[module.layer_idx] = {"cached_key": keys, "cached_value": values}
        keys, values = keys.astype(q.dtype), values.astype(q.dtype)
        if q.shape[1] == 1 and not drawn:
            # One token a row reads its row's filled slots before its own, from its window's first: a
            # key count and start a row, which the decode kernels take as Dew's decoders give them
            # (`CausalSelfAttention._decode_attention`); a padding token reads one key, its output unused.
            slot = call.cache.slots[:, 0]
            reads = jnp.maximum(slot + 1, 1).astype(jnp.int32)
            starts = None if window is None else jnp.maximum(slot + 1 - window, 0).astype(jnp.int32)
            out = scaled_dot_product_attention(q, keys, values, implementation=call.implementation,
                                               causal=False, softcap=softcap, sinks=sinks,
                                               key_value_seq_lengths=reads, key_value_seq_starts=starts)
        else:
            mask = causal_attention_mask(call.cache.slots, keys.shape[1], window, key_valid=call.cache.filled)
            out = scaled_dot_product_attention(q, keys, values, implementation=call.implementation, mask=mask,
                                               softcap=softcap, sinks=sinks, **dropping)
    elif call.visible is not None:
        # The call's own visibility is the whole of it; a window counts the stated keys' positions.
        visible = jnp.asarray(call.visible, bool)
        if window is not None and call.key_places is not None:
            batch, length = visible.shape[:2]
            places = (jnp.broadcast_to(jnp.arange(length), (batch, length)) if call.places is None
                      else call.places)
            reach = windowed(places[:, :, None], call.key_places[:, None, :], window, causal=False)
            visible = visible & reach
        out = scaled_dot_product_attention(q, k, v, implementation=call.implementation,
                                           mask=visible[:, None], softcap=softcap, sinks=sinks, **dropping)
    elif call.valid is None:
        out = attend(q, k, v, places=None, segment_ids=call.documents, valid=None)
    else:
        out = compacted(attend, q, k, v, None, call.documents, call.valid)
    return torch_view(out.astype(q.dtype)), None


@functools.cache
def _registered() -> None:
    """Dew's attention (`_dew_attention`) and the probe's (`_probe_attention`, over transformers' eager
    mask, additive and always built, which a model computing its own attention adds to its scores) in
    transformers' attention interfaces, once a process."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

    AttentionInterface.register(_ATTENTION, _dew_attention)
    AttentionInterface.register(_PROBE, _probe_attention)
    AttentionMaskInterface.register(_PROBE, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


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


@functools.cache
def _functional_experts() -> type:
    """The torch module `functional_experts` substitutes (built on first use)."""
    import torch

    from dew.interop.parts import gated_activation
    from dew.nn.activations import gated_product
    from dew.nn.moe import EXPERT_AXES, expert_compute_dtype, expert_dispatch, expert_projection

    class FunctionalExperts(torch.nn.Module):
        """Routed experts beside identity experts at static shapes, through Dew's grouped expert
        dispatch (`dew.nn.moe.expert_dispatch`): each routed slot runs its expert's gated product
        (`expert_projection` over the rows sorted by expert), an identity slot (numbered past the
        routed experts, which the dispatch's sentinel holds at zero) passes its token, and the slots sum
        weighted by their routing weights in at least fp32, as transformers' loop computes them expert by
        expert. Its parameters are the replaced module's; one of identity experts alone holds none."""

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
            identity = index >= self.routed
            slots = jnp.zeros((*index.shape, x.shape[-1]), x.dtype)
            if self.routed:
                # Under the trace these are its views of the parameters, not torch's parameters.
                gate_up = jax_view(self.gate_up_proj)[:self.routed]
                width = gate_up.shape[1] // 2
                kernels = (jnp.swapaxes(gate_up[:, :width], 1, 2), jnp.swapaxes(gate_up[:, width:], 1, 2),
                           jnp.swapaxes(jax_view(self.down_proj), 1, 2))
                compute = expert_compute_dtype(x, *kernels, dtype=None)

                def project(tokens, sizes, _ids, held):
                    def linear(rows, kernel):
                        return jnp.asarray(expert_projection(rows, kernel, sizes, compute, "xla", None))
                    gated = gated_product(self.activation)(linear(tokens, held[0]), linear(tokens, held[1]))
                    return linear(gated, held[2])

                slots = expert_dispatch(project, x, jnp.where(identity, self.routed, index), kernels,
                                        tuple(EXPERT_AXES[name] for name in ("gate_proj", "up_proj",
                                                                             "down_proj")),
                                        num_experts=self.routed, dispatch="global", output_dtype=compute)
            held = jnp.where(identity[..., None], x[:, None, :], slots)
            wide = at_least_fp32(x.dtype)
            out = jnp.sum(held.astype(wide) * weights[..., None].astype(wide), axis=1)
            return torch_view(out.astype(x.dtype))

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


TORCH_AXES: tuple[tuple[str, LogicalAxes], ...] = (
    # nn.Linear stores [out, in]; its bias is the output side alone.
    ("*.q_proj.weight", ("heads", "embed")), ("*.q_proj.bias", ("heads",)),
    ("*.k_proj.weight", ("kv", "embed")), ("*.k_proj.bias", ("kv",)),
    ("*.v_proj.weight", ("kv", "embed")), ("*.v_proj.bias", ("kv",)),
    ("*.o_proj.weight", ("embed", "heads")),
    ("*.self_attn.out_proj.weight", ("embed", "heads")),
    ("*.attention.query_key_value.weight", ("heads", "embed")),
    ("*.attention.query_key_value.bias", ("heads",)),
    ("*.attention.dense.weight", ("embed", "heads")),
    ("*.gate_proj.weight", ("mlp", "embed")), ("*.up_proj.weight", ("mlp", "embed")),
    ("*.down_proj.weight", ("embed", "mlp")),
    ("*.fc1.weight", ("mlp", "embed")), ("*.fc1.bias", ("mlp",)),
    ("*.fc2.weight", ("embed", "mlp")),
    ("*.dense_h_to_4h.weight", ("mlp", "embed")), ("*.dense_h_to_4h.bias", ("mlp",)),
    ("*.dense_4h_to_h.weight", ("embed", "mlp")),
    # GPT-2's Conv1D stores [in, out].
    ("*.attn.c_attn.weight", ("embed", "heads")), ("*.attn.c_attn.bias", ("heads",)),
    ("*.attn.c_proj.weight", ("heads", "embed")),
    ("*.mlp.c_fc.weight", ("embed", "mlp")), ("*.mlp.c_fc.bias", ("mlp",)),
    ("*.mlp.c_proj.weight", ("mlp", "embed")),
    # Token tables and output heads are [vocab, embed], positions [positions, embed].
    ("*embed_tokens.weight", ("vocab", "embed")), ("*.wte.weight", ("vocab", "embed")),
    ("*embed_in.weight", ("vocab", "embed")), ("*word_embeddings.weight", ("vocab", "embed")),
    ("lm_head.weight", ("vocab", "embed")), ("embed_out.weight", ("vocab", "embed")),
    ("*.wpe.weight", (None, "embed")), ("*embed_positions.weight", (None, "embed")),
)
"""The logical axes of the tensors the common transformers conventions name,
in their torch orientation. A rule applies to a tensor its glob matches and
whose rank equals the length of its axes; the first such rule wins."""


@dataclass(frozen=True)
class TorchLayout(Layout):
    """A `Layout` for a tree keyed by torch parameter names.

    `names` maps a glob over a tensor's torch name to its logical axes;
    `rules` then map those onto the mesh exactly as they do for a native
    model, so the tensor axis, fsdp, `min_shard` and `check` mean what they
    mean there. A tensor no name matches takes `Layout`'s shape heuristic.
    Optimizer moments and EMA copies end in their parameter's name, so they
    take its placement.

    The placement decides where GSPMD keeps each tensor, not what the forward
    computes: a spec that suits the model badly costs collectives, never
    correctness.
    """

    names: tuple[tuple[str, LogicalAxes], ...] = TORCH_AXES

    def axes(self, name: str, ndim: int) -> LogicalAxes | None:
        """The logical axes the first matching rule gives `name`, or None."""
        for pattern, axes in self.names:
            if len(axes) == ndim and fnmatch.fnmatchcase(name, pattern):
                return axes
        return None

    def _axes(self, path, ndim: int, declared: Mapping[Suffix, LogicalAxes]) -> LogicalAxes | None:
        """The axes the name table gives the leaf's torch name, else `Layout`'s."""
        names = parameter_path(path)
        axes = self.axes(names[-1], ndim) if names else None
        return super()._axes(path, ndim, declared) if axes is None else axes


@dataclass(frozen=True, eq=False)
class TorchGraph:
    """The transformers module torchax runs, as a function of its tensors.

    `persistent` names the buffers the source checkpoint stores. A training
    forward runs the module's own training forward, its dropout and other
    draws taking `key` as torchax's random key; one that draws on the host
    instead (OPT's layer drop, `torch.rand([])`) is decided once at trace
    time, and is refused. `attention_impl` is the Dew attention kernel a module
    built with Dew's attention (`_ATTENTION`) runs, None for one that runs
    transformers' own.
    """

    module: JittableModule
    persistent: tuple[str, ...]
    attention_impl: str | None = None
    shapes: Mapping[str, Mapping[str, jax.ShapeDtypeStruct]] = field(default_factory=dict)
    """The source's variables, `params` and `buffers`, as shapes and dtypes: what an initialization,
    which holds none of them, traces the forward over (`TorchCausalLM._run`)."""

    def __call__(self, params: Mapping[str, jax.Array], buffers: Mapping[str, jax.Array],
                 tokens: jax.Array, *, train: bool, key: jax.Array | None = None,
                 precision: jax.lax.PrecisionLike = None, laid: Coordinates | None = None,
                 visible: jax.Array | None = None, key_places: jax.Array | None = None,
                 head_input: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        """The logits and the output head's input states for `tokens`, in Dew's torchax
        environment (`_environment`), every product at `precision` (`configured_precision`):
        torch's operators state none. Each broadcast of a value narrower than fp32 sums
        its gradient in fp32 (`_accumulating`), as torch's reduction of a broadcast's
        gradient does. `laid` places the tokens in their documents, which Dew's attention
        keeps apart and whose positions transformers' rotary reads; `visible` `[B, S, S]`, where
        given, is what each token sees in their place, and `key_places` `[B, S]` the keys' positions a
        window counts (`_Attending`). `head_input`, where
        given, is the output head's input in place of the states the forward computes, so the
        logits are the head's over it, every step the forward takes after the head included."""
        documents, max_segments = _documents(laid)
        positions = None if laid is None else laid.places()
        valid = None if laid is None or laid.valid is None else jnp.asarray(laid.valid, bool)
        arguments = (dict(params), dict(buffers), tokens, key, positions, documents, valid, visible,
                     key_places, head_input, None, None)
        (logits, hidden), _ = self._run(arguments, train=train, precision=precision,
                                        max_segments=max_segments)
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

    def _run(self, arguments: tuple, *, train: bool, precision: jax.lax.PrecisionLike, max_segments: int):
        traced, shapes, draws = self.traced(*arguments, train=train, precision=precision,
                                            max_segments=max_segments)
        outputs = jax.tree.unflatten(jax.tree.structure(shapes),
                                     _accumulating(traced, *jax.tree.leaves(arguments)))
        key = arguments[3]
        if draws.host:
            raise ValueError(
                f"tier 3 traces this model's training forward once, and it draws on the host "
                f"({draws.host} draws, e.g. a layer drop), which tracing would fix for every step; set "
                f"that rate to 0 in its config, or train a family Dew loads natively")
        if draws.device and key is None:
            raise ValueError("this model's training forward draws (dropout) and was given no 'dropout' rng")
        return outputs

    def traced(self, params: Mapping[str, jax.typing.ArrayLike], buffers: Mapping[str, jax.typing.ArrayLike],
               tokens: jax.typing.ArrayLike, key: jax.Array | None,
               positions: jax.typing.ArrayLike | None = None, documents: jax.typing.ArrayLike | None = None,
               valid: jax.typing.ArrayLike | None = None, visible: jax.typing.ArrayLike | None = None,
               key_places: jax.typing.ArrayLike | None = None, head_input: jax.typing.ArrayLike | None = None,
               cache: tuple | None = None, reach: jax.typing.ArrayLike | None = None, *, train: bool,
               precision: jax.lax.PrecisionLike = None, max_segments: int = 1):
        """The forward over these arguments traced once (`jax.make_jaxpr`), its output shapes, and the
        draws it made (`_Draws`): on torchax's device, a key's, or on the host, fixed by the trace. Its
        outputs are the forward's and, for a decoding call (`cache`, the held caches, slots and filled
        slots of `_Cached`), every layer's cache after it."""
        env = _environment()
        draws = _Draws()

        def forward(params, buffers, tokens, key, positions, documents, valid, visible, key_places,
                    head_input, cache, reach):
            arguments = env.j2t_iso((dict(params), dict(buffers), tokens, positions, head_input))
            if reach is None and documents is not None:
                reach = document_reach(documents, valid, positions)
            cached = None if cache is None else _Cached(*cache)
            attending = (contextlib.nullcontext() if self.attention_impl is None else _attending(
                _Attending(self.attention_impl, documents, valid, max_segments, train, key, visible,
                           positions, key_places, cached)))
            self.module.train(train)
            try:
                with (env, env.override_property(prng=key) if key is not None else contextlib.nullcontext(),
                      draws, attending, _reaching(reach)):
                    outputs = env.t2j_iso(self.module.functional_call(_forward, *arguments))
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

        def __torch_function__(self, func, types, args=(), kwargs=None):
            tensor = func(*args, **(kwargs or {}))
            if func in drawing:
                if isinstance(tensor, torchax.tensor.Tensor):
                    self.device += 1
                else:
                    self.host += 1
            return tensor

    return Draws()


_OPMATH: Mapping[str, tuple[int, str] | None] = {
    "gelu": None, "silu": None, "native_layer_norm": None, "native_group_norm": None,
    "_softmax": (2, "half_to_float"), "_log_softmax": (2, "half_to_float")}
"""The aten ops torchax lowers into several roundings of a narrow float (jax.nn's gelu, silu,
softmax and log_softmax, and its layer and group norms' mean, variance and normalization),
where torch's kernels compute a bf16 or fp16 operand in fp32 and round each output once; by
the position and name of the argument that keeps the output fp32 (`half_to_float`), where the
op has one. Left to torchax, a GPT-NeoX's LayerNorm took the variance of its bf16-rounded deviations,
which scaled its logits by 1 - 8.9e-4 and put its bf16 loss 7.7 times torch's distance from
float64."""


@functools.cache
def _environment():
    """The torchax environment Dew's fallback runs transformers' forward in: torchax's own
    lowerings, but the `_OPMATH` ops computed at torch's opmath (`_at_opmath`). Its own, so
    the process's default environment keeps torchax's."""
    import torch
    from torchax.ops import ops_registry
    from torchax.tensor import Environment

    env = Environment()
    for name, keeps_fp32 in _OPMATH.items():
        packet = getattr(torch.ops.aten, name)
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


def _accumulating(closed: Jaxpr, *arguments: jax.Array) -> list[jax.Array]:
    """`closed` evaluated on `arguments` with each broadcast of a floating value
    narrower than fp32 summing its gradient in fp32 (`_accumulated`): an explicit
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
            if equation.primitive is primitives.jit_p:
                inner = equation.params["jaxpr"]
                outputs = evaluate(inner.jaxpr, inner.consts, inputs)
            elif equation.primitive is primitives.broadcast_in_dim_p and _narrow_float(inputs[0]):
                outputs = [_accumulated(inputs[0], tuple(equation.params["shape"]),
                                        tuple(equation.params["broadcast_dimensions"]))]
            else:
                if equation.primitive in _BROADCASTING:
                    shape = equation.outvars[0].aval.shape
                    rank = len(shape)
                    # An operand broadcasts on its trailing dimensions: a scalar on none.
                    inputs = [_accumulated(value, shape, tuple(range(rank - jnp.ndim(value), rank)))
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


_BROADCASTING = frozenset((primitives.add_p, primitives.sub_p, primitives.mul_p, primitives.div_p,
                           primitives.max_p, primitives.min_p, primitives.pow_p, primitives.atan2_p,
                           primitives.rem_p, primitives.nextafter_p, primitives.select_n_p,
                           primitives.clamp_p))
"""The elementwise primitives that broadcast an operand's size-1 dimensions implicitly."""


def _narrow_float(value) -> bool:
    """Whether `value` is a floating array narrower than fp32."""
    dtype = jnp.result_type(value)
    return jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits < 32


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _accumulated(value: jax.Array, shape: tuple[int, ...], dimensions: tuple[int, ...]) -> jax.Array:
    """`jax.lax.broadcast_in_dim`, whose gradient sums in fp32 and rounds once to `value`'s dtype."""
    return jax.lax.broadcast_in_dim(value, shape, dimensions)


def _accumulated_forward(value, shape, dimensions):
    return _accumulated(value, shape, dimensions), value


def _accumulated_backward(shape, dimensions, value, cotangent):
    _, transpose = jax.vjp(lambda wide: jax.lax.broadcast_in_dim(wide, shape, dimensions),
                           value.astype(jnp.float32))
    return (transpose(cotangent.astype(jnp.float32))[0].astype(value.dtype),)


_accumulated.defvjp(_accumulated_forward, _accumulated_backward)


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
    runs on Dew's (`TorchGraph.attention_impl`) reads packed and padded rows:
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

    def bank_sites(self, store: Variables) -> tuple[DecoderBank, ...]:
        return ()

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
                 positions=None, **packing) -> jax.Array:
        """The logits of `tokens`, in at least float32: of whole rows laid as `logits` lays them, or
        with `decode` of a decoding call into the cache `init_cache` opened, `attention_mask` `[B, S]`
        the tokens that write and `positions` `[B, S]` their places in their rows (their slots where
        None), or the rows' `coordinates` stating both; a decoding row holds one document."""
        fields = {name: value for name, value in ((VALIDITY_FIELD, attention_mask), ("positions", positions))
                  if value is not None}
        if not decode:
            return self.logits(tokens, train=train, **fields, **packing)
        if self.graph.attention_impl is None:
            raise Lacks("cache", "this model runs transformers' own attention, and a decode cache is the one "
                                 "Dew's attention writes")
        laid = self._laid(tokens, {**fields, **packing})
        if laid is not None and laid.max_segments > 1:
            raise ValueError("a decoding row continues one document, and these rows hold several")
        # The tokens that write: those the validity marks less the padding a row numbers 0 in its documents.
        real = real_slots({**fields, **packing}, jnp.shape(tokens)[:2])
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
        """Open the decode cache of `batch_size` rows (`Decoding`), every slot unfilled: one decoding
        call of no valid token, which writes nothing, as Dew's decoders open theirs."""
        tokens = jnp.zeros((batch_size, 1), jnp.int32)
        self(tokens, decode=True, attention_mask=jnp.zeros((batch_size, 1), bool))

    def with_cache_capacity(self, capacity: int) -> Self:
        """The model with a decode cache of `capacity` slots a row (`CacheCapacity`)."""
        return self.clone(max_seq_len=capacity)

    def _cached(self, rows: int) -> dict[int, Variable]:
        """Each attention layer's keys and values by its `layer_idx`, opened at `rows` rows of
        `max_seq_len` slots where they are not yet: their shapes and dtypes are those one decoding token
        of each row writes (`TorchGraph.step` over no held cache). The rows' cursor is the model's own
        (`cache_positions`), one for every layer."""
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

        return {layer: self.variable("cache", f"layer_{layer}", opened(written))
                for layer, written in sorted(shapes.items())}

    def _decoded(self, tokens, real, positions) -> jax.Array:
        tokens = jnp.asarray(tokens)
        rows, width = tokens.shape[:2]
        cache = self._cached(rows)
        # Each real token fills its row's next slot, the rest none (-1), as Dew's caches fill.
        slots, _ = cache_positions(self, rows, width, real)
        filled = filled_slots(self.get_variable("cache", "cache_index"), self.max_seq_len)
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
        precision = float32_products(compute, self.precision)
        return self.graph.step(params, buffers, tokens, cache=held, slots=slots, filled=filled,
                               positions=places, reach=reach, precision=precision)

    def logits(self, tokens, *, train: bool = False, **packing) -> jax.Array:
        """The logits of the whole rows, as `__call__` returns them: in at least float32
        (`at_least_fp32`), so a float64 model's are float64, as its output table's are."""
        logits, _ = self._run(tokens, train, self._laid(tokens, packing))
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
        laid = self._laid(tokens, packing)
        if attention_key_runs is not None:
            from dew.nn.attention_masks import visibility_mask

            # The runs as the mask they state, the stated visibility this path reads.
            length = jnp.shape(tokens)[1]
            attention_pairwise_mask = visibility_mask(length, length, causal=False, sliding_window=None,
                                                      key_runs=attention_key_runs)[:, 0]
        if attention_pairwise_mask is None:
            return self._run(tokens, train, laid)[1]
        if self.graph.attention_impl is None:
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

    def _laid(self, tokens, packing: Mapping[str, object]) -> Coordinates | None:
        """The rows' `Coordinates` from their packing fields, None for none: a model on Dew's attention
        reads `coordinates`, else `segment_ids`, `positions` and the validity field; one on
        transformers' own refuses them (`_refuse_packing`)."""
        if self.graph.attention_impl is None:
            _refuse_packing(packing)
            return None
        unknown = sorted(set(packing) - {COORDINATES_FIELD, VALIDITY_FIELD, "segment_ids", "positions"})
        if unknown:
            raise Lacks("packing", f"a tier-3 model reads its rows' documents, positions and validity, "
                                   f"not {unknown}")
        if not packing:
            return None
        laid = packing.get(COORDINATES_FIELD)
        if isinstance(laid, Coordinates):
            return laid
        batch, length = jnp.shape(tokens)[:2]
        return coordinates((batch, length), segments=packing.get("segment_ids"),
                           valid=packing.get(VALIDITY_FIELD), positions=packing.get("positions"))

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
        return self.graph(params, buffers, jnp.asarray(tokens), train=train, key=key,
                          precision=float32_products(compute, self.precision), laid=laid, visible=visible,
                          key_places=key_places, head_input=head_input)

    def _operands(self, held) -> tuple[dict[str, jax.Array], dict[str, jax.Array], Dtype]:
        """The forward's parameters at `dtype`, its buffers, and the dtype its products take: torch's
        operators state no precision, so the products take the one its compute dtype does
        (`float32_products`), `dtype` or the dtype its floating parameters hold."""
        params = {name: self._compute(leaf) for name, leaf in held["params"].items()}
        buffers = {name: jnp.asarray(leaf) for name, leaf in held.get("buffers", {}).items()}
        compute = self.dtype or jnp.result_type(*[leaf for leaf in params.values()
                                                  if jnp.issubdtype(leaf.dtype, jnp.floating)])
        return params, buffers, compute

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
         param_dtype: str, attention_impl: str, max_seq_len: int | None,
         admission: TorchAdmission | None = None, weights: bool = True) -> PretrainedFallback:
    """Load the causal LM `source` selects through transformers and torchax.

    `root` is the metadata snapshot `Pretrained.load` resolved at `source`'s
    revision, and `source`'s folder selects the model below it; the weights
    come from the same commit and folder (`SourceReference.weights`), so the
    file selection is the native loader's. `admission` is the decision that
    sent an unregistered type here (`admitted`), None for
    `fallback="torchax"`. Without `weights` the model is built from its
    config on torch's meta device, nothing but metadata fetched or read, and
    its variables are their shapes and dtypes alone (`jax.ShapeDtypeStruct`).
    """
    name_or_dir, revision = source.name, source.revision
    directory = source.directory(root)
    attends = admission is not None and admission.attends
    if attention_impl != "auto" and not attends:
        raise ValueError(
            f"attention_impl={attention_impl!r} selects a Dew kernel, and this source runs "
            f"transformers' own attention ({'fallback=torchax' if admission is None else admission.why}); "
            "leave attention_impl at 'auto'")
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(
            f"fallback='torchax' builds the model a transformers config.json describes, and "
            f"{name_or_dir} has none; load a repo that ships config.json and its weights")
    config = json.loads((directory / "config.json").read_text())
    refuse_quantized(config)
    try:
        import torch
        from torchax.interop import JittableModule, extract_all_buffers
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ImportError(f"fallback='torchax' needs torch, torchax and transformers: {INSTALL}") from error
    if weights:
        directory = source.directory(source.weights(root))
    # Before transformers, whose reader fails on a malformed file without naming it.
    generation_config = read_generation_config(directory)
    storage = "auto" if param_dtype == AUTO else getattr(torch, param_dtype)
    if weights:
        model = AutoModelForCausalLM.from_pretrained(str(directory), dtype=storage, local_files_only=True,
                                                     experts_implementation=EXPERTS)
    else:
        from transformers import AutoConfig

        stated = AutoConfig.from_pretrained(str(directory), local_files_only=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                stated, dtype=stated.dtype if storage == "auto" else storage, experts_implementation=EXPERTS)
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
    if head_name is not None and not (_plain_head(model, head) if weights else _shrunk_plain_head(config)):
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
        if weights:
            return host_view(leaf, name)
        return jax.ShapeDtypeStruct(tuple(leaf.shape), host_dtype(leaf, name))

    params = {name: held(leaf, name) for name, leaf in jittable.params.items()}
    buffers = {name: held(leaf, name) for name, leaf in extract_all_buffers(model)[1].items()}
    stored = model.state_dict()
    persistent = tuple(name for name in buffers if name in stored)
    vocab = int(head_weight.shape[0])
    # Torch's copies are no longer read: the forward runs on the arrays above.
    torch.nn.Module.to(model, "meta")
    graph = TorchGraph(jittable, persistent, attention_impl=attention_impl if attends else None,
                       shapes={"params": _shapes(params), "buffers": _shapes(buffers)})
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
        + "no sharding rules beyond TorchLayout's name table; the load holds the "
        "torch model on the host, at least 2x the checkpoint", TransformersTierWarning, stacklevel=3)
    built = {**config, "dtype": dtype, "fallback": "torchax", **versions}
    return PretrainedFallback(module, {"params": params, "buffers": buffers}, processor, config, directory,
                              built, generation_config, export_adapter=graph.export, revision=revision)


def _shapes(leaves: Mapping[str, np.ndarray | jax.ShapeDtypeStruct]) -> dict[str, jax.ShapeDtypeStruct]:
    """Each leaf's shape and dtype."""
    return {name: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for name, leaf in leaves.items()}


def _shrunk_plain_head(config: Mapping[str, object]) -> bool:
    """`_plain_head` of the class `config` builds, at the probe's sizes (`verify.shrink`) with
    transformers' own initialization: the head's structure, which no size changes."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    fields = {key: value for key, value in verify.shrink(config).items() if key != "model_type"}
    stated = AutoConfig.for_model(str(config["model_type"]), **fields)
    model = AutoModelForCausalLM.from_config(stated, dtype=torch.float32, experts_implementation=EXPERTS)
    return _plain_head(model.eval(), model.get_output_embeddings())


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
