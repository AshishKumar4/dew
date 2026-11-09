"""Load an unregistered decoder through a registered family's convention, verified against transformers.

This is the second loading tier. A registered family (tier 1) is parity-tested
against a pinned transformers release. A `model_type` Dew does not register
loads here as the first registered family whose convention it follows: its
config reads as that family reads one and every tensor name maps as that
family maps it. The candidates (`CONVENTIONS`) are the Llama convention,
which reads every field the families share (`decoder_config.PLAIN_DECODER`), and
then every registered causal family in `decoder_families.ENTRIES` order.
The fields a candidate does not read are not trusted by name. Before any
weight downloads, the installed transformers builds its own class for the
type from the same config with the sizes shrunk, gives it random weights and
runs it; Dew loads that checkpoint through each candidate and runs it too.
The load goes ahead as the first candidate that agrees. A type none computes
is `Unmapped`, with each candidate's reason and the probe's shrunken config,
on which tier 3 decides whether transformers' own forward runs it
(`torchax_fallback.admitted`). A type whose probe cannot run (the config does
not shrink by the standard sizes, or transformers builds or runs no shrunken
model) is refused as `ProbeUnavailable`, naming the generic route
(`fallback="torchax"`).
The probe runs LENGTH tokens, past every position-dependent length the config
states that `shrink` knows to shrink: a window, a chunk and a rope ramp's
original context. A field of another name that acts only past LENGTH
positions agrees there and is not caught.

The same probe tools/hf_reference.py writes the committed fixtures with:
`scatter_weights`, `probe_ids` and `reference_logits` live here, so the
fixtures and the load-time check run one reference recipe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import tempfile
import warnings
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import NoneType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util as flax_traverse

from dew import _records, parity as parity_check
from dew._cache import package_digest
from dew.interop import decoder_config, decoder_parts, hf_decoders as decoders, parts, sources
from dew.interop.decoder_families import ENTRIES
from dew.nn import precision
from dew.nn.blocks import padded_row
from dew.nn.kernels.options import owned_program
from dew.nn.sharding import abstract
from dew.objectives.base import Variables
from dew.registry import models

if TYPE_CHECKING:
    import torch
    from flax import linen as nn
    from transformers import PretrainedConfig, PreTrainedModel

CONVENTIONS: tuple[tuple[str, decoder_parts.DecoderFamily], ...] = (
    # Llama's convention reads every field the shared decoder states, windows
    # included (`decoder_config.PLAIN_DECODER`), where the registered Llama
    # family reads only what LlamaConfig declares; the probe, not the
    # reference's declarations, is what admits a type's reading.
    ('llama', replace(decoders.families()['llama'], config=decoder_config.decoder_layout())),
    # A family whose tensors are rewritten or split before its path map
    # reads them (`prepare`, `packed`: GPT-2's buffers, fused experts) is
    # left out: its names cannot be checked before the weights are read.
    *((family.model_types[0], family) for family in ENTRIES
      if family.model_types[0] != 'llama' and not family.bidirectional
      and family.prepare is decoder_parts.DecoderFamily.prepare and not family.packed),
)
"""The registered families an unregistered type is tried as, in order, each
by its name and the entry it translates and maps through."""

PARTS = 'parts'
"""The candidate assembled from the registered families' parts (`dew.interop.parts`), tried first: each of
its parts is read from the source's own modeling code."""

BATCH, LENGTH = 2, 12
"""The probe's token grid, which the committed fixtures share."""
_DRAWN = (8, 16)
"""Each of check 1's draws of the logits' token grid (`logits_parity`), as a test's `model_draw` draws:
past LENGTH, so a window the probe tells conventions apart by is applied on every draw."""
_GRADIENT_DRAWN = (128, 4)
"""Each of check 1's draws of the gradients' token grid (`gradient_parity`): many short sequences, so a
gradient summed over the draw is filled, and a recurrence's backward, which carries one rounding event
along a sequence into every parameter, sums over enough independent sequences for its error to size
(Falcon-H1's mamba sizes at 128 x 4, not at 64 x 8 nor 8 x 16)."""

_HEAD_DIM = 16
_VOCAB = 256
# Below LENGTH, so a window the reference applies and the convention does not
# (or the reverse) shows in the probe's logits.
_WINDOW = 4
# A rope ramp's original context, below LENGTH too: a ramp rescales the
# frequencies whose wavelengths pass the original context, which within it
# turn too little to show, so a ramp the reference applies and the convention
# does not (or the reverse) shows only past it.
_CONTEXT = 8
# The groups a type splits its heads into, which its heads stay a multiple of at the probe's size:
# CodeGen's fused qkv in four (transformers' `mp_num`).
_HEAD_GROUPS = {'codegen': 4}
# A rope record's per-frequency factors, one per rotated pair (Phi-3's and PhiMoE's LongRoPE).
_FREQUENCY_FACTORS = frozenset({'long_factor', 'short_factor'})
# Probe fields Dew never reads that would change what the reference builds or
# runs, not what it computes: `auto_map` asks transformers to run the
# checkpoint's own code, and the probe runs in fp32 whatever the checkpoint
# stores. The quantization record is the codec's, and the probe's weights are
# plain floats.
_PROBE_DROPPED = frozenset({'auto_map', 'dtype', 'torch_dtype'}) | decoder_parts.CODEC_FIELDS
_INSTALL = "pip install 'dewml[torch]'"
_FALLBACK = 'Pretrained.load(..., fallback="torchax")'
# Tier 2 judges a candidate as every test of Dew against a reference is judged: check 1 (`dew.parity`),
# Dew's RMS error from the source's float64 twin on par with the source's own float32 one, leaf by leaf,
# over draws `dew.parity.seeds_for` sizes. Two float32 roundings of one computation pass, and a Dew more
# exact than the source; a computation Dew does not share, in the forward or in a backward the forward
# does not show (a hook, a custom autograd function), sets Dew as far from the twin as it moves a value.


_log = logging.getLogger(__name__)

_REMOTE_CLASSES = ('AutoConfig', 'AutoModelForCausalLM')
"""The `auto_map` entries a remote source's verification builds from: its config and its causal LM."""


@dataclass(frozen=True)
class ModelSource:
    """Where an unregistered type's classes come from: transformers' own, or, only where the user opted
    in (`Pretrained.load(..., trust_remote_code=True)`, transformers' own flag), the repository's remote
    code at `remote`, each class its config's `auto_map` names (`references`) by its own and the rest
    transformers', as transformers resolves them."""

    model_type: str
    remote: Path | None = None
    references: Mapping[str, str] = field(default_factory=dict)
    """Each auto class's remote class reference, for the classes the `auto_map` names."""
    pinned: sources.SourceReference | None = None
    """Where the remote code was fetched: its repository, the Hub commit the load resolved and the folder
    `subfolder` selected, which together name that code (a commit names the whole repository, whose
    folders may each hold code of their own); None for transformers' own classes. A local directory's
    reference names no commit, so no code."""

    def config_class(self) -> type[PretrainedConfig]:
        from transformers import AutoConfig

        if 'AutoConfig' not in self.references:
            return type(AutoConfig.for_model(self.model_type))
        return self._remote_class('AutoConfig')

    def configured(self, fields: Mapping[str, object]) -> PretrainedConfig:
        """The source's config of `fields`, a config's JSON, its records of infinite floats read back as
        its own load reads them (`_records.decoded_floats`)."""
        from transformers import AutoConfig

        stated = _records.decoded_floats(fields)
        if 'AutoConfig' not in self.references:
            return AutoConfig.for_model(self.model_type, **stated)
        return self.config_class().from_dict(dict(stated))

    def causal_lm(self, config: PretrainedConfig) -> PreTrainedModel:
        """The source's causal LM of `config`, float32, on torch's current device."""
        from transformers import AutoModelForCausalLM

        if 'AutoModelForCausalLM' not in self.references:
            return AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        return self._remote_class('AutoModelForCausalLM')(config)

    def recordable(self) -> bool:
        """Whether a verification of these classes may be recorded for a later load (`_record_key`):
        transformers' own, which its release names, or remote code at the commit and folder that name
        it."""
        return self.remote is None or (self.pinned is not None and self.pinned.revision is not None)

    def _remote_class(self, auto: str):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(self.references[auto], str(self.remote))

    @classmethod
    def of(cls, model_type: str, hf_config: Mapping[str, object], remote: Path | None,
           pinned: sources.SourceReference | None = None) -> ModelSource:
        """The classes `hf_config` names: with `remote` (the opted-in repository's code, fetched where
        `pinned` names), each of `_REMOTE_CLASSES` its `auto_map` names is the repository's own, the
        others transformers'. A class in another repository (transformers' `repo--module.Class`) is
        refused: its code is neither the repository's, which the load pinned, nor named by the reference
        a verification is recorded under (`pinned`)."""
        held = _records.record(hf_config.get('auto_map') or {}, 'auto_map')
        if remote is None or not any(auto in held for auto in _REMOTE_CLASSES):
            return cls(model_type)
        references = {auto: _records.text(held[auto], f'auto_map {auto}') for auto in _REMOTE_CLASSES
                      if auto in held}
        elsewhere = sorted(auto for auto, reference in references.items() if '--' in reference)
        if elsewhere:
            raise _refuse(model_type, f"its auto_map names {elsewhere} in another repository, whose code the "
                          "load neither pinned nor verifies; load a repository that holds its own code")
        return cls(model_type, Path(remote), references, pinned)


class VerifiedMappingWarning(UserWarning):
    """A checkpoint loaded through the verified Llama convention (tier 2), not a registered family."""


def scatter_weights(model: torch.nn.Module, seed: int = 1234) -> None:
    """Random weights with something in every tensor.

    A freshly constructed model leaves the RMSNorm scales at their identity
    value, and a fixture whose norms are all ones or all zeros would pass a
    parity test that had the (1 + w) offset backwards.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            noise = torch.randn(tensor.shape, generator=generator) * 0.05
            tensor.copy_(tensor + noise if "norm" in name or "layernorm" in name
                         else noise * 4.0)
        # DeepSeek's balancing bias is a buffer the checkpoint carries, and
        # the reference selects on it: nonzero, or the load path that reads
        # it would agree with one that drops it. LFM2-MoE's and AFMoE's hold
        # it as `expert_bias`.
        for name, tensor in model.named_buffers():
            if name.endswith(("e_score_correction_bias", "expert_bias")):
                tensor.copy_(torch.linspace(-0.4, 0.4, tensor.shape[0]))


def probe_ids(vocab_size: int, seed: int = 0, shape: tuple[int, int] = (BATCH, LENGTH)) -> np.ndarray:
    """Return the probe's token ids of draw `seed`, `shape` (BATCH x LENGTH, the fixed ones a reference
    is run on, at seed 0)."""
    return np.random.RandomState(7 + seed).randint(0, vocab_size, shape).astype(np.int32)


def twin_logits(twin: PreTrainedModel, ids: np.ndarray) -> np.ndarray:
    """A source's float64 twin's (`source_twin`) logits over `ids`, run inside `_wide`: the truth tier
    2's parities measure both arms from."""
    import torch

    with _wide(), torch.no_grad():
        out = twin(input_ids=torch.from_numpy(ids).to(twin.device), use_cache=False)
    return out.logits.to(torch.float64).cpu().numpy()


def source_twin(model: PreTrainedModel, build: Callable[[], PreTrainedModel]) -> PreTrainedModel:
    """`model`'s float64 twin, eager attention and eager experts (torch's grouped product takes no
    float64): built by `build`, the source's own constructor, inside `_dew_torch.precision.float64`, so
    its constructor's tables are float64's and whatever its code attaches to what it builds (a hook on
    a gradient) is attached as it was to `model`, and its weights `model`'s (`float64_twin`). Run it
    inside `_wide`, under which the float32 pins the source's code carries (a rotary's `.float()`, eager
    softmax's `dtype=torch.float32`) are widened too."""
    from _dew_torch.precision import float64_twin

    twin = float64_twin(model, build)
    twin.set_attn_implementation("eager")
    twin.set_experts_implementation("eager")
    return twin


def _wide():
    """The context a float64 twin runs in (`_dew_torch.precision.float64`)."""
    from _dew_torch.precision import float64

    return float64()


def reference_logits(model: PreTrainedModel, ids: np.ndarray) -> np.ndarray:
    """Run a transformers model in eval mode with eager attention on its
    device and return fp32 logits."""
    import torch

    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).to(model.device), use_cache=False)
    return out.logits.to(torch.float32).cpu().numpy()


_SPELLED = frozenset({'hidden_size', 'num_attention_heads', 'num_hidden_layers'})
"""The sizes a config class may spell under its own keys that the probe reads by the class's
spelling (`PretrainedConfig.attribute_map`). A class's map also aliases other names to fields of
another meaning (GLM-4.7-Flash's `head_dim` to its rotary width), which the probe reads as stated."""


@dataclass(frozen=True)
class _Sizes:
    """A source config's standard sizes as its transformers config class reads them: the width, the
    heads and the depth under the config's own key for each (`_SPELLED`: OpenAI-GPT's `n_embd` for
    `hidden_size`), and one the config leaves unstated at the class's default."""

    config: Mapping[str, object]
    keys: Mapping[str, str]
    defaults: PretrainedConfig | None

    @classmethod
    def of(cls, hf_config: Mapping[str, object], source: ModelSource | None = None) -> _Sizes:
        """`hf_config`'s sizes as its `source`'s config class reads them; a type transformers does not
        know and no remote code states reads its keys as they are, with no defaults."""
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        model_type = hf_config.get('model_type')
        if source is None:
            if not isinstance(model_type, str) or model_type not in CONFIG_MAPPING:
                return cls(hf_config, {}, None)
            source = ModelSource(model_type)
        # A remote class may take required arguments, which the config states: read it whole.
        stated = {key: value for key, value in hf_config.items() if key != 'model_type'}
        defaults = source.configured(stated) if 'AutoConfig' in source.references else source.config_class()()
        spelled = {name: key for name, key in type(defaults).attribute_map.items() if name in _SPELLED}
        return cls(hf_config, spelled, defaults)

    def aliased(self, name: str) -> str:
        """The key the config class reads `name` under through its attribute map (Longcat Flash's
        `moe_topk` for `num_experts_per_tok`), every alias of it: for the routed and latent sizes the
        probe shrinks by their meaning, which each alias keeps."""
        return name if self.defaults is None else type(self.defaults).attribute_map.get(name, name)

    def key(self, name: str) -> str:
        """The config's own key for the standard size `name`."""
        return self.keys.get(name, name)

    def given(self, name: str) -> int | None:
        """The size `name` as the config states it under its own key or the standard one, which its
        class reads alike, else None, a null stating nothing (CWM's `head_dim`, derived from the width
        and the heads); a stated size that is no whole number is a ValueError."""
        for spelled in (self.key(name), name):
            if self.config.get(spelled) is not None:
                return _records.integer(self.config[spelled], spelled)
        return None

    def stated(self, name: str) -> int:
        """The size `name` as the config states it, else its class's default, a KeyError naming it
        where neither gives it."""
        size = self.given(name)
        if size is None and name in _SPELLED and getattr(self.defaults, name, None) is not None:
            size = _records.integer(getattr(self.defaults, name), name)
        if size is None:
            raise KeyError(name)
        return size


def shrink(hf_config: Mapping[str, object], source: ModelSource | None = None) -> Mapping[str, object]:
    """Return a config with the sizes the convention reads made small and the rest kept.

    The layer count stays here: a per-layer field the convention does not
    read (SmolLM3's `no_rope_layers`, one entry per layer with every fourth
    layer off) would read as all-on over two layers and hide the difference,
    so `shallow` cuts it only to the layers that hold every kind the stated
    depth builds. The query-to-key-head ratio stays, and so does whether the hidden width
    is the heads' total width; a stated window or chunk, and a rope ramp's
    stated original context, shrink below the probe's length so each acts,
    and a token id past the probe's vocabulary moves inside it. Each standard
    size is read and written under the source's own key for it (`_Sizes`), a
    size the network needs (the heads, the width, the depth) the config leaves
    unstated at its class's default, and a size neither states is a KeyError.
    """
    sizes = _Sizes.of(hf_config, source)
    key = sizes.key
    heads, hidden = sizes.stated('num_attention_heads'), sizes.stated('hidden_size')
    kv_heads = sizes.given('num_key_value_heads') or heads
    head_dim = sizes.given('head_dim') or hidden // heads
    # Two key heads, or as many as a type splits its heads into (`_HEAD_GROUPS`).
    small_kv = max(min(kv_heads, 2), _HEAD_GROUPS.get(str(hf_config.get('model_type')), 1))
    small_heads = small_kv * (heads // kv_heads)
    small_hidden = small_heads * _HEAD_DIM * (1 if hidden == heads * head_dim else 2)
    # Each spelled size is written once, under the config's own key: a standard name its class
    # aliases (`_SPELLED`) would set the size again over it.
    aliases = {name for name in _SPELLED if key(name) != name}
    fields = {key: value for key, value in hf_config.items()
              if key not in _PROBE_DROPPED and key != 'model_type' and key not in aliases}
    fields.update({key('hidden_size'): small_hidden, key('num_attention_heads'): small_heads,
                   key('intermediate_size'): 2 * small_hidden, key('vocab_size'): _VOCAB})
    if (depth := sizes.given('num_hidden_layers')) is not None:
        # The depth stays as stated, under the config's own key for it.
        fields[key('num_hidden_layers')] = depth
    # A routed or latent-attention type keeps its structure at small widths:
    # four experts, two per token, and small latent ranks.
    # A stated query-key width is the latent heads' two parts together; identity ('zero') experts keep
    # their place beside the routed ones.
    small = {'num_experts': 4, 'n_routed_experts': 4, 'num_local_experts': 4, 'num_experts_per_tok': 2,
             'moe_intermediate_size': 2 * small_hidden, 'shared_expert_intermediate_size': 2 * small_hidden,
             'expert_ffn_hidden_size': 2 * small_hidden, 'zero_expert_num': 2,
             'kv_lora_rank': _HEAD_DIM, 'q_lora_rank': 2 * _HEAD_DIM, 'qk_nope_head_dim': _HEAD_DIM,
             'qk_rope_head_dim': _HEAD_DIM // 2, 'qk_head_dim': _HEAD_DIM + _HEAD_DIM // 2,
             'v_head_dim': _HEAD_DIM, 'n_group': 1, 'topk_group': 1}
    fields.update({sizes.aliased(name): value for name, value in small.items()
                   if sizes.aliased(name) in hf_config})
    if 'first_k_dense_replace' in hf_config:
        fields['first_k_dense_replace'] = min(_records.integer(hf_config['first_k_dense_replace'],
                                                              'first_k_dense_replace'), 1)
    if key('num_key_value_heads') in hf_config:
        fields[key('num_key_value_heads')] = small_kv
    if isinstance(per_layer := hf_config.get('num_attention_heads_per_layer'), list):
        # Each layer's own query heads (Laguna's) keep their share of the key heads.
        counts = _records.integers(per_layer, 'num_attention_heads_per_layer')
        fields['num_attention_heads_per_layer'] = [count // kv_heads * small_kv for count in counts]
    if key('head_dim') in hf_config:
        # A latent-attention config whose head_dim states its rotary width (Longcat Flash's) keeps the
        # two equal.
        stated = hf_config[key('head_dim')]
        rotary = 'qk_rope_head_dim' in hf_config and stated == hf_config['qk_rope_head_dim']
        fields[key('head_dim')] = _HEAD_DIM // 2 if rotary else _HEAD_DIM
    # A window, a chunk and a ramp's original context shrink below the probe's length, so the probe runs
    # past each. A state space layer's scan chunk is one too: its chunks are the scan's own exact
    # arithmetic, which the probe crosses between chunks only below its length, and a stated one past it
    # (Falcon-H1-3B's 128) pads each of the gradients' four-token rows to the whole chunk, where
    # transformers' torch scan holds a [rows, chunk, chunk, heads, head width] product: its float32
    # gradient of 128 rows took 9.8 GiB.
    for positional in ('sliding_window', 'attention_chunk_size', 'mamba_chunk_size', 'chunk_size'):
        if isinstance(hf_config.get(positional), int):
            fields[positional] = _WINDOW
    small_head_dim = _HEAD_DIM if key('head_dim') in hf_config else small_hidden // small_heads
    partial = _records.number(hf_config.get('partial_rotary_factor', 1.0), 'partial_rotary_factor')
    if isinstance(rotary := hf_config.get('rotary_dim'), int):
        # A stated rotary width (CodeGen's, GPT-J's) keeps its share of the head (`_HEAD_DIM`'s).
        fields['rotary_dim'] = max(2, rotary * small_head_dim // head_dim // 2 * 2)
    for record in ('rope_parameters', 'rope_scaling'):
        if isinstance(rope := hf_config.get(record), Mapping):
            fields[record] = _within_probe(rope, small_head_dim, partial)
    if 'mamba_n_heads' in hf_config:
        # A Mamba mixer's heads tile its expanded width, which GraniteMoeHybrid's config validates: two
        # heads, one group, a small state.
        # A stated inner width (Falcon-H1's mamba_d_ssm) is the expanded hidden's, shrunk with it.
        width = _records.integer(hf_config.get('mamba_expand', 2), 'mamba_expand') * small_hidden
        fields['mamba_n_heads'] = 2
        if isinstance(hf_config.get('mamba_d_head'), int):
            fields['mamba_d_head'] = width // 2
        if isinstance(hf_config.get('mamba_d_ssm'), int):
            fields['mamba_d_ssm'] = width
        fields.update({name: value for name, value in (('mamba_n_groups', 1), ('mamba_d_state', _HEAD_DIM))
                       if name in hf_config})
    if isinstance(hf_config.get('original_max_position_embeddings'), int):
        fields['original_max_position_embeddings'] = _CONTEXT
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        value = hf_config.get(key)
        if isinstance(value, int):
            fields[key] = min(value, _VOCAB - 1)
        elif isinstance(value, list):
            fields[key] = [min(token, _VOCAB - 1) for token in _records.integers(value, key)]
    return fields


def _within_probe(record: Mapping[str, object], head_dim: int, partial: float) -> Mapping[str, object]:
    """A rope record with the original context it states at `_CONTEXT`, and its per-frequency factors
    (LongRoPE's `long_factor` and `short_factor`, one per rotated pair) cut to the pairs the probe's
    `head_dim` rotates, the record's partial rotary share where it states one, else the config's
    `partial`: a record's own, or each layer type's of a record nested by layer type."""
    if 'original_max_position_embeddings' in record or _FREQUENCY_FACTORS & set(record):
        share = _records.number(record.get('partial_rotary_factor', partial), 'partial_rotary_factor')
        half = int(head_dim * share) // 2
        cut = {name: factors[:half] for name in _FREQUENCY_FACTORS
               if isinstance(factors := record.get(name), list)}
        stated = 'original_max_position_embeddings' in record
        return {**record, **({'original_max_position_embeddings': _CONTEXT} if stated else {}), **cut}
    return {key: _within_probe(value, head_dim, partial) if isinstance(value, Mapping) else value
            for key, value in record.items()}


def _unmapped(names: Collection[str], record: decoder_parts.DecoderFields,
              family: decoder_parts.DecoderFamily) -> list[str]:
    """Return the tensor names `family` has no path for; a `packed` one is placed by its parts'
    paths, which reading it checks."""
    missing = []
    for name in sorted(names):
        try:
            if family.packing(name) is None:
                family.weight_path(name, record)
        except ValueError:
            missing.append(name)
    return missing


def _refuse(model_type: str, detail: str) -> ValueError:
    return ValueError(
        f"model_type {model_type!r} is not a registered family and cannot load through a "
        f"registered family's verified convention (tier 2): {detail}. To run transformers' own "
        f"forward instead, opt into the generic route with {_FALLBACK}")


class ProbeUnavailable(ValueError):
    """An unregistered type whose probe cannot run: its config does not shrink by the standard sizes,
    or transformers builds or runs no shrunken model of it. Tier 2 cannot map such a source; tier 3 may
    qualify it on its full config, each program it runs proven causal (`torchax_fallback.qualified`,
    ledger 1595)."""


def _unavailable(model_type: str, detail: str) -> ProbeUnavailable:
    return ProbeUnavailable(*_refuse(model_type, detail).args)


@dataclass(frozen=True)
class Unmapped:
    """The verdict on an unregistered type no registered convention computes: why each candidate did
    not (`reasons`), and the probe's shrunken config (`fields`, `shrink`), on which tier 3 decides
    whether transformers' own forward runs the source (`torchax_fallback.admitted`)."""

    model_type: str
    reasons: Mapping[str, str]
    fields: Mapping[str, object]


class _Mismatch(ValueError):
    """One candidate convention that does not compute the reference's model."""


def _translate(hf_config: Mapping[str, object], convention: str,
               family: decoder_parts.DecoderFamily) -> tuple[decoder_parts.DecoderFields, set[str]]:
    # Another family's reader on a foreign config may trip on a field's
    # type or length as well as refuse it; all mean it does not read it.
    try:
        return decoder_parts.translate_family_config(hf_config, family)
    except (KeyError, ValueError, TypeError, IndexError) as error:
        raise _Mismatch(f"its config does not read as {convention}'s ({error!r})") from error


@contextmanager
def _quiet() -> Iterator[None]:
    """Hold transformers' progress bars and warnings while the probe runs out of sight."""
    from transformers.utils import logging

    verbosity, bars = logging.get_verbosity(), logging.is_progress_bar_enabled()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        yield
    finally:
        logging.set_verbosity(verbosity)
        if bars:
            logging.enable_progress_bar()


@dataclass(frozen=True)
class VerifiedMapping:
    """The outcome of a passed probe: what agreed, by how much, against what."""

    model_type: str
    family: str
    """The registered family the type loads as (a `CONVENTIONS` name)."""
    reference: str
    """The transformers class and release the probe ran."""
    seeds: int
    """The probe's draws check 1 held Dew's route's fp32 logits on par with the reference's over, each
    from the reference's float64 twin (`logits_parity`), as `dew.parity.seeds_for` sized them."""
    inert: tuple[str, ...]
    """The config fields the convention does not read, which changed nothing on the probe's shrunken
    instance: a field whose effect starts past a context the probe never reaches (one `shrink` does not
    know to shrink below it) is not shown inert by it."""
    source: ModelSource | None = None
    """Where the type's classes came from, where not transformers' own (remote code, opted into)."""

    def entry(self, hf_config: Mapping[str, object],
              inventory: Collection[str] | None = None) -> decoder_parts.DecoderFamily:
        """The family the real config loads through: the convention's, or, for a type verified from parts,
        the one its real-sized skeleton assembles (`parts.family`), its stored names read over the
        checkpoint's `inventory` (`parts.assemble`)."""
        if self.family == PARTS:
            source = self.source or ModelSource(self.model_type)
            return parts.family(self.model_type, _assembled(source, hf_config, inventory))
        return dict(CONVENTIONS)[self.family]

    def record(self, hf_config: Mapping[str, object],
               entry: decoder_parts.DecoderFamily | None = None) -> decoder_parts.DecoderFields:
        """The real config's record through `entry` (`entry`'s family by default), its tensor names
        unchecked: what a load's selection of the checkpoint's tensors (its prediction depths,
        `DecoderFamily.prediction_depth`) reads before `translate` checks the selection's names."""
        entry = self.entry(hf_config) if entry is None else entry
        try:
            record, _ = _translate(hf_config, self.family, entry)
        except _Mismatch as error:
            raise _refuse(self.model_type, str(error)) from error
        return record

    def translate(self, hf_config: Mapping[str, object], tensor_names: Collection[str],
                  entry: decoder_parts.DecoderFamily | None = None) -> decoder_parts.DecoderFields:
        """Translate the real config and check the real tensor names, then warn once.

        The probe held the shrunk config; the real one differs only in sizes,
        and its tensors may carry names the shrunk reference did not build.
        `entry` is the family `entry` gave for this config, where the caller holds it.
        """
        entry = self.entry(hf_config, tensor_names) if entry is None else entry
        record = self.record(hf_config, entry)
        unmapped = _unmapped(tensor_names, record, entry)
        if unmapped:
            raise _refuse(self.model_type, f"the tensors {unmapped[:8]} have no path in the {self.family!r} "
                          "convention it was verified as")
        loads = ("from the parts of the registered families, each read from its own modeling code"
                 if self.family == PARTS else f"as the registered {self.family!r} family")
        warnings.warn(
            f"tier 2: verified mapping: model_type {self.model_type!r} loads {loads}; "
            f"its fp32 logits on a random shrunken instance are on par with {self.reference}'s from its "
            f"float64 twin by check 1 over {self.seeds} draws; it does not read the config fields "
            f"{list(self.inert)}, which changed nothing on that instance",
            VerifiedMappingWarning, stacklevel=3)
        return record


def _saved_names(directory: Path) -> frozenset[str]:
    """The tensor names a source's `save_pretrained` wrote to `directory`, read from its safetensors'
    headers."""
    from safetensors import safe_open

    names: set[str] = set()
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, "np") as stored:
            names.update(stored.keys())
    return frozenset(names)


def _assembled(source: ModelSource, hf_config: Mapping[str, object],
               inventory: Collection[str] | None = None) -> parts.Assembly:
    """The source's causal LM at `hf_config`'s own sizes, built on torch's meta device (so no layer
    allocates), assembled from parts over the checkpoint's `inventory` (`parts.assemble`)."""
    import torch

    fields = {key: value for key, value in hf_config.items()
              if key not in _PROBE_DROPPED and key != 'model_type'}
    with torch.device('meta'):
        skeleton = source.causal_lm(source.configured(fields))
    try:
        return parts.assemble(skeleton, inventory)
    except parts.Ambiguous as error:
        raise _refuse(source.model_type, f"its real-sized model does not assemble as its probe did "
                      f"({error})") from error


_LAYER_INDEXES = frozenset({'layer_idx', 'layer_id', 'layer_index', 'layer_number'})
"""The attributes a layer's modules hold its own place under, which tell layers apart without telling
their computations apart."""


def _skeleton(source: ModelSource, fields: Mapping[str, object]) -> tuple[PretrainedConfig, PreTrainedModel]:
    """transformers' config for the type at `fields` and the causal LM it builds on torch's meta device, so
    an unshrunk layer allocates nothing; `ProbeUnavailable` where it builds none."""
    model_type = source.model_type
    try:
        import torch
        from transformers import __version__
    except ImportError as error:
        raise _refuse(model_type, f"the check runs transformers' {model_type} model, which needs torch "
                      f"({error}); install it with {_INSTALL}") from error
    # The reference is transformers' code (or the opted-into remote code) for a type Dew does not
    # know, so anything it raises means the probe cannot run, not a Dew bug.
    try:
        config = source.configured(fields)
        with torch.device('meta'):
            return config, source.causal_lm(config)
    except Exception as error:
        raise _unavailable(model_type, f"transformers {__version__} builds no causal LM from its shrunken "
                           f"config ({type(error).__name__}: {error})") from error


def _module_kind(path: str, module: torch.nn.Module) -> tuple:
    """A module's kind: its place under its layer or model, its class, its public scalar attributes (a
    layer's place aside, `_LAYER_INDEXES`) and its own parameters' and buffers' shapes."""
    return (path, type(module).__module__, type(module).__qualname__,
            tuple(sorted((key, value) for key, value in vars(module).items()
                         if not key.startswith('_') and key not in _LAYER_INDEXES
                         and isinstance(value, (bool, int, float, str, NoneType)))),
            tuple((name, tuple(tensor.shape)) for name, tensor in (
                *module.named_parameters(recurse=False), *module.named_buffers(recurse=False))))


def _layer_kinds(source: ModelSource, fields: Mapping[str, object], listed: Collection[str] | None = None
                 ) -> tuple[list[tuple], tuple, tuple[str, ...]] | None:
    """The kinds of the model `fields` builds (`_skeleton`): each decoder layer's, its entry in each of the
    config's per-layer lists (one as long as the depth, stated or derived: SmolLM3's `no_rope_layers`,
    HY-3's `mlp_layer_types`; those `listed` by name where given, so a list as long as a shallower depth
    by chance, Falcon-H1's two `mlp_multipliers` at two layers, is no layer's) and each of its modules'
    (`_module_kind`) across every module list as long as the depth; every module's outside those lists
    (its table, its rotary, its final norm); and the lists' names. None
    where no module list is as long as the depth, or where the model is not built by transformers' own
    code of transformers' and torch's own classes (a remote source; a config class, or a causal LM class
    the auto registry resolves it to, registered from elsewhere), whose layers differ only as their kinds
    show: a source's own code may add what no kind shows (a hook on its later layers, a rotary built by
    its depth)."""
    import torch

    config, model = _skeleton(source, fields)
    depth = _records.integer(config.num_hidden_layers, 'num_hidden_layers')
    # The code that built the model, a remote class's constructor or a class registered from elsewhere, is
    # the source's own even where what it returns is transformers' classes.
    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

    if source.remote is not None:
        return None
    builders = (type(config), MODEL_FOR_CAUSAL_LM_MAPPING[type(config)])
    own = ('transformers.', 'torch.')
    if any(not built.__module__.startswith('transformers.') for built in builders) or any(
            not type(module).__module__.startswith(own) for module in model.modules()):
        return None
    stacks = [module for module in model.modules() if isinstance(module, torch.nn.ModuleList)
              and len(module) == depth]
    if not stacks:
        return None
    lists = sorted((name, tuple(value)) for name, value in vars(config).items()
                   if isinstance(value, (list, tuple))
                   and (len(value) == depth if listed is None else name in listed))
    inside = {id(module) for stack in stacks for module in stack.modules()}
    outside = tuple(_module_kind(path, module) for path, module in model.named_modules()
                    if id(module) not in inside)
    layers = [(tuple(json.dumps(values[index], sort_keys=True, default=str) for _, values in lists),
               tuple(tuple(_module_kind(path, module) for path, module in stack[index].named_modules())
                     for stack in stacks)) for index in range(depth)]
    return layers, outside, tuple(name for name, _ in lists)


def shallow(source: ModelSource, fields: Mapping[str, object]) -> Mapping[str, object]:
    """`fields` (`shrink`'s) cut to the fewest first layers that hold every kind of layer its stated depth
    builds (`_layer_kinds`), at least two, each stated per-layer list cut alike: HY-3's 80 layers probe as
    its dense first and a routed one, SmolLM3's 36 as its first four, the fourth without positions. The
    cut holds only where its layers are the deep model's first ones kind for kind and every module
    outside them is the deep model's; the depth stays otherwise, and where no kind is told
    (`_layer_kinds` is None)."""
    deep = _layer_kinds(source, fields)
    if deep is None:
        return fields
    kinds, outside, listed = deep
    firsts = {}
    for index, layer in enumerate(kinds):
        firsts.setdefault(layer, index)
    count = max(max(firsts.values()) + 1, min(2, len(kinds)))
    if count == len(kinds):
        return fields
    key = _Sizes.of(fields, source).key('num_hidden_layers')
    cut = {name: (list(value[:count]) if isinstance(value, (list, tuple)) and len(value) == len(kinds)
                  and name != key else value) for name, value in fields.items()}
    cut[key] = count
    return cut if _layer_kinds(source, cut, listed) == (kinds[:count], outside, listed) else fields


def _reference_config(source: ModelSource, fields: Mapping[str, object]) -> tuple[PretrainedConfig, int, str]:
    """transformers' config for the type at the probe's sizes, the parameter
    count of the model it builds (on torch's meta device, so an unshrunk
    layer allocates nothing), and the class's name and release."""
    config, skeleton = _skeleton(source, fields)
    from transformers import __version__

    size = sum(parameter.numel() for parameter in skeleton.parameters())
    origin = f"transformers {__version__}'s" if source.remote is None else f"{source.remote.name}'s remote"
    return config, size, f"{origin} {type(skeleton).__name__}"


def _built(model_type: str, convention: str, family: decoder_parts.DecoderFamily,
           fields: Mapping[str, object]) -> tuple[nn.Module, decoder_parts.DecoderFields, set[str], int]:
    """A candidate's model at the probe's sizes, its record, the fields it
    left unread and its parameter count."""
    record, inert = _translate({**fields, 'model_type': model_type}, convention, family)
    model = models.build("causal_transformer", {**record, "dtype": "float32", "attention_impl": "reference"})
    try:
        template = abstract(lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))
    except ValueError as error:
        raise _Mismatch(f"{convention}'s model does not build at the probe's sizes ({error})") from error
    return model, record, inert, sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(template['params']))


def _agreement(convention: str, family: decoder_parts.DecoderFamily, model: nn.Module,
               record: decoder_parts.DecoderFields, inert: set[str], name: str, source: _Source,
               checkpoint: Path, pad: int | None) -> int:
    """One candidate against the reference's saved checkpoint, `source` the reference and its float64
    twin: the draws check 1 held its logits on par over, or `_Mismatch` naming how it differs. `pad` is
    the row the reference's own table pads (its `padding_idx`, which torch counts from
    the table's end where negative), which takes no gradient there (ledger
    1593) and which eval logits do not show, so the candidate pads that row,
    its own id read as its lookup reads it (`padded_row`), or is no reading
    of the reference."""
    stated = record.get('padding_idx')
    row = padded_row(None if stated is None else _records.integer(stated, 'padding_idx'),
                     _records.integer(record['vocab_size'], 'vocab_size'))
    if row != pad:
        raise _Mismatch(f"{convention} pads row {row} of the token table where {name}'s table pads {pad}")
    tensors = sources.load_shards(checkpoint)
    unmapped = _unmapped(tensors, record, family)
    if unmapped:
        raise _Mismatch(f"the tensors {unmapped[:8]} have no {convention} path")
    try:
        variables = decoders.translate_weights(tensors, record, family)
        decoder_parts.check_decoder_tree(variables, model)
    except (KeyError, ValueError) as error:
        # A candidate's reading of tensors it does not hold disqualifies that candidate alone.
        raise _Mismatch(f"its tensors do not fill {convention}'s model ({error})") from error
    try:
        return logits_parity(source.logits, model, variables, f"{convention}'s logits against {name}'s")
    except parity_check.GateFailure as failure:
        raise _Mismatch(f"{convention}'s fp32 logits on a random shrunken instance are not on par with "
                        f"{name}'s from its float64 twin ({failure}), so its config fields {sorted(inert)} "
                        f"or its modeling compute something {convention} does not") from failure


def _users_arithmetic():
    """The precision the parities run Dew's decoder at (`logits_parity`, `gradient_parity`): a float32 run
    as a user's process runs it, its products summed in float32 (`precision.FP32_SUMS`), not in TF32 on a
    GPU nor in float64 as a test process's wide CPU arithmetic sums them
    (`precision.WIDE_CPU_ARITHMETIC`), which would set it nearer the truth than users get, as check 1's
    arms run (`_summed_as_run`)."""
    return precision.configured_precision(precision.FP32_SUMS)


@dataclass(frozen=True)
class _Source:
    """The probe's reference at float32 and its float64 twin (`source_twin`), whose runs over each draw
    the parities judge Dew's against."""

    reference: PreTrainedModel
    twin: PreTrainedModel

    def logits(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The reference's float32 logits over `ids` and its twin's, as float64."""
        return (np.asarray(reference_logits(self.reference, ids.astype(np.int64)), np.float64),
                twin_logits(self.twin, ids.astype(np.int64)))


def _judged(draw, label: str) -> int:
    """Check 1's verdict over `draw` (seed -> Dew's, the source's and the twin's trees), at the seeds
    `dew.parity.seeds_for` sizes on draws the verdict never sees: those seeds where every leaf is on par,
    or `dew.parity.GateFailure` naming each leaf that is not, `dew.parity.Inconclusive` one unresolved."""
    seeds = parity_check.seeds_for(draw, label)
    parity_check.judge(draw, seeds, label)
    return seeds


def logits_parity(source: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], model: nn.Module,
                  variables: Variables, label: str) -> int:
    """Dew's float32 logits over the probe's draws of ids (`probe_ids`) on par with the source's from its
    float64 twin (`_judged`), `source` the two over a draw's ids (`_Source.logits`), Dew's run as a user's
    process runs it (`_users_arithmetic`)."""
    held = jax.tree.map(lambda leaf: jnp.asarray(leaf, 'float32'), variables)
    forward = owned_program(lambda variables, ids: model.apply(variables, ids), model)

    def draw(seed: int):
        ids = probe_ids(_VOCAB, seed, _DRAWN)
        with _users_arithmetic():
            ours = np.asarray(forward(held, ids), np.float64)
        theirs, truth = source(ids)
        return ours, theirs, truth

    return _judged(draw, label)


def _probe_gradients(model: PreTrainedModel, ids: np.ndarray, cotangent: np.ndarray, *,
                     train: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The source's logits over `ids` and every trained parameter's gradient of their sum against
    `cotangent`, by the parameter's stored name, in training or eval mode, eager attention. A trained
    parameter the backward reaches no path to (one read under `torch.no_grad`, as DeepSeek V3.2's
    lightning indexer is, whose top-k selection carries no gradient) has a gradient of zeros, which torch
    leaves unset."""
    import torch

    model.train(train)
    model.zero_grad()
    logits = model(input_ids=torch.from_numpy(ids.astype(np.int64)), use_cache=False).logits
    (logits * torch.from_numpy(cotangent)).sum().backward()
    gradients = {name: (np.zeros(tuple(parameter.shape), np.float32) if parameter.grad is None
                        else parameter.grad.detach().numpy().copy())
                 for name, parameter in model.named_parameters() if parameter.requires_grad}
    model.eval()
    return logits.detach().numpy(), gradients


def _as_stored(model: PreTrainedModel, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`tensors`, by `model`'s parameter names, as its `save_pretrained` stores them: through the reverse of
    the checkpoint conversion its load applies (`revert_weight_conversion`; a routed type's fused experts
    are stored one tensor per expert), so each lands where its parameter's stored tensor does."""
    import torch
    from transformers.core_model_loading import revert_weight_conversion

    stored = revert_weight_conversion(model, {name: torch.from_numpy(np.asarray(value))
                                              for name, value in tensors.items()})
    return {name: tensor.numpy() for name, tensor in stored.items()}


def gradient_parity(source: _Source, family: decoder_parts.DecoderFamily,
                    record: decoder_parts.DecoderFields, variables: Variables, label: str) -> int:
    """Dew's float32 gradient of every parameter on par with the source's from its float64 twin's
    (`_judged`), over the probe's draws of ids and cotangents, every parameter's gradient laid end to end
    as one leaf: a backward the forward does not show moves some parameter's gradient by a share of its
    value, which no other's rounding hides, where a parameter's own leaf holds too few values a draw for
    check 1 to size (a learned activation's scalars, an embedding's rows a few tokens touch, a norm's
    scale). The source's eval-mode gradients,
    its float32 ones and its twin's, are stored as `save_pretrained` stores its weights (`_as_stored`) and
    read into Dew's tree as they are (`translate_weights`; a gradient rearranges as its parameter does)."""
    model = models.build('causal_transformer', {**record, 'dtype': 'float32', 'attention_impl': 'reference'})
    held = jax.tree.map(lambda leaf: jnp.asarray(leaf, 'float32'), variables)

    def gradient(params, ids, cotangent):
        return jax.grad(lambda params: jnp.sum(model.apply({**held, 'params': params}, ids) * cotangent))(
            params)

    # Held by this qualification alone, which a closure over its probe's arrays is (`owned_program`).
    compiled = owned_program(gradient, model)

    def theirs(network: PreTrainedModel, ids: np.ndarray, cotangent: np.ndarray, dtype: str) -> dict:
        _, gradients = _probe_gradients(network, ids, cotangent, train=False)
        tree = decoders.translate_weights(_as_stored(source.reference, gradients), record, family,
                                          param_dtype=dtype)
        return {'/'.join(path): np.asarray(leaf, np.float64)
                for path, leaf in flax_traverse.flatten_dict(tree['params']).items()}

    def draw(seed: int):
        ids = probe_ids(_VOCAB, seed, _GRADIENT_DRAWN)
        cotangent = np.random.RandomState(11 + seed).standard_normal((*_GRADIENT_DRAWN, _VOCAB)).astype(
            np.float32)
        reference = theirs(source.reference, ids, cotangent, 'float32')
        with _wide(), jax.enable_x64(new_val=True):
            truth = theirs(source.twin, ids, cotangent.astype(np.float64), 'float64')
        with _users_arithmetic():
            ours = {'/'.join(path): np.asarray(leaf, np.float64) for path, leaf in
                    flax_traverse.flatten_dict(compiled(held['params'], ids, cotangent)).items()}
        if set(ours) != set(reference):
            raise _Mismatch(f"its gradients fill {sorted(set(ours) ^ set(reference))[:4]} apart")
        return tuple(np.concatenate([tree[path].ravel() for path in sorted(ours)])
                     for tree in (ours, reference, truth))

    return _judged(draw, label)


def _trained(source: _Source, assembly: parts.Assembly, family: decoder_parts.DecoderFamily,
             record: decoder_parts.DecoderFields, name: str) -> None:
    """The training qualification of a decoder assembled from parts, beside the probe's eval logits:
    the source's training forward and gradients, its attention dropout set to zero, are its eval ones
    bitwise (no training-only behavior, a hook or another branch, that Dew's decoder lacks), and
    Dew's gradients are on par with its own from its float64 twin (`gradient_parity`: no backward the
    forward does not show). `_Mismatch` names what differs."""
    reference = source.reference
    for path, attribute in assembly.dropouts:
        setattr(reference.get_submodule(path), attribute, 0.0)
    ids = probe_ids(_VOCAB)
    cotangent = np.random.RandomState(11).standard_normal((BATCH, LENGTH, _VOCAB)).astype(np.float32)
    trained, evaluated = (_probe_gradients(reference, ids, cotangent, train=train) for train in (True, False))
    if not (np.array_equal(trained[0], evaluated[0]) and trained[1].keys() == evaluated[1].keys()
            and all(np.array_equal(trained[1][key], evaluated[1][key]) for key in trained[1])):
        raise _Mismatch(f"{name}'s training forward or gradient differs from its eval one with its dropout "
                        "zeroed, a training behavior Dew's decoder does not have")
    with tempfile.TemporaryDirectory() as folder:
        reference.save_pretrained(folder)
        variables = decoders.translate_weights(sources.load_shards(Path(folder)), record, family)
    try:
        gradient_parity(source, family, record, variables, f"Dew's gradients against {name}'s")
    except parity_check.GateFailure as failure:
        raise _Mismatch(f"Dew's gradients are not on par with {name}'s from its float64 twin ({failure}), so "
                        "its backward computes something Dew's does not") from failure


def verify_mapping(hf_config: Mapping[str, object], *, remote: Path | None = None,
                   pinned: sources.SourceReference | None = None) -> VerifiedMapping | Unmapped:
    """Check an unregistered model_type against transformers before its weights download.

    The first candidate is the decoder assembled from the registered families' parts (`PARTS`,
    `dew.interop.parts`), each read from the source's own code; the whole-family `CONVENTIONS`
    follow. `remote` is the repository whose remote code the user opted into running
    (`Pretrained.load(..., trust_remote_code=True)`): the type's config and model classes are then
    its own, by its config's `auto_map`, and `pinned` the repository, Hub commit and folder that code
    was fetched at. A verification passed before, under the same config, transformers release, Dew
    implementation and remote code at the same commit and folder, is read back without the probe
    (`_recorded`); a local directory's code, which no commit names, is verified at each load.

    transformers builds its class for the type from `shrink(hf_config)` with
    random weights (`scatter_weights`) and saves it as safetensors; each
    `CONVENTIONS` candidate whose reading takes the real config loads that
    directory and runs the same ids in fp32. The first whose logits are on
    par with the reference's from its float64 twin by check 1
    (`logits_parity`), is the mapping. When several read the config and match,
    the `CONVENTIONS` order is the tie-break: each computed the probe's
    logits, so any of them is sound. With none, the type is `Unmapped`, with
    every candidate's reason and the shrunken config. A type transformers has
    no class for is refused (its repo's own code is not run), and one whose
    probe cannot run is refused as `ProbeUnavailable`.
    """
    model_type = hf_config.get('model_type')
    if not isinstance(model_type, str):
        raise ValueError("config.json states no model_type, so neither a family nor a reference "
                         "class can be chosen for it")
    from transformers import __version__
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    source = ModelSource.of(model_type, hf_config, remote, pinned)
    if model_type not in CONFIG_MAPPING and len(source.references) < len(_REMOTE_CLASSES):
        raise _refuse(model_type, f"transformers {__version__} has no class for it, and its repo's own "
                      "code is run only where the load opts in (trust_remote_code=True) and its config's "
                      f"auto_map names both {list(_REMOTE_CLASSES)}")
    held = None if source.remote is None else source
    key = _record_key(hf_config, source) if source.recordable() else None
    recorded = None if key is None else _recorded(key)
    if recorded is not None:
        return replace(recorded, source=held)
    _log.warning("tier 2: model_type %r is no registered family; verifying which convention computes it "
                 "against its %s code by check 1 on a shrunken instance, its forward and its backward over "
                 "drawn inputs: seconds for most types, about two minutes for a recurrent hybrid's "
                 "backward (Falcon-H1's on 24 CPU cores), done once a type%s.", model_type,
                 'remote' if source.remote is not None else "transformers'",
                 ' and recorded' if key is not None else ' (its local code at each load)')
    verdict = _verified(hf_config, source)
    if isinstance(verdict, VerifiedMapping):
        if key is not None:
            _record(key, verdict)
        return replace(verdict, source=held)
    if source.remote is not None:
        # Tier 3 runs transformers' own classes, which a remote type has none of.
        raise _refuse(model_type, f"its remote code assembles from no known parts and follows no "
                      f"registered convention ({verdict.reasons})")
    return verdict


_VERIFIED = 'verified'
"""The directory under Dew's cache a passed verification is recorded in, one file per key."""


def _record_key(hf_config: Mapping[str, object], source: ModelSource) -> str:
    """What a verification holds for: the config, the transformers and torch releases, Dew's own
    implementation (`package_digest`, so any change of a reader, the shrink or a model verifies again)
    and the repository, commit and folder the remote code was fetched at (`ModelSource.pinned`), as one
    digest."""
    import importlib.metadata

    from transformers import __version__

    remote = None if source.pinned is None else asdict(source.pinned)
    return hashlib.sha256(json.dumps({'config': hf_config, 'transformers': __version__, 'remote': remote,
                                      'torch': importlib.metadata.version('torch'),
                                      'dew': package_digest()},
                                     sort_keys=True, default=str).encode()).hexdigest()


def _recorded(key: str) -> VerifiedMapping | None:
    from dew._cache import dew_cache_dir

    path = Path(dew_cache_dir()) / _VERIFIED / key / 'mapping.json'
    if not path.is_file():
        return None
    record = _records.record(json.loads(path.read_text()), 'verified mapping')
    return VerifiedMapping(
        _records.text(record['model_type'], 'model_type'), _records.text(record['family'], 'family'),
        _records.text(record['reference'], 'reference'), _records.integer(record['seeds'], 'seeds'),
        tuple(_records.strings(record['inert'], 'inert')))


def _record(key: str, verified: VerifiedMapping) -> None:
    from dew import _files
    from dew._cache import dew_cache_dir

    directory = Path(dew_cache_dir()) / _VERIFIED / key
    with _files.staged(directory) as staging:
        (staging / 'mapping.json').write_text(json.dumps({
            'model_type': verified.model_type, 'family': verified.family, 'reference': verified.reference,
            'seeds': verified.seeds, 'inert': list(verified.inert)}))


def _verified(hf_config: Mapping[str, object], source: ModelSource) -> VerifiedMapping | Unmapped:
    """The probe itself (`verify_mapping`)."""
    model_type = source.model_type
    reasons: dict[str, str] = {}
    candidates = []
    # Remote code is admitted by its code alone (`parts`): a whole family's convention is checked on the
    # probe's eval forward, which another training forward, gradient or rotary past the probe's sizes
    # would pass.
    for convention, family in () if source.remote is not None else CONVENTIONS:
        try:
            _translate(hf_config, convention, family)
            candidates.append((convention, family))
        except _Mismatch as error:
            reasons[convention] = str(error)
    with _quiet():
        try:
            shrunk = shrink(hf_config, source)
            _Sizes.of(hf_config, source).stated('num_hidden_layers')
        except (KeyError, ValueError) as error:
            # A reader may take a config by its own names or its defaults (OPT's
            # defaults every size); the probe shrinks it by the standard names.
            raise _unavailable(model_type, f"its config states no {error.args[0]!r} the probe shrinks it "
                               "by") from error
        fields = shallow(source, shrunk)
        config, size, name = _reference_config(source, fields)
        # A candidate that builds another architecture is dismissed on its
        # parameter count before its own model is run.
        sized = []
        for convention, family in candidates:
            try:
                model, record, inert, expected_size = _built(model_type, convention, family, fields)
            except _Mismatch as mismatch:
                reasons[convention] = str(mismatch)
                continue
            if expected_size == size:
                sized.append((convention, family, model, record, inert))
            else:
                reasons[convention] = (f"{name} holds {size} parameters at the probe's sizes where "
                                       f"{convention}'s model holds {expected_size}")
        # The probe runs whether or not a candidate is left, so a type no convention computes is
        # one whose shrunken model ran (`Unmapped`), and one whose model does not run is
        # `ProbeUnavailable`, whatever the candidates.
        from _dew_torch.precision import widened_config

        try:
            reference = source.causal_lm(config).float()
            scatter_weights(reference)
            reference_logits(reference, probe_ids(_VOCAB).astype(np.int64))
            probed = _Source(reference, source_twin(
                reference, lambda: source.causal_lm(widened_config(copy.deepcopy(config)))))
        except Exception as error:
            raise _unavailable(model_type, f"{name} does not run its shrunken config "
                               f"({type(error).__name__}: {error})") from error
        # The row the source's table pads, nn.Embedding's notion (Gemma's scaled table is one); a lookup
        # of another kind has no padding_idx to pad by. The reference ran, so torch is installed.
        import torch

        lookup = reference.get_input_embeddings()
        pad = lookup.padding_idx if isinstance(lookup, torch.nn.Embedding) else None
        with tempfile.TemporaryDirectory() as scratch:
            # Saved first: the parts read the checkpoint's names as its load does, over its whole
            # inventory where its conversion renames a key by another's presence (`parts.assemble`).
            reference.save_pretrained(scratch)
            assembly = None
            try:
                assembly = parts.assemble(reference, _saved_names(Path(scratch)))
            except parts.Ambiguous as ambiguous:
                reasons[PARTS] = str(ambiguous)
            else:
                record = assembly.fields
                try:
                    model = models.build("causal_transformer",
                                         {**record, "dtype": "float32", "attention_impl": "reference"})
                except ValueError as refused:
                    # Parts each read alike may still combine into a decoder Dew refuses to build.
                    reasons[PARTS] = f"Dew builds no decoder of its assembled parts ({refused})"
                else:
                    sized.insert(0, (PARTS, parts.family(model_type, assembly), model, record, set()))
            if sized:
                for convention, family, model, record, inert in sized:
                    try:
                        seeds = _agreement(convention, family, model, record, inert, name, probed,
                                           Path(scratch), pad)
                        if convention == PARTS:
                            assert assembly is not None
                            _trained(probed, assembly, family, record, name)
                    except _Mismatch as mismatch:
                        reasons[convention] = str(mismatch)
                        continue
                    return VerifiedMapping(model_type, convention, name, seeds, tuple(sorted(inert)))
    return Unmapped(model_type,
                    {name: reasons[name] for name in (PARTS, *dict(CONVENTIONS)) if name in reasons},
                    {**fields, 'model_type': model_type})
