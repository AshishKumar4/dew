"""Load an unregistered decoder through a registered family's convention, verified against transformers.

This is the second loading tier. A registered family (tier 1) is parity-tested
against a pinned transformers release. A `model_type` Dew does not register
loads here as the first registered family whose convention it follows: its
config reads as that family reads one and every tensor name maps as that
family maps it. The candidates (`CONVENTIONS`) are the Llama convention,
which reads every field the families share (`hf_decoders.base_config`), and
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
The probe runs LENGTH tokens, so a field that acts only past that many
positions agrees there and is not caught.

The same probe tools/hf_reference.py writes the committed fixtures with:
`scatter_weights`, `probe_ids` and `reference_logits` live here, so the
fixtures and the load-time check run one reference recipe.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import warnings
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util as flax_traverse

from dew import records
from dew.interop import decoder_parts, hf_decoders as decoders, parts, sources
from dew.interop.decoder_families import ENTRIES
from dew.objectives.base import Variables
from dew.registry import models

if TYPE_CHECKING:
    import torch
    from flax import linen as nn
    from transformers import PretrainedConfig, PreTrainedModel

CONVENTIONS: tuple[tuple[str, decoder_parts.DecoderFamily], ...] = (
    # Llama's convention reads every field `base_config` shares, windows
    # included, where the registered Llama family reads only what
    # LlamaConfig declares; the probe, not the reference's declarations, is
    # what admits a type's reading.
    ('llama', replace(decoders.families()['llama'], translate_config=lambda hf_config, used:
                      decoder_parts.base_config(hf_config, used, rope=decoder_parts.read_rope(
                          hf_config, used, decoder_parts.PLAIN_SPELLING)))),
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

_HEAD_DIM = 16
_VOCAB = 256
# Below LENGTH, so a window the reference applies and the convention does not
# (or the reverse) shows in the probe's logits.
_WINDOW = 4
# Probe fields Dew never reads that would change what the reference builds or
# runs, not what it computes: `auto_map` asks transformers to run the
# checkpoint's own code, and the probe runs in fp32 whatever the checkpoint
# stores. The quantization record is the codec's, and the probe's weights are
# plain floats.
_PROBE_DROPPED = frozenset({'auto_map', 'dtype', 'torch_dtype'}) | decoder_parts.CODEC_FIELDS
_INSTALL = "pip install 'dewml[torch]'"
_FALLBACK = 'Pretrained.load(..., fallback="torchax")'
# fp32 rounding between Dew and transformers on a registered family, in eps
# per layer per unit of the largest reference logit. The thirteen dense and
# routed tier-1 tiny fixtures (tests/fixtures/hf, written by
# tools/hf_reference.py with the probe's weight recipe) measure from 1.44
# (gemma3-tiny) to 6.46 (llama31-tiny: 9.89e-06 over two layers, logits up
# to 6.43). The probe's bound is twice the largest, as the
# released-checkpoint tests derive theirs. A field that changes the
# computation moves the probe's logits by tenths or more: Granite's
# multipliers by 4.2, SmolLM3's NoPE layers by 1.7.
_ROUNDING = 6.46
# How much farther the source's float32 gradients may be from the float64 truth than Dew's own float32
# gradients are (`gradient_parity`, pooled over every element of every parameter's gradient on the
# probe's draw): rounding alone puts the two arms alike whatever the model's conditioning, and a backward
# the source's forward does not show (a hook, a custom autograd function) orders of magnitude apart.
# Twice the largest tools/gradient_parity.py measured over the types and compositions that assemble
# from parts.
_GRADIENT_PARITY = 4.0


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

    def config_class(self) -> type[PretrainedConfig]:
        from transformers import AutoConfig

        if 'AutoConfig' not in self.references:
            return type(AutoConfig.for_model(self.model_type))
        return self._remote_class('AutoConfig')

    def configured(self, fields: Mapping[str, object]) -> PretrainedConfig:
        from transformers import AutoConfig

        if 'AutoConfig' not in self.references:
            return AutoConfig.for_model(self.model_type, **fields)
        return self.config_class().from_dict(dict(fields))

    def causal_lm(self, config: PretrainedConfig) -> PreTrainedModel:
        """The source's causal LM of `config`, float32, on torch's current device."""
        from transformers import AutoModelForCausalLM

        if 'AutoModelForCausalLM' not in self.references:
            return AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        return self._remote_class('AutoModelForCausalLM')(config)

    def code(self) -> str | None:
        """The remote code's identity, which a recorded verification names: the digest of every Python
        file the repository holds, the modules its `auto_map` names and every one they import from it;
        None for transformers' own."""
        if self.remote is None:
            return None
        digest = hashlib.sha256()
        for path in sorted(self.remote.rglob('*.py')):
            digest.update(path.relative_to(self.remote).as_posix().encode() + b'\0' + path.read_bytes())
        return digest.hexdigest()

    def _remote_class(self, auto: str):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(self.references[auto], str(self.remote))

    @classmethod
    def of(cls, model_type: str, hf_config: Mapping[str, object], remote: Path | None) -> ModelSource:
        """The classes `hf_config` names: with `remote` (the opted-in repository), each of
        `_REMOTE_CLASSES` its `auto_map` names is the repository's own, the others transformers'. A
        class in another repository (transformers' `repo--module.Class`) is refused: its code is
        neither the repository's, which the load pinned, nor in the digest a verification is recorded
        under (`code`)."""
        held = records.record(hf_config.get('auto_map') or {}, 'auto_map')
        if remote is None or not any(auto in held for auto in _REMOTE_CLASSES):
            return cls(model_type)
        references = {auto: records.text(held[auto], f'auto_map {auto}') for auto in _REMOTE_CLASSES
                      if auto in held}
        elsewhere = sorted(auto for auto, reference in references.items() if '--' in reference)
        if elsewhere:
            raise _refuse(model_type, f"its auto_map names {elsewhere} in another repository, whose code the "
                          "load neither pinned nor verifies; load a repository that holds its own code")
        return cls(model_type, Path(remote), references)


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
        # it would agree with one that drops it.
        for name, tensor in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                tensor.copy_(torch.linspace(-0.4, 0.4, tensor.shape[0]))


def probe_ids(vocab_size: int) -> np.ndarray:
    """Return the fixed BATCH x LENGTH token ids a reference is run on."""
    return np.random.RandomState(7).randint(0, vocab_size, (BATCH, LENGTH)).astype(np.int32)


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

    def key(self, name: str) -> str:
        """The config's own key for the standard size `name`."""
        return self.keys.get(name, name)

    def given(self, name: str) -> int | None:
        """The size `name` as the config states it under its own key or the standard one, which its
        class reads alike, else None, a null stating nothing (CWM's `head_dim`, derived from the width
        and the heads); a stated size that is no whole number is a ValueError."""
        for spelled in (self.key(name), name):
            if self.config.get(spelled) is not None:
                return records.integer(self.config[spelled], spelled)
        return None

    def stated(self, name: str) -> int:
        """The size `name` as the config states it, else its class's default, a KeyError naming it
        where neither gives it."""
        size = self.given(name)
        if size is None and name in _SPELLED and getattr(self.defaults, name, None) is not None:
            size = records.integer(getattr(self.defaults, name), name)
        if size is None:
            raise KeyError(name)
        return size


def shrink(hf_config: Mapping[str, object], source: ModelSource | None = None) -> Mapping[str, object]:
    """Return a config with the sizes the convention reads made small and the rest kept.

    The layer count stays: a per-layer field the convention does not read
    (SmolLM3's `no_rope_layers`, one entry per layer with every fourth
    layer off) would read as all-on over two layers and hide the difference.
    The query-to-key-head ratio stays, and so does whether the hidden width
    is the heads' total width; a stated window shrinks below the probe's
    length so it acts, and a token id past the probe's vocabulary moves
    inside it. Each standard size is read and written under the source's own key for it (`_Sizes`),
    a size the network needs (the heads, the width, the depth) the config leaves unstated at its
    class's default, and a size neither states is a KeyError.
    """
    sizes = _Sizes.of(hf_config, source)
    key = sizes.key
    heads, hidden = sizes.stated('num_attention_heads'), sizes.stated('hidden_size')
    kv_heads = sizes.given('num_key_value_heads') or heads
    head_dim = sizes.given('head_dim') or hidden // heads
    small_kv = min(kv_heads, 2)
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
    small = {'num_experts': 4, 'n_routed_experts': 4, 'num_local_experts': 4, 'num_experts_per_tok': 2,
             'moe_intermediate_size': 2 * small_hidden, 'shared_expert_intermediate_size': 2 * small_hidden,
             'kv_lora_rank': _HEAD_DIM, 'q_lora_rank': 2 * _HEAD_DIM, 'qk_nope_head_dim': _HEAD_DIM,
             'qk_rope_head_dim': _HEAD_DIM // 2, 'v_head_dim': _HEAD_DIM, 'n_group': 1, 'topk_group': 1}
    fields.update({key: value for key, value in small.items() if key in hf_config})
    if 'first_k_dense_replace' in hf_config:
        fields['first_k_dense_replace'] = min(records.integer(hf_config['first_k_dense_replace'],
                                                              'first_k_dense_replace'), 1)
    if key('num_key_value_heads') in hf_config:
        fields[key('num_key_value_heads')] = small_kv
    if key('head_dim') in hf_config:
        fields[key('head_dim')] = _HEAD_DIM
    # A window or a chunk shrinks below the probe's length, so the probe runs past it.
    for positional in ('sliding_window', 'attention_chunk_size'):
        if isinstance(hf_config.get(positional), int):
            fields[positional] = _WINDOW
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        value = hf_config.get(key)
        if isinstance(value, int):
            fields[key] = min(value, _VOCAB - 1)
        elif isinstance(value, list):
            fields[key] = [min(token, _VOCAB - 1) for token in records.integers(value, key)]
    return fields


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
    or transformers builds or runs no shrunken model of it. Neither tier 2 nor tier 3 can qualify such
    a source without the probe (ledger 1595)."""


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
    error: float
    """max |Δlogits| between Dew's convention route and the reference, fp32."""
    bound: float
    """What fp32 rounding alone reaches on the probe (`_ROUNDING`)."""
    inert: tuple[str, ...]
    """The config fields the convention does not read, which changed nothing on the probe's shrunken
    instance: a field whose effect starts past the probe's sizes (a context it never reaches) is not
    shown inert by it."""
    source: ModelSource | None = None
    """Where the type's classes came from, where not transformers' own (remote code, opted into)."""

    def entry(self, hf_config: Mapping[str, object]) -> decoder_parts.DecoderFamily:
        """The family the real config loads through: the convention's, or, for a type verified from parts,
        the one its real-sized skeleton assembles (`parts.family`)."""
        if self.family == PARTS:
            source = self.source or ModelSource(self.model_type)
            return parts.family(self.model_type, _assembled(source, hf_config))
        return dict(CONVENTIONS)[self.family]

    def translate(self, hf_config: Mapping[str, object], tensor_names: Collection[str],
                  entry: decoder_parts.DecoderFamily | None = None) -> decoder_parts.DecoderFields:
        """Translate the real config and check the real tensor names, then warn once.

        The probe held the shrunk config; the real one differs only in sizes,
        and its tensors may carry names the shrunk reference did not build.
        `entry` is the family `entry` gave for this config, where the caller holds it.
        """
        entry = self.entry(hf_config) if entry is None else entry
        try:
            record, _ = _translate(hf_config, self.family, entry)
        except _Mismatch as error:
            raise _refuse(self.model_type, str(error)) from error
        unmapped = _unmapped(tensor_names, record, entry)
        if unmapped:
            raise _refuse(self.model_type, f"the tensors {unmapped[:8]} have no path in the {self.family!r} "
                          "convention it was verified as")
        loads = ("from the parts of the registered families, each read from its own modeling code"
                 if self.family == PARTS else f"as the registered {self.family!r} family")
        warnings.warn(
            f"tier 2: verified mapping: model_type {self.model_type!r} loads {loads}; "
            f"{self.reference} agrees on a random shrunken instance to max "
            f"|Δlogits| {self.error:.2e} in fp32 (bound {self.bound:.2e}); it does not read the config "
            f"fields {list(self.inert)}, which changed nothing on that instance",
            VerifiedMappingWarning, stacklevel=3)
        return record


def _assembled(source: ModelSource, hf_config: Mapping[str, object]) -> parts.Assembly:
    """The source's causal LM at `hf_config`'s own sizes, built on torch's meta device (so no layer
    allocates), assembled from parts (`parts.assemble`)."""
    import torch

    fields = {key: value for key, value in hf_config.items()
              if key not in _PROBE_DROPPED and key != 'model_type'}
    with torch.device('meta'):
        skeleton = source.causal_lm(source.configured(fields))
    try:
        return parts.assemble(skeleton)
    except parts.Ambiguous as error:
        raise _refuse(source.model_type, f"its real-sized model does not assemble as its probe did "
                      f"({error})") from error


def _reference_config(source: ModelSource, fields: Mapping[str, object]) -> tuple[PretrainedConfig, int, str]:
    """transformers' config for the type at the probe's sizes, the parameter
    count of the model it builds (on torch's meta device, so an unshrunk
    layer allocates nothing), and the class's name and release."""
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
            skeleton = source.causal_lm(config)
    except Exception as error:
        raise _unavailable(model_type, f"transformers {__version__} builds no causal LM from its shrunken "
                           f"config ({type(error).__name__}: {error})") from error
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
        template = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))
    except ValueError as error:
        raise _Mismatch(f"{convention}'s model does not build at the probe's sizes ({error})") from error
    return model, record, inert, sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(template['params']))


def _agreement(convention: str, family: decoder_parts.DecoderFamily, model: nn.Module,
               record: decoder_parts.DecoderFields, inert: set[str], name: str, expected: np.ndarray,
               checkpoint: Path, layers: int) -> tuple[float, float]:
    """One candidate against the reference's saved checkpoint: its error and
    the bound, or `_Mismatch` naming how it differs."""
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
    # `_ROUNDING` was measured at fp32 matmul precision; a GPU's default runs
    # fp32 matmuls in TF32, which alone misses it.
    with jax.default_matmul_precision("highest"):
        actual = np.asarray(model.apply(variables, probe_ids(_VOCAB)))
    error = float(np.abs(actual - expected).max())
    bound = float(2 * _ROUNDING * np.finfo(np.float32).eps * layers * np.abs(expected).max())
    if not error <= bound:
        raise _Mismatch(f"{name} and {convention} disagree on a random shrunken instance by max "
                        f"|Δlogits| {error:.2e} in fp32, over the bound {bound:.2e}, so its config "
                        f"fields {sorted(inert)} or its modeling compute something {convention} does not")
    return error, bound


def _probe_gradients(model: PreTrainedModel, ids: np.ndarray, cotangent: np.ndarray, *,
                     train: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The source's logits over `ids` and every parameter's gradient of their sum against `cotangent`,
    by the parameter's stored name, in training or eval mode, eager attention."""
    import torch

    model.train(train)
    model.zero_grad()
    logits = model(input_ids=torch.from_numpy(ids.astype(np.int64)), use_cache=False).logits
    (logits * torch.from_numpy(cotangent)).sum().backward()
    gradients = {name: parameter.grad.detach().numpy().copy() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    model.eval()
    return logits.detach().numpy(), gradients


def gradient_parity(reference: PreTrainedModel, family: decoder_parts.DecoderFamily,
                    record: decoder_parts.DecoderFields, variables: Variables) -> float:
    """How much farther the source's float32 gradient of every parameter is from the float64 truth than
    Dew's float32 gradient is, pooled over every element on the probe's draw (the root of the ratio of
    their summed squared errors): the source's eval-mode gradients read into Dew's tree as its weights
    are (`translate_weights`; a gradient rearranges as its parameter does), the truth Dew's decoder of
    the same record and weights in float64."""
    ids = probe_ids(_VOCAB)
    cotangent = np.random.RandomState(11).standard_normal((BATCH, LENGTH, _VOCAB)).astype(np.float32)
    _, theirs = _probe_gradients(reference, ids, cotangent, train=False)
    their_tree = flax_traverse.flatten_dict(decoders.translate_weights(theirs, record, family)['params'])

    def gradient(dtype: str) -> dict:
        model = models.build('causal_transformer', {**record, 'dtype': dtype, 'attention_impl': 'reference'})
        held = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), variables)

        def loss(params) -> jax.Array:
            logits = model.apply({**held, 'params': params}, jnp.asarray(ids))
            return jnp.sum(jnp.asarray(logits) * jnp.asarray(cotangent, dtype))

        with jax.default_matmul_precision('highest'):
            return {path: np.asarray(leaf, np.float64) for path, leaf in
                    flax_traverse.flatten_dict(jax.grad(loss)(held['params'])).items()}

    ours = gradient('float32')
    with jax.enable_x64(new_val=True):
        truth = gradient('float64')
    if set(ours) != set(their_tree):
        raise _Mismatch(f"its gradients fill {sorted(set(ours) ^ set(their_tree))[:4]} apart")
    source_error = sum(float(np.sum((np.asarray(their_tree[path], np.float64) - truth[path]) ** 2))
                       for path in ours)
    dew_error = sum(float(np.sum((ours[path] - truth[path]) ** 2)) for path in ours)
    return math.sqrt(source_error / max(dew_error, np.finfo(np.float64).tiny))


def _trained(reference: PreTrainedModel, assembly: parts.Assembly, family: decoder_parts.DecoderFamily,
             record: decoder_parts.DecoderFields, name: str) -> None:
    """The training qualification of a decoder assembled from parts, beside the probe's eval logits:
    the source's training forward and gradients, its attention dropout set to zero, are its eval ones
    bitwise (no training-only behavior, a hook or another branch, that Dew's decoder lacks), and
    its gradients are as near the float64 truth as Dew's (`_GRADIENT_PARITY`: no backward the forward does
    not show). `_Mismatch` names what differs."""
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
    parity = gradient_parity(reference, family, record, variables)
    if not parity <= _GRADIENT_PARITY:
        raise _Mismatch(f"{name}'s gradients are {parity:.2f} times as far from the float64 truth as Dew's, "
                        f"past the {_GRADIENT_PARITY} rounding reaches")


def verify_mapping(hf_config: Mapping[str, object], *,
                   remote: Path | None = None) -> VerifiedMapping | Unmapped:
    """Check an unregistered model_type against transformers before its weights download.

    The first candidate is the decoder assembled from the registered families' parts (`PARTS`,
    `dew.interop.parts`), each read from the source's own code; the whole-family `CONVENTIONS`
    follow. `remote` is the repository whose remote code the user opted into running
    (`Pretrained.load(..., trust_remote_code=True)`): the type's config and model classes are then
    its own, by its config's `auto_map`. A verification passed before, under the same config,
    transformers release, Dew implementation and remote code, is read back without the probe
    (`_recorded`).

    transformers builds its class for the type from `shrink(hf_config)` with
    random weights (`scatter_weights`) and saves it as safetensors; each
    `CONVENTIONS` candidate whose reading takes the real config loads that
    directory and runs the same ids in fp32. The first whose max |Δlogits|
    is within twice `_ROUNDING` eps per layer per unit of the reference's
    largest logit is the mapping. When several read the config and match,
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

    source = ModelSource.of(model_type, hf_config, remote)
    if model_type not in CONFIG_MAPPING and len(source.references) < len(_REMOTE_CLASSES):
        raise _refuse(model_type, f"transformers {__version__} has no class for it, and its repo's own "
                      "code is run only where the load opts in (trust_remote_code=True) and its config's "
                      f"auto_map names both {list(_REMOTE_CLASSES)}")
    held = None if source.remote is None else source
    key = _record_key(hf_config, source)
    recorded = _recorded(key)
    if recorded is not None:
        return replace(recorded, source=held)
    verdict = _verified(hf_config, source)
    if isinstance(verdict, VerifiedMapping):
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
    implementation (every source file of `dew`, so any change of a reader, the shrink or a model
    verifies again) and the remote code, as one digest."""
    from transformers import __version__

    import dew

    digest = hashlib.sha256()
    import importlib.metadata

    digest.update(json.dumps({'config': hf_config, 'transformers': __version__, 'remote': source.code(),
                              'torch': importlib.metadata.version('torch')},
                             sort_keys=True, default=str).encode())
    package = Path(dew.__file__).parent
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def _recorded(key: str) -> VerifiedMapping | None:
    from dew.cache import dew_cache_dir

    path = Path(dew_cache_dir()) / _VERIFIED / key / 'mapping.json'
    if not path.is_file():
        return None
    record = records.record(json.loads(path.read_text()), 'verified mapping')
    return VerifiedMapping(
        records.text(record['model_type'], 'model_type'), records.text(record['family'], 'family'),
        records.text(record['reference'], 'reference'), records.number(record['error'], 'error'),
        records.number(record['bound'], 'bound'), tuple(records.strings(record['inert'], 'inert')))


def _record(key: str, verified: VerifiedMapping) -> None:
    from dew import files
    from dew.cache import dew_cache_dir

    directory = Path(dew_cache_dir()) / _VERIFIED / key
    with files.staged(directory) as staging:
        (staging / 'mapping.json').write_text(json.dumps({
            'model_type': verified.model_type, 'family': verified.family, 'reference': verified.reference,
            'error': verified.error, 'bound': verified.bound, 'inert': list(verified.inert)}))


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
            fields = shrink(hf_config, source)
            layers = _Sizes.of(hf_config, source).stated('num_hidden_layers')
        except (KeyError, ValueError) as error:
            # A reader may take a config by its own names or its defaults (OPT's
            # defaults every size); the probe shrinks it by the standard names.
            raise _unavailable(model_type, f"its config states no {error.args[0]!r} the probe shrinks it "
                               "by") from error
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
        try:
            reference = source.causal_lm(config).float()
            scatter_weights(reference)
            expected = reference_logits(reference, probe_ids(_VOCAB).astype(np.int64))
        except Exception as error:
            raise _unavailable(model_type, f"{name} does not run its shrunken config "
                               f"({type(error).__name__}: {error})") from error
        assembly = None
        try:
            assembly = parts.assemble(reference)
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
            with tempfile.TemporaryDirectory() as scratch:
                reference.save_pretrained(scratch)
                for convention, family, model, record, inert in sized:
                    try:
                        error, bound = _agreement(convention, family, model, record, inert, name, expected,
                                                  Path(scratch), layers)
                        if convention == PARTS:
                            assert assembly is not None
                            _trained(reference, assembly, family, record, name)
                    except _Mismatch as mismatch:
                        reasons[convention] = str(mismatch)
                        continue
                    return VerifiedMapping(model_type, convention, name, error, bound, tuple(sorted(inert)))
    return Unmapped(model_type,
                    {name: reasons[name] for name in (PARTS, *dict(CONVENTIONS)) if name in reasons},
                    {**fields, 'model_type': model_type})
