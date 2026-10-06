"""Load an unregistered decoder through a registered family's convention, verified against transformers.

This is the second loading tier. A registered family (tier 1) is parity-tested
against a pinned transformers release. A `model_type` Dew does not register
loads here as the first registered family whose convention it follows: its
config reads as that family reads one and every tensor name maps as that
family maps it. The candidates (`CONVENTIONS`) are the Llama convention,
which reads every field the families share (`decoder_config._base_config`), and
then every registered causal family in `hf_decoders.family_entries()` order.
The fields a candidate does not read are not trusted by name. Before any
weight downloads, the installed transformers builds its own class for the
type from the same config with the sizes shrunk, gives it random weights and
runs it; Dew loads that checkpoint through each candidate and runs it too.
The load goes ahead as the first candidate that agrees. Anything else is
refused with each candidate's reason, naming the generic route
(`fallback="torchax"`).
The probe runs LENGTH tokens, so a field that acts only past that many
positions agrees there and is not caught.

The same probe tools/hf_reference.py writes the committed fixtures with:
`scatter_weights`, `probe_ids` and `reference_logits` live here, so the
fixtures and the load-time check run one reference recipe.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew import records
from dew.interop import hf_decoders as decoders, sources
from dew.interop.decoder_config import _CODEC_FIELDS, _base_config, _translated
from dew.interop.decoder_family import _check_tree
from dew.registry import models, with_precision

if TYPE_CHECKING:
    import torch
    from flax import linen as nn
    from transformers import PretrainedConfig, PreTrainedModel

_MASKED = frozenset({'llada', 'dream', 'Dream', 'diffusion_gemma_text'})
"""The masked-diffusion families: bidirectional, never a causal LM's convention."""

CONVENTIONS: tuple[tuple[str, decoders.DecoderFamily], ...] = (
    # Llama's convention reads every field `_base_config` shares, windows
    # included, where the registered Llama family reads only what
    # LlamaConfig declares; the probe, not the reference's declarations, is
    # what admits a type's reading.
    ('llama', replace(decoders.families()['llama'], translate_config=_base_config)),
    # A family whose tensors are rewritten or split before its path map
    # reads them (`prepare`, `packed`: GPT-2's buffers, fused experts) is
    # left out: its names cannot be checked before the weights are read.
    *((family.model_types[0], family) for family in decoders.family_entries()
      if family.model_types[0] != 'llama' and not set(family.model_types) & _MASKED
      and family.prepare is decoders.DecoderFamily.prepare and not family.packed),
)
"""The registered families an unregistered type is tried as, in order, each
by its name and the entry it translates and maps through."""

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
_PROBE_DROPPED = frozenset({'auto_map', 'dtype', 'torch_dtype'}) | _CODEC_FIELDS
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
    """Run a transformers model in eval mode with eager attention and return fp32 logits."""
    import torch

    model.eval()
    model.set_attn_implementation("eager")
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids), use_cache=False)
    return out.logits.to(torch.float32).numpy()


def shrink(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """Return a config with the sizes the convention reads made small and the rest kept.

    The layer count stays: a per-layer field the convention does not read
    (SmolLM3's `no_rope_layers`, one entry per layer with every fourth
    layer off) would read as all-on over two layers and hide the difference.
    The query-to-key-head ratio stays, and so does whether the hidden width
    is the heads' total width; a stated window shrinks below the probe's
    length so it acts, and a token id past the probe's vocabulary moves
    inside it.
    """
    heads = records.integer(hf_config['num_attention_heads'], 'num_attention_heads')
    kv_heads = records.integer(hf_config.get('num_key_value_heads') or heads, 'num_key_value_heads')
    hidden = records.integer(hf_config['hidden_size'], 'hidden_size')
    head_dim = records.integer(hf_config.get('head_dim') or hidden // heads, 'head_dim')
    small_kv = min(kv_heads, 2)
    small_heads = small_kv * (heads // kv_heads)
    small_hidden = small_heads * _HEAD_DIM * (1 if hidden == heads * head_dim else 2)
    fields = {key: value for key, value in hf_config.items()
              if key not in _PROBE_DROPPED and key != 'model_type'}
    fields.update(hidden_size=small_hidden, num_attention_heads=small_heads,
                  intermediate_size=2 * small_hidden, vocab_size=_VOCAB)
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
    if 'num_key_value_heads' in hf_config:
        fields['num_key_value_heads'] = small_kv
    if 'head_dim' in hf_config:
        fields['head_dim'] = _HEAD_DIM
    if isinstance(hf_config.get('sliding_window'), int):
        fields['sliding_window'] = _WINDOW
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        value = hf_config.get(key)
        if isinstance(value, int):
            fields[key] = min(value, _VOCAB - 1)
        elif isinstance(value, list):
            fields[key] = [min(token, _VOCAB - 1) for token in records.integers(value, key)]
    return fields


def _unmapped(names: Collection[str], record: decoders.DecoderFields,
              family: decoders.DecoderFamily) -> list[str]:
    """Return the tensor names `family` has no path for."""
    missing = []
    for name in sorted(names):
        try:
            family.weight_path(name, record)
        except ValueError:
            missing.append(name)
    return missing


def _refuse(model_type: str, detail: str) -> ValueError:
    return ValueError(
        f"model_type {model_type!r} is not a registered family and cannot load through a "
        f"registered family's verified convention (tier 2): {detail}. To run transformers' own "
        f"forward instead, opt into the generic route with {_FALLBACK}")


class _Mismatch(ValueError):
    """One candidate convention that does not compute the reference's model."""


def _translate(hf_config: Mapping[str, object], convention: str,
               family: decoders.DecoderFamily) -> tuple[decoders.DecoderFields, set[str]]:
    # Another family's reader on a foreign config may trip on a field's
    # type or length as well as refuse it; all mean it does not read it.
    try:
        return _translated(hf_config, family)
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
    """The config fields the convention does not read, which the probe showed inert."""

    def translate(self, hf_config: Mapping[str, object],
                  tensor_names: Collection[str]) -> decoders.DecoderFields:
        """Translate the real config and check the real tensor names, then warn once.

        The probe held the shrunk config; the real one differs only in sizes,
        and its tensors may carry names the shrunk reference did not build.
        """
        entry = dict(CONVENTIONS)[self.family]
        try:
            record, _ = _translate(hf_config, self.family, entry)
        except _Mismatch as error:
            raise _refuse(self.model_type, str(error)) from error
        unmapped = _unmapped(tensor_names, record, entry)
        if unmapped:
            raise _refuse(self.model_type, f"the tensors {unmapped[:8]} have no path in the {self.family!r} "
                          "convention it was verified as")
        warnings.warn(
            f"tier 2: verified mapping: model_type {self.model_type!r} loads as the registered "
            f"{self.family!r} family; {self.reference} agrees on a random shrunken instance to max "
            f"|Δlogits| {self.error:.2e} in fp32 (bound {self.bound:.2e}), with the config fields "
            f"{list(self.inert)} shown inert",
            VerifiedMappingWarning, stacklevel=3)
        return record


def _reference_config(model_type: str, fields: Mapping[str, object]) -> tuple[PretrainedConfig, int, str]:
    """transformers' config for the type at the probe's sizes, the parameter
    count of the model it builds (on torch's meta device, so an unshrunk
    layer allocates nothing), and the class's name and release."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, __version__
    except ImportError as error:
        raise _refuse(model_type, f"the check runs transformers' {model_type} model, which needs torch "
                      f"({error}); install it with {_INSTALL}") from error
    # The reference is transformers' code for a type Dew does not know, so
    # anything it raises means the probe cannot run, not a Dew bug.
    try:
        config = AutoConfig.for_model(model_type, **fields)
        with torch.device('meta'):
            skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    except Exception as error:
        raise _refuse(model_type, f"transformers {__version__} builds no causal LM from its shrunken "
                      f"config ({type(error).__name__}: {error})") from error
    size = sum(parameter.numel() for parameter in skeleton.parameters())
    return config, size, f"transformers {__version__}'s {type(skeleton).__name__}"


def _built(model_type: str, convention: str, family: decoders.DecoderFamily,
           fields: Mapping[str, object]) -> tuple[nn.Module, decoders.DecoderFields, set[str], int]:
    """A candidate's model at the probe's sizes, its record, the fields it
    left unread and its parameter count."""
    record, inert = _translate({**fields, 'model_type': model_type}, convention, family)
    model = models.build("causal_transformer", with_precision(
        "causal_transformer", record, dtype="float32", attention_impl="reference"))
    try:
        template = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), jnp.int32)))
    except ValueError as error:
        raise _Mismatch(f"{convention}'s model does not build at the probe's sizes ({error})") from error
    return model, record, inert, sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(template['params']))


def _agreement(convention: str, family: decoders.DecoderFamily, model: nn.Module,
               record: decoders.DecoderFields, inert: set[str], name: str, expected: np.ndarray,
               checkpoint: Path, layers: int) -> tuple[float, float]:
    """One candidate against the reference's saved checkpoint: its error and
    the bound, or `_Mismatch` naming how it differs."""
    tensors = sources.load_shards(checkpoint)
    unmapped = _unmapped(tensors, record, family)
    if unmapped:
        raise _Mismatch(f"the tensors {unmapped[:8]} have no {convention} path")
    variables = decoders.translate_weights(tensors, record, convention)
    try:
        _check_tree(variables, model)
    except ValueError as error:
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


def verify_mapping(hf_config: Mapping[str, object]) -> VerifiedMapping:
    """Check an unregistered model_type against transformers before its weights download.

    transformers builds its class for the type from `shrink(hf_config)` with
    random weights (`scatter_weights`) and saves it as safetensors; each
    `CONVENTIONS` candidate whose reading takes the real config loads that
    directory and runs the same ids in fp32. The first whose max |Δlogits|
    is within twice `_ROUNDING` eps per layer per unit of the reference's
    largest logit is the mapping. When several read the config and match,
    the `CONVENTIONS` order is the tie-break: each computed the probe's
    logits, so any of them is sound. With none, the refusal names every
    candidate's reason.
    """
    model_type = hf_config.get('model_type')
    if not isinstance(model_type, str):
        raise ValueError("config.json states no model_type, so neither a family nor a reference "
                         "class can be chosen for it")
    reasons: dict[str, str] = {}
    candidates = []
    for convention, family in CONVENTIONS:
        try:
            _translate(hf_config, convention, family)
            candidates.append((convention, family))
        except _Mismatch as error:
            reasons[convention] = str(error)
    if not candidates:
        raise _refuse(model_type, f"its config reads as no registered family's: {_listed(reasons)}")
    try:
        fields = shrink(hf_config)
        layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    except KeyError as error:
        # A reader may take a config by its own names or its defaults (OPT's
        # defaults every size); the probe shrinks it by the shared names.
        read = ", ".join(convention for convention, _ in candidates)
        raise _refuse(model_type, f"its config reads as {read}'s but states no {error.args[0]!r}, "
                      "by which the probe shrinks it") from error
    with _quiet():
        config, size, name = _reference_config(model_type, fields)
        # A candidate that builds another architecture is dismissed on its
        # parameter count before the reference is allocated or run.
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
        if sized:
            import torch
            from transformers import AutoModelForCausalLM

            try:
                reference = AutoModelForCausalLM.from_config(config, trust_remote_code=False,
                                                             dtype=torch.float32)
                scatter_weights(reference)
                expected = reference_logits(reference, probe_ids(_VOCAB).astype(np.int64))
            except Exception as error:
                raise _refuse(model_type, f"{name} does not run its shrunken config "
                              f"({type(error).__name__}: {error})") from error
            with tempfile.TemporaryDirectory() as scratch:
                reference.save_pretrained(scratch)
                for convention, family, model, record, inert in sized:
                    try:
                        error, bound = _agreement(convention, family, model, record, inert, name, expected,
                                                  Path(scratch), layers)
                    except _Mismatch as mismatch:
                        reasons[convention] = str(mismatch)
                        continue
                    return VerifiedMapping(model_type, convention, name, error, bound, tuple(sorted(inert)))
    tried = {convention: reasons[convention] for convention, _ in candidates}
    raise _refuse(model_type, f"no family whose config reading it takes computes {name}'s model: "
                  f"{_listed(tried)}")


def _listed(reasons: Mapping[str, str]) -> str:
    return "; ".join(f"{convention}: {reason}" for convention, reason in reasons.items())
