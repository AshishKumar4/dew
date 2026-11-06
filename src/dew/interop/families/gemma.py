"""Translate the Gemma decoders: gemma, gemma2, gemma3_text, gemma3n_text, gemma4_text.

Each release keeps the previous block and adds one thing: Gemma 2 the
sandwich norms and the two softcaps, Gemma 3 the q/k norms and the
alternating local window, Gemma 3n AltUp's residual copies with the LAuReL
projections and the per-layer inputs, Gemma 4 the value norm, the shared KV
layers and the routed branch. The export side is here too, because a Gemma
config is written back from the computation rather than from a template.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from dew import _records
from dew._model_types import LAYER_SCALAR_COLLECTIONS
from dew.interop.config_records import native_fields
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, flag, inert, integer, rope_part
from dew.interop.decoder_parts import (
    DECODER_ROUTES,
    FUSED_EXPERTS,
    GEMMA3_MODEL_TYPE,
    MOE_SHARED,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    Leaves,
    RopeSpelling,
    both,
    config_flag,
    fixed_fields,
    in_layer,
    in_trunk,
    kinds_of,
    own_routes,
    read_rope_theta,
    refuse,
    refuse_export,
    specified_layer_types,
)
from dew.interop.layouts import (
    Condition,
    Config,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Nested,
    Route,
    Table,
    constant,
    one_of,
    unread,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.gemma3n import AltUp
from dew.nn.mixers import AttentionMixer
from dew.registry import Configured


def _trailing_sharers(hf_config: Mapping[str, object], layers: int, default: int) -> tuple[int, ...] | None:
    """`num_kv_shared_layers` as the indices of the trailing layers that share
    their providers' keys and values (`first_kv_shared_layer_idx`,
    modeling_gemma3n.py:1178, modeling_gemma4.py:1068), None for none."""
    count = _records.integer(hf_config.get("num_kv_shared_layers", default), "num_kv_shared_layers")
    if not 0 <= count < layers:
        refuse(f"num_kv_shared_layers {count}", f"it has to leave a provider among {layers} layers")
    return tuple(range(layers - count, layers)) or None


_SHARED_UNUSED = ('k_proj', 'v_proj', 'k_norm')
"""A kv-shared layer's own key and value projections and key norm, which transformers builds and a
release may store (google/gemma-4-E4B-it @ee0ef60 does, layers 24-41) but never computes with: the
layer reads its provider's keys and values (modeling_gemma4.py, Gemma4TextAttention.forward)."""


def _kv_shared(config: Config, slots: Mapping[str, str]) -> bool:
    """Whether a route's `layer` is a kv-shared one (`kv_shared_layers`)."""
    sharing = config.get('kv_shared_layers') or ()
    return isinstance(sharing, (tuple, list)) and int(slots['layer']) in sharing


def _shared_unused() -> Route:
    """A kv-shared layer's unused keys, values and key norm (`_SHARED_UNUSED`), which a load leaves
    unplaced and a save writes back as stored."""
    return in_layer('self_attn.{projection}.weight', None, {'projection': one_of(*_SHARED_UNUSED)},
                    _kv_shared)


def _gemma_layer_types(hf_config: Mapping[str, object], used: set[str], *,
                       last_full: bool = False) -> tuple[str, ...]:
    period = 6
    if not last_full and hf_config.get('layer_types') is None:
        used.add('sliding_window_pattern')
        period = _records.integer(hf_config.get("sliding_window_pattern", 6), "sliding_window_pattern")
    types = specified_layer_types(hf_config, used,
                                  ('sliding_attention',) * (period - 1) + ('full_attention',))
    # Gemma4TextConfig rewrites the final layer before building the model.
    return (*types[:-1], 'full_attention') if last_full and types else types


GEMMA4_ROPE = {'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
               'full_attention': {'rope_type': 'proportional', 'partial_rotary_factor': 0.25,
                                  'rope_theta': 1000000.0}}
"""The records Gemma4TextConfig and DiffusionGemmaTextConfig install when
rope_parameters is absent or null (configuration_gemma4.py:203-208)."""

GEMMA4_DEFAULTS = {'hidden_activation': 'gelu_pytorch_tanh', 'head_dim': 256, 'num_key_value_heads': 4,
                   'sliding_window': 512, 'pad_token_id': 0}
"""The class defaults Gemma 4's text configs share where Dew's readers would
otherwise fall back to another value (configuration_gemma4.py:152-183)."""


def _gemma4_rope(entries: Mapping[str, object]) -> tuple[float, float | None, float | None]:
    """Return (rope_theta, rope_local_theta, partial_rotary_factor) for gemma4.

    The full layers may rotate a fraction of their head dims (proportional
    partial rotary); the sliding layers rotate all of theirs. Anything but
    those two shapes refuses with the entry named.
    """
    full = _records.record(entries.get('full_attention') or {}, 'rope_parameters.full_attention')
    sliding = _records.record(entries.get('sliding_attention') or {}, 'rope_parameters.sliding_attention')
    factor = sliding.get('partial_rotary_factor')
    if factor not in (None, 1, 1.0):
        refuse(f"rope_parameters.sliding_attention partial_rotary_factor {factor}",
                "partial rotary applies to the full layers only")
    read_rope_theta({**sliding, 'rope_type': sliding.get('rope_type', sliding.get('type', 'default'))},
                "rope_parameters.sliding_attention")
    local = sliding.get('rope_theta', 10000.0)
    rope_type = full.get('rope_type', full.get('type', 'default'))
    factor = full.get('partial_rotary_factor')
    if rope_type == 'proportional':
        if factor is None:
            refuse("rope_parameters.full_attention",
                    "proportional rope needs its partial_rotary_factor")
        extra = sorted(set(full) - {'rope_type', 'type', 'rope_theta',
                                    'partial_rotary_factor', 'factor'})
        if extra or full.get('factor', 1.0) not in (1, 1.0):
            refuse("rope_parameters.full_attention scaling",
                    "the backbone applies plain rotary positions at rope_theta")
        partial = _records.number(factor, 'rope_parameters.full_attention partial_rotary_factor')
        theta = _records.number(
            full.get("rope_theta", 1000000.0), "rope_parameters.full_attention rope_theta"
        )
    elif rope_type in ('default', 'none'):
        if factor not in (None, 1, 1.0):
            refuse("rope_parameters.full_attention partial_rotary_factor",
                    "partial rotary comes spelled proportional")
        partial = None
        theta = read_rope_theta(full, 'rope_parameters.full_attention') or 10000.0
    else:
        refuse(f"rope_parameters.full_attention (rope_type {rope_type!r})",
                "the backbone applies plain rotary positions at rope_theta")
    local = _records.number(local, 'rope_parameters.sliding_attention rope_theta')

    return theta, (None if local == theta else local), partial


def _number(value: Configured, key: str) -> float | None:
    return None if value is None else _records.number(value, key)


_GEMMA_DECODER = DecoderSpelling(reads=frozenset({'attention_bias'}), scale_after_cast=False,
                                 tie_embeddings=True)
"""Gemma 1's decoder: (1 + w) norms scaled in fp32, a tied head (modeling_gemma.py:77, :374)."""

GEMMA_CONFIG = ConfigLayout(decoder_layout(_GEMMA_DECODER), constant('scale_offset', value=True),
                            constant('embedding_scale', value=True))
"""Gemma 1's config, both ways: sqrt(d)-scaled embeddings, and no norms beyond the two pre-norms. Its
released config names hidden_act 'gelu', which the reference computes as the erf gelu
(modeling_gemma.py:93, ACT2FN['gelu'])."""

def _inverse_root(value: Configured, key: str) -> float | None:
    return None if value is None else _records.number(value, key) ** -0.5


_QUERY_SCALE = ConfigField('query_pre_attn_scalar', 'attention_scale', None, parse=_inverse_root,
                           spell=lambda scale: None if scale is None
                           else round(1.0 / _records.number(scale, 'attention_scale') ** 2))
"""The query's scale, spelled as the scalar whose inverse square root it is."""

_SANDWICH = (constant('scale_offset', value=True), constant('embedding_scale', value=True),
             constant('sandwich_norms', value=True), _QUERY_SCALE,
             ConfigField('final_logit_softcapping', 'final_logit_softcap', None, parse=_number))
"""Gemma 2's and 3's block beyond Gemma 1's: the sandwich norms, the query's scalar and the head's softcap,
both ways."""


def _sliding_pattern() -> ConfigPart[CausalTransformer]:
    """Gemma 3's pattern where a config states no layer_types: a full layer every
    `sliding_window_pattern` (6 where unstated), sliding layers between; written as the layers' kinds."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        if hf_config.get('layer_types') is None:
            used.add('sliding_window_pattern')
            period = _records.integer(hf_config.get('sliding_window_pattern', 6), 'sliding_window_pattern')
            record['layer_types'] = ('sliding_attention',) * (period - 1) + ('full_attention',)
        return record
    return ConfigPart(read, lambda model: {})


_GEMMA3_ROPE = RopeSpelling(theta=1000000.0, per_kind=True, local_theta=10000.0,
                            local_field='rope_local_base_freq')
"""Gemma3TextConfig's and Gemma3nTextConfig's rope: a record per layer type,
the full layers at 1e6 and the sliding ones at rope_local_base_freq or 1e4
(configuration_gemma3.py:102, 136-163; configuration_gemma3n.py:120, 179-206)."""

GEMMA2_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(pattern=('sliding_attention', 'full_attention'), scale_after_cast=False,
                                   tie_embeddings=True, activation='hidden_activation'),
                   # Gemma2RotaryEmbedding rotates every layer at one base and ramp.
                   rope=rope_part(PLAIN_SPELLING, per_kind=False)),
    *_SANDWICH, ConfigField('attn_logit_softcapping', 'attn_logit_softcap', None, parse=_number))
"""Gemma 2's config, both ways: Gemma 3's block without the q/k norms, alternating sliding and full layers
at one rope base and softcapping the attention logits with tanh (configuration_gemma2.py:95-98,
modeling_gemma2.py:203-206)."""

GEMMA3_CONFIG = ConfigLayout(
    _sliding_pattern(),
    decoder_layout(DecoderSpelling(rope=_GEMMA3_ROPE, scale_after_cast=False, tie_embeddings=True,
                                   qk_norm=True, activation='hidden_activation')),
    # Gemma 3 reads attn_logit_softcapping into its attention without passing it on
    # (modeling_gemma3.py:334, :370-379), so it changes nothing.
    *_SANDWICH, unread('attn_logit_softcapping'))
"""Gemma 3's config, both ways."""


def _sharers(default: int) -> ConfigPart[CausalTransformer]:
    """The trailing layers that share their providers' keys and values, both ways (`_trailing_sharers`,
    `default` of them where the config counts none), refused where they are no trailing run."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add('num_kv_shared_layers')
        layers = _records.integer(record['num_layers'], 'num_layers')
        record['kv_shared_layers'] = _trailing_sharers(hf_config, layers, default)
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        sharing = model.sharing_layers
        if sharing != tuple(range(model.num_layers - len(sharing), model.num_layers)):
            refuse_export('kv_shared_layers', 'Gemma can express only a trailing run of shared-KV layers')
        return {'num_kv_shared_layers': len(sharing)}
    return ConfigPart(read, write)


def _sparsity() -> ConfigPart[CausalTransformer]:
    """Gemma 3n's gaussian top-k sparsity, one fraction a layer, both ways; the reference's default is the
    first ten layers at 0.95 when there are more than ten, else none (configuration_gemma3n.py)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add('activation_sparsity_pattern')
        layers = _records.integer(record['num_layers'], 'num_layers')
        sparsity = hf_config.get('activation_sparsity_pattern')
        if sparsity is None:
            sparse = 10 if layers > 10 else 0
            sparsity = [0.95] * sparse + [0.0] * (layers - sparse)
        if not isinstance(sparsity, (list, tuple)) or len(sparsity) != layers:
            refuse(f"activation_sparsity_pattern {sparsity!r}",
                   f"the reference takes one fraction per layer of {layers}")
        record['activation_sparsity_pattern'] = tuple(_records.number(fraction, 'activation_sparsity_pattern')
                                                      for fraction in sparsity)
        return record
    return ConfigPart(read, lambda model: {'activation_sparsity_pattern':
                                           list(model.activation_sparsity_pattern or ())})


_ALTUP: ConfigLayout[AltUp] = ConfigLayout(
    integer('altup_num_inputs', 'num_inputs', 4), integer('altup_active_idx', 'active_idx', 0),
    ConfigField('altup_coef_clip', 'coef_clip', 120.0, parse=_number),
    flag('altup_correct_scale', 'correct_scale', default=True))
"""AltUp's residual copies, both ways, at Gemma3nTextConfig's defaults."""

GEMMA3N_CONFIG = ConfigLayout(
    decoder_layout(DecoderSpelling(rope=_GEMMA3_ROPE,
                                   pattern=('sliding_attention',) * 4 + ('full_attention',),
                                   scale_after_cast=False, tie_embeddings=True, qk_norm=True,
                                   activation='hidden_activation')),
    _sparsity(), integer('laurel_rank', 'laurel_rank', 64),
    Nested('altup', AltUp, _ALTUP, lambda model: model.altup),
    integer('hidden_size_per_layer_input', 'per_layer_input_dim', 256),
    integer('vocab_size_per_layer_input', 'per_layer_input_vocab', 262144), _sharers(15),
    ConfigField('final_logit_softcapping', 'final_logit_softcap', 30.0, parse=_number),
    constant('sandwich_norms', value=True), constant('embedding_scale', value=True),
    constant('attention_scale', value=1.0), constant('v_norm', value=True),
    # PreTrainedConfig serializes this field; Gemma3nTextMLP never reads it.
    unread('chunk_size_feed_forward'))
"""Gemma 3n's config, both ways (E2B, E4B): AltUp's stack of residual copies, the LAuReL block, gaussian
top-k sparsity on the first layers, one feed-forward width per layer, per-layer inputs and KV sharing over
the last layers; every fifth layer full."""


def _gemma4_pattern() -> ConfigPart[CausalTransformer]:
    """Gemma 4's layers' kinds, both ways: `layer_types` (a full layer every six where unstated), the
    final layer rewritten to full attention as Gemma4TextConfig does before building the model, which takes
    precedence over an explicit pattern; written as the kinds, which end on a full layer."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        types = specified_layer_types(hf_config, used, ('sliding_attention',) * 5 + ('full_attention',))
        record['layer_types'] = (*types[:-1], 'full_attention') if types else types
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        types = model.per_layer_types
        if not types or set(types) - {'sliding_attention', 'full_attention'}:
            refuse_export('layer_types', 'Gemma4 has only sliding and full attention')
        if types[-1] != 'full_attention':
            refuse_export('layer_types', 'Gemma4TextConfig forces the final layer to full attention')
        return {'layer_types': list(types)}
    return ConfigPart(read, write)


def _gemma4_rope_part() -> ConfigPart[CausalTransformer]:
    """Gemma 4's rotary, both ways: a plain sliding record and a full one that turns whole heads or a
    proportional fraction of them (`_gemma4_rope`), Gemma4TextConfig's records where rope_parameters is
    absent or null (`GEMMA4_ROPE`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update(('rope_parameters', 'rope_theta'))
        stated = hf_config.get('rope_parameters')
        entries = GEMMA4_ROPE if stated is None else _records.record(stated, 'rope_parameters')
        theta, local, partial = _gemma4_rope(entries)
        record['rope_theta'] = theta
        record['partial_rotary_factor'] = partial
        kinds = kinds_of(record)
        if 'sliding_attention' in kinds:
            if local is not None:
                kinds['sliding_attention']['rope_theta'] = local
            if partial is not None:
                # Gemma 4's sliding layers turn whole heads; its partial factor is the full layers' alone.
                kinds['sliding_attention']['partial_rotary_factor'] = 1.0
        record['kinds'] = kinds
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        if model.partial_rotary_factor is not None and model.partial_rotary_type != 'proportional':
            refuse_export('partial_rotary_type', 'Gemma4 full layers use proportional rotary')
        for name in set(model.per_layer_types):
            if model.kind_of(name).rope_scaling is not None:
                refuse_export(f'kinds.{name}.rope', 'Gemma4 uses plain local and proportional global rotary')
        full = model.kind_of('full_attention')
        local = model.kind_of('sliding_attention') if 'sliding_attention' in model.per_layer_types else full
        return {'rope_parameters': {
            'sliding_attention': {'rope_type': 'default', 'rope_theta': local.rope_theta},
            'full_attention': {'rope_type': 'proportional', 'rope_theta': full.rope_theta,
                               'partial_rotary_factor': model.partial_rotary_factor or 1.0}}}
    return ConfigPart(read, write)


def _gemma4_geometry(*, k_eq_v: bool) -> ConfigPart[CausalTransformer]:
    """The full layers' head width and key/value heads, both ways. Every layer reads its geometry from
    per_layer_config (modeling_gemma4.py, Gemma4TextAttention reads layer_config), whose entries the sliding
    layers leave at the model's head_dim and num_key_value_heads. Where the config carries no
    per_layer_config key at all, configuration_gemma4.py builds the full layers' entries from
    global_head_dim (512 unless named) and, under attention_k_eq_v alone, num_global_key_value_heads; a
    config that carries the key, null or filled, leaves those two fields unread. The released E2B is one,
    its global q_proj 8 heads of 256 against the 512 it names. A family that always reads values off keys
    on full layers (`k_eq_v`, DiffusionGemma's) declares it rather than its config."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update(('attention_k_eq_v', 'per_layer_config', 'global_head_dim', 'num_global_key_value_heads'))
        sliding_dim = _records.integer(record['head_dim'], 'head_dim')
        kv_heads = _records.integer(record['num_kv_heads'], 'num_key_value_heads')
        tied = k_eq_v or config_flag(hf_config, 'attention_k_eq_v', default=False)
        full_dim, full_kv = sliding_dim, kv_heads
        if 'per_layer_config' in hf_config:
            entries = hf_config['per_layer_config'] or {}
            per_layer = (entries.values() if isinstance(entries, Mapping)
                         else entries if isinstance(entries, (list, tuple))
                         else refuse(f"per_layer_config={entries!r}",
                                     "the per-layer overrides are records by layer"))
            for entry in per_layer:
                if not isinstance(entry, Mapping):
                    continue
                if entry.get('head_dim') is not None:
                    full_dim = _records.integer(entry['head_dim'], 'per_layer_config head_dim')
                if entry.get('num_key_value_heads') is not None:
                    stated = _records.integer(entry['num_key_value_heads'],
                                              'per_layer_config num_key_value_heads')
                    if full_kv != kv_heads and stated != full_kv:
                        refuse("per_layer_config num_key_value_heads",
                               f"the full layers name both {full_kv} and {stated}")
                    full_kv = stated
        else:
            full_dim = _records.integer(hf_config.get('global_head_dim', 512), 'global_head_dim')
            global_kv = hf_config.get('num_global_key_value_heads')
            if global_kv is not None and tied:
                full_kv = _records.integer(global_kv, 'num_global_key_value_heads')
        kinds = kinds_of(record)
        if 'full_attention' in _records.strings(record['layer_types'], 'layer_types'):
            full = kinds.setdefault('full_attention', native_fields(LayerKind)())
            if full_dim != sliding_dim:
                full['head_dim'] = full_dim
            if full_kv != kv_heads:
                full['num_kv_heads'] = full_kv
        record['kinds'] = kinds
        record['attention_k_eq_v'] = tied
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        types = model.per_layer_types
        full = model.kind_of('full_attention')
        local = model.kind_of('sliding_attention') if 'sliding_attention' in types else full
        if full.window is not None:
            refuse_export('kinds.full_attention.window', 'full attention is unwindowed')
        if 'sliding_attention' in types and (local.window is None or local.window < 1):
            refuse_export('kinds.sliding_attention.window', 'sliding attention needs a positive window')
        return {'head_dim': local.head_dim, 'num_key_value_heads': local.num_kv_heads,
                'global_head_dim': full.head_dim, 'num_global_key_value_heads': full.num_kv_heads,
                # Explicit overrides keep configuration_gemma4.py:210-223 from replacing the full
                # geometry with global defaults or gating KV heads on K=V.
                'per_layer_config': {str(index): {'head_dim': full.head_dim,
                                                  'num_key_value_heads': full.num_kv_heads}
                                     for index, name in enumerate(types) if name == 'full_attention'},
                **({} if k_eq_v else {'attention_k_eq_v': model.attention_k_eq_v})}
    return ConfigPart(read, write)


def _per_layer_inputs() -> ConfigPart[CausalTransformer]:
    """Gemma 4's per-layer inputs, both ways: their width (none at 0) and their vocabulary (the model's
    where unstated)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update(('hidden_size_per_layer_input', 'vocab_size_per_layer_input'))
        width = _records.integer(hf_config.get('hidden_size_per_layer_input', 0),
                                 'hidden_size_per_layer_input')
        record['per_layer_input_dim'] = width or None
        record['per_layer_input_vocab'] = _records.integer(
            hf_config.get('vocab_size_per_layer_input', record['vocab_size']), 'vocab_size_per_layer_input')
        return record
    return ConfigPart(read, lambda model: {
        'hidden_size_per_layer_input': model.per_layer_input_dim or 0,
        'vocab_size_per_layer_input': model.per_layer_input_vocab or model.vocab_size})


def _double_wide() -> ConfigPart[CausalTransformer]:
    """The widened MLP of Gemma 4's sharing layers, both ways: the reference widens only sharing layers,
    so with none this flag changes neither the weights nor the forward operation."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add('use_double_wide_mlp')
        record['use_double_wide_mlp'] = (config_flag(hf_config, 'use_double_wide_mlp', default=False)
                                         and record.get('kv_shared_layers') is not None)
        return record
    return ConfigPart(read, lambda model: {'use_double_wide_mlp': model.use_double_wide_mlp})


def _read_moe_flags(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    used.update(('enable_moe_block', 'expert_intermediate_size', 'chunk_size_feed_forward'))
    return record


def _parallel_experts(*, flagged: bool) -> ConfigLayout[CausalTransformer]:
    """The routed branch a Gemma 4 layer runs beside its dense MLP, both ways: on every layer where the
    config enables it (`enable_moe_block`, `flagged`) or, for a family that names no such flag
    (DiffusionGemma's), where it carries the three routed widths."""
    keys = ('num_experts', 'top_k_experts', 'moe_intermediate_size')

    def present(hf_config: Mapping[str, object]) -> bool:
        if flagged and hf_config.get('enable_moe_block'):
            for key in keys:
                if hf_config.get(key) is None:
                    refuse("enable_moe_block=True", f"the routed branch needs {key}")
            return True
        return not flagged and all(hf_config.get(key) is not None for key in keys)

    def write(model: CausalTransformer) -> Mapping[str, object]:
        mixture = model.mixture
        if mixture is not None and (not mixture.parallel
                                    or tuple(model.sparse_layers) != tuple(range(model.num_layers))):
            refuse_export('mixture', 'Gemma4 routes a parallel expert branch on every layer')
        return {'enable_moe_block': mixture is not None} if flagged else {}
    return ConfigLayout(
        Nested('mixture', Mixture, ConfigLayout(
            integer('num_experts', 'experts'), integer('top_k_experts', 'top_k'),
            integer('moe_intermediate_size', 'expert_features'), constant('parallel', value=True)),
            lambda model: model.mixture, present=present),
        ConfigPart(_read_moe_flags, write))


def _gemma4_held(model: CausalTransformer) -> Mapping[str, object]:
    """What a Gemma 4 export holds of `model` beyond its config: how it runs, which the config does not
    carry, and what the standalone reference has no computation for (its mixers' multimodal metadata
    needs its source wrapper)."""
    fixed_fields(model, {'scale_after_cast': False}, 'Gemma4 computes {0!r} for this field')
    for name in ('output_gate', 'attention_sinks', 'attn_logit_softcap', 'altup',
                 'laurel_rank', 'activation_sparsity_pattern', 'num_nextn_predict_layers', 'dropout_rate'):
        if getattr(model, name):
            refuse_export(name, 'the standalone Gemma4 reference has no such computation')
    if model.layer_scalar not in ('frozen', 'trainable'):
        refuse_export('layer_scalar', 'Gemma4 exports an explicit frozen or trainable scalar value')
    for name in set(model.per_layer_types):
        mixer = model.kind_of(name).mixer or model.mixer
        if mixer is not None and not isinstance(mixer, AttentionMixer):
            refuse_export(f'kinds.{name}.mixer', 'Gemma4 uses ordinary attention')
        if isinstance(mixer, AttentionMixer) and (
                mixer.bidirectional_images or mixer.bidirectional_blocks or mixer.mrope_section is not None):
            refuse_export(f'kinds.{name}.mixer', 'multimodal attention metadata needs its source wrapper')
    return {}


def gemma4_layout(*, k_eq_v: bool = False, flagged: bool = True, softcap: bool = True
                  ) -> ConfigLayout[CausalTransformer]:
    """Gemma 4's text decoder, both ways: Gemma 3's block with the value norm, layer scalars, KV sharing over
    its trailing layers, per-layer inputs and a parallel routed branch (`_parallel_experts`), its full layers'
    geometry (`_gemma4_geometry`) and proportional rotary (`_gemma4_rope_part`), and its head's softcap where
    the config states one (`softcap`; a family whose head always caps declares it). attention_logit_cap
    changes nothing on the text path: Gemma4TextAttention never passes it to its attention call
    (modeling_gemma4.py, Gemma4TextAttention.forward); only the audio attention applies one."""
    return ConfigLayout(
        _gemma4_pattern(),
        decoder_layout(DecoderSpelling(reads=frozenset({'sliding_window', 'attention_bias'}),
                                       scale_after_cast=False, tie_embeddings=True, qk_norm=True,
                                       activation='hidden_activation'),
                       rope=_gemma4_rope_part()),
        _gemma4_geometry(k_eq_v=k_eq_v), _per_layer_inputs(), _sharers(0), _double_wide(),
        *((ConfigField('final_logit_softcapping', 'final_logit_softcap', None, parse=_number),) if softcap
          else (unread('final_logit_softcapping'),)),
        _parallel_experts(flagged=flagged), unread('attention_logit_cap'),
        constant('sandwich_norms', value=True), constant('embedding_scale', value=True),
        constant('attention_scale', value=1.0), constant('v_norm', value=True),
        # Every released Gemma 4 checkpoint carries the layer_scalar buffer the reference initialises to
        # one, so the tree always holds it.
        constant('layer_scalar', value='frozen'),
        ConfigPart(lambda hf_config, used, record: record, _gemma4_held))


# Gemma 4's routed branch, named for what each norm normalises, as the
# block's own sandwich norms are.
_GEMMA4_MOE: Leaves = {
    'router.proj.weight': ('moe', 'router', 'proj', 'kernel'),
    'router.scale': ('moe', 'router', 'scale'),
    'router.per_expert_scale': ('moe', 'router', 'per_expert_scale'),
    'post_feedforward_layernorm_1.weight': ('moe', 'mlp_branch_norm', 'scale'),
    'pre_feedforward_layernorm_2.weight': ('moe', 'experts_input_norm', 'scale'),
    'post_feedforward_layernorm_2.weight': ('moe', 'experts_output_norm', 'scale'),
}


def _scalar_mode(collection: str) -> Condition:
    """A layer scalar's route held where the record keeps the scalars in `collection` (`layer_scalar`),
    refused where it names no mode."""
    def held(config: Config, slots: Mapping[str, str]) -> bool:
        mode = config.get('layer_scalar')
        if not isinstance(mode, str) or mode not in LAYER_SCALAR_COLLECTIONS:
            return refuse('layer_scalar', 'the source scalar requires a frozen or trainable model mode')
        return LAYER_SCALAR_COLLECTIONS[mode] == collection
    return held


GEMMA4_ROUTES = Table(
    _shared_unused(),
    # Each layer scalar in the collection its mode names (`LAYER_SCALAR_COLLECTIONS`).
    *(Route('model.layers.{layer}.layer_scalar', f'{collection}/layers_{{layer}}/layer_scalar',
            when=both(in_trunk, _scalar_mode(collection)))
      for collection in LAYER_SCALAR_COLLECTIONS.values()),
    *own_routes(_GEMMA4_MOE),
    # Each stacked expert kernel under the name its fused tensor is packed from (`FUSED_EXPERTS`).
    in_layer('experts.{projection}', 'moe/experts/{projection}/kernel', {'projection': one_of(*MOE_SHARED)}),
    DECODER_ROUTES)
"""Gemma 4's names: its routed branch's (`_GEMMA4_MOE`) and its layer scalars' beside the shared decoder's;
a kv-shared layer's unused keys, values and key norm land nowhere."""

GEMMA3N_ROUTES = Table(
    _shared_unused(),
    # The copies' projections are indexed modules (altup_projections.{i}), which land as
    # altup_projections_{i} the way the layers do; the coefficient maps are Linears, so they transpose
    # like any kernel.
    Route('model.{module}.{index}.weight', 'params/{module}_{index}/kernel',
          MappingProxyType({'module': one_of('altup_projections', 'altup_unembed_projections')})),
    in_layer('{module}.correct_output_scale', '{module}/correct_output_scale',
             {'module': one_of('altup', 'laurel')}),
    in_layer('{module}.{norm}.weight', '{module}/{norm}/scale',
             {'module': one_of('altup', 'laurel'), 'norm': one_of('router_norm', 'post_laurel_norm')}),
    in_layer('{module}.{linear}.weight', '{module}/{linear}/kernel',
             {'module': one_of('altup', 'laurel'), 'linear': one_of(
                 'correction_coefs', 'prediction_coefs', 'modality_router', 'linear_left', 'linear_right')}),
    DECODER_ROUTES)
"""Gemma 3n's names: the AltUp and LAuReL leaves beside a Gemma 4 style layer; a kv-shared layer's unused
keys, values and key norm land nowhere."""




GEMMA3N_TEXT = DecoderFamily(
    ('gemma3n_text',),
    GEMMA3N_CONFIG,
    lambda fields: fields.altup is not None,
    'gemma3n_text',
    'Gemma3nForCausalLM',
    routes=GEMMA3N_ROUTES,
    preserve_source_layout=True,
    # Gemma3nTextConfig's window, which an omitted sliding_window means.
    defaults={'max_position_embeddings': 32768, 'hidden_activation': 'gelu_pytorch_tanh',
              'final_logit_softcapping': 30.0, 'pad_token_id': 0,
              'sliding_window': 512},
    padding_idx=True,
)

GEMMA4_TEXT = DecoderFamily(
    ('gemma4_text',),
    gemma4_layout(),
    lambda fields: bool(
        fields.v_norm or fields.per_layer_input_dim or fields.kv_shared_layers
    ),
    'gemma4_text',
    'Gemma4ForCausalLM',
    routes=GEMMA4_ROUTES,
    packed=FUSED_EXPERTS,
    preserve_source_layout=True,
    defaults={**GEMMA4_DEFAULTS, 'hidden_size_per_layer_input': 256,
              'vocab_size_per_layer_input': 262144},
    padding_idx=True,
)

GEMMA4_UNIFIED_TEXT = dataclasses.replace(
    GEMMA4_TEXT, model_types=('gemma4_unified_text',), export_model_type='gemma4_unified_text',
    # Only its causal text path with optional vision blocks is mapped: the unified decoder has no MoE or
    # per-layer input branch.
    config=ConfigLayout(inert('is_causal', 'only its causal text path is mapped', neutral=True),
                        inert('enable_moe_block', 'the unified decoder has no MoE branch', neutral=False),
                        inert('hidden_size_per_layer_input', 'the unified decoder has no per-layer input',
                              neutral=0),
                        gemma4_layout()),
    architecture='Gemma4UnifiedForCausalLM',
    defaults={**GEMMA4_DEFAULTS, 'sliding_window': 1024, 'max_position_embeddings': 262144,
              'use_bidirectional_attention': 'vision'})


GEMMA3_TEXT = DecoderFamily(
    (GEMMA3_MODEL_TYPE,),
    GEMMA3_CONFIG,
    lambda fields: bool(fields.sandwich_norms and fields.qk_norm),
    GEMMA3_MODEL_TYPE,
    'Gemma3ForCausalLM',
    preserve_source_layout=False,
    # Gemma3TextConfig's defaults for the fields its decoder reads.
    defaults={'hidden_activation': 'gelu_pytorch_tanh', 'query_pre_attn_scalar': 256, 'pad_token_id': 0,
              'sliding_window': 4096, 'head_dim': 256, 'max_position_embeddings': 131072},
    padding_idx=True,
)

GEMMA2 = DecoderFamily(
    ('gemma2',),
    GEMMA2_CONFIG,
    lambda fields: bool(fields.sandwich_norms),
    'gemma2',
    'Gemma2ForCausalLM',
    preserve_source_layout=False,
    # Gemma2Config's defaults for the fields its decoder reads.
    defaults={'hidden_activation': 'gelu_pytorch_tanh', 'query_pre_attn_scalar': 256, 'pad_token_id': 0,
              'final_logit_softcapping': 30.0, 'attn_logit_softcapping': 50.0, 'sliding_window': 4096,
              'head_dim': 256, 'max_position_embeddings': 8192},
    padding_idx=True,
)

GEMMA = DecoderFamily(
    ('gemma',),
    GEMMA_CONFIG,
    lambda fields: bool(fields.embedding_scale),
    'gemma',
    'GemmaForCausalLM',
    preserve_source_layout=False,
    defaults={'max_position_embeddings': 8192, 'hidden_act': 'gelu_pytorch_tanh', 'pad_token_id': 0},
    padding_idx=True,
)
