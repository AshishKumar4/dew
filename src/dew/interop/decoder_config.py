"""The config every decoder family's layout composes, both ways (`dew.interop.layouts.ConfigLayout`): the
shared decoder's geometry under transformers' names (`DECODER_FIELDS`) and as each family's class spells it
(`DecoderSpelling`, `decoder_layout`), its layers' kinds and rotary, the parts a family declares beside it
(a classic MLP's activation, the routed experts, a mixer a kind holds, keys its reference reads to no effect
or computes at one value alone), and the layout of a family read and never written (`read_only`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dew import _records
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.decoder_parts import (
    ACTIVATIONS,
    PLAIN_SPELLING,
    DecoderFields,
    ExportRefused,
    Ropes,
    RopeSpelling,
    config_flag,
    context_lengths,
    decoder_kinds,
    hf_activation,
    hf_ungated,
    kinds_of,
    plain_partial_rope,
    read_partial_rope,
    read_rope,
    refuse,
    specified_layer_types,
    ungated_mlp,
)
from dew.interop.layouts import (
    REQUIRED,
    ConfigEntry,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Nested,
    Required,
    Restated,
    constant,
    unread,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.registry import Configured


def ungated(key: str, accepted: tuple[str, ...], family: str, *, default: str = 'gelu_new',
            tanh: str = 'gelu_new', spelled: Mapping[str, str] | None = None
            ) -> ConfigPart[CausalTransformer]:
    """A classic family's ungated MLP activation under `key`, both ways (`ungated_mlp`, `hf_ungated`): the
    names its reference reads (`accepted`, `default` where unstated), each an own spelling of transformers'
    where `spelled` maps them (GPT-1's `afn` 'gelu' is OpenAI's tanh GELU), its tanh GELU written as
    `tanh`."""
    spelled = spelled or {}
    back = {ours: own for own, ours in spelled.items()}

    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add(key)
        stated = _records.text(hf_config.get(key, default) or default, key)
        if spelled and stated not in spelled:
            refuse(f'{key}={stated!r}', f"{family}'s MLP reads only {sorted(spelled)}")
        record['mlp'] = ungated_mlp(spelled.get(stated, stated), accepted, key, family)
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        name = hf_ungated(model.mlp, family, tanh=tanh)
        return {key: back.get(name, name) if spelled else name}
    return ConfigPart(read, write)


def _mlp_features(hf_config: Mapping[str, object]) -> int | tuple[int, ...]:
    """Return one feed-forward width, or Gemma 3n's list of one per layer.

    configuration_gemma3n.py expands an int to a list, so a config it wrote
    carries the list. A list of one repeated value is that value.
    """
    stated = hf_config['intermediate_size']
    if isinstance(stated, (list, tuple)):
        widths = _records.integers(stated, 'intermediate_size')
        return widths[0] if len(set(widths)) == 1 else widths
    return _records.integer(stated, 'intermediate_size')


_OPTIONAL_FIELDS = frozenset({'layer_types', 'sliding_window', 'rope_local_base_freq', 'attention_bias',
                              'mlp_bias'})
"""Fields the shared decoder reads for a family whose reference reads them (`DecoderSpelling.reads`),
and accepts unread at a value that computes nothing. rope_local_base_freq is read by the families whose
class reads it, through their `RopeSpelling`."""


def _neutral(hf_config: Mapping[str, object], field: str) -> bool:
    """Whether the config's `field` computes what leaving it out computes."""
    value = hf_config[field]
    if field == 'layer_types' and isinstance(value, (list, tuple)):
        return all(layer == 'full_attention' for layer in value)
    return value is None or value is False


def _read_heads(hf_config: Mapping[str, object], used: set[str],
                record: DecoderFields) -> DecoderFields:
    """The key/value heads, the query heads' count where unstated, and each head's width, the residual's
    over the heads where unstated."""
    used.update(('num_key_value_heads', 'head_dim'))
    heads = _records.integer(record['num_heads'], 'num_heads')
    kv_heads = hf_config.get('num_key_value_heads')
    record['num_kv_heads'] = heads if kv_heads is None else _records.integer(kv_heads, 'num_key_value_heads')
    width = _records.integer(record['emb_features'], 'hidden_size')
    record['head_dim'] = _records.integer(hf_config.get('head_dim') or width // heads, 'head_dim')
    return record


def _read_activation(hf_config: Mapping[str, object], used: set[str], record: DecoderFields
                     ) -> DecoderFields:
    """The gated MLP's activation, `hidden_act` (or Gemma's `hidden_activation`) as transformers names it,
    refused outside the ones the gated MLP computes (`ACTIVATIONS`)."""
    used.update(('hidden_act', 'hidden_activation'))
    activation = _records.text(hf_config.get('hidden_act', hf_config.get('hidden_activation', 'silu')),
                               'hidden_act/hidden_activation')
    mapped = ACTIVATIONS.get(activation)
    if mapped is None:
        refuse(f"hidden_act {activation!r}", f"the gated MLP supports {sorted(ACTIVATIONS)}")
    record['mlp'] = mapped
    return record


def _read_context(hf_config: Mapping[str, object], used: set[str], record: DecoderFields
                  ) -> DecoderFields:
    used.add('max_position_embeddings')
    record['max_seq_len'], record['context_length'] = context_lengths(hf_config)
    return record


def _read_widths(hf_config: Mapping[str, object], used: set[str], record: DecoderFields
                 ) -> DecoderFields:
    used.add('intermediate_size')
    record['mlp_features'] = _mlp_features(hf_config)
    return record


def _widths_written(model: CausalTransformer) -> Mapping[str, object]:
    """The feed-forward's width, or one a layer as Gemma 3n's config writes them (`_mlp_features`)."""
    widths = model.mlp_features
    return {'intermediate_size': list(widths) if isinstance(widths, tuple) else model.hidden_features}


_GEOMETRY: tuple[ConfigEntry[CausalTransformer], ...] = (
    ConfigField('vocab_size', 'vocab_size', parse=_records.integer),
    ConfigField('hidden_size', 'emb_features', parse=_records.integer),
    ConfigField('num_hidden_layers', 'num_layers', parse=_records.integer),
    ConfigField('num_attention_heads', 'num_heads', parse=_records.integer),
    ConfigPart(_read_heads, lambda model: {'num_key_value_heads': model.kv_heads,
                                           'head_dim': model.features_per_head}),
    ConfigPart(_read_context,
               lambda model: {'max_position_embeddings': model.context_length or model.max_seq_len}),
    ConfigField('rms_norm_eps', 'norm_eps', 1e-6, parse=_records.number))

WIDTH: ConfigPart[CausalTransformer] = ConfigPart(_read_widths, _widths_written)
"""The feed-forward's width under transformers' shared `intermediate_size`, both ways."""

DECODER_FIELDS: ConfigLayout[CausalTransformer] = ConfigLayout(*_GEOMETRY, WIDTH)
"""The decoder's geometry every family's config states under transformers' shared names, both ways
(`decoder_layout`)."""


def activation_part(key: str = 'hidden_act') -> ConfigPart[CausalTransformer]:
    """The gated MLP's activation (`_read_activation`: `hidden_act`, else Gemma's `hidden_activation`),
    written under the key the family's class reads (`key`)."""
    return ConfigPart(_read_activation, lambda model: {key: hf_activation(model.mlp)})


def inert[Model](key: str, why: str, *, neutral: Configured = None) -> ConfigPart[Model]:
    """A key the reference reads to no effect at `neutral` (or absent, or null), refused at any other
    value, saying `why`; written as nothing."""
    def read(hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        used.add(key)
        if hf_config.get(key) not in (None, neutral):
            refuse(f"{key}={hf_config[key]!r}", why)
        return record
    return ConfigPart(read, lambda model: {}, keys=(key,))


def read_only(translate: Callable[[Mapping[str, object], set[str]], DecoderFields],
              why: str) -> ConfigLayout[CausalTransformer]:
    """The config of a family that is read and never written (`why`: a bundle read as its decoder, whose
    model is written as the bundle): its whole record `translate`'s, any write of it refused."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        return translate(hf_config, used)

    def write(model: CausalTransformer) -> Mapping[str, object]:
        raise ExportRefused(why)
    return ConfigLayout(ConfigPart(read, write))


def unwritten(why: str) -> ConfigPart[CausalTransformer]:
    """A family whose model is written as another (`why`: DiffusionGemma's text view, written as its
    whole bundle): read as nothing, any write of it refused."""
    def write(model: CausalTransformer) -> Mapping[str, object]:
        raise ExportRefused(why)
    return ConfigPart(lambda hf_config, used, record: record, write)


def expect[Model](key: str, why: str, *, value: Configured, written: bool = False,
                  required: bool = False) -> ConfigPart[Model]:
    """A key the family's reference computes only at `value`, refused at any other, saying `why`: its
    class's default where unstated, or, where the class defaults to another (`required`: K2 Horizon's
    per-head q/k norm), refused unless stated; written as `value` where readers take another where it is
    unstated (`written`: vLLM's DeepSeek router reads a softmax where no `scoring_func` is), else as
    nothing."""
    def read(hf_config: Mapping[str, object], used: set[str],
             record: NativeFields[Model]) -> NativeFields[Model]:
        used.add(key)
        if (required and key not in hf_config) or hf_config.get(key, value) != value:
            refuse(key, f'{why}; requires {value!r}')
        return record
    return ConfigPart(read, lambda model: {key: value} if written else {}, keys=(key,))


def remote_code(repo: str, module: str, name: str) -> ConfigPart[CausalTransformer]:
    """A family whose reference is its release's own code, which transformers has no class for: read as
    nothing, written as the `auto_map` the released configs name it by, `repo`'s `configuration_{module}`
    and `modeling_{module}` classes (`{name}Config`, `{name}Model`, `{name}ForCausalLM`)."""
    return ConfigPart(lambda hf_config, used, record: record, lambda model: {'auto_map': {
        'AutoConfig': f'{repo}--configuration_{module}.{name}Config',
        'AutoModel': f'{repo}--modeling_{module}.{name}Model',
        'AutoModelForCausalLM': f'{repo}--modeling_{module}.{name}ForCausalLM'}})


@dataclass(frozen=True)
class DecoderSpelling:
    """How a family's config class spells the decoder every family shares, both ways
    (`decoder_layout`): its rotary (`rope`), the optional fields its reference reads (`reads`, of
    `_OPTIONAL_FIELDS`; another stated at a value that computes nothing is accepted unread), the layer
    pattern it cycles where the config states none (`pattern`), its ramps (`ramps`, None for every one
    the rope utilities build), its tying where unstated (`tie_embeddings`), and what its reference fixes
    beside them: q/k head norms (`qk_norm`) and a norm's scale applied after its cast
    (`scale_after_cast`). `windowed` is a class whose every layer slides where its config states a
    window (MistralConfig's, MinistralConfig's without layer_types)."""

    rope: RopeSpelling = PLAIN_SPELLING
    reads: frozenset[str] = frozenset({'layer_types', 'sliding_window', 'attention_bias'})
    pattern: tuple[str, ...] | None = None
    windowed: bool = False
    keys: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    """The family's own keys for the shared geometry's (`Restated`: Qwen3.5-MoE's dense width is its
    experts', `moe_intermediate_size`)."""
    activation: str | None = 'hidden_act'
    """The key the class reads its gated activation under (Gemma 2's and later's `hidden_activation`);
    None for a family with an ungated MLP, which declares its activation itself (`ungated`)."""
    fixed: Mapping[str, object] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    """Shared geometry the class fixes and states nowhere (`Restated`)."""
    defaults: Mapping[str, object] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    """The shared geometry's defaults where the class's differ from the shared reader's (`Restated`)."""
    undeclared: frozenset[str] = frozenset()
    """Shared keys the class declares none of, written as nothing (`Restated`)."""
    ramps: tuple[str, ...] | None = None
    tie_embeddings: bool = False
    qk_norm: bool = False
    scale_after_cast: bool = True
    windows: tuple[str, ...] = ('sliding_window',)
    """The keys the class states its window under, the first a config states (not null) read, written as
    the first (MiMoV2Config's `sliding_window`, else its `sliding_window_size`)."""
    kinds: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))
    """The class's own names for the shared layer kinds, own to shared, its layer_types read and written
    under them (Qwen4ExpTextConfig's `qwen_sparse_attention`, its full attention)."""


def _layer_kinds(spelling: DecoderSpelling) -> ConfigPart[CausalTransformer]:
    """The layers' kinds and the sliding window, both ways: `layer_types` where the reference reads it,
    else the pattern an earlier part of the family set (`record['layer_types']`) or the spelling's,
    every layer full without either; the window where it reads one, refused for sliding layers without."""
    reads = spelling.reads

    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        for optional in _OPTIONAL_FIELDS - reads:
            if optional in hf_config and _neutral(hf_config, optional):
                used.add(optional)
        preset = record.get('layer_types')
        pattern = spelling.pattern if preset is None else _records.strings(preset, 'layer_types')
        if pattern is None and spelling.windowed and hf_config.get('sliding_window') is not None:
            pattern = ('sliding_attention',)
        if 'layer_types' in reads:
            layer_types = tuple(spelling.kinds.get(kind, kind)
                                for kind in specified_layer_types(hf_config, used, pattern))
        else:
            layers = _records.integer(record['num_layers'], 'num_layers')
            cycle = pattern or ('full_attention',)
            layer_types = tuple(cycle[index % len(cycle)] for index in range(layers))
        stated = (next((hf_config[key] for key in spelling.windows if hf_config.get(key) is not None), None)
                  if 'sliding_window' in reads else None)
        if 'sliding_window' in reads:
            used.update(spelling.windows)
        if 'sliding_attention' in layer_types and stated is None:
            refuse("layer_types with sliding attention",
                   "sliding_window is not set, so the window has no size")
        window = _records.integer(stated, 'sliding_window') if 'sliding_attention' in layer_types else None
        record['layer_types'] = layer_types
        record['kinds'] = decoder_kinds(layer_types, window, None, None, None)
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        types = model.per_layer_types
        written: dict[str, object] = {}
        if 'layer_types' in reads and any(layer != 'full_attention' for layer in types):
            own = {shared: kind for kind, shared in spelling.kinds.items()}
            written['layer_types'] = [own.get(layer, layer) for layer in types]
        sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
        if 'sliding_window' in reads and sliding is not None and sliding.window is not None:
            written['sliding_window'] = sliding.window
        return written
    return ConfigPart(read, write)


def place_ropes(record: DecoderFields, ropes: Ropes) -> None:
    """Set a read rotary (`Ropes`) on `record`: the model's base and ramp, the sliding kind's own where
    they differ. A ramp both kinds share is the model's; one the full layers alone carry (OLMo 3's
    spelling) lands on the full kind, because a kind's None rides the model's value and cannot turn a
    ramp off; where the kinds' ramps differ, each lands on its own kind (transformers dispatches each
    kind's record on its own)."""
    record['rope_theta'] = ropes.theta
    kinds = kinds_of(record)
    layer_types = _records.strings(record['layer_types'], 'layer_types')
    if ropes.local_theta is not None and 'sliding_attention' in kinds:
        kinds['sliding_attention']['rope_theta'] = ropes.local_theta
    apart = ropes.scaling is not None and (ropes.full_only or ropes.local_scaling is not None)
    if ropes.scaling is not None:
        if apart and 'sliding_attention' in layer_types:
            kinds.setdefault('full_attention', native_fields(LayerKind)())['rope_scaling'] = ropes.scaling
        else:
            record['rope_scaling'] = ropes.scaling
    if ropes.local_scaling is not None:
        kinds['sliding_attention']['rope_scaling'] = ropes.local_scaling
    record['kinds'] = kinds


def written_ropes(model: CausalTransformer, spelling: RopeSpelling) -> Mapping[str, object]:
    """`place_ropes` backwards, as `spelling`'s class reads its records (`rope_records`): one base and
    ramp flat (`rope_theta`, `rope_scaling`, as Llama 3.1 ships them), a sliding base under the class's
    own field (Gemma 3's rope_local_base_freq), or a record per kind (`rope_parameters`, the spelling
    Olmo3Config.to_dict writes), which a class with one table for the whole model cannot say: refused,
    naming the kinds. A class with a default of its own for the sliding base states both bases."""
    types = model.per_layer_types
    sliding = model.kind_of('sliding_attention') if 'sliding_attention' in types else None
    local_theta = None if sliding is None or sliding.rope_theta == model.rope_theta else sliding.rope_theta
    if sliding is not None and spelling.local_theta is not None:
        local_theta = sliding.rope_theta or model.rope_theta
    tables = {kind: model.kind_of(kind).rope_scaling for kind in sorted(set(types))}
    ramped = {kind: table for kind, table in tables.items() if table is not None}
    per_kind = bool(ramped) and (set(ramped) != set(tables) or len(set(ramped.values())) > 1
                                 or local_theta is not None)
    if per_kind or (local_theta is not None and spelling.local_field is None):
        if not spelling.per_kind:
            raise ExportRefused(f"the rotary differs between the layer kinds {sorted(tables)}, and the "
                                "reference rotates every layer at one table; the model cannot be written "
                                "back")
        return {'rope_parameters': {
            kind: {**({'rope_type': 'default'} if table is None else dataclasses.asdict(table)),
                   'rope_theta': model.kind_of(kind).rope_theta} for kind, table in tables.items()}}
    written: dict[str, object] = {'rope_theta': model.rope_theta}
    if local_theta is not None and spelling.local_field is not None:
        written[spelling.local_field] = local_theta
    if ramped:
        written['rope_scaling'] = dataclasses.asdict(next(iter(ramped.values())))
    return written


def rope_part(spelling: RopeSpelling, *, ramps: tuple[str, ...] | None = None,
              per_kind: bool = True) -> ConfigPart[CausalTransformer]:
    """The rotary, both ways: read as `spelling`'s class resolves its records (`read_rope`) and placed
    on the model and its kinds (`place_ropes`), written back as that class reads them
    (`written_ropes`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        place_ropes(record, read_rope(hf_config, used, spelling, ramps=ramps, per_kind=per_kind))
        return record
    return ConfigPart(read, lambda model: written_ropes(model, spelling))


def partial_rope_part(spelling: RopeSpelling = PLAIN_SPELLING, *, default_factor: float,
                      layout: tuple[str, ...] = (), kind: str = 'default',
                      whole_is_none: bool = False) -> ConfigPart[CausalTransformer]:
    """A rotary turning a fraction of each head, both ways: read as its one record states it
    (`read_partial_rope`: the record's fraction, else the config's flat one, else `default_factor`, the
    class's), the ramp over the rotated width under the `kind` convention; written as that one record, its
    base and fraction beside its ramp, as transformers 5 writes a config. `layout` names the record's
    fields that place the rotated pairs without changing their angles (Qwen 3.5's mRoPE sections), which
    its wrapper reads (`hf_decoders`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        ropes, factor = read_partial_rope(hf_config, used, spelling, default_factor=default_factor,
                                          layout=layout)
        place_ropes(record, ropes)
        # A family whose model turns whole heads by default holds the whole head as none.
        record['partial_rotary_factor'] = None if whole_is_none and factor == 1.0 else factor
        record['partial_rotary_type'] = kind
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        ramp = model.rope_scaling
        entry = ({'rope_type': 'default'} if ramp is None
                 else {key: value for key, value in dataclasses.asdict(ramp).items() if value is not None})
        return {'rope_parameters': {**entry, 'rope_theta': model.rope_theta,
                                    'partial_rotary_factor': model.partial_rotary_factor or 1.0}}
    return ConfigPart(read, write)


def plain_rope_part(spelling: RopeSpelling | None = PLAIN_SPELLING, *, default_factor: float | None,
                    reference: str, factor_field: str = 'partial_rotary_factor',
                    theta_field: str = 'rope_theta', kind: str = 'default') -> ConfigPart[CausalTransformer]:
    """A plain rotary turning a fraction of each head, both ways (`plain_partial_rope`): its base and
    fraction under the family's flat fields (`theta_field`, `factor_field`; GPT-NeoX's `rotary_emb_base`
    and `rotary_pct`) or its rope_parameters entry, the class's `default_factor` where unstated (None for
    a class with none, whose config states it), the whole head none; written flat."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.update((theta_field, factor_field, 'rope_theta', 'rope_parameters', 'rope_scaling',
                     'partial_rotary_factor'))
        theta, factor = plain_partial_rope(hf_config, spelling, default_factor=default_factor,
                                           reference=reference, factor_field=factor_field,
                                           theta_field=theta_field)
        record['rope_theta'] = theta
        record['partial_rotary_factor'] = None if factor == 1 else factor
        record['partial_rotary_type'] = kind
        return record
    return ConfigPart(read, lambda model: {theta_field: model.rope_theta,
                                           factor_field: model.partial_rotary_factor or 1.0})


def single_depth(key: str) -> ConfigField[CausalTransformer]:
    """An optional single shared-embedding prediction depth under `key`, both ways."""
    def parse(value: Configured, key: str) -> int:
        if type(value) is not int or value not in (0, 1):
            refuse(key, 'only a single shared prediction layer is supported')
        return value
    return ConfigField(key, 'num_nextn_predict_layers', 0, parse=parse)


def biases(key: str, *, default: bool) -> ConfigPart[CausalTransformer]:
    """One flag biasing every projection, the attention's and the MLP's, both ways: `key` (Falcon's `bias`,
    OPT's `enable_bias`, StarCoder2's `use_bias`), its class's `default` where unstated (`config_flag`)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add(key)
        biased = config_flag(hf_config, key, default=default)
        record.update(mlp_bias=biased, attention_bias=biased)
        return record
    return ConfigPart(read, lambda model: {key: model.attention_bias}, keys=(key,))


def four_wide(key: str) -> ConfigPart[CausalTransformer]:
    """The MLP's width under `key`, four times the model's where the config states none (GPT-2's `n_inner`,
    Falcon's `ffn_hidden_size`), both ways."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        used.add(key)
        record['mlp_features'] = _records.integer(
            hf_config.get(key) or 4 * _records.integer(record['emb_features'], 'hidden_size'), key)
        return record
    return ConfigPart(read, lambda model: {key: model.hidden_features}, keys=(key,))


def kind_mixer[Mixer](kind: str, name: str, mixer: type[Mixer], layout: ConfigLayout[Mixer], *,
                      fixed: Mapping[str, Configured] | None = None) -> ConfigPart[CausalTransformer]:
    """The mixer the layers of `kind` hold (a hybrid's delta net, SSD or lightning attention), its keys
    `layout`'s, both ways: read into the kind's record as the `mixers` registry's `name` beside what the
    family's reference fixes of it (`fixed`), its keys read as nothing where no layer or prediction step
    is of the kind; written from the mixer the kind holds."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        if kind not in _record_kinds(record):
            used.update(layout.names())
            return record
        fields = layout.read(hf_config, used, NativeFields(mixer, {}))
        fields.update(fixed or {})
        kinds = kinds_of(record)
        kinds[kind] = NativeFields(LayerKind, {**kinds.get(kind, {}),
                                               'mixer': {'class': name, 'fields': fields}})
        record['kinds'] = kinds
        return record

    def write(model: CausalTransformer) -> Mapping[str, object]:
        steps = (model.mtp_layer_type, model.mtp_feedforward_type)
        held = model.kind_of(kind).mixer if kind in (*model.per_layer_types, *steps) else None
        return layout.write(held) if isinstance(held, mixer) else {}
    return ConfigPart(read, write)


def _record_kinds(record: DecoderFields) -> set[str]:
    """The kinds a translated record's layers and its prediction steps hold (`mtp_layer_type`,
    `mtp_feedforward_type`: Nemotron-H's step builds blocks of kinds no trunk layer need hold)."""
    steps = (record.get('mtp_layer_type'), record.get('mtp_feedforward_type'))
    return {*_records.strings(record['layer_types'], 'layer_types'),
            *(_records.text(kind, 'mtp_layer_type') for kind in steps if kind is not None)}


def kinds_among(kinds: tuple[str, ...], why: str) -> ConfigPart[CausalTransformer]:
    """Refuse a layer of a kind outside `kinds`, saying `why`; written as nothing (`_layer_kinds`
    writes the layers' kinds)."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        unknown = sorted(set(_records.strings(record['layer_types'], 'layer_types')) - set(kinds))
        if unknown:
            refuse(f"layer_types {unknown}", why)
        return record
    return ConfigPart(read, lambda model: {})


ROPE_APART: ConfigPart[CausalTransformer] = ConfigPart(lambda hf_config, used, record: record,
                                                       lambda model: {})
"""The shared decoder's rotary where its family reads and writes it in a part of its own (DeepSeek V4's two
entries, a learned table in place of one)."""


PLAIN_DECODER = DecoderSpelling()
"""The decoder as LlamaConfig's mixin spells it."""


def decoder_layout(spelling: DecoderSpelling = PLAIN_DECODER, *,
                   rope: ConfigEntry[CausalTransformer] | None = None,
                   width: ConfigPart[CausalTransformer] = WIDTH) -> ConfigLayout[CausalTransformer]:
    """The decoder every family shares, both ways, as `spelling` states it: its geometry
    (`DECODER_FIELDS`), its feed-forward's width (`width`: a class that states it under its own key with
    a default of its own, Falcon's `ffn_hidden_size`, declares its own), its layers' kinds and window
    (`_layer_kinds`), its rotary (`rope`, the spelling's `rope_part` where None: a family that reads its
    rotary otherwise, a partial one, declares its own), the biases and tying its class declares, and what
    its reference fixes."""
    reads = spelling.reads
    fields = DECODER_FIELDS if width is WIDTH else ConfigLayout(*_GEOMETRY)
    restated = spelling.keys or spelling.fixed or spelling.defaults or spelling.undeclared
    geometry = (Restated(fields, spelling.keys, spelling.fixed, spelling.defaults, spelling.undeclared)
                if restated else fields)
    return ConfigLayout(
        geometry, *(() if width is WIDTH else (width,)),
        *(() if spelling.activation is None else (activation_part(spelling.activation),)),
        _layer_kinds(spelling),
        rope_part(spelling.rope, ramps=spelling.ramps) if rope is None else rope,
        *(flag(name, name, default=False) if name in reads else constant(name, value=False)
          for name in ('attention_bias', 'mlp_bias')),
        flag('tie_word_embeddings', 'tie_embeddings', default=spelling.tie_embeddings),
        constant('qk_norm', value=spelling.qk_norm),
        constant('scale_after_cast', value=spelling.scale_after_cast))


def integer(key: str, field: str, default: int | None | Required = REQUIRED, *aliases: str) -> ConfigField:
    """A config key holding an integer field, its class's `default` where it states none (none where
    REQUIRED), read under `aliases` where `key` is absent."""
    return ConfigField(key, field, default, parse=_records.integer, aliases=aliases)


def flag(key: str, field: str, *, default: bool) -> ConfigField:
    """A boolean the config states under `key`, its class's `default` where it states none or null
    (`config_flag`)."""
    return ConfigField(key, field, default,
                       parse=lambda value, key: config_flag({key: value}, key, default=default))


TRAINING_ROUTER = unread('output_router_logits', 'router_aux_loss_coef')
"""A router's training controls, which a forward reads to no effect."""


def stated_experts(model: CausalTransformer) -> Mixture | None:
    """The model's routed experts as a config states them: their width the model's feed-forward's where
    they name none of their own (`Mixture.expert_features`)."""
    mixture = model.mixture
    return None if mixture is None else dataclasses.replace(
        mixture, expert_features=mixture.expert_features or model.hidden_features)


def experts(*fields: ConfigEntry[Mixture], present: Callable[[Mapping[str, object]], bool] | None = None
            ) -> Nested[CausalTransformer, Mixture]:
    """A model's routed experts (`Mixture`), their keys `fields`, under its `mixture` (none where
    `present`, of the source config, says the config routes no layer)."""
    return Nested('mixture', Mixture, ConfigLayout(*fields), stated_experts, present=present)


LOCAL_EXPERTS = ConfigLayout(experts(integer('num_local_experts', 'experts'),
                                     integer('num_experts_per_tok', 'top_k')), TRAINING_ROUTER)
"""Mixtral's routed experts, which GraniteMoE's share: `num_local_experts` softmax experts on every layer,
the top `num_experts_per_tok` of them renormalized, both ways."""
