"""Translate MiniMax-Text-01 and MiniMax-M1 (`minimax`, and the releases' own `minimax_text_01` and
`minimax_m1`).

The stack alternates MiniMax's lightning attention (`dew.nn.mixers.lightning`) with softmax attention,
seven to one in the releases, over Mixtral's routed experts. Every sublayer joins the stream as its
norm left it, scaled by an alpha, the branch by a beta (`postnorm`, modeling_minimax_text_01.py at
MiniMaxAI/MiniMax-Text-01 a7351bf2, :1024-1058; `BlockWiring.normed_residual`). The softmax layers
rotate the first `rotary_dim` of each head (:482-489, :430-437); transformers 5.16.1's MiniMax, which
reads the `minimax` spelling of the same releases, turns every dimension of a head instead
(`MiniMaxRotaryEmbedding`, modeling_minimax.py:266-310), and Dew reads the releases as their own code
computes them. No runtime outside Dew computes them so, so a model of this family writes no Hugging
Face checkpoint: transformers' MiniMax turns every dimension of a head and scales each layer's rates by
`1 - layer / (layers - 1 + 1e-5) + 1e-5` (`get_slope_rate`), its slopes a power of two's whatever the
heads, and the release's own code does not import under transformers 5 (`is_torch_fx_available`) and
drops a padded row's mask in its softmax layers (ledgers 2534, 2535); its export is refused, naming that.
A lightning layer's rates follow its place in the stack (`lightning_rates`), so each is a kind of its
own, `linear_attention_{index}`.

The `minimax` configs spell the layer pattern as `layer_types` and the factors as
`{full_attn,linear_attn,mlp}_{alpha,beta}_factor`; the releases' own as `attn_type_list` (0 for a
lightning layer, 1 for a softmax one) and `layernorm_{full_attention,linear_attention,mlp}_{alpha,beta}`.
One factor of each kind scales every sublayer (every release states one), and a shared expert, which
no release has, is refused by name.
"""

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    MOE_NAMES,
    DecoderFamily,
    DecoderFields,
    RopeSpelling,
    base_config,
    kind_mixers,
    kinds_of,
    read_partial_rope,
    refuse,
    renamed_path,
    softmax_top_k,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.nn.backbones.layer_plan import LayerKind
from dew.nn.mixers.lightning import ACTIVATIONS, LightningAttentionMixer

_SPELLING = RopeSpelling(theta=1000000.0)
"""MiniMaxConfig's and MiniMaxText01Config's default rope base."""

_BLOCK = 256
"""MiniMaxConfig's `block_size` where a config states none, the remote code's `BLOCK` (:51)."""

_OWN = ('minimax_text_01', 'minimax_m1')
"""The releases' own model types, read by their remote code (`attn_type_list`, `layernorm_*`)."""

_FACTORS = {
    'alpha': ('full_attn_alpha_factor', 'linear_attn_alpha_factor', 'mlp_alpha_factor'),
    'beta': ('full_attn_beta_factor', 'linear_attn_beta_factor', 'mlp_beta_factor'),
}
"""The factors of the stream (alpha) and of the branch (beta) each sublayer joins with, as `minimax`
spells them."""

_OWN_FACTORS = {
    'full_attn_alpha_factor': 'layernorm_full_attention_alpha',
    'full_attn_beta_factor': 'layernorm_full_attention_beta',
    'linear_attn_alpha_factor': 'layernorm_linear_attention_alpha',
    'linear_attn_beta_factor': 'layernorm_linear_attention_beta',
    'mlp_alpha_factor': 'layernorm_mlp_alpha', 'mlp_beta_factor': 'layernorm_mlp_beta',
}
"""Each factor as the releases' own configs spell it (MiniMaxText01DecoderLayer, :963-969)."""

_LIGHTNING_LEAVES = {
    'self_attn.qkv_proj.weight': ('self_attn', 'qkv_proj', 'kernel'),
    'self_attn.output_gate.weight': ('self_attn', 'output_gate', 'kernel'),
    'self_attn.out_proj.weight': ('self_attn', 'out_proj', 'kernel'),
    'self_attn.norm.weight': ('self_attn', 'norm', 'scale'),
}
"""A lightning layer's tensors below `model.layers.N.`, at their place in its mixer."""


def _factor(hf: Mapping[str, object], used: set[str], kind: str) -> float:
    """The one factor of `kind` ('alpha' or 'beta') every sublayer joins with, in either spelling."""
    values = {}
    for name in _FACTORS[kind]:
        own = _OWN_FACTORS[name]
        used.update((name, own))
        values[name] = float(_records.number(hf.get(name, hf.get(own, 1.0)), name))
    if len(set(values.values())) > 1:
        refuse(f'{kind} factors {values}',
               'Dew joins every sublayer at one factor, and each release states one')
    return next(iter(values.values()))


def _layer_types(hf: Mapping[str, object], used: set[str], layers: int) -> tuple[str, ...]:
    """Each layer's attention, `linear_attention` or `full_attention`: `layer_types`, else the releases'
    `attn_type_list`, else MiniMaxConfig's alternation from a softmax layer."""
    used.update(('layer_types', 'attn_type_list'))
    if 'layer_types' in hf:
        types = _records.strings(hf['layer_types'], 'layer_types')
    elif 'attn_type_list' in hf:
        names = {0: 'linear_attention', 1: 'full_attention'}
        listed = _records.integers(hf['attn_type_list'], 'attn_type_list')
        if set(listed) - set(names):
            refuse(f'attn_type_list {listed}', 'a layer is lightning (0) or softmax (1) attention')
        types = tuple(names[kind] for kind in listed)
    else:
        types = tuple('full_attention' if (index + 1) % 2 else 'linear_attention' for index in range(layers))
    unknown = sorted(set(types) - {'linear_attention', 'full_attention'})
    if unknown or len(types) != layers:
        refuse(f'layer_types {list(types)}',
               f'each of the {layers} layers is linear_attention or full_attention')
    return types


def _minimax_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    layers = _records.integer(hf['num_hidden_layers'], 'num_hidden_layers')
    types = _layer_types(hf, used, layers)
    own = hf.get('model_type') in _OWN
    # transformers' MiniMax builds the postnorm block whatever its config states; the releases' own
    # code reads the flag, False where it is absent (:963).
    postnorm = bool(hf.get('postnorm', not own))
    if not own and not postnorm:
        refuse('postnorm',
               "transformers' MiniMax joins every sublayer to its normed input, as the releases do")
    used.add('postnorm')
    for name, neutral, reason in (
            ('shared_intermediate_size', 0, 'no release routes beside a shared expert'),
            ('router_jitter_noise', 0.0, 'training-time input jitter has no counterpart')):
        used.add(name)
        if hf.get(name, neutral) != neutral:
            refuse(name, f'{reason}; requires {neutral!r}')
    # The shared expert's gate, which builds nothing without one (:972-976).
    used.add('shared_moe_mode')
    width = _records.integer(hf['hidden_size'], 'hidden_size')
    head_dim = _records.integer(
        hf.get('head_dim') or width // _records.integer(hf['num_attention_heads'], 'num_attention_heads'),
        'head_dim')
    rotary_dim = _records.integer(hf.get('rotary_dim', head_dim), 'rotary_dim')
    used.add('rotary_dim')
    ropes, factor = read_partial_rope(hf, used, _SPELLING, default_factor=rotary_dim / head_dim)
    if factor * head_dim != rotary_dim or rotary_dim % 2 or not 0 < rotary_dim <= head_dim:
        refuse(f'rotary_dim {rotary_dim}', f'it turns an even part of each {head_dim}-wide head, '
               f'{factor} of it as the rotary states')
    config = base_config({**hf, 'head_dim': head_dim}, used, rope=ropes, layer_types=types, reads=frozenset())
    activation = _records.text(hf.get('hidden_act', 'silu'), 'hidden_act')
    if 'linear_attention' in types and activation not in ACTIVATIONS:
        refuse(f'hidden_act {activation!r}', f'the lightning projection takes {sorted(ACTIVATIONS)}')
    chunk = _records.integer(hf.get('block_size', _BLOCK), 'block_size')
    used.add('block_size')
    kinds = {name: kind for name, kind in kinds_of(config).items() if name != 'linear_attention'}
    named = []
    for index, kind in enumerate(types):
        if kind == 'full_attention':
            named.append(kind)
            continue
        name = f'linear_attention_{index}'
        named.append(name)
        kinds[name] = native_fields(LayerKind)(mixer=None)
        kinds[name]['mixer'] = {'class': 'lightning_attention', 'fields': {
            'layer_idx': index, 'num_hidden_layers': layers, 'hidden_act': activation, 'block_size': chunk}}
    used.update(('num_local_experts', 'num_experts'))
    config.update(
        layer_types=tuple(named), kinds=kinds, partial_rotary_factor=factor, partial_rotary_type='default',
        normed_residual=postnorm, stream_multiplier=_factor(hf, used, 'alpha'),
        residual_multiplier=_factor(hf, used, 'beta'),
        mixture=native_fields(Mixture)(
            top_k=softmax_top_k(hf, used),
            experts=_records.integer(hf.get('num_local_experts', 8), 'num_local_experts')))
    return config


def _lightning(model: CausalTransformer) -> list[LightningAttentionMixer]:
    return [mixer for mixer in kind_mixers(model) if isinstance(mixer, LightningAttentionMixer)]


def _minimax_export(model: CausalTransformer) -> Mapping[str, object]:
    """No config: no runtime outside Dew computes this family as its release does (the module docstring),
    so a Hugging Face checkpoint of it would open as another model."""
    raise ValueError("MiniMax-Text-01 and M1 write no Hugging Face checkpoint: transformers' MiniMax turns "
                     "every dimension of a head and scales its lightning rates otherwise, and the release's "
                     "own code does not import under transformers 5 (ledgers 2534, 2535)")


MINIMAX = DecoderFamily(
    ('minimax', *_OWN),
    _minimax_config,
    lambda fields: bool(_lightning(fields) or fields.normed_residual or fields.stream_multiplier != 1.0),
    'minimax',
    'MiniMaxForCausalLM',
    _minimax_export,
    weight_path=partial(renamed_path, MOE_NAMES, layer_leaves=_LIGHTNING_LEAVES),
    preserve_source_layout=False,
    padding_idx=True,
    # MiniMaxConfig's defaults (configuration_minimax.py), the releases' own class's too.
    defaults={'num_experts_per_tok': 2, 'num_local_experts': 8, 'rms_norm_eps': 1e-5,
              'num_key_value_heads': 8, 'max_position_embeddings': 131072},
    aliases={'num_experts': 'num_local_experts'},
)
