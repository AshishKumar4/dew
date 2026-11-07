"""Phi's one-norm parallel biased block, sliced rotary and affine output head."""

from dew import _records
from dew.interop.decoder_config import DecoderSpelling, decoder_layout, inert, partial_rope_part, ungated
from dew.interop.decoder_parts import DecoderFamily, Renames, spelled
from dew.interop.layouts import ConfigField, ConfigLayout, constant

_PHI_NAMES: Renames = (
    ('model.final_layernorm', 'model.norm'), ('self_attn.dense', 'self_attn.o_proj'),
    ('mlp.fc1', 'mlp.up_proj'), ('mlp.fc2', 'mlp.down_proj'))


PHI_CONFIG = ConfigLayout(
    # PhiRotaryEmbedding builds any rope_type the rope utilities do over the rotated half, YaRN among them
    # (modeling_phi.py:42-45, `_compute_yarn_parameters`).
    decoder_layout(DecoderSpelling(reads=frozenset(), scale_after_cast=False, activation=None,
                                   keys={'rms_norm_eps': 'layer_norm_eps'}, defaults={'layer_norm_eps': 1e-5},
                                   undeclared=frozenset({'head_dim'})),
                   rope=partial_rope_part(default_factor=.5)),
    ungated('hidden_act', ('gelu_new', 'gelu_pytorch_tanh', 'gelu', 'relu'), 'Phi'),
    inert('qk_layernorm', 'Phi q/k LayerNorm has no RMSNorm counterpart', neutral=False),
    ConfigField('resid_pdrop', 'dropout_rate', 0.0, parse=_records.number),
    ConfigField('embd_pdrop', 'embedding_dropout_rate', 0.0, parse=_records.number),
    ConfigField('attention_dropout', 'attention_dropout_rate', 0.0,
                parse=lambda value, key: _records.number(value or 0., key)),
    constant('mlp_bias', value=True), constant('norm_type', value='layer'), constant('norm_bias', value=True),
    constant('parallel_residual', value=True), constant('shared_parallel_norm', value=True),
    constant('head_bias', value=True), constant('attention_bias', value=True))
"""Phi-1's and Phi-2's config, both ways: a parallel biased-LayerNorm block under one norm, half of each
head rotated."""


PHI = DecoderFamily(
    ('phi',), PHI_CONFIG,
    lambda fields: fields.shared_parallel_norm and fields.head_bias and fields.attention_bias,
    'phi', 'PhiForCausalLM',
    routes=spelled(_PHI_NAMES), preserve_source_layout=False,
    # PhiConfig's context (configuration_phi.py:73), which a YaRN record's omitted lengths read.
    defaults={'max_position_embeddings': 2048},
    padding_idx=True,
)
