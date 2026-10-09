"""StableLM's biased LayerNorm block: grouped-query attention over a partial
rotary, biased queries, keys and values under a bias-free output, and a
bias-free gated feed-forward, the branches sequential or in parallel off one
norm. `qk_layernorm` norms each query and key head under its own LayerNorm,
which the checkpoint stores one tensor a head."""

from collections.abc import Mapping

import jax
import numpy as np

from dew import records
from dew.interop.decoder_parts import (
    DecoderFamily,
    DecoderFields,
    Ropes,
    base_config,
    decoder_tensors,
    plain_partial_rope,
    refuse,
)
from dew.interop.safetensors_io import LazyTensors
from dew.nn.backbones.causal_transformer import CausalTransformer

_HEAD_NORMS = (('q_layernorm', 'q_norm'), ('k_layernorm', 'k_norm'))
"""StableLmLayerNormPerHead's module names, and the norm each stacks into."""


def _stablelm_config(hf: Mapping[str, object], used: set[str]) -> DecoderFields:
    if hf.get('rope_scaling') is not None:
        refuse('rope_scaling', 'StableLmRotaryEmbedding is the plain rotary')
    # StableLmDecoderLayer drops the feed-forward branch alone, which the
    # shared block's dropout, over both branches, does not compute.
    if hf.get('hidden_dropout', 0.):
        refuse(f"hidden_dropout {hf['hidden_dropout']}", 'StableLM drops its feed-forward branch alone')
    theta, factor = plain_partial_rope(hf, default_factor=0.25, reference='StableLmRotaryEmbedding')
    parallel = bool(hf.get('use_parallel_residual', False))
    used.update(('rope_theta', 'rope_parameters', 'rope_scaling', 'partial_rotary_factor', 'use_qkv_bias',
                 'qk_layernorm', 'use_parallel_residual', 'layer_norm_eps', 'hidden_dropout',
                 'attention_dropout'))
    config = base_config(hf, used, rope=Ropes(theta), qk_norm=bool(hf.get('qk_layernorm', False)),
                         reads=frozenset())
    config.update(
        norm_type='layer', norm_bias=True,
        norm_eps=records.number(hf.get('layer_norm_eps', 1e-5), 'layer_norm_eps'),
        attention_bias=bool(hf.get('use_qkv_bias', False)), o_proj_bias=False,
        qk_norm_scope='head_layernorm', parallel_residual=parallel, shared_parallel_norm=parallel,
        partial_rotary_factor=None if factor == 1.0 else factor, partial_rotary_type='default',
        attention_dropout_rate=records.number(hf.get('attention_dropout', 0.), 'attention_dropout'))
    return config


def _stablelm_prepare(tensors: Mapping[str, np.ndarray],
                      config: Mapping[str, object] | None = None) -> Mapping[str, np.ndarray]:
    """Each layer's per-head norm weights, `{q,k}_layernorm.norms.<h>.weight`,
    stacked by head into the one `[heads, head_dim]` scale they compute."""
    prepared: dict[str, np.ndarray] = {}
    heads: dict[str, dict[int, np.ndarray]] = {}
    for name, tensor in tensors.items():
        for theirs, ours in _HEAD_NORMS:
            stem, found, rest = name.partition(f'.self_attn.{theirs}.norms.')
            if found:
                index, _, leaf = rest.partition('.')
                if leaf != 'weight':
                    raise ValueError(f'{name}: StableLM norms each head without a bias')
                heads.setdefault(f'{stem}.self_attn.{ours}.weight', {})[int(index)] = tensor
                break
        else:
            prepared[name] = tensor
    for name, rows in heads.items():
        if sorted(rows) != list(range(len(rows))):
            raise ValueError(f'{name} needs one norm a head, got heads {sorted(rows)}')
        prepared[name] = np.stack([rows[index] for index in range(len(rows))])
    return prepared


def _stablelm_export_weights(family: DecoderFamily, model: CausalTransformer, variables: Mapping[str, object],
                             config: Mapping[str, object]) -> LazyTensors:
    """The shared writer's tensors with each `[heads, head_dim]` head norm
    written one tensor a head, the inverse of `_stablelm_prepare`."""
    tensors = decoder_tensors(family, model, variables, config)
    rows: dict[str, tuple[str, int]] = {}
    for name, spec in tensors.specs.items():
        for theirs, ours in _HEAD_NORMS:
            stem, found, _ = name.partition(f'.self_attn.{ours}.')
            if found:
                rows.update({f'{stem}.self_attn.{theirs}.norms.{index}.weight': (name, index)
                             for index in range(spec.shape[0])})
    stacked = {name for name, _ in rows.values()}
    specs = {name: spec for name, spec in tensors.specs.items() if name not in stacked}
    for name, (whole, _) in rows.items():
        spec = tensors.specs[whole]
        specs[name] = jax.ShapeDtypeStruct(spec.shape[1:], spec.dtype)

    def build(name: str) -> np.ndarray:
        if name not in rows:
            return tensors[name]
        whole, index = rows[name]
        return np.ascontiguousarray(tensors[whole][index])

    return LazyTensors(specs, build)


def _stablelm_export(model: CausalTransformer) -> Mapping[str, object]:
    return {
        'layer_norm_eps': model.norm_eps, 'use_qkv_bias': model.attention_bias, 'qk_layernorm': model.qk_norm,
        'use_parallel_residual': model.parallel_residual, 'hidden_dropout': 0.,
        'attention_dropout': model.attention_dropout_rate,
        'rope_parameters': {'rope_type': 'default', 'rope_theta': model.rope_theta,
                            'partial_rotary_factor': model.partial_rotary_factor or 1.},
        'rope_theta': None, 'rms_norm_eps': None, 'attention_bias': None, 'head_dim': None,
    }


def _matches(fields: CausalTransformer) -> bool:
    """A biased-LayerNorm rotary decoder with a gated feed-forward."""
    return bool(fields.norm_type == 'layer' and fields.norm_bias and fields.position_embedding == 'rotary'
                and fields.mlp in ('swiglu', 'geglu', 'geglu_exact') and not fields.mlp_bias
                and (not fields.qk_norm or fields.qk_norm_scope == 'head_layernorm'))


STABLELM = DecoderFamily(
    ('stablelm',), _stablelm_config, _matches, 'stablelm', 'StableLmForCausalLM', _stablelm_export,
    prepare=_stablelm_prepare, export_weights=_stablelm_export_weights, preserve_source_layout=False,
)
