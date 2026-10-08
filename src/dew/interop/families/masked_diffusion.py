"""Translate the bidirectional decoders: LLaDA, Dream and DiffusionGemma's text half.

Each is a causal family's block with the mask replaced: full attention in
place of the causal mask, and a mask token the sampler denoises to. Dream
rides the qwen2 layout and DiffusionGemma the Gemma 4 one, so only LLaDA's
OLMo-style tensor names need a map of their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    FUSED_EXPERTS,
    DecoderFamily,
    DecoderFields,
    Renames,
    Ropes,
    base_config,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.gemma import (
    gemma4_config,
    gemma4_export_path,
    gemma4_export_weights,
    gemma4_path,
    parallel_experts,
)
from dew.interop.families.qwen import qwen2_config
from dew.nn.backbones.causal_transformer import CausalTransformer


def _mask_token(hf_config: Mapping[str, object], used: set[str]) -> int:
    """Return the mask id a masked-diffusion release reserves, under either spelling.

    The releases write it as `mask_token_id` or as `mask_id`.
    """
    mask = hf_config.get('mask_token_id', hf_config.get('mask_id'))
    if mask is None:
        refuse('mask_token_id', 'a masked diffusion checkpoint reserves its mask id')
    used.update(('mask_token_id', 'mask_id'))
    return records.integer(mask, 'mask_token_id/mask_id')


def _llada_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a LLaDA-8B config into `CausalTransformer` fields.

    GSAI-ML/LLaDA-8B-Base (model_type 'llada', architectures ['LLaDAModelLM'])
    computes a Llama block: RMSNorm pre-norms, rotate-half rope, SwiGLU. It
    carries no causal mask anywhere (modeling_llada.py, LLaDAModel
    bidirectional bias, LLaDALlamaBlock is_causal=False) and trains masked
    diffusion on the id in mask_token_id.

    The checkpoint names its tensors OLMo-style
    (model.transformer.blocks.N.{attn_norm,q_proj,k_proj,v_proj,attn_out,
    ff_norm,ff_proj,up_proj,ff_out}, wte, ln_f, ff_out for the head), so
    `_llada_path` renames them onto the llama layout the shared map reads.
    """
    std = _llada_geometry(hf_config)
    layers = records.integer(std['num_hidden_layers'], 'num_hidden_layers/n_layers')
    config = base_config(std, set(), layer_types=('full_attention',) * layers,
                          rope=Ropes(records.number(std['rope_theta'], 'rope_theta')))
    used.update(key for key in _LLADA_READ if key in hf_config)
    mask = _mask_token(hf_config, used)
    _llada_refusals(hf_config, used, std)
    config.update(causal=False, mask_token_id=mask)
    return config


# The config aliases the release writes, beside the standard spellings the
# shared reader takes; a key present under either name has been read.
_LLADA_READ = ('hidden_size', 'd_model', 'num_hidden_layers', 'n_layers', 'num_layers',
               'num_attention_heads', 'n_heads', 'num_key_value_heads', 'n_kv_heads',
               'head_dim', 'intermediate_size', 'mlp_hidden_size', 'vocab_size',
               'embedding_size', 'max_position_embeddings', 'max_sequence_length',
               'rms_norm_eps', 'norm_eps', 'rope_theta', 'attention_bias', 'include_bias',
               'include_qkv_bias', 'tie_word_embeddings', 'weight_tying', 'hidden_act',
               'hidden_activation', 'activation_type')


def _llada_geometry(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """Rewrite LLaDA's own config spellings under the standard names `base_config` reads.

    The aliases are d_model, n_layers, n_heads, n_kv_heads, mlp_hidden_size,
    embedding_size and max_sequence_length. Each is read beside the standard
    spelling it stands in for.
    """
    hidden = hf_config.get('hidden_size', hf_config.get('d_model'))
    layers = hf_config.get('num_hidden_layers', hf_config.get('n_layers',
                           hf_config.get('num_layers')))
    heads = hf_config.get('num_attention_heads', hf_config.get('n_heads'))
    kv_heads = hf_config.get('num_key_value_heads', hf_config.get('n_kv_heads'))
    intermediate = hf_config.get('intermediate_size', hf_config.get('mlp_hidden_size'))
    vocab = hf_config.get('vocab_size', hf_config.get('embedding_size'))
    if hidden is None or layers is None or heads is None or intermediate is None or vocab is None:
        refuse('llada geometry',
                'd_model/hidden_size, n_layers/num_hidden_layers, n_heads/num_attention_heads, '
                'mlp_hidden_size/intermediate_size and embedding_size/vocab_size are required')
    kv_heads = heads if kv_heads is None else kv_heads
    head_dim = hf_config.get('head_dim')
    head_dim = (
        records.integer(head_dim, "head_dim")
        if head_dim is not None
        else records.integer(hidden, "hidden_size/d_model")
        // records.integer(heads, "num_attention_heads/n_heads")
    )
    max_pos = hf_config.get('max_position_embeddings',
                            hf_config.get('max_sequence_length', DEFAULT_MAX_SEQ_LEN))
    std: dict[str, object] = {
        "hidden_size": records.integer(hidden, "hidden_size/d_model"),
        "num_attention_heads": records.integer(heads, "num_attention_heads/n_heads"),
        "num_key_value_heads": records.integer(kv_heads, "num_key_value_heads/n_kv_heads"),
        "head_dim": head_dim,
        "intermediate_size": records.integer(intermediate, "intermediate_size/mlp_hidden_size"),
        "vocab_size": records.integer(vocab, "vocab_size/embedding_size"),
        "num_hidden_layers": records.integer(layers, "num_hidden_layers/n_layers"),
        "max_position_embeddings": min(
            records.integer(max_pos, "max_position_embeddings/max_sequence_length"), DEFAULT_MAX_SEQ_LEN
        ),
        "rms_norm_eps": records.number(
            hf_config.get("rms_norm_eps", hf_config.get("norm_eps", 1e-6)), "rms_norm_eps"
        ),
        "rope_theta": records.number(hf_config.get("rope_theta", 500000.0), "rope_theta"),
        "attention_bias": bool(
            hf_config.get(
                "attention_bias", hf_config.get("include_bias", hf_config.get("include_qkv_bias", False))
            )
        ),
        "tie_word_embeddings": bool(
            hf_config.get("tie_word_embeddings", hf_config.get("weight_tying", False))
        ),
        "hidden_act": hf_config.get(
            "hidden_act", hf_config.get("hidden_activation", hf_config.get("activation_type", "silu"))
        ),
    }
    return std


def _llada_refusals(hf_config: Mapping[str, object], used: set[str],
                    std: Mapping[str, object]) -> None:
    """Check the release's own config flags against the computation this entry builds.

    Dropout, init and kernel flags describe training or the kernel, not the
    eval forward, so they are marked read and left alone. A flag that would
    change the eval computation refuses. `std` is the geometry already read, so
    the two checks against it (the biases and the activation) see the resolved
    value rather than an alias.
    """
    layers = records.integer(std['num_hidden_layers'], 'num_hidden_layers/n_layers')
    if std['attention_bias']:
        refuse('include_bias/include_qkv_bias', 'the released checkpoint carries no biases')
    if std['hidden_act'] != 'silu':
        refuse(f"activation_type {std['hidden_act']!r}", 'LLaDA-8B computes SwiGLU')
    if hf_config.get('rope') is False:
        refuse('rope=False', 'the released checkpoint rotates every layer')
    used.add('rope')
    if (hf_config.get('rope_parameters') is not None or hf_config.get('rope_scaling') is not None
            or hf_config.get('rope_local_base_freq') is not None):
        refuse('rope_parameters/rope_scaling', 'LLaDA-8B carries plain rope at rope_theta')
    used.update(('rope_parameters', 'rope_scaling', 'rope_local_base_freq'))
    stated = hf_config.get('layer_types')
    if (stated is not None and records.strings(stated, 'layer_types')
            != ('full_attention',) * layers):
        refuse(f'layer_types {list(records.strings(stated, "layer_types"))!r}',
                'LLaDA-8B attends every layer fully')
    if stated is not None:
        used.add('layer_types')
    if hf_config.get('sliding_window') is not None:
        refuse('sliding_window', 'LLaDA-8B windows no layer')
    used.add('sliding_window')
    if hf_config.get('attention_layer_norm'):
        refuse('attention_layer_norm', 'the dense checkpoint norms no queries or keys')
    used.update(('attention_layer_norm', 'attention_layer_norm_with_affine'))
    if hf_config.get('bias_for_layer_norm'):
        refuse('bias_for_layer_norm', 'the RMS norms carry no bias')
    used.add('bias_for_layer_norm')
    if hf_config.get('layer_norm_type', 'rms') != 'rms':
        refuse(f"layer_norm_type {hf_config.get('layer_norm_type')!r}", 'the norms are RMS')
    used.add('layer_norm_type')
    if not hf_config.get('layer_norm_with_affine', True):
        refuse('layer_norm_with_affine=False', 'the released norms scale')
    used.add('layer_norm_with_affine')
    if hf_config.get('input_emb_norm'):
        refuse('input_emb_norm', 'the embeddings enter the first block unnormed')
    used.add('input_emb_norm')
    if hf_config.get('block_type', 'llama') != 'llama':
        refuse(f"block_type {hf_config.get('block_type')!r}", 'this entry is the llama block')
    used.add('block_type')
    if records.integer(hf_config.get('block_group_size', 1), 'block_group_size') != 1:
        refuse('block_group_size', 'the released stack groups no blocks')
    used.add('block_group_size')
    if hf_config.get('alibi'):
        refuse('alibi', 'the attention carries no alibi slopes')
    used.update(('alibi', 'alibi_bias_max'))
    if hf_config.get('scale_logits'):
        refuse('scale_logits', 'the head writes raw logits')
    used.add('scale_logits')
    if not hf_config.get('rope_full_precision', True):
        refuse('rope_full_precision=False', 'the parity fixture runs the rope in fp32')
    used.add('rope_full_precision')
    if hf_config.get('multi_query_attention') not in (None, False):
        refuse('multi_query_attention', 'the head counts come from n_heads/n_kv_heads')
    used.add('multi_query_attention')
    used.update(('embedding_dropout', 'residual_dropout', 'flash_attention', 'precision',
                 'init_device', 'init_fn', 'init_std', 'init_cutoff_factor', 'mlp_ratio',
                 'eos_token_id', 'pad_token_id'))
    embedding = hf_config.get('embedding_size')
    vocab = hf_config.get('vocab_size', embedding)
    if embedding is not None and records.integer(embedding, "embedding_size") != records.integer(
        vocab, "vocab_size"
    ):
        refuse('embedding_size', f'it names {embedding} rows for a {vocab} vocabulary')


def _dream_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a Dream-v0 config into `CausalTransformer` fields.

    Dream-org/Dream-v0-Base-7B (model_type 'Dream', architectures
    ['DreamModel']) is a 28-layer Qwen2.5-7B geometry: 3584 wide, 28 heads, 4
    kv heads. Its attention hard-codes is_causal=False over biased q/k/v and a
    bias-free o_proj (modeling_dream.py, DreamAttention/DreamSdpaAttention),
    and its MLP is bias-free SwiGLU. Tensor names are the qwen2 layout, so the
    shared map reads them with no new table. use_mrope=False is the only
    Dream-only flag and changes nothing at that value.
    """
    if hf_config.get('use_mrope'):
        refuse('use_mrope=True', 'the backbone rotates plain positions')
    used.add('use_mrope')
    config = qwen2_config(hf_config, used)
    mask = _mask_token(hf_config, used)
    config.update(causal=False, mask_token_id=mask)
    return config


def _diffusion_gemma_text_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a DiffusionGemma text config into `CausalTransformer` fields.

    google/diffusiongemma-26B-A4B-it (model_type 'diffusion_gemma_text') is the
    Gemma 4 text geometry with canvas denoising around it. A causal encoder
    over the prompt fills a KV cache, a bidirectional decoder over the canvas
    attends to that cache, and a self-conditioning MLP folds the previous
    step's logits into the input embeddings
    (TF/models/diffusion_gemma/modeling_diffusion_gemma.py:281, :383, :790-823,
    :1326-1440).

    The parameter tree is the same either way, since `causal` changes the mask
    and not the parameters. So this entry maps the weights onto the Gemma 4
    table and marks the record decoder-mode (causal=False). The encoder cache,
    the canvas positions and the self-conditioning loop need a block-diffusion
    Process and objective, which live outside this module.
    """
    # The reference builds no v_proj on full layers whatever the config says
    # (modeling_diffusion_gemma.py, DiffusionGemmaEncoderTextAttention: v_proj
    # only if the layer slides), so the record always reads values off keys
    # there; the sliding layers keep their own v_proj. The family declares it
    # rather than setting it afterwards, because the global key count is read
    # under it: gemma4 spells the same regime as an attention_k_eq_v field,
    # and DiffusionGemma carries the behaviour in its modules instead.
    config = gemma4_config(hf_config, used, k_eq_v=True)
    if config.get("mixture") is None and all(
            hf_config.get(field) is not None
            for field in ("num_experts", "top_k_experts", "moe_intermediate_size")):
        # DiffusionGemma names no enable_moe_block flag; a config carrying the
        # three routed widths routes every layer beside its dense MLP, which
        # is what its encoder and decoder layers both build.
        config["mixture"] = parallel_experts(hf_config)
    # final_logit_softcapping is a class attribute of the reference text
    # config, not an instance field a config.json carries; the head always
    # divides by 30 under tanh.
    config["final_logit_softcap"] = 30.0
    config.update(causal=False)
    return config


# LLaDA's own names beside the llama-layout spellings the shared map reads.
# The computation matches the llama block (pre-norm RMS, rotate-half rope,
# SwiGLU, untied head) and only the names differ.
_LLADA_NAMES: Renames = (
    ('model.transformer.wte', 'model.embed_tokens'), ('model.transformer.ln_f', 'model.norm'),
    ('model.transformer.ff_out', 'lm_head'), ('model.transformer.blocks', 'model.layers'),
    ('attn_norm', 'input_layernorm'), ('attn_out', 'self_attn.o_proj'),
    ('ff_norm', 'post_attention_layernorm'), ('ff_proj', 'mlp.gate_proj'), ('up_proj', 'mlp.up_proj'),
    ('ff_out', 'mlp.down_proj'), ('q_proj', 'self_attn.q_proj'), ('k_proj', 'self_attn.k_proj'),
    ('v_proj', 'self_attn.v_proj'))


def _llada_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Return the variables-tree path for one LLaDA tensor name, every one of
    which the release nests under `model.transformer.`."""
    if not name.startswith('model.transformer.'):
        raise ValueError(f'unknown tensor name {name!r}')
    return renamed_path(_LLADA_NAMES, name, config)


def _mask_token_export(model: CausalTransformer) -> Mapping[str, object]:
    """Return the config field a masked-diffusion export writes: its mask id.

    LLaDA and Dream both declare the reserved id and nothing else of their own.
    """
    return {'mask_token_id': model.mask_token_id}


def _diffusion_gemma_export(model: CausalTransformer) -> Mapping[str, object]:
    raise ValueError(
        "diffusion_gemma_text is a cache-reading view; export the complete native DiffusionGemma wrapper"
    )


DIFFUSION_GEMMA_TEXT = DecoderFamily(
    ('diffusion_gemma_text',),
    _diffusion_gemma_text_config,
    lambda fields: bool(
        fields.causal is False
        and (
            fields.v_norm
            or fields.per_layer_input_dim
            or fields.kv_shared_layers
        )
    ),
    'diffusion_gemma_text',
    'DiffusionGemmaForBlockDiffusion',
    _diffusion_gemma_export,
    sandwich_norms=True,
    weight_path=gemma4_path,
    export_path=gemma4_export_path,
    packed=FUSED_EXPERTS,
    export_weights=gemma4_export_weights,
    preserve_source_layout=False,
)

DREAM = DecoderFamily(
    ('dream', 'Dream'),
    _dream_config,
    lambda fields: bool(
        fields.causal is False
        and fields.attention_bias
        and fields.o_proj_bias is False
    ),
    'dream',
    'DreamModel',
    _mask_token_export,
    preserve_source_layout=True,
)

LLADA = DecoderFamily(
    ('llada',),
    _llada_config,
    lambda fields: bool(
        fields.causal is False
        and not fields.attention_bias
        and fields.mixture is None
        and not (
            fields.v_norm
            or fields.per_layer_input_dim
            or fields.kv_shared_layers
        )
        and not fields.output_gate
        and not fields.qk_norm
    ),
    'llada',
    'LLaDAModelLM',
    _mask_token_export,
    weight_path=_llada_path,
    export_path=partial(renamed_name, _LLADA_NAMES),
    preserve_source_layout=True,
)
