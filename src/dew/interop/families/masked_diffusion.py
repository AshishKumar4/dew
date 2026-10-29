"""Translate the bidirectional LLaDA, Dream and DiffusionGemma decoders.

LLaDA's dense and routed releases configure the shared GQA and expert
modules. Dream rides Qwen2's layout and DiffusionGemma Gemma 4's; the
adapters below own only their configuration and tensor spellings.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import _records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    FUSED_EXPERTS,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    MaskedWalk,
    Packed,
    Renames,
    Ropes,
    RopeSpelling,
    base_config,
    decoder_tensors,
    plain_partial_rope,
    qkv_widths,
    record_float,
    record_int,
    refuse,
    renamed_name,
    renamed_path,
)
from dew.interop.families.deepseek import deepseek_mixture
from dew.interop.families.gemma import (
    GEMMA4_DEFAULTS,
    gemma4_config,
    gemma4_export_path,
    gemma4_export_weights,
    gemma4_path,
    parallel_experts,
)
from dew.interop.families.qwen import qwen2_config
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture

# The 7B tokenizer reserves <|mask|> here; LLaDA2's generate also defaults to it.
_MOE_MASK_ID = 156895


def _mask_token(hf_config: Mapping[str, object], used: set[str]) -> int:
    """Return the mask id a masked-diffusion release reserves, under either spelling.

    The releases write it as `mask_token_id` or as `mask_id`.
    """
    mask = hf_config.get('mask_token_id', hf_config.get('mask_id'))
    if mask is None:
        refuse('mask_token_id', 'a masked diffusion checkpoint reserves its mask id')
    used.update(('mask_token_id', 'mask_id'))
    return _records.integer(mask, 'mask_token_id/mask_id')


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
    if 'num_experts' in hf_config:
        return _llada_moe_config(hf_config, used)
    std = _llada_geometry(hf_config)
    layers = _records.integer(std['num_hidden_layers'], 'num_hidden_layers/n_layers')
    config = base_config(std, set(), layer_types=('full_attention',) * layers,
                          rope=Ropes(_records.number(std['rope_theta'], 'rope_theta')))
    used.update(key for aliases in _LLADA_ALIASES for key in aliases if key in hf_config)
    mask = _mask_token(hf_config, used)
    _llada_refusals(hf_config, used, std)
    config.update(causal=False, mask_token_id=mask)
    return config


# The standard spelling wins over the release's aliases, but every present
# spelling has been read when the generic reader accounts for source keys.
_LLADA_ALIASES = (
    ('hidden_size', 'd_model'),
    ('num_hidden_layers', 'n_layers', 'num_layers'),
    ('num_attention_heads', 'n_heads'),
    ('num_key_value_heads', 'n_kv_heads'),
    ('head_dim',),
    ('intermediate_size', 'mlp_hidden_size'),
    ('vocab_size', 'embedding_size'),
    ('max_position_embeddings', 'max_sequence_length'),
    ('rms_norm_eps', 'norm_eps'),
    ('rope_theta',),
    ('attention_bias', 'include_bias', 'include_qkv_bias'),
    ('tie_word_embeddings', 'weight_tying'),
    ('hidden_act', 'hidden_activation', 'activation_type'),
)


def _llada_geometry(hf_config: Mapping[str, object]) -> Mapping[str, object]:
    """Rewrite LLaDA's own config spellings under the standard names `base_config` reads.

    The first present spelling in each alias group wins, even when its
    value is null, as the release's nested get calls do.
    """
    values: dict[str, object] = {}
    for aliases in _LLADA_ALIASES:
        for key in aliases:
            if key in hf_config:
                values[aliases[0]] = hf_config[key]
                break
    hidden = values.get('hidden_size')
    layers = values.get('num_hidden_layers')
    heads = values.get('num_attention_heads')
    kv_heads = values.get('num_key_value_heads')
    intermediate = values.get('intermediate_size')
    vocab = values.get('vocab_size')
    if hidden is None or layers is None or heads is None or intermediate is None or vocab is None:
        refuse('llada geometry',
                'd_model/hidden_size, n_layers/num_hidden_layers, n_heads/num_attention_heads, '
                'mlp_hidden_size/intermediate_size and embedding_size/vocab_size are required')
    kv_heads = heads if kv_heads is None else kv_heads
    head_dim = values.get('head_dim')
    head_dim = (
        _records.integer(head_dim, "head_dim")
        if head_dim is not None
        else _records.integer(hidden, "hidden_size/d_model")
        // _records.integer(heads, "num_attention_heads/n_heads")
    )
    max_pos = values.get('max_position_embeddings', DEFAULT_MAX_SEQ_LEN)
    std: dict[str, object] = {
        "hidden_size": _records.integer(hidden, "hidden_size/d_model"),
        "num_attention_heads": _records.integer(heads, "num_attention_heads/n_heads"),
        "num_key_value_heads": _records.integer(kv_heads, "num_key_value_heads/n_kv_heads"),
        "head_dim": head_dim,
        "intermediate_size": _records.integer(intermediate, "intermediate_size/mlp_hidden_size"),
        "vocab_size": _records.integer(vocab, "vocab_size/embedding_size"),
        "num_hidden_layers": _records.integer(layers, "num_hidden_layers/n_layers"),
        "max_position_embeddings": min(
            _records.integer(max_pos, "max_position_embeddings/max_sequence_length"), DEFAULT_MAX_SEQ_LEN
        ),
        "rms_norm_eps": _records.number(
            values.get("rms_norm_eps", 1e-6), "rms_norm_eps"
        ),
        # The pinned LLaDA-8B's ModelConfig, which LLaDAConfig starts from, sets 10000; the release
        # states its own 500000.
        "rope_theta": _records.number(values.get("rope_theta", 10000.0), "rope_theta"),
        "attention_bias": bool(values.get("attention_bias", False)),
        "tie_word_embeddings": bool(values.get("tie_word_embeddings", False)),
        "hidden_act": values.get("hidden_act", "silu"),
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
    layers = _records.integer(std['num_hidden_layers'], 'num_hidden_layers/n_layers')
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
    if (stated is not None and _records.strings(stated, 'layer_types')
            != ('full_attention',) * layers):
        refuse(f'layer_types {list(_records.strings(stated, "layer_types"))!r}',
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
    if _records.integer(hf_config.get('block_group_size', 1), 'block_group_size') != 1:
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
    if embedding is not None and _records.integer(embedding, "embedding_size") != _records.integer(
        vocab, "vocab_size"
    ):
        refuse('embedding_size', f'it names {embedding} rows for a {vocab} vocabulary')


def _moe_decoder(hf_config: Mapping[str, object], used: set[str], *, qk_norm: bool, intermediate: int,
                 rope_defaults: tuple[RopeSpelling, float] | None) -> DecoderFields:
    """The released MoE decoders' shared pre-norm GQA and sliced rotate-half rope.

    `rope_defaults` is the config class's base and factor for a config that states neither, or None
    for a class whose defaults are no rotary at all: LLaDA-MoE's and LLaDA-MoE v2's default both to
    -1 (configuration_lladamoe.py:32-33), so their configs state them."""
    if hf_config.get('rope_scaling') is not None:
        refuse('rope_scaling', 'the released MoE checkpoints use plain rotary positions')
    spelling, rotary = (None, None) if rope_defaults is None else rope_defaults
    theta, factor = plain_partial_rope(hf_config, spelling, default_factor=rotary,
                                       reference='LLaDA MoE rotary')
    used.update(('rope_theta', 'rope_scaling', 'rope_parameters', 'partial_rotary_factor'))
    layers = record_int(hf_config, 'num_hidden_layers')
    config = base_config({'rms_norm_eps': 1e-5, **hf_config, 'intermediate_size': intermediate}, used,
                          rope=Ropes(theta), qk_norm=qk_norm,
                          layer_types=('full_attention',) * layers,
                          reads=frozenset({'attention_bias'}))
    config.update(causal=False, partial_rotary_factor=None if factor == 1 else factor,
                  partial_rotary_type='default',
                  mask_token_id=_mask_token({
                      'mask_token_id': hf_config.get('mask_id', _MOE_MASK_ID), **hf_config}, used))
    return config


def _llada_moe_config(hf_config: Mapping[str, object], used: set[str], *,
                       v2: bool = False) -> DecoderFields:
    """OLMoE's softmax experts, per-head QK norms and optional shared branch.

    The 7B code fixes norm_topk_prob=False and never reads its scaling;
    v2 softmaxes the selected logits and scales them. Its grouped router
    class is not the class its DecoderLayer instantiates.

    Both tables pad config.pad_token_id (modeling_lladamoe.py:895-898 at
    daccaf1, modeling_lladamoe_v2.py:1094-1097 at 2a513f7), which both
    config classes default to 1. Read here rather than by the family, since
    model_type 'llada' also names dense LLaDA, whose table pads nothing.
    """
    config = _moe_decoder(hf_config, used, qk_norm=bool(hf_config.get('qk_layernorm')),
                           intermediate=record_int(hf_config, 'dense_intermediate_size'), rope_defaults=None)
    pad = hf_config.get('pad_token_id', 1)
    config['padding_idx'] = None if pad is None else _records.integer(pad, 'pad_token_id')
    used.update(('dense_intermediate_size', 'expert_intermediate_size',
                 'shared_expert_intermediate_size', 'moe_layer_freq', 'qk_layernorm',
                 'clip_qkv', 'num_experts', 'num_experts_per_tok', 'norm_topk_prob',
                 'moe_router_enable_expert_bias', 'moe_router_score_function',
                 'routed_scaling_factor', 'router_num_group', 'router_topk_group',
                 'output_router_logits', 'router_aux_loss_coef'))
    if hf_config.get('clip_qkv') is not None:
        refuse('clip_qkv', 'clipping normalized queries, keys and values is not configured')
    if v2 and hf_config.get('moe_router_enable_expert_bias'):
        refuse('moe_router_enable_expert_bias',
                'the released v2 SparseMoeBlock reads routing_weights before assigning it')
    layers = record_int(hf_config, 'num_hidden_layers')
    pattern = _records.integers(hf_config['moe_layer_freq'], 'moe_layer_freq')
    if len(pattern) < layers:
        refuse('moe_layer_freq', 'it must name each decoder layer')
    routed = tuple(index for index, enabled in enumerate(pattern[:layers]) if enabled != 0)
    if routed:
        bias = bool(hf_config.get('moe_router_enable_expert_bias'))
        # The release defaults to -1; only an explicit null disables this branch.
        shared = hf_config.get('shared_expert_intermediate_size', -1)
        config['mixture'] = native_fields(Mixture)(
            experts=record_int(hf_config, 'num_experts'),
            top_k=record_int(hf_config, 'num_experts_per_tok'), layers=routed,
            expert_features=record_int(hf_config, 'expert_intermediate_size'),
            shared_features=_records.integer(shared, 'shared_expert_intermediate_size')
            if shared is not None else 0,
            norm_topk_prob=v2, scaling=(record_float(hf_config, 'routed_scaling_factor', 1.0)
                                       or 1.0) if v2 else 1.0,
            bias=bias, bias_in_weights=bias)
    return config


def _llada_moe_v2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    return _llada_moe_config(hf_config, used, v2=True)


def _llada2_walk(config: Mapping[str, object]) -> MaskedWalk | None:
    """LLaDA2's own generation, where Dew runs it: 2.0's `generate` walks blocks of 32 over 32 steps
    each at temperature 0, revealing past 0.95 (modeling_llada2_moe.py:1233-1247 at dad945c). 2.1's
    configs read as 2.0's and take the same walk, without the release's editing of revealed tokens,
    which Dew does not run; 2.2's (its DELETE and SPLIT ids) generates with editing past another
    threshold, and unmasks the whole response with Dew's MDLM instead."""
    if config.get("delete_token_id") is not None or config.get("split_token_id") is not None:
        return None
    return MaskedWalk(block=32, steps=32, threshold=0.95, temperature=0.0)


def _llada2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """DeepSeek's grouped sigmoid router, or 2.2's block-level expert shortlist.

    The 2.0 forward requires a 4-D additive block mask. 2.1 and 2.2 accept
    it through create_bidirectional_mask too. Without an explicit mask Dew
    exposes their full bidirectional view, as the other masked LMs do;
    a 2.0 or 2.1 source generates by 2.0's blockwise walk (`_llada2_walk`), a
    2.2 one by Dew's MDLM procedure.

    2.2's block routing departs from the release in one respect: the release
    cuts its `block_size` blocks from the batch flattened into one stream (and
    asserts whole blocks), so a row's experts there depend on the rows beside
    it and a packed document's on its neighbours. Dew routes each request and
    each packed document by its own blocks of real tokens, padding voting for
    none (`dew.nn.moe.Router`), which is the release's routing wherever a row
    is a whole number of blocks of real tokens: one request, or rows that
    many blocks long (tools/llada_moe_reference.py's recorded draws).
    """
    block = (record_int(hf_config, 'block_size', 32)
             if 'block_size' in hf_config or 'expert_capacity' in hf_config else None)
    config = _moe_decoder(hf_config, used, qk_norm=bool(hf_config.get('use_qk_norm', block is None)),
                           intermediate=record_int(hf_config, 'intermediate_size'),
                           # LLaDA2MoeConfig's defaults (configuration_llada2_moe.py:29, :46).
                           rope_defaults=(PLAIN_SPELLING, 0.5))
    config.update(attention_bias=bool(hf_config.get('use_qkv_bias', False)),
                  o_proj_bias=bool(hf_config.get('use_bias', True)))
    used.update(('use_qkv_bias', 'use_qk_norm', 'use_bias', 'num_experts', 'num_shared_experts',
                 'n_group', 'topk_group', 'moe_intermediate_size', 'first_k_dense_replace',
                 'num_experts_per_tok', 'routed_scaling_factor', 'output_router_logits',
                 'block_size', 'expert_capacity', 'sliding_window',
                 'moe_router_enable_expert_bias', 'norm_head', 'norm_softmax', 'norm_topk_prob',
                 'router_dtype', 'score_function', 'rotary_dim', 'use_rmsnorm',
                 'using_split_qkv_in_self_attention', 'embedding_dropout', 'output_dropout',
                 'delete_token_id', 'split_token_id'))
    if hf_config.get('num_experts') is None:
        return config
    # These names differ, but the inference gate and dense-first layout are DeepSeek V3's.
    mixture = deepseek_mixture({
        **hf_config, 'n_routed_experts': hf_config['num_experts'],
        'n_shared_experts': hf_config.get('num_shared_experts'), 'norm_topk_prob': True,
        'n_group': 1 if block is not None else hf_config.get('n_group', 8),
        'topk_group': 1 if block is not None else hf_config.get('topk_group', 4),
    }, record_int(hf_config, 'num_hidden_layers'), set())
    mixture['norm_topk_prob'] = record_int(hf_config, 'num_experts_per_tok') > 1
    if block is not None:
        mixture.update(routing_block=block,
                       block_experts=record_int(hf_config, 'expert_capacity', 48))
    if mixture['layers']:
        config['mixture'] = mixture
    return config


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
    # Dream continues Qwen2.5's next-token head, so its reference reads position i from logits i-1
    # (`torch.cat([logits[:, :1], logits[:, :-1]], dim=1)` in its generation and training code).
    config.update(causal=False, mask_token_id=mask, shifted_logits=True)
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


_LLADA_MOE_NAMES: Renames = (
    ('shared_expert', 'mlp.shared_experts'),
    ('mlp.expert_bias', 'mlp.gate.e_score_correction_bias'))
_LLADA_MOE_NULLS = ('shared_expert_intermediate_size',)
_LLADA2_NAMES: Renames = (
    ('model.word_embeddings', 'model.embed_tokens'),
    ('attention.query_layernorm', 'self_attn.q_norm'),
    ('attention.key_layernorm', 'self_attn.k_norm'),
    ('attention.dense', 'self_attn.o_proj'), ('attention', 'self_attn'),
    ('mlp.gate.expert_bias', 'mlp.gate.e_score_correction_bias'))
_LLADA2_PACKED = tuple(
    Packed(f'.attention.query_key_value.{leaf}',
           tuple(f'.attention.{part}.{leaf}' for part in ('q_proj', 'k_proj', 'v_proj')),
           0, widths=qkv_widths)
    for leaf in ('weight', 'bias'))


def _llada_path(name: str, config: Mapping[str, object]) -> tuple[str, ...] | None:
    """Read dense LLaDA's OLMo envelope or the MoE release's decoder names."""
    return renamed_path(_LLADA_NAMES if name.startswith('model.transformer.') else _LLADA_MOE_NAMES,
                         name, config)


def _llada_export_path(name: str, config: Mapping[str, object]) -> str | None:
    names = _LLADA_MOE_NAMES if 'num_experts' in config else _LLADA_NAMES
    return renamed_name(names, name, config)


def _llada_moe_export(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        return _mask_token_export(model)
    return {
        **_mask_token_export(model),
        **({} if mixture.norm_topk_prob else {'architectures': ['LLaDAMoEModelLM']}),
        'dense_intermediate_size': model.hidden_features,
        'expert_intermediate_size': mixture.expert_features or model.hidden_features,
        'shared_expert_intermediate_size': mixture.shared_features or None,
        'num_experts': mixture.experts, 'num_experts_per_tok': mixture.top_k,
        'moe_layer_freq': [int(mixture.layers is None or index in mixture.layers)
                           for index in range(model.num_layers)],
        'moe_router_enable_expert_bias': mixture.bias,
        'moe_router_score_function': 'softmax', 'norm_topk_prob': mixture.norm_topk_prob,
        'routed_scaling_factor': mixture.scaling, 'qk_layernorm': model.qk_norm,
        'partial_rotary_factor': model.partial_rotary_factor or 1.0,
        'pad_token_id': model.padding_idx,
    }


def _llada2_export(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        refuse('mixture', 'the LLaDA2 source format describes a routed decoder')
    layers = mixture.layers if mixture.layers is not None else tuple(range(model.num_layers))
    first = layers[0]
    if layers != tuple(range(first, model.num_layers)):
        refuse('mixture.layers', 'LLaDA2 routes every layer after first_k_dense_replace')
    width = mixture.expert_features or model.hidden_features
    if mixture.shared_features % width:
        refuse('mixture.shared_features', 'LLaDA2 counts whole shared experts of the routed width')
    return {
        **_mask_token_export(model), 'num_experts': mixture.experts,
        'num_experts_per_tok': mixture.top_k,
        'num_shared_experts': mixture.shared_features // width or None,
        'moe_intermediate_size': width, 'first_k_dense_replace': first,
        'routed_scaling_factor': mixture.scaling, 'n_group': mixture.groups,
        'topk_group': mixture.groups_per_token,
        'use_qkv_bias': model.attention_bias,
        'use_bias': model.attention_bias if model.o_proj_bias is None else model.o_proj_bias,
        'use_qk_norm': model.qk_norm,
        'partial_rotary_factor': model.partial_rotary_factor or 1.0,
        **({} if mixture.routing_block is None else {
            'block_size': mixture.routing_block, 'expert_capacity': mixture.block_experts}),
    }


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
    bidirectional=True,
    defaults=GEMMA4_DEFAULTS,
    padding_idx=True,
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
    bidirectional=True,
    # DreamBaseModel's table pads config.pad_token_id (modeling_dream.py:602-605 at 6572adb), whose
    # DreamConfig default is 151643.
    defaults={'pad_token_id': 151643},
    padding_idx=True,
)

LLADA = DecoderFamily(
    ('llada',),
    _llada_config,
    lambda fields: bool(
        fields.causal is False
        and not fields.attention_bias
        and not (
            fields.v_norm
            or fields.per_layer_input_dim
            or fields.kv_shared_layers
        )
        and not fields.output_gate
        and (not fields.qk_norm or fields.mixture is not None)
    ),
    'llada',
    'LLaDAModelLM',
    _llada_moe_export,
    weight_path=_llada_path,
    export_path=_llada_export_path,
    export_weights=decoder_tensors,
    export_nulls=_LLADA_MOE_NULLS,
    preserve_source_layout=True,
    bidirectional=True,
)

LLADA_MOE_V2 = DecoderFamily(
    ('llada_moe_v2',), _llada_moe_v2_config,
    lambda fields: bool(not fields.causal and (mixture := fields.mixture) is not None
                        and mixture.score_function == 'softmax' and mixture.norm_topk_prob),
    'llada_moe_v2', 'LLaDAMoEv2ModelLM', _llada_moe_export,
    weight_path=partial(renamed_path, _LLADA_MOE_NAMES),
    export_path=partial(renamed_name, _LLADA_MOE_NAMES), export_weights=decoder_tensors,
    export_nulls=_LLADA_MOE_NULLS,
    preserve_source_layout=True,
    bidirectional=True,
)

LLADA2_MOE = DecoderFamily(
    ('llada2_moe',), _llada2_config,
    lambda fields: bool(not fields.causal and (mixture := fields.mixture) is not None
                        and mixture.score_function == 'sigmoid'),
    'llada2_moe', 'LLaDA2MoeModelLM', _llada2_export,
    weight_path=partial(renamed_path, _LLADA2_NAMES),
    export_path=partial(renamed_name, _LLADA2_NAMES), export_weights=decoder_tensors,
    packed=_LLADA2_PACKED, tied_head_names=('lm_head.weight', 'model.word_embeddings.weight'),
    export_nulls=('num_shared_experts',),
    preserve_source_layout=True,
    bidirectional=True,
    # LLaDA2MoeModel's table pads config.pad_token_id (modeling_llada2_moe.py:750-754 of LLaDA2.0 at
    # dad945c, :761-765 of 2.1 at 20e64e2, :770-774 of 2.2 at 9b78da3), whose LLaDA2MoeConfig default
    # is 126081.
    defaults={'pad_token_id': 126081},
    padding_idx=True,
    masked_walk=_llada2_walk,
)
