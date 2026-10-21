"""Translate the bidirectional decoders: LLaDA, LLaDA-MoE, LLaDA2, Dream and DiffusionGemma's text half.

Each is a causal family's block with the mask replaced: full attention in
place of the causal mask, or LLaDA2's blocks, and a mask token the sampler
denoises to. Dream rides the qwen2 layout and DiffusionGemma the Gemma 4 one;
LLaDA's OLMo-style tensor names, LLaDA-MoE's layer-level shared expert and
LLaDA2's Ling-style names each take a map of their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from dew import records
from dew.interop.config_records import native_fields
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    FUSED_EXPERTS,
    DecoderFamily,
    DecoderFields,
    MixtureFields,
    Packed,
    Renames,
    Ropes,
    base_config,
    qkv_widths,
    read_rope,
    refuse,
    renamed_name,
    renamed_path,
    softmax_top_k,
)
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

    inclusionAI/LLaDA-MoE-7B-A1B releases its MoE under the same model_type
    with a config class of its own (configuration_lladamoe.LLaDAConfig),
    which always names its experts, so a config with `num_experts` is read as
    that release (`_llada_moe_config`).
    """
    if 'num_experts' in hf_config:
        return _llada_moe_config(hf_config, used, renormalized=False)
    std = _llada_geometry(hf_config)
    layers = records.integer(std['num_hidden_layers'], 'num_hidden_layers/n_layers')
    config = base_config(std, set(), layer_types=('full_attention',) * layers,
                          rope=Ropes(records.number(std['rope_theta'], 'rope_theta')))
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
        records.integer(head_dim, "head_dim")
        if head_dim is not None
        else records.integer(hidden, "hidden_size/d_model")
        // records.integer(heads, "num_attention_heads/n_heads")
    )
    max_pos = values.get('max_position_embeddings', DEFAULT_MAX_SEQ_LEN)
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
            values.get("rms_norm_eps", 1e-6), "rms_norm_eps"
        ),
        "rope_theta": records.number(values.get("rope_theta", 500000.0), "rope_theta"),
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


_RELEASED_MASK = 156895
"""The `<|mask|>` id of the vocabulary LLaDA-MoE and LLaDA2 share: LLaDAMoEv2Config's mask_token_id
default, and the mask_id both families' own samplers default to (LLaDA-MoE-7B-A1B's README
`generate`, LLaDA2MoeModelLM.generate). LLaDA-MoE-7B-A1B's and LLaDA2's configs name none."""


def _llada_moe_config(hf_config: Mapping[str, object], used: set[str], *,
                      renormalized: bool) -> DecoderFields:
    """Read a LLaDA-MoE config into `CausalTransformer` fields.

    inclusionAI/LLaDA-MoE-7B-A1B (model_type 'llada', modeling_lladamoe.py) and
    GSAI-ML/LLaDA-MoE-v2-30B-A3B ('llada_moe_v2', modeling_lladamoe_v2.py) compute OLMoE's block with
    no causal mask (the attention's is_causal False, the decoder layer's mask None): q/k RMSNorm per
    head before a rotate-half rotary over the first partial_rotary_factor of each head (transformers'
    default rope), SwiGLU, a dense MLP dense_intermediate_size wide where moe_layer_freq is 0, routed
    experts elsewhere, and an ungated shared expert beside them where
    shared_expert_intermediate_size is set.

    They route differently, and neither reads most of the routing fields its config carries
    (moe_router_score_function, router_num_group, router_topk_group): 7B-A1B weighs its top-k by the
    softmax over every expert, unnormalized and unscaled (LLaDAMoESparseMoeBlock fixes
    norm_topk_prob False); v2, `renormalized`, softmaxes its top-k logits alone and scales them by a
    nonzero routed_scaling_factor (LLaDAMoEv2SparseMoeBlock). The expert bias neither release
    enables adds to 7B-A1B's routing weights themselves and is unbound in v2's forward, so it is
    refused.
    """
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    dense = records.integer(hf_config.get('dense_intermediate_size', -1), 'dense_intermediate_size')
    if dense < 1:
        refuse('dense_intermediate_size', 'the dense MLP needs its width; the config class leaves it at -1')
    heads = records.integer(hf_config['num_attention_heads'], 'num_attention_heads')
    width = records.integer(hf_config['hidden_size'], 'hidden_size') // heads
    if not renormalized and hf_config.get('head_dim') not in (None, width):
        # 7B-A1B's attention splits hidden_size // num_attention_heads whatever head_dim says, and
        # its rotary reads head_dim.
        refuse(f"head_dim {hf_config['head_dim']!r}", f'LLaDA-MoE-7B-A1B attends {width}-wide heads')
    # Both config classes default rms_norm_eps to 1e-05 where the shared reader takes 1e-06.
    shared_geometry = {**hf_config, 'intermediate_size': dense,
                       'rms_norm_eps': hf_config.get('rms_norm_eps', 1e-5)}
    config = base_config(shared_geometry, used, rope=read_rope(hf_config, used),
                         qk_norm=bool(hf_config.get('qk_layernorm')), reads=frozenset({'attention_bias'}))
    factor = records.number(hf_config.get('partial_rotary_factor', -1), 'partial_rotary_factor')
    if not 0 < factor <= 1:
        refuse(f'partial_rotary_factor {factor!r}', 'the rotary turns a share of each head in (0, 1]; the '
               'config class leaves it at -1')
    if hf_config.get('clip_qkv') is not None:
        refuse('clip_qkv', 'the projections are not clamped here')
    if hf_config.get('moe_router_enable_expert_bias'):
        refuse('moe_router_enable_expert_bias', 'no release enables it, and it would add to the routing '
               'weights rather than choose by them')
    stated = hf_config.get('moe_layer_freq')
    if not isinstance(stated, list) or len(stated) < layers:
        refuse('moe_layer_freq', f'the reference reads one entry a layer, 0 for a dense one, of {layers}')
    sparse = tuple(index for index in range(layers)
                   if records.integer(stated[index], 'moe_layer_freq') != 0)
    if not sparse:
        refuse('moe_layer_freq', 'it routes no layer, which no LLaDA-MoE release builds')
    shared = hf_config.get('shared_expert_intermediate_size')
    if shared is not None and records.integer(shared, 'shared_expert_intermediate_size') < 1:
        refuse('shared_expert_intermediate_size', 'the shared expert needs its width, or null for none')
    scaling = records.number(hf_config.get('routed_scaling_factor') or 1.0, 'routed_scaling_factor')
    used.update(('dense_intermediate_size', 'partial_rotary_factor', 'qk_layernorm', 'clip_qkv',
                 'moe_router_enable_expert_bias', 'moe_layer_freq', 'shared_expert_intermediate_size',
                 'num_experts', 'expert_intermediate_size', 'routed_scaling_factor', 'norm_topk_prob',
                 'moe_router_score_function', 'router_num_group', 'router_topk_group', 'mask_token_id'))
    config.update(
        partial_rotary_factor=None if factor == 1 else factor, partial_rotary_type='default',
        mixture=native_fields(Mixture)(
            experts=records.integer(hf_config['num_experts'], 'num_experts'),
            top_k=softmax_top_k(hf_config, used), layers=sparse, score_function='softmax',
            norm_topk_prob=renormalized, scaling=scaling if renormalized else 1.0,
            expert_features=records.integer(hf_config['expert_intermediate_size'],
                                            'expert_intermediate_size'),
            shared_features=(0 if shared is None
                             else records.integer(shared, 'shared_expert_intermediate_size'))),
        causal=False,
        mask_token_id=records.integer(hf_config.get('mask_token_id', _RELEASED_MASK), 'mask_token_id'))
    return config


_LLADA2_BLOCK = 32
"""The block LLaDA2 attends and denoises by: LLaDA2MoeModelLM.generate's block_length default
(modeling_llada2_moe.py:1237 of LLaDA2.0-mini), which every release's README samples with."""


def _llada2_config(hf_config: Mapping[str, object], used: set[str]) -> DecoderFields:
    """Read a LLaDA2.x config (model_type 'llada2_moe') into `CausalTransformer` fields.

    inclusionAI's LLaDA2.0-mini/-flash, LLaDA2.1-mini/-flash and LLaDA2.2-flash compute Ling 2.0's
    block (modeling_llada2_moe.py): one query_key_value projection stacking every query head, then
    every key and value head; q/k RMSNorm per head before a rotate-half rotary over the first
    partial_rotary_factor of each head (transformers' default rope); the output projection `dense`;
    a dense MLP on the first first_k_dense_replace layers and routed SwiGLU experts after them beside
    num_shared_experts' ungated MLP. Each token chooses by its sigmoid scores plus the gate's
    expert_bias and weighs by its unbiased scores, normalized past one expert and scaled by
    routed_scaling_factor (LLaDA2MoeGate). 2.0 and 2.1 keep a token to its topk_group best of n_group
    groups, each scored by its two best experts; 2.2 has each block of block_size tokens choose
    expert_capacity experts together instead (block_routing), reading no group field.

    The model attends in blocks: its generate cuts the row into blocks of 32 from the prompt's first
    token, lets each block read itself and the blocks before it (block_diffusion_attention_mask)
    and denoises them in order, so the record carries that block (`diffusion_block`).

    The release's model reads none of score_function, router_dtype, moe_router_enable_expert_bias,
    norm_topk_prob, norm_softmax, norm_head, use_rmsnorm, using_split_qkv_in_self_attention,
    rotary_dim, sliding_window (which only flash attention, which the class refuses, would apply) or
    the dropouts: it fixes a sigmoid fp32 router with its bias, RMSNorm and the fused projection
    whatever they say. delete_token_id and split_token_id are 2.2's sampler's edit tokens.
    """
    if not hf_config.get('num_key_value_heads'):
        refuse('num_key_value_heads', 'LLaDA2MoeConfig leaves it at 0, which no head divides into')
    # LLaDA2.0's and 2.1's config class norms queries and keys unless told not to, and 2.2's, which
    # adds block_size and expert_capacity, only when told to.
    qk_norm = bool(hf_config.get('use_qk_norm', 'block_size' not in hf_config))
    config = base_config(hf_config, used, rope=read_rope(hf_config, used), qk_norm=qk_norm,
                         reads=frozenset())
    head_dim = records.integer(config['head_dim'], 'head_dim')
    factor = records.number(hf_config.get('partial_rotary_factor', 0.5), 'partial_rotary_factor')
    if not 0 < factor <= 1 or int(head_dim * factor) % 2:
        refuse(f'partial_rotary_factor {factor!r}',
               f'the rotary turns an even share of a {head_dim}-wide head')
    layers = records.integer(hf_config['num_hidden_layers'], 'num_hidden_layers')
    first_k = records.integer(hf_config.get('first_k_dense_replace', 0), 'first_k_dense_replace')
    experts = hf_config.get('num_experts', 16)
    sparse = () if experts is None else tuple(range(min(first_k, layers), layers))
    used.update(('use_qk_norm', 'partial_rotary_factor', 'use_qkv_bias', 'use_bias', 'first_k_dense_replace',
                 'num_experts', 'num_experts_per_tok', 'num_shared_experts', 'moe_intermediate_size',
                 'routed_scaling_factor', 'n_group', 'topk_group', 'block_size', 'expert_capacity',
                 'score_function', 'router_dtype', 'moe_router_enable_expert_bias', 'norm_topk_prob',
                 'norm_softmax', 'norm_head', 'use_rmsnorm', 'using_split_qkv_in_self_attention',
                 'rotary_dim', 'sliding_window', 'embedding_dropout', 'output_dropout',
                 'output_router_logits', 'router_aux_loss_coef', 'delete_token_id', 'split_token_id',
                 'mask_token_id'))
    config.update(
        partial_rotary_factor=None if factor == 1 else factor, partial_rotary_type='default',
        attention_bias=bool(hf_config.get('use_qkv_bias', False)),
        o_proj_bias=bool(hf_config.get('use_bias', True)),
        mixture=(_llada2_mixture(hf_config, records.integer(experts, 'num_experts'), sparse)
                 if sparse else None),
        causal=False, diffusion_block=_LLADA2_BLOCK,
        mask_token_id=records.integer(hf_config.get('mask_token_id', _RELEASED_MASK), 'mask_token_id'))
    return config


def _llada2_mixture(hf_config: Mapping[str, object], experts: int,
                    sparse: tuple[int, ...]) -> MixtureFields:
    """LLaDA2's routed experts: groups for 2.0 and 2.1, blocks for 2.2 (`_llada2_config`)."""
    top_k = records.integer(hf_config.get('num_experts_per_tok', 2), 'num_experts_per_tok')
    width = hf_config.get('moe_intermediate_size')
    if width is None:
        refuse('moe_intermediate_size', 'the routed experts need their width')
    expert_width = records.integer(width, 'moe_intermediate_size')
    shared = records.integer(hf_config.get('num_shared_experts', 0) or 0, 'num_shared_experts')
    blocks = hf_config.get('block_size')
    by_block = blocks is not None
    return native_fields(Mixture)(
        experts=experts, top_k=top_k, layers=sparse, score_function='sigmoid', norm_topk_prob=top_k > 1,
        scaling=records.number(hf_config.get('routed_scaling_factor', 2.5), 'routed_scaling_factor'),
        bias=True, expert_features=expert_width, shared_features=shared * expert_width,
        groups=1 if by_block else records.integer(hf_config.get('n_group', 8), 'n_group'),
        groups_per_token=1 if by_block else records.integer(hf_config.get('topk_group', 4), 'topk_group'),
        routing_block=records.integer(blocks, 'block_size') if by_block else None,
        block_experts=(records.integer(hf_config.get('expert_capacity', 48), 'expert_capacity')
                       if by_block else None))


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
    which the dense release nests under `model.transformer.`. LLaDA-MoE-7B-A1B,
    released under the same model_type, names its tensors OLMoE's way."""
    if config.get('mixture') is not None:
        return renamed_path(_LLADA_MOE_NAMES, name, config)
    if not name.startswith('model.transformer.'):
        raise ValueError(f'unknown tensor name {name!r}')
    return renamed_path(_LLADA_NAMES, name, config)


# LLaDA-MoE's shared expert sits beside its layer's routed block, where the
# shared map reads it inside that block.
_LLADA_MOE_NAMES: Renames = (('shared_expert', 'mlp.shared_experts'),)

# LLaDA2's Ling-style names beside the shared ones; its fused query_key_value
# is split by `_LLADA2_PACKED`.
_LLADA2_NAMES: Renames = (
    ('model.word_embeddings', 'model.embed_tokens'), ('attention.dense', 'self_attn.o_proj'),
    ('attention.query_layernorm', 'self_attn.q_norm'), ('attention.key_layernorm', 'self_attn.k_norm'),
    ('mlp.gate.expert_bias', 'mlp.gate.e_score_correction_bias'))
_LLADA2_PACKED = tuple(
    Packed(f'.attention.query_key_value.{leaf}',
           tuple(f'.self_attn.{part}.{leaf}' for part in ('q_proj', 'k_proj', 'v_proj')), 0,
           widths=qkv_widths)
    for leaf in ('weight', 'bias'))


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
    defaults=GEMMA4_DEFAULTS,
)

LLADA2 = DecoderFamily(
    ('llada2_moe',),
    _llada2_config,
    lambda fields: bool(fields.causal is False and fields.diffusion_block is not None),
    'llada2_moe',
    'LLaDA2MoeModelLM',
    _mask_token_export,
    weight_path=partial(renamed_path, _LLADA2_NAMES),
    packed=_LLADA2_PACKED,
    preserve_source_layout=True,
    # LLaDA2MoeConfig's default; the shared reader's is 1e-06.
    defaults={'rms_norm_eps': 1e-5},
)

# LLaDA-MoE-7B-A1B's 'llada' configs reach this computation through LLADA's
# reader (`_llada_config`); its weights through LLADA's map (`_llada_path`).
LLADA_MOE = DecoderFamily(
    ('llada_moe_v2',),
    partial(_llada_moe_config, renormalized=True),
    lambda fields: bool(fields.causal is False and fields.mixture is not None
                        and fields.mixture.score_function == 'softmax' and not fields.mixture.parallel),
    'llada_moe_v2',
    'LLaDAMoEv2ModelLM',
    _mask_token_export,
    weight_path=partial(renamed_path, _LLADA_MOE_NAMES),
    preserve_source_layout=True,
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
