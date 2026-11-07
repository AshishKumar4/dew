"""Translate the bidirectional LLaDA, Dream and DiffusionGemma decoders.

LLaDA's dense and routed releases configure the shared GQA and expert
modules. Dream rides Qwen2's layout and DiffusionGemma Gemma 4's; the
adapters below own only their configuration and tensor spellings.
"""

from __future__ import annotations

from collections.abc import Mapping

from dew import _records
from dew.interop.config_records import NativeFields
from dew.interop.decoder_config import (
    DecoderSpelling,
    decoder_layout,
    experts,
    flag,
    inert,
    integer,
    plain_rope_part,
    unwritten,
)
from dew.interop.decoder_parts import (
    DEFAULT_MAX_SEQ_LEN,
    FUSED_EXPERTS,
    PLAIN_SPELLING,
    DecoderFamily,
    DecoderFields,
    MaskedWalk,
    MixtureFields,
    Packed,
    Renames,
    RopeSpelling,
    SourceCode,
    qkv_widths,
    record_int,
    refuse,
    refuse_export,
    spelled,
)
from dew.interop.families.deepseek import DENSE_FIRST, V3_ROUTER, deepseek_experts, routes_some
from dew.interop.families.gemma import GEMMA4_DEFAULTS, GEMMA4_ROUTES, gemma4_layout
from dew.interop.families.qwen import QWEN2_CONFIG
from dew.interop.layouts import (
    Choice,
    ConfigField,
    ConfigLayout,
    ConfigPart,
    Restated,
    Table,
    When,
    constant,
    unread,
)
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import Mixture
from dew.registry import Configured

# The 7B tokenizer reserves <|mask|> here; LLaDA2's generate also defaults to it.
_MOE_MASK_ID = 156895


def _mask(default: int | None = None) -> ConfigPart[CausalTransformer]:
    """The mask id a masked-diffusion release reserves, both ways: `mask_token_id` or `mask_id` as the
    releases write it, `default` where neither is (refused where it has none); written as mask_token_id.
    LLaDA and Dream declare the reserved id and nothing else of their own."""
    def read(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
        mask = hf_config.get('mask_token_id', hf_config.get('mask_id', default))
        if mask is None:
            refuse('mask_token_id', 'a masked diffusion checkpoint reserves its mask id')
        used.update(('mask_token_id', 'mask_id'))
        record['mask_token_id'] = _records.integer(mask, 'mask_token_id/mask_id')
        return record
    return ConfigPart(read, lambda model: {'mask_token_id': model.mask_token_id})


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
    """Rewrite LLaDA's own config spellings under the standard names the shared decoder reads.

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


_LLADA_DECODER = decoder_layout(DecoderSpelling(reads=frozenset({'attention_bias'})))
"""Dense LLaDA's decoder under the standard spellings its geometry resolves (`_llada_geometry`)."""


def _read_llada(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """GSAI-ML/LLaDA-8B-Base's geometry, its own spellings resolved under the standard ones the shared
    decoder reads, and its own flags checked (`_llada_refusals`); written under the standard spellings,
    which win over its aliases on a read."""
    std = _llada_geometry(hf_config)
    used.update(key for aliases in _LLADA_ALIASES for key in aliases if key in hf_config)
    _llada_refusals(hf_config, used, std)
    return _LLADA_DECODER.read(std, set(), record)


LLADA_DENSE = ConfigLayout(ConfigPart(_read_llada, _LLADA_DECODER.write), _mask(),
                           constant('causal', value=False))
"""Dense LLaDA's config, both ways.

GSAI-ML/LLaDA-8B-Base (model_type 'llada', architectures ['LLaDAModelLM']) computes a Llama block: RMSNorm
pre-norms, rotate-half rope, SwiGLU. It carries no causal mask anywhere (modeling_llada.py, LLaDAModel
bidirectional bias, LLaDALlamaBlock is_causal=False) and trains masked diffusion on the id in
mask_token_id. The checkpoint names its tensors OLMo-style, which its routes rename onto the llama
layout the shared map reads (`_LLADA_ROUTES`)."""


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


def _moe_decoder(*, keys: Mapping[str, str], spelling: RopeSpelling | None,
                 default_factor: float | None) -> ConfigLayout[CausalTransformer]:
    """The released MoE decoders' shared pre-norm GQA and sliced rotate-half rope, both ways (RMS norms
    at 1e-5 unless stated, the dense width under the family's own key, `keys`), bidirectional, and the
    mask id (156895 where unstated). `spelling` and `default_factor` are the config class's base and
    factor for a config that states neither, or None for a class whose defaults are no rotary at all:
    LLaDA-MoE's and LLaDA-MoE v2's default both to -1 (configuration_lladamoe.py:32-33), so their configs
    state them."""
    return ConfigLayout(
        inert('rope_scaling', 'the released MoE checkpoints use plain rotary positions'),
        decoder_layout(DecoderSpelling(reads=frozenset({'attention_bias'}), keys=keys,
                                       defaults={'rms_norm_eps': 1e-5}),
                       rope=plain_rope_part(spelling, default_factor=default_factor,
                                            reference='LLaDA MoE rotary')),
        constant('causal', value=False), _mask(_MOE_MASK_ID))


def _llada_moe_routed(hf_config: Mapping[str, object]) -> tuple[int, ...]:
    """The routed layers, those `moe_layer_freq` enables, one entry a decoder layer."""
    layers = record_int(hf_config, 'num_hidden_layers')
    pattern = _records.integers(hf_config['moe_layer_freq'], 'moe_layer_freq')
    if len(pattern) < layers:
        refuse('moe_layer_freq', 'it must name each decoder layer')
    return tuple(index for index, enabled in enumerate(pattern[:layers]) if enabled != 0)


def _llada_moe_schedule(hf_config: Mapping[str, object], used: set[str],
                        record: DecoderFields) -> DecoderFields:
    used.add('moe_layer_freq')
    routed = _llada_moe_routed(hf_config)
    if record.get('mixture') is not None:
        record['mixture'] = NativeFields(Mixture, {**_records.record(record['mixture'], 'mixture'),
                                                   'layers': routed})
    return record


def _llada_moe_router(*, v2: bool) -> ConfigPart[Mixture]:
    """OLMoE's softmax router: the 7B code fixes norm_topk_prob=False and never reads its scaling or score
    function; v2 softmaxes the selected logits and renormalises them. Its grouped router class is not the
    class its DecoderLayer instantiates. A model that does not renormalise is the 7B's, its own
    architecture written."""
    def read(hf_config: Mapping[str, object], used: set[str], record: MixtureFields) -> MixtureFields:
        used.update(('norm_topk_prob', 'moe_router_score_function', 'moe_router_enable_expert_bias'))
        if v2 and hf_config.get('moe_router_enable_expert_bias'):
            refuse('moe_router_enable_expert_bias',
                   'the released v2 SparseMoeBlock reads routing_weights before assigning it')
        bias = bool(hf_config.get('moe_router_enable_expert_bias'))
        record.update(norm_topk_prob=v2, bias=bias, bias_in_weights=bias)
        return record

    def write(mixture: Mixture) -> Mapping[str, object]:
        return {'norm_topk_prob': mixture.norm_topk_prob, 'moe_router_score_function': 'softmax',
                'moe_router_enable_expert_bias': mixture.bias,
                **({} if mixture.norm_topk_prob else {'architectures': ['LLaDAMoEModelLM']})}
    return ConfigPart(read, write, keys=('norm_topk_prob', 'moe_router_score_function',
                                         'moe_router_enable_expert_bias'))


def _shared_width(value: Configured, key: str) -> int:
    """The shared branch's width; the release defaults it to -1, and only an explicit null disables it."""
    return 0 if value is None else _records.integer(value, key)


def _scaling(value: Configured, key: str) -> float:
    return _records.number(1.0 if value is None else value, key) or 1.0


def _llada_moe(*, v2: bool) -> ConfigLayout[CausalTransformer]:
    """LLaDA-MoE's config (`v2`: LLaDA-MoE v2's), both ways: OLMoE's softmax experts on the layers
    `moe_layer_freq` enables, per-head QK norms and an optional shared branch.

    Both tables pad config.pad_token_id (modeling_lladamoe.py:895-898 at daccaf1,
    modeling_lladamoe_v2.py:1094-1097 at 2a513f7), which both config classes default to 1. Read here
    rather than by the family, since model_type 'llada' also names dense LLaDA, whose table pads
    nothing."""
    return ConfigLayout(
        _moe_decoder(keys={'intermediate_size': 'dense_intermediate_size'}, spelling=None,
                     default_factor=None),
        flag('qk_layernorm', 'qk_norm', default=False),
        ConfigField('pad_token_id', 'padding_idx', 1,
                    parse=lambda value, key: None if value is None else _records.integer(value, key)),
        inert('clip_qkv', 'clipping normalized queries, keys and values is not configured'),
        experts(integer('num_experts', 'experts'), integer('num_experts_per_tok', 'top_k'),
                integer('expert_intermediate_size', 'expert_features'),
                ConfigField('shared_expert_intermediate_size', 'shared_features', -1, parse=_shared_width,
                            spell=lambda width: width or None),
                ConfigField('routed_scaling_factor', 'scaling', 1.0, parse=_scaling) if v2
                else unread('routed_scaling_factor'),
                _llada_moe_router(v2=v2),
                unread('router_num_group', 'router_topk_group', 'output_router_logits',
                       'router_aux_loss_coef'),
                present=lambda hf_config: bool(_llada_moe_routed(hf_config))),
        ConfigPart(_llada_moe_schedule, lambda model: {'moe_layer_freq': [
            int(model.mixture is not None and index in model.sparse_layers)
            for index in range(model.num_layers)]}, keys=('moe_layer_freq',)))


def _llada2_walk(config: Mapping[str, object], generation: Mapping[str, object],
                 code: SourceCode | None) -> MaskedWalk | None:
    """LLaDA2's own generation, where Dew runs it: 2.0's `generate` walks blocks of 32 over 32 steps
    each at temperature 0, revealing past 0.95 (modeling_llada2_moe.py:1233-1247 at dad945c). 2.1's
    configs read as 2.0's and take the same walk, without the release's editing of revealed tokens,
    which Dew does not run; 2.2's generates with editing past another threshold, and unmasks the whole
    response with Dew's MDLM instead: a 2.2 config names its DELETE and SPLIT ids, or, as Dew's own
    export writes it, its block routing (`block_size`, `expert_capacity`)."""
    if any(config.get(field) is not None
           for field in ("delete_token_id", "split_token_id", "block_size", "expert_capacity")):
        return None
    from dew.diffusion.discrete import Confident

    return MaskedWalk(Confident(threshold=0.95, temperature=0.0), steps=32, block=32)


DREAM_SAMPLING = {'7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921': None,
                  '99318e7261e858be87e0aa5c7cccfbb2912d166c2b4218f0f81a07f922b5cf55': 1e-12}
"""The Dream releases' generation code Dew walks as they sample, by the sha256 of their
`generation_utils.py`, each with the time its grid runs to: Dream-v0's (Dream-org/Dream-v0-Base-7B at
6572adb, Dream-v0-Instruct-7B the same file) reads `eps` from the generation config (None here);
DiffuCoder's (apple/DiffuCoder-7B-Base at a2c33054, its Instruct and cpGRPO the same file) fixes 1e-12
whatever the config states. Dream-Coder's adds an end-token penalty to its logits, a walk Dew does not
run."""

DREAM_GENERATION = {'alg': 'origin', 'alg_temp': None, 'eps': 1e-3, 'steps': 512, 'temperature': 0.0,
                    'top_p': None, 'top_k': None}
"""DreamGenerationConfig's defaults for the controls its `_sample` reads (generation_utils.py:96-110)."""


def _dream_walk(config: Mapping[str, object], generation: Mapping[str, object],
                code: SourceCode | None) -> MaskedWalk | None:
    """Dream's own generation, where Dew runs it: `diffusion_generate`'s `alg='origin'`
    (generation_utils.py `_sample`), each of `steps` revealing every masked position with chance
    `1 - s/t` over `linspace(1, eps, steps + 1)`, all at the last, each revealed position taking its
    token at `temperature` (its best at 0), which is `Unmask` on that grid (`DiscreteProcess.endpoint`).
    The grid's end is the release's code's (`DREAM_SAMPLING`, by its `generation_utils.py`), the other
    controls the generation config's, DreamGenerationConfig's defaults where it states none. None for
    a source whose code Dew does not know, or that asks what `origin` does not take (another `alg`, a
    `top_p` or `top_k`), which the task then refuses by name (`audit_masked`)."""
    import hashlib

    from dew.diffusion.discrete import Unmask
    from dew.interop.sources import code_modules

    # The modeling code imports its sampling from the file beside it (`from .generation_utils import`),
    # which its modules' resolution brings, as `Pretrained.save` writes it.
    sampling = None if code is None else code_modules(code).get('generation_utils.py')
    if sampling is None:
        return None
    digest = hashlib.sha256(sampling.read_bytes()).hexdigest()
    if digest not in DREAM_SAMPLING:
        return None
    stated = {**DREAM_GENERATION,
              **{name: generation[name] for name in DREAM_GENERATION if name in generation}}
    if stated['alg'] != 'origin' or stated['top_k'] is not None or stated['top_p'] not in (None, 1, 1.0):
        return None
    endpoint = DREAM_SAMPLING[digest]
    return MaskedWalk(
        Unmask(temperature=float(_records.number(stated['temperature'], 'temperature'))),
        steps=_records.integer(stated['steps'], 'steps'),
        endpoint=float(_records.number(stated['eps'], 'eps')) if endpoint is None else endpoint,
        reads=frozenset(DREAM_GENERATION))


def _renormalized(hf_config: Mapping[str, object], used: set[str], record: MixtureFields) -> MixtureFields:
    """The top-k weights renormalised wherever the router picks more than one, whatever `norm_topk_prob`
    says (LLaDA2MoeSparseMoeBlock)."""
    used.add('norm_topk_prob')
    record['norm_topk_prob'] = _records.integer(record['top_k'], 'num_experts_per_tok') > 1
    return record


def _block_routing(hf_config: Mapping[str, object], used: set[str], record: DecoderFields) -> DecoderFields:
    """2.2's block-level expert shortlist (`block_size`, `expert_capacity`), which picks no groups, and the
    q/k norms, which only the block-routed release leaves off where unstated."""
    used.update(('block_size', 'expert_capacity', 'use_qk_norm'))
    block = (record_int(hf_config, 'block_size', 32)
             if 'block_size' in hf_config or 'expert_capacity' in hf_config else None)
    record['qk_norm'] = bool(hf_config.get('use_qk_norm', block is None))
    if block is not None and record.get('mixture') is not None:
        record['mixture'] = NativeFields(Mixture, {
            **_records.record(record['mixture'], 'mixture'), 'groups': 1, 'groups_per_token': 1,
            'routing_block': block, 'block_experts': record_int(hf_config, 'expert_capacity', 48)})
    return record


def _block_routing_written(model: CausalTransformer) -> Mapping[str, object]:
    mixture = model.mixture
    if mixture is None:
        refuse_export('mixture', 'the LLaDA2 source format describes a routed decoder')
    return {'use_qk_norm': model.qk_norm,
            # LLaDA2MoeModel builds a shared branch wherever its count is not null.
            **({} if mixture.shared_features else {'num_shared_experts': None}),
            **({} if mixture.routing_block is None else {
                'block_size': mixture.routing_block, 'expert_capacity': mixture.block_experts})}


def _bias(value: Configured, key: str) -> bool:
    return bool(True if value is None else value)


LLADA2_CONFIG = ConfigLayout(
    # LLaDA2MoeConfig's rotary defaults (configuration_llada2_moe.py:29, :46).
    _moe_decoder(keys={}, spelling=PLAIN_SPELLING, default_factor=0.5),
    flag('use_qkv_bias', 'attention_bias', default=False),
    ConfigField('use_bias', 'o_proj_bias', default=True, parse=_bias,
                value=lambda model: model.attention_bias if model.o_proj_bias is None else model.o_proj_bias),
    # These names differ, but the inference gate and dense-first layout are DeepSeek V3's.
    Restated(ConfigLayout(
        deepseek_experts(
            (V3_ROUTER[0], ConfigPart(_renormalized, lambda mixture: {}, keys=('norm_topk_prob',))),
            lambda hf_config: hf_config.get('n_routed_experts') is not None and routes_some(hf_config)),
        DENSE_FIRST), own={'n_routed_experts': 'num_experts', 'n_shared_experts': 'num_shared_experts'},
        # LLaDA2MoeConfig declares no router choice: its router is the sigmoid one alone.
        undeclared=frozenset({'scoring_func', 'topk_method'})),
    ConfigPart(_block_routing, _block_routing_written),
    unread('output_router_logits', 'sliding_window', 'moe_router_enable_expert_bias', 'norm_head',
           'norm_softmax', 'router_dtype', 'score_function', 'rotary_dim', 'use_rmsnorm',
           'using_split_qkv_in_self_attention', 'embedding_dropout', 'output_dropout', 'delete_token_id',
           'split_token_id'))
"""LLaDA 2's MoE config, both ways: DeepSeek's grouped sigmoid router, or 2.2's block-level expert shortlist.

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
many blocks long (tools/llada_moe_reference.py's recorded draws)."""


DREAM_CONFIG = ConfigLayout(
    inert('use_mrope', 'the backbone rotates plain positions', neutral=False),
    QWEN2_CONFIG, _mask(), constant('causal', value=False),
    # Dream continues Qwen2.5's next-token head, so its reference reads position i from logits i-1
    # (`torch.cat([logits[:, :1], logits[:, :-1]], dim=1)` in its generation and training code).
    constant('shifted_logits', value=True))
"""Dream-v0's config, both ways.

Dream-org/Dream-v0-Base-7B (model_type 'Dream', architectures ['DreamModel']) is a 28-layer Qwen2.5-7B
geometry: 3584 wide, 28 heads, 4 kv heads. Its attention hard-codes is_causal=False over biased q/k/v and a
bias-free o_proj (modeling_dream.py, DreamAttention/DreamSdpaAttention), and its MLP is bias-free SwiGLU.
Tensor names are the qwen2 layout, so the shared map reads them with no new table. use_mrope=False is the
only Dream-only flag and changes nothing at that value."""


DIFFUSION_GEMMA_TEXT_CONFIG = ConfigLayout(
    # The reference builds no v_proj on full layers whatever the config says
    # (modeling_diffusion_gemma.py, DiffusionGemmaEncoderTextAttention: v_proj only if the layer slides),
    # so the record always reads values off keys there; the sliding layers keep their own v_proj. The
    # family declares it rather than setting it afterwards, because the global key count is read under
    # it. DiffusionGemma names no enable_moe_block flag; a config carrying the three routed widths routes
    # every layer beside its dense MLP. final_logit_softcapping is a class attribute of the reference text
    # config, not an instance field a config.json carries; the head always divides by 30 under tanh.
    gemma4_layout(k_eq_v=True, flagged=False, softcap=False),
    constant('final_logit_softcap', value=30.0), constant('causal', value=False),
    unwritten('diffusion_gemma_text is a cache-reading view; export the complete native DiffusionGemma '
              'wrapper'))
"""DiffusionGemma's text config: google/diffusiongemma-26B-A4B-it (model_type 'diffusion_gemma_text') is
the Gemma 4 text geometry with canvas denoising around it, read here and written whole as its bundle."""


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


_LLADA_ROUTES = Table(When(lambda config: config.get('mixture') is None, spelled(_LLADA_NAMES)),
                      spelled(_LLADA_MOE_NAMES), spelled(_LLADA_NAMES))
"""LLaDA's names: dense LLaDA's OLMo envelope, which a dense model writes, or the MoE release's decoder
names; a load reads either."""


DIFFUSION_GEMMA_TEXT = DecoderFamily(
    ('diffusion_gemma_text',),
    DIFFUSION_GEMMA_TEXT_CONFIG,
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
    routes=GEMMA4_ROUTES,
    packed=FUSED_EXPERTS,
    preserve_source_layout=False,
    bidirectional=True,
    defaults=GEMMA4_DEFAULTS,
    padding_idx=True,
)

DREAM = DecoderFamily(
    ('dream', 'Dream'),
    DREAM_CONFIG,
    lambda fields: bool(
        fields.causal is False
        and fields.attention_bias
        and fields.o_proj_bias is False
    ),
    'dream',
    'DreamModel',
    preserve_source_layout=True,
    bidirectional=True,
    # DreamBaseModel's table pads config.pad_token_id (modeling_dream.py:602-605 at 6572adb), whose
    # DreamConfig default is 151643.
    defaults={'pad_token_id': 151643},
    padding_idx=True,
    masked_walk=_dream_walk,
)

LLADA = DecoderFamily(
    ('llada',),
    ConfigLayout(Choice(lambda hf_config: 'num_experts' in hf_config, lambda model: model.mixture is not None,
                        _llada_moe(v2=False), LLADA_DENSE)),
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
    routes=_LLADA_ROUTES,
    export_nulls=_LLADA_MOE_NULLS,
    preserve_source_layout=True,
    bidirectional=True,
)

LLADA_MOE_V2 = DecoderFamily(
    ('llada_moe_v2',), _llada_moe(v2=True),
    lambda fields: bool(not fields.causal and (mixture := fields.mixture) is not None
                        and mixture.score_function == 'softmax' and mixture.norm_topk_prob),
    'llada_moe_v2', 'LLaDAMoEv2ModelLM',
    routes=spelled(_LLADA_MOE_NAMES),
    export_nulls=_LLADA_MOE_NULLS,
    preserve_source_layout=True,
    bidirectional=True,
)

LLADA2_MOE = DecoderFamily(
    ('llada2_moe',), LLADA2_CONFIG,
    lambda fields: bool(not fields.causal and (mixture := fields.mixture) is not None
                        and mixture.score_function == 'sigmoid'),
    'llada2_moe', 'LLaDA2MoeModelLM',
    routes=spelled(_LLADA2_NAMES),
    packed=_LLADA2_PACKED, tied_head_names=('lm_head.weight', 'model.word_embeddings.weight'),
    export_nulls=('num_shared_experts',),
    preserve_source_layout=True,
    bidirectional=True,
    # LLaDA2MoeModel's table pads config.pad_token_id (modeling_llada2_moe.py:750-754 of LLaDA2.0 at
    # dad945c, :761-765 of 2.1 at 20e64e2, :770-774 of 2.2 at 9b78da3), whose LLaDA2MoeConfig default
    # is 126081.
    # LLaDA2MoeConfig's groups where it states none (configuration_llada2_moe.py).
    defaults={'pad_token_id': 126081, 'n_group': 8, 'topk_group': 4},
    padding_idx=True,
    masked_walk=_llada2_walk,
)
