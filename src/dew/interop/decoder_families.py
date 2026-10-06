"""Decoder families, ordered from the most specific layout to the plain decoder."""

from functools import partial

from dew.interop import mamba2
from dew.interop.decoder_config import _GEMMA, _QWEN35
from dew.interop.decoder_export import _decoder_tensors
from dew.interop.decoder_family import DecoderFamily, _every_layer_windowed, _kind_mixers
from dew.interop.decoder_paths import _FUSED_EXPERTS, _renamed_name, _renamed_path
from dew.interop.families.bloom import _BLOOM_NAMES, _bloom_config, _bloom_export, _bloom_path
from dew.interop.families.deepseek import (
    _MINIMAX_M2_NAMES,
    _deepseek_config,
    _deepseek_v2_mixture,
    _deepseek_v4_config,
    _deepseek_v4_path,
    _deepseek_v4_prepare,
    _kimi_k25_config,
    _kimi_k25_path,
    _minimax_m2_config,
)
from dew.interop.families.deepseek_v41 import DEEPSEEK_V41
from dew.interop.families.falcon import (
    _FALCON_NAMES,
    _falcon_config,
    _falcon_export,
    _falcon_export_weights,
    _falcon_path,
    _falcon_prepare,
)
from dew.interop.families.gemma import (
    _gemma2_config,
    _gemma2_export,
    _gemma3_config,
    _gemma3_export,
    _gemma3n_config,
    _gemma3n_path,
    _gemma4_config,
    _gemma4_export,
    _gemma4_export_path,
    _gemma4_export_weights,
    _gemma4_path,
    _gemma_config,
)
from dew.interop.families.glm import (
    _glm4_moe_config,
    _glm4_moe_path,
    _glm5_next_config,
    _glm5_next_export,
    _glm5_next_export_weights,
    _glm_moe_dsa_config,
)
from dew.interop.families.gpt2 import (
    _GPT2_NAMES,
    _GPT2_PACKED,
    _GPT_NEO_NAMES,
    _GPTJ_NAMES,
    _gpt2_config,
    _gpt2_export,
    _gpt2_path,
    _gpt2_prepare,
    _gpt_neo_config,
    _gpt_neo_export,
    _gptj_config,
    _gptj_export,
    _gptj_export_weights,
    _gptj_path,
    _gptj_prepare,
)
from dew.interop.families.gpt_neox import (
    _GPT_NEOX_NAMES,
    _gpt_neox_config,
    _gpt_neox_export,
    _gpt_neox_export_weights,
    _gpt_neox_path,
    _gpt_neox_prepare,
)
from dew.interop.families.gpt_oss import _gpt_oss_config, _gpt_oss_export, _gpt_oss_export_path, _gpt_oss_path
from dew.interop.families.kimi import (
    _KDA_ZERO_PADDED,
    _kimi_k3_config,
    _kimi_k3_path,
    _kimi_k3_prepare,
    _kimi_linear_config,
    _kimi_linear_path,
    _kimi_linear_prepare,
)
from dew.interop.families.llama import (
    _GRANITEMOE_NAMES,
    _GRANITEMOE_PACKED,
    _granitemoe_config,
    _granitemoe_path,
    _llama_config,
    _ministral_config,
    _mistral_config,
    _mixtral_config,
    _mixtral_path,
)
from dew.interop.families.llama4 import _LLAMA4_PACKED, _llama4_config, _llama4_export, _llama4_path
from dew.interop.families.masked_diffusion import (
    _LLADA_NAMES,
    _diffusion_gemma_export,
    _diffusion_gemma_text_config,
    _dream_config,
    _llada_config,
    _llada_path,
    _mask_token_export,
)
from dew.interop.families.modernbert import (
    _MODERNBERT_PACKED,
    _modernbert_config,
    _modernbert_export,
    _modernbert_export_path,
    _modernbert_path,
    _modernbert_prepare,
)
from dew.interop.families.nemotron_h import (
    PACKED as _NEMOTRON_H_PACKED,
    config_from_hf as _nemotron_h_config,
    export_path as _nemotron_h_export_path,
    matches as _nemotron_h_matches,
    weight_path as _nemotron_h_path,
)
from dew.interop.families.olmo import _olmo3_config
from dew.interop.families.opt import _OPT_NAMES, _opt_config, _opt_export
from dew.interop.families.phi import _PHI_NAMES, _phi_config, _phi_export
from dew.interop.families.qwen import (
    _qwen2_config,
    _qwen2_moe_config,
    _qwen3_config,
    _qwen3_export,
    _qwen3_moe_config,
    _qwen3_next_config,
    _qwen35_config,
    _qwen35_moe_config,
    _qwen35_moe_path,
    _qwen35_path,
)
from dew.nn.deepseek_v4 import DeepseekV4Mixer
from dew.nn.dsa_kpool import KPoolSparseAttentionMixer
from dew.nn.kda import KimiDeltaAttentionMixer
from dew.nn.llama4 import Llama4Mixer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mixers.mamba2 import Mamba2Mixer
from dew.nn.mla import MLAMixer

ENTRIES = (
    DecoderFamily(
        ('bloom',), _bloom_config,
        lambda fields: fields.position_embedding == 'alibi' and fields.embedding_norm,
        'bloom', 'BloomForCausalLM', _bloom_export,
        weight_path=_bloom_path, export_path=partial(_renamed_name, _BLOOM_NAMES),
        prepare=partial(_gpt_neox_prepare, attention_name='self_attention'),
        export_weights=partial(_gpt_neox_export_weights, attention_name='self_attention'),
        preserve_source_layout=False,
        tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
    ),
    DecoderFamily(
        ('gpt_neo',), _gpt_neo_config,
        lambda fields: fields.position_embedding == 'learned' and fields.attention_scale == 1.0
                       and fields.attention_bias is False and fields.o_proj_bias is True,
        'gpt_neo', 'GPTNeoForCausalLM', _gpt_neo_export,
        weight_path=partial(_renamed_path, _GPT_NEO_NAMES),
        export_path=partial(_renamed_name, _GPT_NEO_NAMES), preserve_source_layout=False,
        tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
    ),
    DecoderFamily(
        ('phi',), _phi_config,
        lambda fields: fields.shared_parallel_norm and fields.head_bias and fields.attention_bias,
        'phi', 'PhiForCausalLM', _phi_export,
        weight_path=partial(_renamed_path, _PHI_NAMES), export_path=partial(_renamed_name, _PHI_NAMES),
        export_weights=_decoder_tensors, preserve_source_layout=False,
    ),
    DecoderFamily(
        ('gptj',), _gptj_config,
        lambda fields: fields.shared_parallel_norm and fields.head_bias and not fields.attention_bias,
        'gptj', 'GPTJForCausalLM', _gptj_export,
        weight_path=_gptj_path, export_path=partial(_renamed_name, _GPTJ_NAMES),
        prepare=_gptj_prepare, export_weights=_gptj_export_weights, preserve_source_layout=False,
        tied_head_names=('lm_head.weight', 'transformer.wte.weight'),
    ),
    DecoderFamily(
        ('falcon',), _falcon_config,
        lambda fields: fields.shared_parallel_norm and fields.mlp == 'gelu_exact'
                       and fields.partial_rotary_factor is None,
        'falcon', 'FalconForCausalLM', _falcon_export,
        weight_path=_falcon_path, export_path=partial(_renamed_name, _FALCON_NAMES),
        prepare=_falcon_prepare, export_weights=_falcon_export_weights, preserve_source_layout=False,
        tied_head_names=('lm_head.weight', 'transformer.word_embeddings.weight'),
    ),
    DecoderFamily(
        ('minimax_m2',),
        _minimax_m2_config,
        lambda fields: bool(fields.qk_norm and fields.qk_norm_scope == 'projection'
                            and fields.pre_norms and fields.mixture is not None),
        'minimax_m2',
        'MiniMaxM2ForCausalLM',
        lambda model: {},
        weight_path=partial(_renamed_path, _MINIMAX_M2_NAMES),
        export_path=partial(_renamed_name, _MINIMAX_M2_NAMES),
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ('modernbert',), _modernbert_config,
        lambda fields: not fields.causal and fields.embedding_norm and not fields.first_attention_norm,
        'modernbert', 'ModernBertForMaskedLM', _modernbert_export,
        weight_path=_modernbert_path, export_path=_modernbert_export_path,
        prepare=_modernbert_prepare, packed=_MODERNBERT_PACKED, preserve_source_layout=False,
        tied_head_names=('decoder.weight', 'model.embeddings.tok_embeddings.weight'),
    ),
    DecoderFamily(
        ("nemotron_h",),
        _nemotron_h_config,
        _nemotron_h_matches,
        "nemotron_h",
        "NemotronHForCausalLM",
        lambda model: {},
        weight_path=_nemotron_h_path,
        export_path=_nemotron_h_export_path,
        packed=_NEMOTRON_H_PACKED,
        preserve_source_layout=True,
        tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
    ),
    DecoderFamily(
        ("gpt_neox",),
        _gpt_neox_config,
        lambda fields: bool(
            fields.norm_type == "layer"
            and fields.norm_bias
            and fields.mlp_bias
            and fields.position_embedding == "rotary"
        ),
        "gpt_neox",
        "GPTNeoXForCausalLM",
        _gpt_neox_export,
        weight_path=_gpt_neox_path,
        export_path=partial(_renamed_name, _GPT_NEOX_NAMES),
        prepare=_gpt_neox_prepare,
        export_weights=_gpt_neox_export_weights,
        preserve_source_layout=False,
        tied_head_names=("embed_out.weight", "gpt_neox.embed_in.weight"),
    ),
    DecoderFamily(
        ("opt",),
        _opt_config,
        lambda fields: fields.position_embedding_offset == 2,
        "opt",
        "OPTForCausalLM",
        _opt_export,
        weight_path=partial(_renamed_path, _OPT_NAMES),
        export_path=partial(_renamed_name, _OPT_NAMES),
        preserve_source_layout=False,
        tied_head_names=("lm_head.weight", "model.decoder.embed_tokens.weight"),
    ),
    DecoderFamily(
        ("gpt2",),
        _gpt2_config,
        lambda fields: fields.position_embedding == "learned",
        "gpt2",
        "GPT2LMHeadModel",
        _gpt2_export,
        weight_path=_gpt2_path,
        export_path=partial(_renamed_name, _GPT2_NAMES),
        prepare=_gpt2_prepare,
        packed=_GPT2_PACKED,
        preserve_source_layout=False,
        tied_head_names=("lm_head.weight", "transformer.wte.weight"),
    ),
    DecoderFamily(
        ("glm5_next_text",),
        _glm5_next_config,
        lambda fields: any(
            isinstance(mixer, (KimiDeltaAttentionMixer, KPoolSparseAttentionMixer))
            for mixer in _kind_mixers(fields)
        ),
        "glm5_next_text",
        "Glm5NextTextForCausalLM",
        _glm5_next_export,
        weight_path=_glm4_moe_path,
        export_weights=_glm5_next_export_weights,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("diffusion_gemma_text",),
        _diffusion_gemma_text_config,
        lambda fields: bool(
            fields.causal is False
            and (
                fields.v_norm
                or fields.per_layer_input_dim
                or fields.kv_shared_layers
            )
        ),
        "diffusion_gemma_text",
        "DiffusionGemmaForBlockDiffusion",
        _diffusion_gemma_export,
        sandwich_norms=True,
        weight_path=_gemma4_path,
        export_path=_gemma4_export_path,
        packed=_FUSED_EXPERTS,
        export_weights=_gemma4_export_weights,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("dream", "Dream"),
        _dream_config,
        lambda fields: bool(
            fields.causal is False
            and fields.attention_bias
            and fields.o_proj_bias is False
        ),
        "dream",
        "DreamModel",
        _mask_token_export,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("llada",),
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
        "llada",
        "LLaDAModelLM",
        _mask_token_export,
        weight_path=_llada_path,
        export_path=partial(_renamed_name, _LLADA_NAMES),
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("gpt_oss",),
        _gpt_oss_config,
        lambda fields: fields.mlp == "swigluoai",
        "gpt_oss",
        "GptOssForCausalLM",
        _gpt_oss_export,
        weight_path=_gpt_oss_path,
        export_path=_gpt_oss_export_path,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("llama4_text",),
        _llama4_config,
        lambda fields: any(isinstance(mixer, Llama4Mixer) for mixer in _kind_mixers(fields)),
        "llama4_text",
        "Llama4ForCausalLM",
        _llama4_export,
        weight_path=_llama4_path,
        packed=_LLAMA4_PACKED,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("glm4_moe",),
        _glm4_moe_config,
        lambda fields: (
            fields.partial_rotary_type == "default"
            and (mixture := fields.mixture) is not None
            and mixture.bias
        ),
        "glm4_moe",
        "Glm4MoeForCausalLM",
        lambda model: {},
        weight_path=_glm4_moe_path,
        preserve_source_layout=True,
    ),
    # GLM's sparse block is V3.2's with the indexer rotating interleaved
    # pairs, which no DeepSeek release does, so that field names the family.
    DecoderFamily(
        ("glm_moe_dsa",),
        _glm_moe_dsa_config,
        lambda fields: (
            isinstance(mixer := fields.mixer, MLAMixer)
            and mixer.index_topk is not None
            and mixer.index_rope_interleave
        ),
        "glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
        lambda model: {},
        weight_path=_glm4_moe_path,
        preserve_source_layout=True,
    ),
    DEEPSEEK_V41,
    # V4's block is nothing another family builds: the mixer kind names its
    # window, its compressor and its grouped output projection at once.
    DecoderFamily(
        ("deepseek_v4",),
        _deepseek_v4_config,
        lambda fields: isinstance(fields.mixer, DeepseekV4Mixer),
        "deepseek_v4",
        "DeepseekV4ForCausalLM",
        lambda model: {},
        weight_path=_deepseek_v4_path,
        prepare=_deepseek_v4_prepare,
        preserve_source_layout=True,
        tied_head_names=("head.weight", "embed.weight"),
    ),
    DecoderFamily(
        ("deepseek_v32",),
        partial(_deepseek_config, sparse=True),
        lambda fields: (isinstance(mixer := fields.mixer, MLAMixer) and mixer.index_topk is not None),
        "deepseek_v32",
        "DeepseekV32ForCausalLM",
        lambda model: {},
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("deepseek_v2",),
        partial(_deepseek_config, mixture=_deepseek_v2_mixture),
        lambda fields: (
            isinstance(fields.mixer, MLAMixer)
            and (mixture := fields.mixture) is not None
            and not mixture.bias
        ),
        "deepseek_v2",
        "DeepseekV2ForCausalLM",
        lambda model: {},
        preserve_source_layout=True,
    ),
    # Kimi and DeepSeek V3 share a computation; only source provenance names Kimi.
    # Derived-model export therefore never selects Kimi via `matches`.
    DecoderFamily(
        ("kimi_k2",),
        _deepseek_config,
        lambda fields: False,
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        lambda model: {},
        preserve_source_layout=True,
    ),
    # Kimi K2.5 wraps that same computation in a vision repo, so it is
    # provenance-only too, and its own tensor names are the wrapper's.
    DecoderFamily(
        ("kimi_k25",),
        _kimi_k25_config,
        lambda fields: False,
        "kimi_k25",
        "Kimi_K25ForConditionalGeneration",
        lambda model: {},
        weight_path=_kimi_k25_path,
        preserve_source_layout=True,
        tied_head_names=("language_model.lm_head.weight", "language_model.model.embed_tokens.weight"),
    ),
    # Kimi Linear's released remote code; provenance-only, like K2.5.
    DecoderFamily(
        ("kimi_linear",),
        _kimi_linear_config,
        lambda fields: False,
        "kimi_linear",
        "KimiLinearForCausalLM",
        lambda model: {},
        weight_path=_kimi_linear_path,
        prepare=_kimi_linear_prepare,
        preserve_source_layout=True,
    ),
    # Kimi K3's text decoder under its vision wrapper; provenance-only, like K2.5.
    DecoderFamily(
        ("kimi_k3",),
        _kimi_k3_config,
        lambda fields: False,
        "kimi_k3",
        "KimiK3ForConditionalGeneration",
        lambda model: {},
        weight_path=_kimi_k3_path,
        prepare=_kimi_k3_prepare,
        zero_padded=_KDA_ZERO_PADDED,
        preserve_source_layout=True,
        tied_head_names=("language_model.lm_head.weight", "language_model.model.embed_tokens.weight"),
    ),
    DecoderFamily(
        ("deepseek_v3",),
        _deepseek_config,
        lambda fields: isinstance(fields.mixer, MLAMixer),
        "deepseek_v3",
        "DeepseekV3ForCausalLM",
        lambda model: {},
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("qwen3_next",),
        _qwen3_next_config,
        lambda fields: any(
            isinstance(mixer, GatedDeltaNetMixer) and mixer.fused_in_proj for mixer in _kind_mixers(fields)
        ),
        "qwen3_next",
        "Qwen3NextForCausalLM",
        lambda model: {},
        weight_path=_qwen35_moe_path,
        packed=_FUSED_EXPERTS,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("qwen3_5_moe_text",),
        _qwen35_moe_config,
        lambda fields: bool(fields.output_gate and fields.mixture is not None),
        "qwen3_5_moe_text",
        "Qwen3_5MoeForCausalLM",
        lambda model: {},
        weight_path=_qwen35_moe_path,
        packed=_FUSED_EXPERTS,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        (_QWEN35,),
        _qwen35_config,
        lambda fields: bool(
            fields.output_gate or "linear_attention" in (fields.layer_types or ())
        ),
        _QWEN35,
        "Qwen3_5ForCausalLM",
        lambda model: {},
        weight_path=_qwen35_path,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("olmo3",),
        _olmo3_config,
        lambda fields: not fields.pre_norms,
        "olmo3",
        "Olmo3ForCausalLM",
        lambda model: {},
        sandwich_norms=True,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("gemma3n_text",),
        _gemma3n_config,
        lambda fields: fields.altup is not None,
        "gemma3n_text",
        "Gemma3nForCausalLM",
        _gemma3_export,
        sandwich_norms=True,
        weight_path=_gemma3n_path,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("gemma4_text",),
        _gemma4_config,
        lambda fields: bool(
            fields.v_norm or fields.per_layer_input_dim or fields.kv_shared_layers
        ),
        "gemma4_text",
        "Gemma4ForCausalLM",
        _gemma4_export,
        sandwich_norms=True,
        weight_path=_gemma4_path,
        export_path=_gemma4_export_path,
        packed=_FUSED_EXPERTS,
        export_weights=_gemma4_export_weights,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        (_GEMMA,),
        _gemma3_config,
        lambda fields: bool(fields.sandwich_norms and fields.qk_norm),
        _GEMMA,
        "Gemma3ForCausalLM",
        _gemma3_export,
        sandwich_norms=True,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("gemma2",),
        _gemma2_config,
        lambda fields: bool(fields.sandwich_norms),
        "gemma2",
        "Gemma2ForCausalLM",
        _gemma2_export,
        sandwich_norms=True,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("gemma",),
        _gemma_config,
        lambda fields: bool(fields.embedding_scale),
        "gemma",
        "GemmaForCausalLM",
        lambda model: {},
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("qwen3_moe",),
        _qwen3_moe_config,
        lambda fields: bool(fields.qk_norm and fields.mixture is not None),
        "qwen3_moe",
        "Qwen3MoeForCausalLM",
        _qwen3_export,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("qwen3",),
        _qwen3_config,
        lambda fields: bool(fields.qk_norm),
        "qwen3",
        "Qwen3ForCausalLM",
        _qwen3_export,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ('qwen2_moe',),
        _qwen2_moe_config,
        lambda fields: bool(not fields.qk_norm and fields.mixture is not None
                            and fields.mixture.shared_gate),
        'qwen2_moe',
        'Qwen2MoeForCausalLM',
        lambda model: {},
        weight_path=_qwen35_moe_path,
        packed=_FUSED_EXPERTS,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("qwen2",),
        _qwen2_config,
        lambda fields: bool(fields.attention_bias and fields.o_proj_bias is False),
        "qwen2",
        "Qwen2ForCausalLM",
        _qwen3_export,
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ('granitemoe',),
        _granitemoe_config,
        lambda fields: bool(fields.mixture is not None and not fields.qk_norm
                            and (fields.embedding_multiplier != 1.0 or fields.residual_multiplier != 1.0
                                 or fields.logits_scaling != 1.0 or fields.attention_scale is not None)),
        'granitemoe',
        'GraniteMoeForCausalLM',
        lambda model: {},
        weight_path=_granitemoe_path,
        export_path=partial(_renamed_name, _GRANITEMOE_NAMES),
        packed=_GRANITEMOE_PACKED,
        preserve_source_layout=True,
    ),
    DecoderFamily(
        ("mixtral",),
        _mixtral_config,
        lambda fields: fields.mixture is not None,
        "mixtral",
        "MixtralForCausalLM",
        lambda model: {},
        weight_path=_mixtral_path,
        preserve_source_layout=True,
    ),
    # MistralConfig has no layer_types: its window is on every layer.
    DecoderFamily(
        ("mistral",),
        _mistral_config,
        _every_layer_windowed,
        "mistral",
        "MistralForCausalLM",
        lambda model: {"layer_types": None},
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("mamba2",),
        mamba2.config_from_hf,
        lambda fields: isinstance(fields.mixer, Mamba2Mixer),
        "mamba2",
        "Mamba2ForCausalLM",
        lambda model: {},
        weight_path=mamba2.weight_path,
        export_path=mamba2.export_path,
        preserve_source_layout=True,
        tied_head_names=("lm_head.weight", "backbone.embeddings.weight"),
    ),
    DecoderFamily(
        ("ministral",),
        _ministral_config,
        lambda fields: "sliding_attention" in (fields.layer_types or ()),
        "ministral",
        "MinistralForCausalLM",
        lambda model: {},
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("llama",),
        _llama_config,
        lambda fields: True,
        "llama",
        "LlamaForCausalLM",
        lambda model: {},
        preserve_source_layout=False,
    ),
)
