"""Decoder families, ordered from the most specific layout to the plain decoder."""

from dew.interop import mamba2
from dew.interop.decoder_parts import DecoderFamily
from dew.interop.families.bloom import BLOOM
from dew.interop.families.deepseek import (
    DEEPSEEK_V2,
    DEEPSEEK_V3,
    DEEPSEEK_V4,
    DEEPSEEK_V32,
    KIMI_K2,
    KIMI_K25,
    MINIMAX_M2,
)
from dew.interop.families.deepseek_v41 import DEEPSEEK_V41
from dew.interop.families.falcon import FALCON
from dew.interop.families.gemma import GEMMA, GEMMA2, GEMMA3_TEXT, GEMMA3N_TEXT, GEMMA4_TEXT
from dew.interop.families.glm import GLM4_MOE, GLM5_NEXT_TEXT, GLM_MOE_DSA
from dew.interop.families.gpt2 import GPT2, GPT_NEO, GPTJ
from dew.interop.families.gpt_neox import GPT_NEOX
from dew.interop.families.gpt_oss import GPT_OSS
from dew.interop.families.kimi import KIMI_K3, KIMI_LINEAR
from dew.interop.families.llama import GRANITEMOE, MISTRAL, MIXTRAL, llama_config, ministral_config
from dew.interop.families.llama4 import LLAMA4_TEXT
from dew.interop.families.masked_diffusion import DIFFUSION_GEMMA_TEXT, DREAM, LLADA
from dew.interop.families.modernbert import MODERNBERT
from dew.interop.families.nemotron_h import NEMOTRON_H
from dew.interop.families.olmo import OLMO3
from dew.interop.families.opt import OPT
from dew.interop.families.phi import PHI
from dew.interop.families.phi3 import PHI3
from dew.interop.families.qwen import (
    QWEN2,
    QWEN2_MOE,
    QWEN3,
    QWEN3_5_MOE_TEXT,
    QWEN3_5_TEXT,
    QWEN3_MOE,
    QWEN3_NEXT,
)
from dew.nn.mixers.mamba2 import Mamba2Mixer

ENTRIES = (
    BLOOM,
    GPT_NEO,
    PHI,
    GPTJ,
    FALCON,
    PHI3,
    MINIMAX_M2,
    MODERNBERT,
    NEMOTRON_H,
    GPT_NEOX,
    OPT,
    GPT2,
    GLM5_NEXT_TEXT,
    DIFFUSION_GEMMA_TEXT,
    DREAM,
    LLADA,
    GPT_OSS,
    LLAMA4_TEXT,
    GLM4_MOE,
    GLM_MOE_DSA,
    DEEPSEEK_V41,
    DEEPSEEK_V4,
    DEEPSEEK_V32,
    DEEPSEEK_V2,
    KIMI_K2,
    KIMI_K25,
    KIMI_LINEAR,
    KIMI_K3,
    DEEPSEEK_V3,
    QWEN3_NEXT,
    QWEN3_5_MOE_TEXT,
    QWEN3_5_TEXT,
    OLMO3,
    GEMMA3N_TEXT,
    GEMMA4_TEXT,
    GEMMA3_TEXT,
    GEMMA2,
    GEMMA,
    QWEN3_MOE,
    QWEN3,
    QWEN2_MOE,
    QWEN2,
    GRANITEMOE,
    MIXTRAL,
    MISTRAL,
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
        ministral_config,
        lambda fields: "sliding_attention" in (fields.layer_types or ()),
        "ministral",
        "MinistralForCausalLM",
        lambda model: {},
        preserve_source_layout=False,
    ),
    DecoderFamily(
        ("llama",),
        llama_config,
        lambda fields: True,
        "llama",
        "LlamaForCausalLM",
        lambda model: {},
        preserve_source_layout=False,
    ),
)
