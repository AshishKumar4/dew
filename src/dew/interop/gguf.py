"""Read a GGUF file as the HF config and safetensors-named tensors a decoder load reads.

A GGUF file carries its own config (key/value metadata), its tokenizer and
its tensors under llama.cpp's names, most of them quantized in blocks. This
module turns one into what `Pretrained.load` reads from a safetensors repo:
the config.json dict, and tensors under the HF names in HF layout.
Everything after that, family translation included, is the safetensors path;
the tokenizer is transformers' own conversion of the one the file carries.

Upstream first. `gguf` (gguf-py) reads the file, dequantizes every block
format and holds the HF-to-GGUF tensor-name table. transformers holds the
metadata-to-config table and the config classes, both torch-free. Its
`load_gguf_checkpoint` and its name resolution refuse to run without torch
(modeling_gguf_pytorch_utils.py, 5.16.1), so the loop that drives those
tables is written here, rule for rule, and so is its Llama Q/K un-permute,
a private method there.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np

from dew import _records

if TYPE_CHECKING:
    from gguf import GGUFReader, ReaderTensor
    from transformers import PreTrainedTokenizerFast

_ARCHITECTURES = {"llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "gemma2": "gemma2",
                  "gemma3": "gemma3_text", "qwen2moe": "qwen2_moe", "qwen3moe": "qwen3_moe"}
"""The GGUF architectures read here, each by the HF model_type it loads as, which is also its Dew
family; transformers' GGUF loader names them alike (modeling_gguf_pytorch_utils.py:609-698,
5.16.1). A `llama` file that states an expert count is Mixtral, which llama.cpp writes as Llama
with experts (`MIXTRAL`). Each architecture's conversion is inverted below: Llama's Q/K row order
(`unpermute`), Gemma 2's and 3's norm offset (`_NORM_SHIFT`) and the MoE files' stacked experts
(`_EXPERTS`); one whose conversion rewrites anything else is refused."""

MIXTRAL = "mixtral"
"""The model_type of a `llama` file with experts (llama.cpp's `LlamaModel` stacks Mixtral's)."""

_SECTIONS = {"qwen2moe": "qwen2_moe", "qwen3moe": "qwen3_moe", MIXTRAL: "llama"}
"""transformers' metadata table under the name it keys an architecture by, where that is not the
file's own."""

_EXTRA = {
    "gemma2": {"attn_logit_softcapping": "attn_logit_softcapping",
               "final_logit_softcapping": "final_logit_softcapping"},
    "gemma3": {"final_logit_softcapping": "final_logit_softcapping",
               "rope.freq_base_swa": "rope_local_base_freq"},
    "qwen2moe": {"expert_feed_forward_length": "moe_intermediate_size",
                 "expert_shared_feed_forward_length": "shared_expert_intermediate_size"},
    "qwen3moe": {"expert_feed_forward_length": "moe_intermediate_size"},
    MIXTRAL: {"expert_count": "num_local_experts", "expert_used_count": "num_experts_per_tok"},
}
"""Metadata llama.cpp writes (conversion/gemma.py, conversion/qwen.py, conversion/llama.py) that
transformers' table does not read, under the config field it states: Gemma 2's softcaps, Gemma 3's
sliding base, the expert widths, Mixtral's expert counts. transformers drops these and builds the
class's defaults; read here, the model the file states loads, and a key neither table reads is
refused."""

_NORM_SHIFT = frozenset({"gemma2", "gemma3"})
"""Architectures whose norm scales llama.cpp stores as 1 + w (Gemma2Model's and Gemma3Model's
`modify_tensors`), which Gemma's (1 + w) norm reads back as w: every `*norm.weight` loads one
lower, as transformers' Gemma2TensorProcessor subtracts (modeling_gguf_pytorch_utils.py:380-389).
Gemma 3n and 4 store w and are not read here."""

_EXPERTS = {"qwen2moe": ("mlp.experts", ("gate_proj", "up_proj", "down_proj")),
            "qwen3moe": ("mlp.experts", ("gate_proj", "up_proj", "down_proj")),
            MIXTRAL: ("block_sparse_moe.experts", ("w1", "w3", "w2"))}
"""Each MoE architecture's experts: llama.cpp stacks every expert's gate, up and down projection
into one [experts, out, in] tensor (`ffn_{gate,up,down}_exps`); they load one tensor per expert
under the HF names, `<module>.<expert>.<projection>`, in gate, up, down order."""

_HEAD_WIDTHS = frozenset({"rope.dimension_count", "attention.key_length", "attention.value_length"})
"""Keys the table drops or maps to head_dim that are inert at the head width:
a rotary over part of each head, or keys and values wider or narrower than
it, would compute another model."""


def _parse(reader: GGUFReader, key: str) -> _records.JSON:
    """One metadata value as gguf decodes it (`ReaderField.contents`), and as
    transformers reads it: a one-element array is its element."""
    value = _records.json_value(reader.fields[key].contents(), key)
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def _architecture(reader: GGUFReader) -> str:
    """The file's architecture as read here (`_ARCHITECTURES`): its `general.architecture`, a Llama
    file with experts Mixtral's."""
    stated = _parse(reader, "general.architecture") if "general.architecture" in reader.fields else None
    if stated == "llama" and "llama.expert_count" in reader.fields:
        return MIXTRAL
    if stated not in _ARCHITECTURES:
        raise ValueError(
            f"GGUF architecture {stated!r} is not read by Dew, which reads "
            f"{', '.join(_ARCHITECTURES)}; load the safetensors repo the file was quantized from "
            "(the model card's base_model)")
    return str(stated)


def _model_type(architecture: str) -> str:
    return MIXTRAL if architecture == MIXTRAL else _ARCHITECTURES[architecture]


def _config(reader: GGUFReader) -> tuple[str, Mapping[str, object]]:
    """Return the file's architecture (`_architecture`) and its HF config, as transformers builds it.

    `load_gguf_checkpoint`'s rules: every key goes through transformers'
    GGUF_TO_TRANSFORMERS_MAPPING["config"] (the special token ids included),
    the head is tied exactly when the file stores no `output.weight`, and a
    file without a vocab_size key takes its tokenizer's length. transformers
    then builds the model from the config class over those fields, so the
    class's defaults belong to the file's config too (Qwen3's head_dim 128 is
    one no metadata key states); `to_diff_dict` is config.json's own form of
    the result. Beside that table, the keys `_EXTRA` names are read under
    their fields. Any other architecture key (llama.cpp's `rope.scaling.*`,
    for one) is refused, not dropped as transformers drops it.
    """
    from transformers import AutoConfig
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING
    from transformers.modeling_gguf_pytorch_utils import GGUF_TO_TRANSFORMERS_MAPPING

    architecture = _architecture(reader)
    prefix = str(_parse(reader, "general.architecture"))
    section = _SECTIONS.get(architecture, architecture)
    extra = _EXTRA.get(architecture, {})
    fields: dict[str, object] = {
        "tie_word_embeddings": all(tensor.name != "output.weight" for tensor in reader.tensors),
        **GGUF_CONFIG_DEFAULTS_MAPPING.get(section, {}), **_mapped(reader, "config", {prefix: section}),
        **{field: _parse(reader, f"{prefix}.{key}") for key, field in extra.items()
           if f"{prefix}.{key}" in reader.fields}}
    fields.pop("model_type", None)
    if "vocab_size" not in fields and "tokenizer.ggml.tokens" in reader.fields:
        fields["vocab_size"] = len(reader.fields["tokenizer.ggml.tokens"].data)
    if architecture in _NORM_SHIFT:
        fields.update(_gemma_attention(architecture, fields))
    config = AutoConfig.for_model(_model_type(architecture), **fields).to_diff_dict()
    table = GGUF_TO_TRANSFORMERS_MAPPING["config"][section]
    head_dim = config.get("head_dim") or _records.integer(
        config["hidden_size"], "hidden_size"
    ) // _records.integer(config["num_attention_heads"], "num_attention_heads")
    for key in reader.fields:
        name = key.removeprefix(f"{prefix}.")
        if name == key or name in extra or (table.get(name, -1) is not None and name in table):
            continue
        if name in _HEAD_WIDTHS and _parse(reader, key) == head_dim:
            continue
        raise ValueError(
            f"GGUF metadata {key}={_parse(reader, key)!r} changes the model and transformers' GGUF table "
            f"does not read it, so the model would load without it; load the safetensors repo the file "
            "was quantized from (the model card's base_model)")
    return architecture, config


_GEMMA_27B = {"gemma2": 46, "gemma3": 62}
"""The depth llama.cpp recognizes each Gemma's 27B model by (src/models/gemma2.cpp, gemma3.cpp)."""


def _gemma_attention(architecture: str, fields: Mapping[str, object]) -> Mapping[str, object]:
    """What a Gemma file's attention computes where no metadata states it, as llama.cpp runs it: the
    query scale's divisor, the width over the heads for the 27B models and the head width otherwise
    (`f_attention_scale`, src/models/gemma2.cpp and gemma3.cpp), where the config class would default
    it to 256; and a Gemma 3 file without a positive sliding window attends every layer whole, which
    is how its converter writes a model of `sliding_window_pattern` 1 (conversion/gemma.py:156-157)."""
    layers, hidden = _records.integer(fields["num_hidden_layers"], "num_hidden_layers"), \
        _records.integer(fields["hidden_size"], "hidden_size")
    heads = _records.integer(fields["num_attention_heads"], "num_attention_heads")
    width = _records.integer(fields.get("head_dim", hidden // heads), "head_dim")
    stated: dict[str, object] = {
        "query_pre_attn_scalar": hidden // heads if layers == _GEMMA_27B[architecture] else width}
    if architecture == "gemma3" and not fields.get("sliding_window"):
        stated.update(sliding_window=None, layer_types=["full_attention"] * layers)
    return stated


def _mapped(reader: GGUFReader, section: str,
            prefixes: Mapping[str, str] | None = None) -> Mapping[str, object]:
    """The file's metadata under transformers' names for one section of
    GGUF_TO_TRANSFORMERS_MAPPING ('config', 'tokenizer' or 'tokenizer_config'), each key's prefix
    read as `prefixes` renames it (an architecture transformers' table keys by another name)."""
    from transformers.modeling_gguf_pytorch_utils import GGUF_TO_TRANSFORMERS_MAPPING

    table = GGUF_TO_TRANSFORMERS_MAPPING[section]
    mapped: dict[str, object] = {}
    for key in reader.fields:
        prefix, _, name = key.partition(".")
        renamed = table.get((prefixes or {}).get(prefix, prefix), {}).get(name)
        if renamed is not None and renamed != -1:
            mapped[renamed] = _parse(reader, key)
    return mapped


def tokenizer(path: str | os.PathLike[str]) -> PreTrainedTokenizerFast:
    """The tokenizer a GGUF file carries, built as transformers builds it
    (`TokenizersBackend` with `gguf_file`) from the same metadata, without
    torch: transformers' own route reads it through `load_gguf_checkpoint`,
    which refuses to run without torch."""
    from gguf import GGUFReader
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    reader = GGUFReader(path)
    architecture, _ = _config(reader)
    # transformers' converters go by model_type; a Mixtral file's vocabulary is Llama's.
    converter = "llama" if architecture == MIXTRAL else _model_type(architecture)
    backend, extra = convert_gguf_tokenizer(converter, _mapped(reader, "tokenizer"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, **_mapped(reader, "tokenizer_config"), **extra)


def _hf_names(architecture: str, layers: int) -> dict[str, str]:
    """Map each GGUF tensor stem this architecture can hold to its HF module name.

    transformers resolves a GGUF name by asking gguf-py's table for each name in
    the torch model's state dict; the dense decoder schema the safetensors codec
    maps (`decoder_parts.dense_modules`) stands in for that state dict, with
    Gemma's two feed-forward norms and each MoE architecture's routers, shared
    expert and stacked experts (`_EXPERTS`, one module for every expert's projection)
    beside it, and the same table answers."""
    from gguf import MODEL_ARCH_NAMES, get_tensor_name_map

    from dew.interop.decoder_parts import dense_modules

    file_arch = "llama" if architecture == MIXTRAL else architecture
    arch = next(key for key, value in MODEL_ARCH_NAMES.items() if value == file_arch)
    table = get_tensor_name_map(arch, layers)
    block = ["pre_feedforward_layernorm", "post_feedforward_layernorm"] if architecture in _NORM_SHIFT else []
    if architecture in _EXPERTS:
        experts, projections = _EXPERTS[architecture]
        block += [f"{experts}.{projection}" for projection in projections]
        block += (["block_sparse_moe.gate"] if architecture == MIXTRAL else
                  ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj",
                   "mlp.shared_expert_gate"])
    modules = (*dense_modules(layers),
               *(f"model.layers.{layer}.{name}" for layer in range(layers) for name in block))
    return {stem: name for name in modules if (stem := table.get_name(name)) is not None}


def unpermute(weight: np.ndarray, heads: int) -> np.ndarray:
    """Undo llama.cpp's rotary row order on a Llama q_proj or k_proj weight.

    llama.cpp's converter reorders each head's rows from the HF half-split
    rotary layout to interleaved pairs (convert_hf_to_gguf.py `LlamaModel.permute`);
    this is transformers' inverse, `LlamaTensorProcessor._reverse_permute_weights`
    (modeling_gguf_pytorch_utils.py:105-115, 5.16.1). `heads` is the query
    head count for q_proj and the key/value head count for k_proj.
    """
    half = weight.shape[0] // heads // 2
    pairs = weight.reshape(heads, half, 2, *weight.shape[1:])
    return pairs.swapaxes(2, 1).reshape(weight.shape)


def _values(tensor: ReaderTensor) -> np.ndarray:
    """One tensor's values in HF layout: float storage as stored, a block format dequantized to float32."""
    from gguf import GGMLQuantizationType, dequantize

    shape = tuple(reversed(tensor.shape.tolist()))
    if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return tensor.data.reshape(shape)
    if tensor.tensor_type == GGMLQuantizationType.BF16:
        return tensor.data.view(ml_dtypes.bfloat16).reshape(shape)
    return dequantize(tensor.data, tensor.tensor_type).reshape(shape)


def read(path: str | os.PathLike[str]) -> tuple[Mapping[str, object], dict[str, np.ndarray]]:
    """Read one GGUF file as its HF config dict and its tensors under HF names.

    F32, F16 and BF16 tensors stay memory mapped in their stored dtype, but
    for Llama's reordered Q/K rows and Gemma's shifted norms; every block
    format is dequantized to float32 by gguf-py. A stacked expert tensor loads
    as one view per expert (`_EXPERTS`). A tensor the name table does not
    place (Llama 3's `rope_freqs`, which carries a frequency scaling the
    metadata does not state) is refused rather than dropped.
    """
    try:
        from gguf import GGUFReader
    except ImportError as missing:
        raise ImportError("reading a GGUF file needs the gguf extra: "
                          "pip install 'dewml[gguf]'") from missing
    reader = GGUFReader(path)
    architecture, config = _config(reader)
    layers = config.get("num_hidden_layers")
    if not isinstance(layers, int):
        raise ValueError(f"{path} states no block_count, so its layers are unknown")
    stems = _hf_names(architecture, layers)
    heads = {"attn_q": config.get("num_attention_heads"), "attn_k": config.get("num_key_value_heads")}
    experts = _EXPERTS.get(architecture, ("", ()))[0]
    tensors: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        stem, _, suffix = tensor.name.rpartition(".")
        if stem not in stems:
            raise ValueError(
                f"GGUF tensor {tensor.name!r} has no counterpart among the {architecture} "
                "HF tensor names; load the safetensors repo the file was quantized from "
                "(the model card's base_model)")
        values = _values(tensor)
        projection = stem.rpartition(".")[2]
        if architecture in ("llama", MIXTRAL) and projection in heads:
            count = heads[projection]
            if not isinstance(count, int):
                raise ValueError(
                    f"{path} states no head counts, so its {projection} rows cannot be reordered"
                )
            values = unpermute(values, count)
        if architecture in _NORM_SHIFT and tensor.name.endswith("norm.weight"):
            values = values - np.ones((), values.dtype)
        if projection == "ffn_gate_inp_shexp":
            # Qwen2-MoE's shared expert gate is a [1, hidden] Linear; older converters store its one
            # row flat, the current one keeps the Linear's shape.
            values = values.reshape(1, -1)
        name = stems[stem]
        prefix, stacked, module = name.rpartition(f".{experts}.")
        if experts and stacked:
            for expert, rows in enumerate(values):
                tensors[f"{prefix}.{experts}.{expert}.{module}.{suffix}"] = rows
        else:
            tensors[f"{name}.{suffix}"] = values
    return config, tensors
