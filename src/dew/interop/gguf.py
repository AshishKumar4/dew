"""Read a GGUF file as the HF config and safetensors-named tensors a decoder load reads.

A GGUF file carries its own config (key/value metadata), its tokenizer and
its tensors under llama.cpp's names, most of them quantized in blocks. This
module turns one into what `load_pretrained` reads from a safetensors repo:
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
from pathlib import Path
from typing import TYPE_CHECKING

import ml_dtypes
import numpy as np

if TYPE_CHECKING:
    from gguf import GGUFReader, ReaderTensor
    from transformers import PreTrainedTokenizerFast

_ARCHITECTURES = ("llama", "qwen2", "qwen3")
"""The GGUF architectures read here. Each is a Llama-convention dense decoder
whose `general.architecture` is also its HF model_type and its Dew family.
Architectures whose llama.cpp conversion rewrites more than the Q/K rows
(Gemma adds one to its norm scales, MoE files fuse experts) are refused."""

_TOP_NAMES = ("model.embed_tokens", "model.norm", "lm_head")
_LAYER_NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "self_attn.q_norm", "self_attn.k_norm", "input_layernorm",
                "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
"""The module names transformers' LlamaForCausalLM, Qwen2ForCausalLM and
Qwen3ForCausalLM hold. transformers resolves a GGUF name by asking gguf-py's
table for each name in the torch model's state dict; these names stand in
for that state dict, and the same table answers."""


def _parse(reader: GGUFReader, key: str) -> object:
    """One metadata value as transformers parses it: a one-element array is its element."""
    from transformers.integrations.ggml import _gguf_parse_value

    field = reader.fields[key]
    values = [_gguf_parse_value(field.parts[index], field.types) for index in field.data]
    return values[0] if len(values) == 1 else values


def _config(reader: GGUFReader) -> tuple[str, dict[str, object]]:
    """Return the file's architecture and its HF config, as transformers builds it.

    `load_gguf_checkpoint`'s rules for these architectures: every key goes
    through transformers' GGUF_TO_TRANSFORMERS_MAPPING["config"] (the
    special token ids included), the head is tied exactly when the file
    stores no `output.weight`, and a file without a vocab_size key takes
    its tokenizer's length. transformers then builds the model from the
    config class over those fields, so the class's defaults belong to the
    file's config too (Qwen3's head_dim 128 is one no metadata key states);
    `to_diff_dict` is config.json's own form of the result.
    """
    from transformers import AutoConfig
    from transformers.integrations.ggml import GGUF_CONFIG_DEFAULTS_MAPPING

    stated = _parse(reader, "general.architecture") if "general.architecture" in reader.fields else None
    if stated not in _ARCHITECTURES:
        raise ValueError(
            f"GGUF architecture {stated!r} is not read by Dew, which reads "
            f"{', '.join(_ARCHITECTURES)}; load the safetensors repo the file was quantized from "
            "(the model card's base_model)")
    architecture = str(stated)
    fields: dict[str, object] = {
        "tie_word_embeddings": all(tensor.name != "output.weight" for tensor in reader.tensors),
        **GGUF_CONFIG_DEFAULTS_MAPPING.get(architecture, {}), **_mapped(reader, "config")}
    fields.pop("model_type", None)
    if "vocab_size" not in fields and "tokenizer.ggml.tokens" in reader.fields:
        fields["vocab_size"] = len(reader.fields["tokenizer.ggml.tokens"].data)
    config = AutoConfig.for_model(architecture, **fields).to_diff_dict()
    return architecture, config


def _mapped(reader: GGUFReader, section: str) -> dict[str, object]:
    """The file's metadata under transformers' names for one section of
    GGUF_TO_TRANSFORMERS_MAPPING ('config', 'tokenizer' or 'tokenizer_config')."""
    from transformers.modeling_gguf_pytorch_utils import GGUF_TO_TRANSFORMERS_MAPPING

    table = GGUF_TO_TRANSFORMERS_MAPPING[section]
    mapped: dict[str, object] = {}
    for key in reader.fields:
        prefix, _, name = key.partition(".")
        renamed = table.get(prefix, {}).get(name)
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
    backend, extra = convert_gguf_tokenizer(architecture, _mapped(reader, "tokenizer"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, **_mapped(reader, "tokenizer_config"), **extra)


def _hf_names(architecture: str, layers: int) -> dict[str, str]:
    """Map each GGUF tensor stem this architecture can hold to its HF module name."""
    from gguf import MODEL_ARCH_NAMES, get_tensor_name_map

    arch = next(key for key, value in MODEL_ARCH_NAMES.items() if value == architecture)
    table = get_tensor_name_map(arch, layers)
    names = [*_TOP_NAMES, *(f"model.layers.{layer}.{name}"
                            for layer in range(layers) for name in _LAYER_NAMES)]
    return {stem: name for name in names if (stem := table.get_name(name)) is not None}


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


def read(path: str | os.PathLike[str]) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Read one GGUF file as its HF config dict and its tensors under HF names.

    F32, F16 and BF16 tensors stay memory mapped in their stored dtype, but
    for Llama's reordered Q/K rows; every block format is dequantized to
    float32 by gguf-py. A tensor the name table does not place (Llama 3's
    `rope_freqs`, which carries a frequency scaling the metadata does not
    state) is refused rather than dropped.
    """
    try:
        from gguf import GGUFReader
    except ImportError as missing:
        raise ImportError("reading a GGUF file needs the gguf extra: "
                          "pip install 'dew-ml[gguf]'") from missing
    reader = GGUFReader(path)
    architecture, config = _config(reader)
    layers = config.get("num_hidden_layers")
    if not isinstance(layers, int):
        raise ValueError(f"{path} states no {architecture}.block_count, so its layers are unknown")
    stems = _hf_names(architecture, layers)
    heads = {"attn_q": config.get("num_attention_heads"), "attn_k": config.get("num_key_value_heads")}
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
        if architecture == "llama" and projection in heads:
            count = heads[projection]
            if not isinstance(count, int):
                raise ValueError(f"{path} states no head counts, so its {projection} rows cannot be reordered")
            values = unpermute(values, count)
        tensors[f"{stems[stem]}.{suffix}"] = values
    return config, tensors


def resolve(name_or_dir: str | Path, directory: Path, gguf_file: str) -> Path:
    """Return the local path of `gguf_file` in a directory, or download it at the snapshot's commit.

    `directory` is the snapshot the metadata fetch resolved, named by its
    commit, so the file comes from that commit even if the branch moves.
    """
    if os.path.isdir(name_or_dir):
        path = directory / gguf_file
        if not path.is_file():
            present = sorted(entry.relative_to(directory).as_posix() for entry in directory.rglob("*.gguf"))
            raise FileNotFoundError(f"{path} does not exist; the GGUF files in {directory} are {present}")
        return path
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(str(name_or_dir), gguf_file, revision=directory.name))
    except EntryNotFoundError as error:
        from dew.interop.hf_decoders import _repo_files

        present = sorted(name for name in _repo_files(str(name_or_dir), directory) if name.endswith(".gguf"))
        raise FileNotFoundError(f"{name_or_dir} at {directory.name} has no {gguf_file!r}; "
                                f"its GGUF files are {present}") from error
