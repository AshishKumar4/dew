"""MiMo's audio tokenizer and encoder as a release stores them: their tensors' names, read one way
(`Unwritten`), and their config's fields, for the model `dew.nn.mimo_audio` computes."""

from collections.abc import Mapping

import numpy as np

from dew import _records
from dew.batch import Variables
from dew.interop.decoder_parts import folder_name
from dew.interop.layouts import Unwritten
from dew.interop.unread import Stored, held
from dew.interop.weights import Tree, checkpoint_dtype, insert, placed
from dew.nn.mimo_audio import MiMoAudio
from dew.registry import Record

from .common import TowerLayout


def translate_mimo_audio_config(audio_config: Mapping[str, object],
                                tokenizer_config: Mapping[str, object]) -> Record:
    """A MiMo-V2 audio_config and its audio tokenizer's config into a MiMoAudio value's fields."""
    if str(tokenizer_config.get("ln_type", "LayerNorm")) != "LayerNorm":
        raise ValueError(f"ln_type {tokenizer_config.get('ln_type')!r} is not the tokenizer's layer norm")
    if str(tokenizer_config.get("activation_function", "gelu")) != "gelu":
        raise ValueError("the tokenizer's feed-forward is GELU")
    if _records.number(audio_config.get("partial_rotary_factor", 1.0), "partial_rotary_factor") != 1.0:
        raise ValueError("the audio encoder rotates every head dimension")
    channels = _records.integer(audio_config["audio_channels"], "audio_channels")
    # The release writes the vocabularies as text: one size for every channel, or each one's, dashed.
    vocab = str(audio_config.get("speech_vocab_size", 1280))
    sizes = [_records.integer(int(size), "speech_vocab_size") for size in vocab.split("-")]
    sizes = sizes if len(sizes) > 1 else sizes * channels
    stages = _records.integer(tokenizer_config.get("num_quantizers", 20), "num_quantizers")
    books = tokenizer_config.get("codebook_size", 1024)
    books = _records.integers(books, "codebook_size") if isinstance(books, (list, tuple)) else (
        _records.integer(books, "codebook_size"),) * stages
    books = (*books, *(books[-1],) * (stages - len(books)))[:stages]
    skip = tokenizer_config.get("encoder_skip_layer_id")
    tokenizer = {
        "n_mels": _records.integer(tokenizer_config["n_mels"], "n_mels"),
        "sampling_rate": _records.integer(tokenizer_config["sampling_rate"], "sampling_rate"),
        "nfft": _records.integer(tokenizer_config["nfft"], "nfft"),
        "hop_length": _records.integer(tokenizer_config["hop_length"], "hop_length"),
        "window_size": _records.integer(tokenizer_config["window_size"], "window_size"),
        "fmin": _records.number(tokenizer_config.get("fmin", 0.0), "fmin"),
        "fmax": (None if tokenizer_config.get("fmax") is None
                 else _records.number(tokenizer_config["fmax"], "fmax")),
        "d_model": _records.integer(tokenizer_config["d_model"], "d_model"),
        "kernel_size": _records.integer(tokenizer_config["kernel_size"], "kernel_size"),
        "stride_size": _records.integer(tokenizer_config["stride_size"], "stride_size"),
        "avg_pooler": _records.integer(tokenizer_config["avg_pooler"], "avg_pooler"),
        "encoder_layers": _records.integer(tokenizer_config["encoder_layers"], "encoder_layers"),
        "encoder_attention_heads": _records.integer(tokenizer_config["encoder_attention_heads"],
                                                   "encoder_attention_heads"),
        "encoder_ffn_dim": _records.integer(tokenizer_config["encoder_ffn_dim"], "encoder_ffn_dim"),
        "encoder_causal": _records.boolean(tokenizer_config.get("encoder_causal", False), "encoder_causal"),
        "encoder_attn_window_size": _records.integers(
            tokenizer_config.get("encoder_attn_window_size", [-1, -1]), "encoder_attn_window_size"),
        "encoder_skip_layer_id": None if skip is None else _records.integer(skip, "encoder_skip_layer_id"),
        "hybrid_attention": _records.boolean(tokenizer_config.get("hybrid_attention", False),
                                            "hybrid_attention"),
        "swa_per_block": _records.integer(tokenizer_config.get("swa_per_block", 2), "swa_per_block"),
        "rope_theta": _records.number(tokenizer_config.get("rope_theta", 10000.0), "rope_theta"),
        "codebook_size": books,
    }
    return {"class": "mimo_audio", "fields": {
        "tokenizer": tokenizer,
        "audio_channels": channels,
        "group_size": _records.integer(audio_config["group_size"], "group_size"),
        "input_local_dim": _records.integer(audio_config["input_local_dim"], "input_local_dim"),
        "input_local_layers": _records.integer(audio_config["input_local_layers"], "input_local_layers"),
        "input_local_attn_heads": _records.integer(audio_config["input_local_attn_heads"],
                                                  "input_local_attn_heads"),
        "input_local_intermediate_size": _records.integer(audio_config["input_local_intermediate_size"],
                                                         "input_local_intermediate_size"),
        "input_full_attention": _records.boolean(audio_config.get("input_full_attention", True),
                                                "input_full_attention"),
        "rope_theta": _records.number(audio_config.get("rope_theta", 640000.0), "rope_theta"),
        "add_post_norm": _records.boolean(audio_config.get("add_post_norm", True), "add_post_norm"),
        "projection_layers": _records.integer(audio_config.get("projection_layers", 2), "projection_layers"),
        "out_hidden_size": _records.integer(audio_config["out_hidden_size"], "out_hidden_size"),
        "speech_vocab_size": tuple(sizes),
        "audio_segment_size": _records.integer(audio_config.get("audio_segment_size", 6000),
                                              "audio_segment_size"),
    }}


_TOKENIZER_NORMS = {"self_attn_layer_norm": "input_layernorm", "final_layer_norm": "post_attention_layernorm"}


_TOKENIZER_MAPS = {("self_attn", "q_proj"): ("self_attn", "q_proj"),
                   ("self_attn", "k_proj"): ("self_attn", "k_proj"),
                   ("self_attn", "v_proj"): ("self_attn", "v_proj"),
                   ("self_attn", "out_proj"): ("self_attn", "o_proj"), ("fc1",): ("mlp", "up_proj"),
                   ("fc2",): ("mlp", "down_proj")}


def mimo_audio_tokenizer_path(name: str) -> tuple[str, ...]:
    """One audio tokenizer tensor, its `encoder.` prefix off, into the tower's tree; anything else
    raises ValueError."""
    parts = name.split(".")
    leaf = parts[-1]
    kind = "kernel" if leaf == "weight" else "bias"
    path: tuple[str, ...] | None = None
    if parts[0] in ("conv1", "conv2"):
        path = ("tokenizer", parts[0], kind)
    elif parts[:2] == ["down_sample_layer", "0"]:
        path = ("tokenizer", "down_sample_layer", kind)
    elif parts[0] in ("layer_norm", "down_sample_norm"):
        path = ("tokenizer", parts[0], "scale" if leaf == "weight" else "bias")
    elif parts[:2] == ["quantizer", "vq"] and parts[-1] == "embed" and parts[3].isdigit():
        path = ("quantizer", f"codebook_{parts[3]}")
    elif parts[0] == "layers" and parts[1].isdigit():
        layer, rest = f"layers_{parts[1]}", tuple(parts[2:-1])
        if rest[0] in _TOKENIZER_NORMS and len(rest) == 1:
            path = ("tokenizer", layer, _TOKENIZER_NORMS[rest[0]], "scale" if leaf == "weight" else "bias")
        elif rest in _TOKENIZER_MAPS:
            path = ("tokenizer", layer, *_TOKENIZER_MAPS[rest], kind)
    if path is None:
        raise ValueError(f"unknown tensor name {name!r}")
    return path


def mimo_audio_encoder_path(name: str) -> tuple[str, ...]:
    """One audio encoder tensor (`speech_embeddings.*`, or `audio_encoder.*` its prefix off) into
    the tower's tree; anything else raises ValueError."""
    parts = name.split(".")
    leaf = parts[-1]
    path: tuple[str, ...] | None = None
    if parts[0] == "speech_embeddings" and parts[1].isdigit():
        path = ("encoder", f"speech_embeddings_{parts[1]}", "embedding")
    elif parts[0] == "projection":
        site = {"mlp.0": "projection_fc1", "mlp.2": "projection_fc2"}.get(".".join(parts[1:-1]), "projection")
        path = ("encoder", site, "kernel")
    elif parts[:2] == ["input_local_transformer", "norm"]:
        path = ("encoder", "norm", "scale")
    elif parts[:2] == ["input_local_transformer", "layers"] and parts[2].isdigit():
        layer, rest = f"layers_{parts[2]}", tuple(parts[3:-1])
        if rest in (("input_layernorm",), ("post_attention_layernorm",)):
            path = ("encoder", layer, rest[0], "scale")
        elif len(rest) == 2 and rest[0] in ("self_attn", "mlp"):
            path = ("encoder", layer, *rest, "kernel" if leaf == "weight" else "bias")
    if path is None:
        raise ValueError(f"unknown tensor name {name!r}")
    return path


AUDIO_TOKENIZER = "audio_tokenizer"
"""The folder a MiMo-V2.6 release keeps its audio tokenizer in, its own config and weights, which the
release's code loads beside the model (`MiMoV2ForCausalLM.load_audio_tokenizer`); a bundle's names hold
its tensors under it (`dew.interop.decoder_parts.folder_name`)."""


_TOKENIZER_ENCODER = folder_name(AUDIO_TOKENIZER, "encoder.")
"""The tokenizer's encoder in a bundle's names: the half its MiMoAudioTokenizer holds."""


_STATISTICS = ("inited", "cluster_size", "embed_avg")
"""The quantizer's running statistics beside its codebooks, which no code it gives reads."""


def mimo_audio_path(name: str) -> tuple[str, ...] | None:
    """One tensor of a MiMo-V2.6 bundle's audio tower, by its name in the bundle, into its path under
    `params`: the encoder's (`audio_encoder.`, `speech_embeddings.`) and the tokenizer folder's encoder's;
    None for the quantizer's running statistics, which the tower does not compute with."""
    if name.startswith(_TOKENIZER_ENCODER):
        rest = name.removeprefix(_TOKENIZER_ENCODER)
        return None if rest.split(".")[-1] in _STATISTICS else ("params", *mimo_audio_tokenizer_path(rest))
    return ("params", *mimo_audio_encoder_path(name.removeprefix("audio_encoder.")))


def _read_mimo_audio(tensors: Mapping[str, Stored], path_of, param_dtype: str
                     ) -> tuple[Tree[Stored], tuple]:
    """The tower's variables and the layouts that write its tensors back (`placed`), each of the
    tokenizer's key maps, which carry no bias, beside a zero one."""
    variables, layouts = placed(tensors, path_of, param_dtype)
    for name, tensor in tensors.items():
        bias = name.removesuffix("weight") + "bias"
        path = path_of(name) if name.endswith("k_proj.weight") and bias not in tensors else None
        if path is not None:
            insert(variables, (*path[:-1], "bias"),
                   held(np.zeros(tensor.shape[0], checkpoint_dtype(held(tensor).dtype, param_dtype))), bias)
    return variables, layouts


_LAYOUT = TowerLayout("", Unwritten(lambda name, config: mimo_audio_path(name)), read=_read_mimo_audio,
                      roots=("audio_encoder.", "speech_embeddings.", _TOKENIZER_ENCODER))


MIMO_AUDIO = _LAYOUT


def translate_mimo_audio_weights(tokenizer: Mapping[str, np.ndarray], encoder: Mapping[str, np.ndarray], *,
                                 param_dtype: str = "float32") -> Variables:
    """The tower's variables from the tokenizer's tensors (`encoder.` prefix off) and the audio
    encoder's (`audio_encoder.` off, `speech_embeddings.` kept), as a bundle's names read them
    (`MIMO_AUDIO`)."""
    named = {**{f"{_TOKENIZER_ENCODER}{name}": tensor for name, tensor in tokenizer.items()},
             **{name if name.startswith("speech_embeddings.") else f"audio_encoder.{name}": tensor
                for name, tensor in encoder.items()}}
    return _LAYOUT.variables(named, param_dtype)[0]

TOWERS = {MiMoAudio: MIMO_AUDIO}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""
