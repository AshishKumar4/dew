"""Model-type spellings and tensor names shared by the decoder, wrapper and vision maps.

Vision translation reads these without importing the interop package, whose
public entry point imports the vision classes itself.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Wrapper:
    """What a multimodal wrapper type does that Dew reads by its kind, stated once for the type (`WRAPPERS`)
    rather than by lists of types each reader keeps."""

    qwen35: bool = False
    """Qwen 3.5's kind: its vision tower and merger, M-RoPE's rotary positions, its media tokens' type ids,
    and a prediction layer (`mtp.*`) at the checkpoint's root that the language model reads."""
    gemma4: bool = False
    """Gemma 4's kind: images paired with their patches' positions, and placeholder ids read as padding."""
    packed_media: bool = False
    """Images and videos packed alike, as runs of patch vectors beside their `(t, h, w)` grids, a video's
    frames each a run of its own (`Processor._qwen_frames`); Kimi's each item whole, its tower attending over
    an item's frames and pooling them (`TowerGeometry.pools_frames`)."""
    video: bool = False
    """Whether its processor carries videos (`pixel_values_videos`)."""
    head_tied_by_default: bool = True
    """Whether its head ties to its embedding where its config leaves `tie_word_embeddings` unset:
    Transformers ties by the wrapper config's own field, which Qwen 3.5's and Muse Glimmer's configs leave
    unset."""


_QWEN35 = Wrapper(qwen35=True, packed_media=True, video=True, head_tied_by_default=False)
_PACKED = Wrapper(packed_media=True, video=True)
_GEMMA4 = Wrapper(gemma4=True, video=True)

WRAPPERS: dict[str, Wrapper] = {
    'qwen3_5': _QWEN35, 'qwen3_5_moe': _QWEN35,
    # Qwen4-Exp's classes are Qwen 3.5's, class for class (modeling_qwen4_exp.py:1660-1964), its processor
    # Qwen3VLProcessor.
    'qwen4_exp': _QWEN35,
    'muse_glimmer': Wrapper(packed_media=True, video=True, head_tied_by_default=False),
    'lfm3_vl': _PACKED, 'mimo_v2': _PACKED, 'kimi_k25': _PACKED, 'kimi_k3': _PACKED,
    'gemma4': _GEMMA4, 'gemma4_unified': _GEMMA4,
}
"""Each multimodal wrapper type whose kind a reader asks of (`wrapper`)."""

_PLAIN = Wrapper()


def wrapper(model_type: str | None) -> Wrapper:
    """The kind of wrapper `model_type` is (`WRAPPERS`), a plain one where it names none."""
    return WRAPPERS.get(model_type or '', _PLAIN)


def config_wrapper(config: Mapping[str, object]) -> Wrapper:
    """The kind of wrapper a config's `model_type` names (`wrapper`), as the config holds it."""
    return next((kind for name, kind in WRAPPERS.items() if name == config.get('model_type')), _PLAIN)

SANDWICH_NORMS = {
    'input_layernorm': 'input_layernorm',
    'post_attention_layernorm': 'attention_output_norm',
    'pre_feedforward_layernorm': 'post_attention_layernorm',
    'post_feedforward_layernorm': 'mlp_output_norm',
}
"""A sandwich-normed layer's four norms, by their Hugging Face names, onto the decoder block's.

Three of the four move, because Hugging Face names a norm after the sublayer
it follows and the block names it after what it normalizes: HF's
post_attention_layernorm normalizes the attention output (the block's
attention_output_norm), its pre_feedforward_layernorm is the MLP's pre-norm
(the block's post_attention_layernorm), and its post_feedforward_layernorm
normalizes the MLP output (mlp_output_norm). Gemma's decoders and Gemma 4's
vision tower share the layout."""

LAYER_SCALAR_COLLECTIONS = {"trainable": "params", "frozen": "constants"}
"""The collection a decoder block keeps its layer scalar in, by mode (`BlockWiring.layer_scalar`):
Google's release makes it a parameter, and Transformers a buffer, which Dew keeps as a constant."""
