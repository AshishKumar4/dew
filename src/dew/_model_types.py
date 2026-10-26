"""Model-type spellings and tensor names shared by the decoder, wrapper and vision maps.

Vision translation reads these without importing the interop package, whose
public entry point imports the vision classes itself.
"""

QWEN35_TYPES = ('qwen3_5', 'qwen3_5_moe')
GEMMA4_TYPES = ('gemma4', 'gemma4_unified')
QWEN35_TEXT_TYPES = tuple(f'{name}_text' for name in QWEN35_TYPES)
_QWEN35_VISION_TYPES = (*QWEN35_TYPES, *(f'{name}_vision' for name in QWEN35_TYPES))

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
