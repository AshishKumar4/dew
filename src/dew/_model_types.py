"""Model-type spellings shared by the Qwen decoder, wrapper and vision maps.

Vision translation reads these without importing the interop package, whose
public entry point imports the vision classes itself.
"""

QWEN35_TYPES = ('qwen3_5', 'qwen3_5_moe')
QWEN35_TEXT_TYPES = tuple(f'{name}_text' for name in QWEN35_TYPES)
_QWEN35_VISION_TYPES = (*QWEN35_TYPES, *(f'{name}_vision' for name in QWEN35_TYPES))
