"""The native multimodal families' fixtures, data alone, so the tests and the reference generator
(tools/multimodal_reference.py, a Torch-only environment) read one declaration."""

FAMILIES = {
    "gemma4": ("gemma4-native-tiny", "gemma4-native-tiny"),
    "qwen35": ("qwen35-native-tiny", "qwen35-native-tiny"),
    "llama4": ("llama4-native-tiny", "llama4-native-tiny"),
    # Images and audio through the same prompt; the checkpoint and its
    # processor are the audio fixture's, the references sit beside them.
    "gemma3n": ("gemma3n-native-tiny", "gemma-3n-audio-tiny"),
}
"""Each family's directory of references under tests/fixtures/hf, beside their float64 truths
(tools/multimodal_reference.py float64), and the checkpoint they were written from."""
