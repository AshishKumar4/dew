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

OUTPUTS = ("logits", "loss", "pixel_gradient", "updated_logits")
"""The leaves a family's numerics give: its valid logits, its summed next-token loss, its pixel gradient
and its logits after one SGD step of every parameter (tools/multimodal_reference.py's `numerics`)."""
DRAWN = frozenset({"llama4"})
"""The families on check 1 over drawn images and audio, every leaf of theirs (`OUTPUTS`) resolved and
recorded as their site (`multimodal-<family>`, tools/multimodal_worker.py); the others hold their leaves
on their source draw until each resolves."""
LEARNING_RATE = 1e-4
"""The rate of the SGD step every family's references and Dew's runs take."""
