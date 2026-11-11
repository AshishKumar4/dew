"""The text corpus the TFDS fixture holds (tools/datasets/tfds_reference.py's `DewPassages`) and
tests/test_text_backends.py reads from every store, with no TensorFlow."""

PASSAGES = [f"passage {index}: " + "the quick brown fox jumps over the lazy dog " * (1 + index % 4)
            for index in range(24)]
"""Twenty-four passages, each its index and one to four repetitions of a pangram, in this order."""
