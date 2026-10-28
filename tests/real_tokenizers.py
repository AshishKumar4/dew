"""Real tokenizers for decision tests at a real geometry, and a tiny Qwen3.5 that reads each: GPT-2's,
which the tests bring (tests/fixtures/tokenizers/gpt2), and Qwen3.5-0.8B's own, pinned to its revision,
which a test reads from the Hub (marked network). A run's rows letter as many options as the tokenizer
spells labels for and tokenize as it does."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "hf" / "qwen38-dense-tiny"
QWEN35, QWEN35_REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
"""The released Qwen3.5 whose tokenizer the network tests read, at the revision they pin."""
TOKENIZERS = [pytest.param("gpt2", id="gpt2"),
              pytest.param("qwen3.5", id="qwen3.5", marks=pytest.mark.network)]
"""The tokenizers a test reads: GPT-2's in every run, Qwen3.5's where the Hub is reachable."""


def tokenizer_files(name: str) -> Path:
    """The directory of `name`'s tokenizer files (`TOKENIZERS`)."""
    if name == "gpt2":
        return FIXTURES / "tokenizers" / "gpt2"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(QWEN35, revision=QWEN35_REVISION, allow_patterns=[
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"]))


def real_tokenizer(name: str):
    """`name`'s tokenizer (`TOKENIZERS`)."""
    from dew.data.text import HFTokenizer

    return HFTokenizer(str(tokenizer_files(name)), local_files_only=True)


def backbone_reading(name: str, directory: Path) -> Path:
    """The tiny Qwen3.5 fixture written to `directory` reading `name`'s tokenizer: its two
    vocabulary-sized tensors drawn at that tokenizer's vocabulary, each new row as the fixture's are
    drawn (seeded), and its vocabulary size in the config; every other weight the fixture's own."""
    from safetensors.numpy import load_file, save_file
    from tokenizers import Tokenizer

    files = tokenizer_files(name)
    shutil.copytree(TINY, directory)
    for path in files.iterdir():
        if path.name != "config.json":
            shutil.copyfile(path, directory / path.name)
    vocabulary = Tokenizer.from_file(str(files / "tokenizer.json")).get_vocab_size(with_added_tokens=True)
    config = json.loads((TINY / "config.json").read_text())
    config["text_config"]["vocab_size"] = vocabulary
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    tensors = load_file(TINY / "model.safetensors")
    rng = np.random.default_rng(0)
    for tensor in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        own = tensors[tensor]
        drawn = rng.normal(0.0, float(own.std()), (vocabulary, own.shape[1])).astype(own.dtype)
        held = min(vocabulary, own.shape[0])
        drawn[:held] = own[:held]
        tensors[tensor] = drawn
    save_file(tensors, directory / "model.safetensors")
    return directory
