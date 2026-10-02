#!/usr/bin/env python3
"""Write the T5 fixtures tests/test_t5_encoders.py checks against.

Everything here runs under torch and transformers, which dew does not depend
on, so this is the only place the reference T5 encoder is executed. The
fixtures it writes are what CI compares against.

Set up the venv once (torch CPU, transformers, safetensors):

    uv venv /tmp/t5ref --python 3.12
    uv pip install --python /tmp/t5ref/bin/python torch transformers \
        safetensors numpy \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
    /tmp/t5ref/bin/python tools/t5_reference.py

--out DIR writes the fixtures somewhere other than tests/fixtures/t5.

What lands in tests/fixtures/t5:

- tiny/: a random-weight T5 encoder in the Hugging Face layout
  (config.json + model.safetensors holding the full T5 state dict, so the
  loader proves it reads past the decoder and lm_head), a tokenizer over an
  explicit ASCII vocabulary (transformers 5 builds the T5 tokenizer from a
  vocab list, no SentencePiece model involved), the prompts and the fp32
  last hidden states of the reference in eval mode. The config uses
  gated-gelu, the T5 v1.1 feed-forward SD3.5 and Flux run, which the t5-small
  network test cannot cover (v1.0 is plain relu). Small enough to live in
  git, so the parity tests need no network.
- tiny-umt5/: the same for a random-weight UMT5 encoder (`UMT5EncoderModel`,
  three layers), whose every layer holds its own relative bias table, as
  Wan 2.1's umt5-xxl text encoder does.
"""

import argparse
import json
import string
from pathlib import Path

import numpy as np
import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer, UMT5Config, UMT5EncoderModel

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "t5"

TINY_PROMPTS = [
    "a red bird",
    "",
    "two cats on a mat, painted",
    "the sun set over hills and the harbour lights came on one by one",
    "x",
]


def tiny_tokenizer() -> T5Tokenizer:
    """A T5 tokenizer over single characters, a few kilobytes of vocabulary.

    The tokenizer falls back to single characters where no multi-character
    piece matches, so the printable ASCII characters plus the word-start
    marker tokenize any of the prompts below. No extra ids: the tiny model
    has no sentinel use.
    """
    pieces = ["<pad>", "</s>", "<unk>", "\u2581"]
    pieces += list(string.ascii_lowercase + string.digits + string.punctuation)
    return T5Tokenizer(vocab=[(piece, 0.0) for piece in pieces], extra_ids=0)


def tiny_model(vocab_size: int) -> T5EncoderModel:
    """A random T5 encoder small enough to commit, gated-gelu like T5-XXL."""
    config = T5Config(
        vocab_size=vocab_size, d_model=64, d_kv=16, d_ff=128,
        num_layers=2, num_heads=4, relative_attention_num_buckets=16,
        relative_attention_max_distance=64, dropout_rate=0.0,
        layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu",
        pad_token_id=0, eos_token_id=1)
    # Every published T5 config.json carries this field. transformers 5
    # accepts it at construction but its typed signature does not declare
    # it, so it is set the way the loader would read it.
    config.decoder_start_token_id = 0
    torch.manual_seed(0)
    return T5EncoderModel(config)


def tiny_umt5(vocab_size: int) -> UMT5EncoderModel:
    """A random UMT5 encoder: the tiny T5's widths, three layers, so two
    later layers each read a table of their own."""
    config = UMT5Config(
        vocab_size=vocab_size, d_model=64, d_kv=16, d_ff=128,
        num_layers=3, num_heads=4, relative_attention_num_buckets=16,
        relative_attention_max_distance=64, dropout_rate=0.0,
        layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu",
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    torch.manual_seed(1)
    return UMT5EncoderModel(config)


def write_tiny(tiny: Path, build=tiny_model) -> None:
    tiny.mkdir(parents=True, exist_ok=True)
    tokenizer = tiny_tokenizer()
    model = build(tokenizer.vocab_size)
    model.config.save_pretrained(tiny)
    # The tokenizer files the loader reads back.
    tokenizer.save_pretrained(tiny)

    tokens = tokenizer(TINY_PROMPTS, padding=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=tokens["input_ids"],
                        attention_mask=tokens["attention_mask"])
    hidden = outputs.last_hidden_state.float().numpy()
    if not np.all(np.isfinite(hidden)):
        raise ValueError("the reference encoder produced a non-finite hidden state")

    # Full state dict on purpose: decoder and lm_head ride along so the
    # loader proves it reads past them. The shared embedding is stored twice
    # under both names, so clone before saving.
    from safetensors.torch import save_file

    save_file({name: tensor.clone() for name, tensor in model.state_dict().items()},
              tiny / "model.safetensors")
    (tiny / "prompts.json").write_text(json.dumps(
        {"prompts": TINY_PROMPTS, "max_length": int(tokens["input_ids"].shape[1])},
        indent=2) + "\n")
    np.savez(tiny / "reference.npz",
             input_ids=tokens["input_ids"].numpy().astype(np.int32),
             attention_mask=tokens["attention_mask"].numpy().astype(np.int32),
             last_hidden_state=hidden)
    size_mb = sum(p.stat().st_size for p in tiny.iterdir()) / 2 ** 20
    print(f"wrote {tiny} ({size_mb:.1f} MB): hidden {hidden.shape}, ids {tuple(tokens['input_ids'].shape)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=FIXTURES)
    out = parser.parse_args(argv).out
    write_tiny(out / "tiny")
    write_tiny(out / "tiny-umt5", tiny_umt5)


if __name__ == "__main__":
    main()
