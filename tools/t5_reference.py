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
  vocab list, no SentencePiece model involved), and the prompts with the
  reference tokenizer's ids and mask. The config uses
  gated-gelu, the T5 v1.1 feed-forward SD3.5 and Flux run, which the t5-small
  network test cannot cover (v1.0 is plain relu). Small enough to live in
  git, so the parity tests need no network.
- tiny-umt5/: the same for a random-weight UMT5 encoder (`UMT5EncoderModel`,
  three layers), whose every layer holds its own relative bias table, as
  Wan 2.1's umt5-xxl text encoder does.

`seeds` records each checkpoint's encoder over check 1's seeds instead
(`record_case`), as `<checkpoint>-seeds.npz` beside the checkpoint it reads,
which it leaves as it is:
the reference's fp32 last hidden state and its float64 truth on each seed's
drawn ids (tests/test_t5_encoders.py's `t5_draw`), kept as the evidence check
1 reads (`summary`), the count planned on Dew's own encoder:

    /tmp/t5ref/bin/python tools/t5_reference.py seeds
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
             attention_mask=tokens["attention_mask"].numpy().astype(np.int32))
    size_mb = sum(p.stat().st_size for p in tiny.iterdir()) / 2 ** 20
    print(f"wrote {tiny} ({size_mb:.1f} MB): ids {tuple(tokens['input_ids'].shape)}")


def write_seeds(directory: Path, name: str, reader) -> None:
    """`directory`'s checkpoint recorded over check 1's seeds as `<directory>-seeds.npz` beside it
    (`record_case`): the
    reference's fp32 encoder and its float64 truth, built and run inside `torch_precision.float64`, on
    each seed's drawn ids, the count planned on Dew's own encoder against its float64 run."""
    import sys

    import jax

    sys.path[:0] = [str(FIXTURES.parents[1]), str(Path(__file__).resolve().parent)]
    from recorded_references import provenance, record_case, save
    from test_t5_encoders import t5_draw
    from torch_precision import float64

    from dew.inputs import T5Text

    torch.set_num_threads(1)
    model = reader.from_pretrained(directory, dtype=torch.float32).eval()
    with float64():
        wide = reader.from_pretrained(directory, dtype=torch.float64).eval()

    def evaluate(inputs):
        ids = torch.from_numpy(inputs["input_ids"]).long()
        mask = torch.from_numpy(inputs["attention_mask"]).long()
        with torch.no_grad():
            reference = model(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()
            with float64():
                truth = wide(input_ids=ids, attention_mask=mask).last_hidden_state.numpy()
        return {"last_hidden_state": reference}, {"last_hidden_state": truth}

    encoder = T5Text.from_pretrained(str(directory))
    with jax.enable_x64(new_val=True):
        wide_encoder = T5Text.from_pretrained(str(directory), dtype="float64", param_dtype="float64")

    def proxy(inputs):
        return {"last_hidden_state": encoder.encode(encoder.params, inputs).hidden}

    def truth_proxy(inputs):
        return {"last_hidden_state": wide_encoder.encode(wide_encoder.params, inputs).hidden}

    arrays = record_case(name, lambda seed: t5_draw(seed, model.config.vocab_size), evaluate, proxy,
                         truth_proxy=truth_proxy, store_inputs=False, summary=True)
    import transformers

    arrays[f"{name}.provenance"] = provenance("transformers", transformers.__version__, "cpu",
                                              "python tools/t5_reference.py seeds", torch=torch.__version__)
    print(save(directory.parent / f"{directory.name}-seeds.npz", arrays,
               reference=f"transformers {transformers.__version__}",
               license="Apache-2.0"), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="?", choices=("checkpoints", "seeds"), default="checkpoints")
    parser.add_argument("--out", type=Path, default=FIXTURES)
    arguments = parser.parse_args(argv)
    out = arguments.out
    if arguments.what == "seeds":
        write_seeds(out / "tiny", "t5", T5EncoderModel)
        write_seeds(out / "tiny-umt5", "umt5", UMT5EncoderModel)
        return
    write_tiny(out / "tiny")
    write_tiny(out / "tiny-umt5", tiny_umt5)


if __name__ == "__main__":
    main()
