#!/usr/bin/env python3
"""Write the CLIP fixtures tests/test_text_encoders.py and tests/test_metrics.py
check against.

Everything here runs under torch and transformers, which dew does not depend
on, so this is the only place the reference implementation of the towers is
executed. The fixtures it writes are what CI compares against.

Set up the venv and run it:

    uv venv /tmp/clipref --python 3.12
    uv pip install --python /tmp/clipref/bin/python torch \
        --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python /tmp/clipref/bin/python transformers safetensors \
        numpy pillow
    HF_HOME=/tmp/clipref-hf /tmp/clipref/bin/python tools/clip_reference.py

What lands in tests/fixtures/clip:

- tiny/: a random-weight CLIP checkpoint in the Hugging Face layout
  (config.json + model.safetensors, both towers and the projection heads), a
  tokenizer small enough to commit, an image processor config at the tower's
  8 pixel input, the four prompts and four synthetic images it was run on, and
  the fp32 outputs of the reference in eval mode with eager attention: the
  text tower's last hidden states and pooled outputs, the projected text
  embeddings, the processor's pixel values and the projected image
  embeddings. Small enough to live in git, so the parity tests need no
  network. Its eos_token_id is 1, which is not the largest id in its
  vocabulary, so the pooled row it fixes is the one the eos branch of the
  reference finds. The images are taller than they are wide, so the
  processor's resize and centre crop both do work.
- large-patch14/: no weights, 340 MB of them would not fit. prompts.json holds
  the three prompts and the recipe for three synthetic images, config.json is
  the repo's own, and reference.npz the fp32 outputs of the real
  openai/clip-vit-large-patch14: the text tower's hidden states and pooled
  outputs and both projected embeddings, which the network tests compare
  against. That config carries eos_token_id 2, so it fixes the argmax branch
  of the pooled row.

--skip-real leaves large-patch14 alone, for a run without the 1.7 GB download.
--fp64 PATH also writes the real text tower's fp64 outputs to PATH, outside
the fixtures, which is the evidence behind the fp32 tolerance the test states.
"""

import argparse
import json
import shutil
import string
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer, CLIPConfig, CLIPImageProcessorPil, CLIPModel, CLIPTextConfig,
    CLIPTokenizer, CLIPVisionConfig,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "clip"
REAL_MODEL = "openai/clip-vit-large-patch14"
MAX_LENGTH = 77

TINY_PROMPTS = ["a red bird", "", "two cats on a mat, painted", "x"]
REAL_PROMPTS = [
    "",
    "a photograph of an astronaut riding a horse on the moon",
    "a dense oil painting of a harbour at dawn, fishing boats tied to a stone "
    "pier, gulls over the water, the town behind it still asleep, warm light "
    "on the rooftops and a cold blue shadow in the alleys",
]
# Synthetic images: seed and (count, height, width, channels). Noise is as good
# as a photograph for parity, and a recipe is smaller than pixels.
TINY_IMAGES = {"seed": 0, "shape": [4, 16, 12, 3]}
REAL_IMAGES = {"seed": 0, "shape": [3, 64, 48, 3]}


def synthetic_images(recipe) -> np.ndarray:
    """uint8 RGB images from a recipe, reproduced the same way by the tests."""
    return np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)


def tiny_tokenizer() -> CLIPTokenizer:
    """A CLIP tokenizer over single characters, a few kilobytes of vocabulary.

    CLIP's tokenizer is byte-level BPE with an end-of-word suffix. With no
    merges every character stays its own token, so a vocabulary of the
    printable ASCII characters and their word-final forms tokenizes any of the
    prompts below without reaching for the unknown token. bos is 0 and eos 1,
    both smaller than every character token, which is what makes the pooled row
    of the eos branch differ from the row the argmax branch would take.
    """
    characters = string.ascii_lowercase + string.digits + string.punctuation
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for character in characters:
        vocab[character] = len(vocab)
        vocab[character + "</w>"] = len(vocab)
    return CLIPTokenizer(vocab=vocab, merges=[], model_max_length=MAX_LENGTH)


def tiny_model(vocab_size: int) -> CLIPModel:
    """A random CLIP small enough to commit, both towers in the usual layout."""
    config = CLIPConfig(
        text_config=CLIPTextConfig(
            vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=MAX_LENGTH, layer_norm_eps=1e-5,
            hidden_act="quick_gelu", projection_dim=16,
            bos_token_id=0, eos_token_id=1, pad_token_id=1),
        vision_config=CLIPVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, image_size=8, patch_size=8,
            projection_dim=16),
        projection_dim=16)
    torch.manual_seed(0)
    return CLIPModel(config)


def encode(model: CLIPModel, tokenizer, prompts):
    """The reference text tower and its projection head on `prompts`, fp32,
    eval mode, eager attention."""
    model.eval()
    model.set_attn_implementation("eager")
    tokens = tokenizer(prompts, padding="max_length", max_length=MAX_LENGTH,
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.text_model(input_ids=tokens["input_ids"],
                                   attention_mask=tokens["attention_mask"])
        text_embeds = model.text_projection(outputs.pooler_output)
    return {
        "input_ids": tokens["input_ids"].to(torch.int32).numpy(),
        "attention_mask": tokens["attention_mask"].to(torch.int32).numpy(),
        "last_hidden_state": outputs.last_hidden_state.to(torch.float32).numpy(),
        "pooler_output": outputs.pooler_output.to(torch.float32).numpy(),
        "text_embeds": text_embeds.to(torch.float32).numpy(),
    }


def encode_images(model: CLIPModel, processor: CLIPImageProcessorPil, images):
    """The reference image processor, vision tower and projection head on
    uint8 images, fp32, eval mode, eager attention."""
    model.eval()
    model.set_attn_implementation("eager")
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = model.visual_projection(outputs.pooler_output)
    return {
        "images": images,
        "pixel_values": pixel_values.to(torch.float32).numpy(),
        "image_embeds": image_embeds.to(torch.float32).numpy(),
    }


def write_fp64(path: Path) -> None:
    """The real tower in fp64 throughout, the evidence behind the fp32 tolerance.

    The reference softmax is hardcoded to fp32 (eager_attention_forward), which
    caps any comparison at fp32 precision, so it is unwrapped here. Nothing
    committed reads this file; it is written where a run can point dew at it:

        JAX_ENABLE_X64=1 python - <<'PY'
        import functools, json, numpy as np, jax, jax.numpy as jnp
        import dew.nn.text_encoders as te
        from dew.nn.attention import scaled_dot_product_attention
        te.scaled_dot_product_attention = functools.partial(
            scaled_dot_product_attention, force_fp32_for_softmax=False)
        fixtures = "tests/fixtures/clip/large-patch14"
        meta = json.load(open(f"{fixtures}/prompts.json"))
        ids = np.load(f"{fixtures}/reference.npz")
        pure = np.load("/tmp/clip-fp64.npz")
        model = te.CLIPTextModel.from_pretrained(meta["repo"], dtype=jnp.float64)
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), model.variables)
        out = model.transformer.apply(variables, ids["input_ids"], ids["attention_mask"])
        print(np.abs(np.asarray(out.last_hidden_state) - pure["last_hidden_state"]).max())
        PY
    """
    original = torch.nn.functional.softmax
    torch.nn.functional.softmax = (
        lambda input, dim=None, dtype=None, **kwargs: original(input, dim=dim, **kwargs))

    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    model = CLIPModel.from_pretrained(REAL_MODEL, dtype=torch.float32).double()
    model.eval()
    model.set_attn_implementation("eager")
    tokens = tokenizer(REAL_PROMPTS, padding="max_length", max_length=MAX_LENGTH,
                       truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.text_model(input_ids=tokens["input_ids"],
                                   attention_mask=tokens["attention_mask"])
    torch.nn.functional.softmax = original
    np.savez(path, last_hidden_state=outputs.last_hidden_state.numpy(),
             pooler_output=outputs.pooler_output.numpy())
    print(f"{path}: fp64 reference for {len(REAL_PROMPTS)} prompts")


def write_tiny(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = tiny_tokenizer()
    tokenizer.save_pretrained(directory)
    model = tiny_model(len(tokenizer.get_vocab()))
    model.save_pretrained(directory, safe_serialization=True)
    side = model.config.vision_config.image_size
    processor = CLIPImageProcessorPil(size={"shortest_edge": side},
                                      crop_size={"height": side, "width": side})
    processor.save_pretrained(directory)

    reference = encode(model, tokenizer, TINY_PROMPTS)
    ids = reference["input_ids"]
    assert ids.max() < len(tokenizer.get_vocab()), "a token id is outside the vocabulary"
    assert (ids == 1).any(axis=-1).all(), "a prompt carries no eos token"
    assert (ids.argmax(axis=-1) != (ids == 1).argmax(axis=-1)).any(), (
        "the argmax and eos branches of the pooled row agree on every prompt, "
        "so this fixture would not tell them apart")
    reference.update(encode_images(model, processor, synthetic_images(TINY_IMAGES)))

    np.savez(directory / "reference.npz", **reference)
    (directory / "prompts.json").write_text(json.dumps(
        {"prompts": TINY_PROMPTS, "max_length": MAX_LENGTH, "images": TINY_IMAGES},
        indent=2) + "\n")


def write_real(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(REAL_MODEL)
    processor = CLIPImageProcessorPil.from_pretrained(REAL_MODEL)
    model = CLIPModel.from_pretrained(REAL_MODEL, dtype=torch.float32)

    reference = encode(model, tokenizer, REAL_PROMPTS)
    # The pixel values are 1.8 MB of upsampled noise; the test recomputes them
    # with the same processor.
    images = encode_images(model, processor, synthetic_images(REAL_IMAGES))
    reference["image_embeds"] = images["image_embeds"]
    np.savez_compressed(directory / "reference.npz", **reference)
    (directory / "prompts.json").write_text(json.dumps(
        {"repo": REAL_MODEL, "prompts": REAL_PROMPTS, "max_length": MAX_LENGTH,
         "images": REAL_IMAGES},
        indent=2) + "\n")

    # The repo's own config.json, byte for byte, so the translation test reads
    # the real nesting and the transformers 4.16 field dump around it without
    # a download.
    shutil.copyfile(hf_hub_download(REAL_MODEL, "config.json"),
                    directory / "config.json")
    print(json.dumps({"hidden": list(reference["last_hidden_state"].shape)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-real", action="store_true",
                        help="write tiny/ only, no 1.7 GB download")
    parser.add_argument("--fp64", type=Path, default=None,
                        help="also write the real tower's fp64 outputs here, "
                             "outside the fixtures, as evidence for the tolerance")
    arguments = parser.parse_args()

    write_tiny(FIXTURES / "tiny")
    if not arguments.skip_real:
        write_real(FIXTURES / "large-patch14")
    if arguments.fp64 is not None:
        write_fp64(arguments.fp64)

    for path in sorted(FIXTURES.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(FIXTURES)}: {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
