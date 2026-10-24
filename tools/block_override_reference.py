"""Record tiny Qwen 3.5 block-attention draws from the release's mask adapter and transformers.

Run in the declared transformers-5.19 reference environment, on armada. The
release's pinned serve.patch supplies install_noncausal/blocks; no attention
mask is transcribed here. The generator writes a native checkpoint, paired
fp32/float64 logits and source.json with exact package/source provenance.
`--seeds` and `--redraw` come from the accuracy gate's recording plan.
"""

import argparse
import contextlib
import contextvars
import copy
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from fetched_fixtures import save
from reference_source import CACHE, defined_in, git_head, verified
from torch_precision import default_dtype, dtype_widened, float_as, replaced, widened_to

ROOT = Path(__file__).resolve().parents[1]
REPO = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
PATCH_SHA256 = "f234c125f1d38d82104b9e1f24e75bde51e4401928acfba2fbf5c59ef2fb46b8"
REFERENCE = f"https://huggingface.co/{REPO}/blob/{REVISION}/serve.patch; transformers"
LICENSE = "MIT AND Apache-2.0"


def release_source() -> tuple[Path, str]:
    """The verified patch and the complete readout module it adds."""
    filename = verified(f"https://huggingface.co/{REPO}/raw/{REVISION}/serve.patch",
                        CACHE / REPO / REVISION / "serve.patch", PATCH_SHA256)
    section = filename.read_text().split("diff --git a/qev/readout.py b/qev/readout.py\n", 1)[1]
    section = section.split("\ndiff --git ", 1)[0]
    source = "\n".join(line[1:] for line in section.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    return filename, source


def release_blocks():
    """The release's own module-level block context and transformers mask adapter."""
    filename, source = release_source()
    return defined_in(source, ("_BLOCKS", "install_noncausal", "blocks", "block_ids"),
                      {"torch": torch, "contextlib": contextlib, "contextvars": contextvars,
                       "importlib": importlib, "Sequence": Sequence}, filename=str(filename))


@contextlib.contextmanager
def oracle_precision():
    """Build and run the same reference with its explicit fp32 casts widened to float64.
    All replacements compose the shared reference precision primitives."""
    patches = [(torch.Tensor, "to", widened_to(torch.Tensor.to)), float_as(torch.float64)]
    constructors = ("arange", "tensor", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like")
    patches.extend((torch, name, dtype_widened(getattr(torch, name))) for name in constructors)
    patches.append((torch.nn.functional, "softmax", dtype_widened(torch.nn.functional.softmax)))
    with default_dtype(torch.float64), replaced(*patches):
        yield


def tiny_config():
    """The existing native Qwen fixture's geometry, with a small vocabulary and explicit blocks."""
    from transformers import Qwen3_5Config

    raw = json.loads((ROOT / "tests/fixtures/hf/qwen38-dense-tiny/config.json").read_text())
    raw["text_config"].update(vocab_size=64, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                               dtype="float32")
    raw.update(image_token_id=3, video_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
               modeling_overrides={"attention": {
                   "full_attention": "block_bidirectional", "linear_attention": "causal"}})
    return Qwen3_5Config.from_dict(raw)


def inputs(seed: int, vocabulary: int) -> dict[str, np.ndarray]:
    """Two independently drawn rows: a shared-state block, a question block and causal trailing tokens."""
    rng = np.random.default_rng(seed)
    batch, length = 2, 12
    ids = rng.integers(7, vocabulary, (batch, length), dtype=np.int64)
    groups = np.full((batch, length), -1, np.int64)
    for row in range(batch):
        split = int(rng.integers(2, length - 4))
        stop = int(rng.integers(split + 2, length))
        groups[row, :split] = 0
        groups[row, split:stop] = 1
    if seed % 4 == 0:
        groups[:] = -1
    return {"input_ids": ids, "block_groups": groups}


def main():
    from transformers import AutoModelForImageTextToText

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--redraw", type=int)
    parser.add_argument("--calibrate", action="store_true",
                        help="size the recorded draws in the checkout's gate environment")
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.output)
        return
    if args.seeds is None or args.redraw is None or args.seeds <= 0 or args.redraw < 0:
        parser.error("seeds must be positive and redraw nonnegative")
    version = importlib.metadata.version("transformers")
    if version != "5.19.0":
        raise RuntimeError(f"this reference is transformers 5.19.0, got {version}")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(43)
    config = tiny_config()
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32,
                                                   attn_implementation="eager").cpu().eval()
    release = release_blocks()
    release["install_noncausal"](model)
    probe = inputs(1, config.text_config.vocab_size)
    ids = torch.from_numpy(probe["input_ids"])
    altered = ids.clone()
    altered[0, 1] = (altered[0, 1] - 6) % (config.text_config.vocab_size - 7) + 7
    with torch.no_grad(), release["blocks"](None):
        causal = model(input_ids=ids, use_cache=False).logits[0, 0]
        causal_changed = model(input_ids=altered, use_cache=False).logits[0, 0]
    with torch.no_grad(), release["blocks"](torch.from_numpy(probe["block_groups"])):
        blocked = model(input_ids=ids, use_cache=False).logits[0, 0]
        blocked_changed = model(input_ids=altered, use_cache=False).logits[0, 0]
    if not torch.equal(causal, causal_changed) or torch.equal(blocked, blocked_changed):
        raise RuntimeError("the reference did not isolate causal rows and admit their future block mate")
    weights = copy.deepcopy(model.state_dict())
    with oracle_precision():
        exact = AutoModelForImageTextToText.from_config(copy.deepcopy(config), dtype=torch.float64,
                                                       attn_implementation="eager").cpu().double().eval()
        exact.load_state_dict(weights)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    arrays = {"blocks.redraw": np.asarray(args.redraw), "precision": np.asarray("ieee")}
    draws, references, truths = [], [], []
    for seed in range(args.seeds * (1 + args.redraw)):
        draw = inputs(seed, config.text_config.vocab_size)
        ids = torch.from_numpy(draw["input_ids"])
        blocks = torch.from_numpy(draw["block_groups"])
        with torch.no_grad(), release["blocks"](blocks):
            reference = model(input_ids=ids, use_cache=False).logits
        with torch.no_grad(), oracle_precision(), release["blocks"](blocks):
            truth = exact(input_ids=ids, use_cache=False).logits
        if reference.dtype != torch.float32 or truth.dtype != torch.float64:
            raise RuntimeError(f"reference/oracle returned {reference.dtype}/{truth.dtype}")
        draws.append(draw)
        references.append(reference.numpy())
        truths.append(truth.numpy())
    for name in draws[0]:
        arrays[f"blocks.input.{name}"] = np.stack([draw[name] for draw in draws])
    arrays["blocks.reference.logits"] = np.stack(references)
    arrays["blocks.truth.logits"] = np.stack(truths)
    save(args.output / "reference.npz", arrays, reference=REFERENCE, license=LICENSE)
    provenance = {"generator": "tools/block_override_reference.py", "generator_revision": git_head(ROOT),
                  "repo": REPO, "revision": REVISION, "patch_sha256": PATCH_SHA256,
                  "packages": {name: importlib.metadata.version(name)
                               for name in ("transformers", "torch", "numpy", "safetensors")},
                  "seeds": args.seeds, "redraw": args.redraw, "parameter_seed": 43,
                  "attention_impl": "eager", "device": "cpu", "precision": "ieee"}
    (args.output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")


def calibrate(directory: Path, sites: tuple[str, ...] = ("blocks",)):
    """Declare the recorded site's seeds with the gate's reference-vs-reference sizing, retaining
    enough recorded draws for its redraw policy. No candidate implementation is measured."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reference_error import REDRAW, indexed_recording, recording, seeds_for

    path = directory / "reference.npz"
    arrays = recording(path)
    counts, capacities = {}, {}
    for site in sites:
        values = next(value for name, value in arrays.items() if name.startswith(f"{site}.reference."))
        capacity = len(values) // (1 + REDRAW)
        draw, reference, truth = indexed_recording(arrays, site, capacity)
        count = seeds_for(reference, draw, site, most=capacity, truth_fn=truth)
        arrays[f"{site}.seeds"] = np.asarray(count, np.int64)
        counts[site], capacities[site] = count, capacity
        print(f"{site}: {count} declared seeds, {capacity} recorded with redraw")
    save(path, arrays, reference=REFERENCE, license=LICENSE)
    provenance = json.loads((directory / "source.json").read_text())
    provenance.update(declared_seeds=counts, recorded_seed_capacity=capacities,
                      sizing_revision=git_head(ROOT))
    (directory / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
