#!/usr/bin/env python3
"""Record GPT OSS 20B's released logits for tests/test_hf_decoder_moe.py: transformers' own bf16
forward of `PROMPT`, the reference, and its float64 truth.

The released checkpoint stores its experts in MXFP4, whose values bf16 holds exactly, so the load
dequantizes them (`Mxfp4Config(dequantize=True)`), as Dew's loader does. The reference computes in the
bf16 the blocks decode to, on a GPU, its matmuls in fp32 (no TF32). The truth is its float64 twin
(`torch_precision.float64_twin`), 168 GB, which no GPU holds whole, run on the host inside
`torch_precision.float64`: a GH200's 480 GB holds it beside the bf16 load. It imports no Dew; an
environment of torch, transformers 5.17.0 and accelerate runs it.

    python tools/gpt_oss_20b_reference.py OUTPUT.npz
"""

import sys
from pathlib import Path

import numpy as np

RELEASED = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
"""The released GPT OSS 20B and the commit both sides read."""
PROMPT = "The Cascade Range runs from northern California through Oregon"
FIXTURE = "tests/fixtures/gpt_oss/released_20b.npz"
"""The recording's place (tools/fetched_fixtures.py keeps it, 30 MB, in the fetched store)."""


def released(device: str):
    """The released checkpoint on `device`, its experts dequantized to bf16, eager attention and experts
    (`eager_experts`)."""
    import torch
    from transformers import GptOssForCausalLM, Mxfp4Config

    repo, revision = RELEASED
    model = GptOssForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.bfloat16,
                                              device_map=device,
                                              quantization_config=Mxfp4Config(dequantize=True)).eval()
    model.set_attn_implementation("eager")
    return eager_experts(model)


def eager_experts(model):
    """`model` with its experts computed one by one, as Dew's reference reads them: transformers'
    grouped product takes no float64 operands."""
    model.set_experts_implementation("eager")
    return model


def logits(model, ids) -> np.ndarray:
    """`model`'s logits `[S, V]` of `ids`, as float32 holds a bf16 run's and float64 a float64 run's."""
    import torch

    dtype = next(model.parameters()).dtype
    assert {parameter.dtype for parameter in model.parameters()} == {dtype}
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).to(model.device)).logits[0]
    return out.to(torch.float64 if dtype == torch.float64 else torch.float32).cpu().numpy()


def main(destination: str) -> None:
    import torch
    import transformers
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from torch_precision import float64, float64_twin

    repo, revision = RELEASED
    ids = AutoTokenizer.from_pretrained(repo, revision=revision)(PROMPT, return_tensors="np")["input_ids"]
    ids = ids.astype(np.int64)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    narrow = released("cuda")
    reference = logits(narrow, ids)
    del narrow
    torch.cuda.empty_cache()
    wide = eager_experts(float64_twin(released("cpu")))
    with float64():
        truth = logits(wide, ids)
    gpu = torch.cuda.get_device_name(0)
    meta = f"transformers {transformers.__version__}, torch {torch.__version__}, {gpu}"
    np.savez_compressed(destination, input_ids=ids.astype(np.int32), reference=reference, truth=truth,
                        meta=np.frombuffer(meta.encode(), np.uint8))
    rms = float(np.sqrt(np.mean((reference - truth) ** 2)))
    print(f"{destination}: ids {ids.shape}, reference from truth {rms:.3e} (rms), "
          f"argmax agreeing {float(np.mean(reference.argmax(-1) == truth.argmax(-1))):.2f}, {gpu}")


if __name__ == "__main__":
    main(sys.argv[1])
