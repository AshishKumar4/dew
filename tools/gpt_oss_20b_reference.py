#!/usr/bin/env python3
"""Record GPT OSS 20B's released logits for tests/test_hf_decoder_moe.py: transformers' own bf16
forward of `PROMPT`, the reference, and its float64 truth.

The released checkpoint stores its experts in MXFP4, whose values bf16 holds exactly, so both loads
dequantize them (`Mxfp4Config(dequantize=True)`), as Dew's loader does: the reference computes in the
bf16 the blocks decode to, on a GPU, its matmuls in fp32 (no TF32); the truth loads the same weights
in float64, which no GPU holds whole (168 GB), so accelerate streams its layers through the GPU from
the host and the disk (`device_map="auto"`), inside `torch_precision.float64`. It needs a GPU, about
100 GB of host memory and 200 GB of disk, and imports no Dew, so the reference environment
(DEW_TORCH_PYTHON, with accelerate) runs it.

    python tools/gpt_oss_20b_reference.py OUTPUT.npz
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

RELEASED = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
"""The released GPT OSS 20B and the commit both sides read."""
PROMPT = "The Cascade Range runs from northern California through Oregon"
FIXTURE = "tests/fixtures/gpt_oss/released_20b.npz"
"""The recording's place (tools/fetched_fixtures.py keeps it, 30 MB, in the fetched store)."""


def logits(dtype, ids, **placement):
    """transformers' logits `[S, V]` of `ids` from the released checkpoint at `dtype`, its experts
    dequantized, eager attention."""
    import torch
    from transformers import GptOssForCausalLM, Mxfp4Config

    repo, revision = RELEASED
    model = GptOssForCausalLM.from_pretrained(repo, revision=revision, dtype=dtype,
                                              quantization_config=Mxfp4Config(dequantize=True),
                                              **placement).eval()
    model.set_attn_implementation("eager")
    assert {parameter.dtype for parameter in model.parameters()} == {dtype}
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).to(model.device)).logits[0]
    return out.to(torch.float64 if dtype == torch.float64 else torch.float32).cpu().numpy()


def main(destination: str) -> None:
    import torch
    import transformers
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from torch_precision import float64

    repo, revision = RELEASED
    ids = AutoTokenizer.from_pretrained(repo, revision=revision)(PROMPT, return_tensors="np")["input_ids"]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    reference = logits(torch.bfloat16, ids.astype(np.int64), device_map="cuda")
    torch.cuda.empty_cache()
    offload = tempfile.mkdtemp(prefix="gpt-oss-20b-offload-")
    try:
        with float64():
            truth = logits(torch.float64, ids.astype(np.int64), device_map="auto",
                           max_memory={0: "60GiB", "cpu": "90GiB"}, offload_folder=offload)
    finally:
        shutil.rmtree(offload)
    gpu = torch.cuda.get_device_name(0)
    meta = f"transformers {transformers.__version__}, torch {torch.__version__}, {gpu}"
    np.savez_compressed(destination, input_ids=ids.astype(np.int32), reference=reference, truth=truth,
                        meta=np.frombuffer(meta.encode(), np.uint8))
    rms = float(np.sqrt(np.mean((reference - truth) ** 2)))
    print(f"{destination}: ids {ids.shape}, reference from truth {rms:.3e} (rms), "
          f"argmax agreeing {float(np.mean(reference.argmax(-1) == truth.argmax(-1))):.2f}, {gpu}")


if __name__ == "__main__":
    main(sys.argv[1])
