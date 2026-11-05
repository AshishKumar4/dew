#!/usr/bin/env python3
"""Record GPT OSS 20B's released logits over check 1's seeds for tests/test_hf_decoder_moe.py:
transformers' own bf16 forward of each seed's drawn tokens (`draw`), the reference, and its float64
twin's, the truth, summarized as check 1 reads them (`recorded_references.record_distances`: each seed's
row distances from its truth and the truth's sketch), which the test's own float64 path is held to.

The released checkpoint stores its experts in MXFP4, whose values bf16 holds exactly, so the load
dequantizes them (`Mxfp4Config(dequantize=True)`), as Dew's loader does. The reference computes in the
bf16 the blocks decode to, on a GPU, its matmuls in fp32 (no TF32), each seed its own call. The truth
is its float64 twin (`torch_precision.float64_twin`), 168 GB, which no GPU holds whole, run on the host
inside `torch_precision.float64`, the seeds' rows in batches: a GH200's 480 GB holds it beside the bf16
load. Two steps, as two environments run them: `run` in one of torch, transformers 5.17.0 and accelerate
(no Dew), keeping each seed's row distances and its truth; `summary` in Dew's, writing the recording:

    python tools/gpt_oss_20b_reference.py run RAW.npz
    python tools/gpt_oss_20b_reference.py summary RAW.npz
"""

import sys
from pathlib import Path

import numpy as np

RELEASED = ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee")
"""The released GPT OSS 20B and the commit both sides read."""
FIXTURE = "tests/fixtures/gpt_oss/released_20b.npz"
"""The recording's place (tools/fetched_fixtures.py keeps it under tests/fixtures)."""
SITE = "gpt-oss-20b"
POSITIONS = 4
"""A seed's tokens: one row of `POSITIONS` drawn ids, every position's logits judged."""
RECORDED = 256
"""The seeds the recording runs, from which `record_distances` keeps the count its evidence resolves
and that count's redraw."""
VOCABULARY = 199998
"""The ids a draw takes, the ordinary tokens below the release's specials."""


def draw(seed: int) -> dict[str, np.ndarray]:
    """Seed `seed`'s tokens: `POSITIONS` ordinary ids, one row."""
    return {"input_ids": np.random.default_rng([0x1719, seed]).integers(0, VOCABULARY, (1, POSITIONS))
            .astype(np.int32)}


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
    """`model`'s logits `[B, S, V]` of `ids` `[B, S]`, as float32 holds a bf16 run's and float64 a
    float64 run's."""
    import torch

    dtype = next(model.parameters()).dtype
    assert {parameter.dtype for parameter in model.parameters()} == {dtype}
    with torch.no_grad():
        out = model(input_ids=torch.from_numpy(ids).to(model.device)).logits
    return out.to(torch.float64 if dtype == torch.float64 else torch.float32).cpu().numpy()


TRUTH_BATCH = 32
"""The seeds' rows the float64 twin runs at once."""


def run(raw: str) -> None:
    """Each seed's reference row distances from its truth and its truth, at `raw`: the bf16 reference a
    seed a call on the GPU, the float64 twin's on the host."""
    import torch
    import transformers

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from torch_precision import float64, float64_twin

    ids = np.concatenate([draw(seed)["input_ids"] for seed in range(RECORDED)]).astype(np.int64)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    narrow = released("cuda")
    reference = np.concatenate([logits(narrow, ids[row:row + 1]) for row in range(RECORDED)])
    gpu = torch.cuda.get_device_name(0)
    del narrow
    torch.cuda.empty_cache()
    wide = eager_experts(float64_twin(released("cpu")))
    with float64():
        truth = np.concatenate([logits(wide, ids[start:start + TRUTH_BATCH])
                                for start in range(0, RECORDED, TRUTH_BATCH)])
    distances = np.sqrt(np.mean(np.square(reference.astype(np.float64) - truth), axis=-1))
    np.savez(raw, distances=distances, truth=truth, gpu=np.asarray(gpu), torch=np.asarray(torch.__version__),
             transformers=np.asarray(transformers.__version__))
    print(f"{raw}: {RECORDED} seeds, distances {distances.mean():.3e} mean, {gpu}", flush=True)


def summary(raw: str) -> None:
    """The recording from `run`'s `raw` (`record_distances`), written through the fetched store."""
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "tests")]
    from fetched_fixtures import save
    from recorded_references import provenance, record_distances

    with np.load(raw) as held:
        distances, truth = held["distances"], held["truth"]
        gpu, torch_version, version = (str(held[name]) for name in ("gpu", "torch", "transformers"))
    arrays = record_distances(SITE, draw, {"logits": distances}, lambda seed: {"logits": truth[seed]},
                              {"logits": "bfloat16"})
    written = save(FIXTURE, {**arrays, "provenance": provenance(
        "transformers", version, gpu, "python tools/gpt_oss_20b_reference.py run, summary",
        torch=torch_version, release=f"{RELEASED[0]}@{RELEASED[1]}")}, reference=f"transformers {version}",
        license="Apache-2.0")
    print(f"{written}: {RECORDED} seeds recorded, {int(arrays[f'{SITE}.seeds'])} declared", flush=True)


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]](sys.argv[2])
