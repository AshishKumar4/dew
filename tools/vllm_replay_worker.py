"""vLLM serving tests/fixtures/hf/qwen3-moe-vllm over drawn prompts and transformers scoring what it
sampled (tools/vllm_replay_reference.py's `generated` and `scored`): the reference arm of
tools/venv_reference.py's `vllm-replay` site (its protocol tools/reference_worker.py), on a GPU.

    python tools/vllm_replay_worker.py vllm-replay    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.prompt_ids` `[rows, length]` and `<draw>.seed` (vLLM's sampling seed). OUT
holds what vLLM generated, which Dew replays, as `<draw>.replayed.<leaf>`, and transformers' scores of it:
`<draw>.reference.log_probs` in float32 and `<draw>.truth.log_probs` from its float64 twin, run in
float64 (src/_dew_torch/precision.py's `float64`, its router's float32 softmax pin among what it widens).
"""

import functools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@functools.cache
def scorers():
    """transformers' model of the checkpoint in float32 and its float64 twin, the experts eager (the
    grouped matmul refuses float64)."""
    import torch
    from transformers import Qwen3MoeForCausalLM
    from vllm_replay_reference import MODEL

    from _dew_torch.precision import float64_twin

    model = Qwen3MoeForCausalLM.from_pretrained(MODEL, dtype=torch.float32, experts_implementation="eager")
    return model.eval(), float64_twin(model)


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's completions, the leaves Dew replays, and transformers' scores of them."""
    import torch
    from vllm_replay_reference import generated, scored

    from _dew_torch.precision import float64

    # One thread sums in one order, so the float32 references regenerate bit for bit.
    torch.set_num_threads(1)
    narrow, twin = scorers()
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        prompts = np.asarray(cases[f"{name}.prompt_ids"])
        replay = generated(prompts, int(cases[f"{name}.seed"]))
        out.update({f"{name}.replayed.{leaf}": value for leaf, value in replay.items()})
        out[f"{name}.reference.log_probs"] = scored(narrow, prompts, replay)
        with float64():
            out[f"{name}.truth.log_probs"] = scored(twin, prompts, replay)
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["vllm-replay"]:
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests"), str(ROOT / "src")]
    from reference_worker import serve

    serve(run)
