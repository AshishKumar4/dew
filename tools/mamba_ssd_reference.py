#!/usr/bin/env python3
"""The fixture tests/test_mamba_ssd.py holds the SSD port's bf16 numbers to: mamba_ssm 2.3.2.post1's own
`mamba_chunk_scan_combined` on a CUDA GPU, forward and every gradient.

    python tools/mamba_ssd_reference.py <mamba_ssm source checkout>

in an environment with torch, Triton and einops on a GPU. What lands is
tests/fixtures/mamba2/ssd_mamba_ssm.npz: fixed-seed operands as Mamba-2 hands them over (`x`, `B` and `C`
in bf16, the step through softplus and `A`, `D` and an initial state in fp32), each token's document
(`seq_idx`, two rows of 200 tokens over chunks of 64, the last chunk ragged, one row starting on a
document and one carrying its initial state into its first), the cotangents of the output (bf16) and
of the final state, and mamba_ssm's output, final state and gradients of `x`, `dt`, `A`, `B`, `C`, `D`
and the initial state.

Every product mamba_ssm's bf16 path takes multiplies bf16 operands, exact in its fp32 sums, so no
TF32 enters the reference and it records 'ieee' (`reference_error.PRECISION`).
"""

import sys
from pathlib import Path

import numpy as np
import torch

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2" / "ssd_mamba_ssm.npz"
SEED = 2026
# batch, tokens, heads, head width, groups, state, chunk: three heads a group, a ragged last chunk.
B, T, H, P, G, N, CHUNK = 2, 200, 6, 16, 2, 32, 64


def operands() -> dict[str, np.ndarray]:
    """The fixture's inputs, bf16 ones stored as the fp32 values they hold."""
    rng = np.random.default_rng(SEED)
    bf16 = lambda shape: torch.tensor(rng.standard_normal(shape), dtype=torch.bfloat16).float().numpy()  # noqa: E731
    starts = rng.random((B, T)) < 3.0 / T
    starts[0, 0] = True
    return {
        "x": bf16((B, T, H, P)), "B": bf16((B, T, G, N)), "C": bf16((B, T, G, N)),
        "dt": np.log1p(np.exp(rng.standard_normal((B, T, H)) - 2.0)).astype(np.float32),
        "A": -np.exp(rng.uniform(0.0, np.log(16.0), H)).astype(np.float32),
        "D": rng.standard_normal(H).astype(np.float32),
        "h0": (0.1 * rng.standard_normal((B, H, P, N))).astype(np.float32),
        "seq": np.cumsum(starts, axis=1).astype(np.int32),
        "dy": bf16((B, T, H, P)), "dfinal": rng.standard_normal((B, H, P, N)).astype(np.float32),
    }


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    from mamba_ssm import __version__
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    arrays = operands()
    cuda = {name: torch.tensor(value, device="cuda") for name, value in arrays.items()}
    narrow = {name: cuda[name].to(torch.bfloat16).requires_grad_() for name in ("x", "B", "C")}
    wide = {name: cuda[name].clone().requires_grad_() for name in ("dt", "A", "D", "h0")}
    y, final = mamba_chunk_scan_combined(narrow["x"], wide["dt"], wide["A"], narrow["B"], narrow["C"], CHUNK,
                                         D=wide["D"], initial_states=wide["h0"], seq_idx=cuda["seq"],
                                         return_final_states=True)
    inputs = (narrow["x"], wide["dt"], wide["A"], narrow["B"], narrow["C"], wide["D"], wide["h0"])
    grads = torch.autograd.grad((y, final), inputs, (cuda["dy"].to(torch.bfloat16), cuda["dfinal"]))
    found = dict(zip(("y", "final", "x", "dt", "A", "B", "C", "D", "h0"), (y, final, *grads), strict=True))
    arrays.update({f"mamba.{name}": value.detach().float().cpu().numpy() for name, value in found.items()})
    np.savez(FIXTURE, chunk=np.int32(CHUNK), precision=np.asarray("ieee"), version=np.asarray(__version__),
             device=np.asarray(torch.cuda.get_device_name()), **arrays)
    print(f"wrote {FIXTURE} from mamba_ssm {__version__} on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
