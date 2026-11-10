"""Mamba's selective scan held to recorded mamba_ssm (tools/mamba_selective_reference.py): a recorded case's
seeded inputs drawn again without torch, and Dew's portable scan over them with every gradient, the
recorder's sizing proxy and its float64 cross-check."""

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import with_gradients
from tool_support import load

from dew.nn.mixers.mamba import selective_scan

_recorder = load("mamba_selective_reference")
SEEDED_CASES = _recorder.SEEDED_CASES


def selective_draw(seed: int, case: str) -> dict[str, np.ndarray]:
    """`case`'s inputs at `seed` as the recorder drew them, its narrow operands' values stored in fp32."""
    *shape, dtype, regime = SEEDED_CASES[case]
    return _recorder.operands(np.random.default_rng(seed), *shape,
                              lambda value: value.astype(jnp.dtype(dtype)).astype(np.float32), regime)


def starts_of(arrays) -> jax.Array | None:
    """Each token's whether it starts a document, from the recorded `seq`, or None where none is packed."""
    seq = arrays.get("seq")
    if seq is None:
        return None
    return jnp.concatenate([jnp.ones_like(seq[:, :1], bool), seq[:, 1:] > seq[:, :-1]], axis=1)


def selective_xla(arrays, dtype=None, scan=selective_scan) -> dict[str, jax.Array]:
    """`scan` (Dew's portable selective scan unless given, its operands `(x, dt, A, B, C, D, state, starts)`)
    over a case's inputs, `x`, the step, `B`, `C` and the cotangent in `dtype` where given: its output and
    each operand's gradient under the recorded cotangent, named as the recording names them."""
    names = ("x", "dt", "A", "B", "C", "D")
    narrow = ("x", "dt", "B", "C")
    operands = tuple(jnp.asarray(arrays[name], dtype if name in narrow else None) for name in names)
    starts = starts_of(arrays)

    def run(*operands):
        return scan(*operands, None, starts)[0]

    out = jax.eval_shape(run, *operands)
    found = with_gradients(run)(*operands, jnp.asarray(arrays["dy"], out.dtype))
    return dict(zip(("y", *names), found, strict=True))


MAMBA_SELECTIVE_SIDE = '''
import sys, time, numpy as np, torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
begun = time.perf_counter()
given = np.load(sys.argv[1])
kinds = dict(zip(str(given["leaves"]).split(","), str(given["dtypes"]).split(",")))
held = {name: torch.from_numpy(given[name]).to("cuda", getattr(torch, kinds[name])).requires_grad_()
        for name in kinds}
draws, rows, length, channels = held["x"].shape
states = held["a"].shape[-1]
# Each draw its own channels (`[B, N D, L]`) and its own group of B and C (`[B, N, states, L]`), its decays
# and skip its channels' (`[N D, states]`, `[N D]`): one call of the draws, each a group of the upstream's.
u, delta = (held[name].permute(1, 0, 3, 2).reshape(rows, draws * channels, length) for name in ("x", "dt"))
b, c = (held[name].permute(1, 0, 3, 2) for name in ("b", "c"))
a, skip = held["a"].reshape(draws * channels, states), held["skip"].reshape(draws * channels)
bounds = [int(x) for x in given["bounds"]]
pieces = [selective_scan_fn(u[..., lo:hi].contiguous(), delta[..., lo:hi].contiguous(), a,
                            b[..., lo:hi].contiguous(), c[..., lo:hi].contiguous(), skip.float())
          for lo, hi in zip(bounds[:-1], bounds[1:])]
y = torch.cat(pieces, dim=-1).reshape(rows, draws, channels, length).permute(1, 0, 3, 2)
torch.cuda.synchronize()
ran = time.perf_counter()
dy = torch.from_numpy(given["dy"]).to("cuda", torch.float32)
torch.autograd.backward(y.float(), dy)
found = {"y": y, **{f"d_{name}": held[name].grad for name in kinds}}
np.savez(sys.argv[2], **{name: value.detach().float().cpu().numpy() for name, value in found.items()})
print(f"K mamba_ssm selective {draws} draws: kernels {ran - begun:.2f} s, "
      f"back {time.perf_counter() - ran:.2f} s", flush=True)
'''
"""mamba_ssm's own `selective_scan_fn` (its CUDA kernel, `exp2f` under `--use_fast_math`) in the interpreter
`DEW_MAMBA_PYTHON` names, over a block of draws run as one call (each draw its own channels and its own group
of B and C), a row's documents each a call of its own from a zero state (the kernel takes no documents) at
the request's `bounds`, its parameters' gradients the fp32 sum of the documents' pulls."""


def mamba_selective_side(drawn: list, starts) -> list:
    """mamba_ssm's selective scan (`MAMBA_SELECTIVE_SIDE`) on a block of draws, each `(x, dt, a, b, c, skip,
    dy)` as the selective template's sites draw them (`[B, L, D]` and `[B, L, N]` in their dtype, the decays
    `[D, N]` and skip `[D]` fp32): each draw's output and its six operands' gradients, `(y, dx, ddt, da, db,
    dc, dskip)` in their operands' dtypes. `starts` `[B, L]` the rows' documents, alike in every row (None:
    one sequence a row)."""
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    import pytest

    python = os.environ.get("DEW_MAMBA_PYTHON")
    if not python:
        pytest.fail("DEW_MAMBA_PYTHON names no interpreter with mamba_ssm 2.3.2 and torch: the GPU tier's "
                    "reference, which a run selecting the mamba reference gives it")
    names = ("x", "dt", "a", "b", "c", "skip")
    length = drawn[0][0].shape[1]
    marks = np.zeros(length, bool) if starts is None else np.asarray(starts, bool)[0]
    if starts is not None:
        assert np.all(np.asarray(starts, bool) == marks), "a block's rows share their documents"
    given = {name: np.stack([np.asarray(d[i], np.float32) for d in drawn]) for i, name in enumerate(names)}
    # B and C in the values' dtype where theirs is narrower (a bf16 C beside fp32 x): the same values, as the
    # upstream's kernel takes B and C in its inputs' type.
    widest = jnp.dtype(drawn[0][0].dtype).name
    kinds = [jnp.dtype(drawn[0][i].dtype).name for i in range(6)]
    kinds[3:5] = [widest, widest]
    given |= {"leaves": np.asarray(",".join(names)), "dtypes": np.asarray(",".join(kinds)),
              "bounds": np.asarray(sorted({0, *np.flatnonzero(marks).tolist(), length}), np.int64),
              "dy": np.stack([np.asarray(d[6], np.float32) for d in drawn])}
    with tempfile.TemporaryDirectory() as root:
        np.savez(Path(root) / "in.npz", **given)
        subprocess.run([python, "-c", MAMBA_SELECTIVE_SIDE, Path(root) / "in.npz", Path(root) / "out.npz"],
                       check=True)
        with np.load(Path(root) / "out.npz") as found:
            outputs = dict(found)
    return [(jnp.asarray(outputs["y"][i], drawn[i][0].dtype),
             *(jnp.asarray(outputs[f"d_{name}"][i], drawn[i][k].dtype) for k, name in enumerate(names)))
            for i in range(len(drawn))]
