"""Stage-by-stage A100 diagnosis of the packed SSD initial-state gradient.

Run --native under DEW_FLA_PYTHON, then --compare under Dew's interpreter. The native phase writes
its installed source paths and source text beside the arrays, so the comparison records the executed
implementation, not an assumed upstream version. Neither phase changes the product implementation.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import types
import urllib.request

import numpy as np

from mamba_ssd_reference import SEEDED_CASES, operands


CASE = "ragged-documents"
SHAPE = SEEDED_CASES[CASE][:6]
CHUNK = SEEDED_CASES[CASE][6]


def draw(seed):
    """The recording's exact native draw and coordinate orbit, without importing JAX in torch."""
    arrays = operands(np.random.default_rng(seed), *SHAPE, lambda x: x.astype(np.float32), True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    _, _, heads, width, groups, state = SHAPE
    group_order = rng.permutation(groups)
    per_group = heads // groups
    head_order = np.concatenate([group * per_group + rng.permutation(per_group) for group in group_order])
    channels, states = rng.permutation(width), rng.permutation(state)
    axes = {
        "x": ((2, head_order), (3, channels)), "dy": ((2, head_order), (3, channels)),
        "dt": ((2, head_order),), "A": ((0, head_order),), "D": ((0, head_order),),
        "B": ((2, group_order), (3, states)), "C": ((2, group_order), (3, states)),
        "h0": ((1, head_order), (2, channels), (3, states)),
        "dfinal": ((1, head_order), (2, channels), (3, states)),
    }
    for name, value in arrays.items():
        for axis, order in axes.get(name, ()):
            value = np.take(value, order, axis=axis)
        arrays[name] = value
    return arrays


def native_source(destination):
    """Import only the Triton package, avoiding mamba_ssm's unrelated compiled CUDA extension."""
    candidates = []
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is not None:
        candidates.extend(Path(root) for root in spec.submodule_search_locations)
    for root in (Path("/home/ubuntu/envs"), Path("/home/ubuntu/s/envs")):
        if root.is_dir():
            candidates.extend(path.parent.parent.parent for path in root.glob("**/ssd_combined.py"))
    source = next((path for path in candidates if (path / "ops/triton/ssd_combined.py").is_file()), None)
    if source is None:
        # Some queue images install fla without optional mamba-ssm. Read the exact distribution named
        # in the port's copyright header instead of silently substituting an unpinned main checkout.
        url = ("https://files.pythonhosted.org/packages/af/48/"
               "aa7bec0efc7453fa0512276c0069c854bf684219050294266534bfef34bb/"
               "mamba_ssm-2.3.2.post1.tar.gz")
        archive = urllib.request.urlopen(url, timeout=120).read()
        digest = hashlib.sha256(archive).hexdigest()
        assert digest == "104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028"
        source = destination / "mamba_ssm"
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            for member in package.getmembers():
                parts = Path(member.name).parts
                if member.isfile() and len(parts) >= 3 and parts[1] == "mamba_ssm":
                    target = destination.joinpath(*parts[1:])
                    if target.suffix == ".py" and ".." not in parts:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(package.extractfile(member).read())
        print(json.dumps({"fallback_distribution": url, "sha256": digest}), flush=True)
    for package, relative in (("mamba_ssm", ""), ("mamba_ssm.ops", "ops"),
                              ("mamba_ssm.ops.triton", "ops/triton"), ("mamba_ssm.utils", "utils")):
        module = types.ModuleType(package)
        module.__path__ = [str(source / relative)]
        sys.modules[package] = module
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("ssd_combined", "ssd_state_passing", "ssd_chunk_state", "ssd_chunk_scan"):
        path = source / "ops/triton" / f"{name}.py"
        (destination / path.name).write_text(path.read_text())
    print(json.dumps({"native_source": str(source)}, sort_keys=True), flush=True)
    return importlib.import_module("mamba_ssm.ops.triton.ssd_combined")


def native(path, seeds):
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    import torch

    combined = native_source(path.with_suffix(".source"))
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                      "dstates": str(inspect.signature(combined._chunk_scan_bwd_dstates)),
                      "passing": str(inspect.signature(combined._state_passing_bwd))}), flush=True)
    stored = {"seeds": np.int32(seeds)}
    for seed in range(seeds):
        arrays = {name: torch.tensor(value, device="cuda") for name, value in draw(seed).items()}
        da, _ = combined._chunk_cumsum_fwd(arrays["dt"], arrays["A"], CHUNK)
        entered = combined._chunk_scan_bwd_dstates(arrays["C"], da, arrays["dy"],
                                                  seq_idx=arrays["seq"], dtype=torch.float32)
        flat = entered.flatten(-2)
        passing = combined._state_passing_bwd(
            torch.zeros_like(flat), da[..., -1], flat, dfinal_states=arrays["dfinal"].flatten(-2),
            seq_idx=arrays["seq"], has_initial_states=True, dstates_dtype=torch.float32,
            states_dtype=torch.float32, chunk_size=CHUNK)
        for name, value in (("da", da.flatten(-2)), ("entered", entered), ("h0", passing[2])):
            stored[f"{seed}.{name}"] = value.detach().cpu().numpy()
    np.savez_compressed(path, **stored)
    print(f"wrote {seeds} native stages: {path}", flush=True)


def exact_stages(arrays):
    """Float64 chain-rule stages; h0 is independent of x, B, D and the value of the initial state."""
    batch, length, heads, width, groups, state = SHAPE
    padded = -(-length // CHUNK) * CHUNK
    seq = np.pad(arrays["seq"], ((0, 0), (0, padded - length)), mode="edge")
    decay = np.pad(arrays["dt"].astype(np.float64).transpose(0, 2, 1)
                   * arrays["A"].astype(np.float64)[None, :, None],
                   ((0, 0), (0, 0), (0, padded - length)))
    da = np.zeros_like(decay)
    entered = np.zeros((batch, padded // CHUNK, heads, width, state), dtype=np.float64)
    relevant = np.zeros((batch, padded), dtype=bool)
    for row in range(batch):
        for chunk in range(padded // CHUNK):
            start, end = chunk * CHUNK, (chunk + 1) * CHUNK
            current = seq[row, start:end]
            for document in np.unique(current):
                mine = current == document
                da[row, :, start:end] += np.where(
                    mine, np.cumsum(np.where(mine, decay[row, :, start:end], 0.0), axis=-1), 0.0)
            earlier = seq[row, start - 1] if chunk else 0
            relevant[row, start:end] = current == earlier
            stop = min(end, length)
            scale = np.exp(da[row, :, start:stop]).T * relevant[row, start:stop, None]
            cm = np.repeat(arrays["C"][row, start:stop].astype(np.float64), heads // groups, axis=1)
            entered[row, chunk] = np.einsum(
                "thp,thn,th->hpn", arrays["dy"][row, start:stop].astype(np.float64), cm, scale)
    carry = arrays["dfinal"].astype(np.float64)
    document = seq[:, -1]
    for chunk in reversed(range(padded // CHUNK)):
        earlier = seq[:, chunk * CHUNK - 1] if chunk else np.zeros(batch, dtype=np.int32)
        scale = np.exp(da[:, :, (chunk + 1) * CHUNK - 1]) * (earlier == document)[:, None]
        carry = scale[:, :, None, None] * carry + entered[:, chunk]
        document = earlier
    return da, entered, carry, relevant


def compare(path):
    import jax
    import jax.numpy as jnp
    from dew.nn.kernels import mamba_ssd as port

    launch = {"chunk": CHUNK, "interpret": False}

    @jax.jit
    def cumsum(arrays):
        seq = port._documents_padded(arrays["seq"], CHUNK)
        return port._CHUNK_CUMSUM(arrays["dt"], arrays["A"], seq, **launch)[1]

    def dstates(arrays, da):
        seq = port._documents_padded(arrays["seq"], CHUNK)
        return port.ChunkScanBwdDstates()._fwd(
            arrays["dy"], arrays["C"], da, seq, config=port.Tiles(16, 16, 32, 2, 4), **launch)[0]

    @jax.jit
    def passing(arrays, da, entered):
        seq = port._documents_padded(arrays["seq"], CHUNK)
        flat = port._flat(entered)
        return port._STATE_PASSING_BWD(
            flat, jnp.zeros_like(flat), da, port._flat(arrays["dfinal"]), seq, dtype="float32",
            **launch)[2].reshape(arrays["dfinal"].shape)

    normal = jax.jit(dstates)
    original_dot = port._dot

    def ieee_dot(a, b, *, exact, **kwargs):
        return original_dot(a, b, exact=exact, ieee=True)

    ieee = jax.jit(lambda arrays, da: dstates(arrays, da))
    sums, counts = {}, {}

    def measure(name, got, truth, mask=None):
        error = np.asarray(got, np.float64) - truth
        if mask is not None:
            error = error[mask]
        sums[name] = sums.get(name, 0.0) + float(np.sum(error * error))
        counts[name] = counts.get(name, 0) + error.size

    with np.load(path) as native_arrays:
        seeds = int(native_arrays["seeds"])
        for seed in range(seeds):
            arrays = draw(seed)
            da64, entered64, h064, relevant = exact_stages(arrays)
            device = jax.tree.map(jnp.asarray, arrays)
            da = cumsum(device)
            entered = normal(device, da)
            with contextlib.ExitStack() as stack:
                port._dot = ieee_dot
                stack.callback(setattr, port, "_dot", original_dot)
                entered_ieee = ieee(device, da)
            native_da = native_arrays[f"{seed}.da"]
            native_entered = native_arrays[f"{seed}.entered"]
            native_h0 = native_arrays[f"{seed}.h0"].reshape(h064.shape)
            mask = np.broadcast_to(relevant[:, None, :], da64.shape)
            measure("da.port", da, da64, mask)
            measure("da.native", native_da, da64, mask)
            measure("exp.port", np.exp(np.asarray(da, np.float64)), np.exp(da64), mask)
            measure("exp.native", np.exp(native_da.astype(np.float64)), np.exp(da64), mask)
            measure("entered.port", entered, entered64)
            measure("entered.ieee", entered_ieee, entered64)
            measure("entered.native", native_entered, entered64)
            measure("h0.port", passing(device, da, entered), h064)
            measure("h0.ieee_dot", passing(device, da, entered_ieee), h064)
            measure("h0.native", native_h0, h064)
            measure("h0.native_stages_port_passing",
                    passing(device, jnp.asarray(native_da), jnp.asarray(native_entered)), h064)
            measure("h0.exact_entered_port_passing",
                    passing(device, da, jnp.asarray(entered64, jnp.float32)), h064)
            measure("h0.port_entered_exact_da",
                    passing(device, jnp.asarray(da64, jnp.float32), entered), h064)
            if seed == 0:
                print(json.dumps({"seed": seed, "first_reset": [
                    np.flatnonzero(arrays["seq"][row] > 0).tolist()[:1] for row in range(SHAPE[0])],
                    "selected_dstates_tiles": "16,16,32,2,4"}), flush=True)
    rms = {name: (sums[name] / counts[name]) ** 0.5 for name in sums}
    print(json.dumps({"seeds": seeds, "rms": rms,
                      "h0_ratios": {name: value / rms["h0.native"] for name, value in rms.items()
                                    if name.startswith("h0.")}}, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--seeds", type=int, default=128)
    args = parser.parse_args()
    if args.native:
        native(args.path, args.seeds)
    elif args.compare:
        compare(args.path)
    else:
        parser.error("choose --native or --compare")


if __name__ == "__main__":
    main()
