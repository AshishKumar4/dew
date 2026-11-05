#!/usr/bin/env python3
"""The fixture tests hold the recurrence template's gated delta rule to: fla-core's own
`chunk_gated_delta_rule` and `chunk_kda` (its Triton kernels) on a CUDA GPU, forward and every gradient, at
each site tests/fla_sites.py names, over the tests' own seeded draws.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_gdn_reference.py <fla> <unit> <site>...
    PYTHONPATH=src:tests:tools python tools/fla_gdn_reference.py --merge <unit>...

on a GPU, Dew's environment recording and a second Python with torch, fla-core and tilelang (`--serve`, the
"fla" cohort, tools/reference_releases.py) running fla and its reference, each run a unit of named sites
(`--list` names them all), the units then merged. What lands is tests/fixtures/fla/gdn_seeded.npz, a
summary recording (`record_case(..., summary=True)`): per site, seed and
leaf, fla's distance from float64 at the site's inputs, which the test draws again (`Site.arrays`), and the
truth's sketch, which Dew's float64 token recurrence is held to.

fla runs at its defaults (an fp32 call's products TF32, as the fast forms' bar is). Packed documents run as
fla runs them, one flattened row of sequences (`cu_seqlens`), a row's first taking its entering memory
unless a document starts it. The oracle is fla's own reference, `naive_recurrent_gated_delta_rule` and
`naive_recurrent_kda` (fla/ops/*/naive.py), on each sequence in torch float64 (`torch_precision.float64`,
which widens their fp32 casts) on the GPU, held at every seed to Dew's float64 token recurrence.
"""

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fla" / "gdn_seeded.npz"
GRADIENTS = ("q", "k", "v", "g", "beta", "h0")


def sequences(arrays) -> list[tuple[int, int, int, bool]]:
    """Each sequence of a site's rows as fla runs them: `(row, start, end, entering)`, a row split at its
    documents' starts, `entering` where it takes the row's entering memory (its row's first, no document
    starting it)."""
    rows, length = arrays["v"].shape[:2]
    starts = arrays.get("starts", np.zeros((rows, length), bool))
    found = []
    for row in range(rows):
        cuts = [*sorted({0, *np.flatnonzero(starts[row]).tolist()}), length]
        for start, end in itertools.pairwise(cuts):
            found.append((row, start, end, start == 0 and not starts[row, 0]))
    return found


def run(arrays, *, kda: bool, scale: float, rule, dtype, device, per_sequence: bool,
        gradients: bool = True) -> dict[str, np.ndarray]:
    """`rule` (fla's chunked rule, or its naive reference with `per_sequence`) on a site's arrays: its rows'
    sequences flattened into one row (`cu_seqlens`), or each alone, the operands in `dtype` but the log decay
    and memory fp32 (in float64 for the reference), and with `gradients` the output's and final memory's
    gradients under the drawn cotangents; leaves as `fla_sites.LEAVES`, the output and final memory alone
    without them."""
    import torch

    wide = dtype == torch.float64
    tensors = {name: torch.tensor(np.asarray(value, np.float64 if wide else np.float32), device=device)
               for name, value in arrays.items() if name != "starts"}
    inputs = {name: tensors[name].to(dtype if name in ("q", "k", "v", "beta") else
                                     (torch.float64 if wide else torch.float32)).requires_grad_()
              for name in GRADIENTS}
    spans = sequences(arrays)
    group = inputs["v"].shape[2] // inputs["q"].shape[2]

    def piece(name, row, start, end):
        x = inputs[name][row:row + 1, start:end]
        return x.repeat_interleave(group, dim=2) if per_sequence and name in ("q", "k") else x

    states = [inputs["h0"][row:row + 1] if entering else torch.zeros_like(inputs["h0"][:1])
              for row, _, _, entering in spans]
    if per_sequence:
        outs, finals = [], []
        for (row, start, end, _), state in zip(spans, states, strict=True):
            q, k, v, g, beta = (piece(name, row, start, end) for name in ("q", "k", "v", "g", "beta"))
            out, final = (rule(q, k, v, g, beta, scale=scale, initial_state=state, output_final_state=True)
                          if kda else rule(q, k, v, beta, g, scale=scale, initial_state=state,
                                           output_final_state=True))
            outs.append(out)
            finals.append(final)
        out = torch.cat(outs, dim=1)
    else:
        q, k, v, g, beta = (torch.cat([piece(name, row, start, end) for row, start, end, _ in spans], dim=1)
                            for name in ("q", "k", "v", "g", "beta"))
        lengths = [end - start for _, start, end, _ in spans]
        cu_seqlens = torch.tensor(np.cumsum([0, *lengths]), device=device, dtype=torch.long)
        out, final = rule(q, k, v, g, beta, scale=scale, initial_state=torch.cat(states, dim=0),
                          output_final_state=True, cu_seqlens=cu_seqlens)
        finals = list(final.split(1, dim=0))
    rows = inputs["v"].shape[0]
    out = out.reshape(rows, -1, *out.shape[2:])
    last = {row: index for index, (row, _, _, _) in enumerate(spans)}
    final = torch.cat([finals[last[row]] for row in range(rows)], dim=0)
    if not gradients:
        return {name: value.detach().to(torch.float64).cpu().numpy()
                for name, value in (("out", out), ("final", final))}
    grads = torch.autograd.grad((out, final), tuple(inputs[name] for name in GRADIENTS),
                                (tensors["dy"].to(out.dtype), tensors["dfinal"].to(final.dtype)),
                                allow_unused=True)
    found = dict(zip(("out", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0"), (out, final, *grads),
                     strict=True))
    return {name: (np.zeros(arrays[name[1:]].shape) if value is None else
                   value.detach().to(torch.float64).cpu().numpy())
            for name, value in found.items()}


def serve() -> None:
    """The torch side (tools/reference_worker.py's protocol): each request a site's inputs with its `dtype`,
    `kind` ('kda' or 'gdn'), `scale` and `leaves` ('all', or 'forward' for the output and final memory
    alone), answered with fla's outputs and the reference's (`reference.<leaf>`, `truth.<leaf>`), after one
    `ready` line of fla's and tilelang's versions and the device."""
    import reference_worker
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    from reference_releases import require
    from torch_precision import float64

    require("fla")
    from importlib.metadata import version

    # On Hopper fla-core 0.5.2 refuses its Triton backward under Triton below 3.7.1 (fla #640, wrong results
    # in chunk_bwd_dqkwg) and takes tilelang's there: the cohort's torch 2.8.0 carries Triton 3.4.
    found = f"fla-core-{version('fla-core')}+tilelang-{version('tilelang')}"

    def answer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dtype, kind, leaves = (str(arrays.pop(name)) for name in ("dtype", "kind", "leaves"))
        scale = float(arrays.pop("scale"))
        kda, gradients = kind == "kda", leaves == "all"
        rule = chunk_kda if kda else chunk_gated_delta_rule
        began = time.perf_counter()
        reference = run(arrays, kda=kda, scale=scale, rule=rule, dtype=getattr(torch, dtype), device="cuda",
                        per_sequence=False, gradients=gradients)
        referenced = time.perf_counter()
        # The truth in float64 on the GPU: on the CPU its token loop and autograd took most of a site's
        # 1432 s for 320 draws (fla-record3-h100-f8fbb058d, the worker at a core's whole time).
        with float64():
            naive = naive_recurrent_kda if kda else naive_recurrent_gated_delta_rule
            truth = run(arrays, kda=kda, scale=scale, rule=naive, dtype=torch.float64, device="cuda",
                        per_sequence=True, gradients=gradients)
        print(f"timed fla {referenced - began:.3f} s truth {time.perf_counter() - referenced:.3f} s",
              file=sys.stderr, flush=True)
        return reference_worker.paired(reference, truth)

    reference_worker.serve(answer, f"{found} {torch.cuda.get_device_name()}")


def write_seeded(worker, names: list[str], out: Path) -> None:
    """Each named site's seeded draws with fla's outputs and gradients and their float64 truth (`worker`,
    `serve`'s `Session`), recorded as a summary (`record_case`) into `out`, Dew's portable rule in the site's
    dtype the sizing proxy and its float64 token recurrence held to fla's reference at every seed: a unit
    of the recording, whose units `merged` joins into the fixture."""
    import functools

    import jax
    from fla_sites import ruled, sites
    from recorded_references import held_to_the_oracle, provenance, record_case
    from reference_error import chain_roundings, widened
    from reference_worker import arms
    from test_recurrence_chunk import chunk_on, token_by_token

    from dew.nn.kernels.selection import KernelRefused

    version, device = worker.started.split(maxsplit=1)
    stored = {}
    for name in names:
        site, began = sites()[name], time.perf_counter()
        sample = site.arrays(0)
        dtype, kda = str(sample["q"].dtype), sample["g"].ndim == 4

        leaves = "all" if set(site.leaves) - {"out", "final"} else "forward"

        def fla(arrays, dtype=dtype, kda=kda, scale=site.hooks.scale, leaves=leaves):
            request = {key: np.asarray(value, np.float32) if key != "starts" else value
                       for key, value in arrays.items()}
            options = {"dtype": dtype, "kind": "kda" if kda else "gdn", "scale": scale, "leaves": leaves}
            request |= {name: np.asarray(value) for name, value in options.items()}
            return arms(worker.ask(request))

        exact_run = jax.jit(functools.partial(lambda hooks, arrays, leaves: ruled(
            functools.partial(token_by_token, hooks), arrays, leaves), site.hooks, leaves=site.leaves))
        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(sample)))

        def evaluate(arrays, fla=fla, exact_run=exact_run, roundings=roundings, name=name):
            reference, truth = fla(arrays)
            held_to_the_oracle(exact_run, arrays, truth, roundings, f"fla {name}")
            return reference, truth

        portable = functools.partial(lambda hooks, arrays, leaves: ruled(
            functools.partial(chunk_on, 'xla', hooks), arrays, leaves), site.hooks, leaves=site.leaves)
        if site.paired:
            # Sized paired with the template's own arm, no proxy planning it: the site's errors are
            # heavy-tailed across draws alike in both arms, which no count of either alone resolves. Where
            # the template refuses the call its test skips the site, which no recording then serves.
            dew = jax.jit(functools.partial(lambda hooks, arrays, leaves: ruled(
                functools.partial(chunk_on, 'mosaic_gpu', hooks), arrays, leaves), site.hooks,
                leaves=site.leaves))
            try:
                dew(sample)
            except KernelRefused as refused:
                print(f"{name}: not recorded, the template refuses it ({refused})", flush=True)
                continue
            stored.update(record_case(name, site.arrays, evaluate, store_inputs=False, summary=True, dew=dew))
        else:
            stored.update(record_case(name, site.arrays, evaluate, portable, truth_proxy=exact_run,
                                      store_inputs=False, summary=True))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        version, version, device.strip(),
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_gdn_reference.py <fla python> <unit> "
        "<site>...",
        truth="fla's naive_recurrent_gated_delta_rule and naive_recurrent_kda on each sequence in torch "
              "float64",
        cross_check="Dew's token recurrence in float64; assert_computes_the_oracle with its traced "
                    "chain_roundings")
    np.savez(out, **stored)
    print(f"wrote {out}: {out.stat().st_size:,} bytes", flush=True)


def merged(units: list[Path]) -> Path:
    """The recording's units (`write_seeded`'s, their sites disjoint) joined and saved as the fixture, its
    provenance the units' own, which must agree but for the device: fla autotunes each device's configs, so
    where units ran on more than one, the provenance names each device's sites (`devices`)."""
    from fetched_fixtures import save

    stored: dict[str, np.ndarray] = {}
    devices: dict[str, list[str]] = {}
    common = None
    for unit in units:
        with np.load(unit, allow_pickle=False) as part:
            recorded = json.loads(str(part["provenance"]))
            device = recorded.pop("device")
            if common is not None and recorded != common:
                raise ValueError(f"{unit} was recorded under another provenance: {recorded}")
            common = recorded
            for key, value in part.items():
                if key == "provenance":
                    continue
                if key in stored:
                    raise ValueError(f"{unit} records {key} again")
                stored[key] = value
                site = key.split(".", 1)[0]
                if site not in devices.setdefault(device, []):
                    devices[device].append(site)
    assert common is not None, "a fixture merges at least one unit"
    named = {"device": next(iter(devices))} if len(devices) == 1 else {"device": " and ".join(devices),
                                                                       "devices": devices}
    stored["provenance"] = np.asarray(json.dumps({**common, **named}))
    return save(FIXTURE, stored, reference=common["reference"], license="MIT")


def main() -> None:
    """`--serve`: the torch side. `--list`: the sites, a line each. `--merge <units>`: the fixture from the
    recording's units. Else `<fla python> <out> <site>...`: those sites' unit, written to `out`."""
    if sys.argv[1] == "--serve":
        serve()
    elif sys.argv[1] == "--list":
        from fla_sites import sites

        print("\n".join(sites()), flush=True)
    elif sys.argv[1] == "--merge":
        written = merged([Path(unit) for unit in sys.argv[2:]])
        print(f"wrote {FIXTURE.name}: {written} ({written.stat().st_size:,} bytes)", flush=True)
    else:
        from reference_worker import session

        with session([sys.argv[1], __file__, "--serve"]) as worker:
            write_seeded(worker, sys.argv[3:], Path(sys.argv[2]))


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
