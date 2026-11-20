#!/usr/bin/env python3
"""The fixture tests hold the recurrence template's gated delta rule to: fla-core's own
`chunk_gated_delta_rule`, `chunk_kda`, `chunk_rwkv6` and `chunk_simple_gla` (its Triton kernels) on a CUDA
GPU, forward and every gradient, at each site tests/fla_sites.py names, over the tests' own seeded draws.

    PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_gdn_reference.py <fla> <unit> <site>...
    PYTHONPATH=src:tests:tools python tools/fla_gdn_reference.py --merge <unit>...

on a GPU, Dew's environment recording and a second Python with torch and fla-core (`--serve`, the
"fla" cohort, tools/reference_releases.py) running fla and its reference, each run a unit of named sites
(`--list` names them all), the units then merged. What lands is tests/fixtures/fla/gdn_seeded.npz, a
summary recording (`record_case(..., summary=True)`): per site, seed and
leaf, fla's distance from float64 at the site's inputs, which the test draws again (`Site.arrays`), and the
truth's sketch, which Dew's float64 token recurrence is held to.

fla runs at its defaults (an fp32 call's products TF32, as the fast forms' bar is). Packed documents run as
fla runs them, one flattened row of sequences (`cu_seqlens`), a row's first taking its entering memory
unless a document starts it. The oracle is fla's own reference, `naive_recurrent_gated_delta_rule`,
`naive_recurrent_kda` and `naive_recurrent_rwkv6` (fla/ops/*/naive.py, rwkv6/recurrent_naive.py), on each
sequence in torch float64 (`_dew_torch.precision.float64`, which widens their fp32 casts) on the GPU, held at
every seed to Dew's float64 token recurrence. RWKV6's sites (an own weight `own`, fla's `u`, and no write
gate) take its `chunk_rwkv6(r, k, v, w, u)`, its naive reference laid `[B, H, T, K]` as it reads them. A
write gate without the delta rule under a decay a head takes `chunk_simple_gla(q, k beta, v, g)`, its gate on
its keys (the template's `gated_keys`), held to `naive_recurrent_simple_gla` alike.
"""

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

COHORT = "fla"
"""The reference releases this tool runs against (tools/reference_releases.py), and the environment
tools/check_tool_types.py checks it in."""

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fla" / "gdn_seeded.npz"
GRADIENTS = ("q", "k", "v", "g", "beta", "h0")
"""The delta rules' differentiated operands; RWKV6's take its own weight in the write gate's place."""

OWN_GRADIENTS = ("q", "k", "v", "g", "own", "h0")


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
        gradients: bool = True, simple: bool = False) -> dict[str, np.ndarray]:
    """`rule` (fla's chunked rule, or its naive reference with `per_sequence`) on a site's arrays: its rows'
    sequences flattened into one row (`cu_seqlens`), or each alone, the operands in `dtype` but the log decay
    and memory fp32 (in float64 for the reference), and with `gradients` the output's and final memory's
    gradients under the drawn cotangents; leaves as `fla_sites.LEAVES`, the output and final memory alone
    without them. A `simple` rule (fla's simple GLA) takes the write gate on its keys, `k beta`, which its
    gradient's chain gives `beta`'s."""
    import torch

    wide = dtype == torch.float64
    owned = "own" in arrays
    differentiated = OWN_GRADIENTS if owned else GRADIENTS
    tensors = {name: torch.tensor(np.asarray(value, np.float64 if wide else np.float32), device=device)
               for name, value in arrays.items() if name != "starts"}
    inputs = {name: tensors[name].to(dtype if name in ("q", "k", "v", "beta") else
                                     (torch.float64 if wide else torch.float32)).requires_grad_()
              for name in differentiated}
    spans = sequences(arrays)
    group = inputs["v"].shape[2] // inputs["q"].shape[2]

    def piece(name, row, start, end):
        # Each value head its key head's queries and keys where the rule reads one a value head: fla's
        # naive references, and RWKV6's chunk, which takes no grouped heads.
        x = inputs[name][row:row + 1, start:end]
        return (x.repeat_interleave(group, dim=2) if (per_sequence or owned or simple) and name in ("q", "k")
                else x)

    def called(q, k, v, g, fifth, state, **flat):
        """The site's rule on one sequence's operands, or the flattened row's (`flat`: `cu_seqlens`):
        RWKV6's (`own` its `u`, the naive reference reading `[B, H, T, K]`), the simple GLA's (`fifth` the
        write gate, on its keys), the KDA's or the gated delta rule's."""
        if owned:
            if per_sequence:
                out, final = rule(*(x.transpose(1, 2) for x in (q, k, v, g)), fifth, scale=scale,
                                  initial_state=state, output_final_state=True)
                return out.transpose(1, 2), final
            return rule(q, k, v, g, fifth, scale=scale, initial_state=state, output_final_state=True, **flat)
        if simple:
            return rule(q, k * fifth.unsqueeze(-1), v, g, scale=scale, initial_state=state,
                        output_final_state=True, **flat)
        if kda:
            return rule(q, k, v, g, fifth, scale=scale, initial_state=state, output_final_state=True, **flat)
        return rule(q, k, v, fifth, g, scale=scale, initial_state=state, output_final_state=True, **flat)

    states = [inputs["h0"][row:row + 1] if entering else torch.zeros_like(inputs["h0"][:1])
              for row, _, _, entering in spans]
    if per_sequence:
        outs, finals = [], []
        for (row, start, end, _), state in zip(spans, states, strict=True):
            q, k, v, g = (piece(name, row, start, end) for name in ("q", "k", "v", "g"))
            fifth = inputs["own"] if owned else piece("beta", row, start, end)
            out, final = called(q, k, v, g, fifth, state)
            outs.append(out)
            finals.append(final)
        out = torch.cat(outs, dim=1)
    else:
        q, k, v, g = (torch.cat([piece(name, row, start, end) for row, start, end, _ in spans], dim=1)
                      for name in ("q", "k", "v", "g"))
        fifth = inputs["own"] if owned else torch.cat([piece("beta", row, start, end)
                                                       for row, start, end, _ in spans], dim=1)
        lengths = [end - start for _, start, end, _ in spans]
        cu_seqlens = torch.tensor(np.cumsum([0, *lengths]), device=device, dtype=torch.long)
        out, final = called(q, k, v, g, fifth, torch.cat(states, dim=0), cu_seqlens=cu_seqlens)
        finals = list(final.split(1, dim=0))
    rows = inputs["v"].shape[0]
    out = out.reshape(rows, -1, *out.shape[2:])
    last = {row: index for index, (row, _, _, _) in enumerate(spans)}
    final = torch.cat([finals[last[row]] for row in range(rows)], dim=0)
    if not gradients:
        return {name: value.detach().to(torch.float64).cpu().numpy()
                for name, value in (("out", out), ("final", final))}
    grads = torch.autograd.grad((out, final), tuple(inputs[name] for name in differentiated),
                                (tensors["dy"].to(out.dtype), tensors["dfinal"].to(final.dtype)),
                                allow_unused=True)
    found = dict(zip(("out", "final", *(f"d{name}" for name in differentiated)), (out, final, *grads),
                     strict=True))
    return {name: (np.zeros(arrays[name[1:]].shape) if value is None else
                   value.detach().to(torch.float64).cpu().numpy())
            for name, value in found.items()}


def serve() -> None:
    """The torch side (tools/reference_worker.py's protocol): each request a site's inputs with its `dtype`,
    `kind` ('kda', 'gdn', 'rwkv6' or 'simple_gla'), `scale` and `leaves` ('all', or 'forward' for the output
    and final memory alone), answered with fla's outputs and the reference's (`reference.<leaf>`,
    `truth.<leaf>`), after one `ready` line of fla's, torch's and Triton's releases and the device."""
    import reference_worker
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    from fla.ops.rwkv6 import chunk_rwkv6
    from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6
    from fla.ops.simple_gla import chunk_simple_gla
    from fla.ops.simple_gla.naive import naive_recurrent_simple_gla
    from reference_releases import require

    from _dew_torch.precision import float64

    require("fla")
    from importlib.metadata import version

    # The releases fla's kernels run on: its Triton, torch's, past fla #640's Hopper miscompile (Triton 3.4 to
    # 3.7.0, chunk_bwd_dqkwg) on the cohort's torch, so fla takes no tilelang backward there.
    torch_release = version('torch').split('+')[0]
    found = f"fla-core-{version('fla-core')}+torch-{torch_release}+triton-{version('triton')}"

    def answer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dtype, kind, leaves = (str(arrays.pop(name)) for name in ("dtype", "kind", "leaves"))
        scale = float(arrays.pop("scale"))
        kda, simple, gradients = kind == "kda", kind == "simple_gla", leaves == "all"
        rule = {"kda": chunk_kda, "gdn": chunk_gated_delta_rule, "rwkv6": chunk_rwkv6,
                "simple_gla": chunk_simple_gla}[kind]
        began = time.perf_counter()
        reference = run(arrays, kda=kda, scale=scale, rule=rule, dtype=getattr(torch, dtype), device="cuda",
                        per_sequence=False, gradients=gradients, simple=simple)
        referenced = time.perf_counter()
        # The truth in float64 on the GPU: on the CPU its token loop and autograd took most of a site's
        # 1432 s for 320 draws (fla-record3-h100-f8fbb058d, the worker at a core's whole time).
        with float64():
            naive = {"kda": naive_recurrent_kda, "gdn": naive_recurrent_gated_delta_rule,
                     "rwkv6": naive_recurrent_rwkv6, "simple_gla": naive_recurrent_simple_gla}[kind]
            truth = run(arrays, kda=kda, scale=scale, rule=naive, dtype=torch.float64, device="cuda",
                        per_sequence=True, gradients=gradients, simple=simple)
        print(f"timed fla {referenced - began:.3f} s truth {time.perf_counter() - referenced:.3f} s",
              file=sys.stderr, flush=True)
        return reference_worker.paired(reference, truth)

    reference_worker.serve(answer, f"{found} {torch.cuda.get_device_name()}")


def write_seeded(worker, names: list[str], out: Path) -> None:
    """Each named site's seeded draws with fla's outputs and gradients and their float64 truth (`worker`,
    `serve`'s `Session`), recorded as a summary (`record_case`) into `out`, Dew's float64 token
    recurrence held to fla's reference at every seed: a unit
    of the recording, whose units `reference_worker.merged_units` joins into the fixture."""
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

        # The query's scale as the rule's prologue takes it (`recurrence._prologue`): its own, else an
        # l2-normed query's keys' width's, else none.
        hooks = site.hooks
        scale = (hooks.scale if hooks.scale is not None else sample["k"].shape[-1] ** -0.5 if hooks.l2norm
                 else 1.0)

        def fla(arrays, dtype=dtype, kda=kda, scale=scale, leaves=leaves, delta=hooks.delta):
            request = {key: np.asarray(value, np.float32) if key != "starts" else value
                       for key, value in arrays.items()}
            kind = "rwkv6" if "own" in arrays else "kda" if kda else "gdn" if delta else "simple_gla"
            options = {"dtype": dtype, "kind": kind, "scale": scale, "leaves": leaves}
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

        if site.paired:
            # Sized paired with the template's own arm: the site's errors are
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
            stored.update(record_case(name, site.arrays, evaluate, store_inputs=False, summary=True))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        version, version, device.strip(),
        "PYTHONPATH=src:tests:tools JAX_PLATFORMS=cpu python tools/fla_gdn_reference.py <fla python> <unit> "
        "<site>...",
        truth="fla's naive_recurrent_gated_delta_rule, naive_recurrent_kda, naive_recurrent_rwkv6 and "
              "naive_recurrent_simple_gla on each sequence in torch float64",
        cross_check="Dew's token recurrence in float64; assert_computes_the_oracle with its traced "
                    "chain_roundings")
    np.savez(out, allow_pickle=False, **stored)
    print(f"wrote {out}: {out.stat().st_size:,} bytes", flush=True)


def main() -> None:
    """The recorder's command line (`reference_worker.recording_command`), its sites `fla_sites`'s."""
    from reference_worker import recording_command

    recording_command(__file__, "fla_sites", serve=serve, write_seeded=write_seeded,
                      fixture=FIXTURE, license="MIT")


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    main()
