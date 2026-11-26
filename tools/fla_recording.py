"""The recorders of fla-core's chunked recurrences in their own cohort's environment
(tools/fla_momentum_reference.py, tools/fla_slots_reference.py): the torch side's answer to each request
(`serve`) and the recording of a unit of named sites (`write_seeded`), each recorder its own `run` and
sites."""

import sys
import time
from collections.abc import Callable

import numpy as np


def serve(cohort: str, run: Callable, options: tuple[str, ...]) -> None:
    """The torch side (tools/reference_worker.py's protocol): each request a site's inputs with its
    `options` (each a string, `dtype` among them, named a torch dtype), answered with fla's outputs
    (`run(..., wide=False)`) and its float64 reference's (`run(..., wide=True)` under
    `_dew_torch.precision.float64`) as `reference.<leaf>` and `truth.<leaf>`, after one `ready` line of fla's,
    torch's and Triton's releases and the device."""
    from importlib.metadata import version

    import reference_worker
    import torch
    from reference_releases import require

    from _dew_torch.precision import float64

    require(cohort)
    torch_release = version('torch').split('+')[0]
    found = f"fla-core-{version('fla-core')}+torch-{torch_release}+triton-{version('triton')}"

    def answer(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        given = {name: str(arrays.pop(name)) for name in options}
        dtype = getattr(torch, given.pop("dtype"))
        began = time.perf_counter()
        reference = run(arrays, dtype=dtype, device="cuda", wide=False, **given)
        referenced = time.perf_counter()
        with float64():
            truth = run(arrays, dtype=torch.float64, device="cuda", wide=True, **given)
        print(f"timed fla {referenced - began:.3f} s truth {time.perf_counter() - referenced:.3f} s",
              file=sys.stderr, flush=True)
        return reference_worker.paired(reference, truth)

    reference_worker.serve(answer, f"{found} {torch.cuda.get_device_name()}")


def write_seeded(worker, names: list[str], out, *, sites: dict, options: Callable[..., dict],
                 exact: Callable[..., Callable], command: str, truth: str) -> None:
    """Each named site's (`sites`) seeded draws with fla's outputs and gradients and their float64 truth
    (`worker`, `serve`'s session, each request the site's `options(site, sample)`), recorded as a summary
    (`record_case`) into `out`, Dew's float64 token recurrence (`exact(site)`, a site's arrays to its leaves)
    held to fla's reference at every seed: a unit of the recording, which `reference_worker.merged_units`
    joins into the fixture."""
    import jax
    from recorded_references import oracle_held, provenance, record_case
    from reference_error import chain_roundings, widened
    from reference_worker import arms

    version, device = worker.started.split(maxsplit=1)
    stored = {}
    for name in names:
        site, began = sites[name], time.perf_counter()
        sample = site.arrays(0)
        extra = {key: np.asarray(value) for key, value in options(site, sample).items()}

        def fla(arrays, extra=extra):
            request = {key: np.asarray(value, np.float32) for key, value in arrays.items()}
            return arms(worker.ask(request | extra))

        exact_run = jax.jit(exact(site))
        with jax.enable_x64(new_val=True):
            roundings = chain_roundings(jax.make_jaxpr(exact_run)(widened(sample)))

        evaluate = oracle_held(fla, exact_run, roundings, f"fla {name}")
        stored.update(record_case(name, site.arrays, evaluate, store_inputs=False, summary=True))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        version, version, device.strip(), command, truth=truth,
        cross_check="Dew's token recurrence in float64; assert_computes_the_oracle with its traced "
                    "chain_roundings")
    np.savez(out, allow_pickle=False, **stored)
    print(f"wrote {out}: {out.stat().st_size:,} bytes", flush=True)


def main(tool: str, sites: str, fixture: str, *, serve: Callable, write_seeded: Callable) -> None:
    """A recorder's command line (`reference_worker.recording_command`): `tool`'s file, its sites' module
    `sites`, and its fixture `tests/fixtures/fla/<fixture>`."""
    import os
    from pathlib import Path

    from reference_worker import recording_command

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    recorded = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fla" / fixture
    recording_command(tool, sites, serve=serve, write_seeded=write_seeded, fixture=recorded, license="MIT")
