#!/usr/bin/env python3
"""The recording the rated scan's stability sites are held to (tests/mamba_sites.py, row 6781): mamba_ssm's
own `mamba_chunk_scan_combined` on a CUDA GPU (tests/ssd_support.py's `mamba_side`, the interpreter
`DEW_MAMBA_PYTHON` names, `BLOCK` draws a call), forward and every gradient, at each site's seeded draws,
which are host arrays the test draws again (`Site.arrays`), Dew's float64 token recurrence the truth,
recorded as a summary (`record_case(..., summary=True)`).

    PYTHONPATH=src:tests:tools python tools/mamba_stability_reference.py <mamba python> [<site>...]

on a GPU, `<mamba python>` the "mamba-ssm" cohort's interpreter (tools/reference_releases.py) serving the
tier's release (`recurrence_matrix.tier_release`), refused otherwise; every site of `mamba_sites.sites()`
where none is named. What lands is tests/fixtures/mamba2/stability_seeded.npz (`mamba_sites.RECORDED`), or
past `fetched_fixtures.IN_GIT` its pin beside the store's copy, uploaded by `fetched_fixtures.upload`.
"""

import os
import subprocess
import sys
import time

COHORT = "mamba-ssm"
"""The reference releases this tool runs against (tools/reference_releases.py)."""

BLOCK = 16
"""The draws a call of mamba_ssm's scan takes at once (`mamba_side`, an interpreter's start a call)."""


def answers(site):
    """mamba_ssm's scan on `site`'s draws (`mamba_side`), `BLOCK` draws a call from the draw asked, one
    block's answers held at a time (`upstream_support.UpstreamBlocks`), its `draw` the site's own: each
    draw's leaves, the site's of mamba_side's (`out`, `final`, `d_<operand>`)."""
    import numpy as np
    from mamba_sites import OPERANDS
    from matrix_reference import BlockAnswers
    from ssd_support import mamba_side
    from upstream_support import UpstreamBlocks

    from dew.nn.kernels import recurrence_chunk

    def run_block(drawn: list) -> list:
        arrays = [each for (each,) in drawn]
        starts = [each.get("starts") for each in arrays]
        # One call's documents are one layout: each regime's restarts are its draws' alike.
        assert all(np.array_equal(found, starts[0]) for found in starts), "a block's documents differ"
        found = mamba_side({name: np.stack([each[name] for each in arrays]) for name in OPERANDS},
                           (np.stack([each["dy"] for each in arrays]),
                            np.stack([each["dfinal"] for each in arrays])),
                           starts[0], recurrence_chunk.CHUNK_SIZE)
        return [{name: found[name][index] for name in site.leaves} for index in range(len(arrays))]

    return BlockAnswers(UpstreamBlocks(run_block, lambda seed: (site.arrays(seed),), BLOCK, bounded=True))


def recorded(name: str) -> dict:
    """Site `name`'s summary recording: mamba_ssm's answer on each seed's draw (`answers`) and Dew's float64
    token recurrence the truth."""
    import functools

    import jax
    import numpy as np
    import test_recurrence_stability as stability
    from mamba_sites import ruled, sites
    from recorded_references import record_case
    from reference_error import widened
    from test_recurrence_chunk import token_by_token

    site = sites()[name]
    answer = answers(site)
    exact = jax.jit(lambda arrays: ruled(functools.partial(token_by_token, stability.SITES["rated"]), arrays,
                                         site.leaves))

    def evaluate(inputs):
        with jax.enable_x64(new_val=True):
            truth = exact(widened(inputs))
        return answer(inputs), {leaf: np.asarray(value, np.float64) for leaf, value in truth.items()}

    return record_case(name, answer.draw, evaluate, store_inputs=False, summary=True)


MAMBA_RELEASE = ("import importlib.metadata as m, torch\n"
                 "print(m.version('mamba_ssm'), torch.cuda.get_device_name())")
"""The torch side's mamba_ssm release (its installed version) and its device, a line."""


def main() -> None:
    """The recording of each named site (every site where none is named), saved as `mamba_sites.RECORDED`."""
    from fetched_fixtures import save
    from mamba_sites import RECORDED, sites
    from recorded_references import provenance
    from recurrence_matrix import tier_release

    python, *names = sys.argv[1:]
    os.environ["DEW_MAMBA_PYTHON"] = python
    version, device = subprocess.run([python, "-c", MAMBA_RELEASE], check=True, capture_output=True,
                                     text=True).stdout.split(maxsplit=1)
    # The release a replay holds the recording to, checked before any GPU time is spent.
    if version != tier_release("mamba_ssm"):
        raise SystemExit(f"{python} serves mamba_ssm {version}; the tier's reference is "
                         f"{tier_release('mamba_ssm')}")
    stored = {}
    for name in names or list(sites()):
        began = time.perf_counter()
        stored.update(recorded(name))
        print(f"{name}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        "mamba_ssm", version, device.strip(),
        "PYTHONPATH=src:tests:tools python tools/mamba_stability_reference.py <mamba python> [<site>...]",
        truth="Dew's token recurrence in float64 on the GPU")
    written = save(RECORDED, stored, reference=f"mamba_ssm {version}", license="Apache-2.0")
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
