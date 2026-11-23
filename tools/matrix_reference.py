#!/usr/bin/env python3
"""The recordings tests/recurrence_matrix.py holds a cell to where no GPU tier's upstream runs (a CPU lane's
interpreted cells, a TPU's): an upstream's own computation on a CUDA GPU over the cell's draws, which are host
NumPy's and the test draws again (`recurrence_matrix._inputs`), Dew's float64 token recurrence on the GPU the
truth, recorded as a summary (`record_case(..., summary=True)`): fla's own layer
(tests/delta_rule_support.py's `FLA_LAYER_SIDE`, a draw a request, served by `--serve`), sized paired with
the cell's own kernel; or mamba_ssm's scan (`recurrence_matrix.UPSTREAMS`, `ssd_support.mamba_side`, a block
of draws a call), sized on the recording alone where the cell's kernel is a TPU's, which no GPU runs (row
12126).

    PYTHONPATH=src:tests:tools python tools/matrix_reference.py <upstream> <python> <case>...

`<upstream>` `fla` or `mamba_ssm`, each case its axis values joined by `-`
(`recurrence_matrix._on_par_recorded`'s site), on a GPU, `<python>` the upstream's cohort's interpreter
(tools/reference_releases.py, `COHORTS`; mamba_ssm's its tier release, `ssd_support.MAMBA_SSM`, refused
otherwise). What lands is the upstream's file (`recurrence_matrix.RECORDED`, `DESTINATIONS`), or past
`fetched_fixtures.IN_GIT` its pin beside the store's copy (tests/fixtures/fetched/), uploaded by
`fetched_fixtures.upload`.
"""

import dataclasses
import functools
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

COHORTS = {"fla": "fla", "mamba_ssm": "mamba-ssm"}
"""Each upstream's reference releases (tools/reference_releases.py): the environment its variant runs and
tools/check_tool_types.py checks it in."""

BLOCK = 16
"""The draws a call of mamba_ssm's scan takes at once (`mamba_side`, an interpreter's start a call)."""


def serve(source: str) -> None:
    """The torch side: fla's layer (`FLA_LAYER_SIDE`'s `fla_layer`, its source at `source`) answering each
    request (`delta_rule_support.fla_request`), its startup line fla's release and the device. fla itself
    imported by the layer's source alone, so this tool's other variant's environment needs none
    (`COHORTS`)."""
    from importlib.metadata import version

    import reference_worker
    import torch

    namespace: dict[str, Any] = {"__name__": "fla_layer"}
    exec(Path(source).read_text(), namespace)

    def answer(request):
        return namespace["fla_layer"](request, str(request["form"]), str(request["dtype"]))

    reference_worker.serve(answer, f"{version('fla-core')} {torch.cuda.get_device_name()}")


def recorded(answered, label: str) -> dict:
    """The cell `label`'s summary recording: `upstream`'s answer on each seed's draw (`answered(named,
    starts)`'s function of a draw's inputs, `fla_answers` or `MambaAnswers`) and Dew's float64 token
    recurrence the truth, its count sized paired with the cell's kernel where a GPU runs it, else on the
    recording alone (a TPU's cell)."""
    import jax
    import numpy as np
    import recurrence_matrix as matrix
    from kernel_matrix import AXES
    from recorded_references import record_case
    from reference_error import widened, with_gradients
    from test_recurrence_chunk import token_by_token

    named = dict(zip(AXES["recurrence"], label.split("-"), strict=True))
    hooks, starts = matrix.FORMS[named["form"]], matrix._starts(named)
    exact = jax.jit(with_gradients(lambda operands: matrix._wide(token_by_token(
        hooks, dataclasses.replace(operands, starts=starts)))))
    answer = answered(named, starts)

    def evaluate(inputs):
        with jax.enable_x64(new_val=True):
            truth = matrix._leaves(exact(*widened(matrix._split(named, inputs))))
        return answer(inputs), {name: np.asarray(value, np.float64) for name, value in truth.items()}

    dew = (None if named["kernel"] == "tpu"
           else functools.partial(dew_leaves, named, jax.jit(matrix._step(named, named["kernel"]))))
    return record_case(label, answer.draw if hasattr(answer, "draw") else functools.partial(
        matrix._inputs, named), evaluate, store_inputs=False, summary=True, dew=dew)


def dew_leaves(named: dict, step, inputs: dict) -> dict:
    """Dew's own cell `named`'s `step` on a recording's draw `inputs`, its leaves host arrays: what the
    recording's count is sized paired with (`record_case`'s `dew`)."""
    import numpy as np
    import recurrence_matrix as matrix

    leaves = matrix._leaves(step(*matrix._split(named, inputs)))
    return {name: np.asarray(value) for name, value in leaves.items()}


def fla_answers(worker):
    """fla's layer through `worker` (`serve`), a draw a request: `recorded`'s `answered`."""
    import recurrence_matrix as matrix
    from delta_rule_support import fla_request

    def answered(named, starts):
        hooks = matrix.FORMS[named["form"]]
        epsilon = 0.0 if hooks.output is None else hooks.output.epsilon

        def answer(inputs):
            operands = {name: value[None] for name, value in inputs.items()
                        if name not in ("d_out", "d_final")}
            found = worker.ask(fla_request(matrix.FLA_LAYERS[named["form"]], operands,
                                           (inputs["d_out"][None], inputs["d_final"][None]), starts,
                                           named["dtype"], epsilon=epsilon))
            return {name: value[0] for name, value in found.items()}

        return answer

    return answered


class MambaAnswers:
    """mamba_ssm's scan on a cell's draws (`recurrence_matrix.UPSTREAMS`), `BLOCK` draws a call from the
    first a draw not yet answered asks: `recorded`'s answer of a cell, its `draw` the cell's own
    (`recurrence_matrix._inputs`), each draw found by its digest (`upstream_support.digest`). One block's
    answers held at a time, and the seeds of draws not yet answered, as `record_case` streams a summary's
    draws in order: every block's held, a cell's sizing draws held every output whole."""

    def __init__(self, named: dict, starts):
        self.named, self.starts, self.seeds, self.held = named, starts, {}, {}

    def draw(self, seed: int) -> dict:
        import recurrence_matrix as matrix
        from upstream_support import digest

        inputs = matrix._inputs(self.named, seed)
        self.seeds[digest(inputs)] = seed
        return inputs

    def __call__(self, inputs: dict) -> dict:
        import numpy as np
        import recurrence_matrix as matrix
        from upstream_support import digest

        key = digest(inputs)
        if key not in self.held:
            first = self.seeds[key]
            self.seeds = {each: seed for each, seed in self.seeds.items() if seed >= first}
            drawn = [matrix._inputs(self.named, seed) for seed in range(first, first + BLOCK)]
            names = [name for name in drawn[0] if name not in ("d_out", "d_final")]
            cotangents = tuple(np.stack([each[name] for each in drawn]) for name in ("d_out", "d_final"))
            found = matrix.UPSTREAMS["mamba_ssm"](
                self.named, {name: np.stack([each[name] for each in drawn]) for name in names}, cotangents,
                self.starts)
            self.held = {digest(each): {name: value[index] for name, value in found.items()}
                         for index, each in enumerate(drawn)}
        return self.held[key]


MAMBA_RELEASE = ("import importlib.metadata as m, torch\n"
                 "print(m.version('mamba_ssm'), torch.cuda.get_device_name())")
"""The torch side's mamba_ssm release (its installed version) and its device, a line."""

DESTINATIONS = {"mamba_ssm": Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mamba2" /
                "matrix_seeded.npz"}
"""Where an upstream's recording lands that tests/recurrence_matrix.py's `RECORDED` does not register yet
(registered beside its fetched manifest once uploaded)."""


def main() -> None:
    """`--serve <source>`: fla's torch side. Else the recording of each named cell of the upstream named,
    saved as its fixture (`recurrence_matrix.RECORDED`)."""
    if sys.argv[1] == "--serve":
        serve(sys.argv[2])
        return
    import os
    import subprocess

    from delta_rule_support import FLA_LAYER_SIDE
    from fetched_fixtures import save
    from recorded_references import provenance
    from recurrence_matrix import RECORDED
    from reference_worker import session

    upstream, python, *labels = sys.argv[1:]
    stored = {}
    if upstream == "fla":
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "fla_layer.py"
            source.write_text(FLA_LAYER_SIDE)
            with session([python, __file__, "--serve", str(source)]) as worker:
                version, device = worker.started.split(maxsplit=1)
                for label in labels:
                    began = time.perf_counter()
                    stored.update(recorded(fla_answers(worker), label))
                    print(f"{label}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
        reference, license = f"fla-core-{version}", "MIT"
    elif upstream == "mamba_ssm":
        from ssd_support import MAMBA_SSM

        os.environ["DEW_MAMBA_PYTHON"] = python
        version, device = subprocess.run([python, "-c", MAMBA_RELEASE], check=True, capture_output=True,
                                         text=True).stdout.split(maxsplit=1)
        # The release a replay holds the recording to, checked before any GPU time is spent.
        if version != MAMBA_SSM:
            raise SystemExit(f"{python} serves mamba_ssm {version}; the tier's reference is {MAMBA_SSM}")
        for label in labels:
            began = time.perf_counter()
            stored.update(recorded(MambaAnswers, label))
            print(f"{label}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
        reference, license = f"mamba_ssm {version}", "Apache-2.0"
    else:
        raise SystemExit(f"an upstream {sorted(COHORTS)} names, not {upstream!r}")
    stored["provenance"] = provenance(
        upstream, version, device.strip(),
        f"PYTHONPATH=src:tests:tools python tools/matrix_reference.py {upstream} <python> <case>...",
        truth="Dew's token recurrence in float64 on the GPU")
    written = save(RECORDED.get(upstream, DESTINATIONS.get(upstream)), stored, reference=reference,
                   license=license)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
