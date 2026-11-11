#!/usr/bin/env python3
"""The recording tests/recurrence_matrix.py holds a cell to where no GPU tier's fla runs (a CPU lane's
interpreted cells): fla's own layer (tests/delta_rule_support.py's `FLA_LAYER_SIDE`, a draw a request) on a
CUDA GPU over the cell's draws, which are host NumPy's and the test draws again (`recurrence_matrix._inputs`),
Dew's float64 token recurrence on the GPU the truth, recorded as a summary (`record_case(..., summary=True)`),
sized paired with the cell's own kernel.

    PYTHONPATH=src:tests:tools python tools/fla_matrix_reference.py <fla python> <case>...

each case its axis values joined by `-` (`recurrence_matrix._on_par_recorded`'s site), on a GPU, `<fla
python>` the "fla" cohort's interpreter (tools/reference_releases.py) serving fla's layer (`--serve`). What
lands is tests/fixtures/fla/matrix_seeded.npz, or past `fetched_fixtures.IN_GIT` its pin beside the store's
copy (tests/fixtures/fetched/), uploaded by `fetched_fixtures.upload`.
"""

import dataclasses
import functools
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def serve(source: str) -> None:
    """The torch side: fla's layer (`FLA_LAYER_SIDE`'s `fla_layer`, its source at `source`) answering each
    request (`delta_rule_support.fla_request`), its startup line fla's release and the device."""
    import fla
    import reference_worker
    import torch

    namespace: dict[str, Any] = {"__name__": "fla_layer"}
    exec(Path(source).read_text(), namespace)

    def answer(request):
        return namespace["fla_layer"](request, str(request["form"]), str(request["dtype"]))

    reference_worker.serve(answer, f"{fla.__version__} {torch.cuda.get_device_name()}")


def recorded(worker, label: str) -> dict:
    """The cell `label`'s summary recording: fla's layer through `worker` on each seed's draw and Dew's
    float64 token recurrence the truth, its count sized paired with the cell's kernel."""
    import jax
    import numpy as np
    import recurrence_matrix as matrix
    from delta_rule_support import fla_request
    from kernel_matrix import AXES
    from recorded_references import record_case
    from reference_error import widened, with_gradients
    from test_recurrence_chunk import token_by_token

    named = dict(zip(AXES["recurrence"], label.split("-"), strict=True))
    hooks, starts = matrix.FORMS[named["form"]], matrix._starts(named)
    step = jax.jit(matrix._step(named, named["kernel"]))
    exact = jax.jit(with_gradients(lambda operands: matrix._wide(token_by_token(
        hooks, dataclasses.replace(operands, starts=starts)))))
    epsilon = 0.0 if hooks.output is None else hooks.output.epsilon

    def evaluate(inputs):
        operands = {name: value[None] for name, value in inputs.items() if name not in ("d_out", "d_final")}
        answer = worker.ask(fla_request(matrix.FLA_LAYERS[named["form"]], operands,
                                        (inputs["d_out"][None], inputs["d_final"][None]), starts,
                                        named["dtype"], epsilon=epsilon))
        with jax.enable_x64(new_val=True):
            truth = matrix._leaves(exact(*widened(matrix._split(named, inputs))))
        return ({name: value[0] for name, value in answer.items()},
                {name: np.asarray(value, np.float64) for name, value in truth.items()})

    def dew(inputs):
        return {name: np.asarray(value) for name, value in
                matrix._leaves(step(*matrix._split(named, inputs))).items()}

    return record_case(label, functools.partial(matrix._inputs, named), evaluate, store_inputs=False,
                       summary=True, dew=dew)


def main() -> None:
    """`--serve <source>`: the torch side. Else the recording of each named cell, saved as the fixture."""
    if sys.argv[1] == "--serve":
        serve(sys.argv[2])
        return
    from delta_rule_support import FLA_LAYER_SIDE
    from fetched_fixtures import save
    from recorded_references import provenance
    from recurrence_matrix import RECORDED
    from reference_worker import session

    python, *labels = sys.argv[1:]
    stored = {}
    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "fla_layer.py"
        source.write_text(FLA_LAYER_SIDE)
        with session([python, __file__, "--serve", str(source)]) as worker:
            version, device = worker.started.split(maxsplit=1)
            for label in labels:
                began = time.perf_counter()
                stored.update(recorded(worker, label))
                print(f"{label}: recorded in {time.perf_counter() - began:.0f} s", flush=True)
    stored["provenance"] = provenance(
        "fla", version, device.strip(),
        "PYTHONPATH=src:tests:tools python tools/fla_matrix_reference.py <fla python> <case>...",
        truth="Dew's token recurrence in float64 on the GPU")
    written = save(RECORDED["fla"], stored, reference=f"fla-core-{version}", license="MIT")
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
