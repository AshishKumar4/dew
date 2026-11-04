"""One rank of a pool running the layout tool's step (`tools/layout_parity.py`): the rows it reads on this
process, each row's place among the step's examples."""
import sys

import jax
import numpy as np
from process_support import REPO_ROOT, serve_mode, worker_parser

from dew.batch import EXAMPLE_ROWS


def layout_rows(arguments) -> dict:
    """Each row's place among the step's examples (`EXAMPLE_ROWS`) the layout tool's step reads on this
    process (`layout_parity.compiled_step`), over an fsdp axis of every device, gathered from every
    process, beside the case's batch."""
    from jax.experimental import multihost_utils

    del arguments
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import layout_parity as tool

    case, _ = tool.model_case("dense", "float32", {}, {})
    _, data, _, _, _ = tool.compiled_step(case, {"fsdp": jax.device_count()}, tool.drawn_batch(case, 0))
    rows = np.asarray(multihost_utils.process_allgather(data[EXAMPLE_ROWS], tiled=True))
    return {"rows": [int(row) for row in rows], "batch": case.batch_size}


if __name__ == "__main__":
    modes = {"layout_rows": layout_rows}
    serve_mode(modes, worker_parser(modes, __doc__).parse_args())
