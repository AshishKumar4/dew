"""One process of a pool settling a step's kernel configs, driven by tests/test_tuning.py.

Each process keeps its picks in a store of its own, which starts as
`--picks`'s `process<index>.json` where the test wrote one and empty
otherwise. Every process tunes the same program, the step at each token count
of `--rows`, as on an sm80 GPU (`generation.assumed`), as a pool compiling one
program does, and records each launch of the program `tune` returns.
"""

import os
from pathlib import Path

import jax
from process_support import serve_mode, worker_parser
from test_tuning import launched, operands, step

from dew import cache
from dew.files import write_atomically
from dew.nn.kernels import generation, tuning


def mode_settle(args) -> list:
    jax.config.update("jax_compilation_cache_dir", str(args.cache / f"process{args.process_id}"))
    held = args.picks / f"process{args.process_id}.json"
    if held.exists():
        path = cache.kernel_picks_path(jax.devices()[0].device_kind)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_atomically(path, held.read_text())
    rows = [int(count) for count in args.rows.split(",")]

    def program():
        return [step()(*operands(count)) for count in rows]

    with generation.assumed("sm80"):
        tuned, _ = tuning.tune(lambda: jax.make_jaxpr(program)())
    return launched(tuned)


MODES = {"settle": mode_settle}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--cache", type=Path, required=True, help="each process's own cache's directory")
    parser.add_argument("--picks", type=Path, required=True, help="the stores processes start with")
    parser.add_argument("--rows", required=True, help="the step's token counts, comma-separated")
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
