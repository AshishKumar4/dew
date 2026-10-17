"""One process of a pool settling a step's kernel configs, driven by tests/test_tuning.py.

Each process keeps its picks in a store of its own; the one `--holder` names
starts with the store `--picks` holds, the others with none. Every process
tunes the same 15-token step as on an sm80 GPU (`generation.assumed`), as a
pool compiling one program does, and records the grid and warps of each launch
of the step it then traces for that GPU.
"""

import os
from pathlib import Path

import jax
from process_support import serve_mode, worker_parser
from test_tuning import launches, operands, step

from dew import cache
from dew.files import write_atomically
from dew.nn.kernels import generation, tuning


def mode_settle(args) -> dict:
    jax.config.update("jax_compilation_cache_dir", str(args.cache / f"process{args.process_id}"))
    if args.process_id == args.holder:
        path = cache.kernel_picks_path(jax.devices()[0].device_kind)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_atomically(path, args.picks.read_text())
    with generation.assumed("sm80"):
        tuning.tune(lambda: jax.make_jaxpr(step())(*operands(15)))
        return launches(step(), *operands(15))


MODES = {"settle": mode_settle}


def main(argv=None) -> None:
    parser = worker_parser(MODES, __doc__)
    parser.add_argument("--cache", type=Path, required=True, help="directory of each process's own cache")
    parser.add_argument("--picks", type=Path, required=True, help="the store the holder starts with")
    parser.add_argument("--holder", type=int, required=True, help="the process that starts with --picks")
    serve_mode(MODES, parser.parse_args(argv))


if __name__ == "__main__":
    main()
