#!/usr/bin/env python3
"""Write the tiny FID extractor under tests/fixtures/inception/tiny/.

The fixture is `dew.eval.inception.InceptionV3` at a sixteenth of every
channel width, its parameters drawn from a fixed seed and its running
statistics taken from one pass of noise: 128 pool3 features in a few hundred
KB, against 2048 in 90 MB, so the offline FID tests score with no download.
Its numbers are its own, and what they are asserted for is the ordering a
distance promises.

The drawn tree is written in the jax-fid layout and converted back by
`dew.interop.inception_fid`, so the committed file comes off the same
converter the published checkpoint does. Every draw is seeded, so rerunning
this script writes the same tensors; only the order of the header's metadata
keys is the safetensors writer's to choose.

    python tools/make_tiny_inception_weights.py
"""

import argparse
import hashlib
import pickle
import tempfile
from pathlib import Path

import jax
import numpy as np

from dew.eval.inception import InceptionV3
from dew.interop.inception_fid import convert, save, upstream_names
from dew.interop.safetensors_io import SEPARATOR

FIXTURE = (Path(__file__).resolve().parents[1]
           / "tests/fixtures/inception/tiny/inception_v3_fid.safetensors")
DIVISOR = 16
SEED = 20260921


def drawn(divisor: int, seed: int):
    """The narrow extractor's variables, every leaf drawn from the seed.

    The running statistics are drawn as well, rather than taken from a pass of
    noise: a pass is a floating-point reduction, and what it sums first depends
    on the machine it runs on, which is not something a committed fixture can
    depend on. Shifted means and spread variances keep the norms from being
    the identity the zeros and ones `init` leaves behind.
    """
    model = InceptionV3(channel_divisor=divisor)
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    variables = model.init(keys[0], jax.random.normal(keys[1], (1, 299, 299, 3)))
    statistics = np.random.default_rng(seed)

    def drawn_statistic(path, leaf):
        spread = path[-1].key == "var"
        draw = (statistics.uniform(0.5, 1.5, leaf.shape) if spread
                else statistics.normal(0.0, 0.25, leaf.shape))
        return draw.astype(np.float32)

    return {
        "params": variables["params"],
        "batch_stats": jax.tree_util.tree_map_with_path(drawn_statistic,
                                                        variables["batch_stats"]),
    }


def upstream_layout(variables) -> dict:
    """The variables tree under the names jax-fid stores, which is what the
    converter reads."""
    flat = {name: np.asarray(leaf) for name, leaf in
            ((SEPARATOR.join(entry.key for entry in path), leaf)
             for path, leaf in jax.tree_util.tree_flatten_with_path(variables)[0])}
    tree: dict = {}
    for name, upstream in upstream_names().items():
        node = tree
        for step in upstream[:-1]:
            node = node.setdefault(step, {})
        node[upstream[-1]] = flat[name]
    return tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURE)
    parser.add_argument("--pickle", type=Path, default=None,
                        help="keep the jax-fid layout written on the way, for inspection")
    args = parser.parse_args()

    tree = upstream_layout(drawn(DIVISOR, SEED))
    with tempfile.TemporaryDirectory() as scratch:
        source = args.pickle or Path(scratch) / "inception_v3_fid.pickle"
        source.write_bytes(pickle.dumps(tree, protocol=4))
        save(convert(source), args.out, DIVISOR)
    with open(args.out, "rb") as handle:
        print(f"{args.out} {hashlib.file_digest(handle, 'sha256').hexdigest()}")


if __name__ == "__main__":
    main()
