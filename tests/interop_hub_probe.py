"""A load without weights from a Hub on the loopback (tests/local_hub.py), in a process of its own, as
`huggingface_hub` reads the Hub's address when it is imported: `python interop_hub_probe.py <request.json>`
loads the request's `repo` with `Pretrained.load(..., weights=False, **options)` (or the pipeline's,
`pipeline`) and writes beside the request each variable's shape and dtype (`shapes`), a missing source's
refusal (`refused`), and where it names a local directory and a sample (`encode`), the loaded
autoencoder's latent of the sample under that directory's autoencoder parameters (`latent.npy`), which
the load without weights holds none of."""

import json
import sys
from pathlib import Path

import jax
import numpy as np

from dew.interop import Pretrained, PretrainedPipeline


def shapes(variables) -> object:
    """Each leaf's shape and dtype, as JSON holds them."""
    return jax.tree.map(lambda leaf: [list(leaf.shape), str(leaf.dtype)], variables)


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    loader = PretrainedPipeline if request.get("pipeline") else Pretrained
    answer: dict[str, object] = {}
    try:
        loaded = loader.load(request["repo"], weights=False, **request.get("options", {}))
    except FileNotFoundError as missing:
        answer["refused"] = type(missing).__name__
    else:
        answer["shapes"] = shapes(loaded.variables)
        if "encode" in request:
            local = PretrainedPipeline.load(request["encode"]["directory"], on_host=True,
                                            **request.get("options", {}))
            sample = np.load(request["encode"]["sample"])
            latent = loaded.autoencoder.encode_batch(local.variables["autoencoder"], sample)
            np.save(request_path.with_name("latent.npy"), np.asarray(latent))
    request_path.with_name("answer.json").write_text(json.dumps(answer))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
