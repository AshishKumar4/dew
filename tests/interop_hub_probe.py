"""A load without weights from a Hub on the loopback (tests/local_hub.py), in a process of its own, as
`huggingface_hub` reads the Hub's address when it is imported: `python interop_hub_probe.py <request.json>`
loads the request's `repo` with `Pretrained.load(..., weights=False, **options)` (or the pipeline's,
`pipeline`; a published autoencoder's with its weights, `load_autoencoder(..., **options)`, where the
request's `call` is "autoencoder"; or only fetches its snapshot with its weights, `sources.snapshot`, where
it is "snapshot") and writes beside the request each variable's shape and dtype (`shapes`), a missing
source's refusal (`refused`), and where it names a local directory and a sample (`encode`), the loaded
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
    answer: dict[str, object] = {}
    if request.get("call") == "autoencoder":
        from dew.interop.autoencoders.pretrained import load_autoencoder

        answer["shapes"] = shapes(load_autoencoder(request["repo"], **request.get("options", {})).params)
    elif request.get("call") == "snapshot":
        from dew.interop.sources import snapshot

        snapshot(request["repo"], request.get("options", {}).get("revision"), weights=True)
    else:
        _load(request, request_path, answer)
    request_path.with_name("answer.json").write_text(json.dumps(answer))


def _load(request: dict, request_path: Path, answer: dict[str, object]) -> None:
    """`Pretrained.load` (or the pipeline's) of the request's `repo` without its weights, into `answer`."""
    loader = PretrainedPipeline if request.get("pipeline") else Pretrained
    try:
        loaded = loader.load(request["repo"], weights=False, **request.get("options", {}))
    except FileNotFoundError as missing:
        answer["refused"] = type(missing).__name__
        answer["reason"] = str(missing)
    else:
        answer["shapes"] = shapes(loaded.variables)
        if "encode" in request:
            local = PretrainedPipeline.load(request["encode"]["directory"], on_host=True,
                                            **request.get("options", {}))
            sample = np.load(request["encode"]["sample"])
            latent = loaded.autoencoder.encode_batch(local.variables["autoencoder"], sample)
            np.save(request_path.with_name("latent.npy"), np.asarray(latent))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
