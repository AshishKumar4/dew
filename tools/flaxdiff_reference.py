#!/usr/bin/env python3
"""Write the FlaxDiff fixture tests/test_flaxdiff.py checks Dew's SimpleUDiT against.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of,
and `dew.interop.flaxdiff` loads its checkpoints. The reference is FlaxDiff's
own `SimpleUDiT` at commit 3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a, the
commit the text-to-image run this loader was written for trained with
(flaxdiff 0.2.8). It runs here, on a model small enough to commit, and the
fixture holds everything a parity test needs without FlaxDiff installed:

- `config.json`: the model's entry of the run config FlaxDiff's trainer logs,
  the record `dew.interop.flaxdiff.simple_udit_fields` translates;
- `reference.npz`: the parameter tree under FlaxDiff's own names, the Fourier
  table FlaxDiff's `FourierEmbedding` draws (`jax.random.normal` at
  PRNGKey(42), which jax 0.5 and later give the same), the inputs, the text
  mask a Dew caller hands the model beside them, and the fp32 output.

Every weight is redrawn from N(0, 0.05^2), the zero-initialized ones
included, so every path of the block reaches the output.

    git clone https://github.com/AshishKumar4/FlaxDiff /tmp/flaxdiff
    git -C /tmp/flaxdiff checkout 3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a
    python tools/flaxdiff_reference.py --flaxdiff-path /tmp/flaxdiff \\
        --out tests/fixtures/flaxdiff
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

COMMIT = "3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a"

# The model entry of a FlaxDiff run config, at a size a fixture can hold.
MODEL = {"output_channels": 4, "patch_size": 2, "emb_features": 32, "num_layers": 4,
         "num_heads": 4, "mlp_ratio": 4, "dropout_rate": 0.1, "norm_groups": 0,
         "use_hilbert": False, "use_flash_attention": False,
         "activation": "jax._src.nn.functions.silu", "dtype": "jax.numpy.float32",
         "precision": "DEFAULT"}
BATCH, SIZE, TEXT_TOKENS, TEXT_WIDTH, REAL_TOKENS = 2, 8, 7, 16, (3, 5)


def arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flaxdiff-path", required=True,
                        help=f"a FlaxDiff checkout at {COMMIT}, the directory holding flaxdiff/")
    parser.add_argument("--out", default="tests/fixtures/flaxdiff")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = arguments(argv)
    sys.path.insert(0, args.flaxdiff_path)
    from flaxdiff.models.common import FourierEmbedding
    from flaxdiff.models.simple_vit import SimpleUDiT

    model = SimpleUDiT(**{key: value for key, value in MODEL.items()
                          if key not in ("activation", "dtype", "precision")},
                       dtype=jnp.float32)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, SIZE, SIZE, MODEL["output_channels"]), dtype=np.float32)
    # log(sigma) / 4 over the Karras grid's range, which is what EDM hands the model
    temb = np.array([-1.3, 0.9], np.float32)
    text = rng.standard_normal((BATCH, TEXT_TOKENS, TEXT_WIDTH), dtype=np.float32)
    mask = (np.arange(TEXT_TOKENS)[None, :] < np.array(REAL_TOKENS)[:, None]).astype(np.int32)

    shapes = jax.eval_shape(model.init, jax.random.key(0), x, temb, text)
    params = jax.tree.map(
        lambda leaf: (0.05 * rng.standard_normal(leaf.shape)).astype(np.float32), shapes)
    with jax.default_matmul_precision("highest"):
        output = np.asarray(model.apply(params, x, temb, text))
    table = np.asarray(FourierEmbedding(features=MODEL["emb_features"]).bind({}).freqs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"flaxdiff_commit": COMMIT, "architecture": "simple_udit", "model": MODEL}, indent=1) + "\n")
    arrays = {f"params/{name}": np.asarray(leaf)
              for name, leaf in flatten_dict(params["params"], sep="/").items()}
    np.savez(out / "reference.npz", **arrays, fourier_table=table, x=x, temb=temb, text=text,
             text_mask=mask, output=output)


if __name__ == "__main__":
    main()
