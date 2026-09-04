"""Same-seed per-step training losses, this branch against main.

Runs the small benchmark preset's language model for a fixed number of steps
from one seed and writes every step's loss as JSON. Run it twice, once with
PYTHONPATH pointing at this worktree and once at the main checkout, then
compare: the chunked head only ships if the two agree to the tolerance
CONTRIBUTING states for a replaced loss, 1e-5 relative.

Usage:
  PYTHONPATH=<checkout>/src python tools/lm_step_parity.py --steps 20 \
      --out /tmp/lm-head/losses-<name>.json
"""
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.objectives.lm import LMObjective, TEXT_KEY
from dew import models  # naming a registry fills it
from dew.registry import with_precision
from dew.training import MeshSpec, Trainer

# tools/benchmark_step.py's small causal_transformer preset.
CONFIG = dict(vocab_size=50304, emb_features=768, num_layers=3, num_heads=12,
              mlp_features=4 * 768, max_seq_len=512)
BATCH, SEQ = 16, 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", default=None,
                        help="model.precision: unset is XLA's default, which "
                             "uses TF32 for fp32 matmuls on this card")
    args = parser.parse_args()

    config = with_precision("causal_transformer", dict(CONFIG),
                            dtype="bfloat16", attention_impl="reference")
    if args.precision is not None:
        config["precision"] = args.precision
    model = models.build("causal_transformer", **config)
    trainer = Trainer(LMObjective(model, SEQ), optax.adam(1e-4),
                      key=jax.random.key(0), mesh=MeshSpec(fsdp=1),
                      checkpoints=None, tracker=None)
    abstract = jax.eval_shape(trainer.initial_state)
    state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
    scale = None

    # The benchmark's fixed batch, byte for byte the same on both sides.
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, CONFIG["vocab_size"],
                                size=(BATCH, SEQ + 1)).astype(np.int32)
    batch = {TEXT_KEY: jnp.asarray(tokens)}

    step = trainer.compile(state, batch)
    losses, accuracies = [], []
    for _ in range(args.steps):
        state, scale, loss, metrics, finite = step(state, scale, batch)
        losses.append(float(loss))
        accuracies.append(float(metrics["token_accuracy"]))
        assert bool(finite), "the loss went non-finite"

    with open(args.out, "w") as handle:
        json.dump({"losses": losses, "token_accuracy": accuracies}, handle, indent=2)
    print(json.dumps({"first": losses[0], "last": losses[-1],
                      "accuracy_last": accuracies[-1]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
