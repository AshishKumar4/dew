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
from dew.registry import apply_precision_policy, build_model
from dew.training import ObjectiveTrainer

# tools/benchmark_step.py's small causal_transformer preset.
CONFIG = dict(vocab_size=50304, emb_features=768, num_layers=3, num_heads=12,
              mlp_ratio=4, max_seq_len=512)
BATCH, SEQ = 16, 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-dir", default="/tmp/lm-head/parity-ckpt")
    parser.add_argument("--precision", default=None,
                        help="model.precision: unset is XLA's default, which "
                             "uses TF32 for fp32 matmuls on this card")
    args = parser.parse_args()

    config = apply_precision_policy("causal_transformer", dict(CONFIG),
                                    dtype="bfloat16", attention_impl="reference")
    if args.precision is not None:
        config["precision"] = args.precision
    model = build_model("causal_transformer", config)
    objective = LMObjective(model, SEQ, vocab_size=CONFIG["vocab_size"])
    trainer = ObjectiveTrainer(
        model=model,
        optimizer=optax.adam(1e-4),
        rngs=jax.random.PRNGKey(0),
        input_config=None,
        objective=objective,
        name="lm-head-parity",
        wandb_config=None,
        distributed_training=True,
        fsdp_size=1,
        checkpoint_base_path=args.checkpoint_dir,
    )
    trainer.global_batch_size = BATCH

    step = trainer._define_train_step(batch_size=BATCH)
    state, rng = trainer.state, trainer.rngstate

    # The benchmark's fixed batch, byte for byte the same on both sides.
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, CONFIG["vocab_size"],
                                size=(BATCH, SEQ + 1)).astype(np.int32)
    batch = {TEXT_KEY: jnp.asarray(tokens)}

    losses, accuracies = [], []
    for _ in range(args.steps):
        state, loss, aux, rng, finite = step(state, rng, batch)
        losses.append(float(loss))
        accuracies.append(float(aux["token_accuracy"]))
        assert bool(finite), "the loss went non-finite"

    with open(args.out, "w") as handle:
        json.dump({"losses": losses, "token_accuracy": accuracies}, handle, indent=2)
    print(json.dumps({"first": losses[0], "last": losses[-1],
                      "accuracy_last": accuracies[-1]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
