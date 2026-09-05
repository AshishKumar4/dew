"""Record the per-step losses of the small language model preset from one seed.

Runs the small benchmark preset's language model for a fixed number of steps
from one seed on one fixed batch and writes every step's loss and token
accuracy as JSON. Run it twice, once with PYTHONPATH at each checkout, and
compare the two files: a replaced loss ships only if they agree to fp32
tolerance, as CONTRIBUTING asks of any replaced computation.

Usage:
  PYTHONPATH=<checkout>/src python tools/lm_step_parity.py --steps 20 \\
      --out /tmp/lm-head/losses-<name>.json

--precision sets `model.precision`; unset is XLA's default, which uses TF32
for fp32 matmuls on an Ampere or later card.
"""

import argparse
import json
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.objectives.lm import LMObjective, TEXT_KEY
from dew import models  # naming a registry fills it
from dew.registry import with_precision
from dew.training import MeshSpec, Trainer

# tools/benchmark_step.py's small causal_transformer preset.
CONFIG: dict[str, int] = dict(vocab_size=50304, emb_features=768, num_layers=3, num_heads=12,
                              mlp_features=4 * 768, max_seq_len=512)
BATCH, SEQ = 16, 512


@dataclass(frozen=True)
class Record:
    """Every step's loss and token accuracy, in step order."""

    losses: list[float]
    token_accuracy: list[float]


def run(config: dict[str, int], batch: int, seq: int, steps: int,
        precision: str | None = None) -> Record:
    fields = with_precision("causal_transformer", config,
                            dtype="bfloat16", attention_impl="reference")
    if precision is not None:
        fields["precision"] = precision
    model = models.build("causal_transformer", **fields)
    trainer = Trainer(LMObjective(model, seq), optax.adam(1e-4),
                      key=jax.random.key(0), mesh=MeshSpec(fsdp=1),
                      checkpoints=None, tracker=None)
    abstract = jax.eval_shape(trainer.initial_state)
    state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
    scale = None

    # The benchmark's fixed batch, byte for byte the same on both sides.
    generator = np.random.default_rng(0)
    tokens = generator.integers(0, config["vocab_size"], size=(batch, seq + 1)).astype(np.int32)
    data = {TEXT_KEY: jnp.asarray(tokens)}

    step = trainer.compile(state, data)
    record = Record([], [])
    for index in range(steps):
        state, scale, loss, metrics, finite = step(state, scale, data)
        if not bool(finite):
            raise RuntimeError(f"the loss went non-finite at step {index}")
        record.losses.append(float(loss))
        record.token_accuracy.append(float(metrics["token_accuracy"]))
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precision", default=None,
                        help="model.precision; unset is XLA's default")
    args = parser.parse_args(argv)

    record = run(CONFIG, BATCH, SEQ, args.steps, args.precision)
    with open(args.out, "w") as handle:
        json.dump(asdict(record), handle, indent=2)
    print(json.dumps({"first": record.losses[0], "last": record.losses[-1],
                      "accuracy_last": record.token_accuracy[-1]}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
