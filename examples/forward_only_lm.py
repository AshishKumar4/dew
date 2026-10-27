"""A language model trained without backpropagation, against the same model trained with it.

    python tools/noise_fineweb.py data/noise
    python examples/forward_only_lm.py --data data/noise --method perturbation --out runs/forward
    python examples/forward_only_lm.py --data data/noise --method backprop --out runs/backprop
    JAX_PLATFORMS=cpu python examples/forward_only_lm.py --smoke --out /tmp/forward-smoke

`tools/noise_fineweb.py` prepares Noise's fixed FineWeb data (qlabs-eng/noise): rows of 2,049 ids of its
BPE-4096 tokenizer, checked against its pinned hashes. This trains Dew's decoder at the shape of Dust's
37.7M-parameter GPT (8 layers of width 512, 8 heads of 64, 4,096 ids, rows of 2,048 predictions, windows
of half a row on three layers in four and the last layer whole, logits capped at 15, an untied head) on
the first `--tokens` of it, one epoch of 8 rows a step, and scores Noise's validation rows every
`--eval-every` steps. Four of Dust's block features are not Dew's (ledger 1852): its value embeddings, its
residual mix with the first layer's input, its parameter-free norms and its ungated ReLU^2 MLP, which a
gated SwiGLU of the same parameter count stands in for. So the model is Dust's shape, not Dust's model,
and the comparison is of the two learning rules on one model.

`--method perturbation` trains it by node perturbation (`dew.objectives.perturbation.NodePerturbation`),
Dust's estimator in Dew's general form: every Dense and Embed output jittered by `--draws` antithetic
pairs a step, each token credited by its own loss, with Adam at 1.5e-3, betas (0.7, 0.95), and a linear
warmdown over the last 30% of steps, Dust-with-Adam's settings in Noise. `--method backprop` trains it by
backpropagation with the same Adam and schedule at `--backprop-lr`. Each run writes `curves.json` to
`--out`: the training and validation losses by step, and the wall time of the training.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import optax
import tyro

from dew.data import Memory
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.sources import source
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.layer_plan import LayerKind
from dew.objectives.lm import LMObjective
from dew.objectives.perturbation import NodePerturbation
from dew.training import LocalTracker, Trainer


class ForwardLM(NodePerturbation, LMObjective):
    """`LMObjective` trained by node perturbation."""


@dataclasses.dataclass
class Config:
    data: Path = Path("data/noise")
    """`tools/noise_fineweb.py`'s output: its `train/` and `val/` stores."""
    out: Path = Path("runs/forward")
    method: str = "perturbation"
    """`perturbation` (no backpropagation) or `backprop`."""
    tokens: str = "10M"
    """The training rows read, in order: 10M, the Dust paper's first 4,880 rows, or 30M, Noise's 14,648."""
    draws: int = 64
    """Node perturbation's antithetic pairs a step."""
    sigma: float = 0.2
    lr: float = 1.5e-3
    backprop_lr: float = 3e-3
    eval_every: int = 50
    seed: int = 0
    smoke: bool = False
    """Four steps of a two-layer model of width 64 over random rows of 33 ids, both stores written here."""


def model(smoke: bool) -> CausalTransformer:
    """Dew's decoder at Dust's GPT shape, its gated MLP of the parameters Dust's 4x ReLU^2 one holds."""
    if smoke:
        return CausalTransformer(vocab_size=256, emb_features=64, num_layers=2, num_heads=2, head_dim=32,
                                 mlp_features=170, max_seq_len=32, final_logit_softcap=15.0,
                                 tie_embeddings=False, layer_types=("short", "long"),
                                 kinds={"short": LayerKind(window=16), "long": LayerKind()}, dtype="float32")
    return CausalTransformer(vocab_size=4096, emb_features=512, num_layers=8, num_heads=8, head_dim=64,
                             mlp_features=1365, max_seq_len=2048, final_logit_softcap=15.0,
                             tie_embeddings=False, layer_types=("short", "short", "short", "long") * 2,
                             kinds={"short": LayerKind(window=1024), "long": LayerKind()}, dtype="bfloat16")


def stores(config: Config) -> tuple[Path, Path]:
    """The training and validation stores: the prepared data, or random rows the smoke writes."""
    if not config.smoke:
        return config.data / "train", config.data / "val"
    rows = np.random.default_rng(config.seed).integers(0, 256, (48, 33)).astype(np.int32)
    for name, part in (("train", rows[:32]), ("val", rows[32:])):
        export(Memory([{"text": row} for row in part]), str(config.out / "smoke-data" / name),
               format="array_record")
    return config.out / "smoke-data" / "train", config.out / "smoke-data" / "val"


def main(config: Config) -> dict:
    train, val = stores(config)
    rows = 32 if config.smoke else {"10M": 4880, "30M": 14648}[config.tokens]
    steps = rows // 8
    decoder = model(config.smoke)
    seq_len = decoder.max_seq_len
    if config.method == "perturbation":
        objective = ForwardLM(decoder, seq_len=seq_len, draws=2 if config.smoke else config.draws,
                              sigma=config.sigma)
        rate = config.lr
    elif config.method == "backprop":
        objective, rate = LMObjective(decoder, seq_len=seq_len), config.backprop_lr
    else:
        raise ValueError(f"--method is perturbation or backprop, not {config.method!r}")
    warmdown = max(1, int(0.3 * steps))
    schedule = optax.join_schedules(
        [optax.constant_schedule(rate), optax.linear_schedule(rate, 0.0, warmdown)], [steps - warmdown])
    tracker = LocalTracker(config.out / "tracking")
    trainer = Trainer(objective, optax.adam(schedule, b1=0.7, b2=0.95), key=config.seed, tracker=tracker)
    began = time.perf_counter()
    trainer.fit(Pipeline(source(str(train))).take(rows).batch(8), steps=steps,
                validation=Pipeline(source(str(val))).batch(8), eval_every=min(config.eval_every, steps),
                log_every=1)
    seconds = time.perf_counter() - began
    curves = {"method": config.method, "steps": steps, "tokens": steps * 8 * seq_len, "seconds": seconds,
              "train": [(step, scalars["train/loss"]) for step, scalars in tracker.scalars()
                        if "train/loss" in scalars],
              "val": [(step, scalars["val/loss"]) for step, scalars in tracker.scalars()
                      if "val/loss" in scalars]}
    (config.out / "curves.json").write_text(json.dumps(curves, indent=1) + "\n")
    print(json.dumps({key: curves[key] for key in ("method", "steps", "tokens", "seconds")}
                     | {"val": curves["val"][-1:]}))
    return curves


if __name__ == "__main__":
    main(tyro.cli(Config))
