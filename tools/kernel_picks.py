"""Measure this GPU's kernel picks (`dew.nn.kernels.tuning`) for a language
model's training step at its real shapes.

The step is lowered from abstract state (`Trainer(lower_for=...)` on this
GPU's generation), so a model this GPU cannot hold still lowers, and each
Pallas launch the step holds is measured here and stored beside the
compilation cache (`dew.cache.kernel_picks_path`), where every later run on
a GPU of this model reads it. FSDP splits the parameters, not the rows a norm
or an expert matmul reads, so one device at a pool's per-device batch holds
the launches each of the pool's devices runs. Run it on a GPU it has to
itself (`gpu-run`, never `-n 0.5`, where nothing is measured):

    gpu-run python tools/kernel_picks.py --config Qwen/Qwen3-30B-A3B --batch 2 --seq 4096
    gpu-run python tools/kernel_picks.py --config path/to/checkpoint --batch 2 --seq 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import tokamax
from huggingface_hub import hf_hub_download

from dew import cache
from dew.interop.hf_decoders import translate_config
from dew.nn.backbones import CausalTransformer
from dew.nn.kernels import generation, tuning
from dew.objectives.lm import LMObjective
from dew.training import Trainer
from dew.training._lowering import Lowered


def model_of(arguments) -> CausalTransformer:
    """The decoder a checkpoint's config.json translates to, in `--dtype`: a
    local directory's, or a Hub repo's, of which nothing else is read."""
    local = Path(arguments.config) / "config.json"
    path = local if local.is_file() else hf_hub_download(arguments.config, "config.json",
                                                         revision=arguments.revision)
    config = json.loads(Path(path).read_text())
    fields = dict(translate_config(config.get("text_config", config)))
    return CausalTransformer(**{**fields, "dtype": jnp.dtype(arguments.dtype)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure this GPU's kernel picks for a decoder's step.")
    parser.add_argument("--config", required=True,
                        help="a decoder checkpoint's directory or Hub repo; only its config.json is read")
    parser.add_argument("--revision", help="the Hub repo's commit")
    parser.add_argument("--batch", type=int, required=True, help="sequences a device holds")
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    arguments = parser.parse_args()
    if not cache.measures_picks():
        raise SystemExit("this process shares its GPU, where no pick is measured: "
                         "run it under gpu-run without -n")
    model = model_of(arguments)
    trainer = Trainer(LMObjective(model, arguments.seq), optax.adamw(1e-4), key=jax.random.key(0),
                      lower_for=generation.device_generation())
    state = jax.eval_shape(trainer.initial_state, trainer.objective.initializer, trainer.key)
    batch = {"text": jax.ShapeDtypeStruct((arguments.batch, arguments.seq + 1), jnp.int32)}

    def step() -> jax.stages.Lowered:
        """The first step, which `compile` under `lower_for` lowers and raises."""
        try:
            trainer.compile(state, batch)
        except Lowered as lowered:
            return lowered.lowered
        raise AssertionError("a trainer under lower_for raises its lowered step")

    _, launches = tuning.tune(step, measure=True)
    launches = json.loads(launches)
    path = Path(cache.kernel_picks_path(jax.devices()[0].device_kind))
    # A store exists once a launch has been measured on this GPU model: a step with no Pallas launch
    # there (the dense 1.1B on an RTX PRO 6000 Blackwell) leaves none.
    stored = [] if not path.is_file() else list(tuning._read(str(path))[0].values())
    for ba, timings in stored:
        fastest = timings.prune()
        print(json.dumps({"kernel": type(ba.op).__name__, "key": str(dict(ba.autotuning_cache_key)),
                          "config": str(fastest.fastest_config),
                          "ms": min(data.median_evaluation_time_ms for data in fastest.values()
                                    if isinstance(data, tokamax.BenchmarkData))}))
    # The step's launches beside the store's: a launch none of whose configs ran is in the first count
    # alone, and takes its heuristic config in every run.
    print(json.dumps({"store": str(path), "launches": len(stored), "step_launches": len(launches)}))


if __name__ == "__main__":
    main()
