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

    gpu-run python tools/kernel_picks.py --hub Qwen/Qwen3-30B-A3B --batch 2 --seq 4096
    gpu-run python tools/kernel_picks.py --fields model1b.json --batch 2 --seq 2048
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
    """The decoder a Hub repo's config.json translates to, or the fields a file holds, in `--dtype`."""
    if arguments.hub:
        with open(hf_hub_download(arguments.hub, "config.json")) as file:
            config = json.load(file)
        fields = dict(translate_config(config.get("text_config", config)))
    else:
        fields = json.loads(Path(arguments.fields).read_text())
    return CausalTransformer(**{**fields, "dtype": jnp.dtype(arguments.dtype)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hub", help="a Hub decoder repo, of which only config.json is read")
    source.add_argument("--fields", help="a JSON file of CausalTransformer fields")
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

    tuning.tune(step)
    kind = jax.devices()[0].device_kind
    path = cache.kernel_picks_path(kind)
    with open(path) as file:
        stored = tokamax.AutotuningResult.load(file)
    for ba, timings in stored.data:
        fastest = timings.prune()
        print(json.dumps({"kernel": type(ba.op).__name__, "key": str(dict(ba.autotuning_cache_key)),
                          "config": str(fastest.fastest_config),
                          "ms": min(data.median_evaluation_time_ms for data in fastest.values())}))
    print(json.dumps({"store": path, "launches": len(stored.data)}))


if __name__ == "__main__":
    main()
