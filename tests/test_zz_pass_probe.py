"""Scratch probe (not for landing): the algebra pass's seconds on Qwen3.5 steps compiled ahead for an H100."""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import jax
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))


def _config(size):
    config = json.load(open(Path(__file__).parent / "fixtures/scratch/qwen3.5-9b.json"))["text_config"]
    if size == "4b":
        # Qwen3.5-4B's widths (its up projection 2560 x 9216, receipts FSTRAFFIC2), the 9B's layout otherwise.
        config = {**config, "hidden_size": 2560, "intermediate_size": 9216}
    return config


@pytest.mark.parametrize("on", [True], ids=["pass-on"])
@pytest.mark.parametrize("size", ["4b", "9b"])
def test_probe(size, on):
    from benchmark_cases import Case
    from benchmark_models import build_objective, global_batch, trainer_mesh
    import optax
    from dew import algebra
    from dew.interop.hf_decoders import translate_config
    from dew.nn.kernels.templates import report
    from dew.training import Layout, Trainer
    from dew.training.ahead import compile_ahead, compile_only_gpus

    case = Case("causal_transformer", dict(translate_config(_config(size))), dtype="bfloat16", batch_size=2,
                seq_len=2048)
    trainer = Trainer(build_objective(case, "auto"), optax.sgd(1e-4), key=jax.random.key(0),
                      **trainer_mesh(case.mesh), layout=Layout(min_shard=case.fsdp_min_param_size),
                      accumulation=case.accumulation, split=1, checkpoints=None, tracker=None)
    trainer.device_mesh = trainer.mesh.build(compile_only_gpus("sm90"))
    profile = cProfile.Profile()
    began = time.perf_counter()
    with algebra.passes(on=on, target="sm90"):
        profile.enable()
        ahead = compile_ahead(trainer, global_batch(case), 80 * 2**30)
        profile.disable()
    total = time.perf_counter() - began
    stats = pstats.Stats(profile)
    def cumulative(module_part, name):
        return sum(v[3] for (f, _, n), v in stats.stats.items() if module_part in f and n == name)
    passes = report.passes()
    rewrites = report.rewrites()
    row = {"size": size, "on": on, "total_s": round(total, 1), "planned_s": round(ahead.planned_seconds, 1),
           "passes": len(passes), "pass_s": round(sum(passes), 2), "max_pass_s": round(max(passes, default=0), 2),
           "windows": len(rewrites), "reused": sum(bool(getattr(r, "reused", False)) for r in rewrites),
           "window_s": round(sum(r.seconds for r in rewrites), 3),
           "cum_rewritten": round(cumulative("algebra/program.py", "_rewritten"), 2),
           "cum_windowed": round(cumulative("algebra/program.py", "_windowed"), 2),
           "cum_within": round(cumulative("algebra/program.py", "_within"), 2),
           "cum_region": round(cumulative("algebra/program.py", "_region"), 2),
           "cum_trace": round(cumulative("jax/_src/interpreters/partial_eval.py", "trace_to_jaxpr_dynamic"), 2),
           "cum_lower": round(cumulative("jax/_src/interpreters/pxla.py", "lower_sharding_computation"), 2),
           "cum_compile": round(cumulative("jax/_src/compiler.py", "backend_compile_and_load")
                                + cumulative("jax/_src/compiler.py", "backend_compile"), 2)}
    top = sorted(((v[3], f"{Path(f).name}:{n}") for (f, _, n), v in stats.stats.items() if "dew/algebra" in f),
                 reverse=True)[:12]
    raise AssertionError("PROBE " + json.dumps(row) + "\nTOP " + json.dumps([(round(s, 2), n) for s, n in top]))
