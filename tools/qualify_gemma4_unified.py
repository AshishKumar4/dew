"""Size one live Unified reference workload without evaluating Dew's candidate arm.

    PYTHONPATH=src python tools/qualify_gemma4_unified.py forward --output qualification.json

Run in the project's reference environment. The tests own the workloads; reference_error owns
the sizing rule. --start and --most bound one reference-only sizing job.

The backward recording, BACKWARD_WINDOW draws a seed, is made in shards of seeds where one process is
too slow for it, each run as one process runs it, then joined:

    PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --shard 0:32 --output 0.npz
    PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --merge *.npz \
        --output tests/fixtures/hf/gemma4-unified-native-tiny/backward.npz
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = "tests/fixtures/hf/gemma4-unified-native-tiny"
"""Where the Unified fixture and its sites' recordings are, from the checkout."""


def workload(site, directory, device):
    from test_gemma4_unified import (
        FIXTURE,
        cached_functions,
        equal_geometry_source,
        forward_functions,
        gradient_functions,
        optional_source,
        source_pair,
        unified_draw,
    )

    source = source_pair(FIXTURE, "bfloat16" if site == "bfloat16" else "float32")
    if site in ("text", "vision", "audio", "equal"):
        source, draw = (equal_geometry_source(source, directory) if site == "equal"
                        else optional_source(source, directory, site))
    else:
        def draw(seed):
            return unified_draw(source[1].config, seed, side=4 if site == "long" else 2)
    source[1].to(device)
    source[2].to(device)
    if site in ("backward", "cached"):
        functions = (gradient_functions if site == "backward" else cached_functions)(source)
        return (*functions[1:], source[0].source)
    _, reference, truth = forward_functions(source)
    return reference, truth, lambda seed: (draw(seed),), source[0].source


def backward_recording(device, seeds=None):
    """The backward site's summarized recording, BACKWARD_WINDOW draws a seed (`windows`): whole
    (`record_case`), or the shard of `seeds` a merge joins (`record_shard`)."""
    import jax
    from recorded_references import record_case, record_shard
    from reference_error import assert_computes_the_oracle, chain_roundings, over_windows, widened, windows
    from test_gemma4_unified import BACKWARD_WINDOW, FIXTURE, gradient_functions, source_pair

    from dew.interop import Pretrained

    source = source_pair(FIXTURE)
    source[1].to(device)
    source[2].to(device)
    _, reference, truth, drawn = gradient_functions(source)
    with jax.enable_x64(new_val=True):
        native = Pretrained.load(FIXTURE, dtype="float64", param_dtype="float64", attention_impl="reference")
        native_truth = gradient_functions((native, source[1], source[2]))[0]
        roundings = chain_roundings(jax.make_jaxpr(native_truth)(widened(drawn(0)[0])))

    started, evaluated = time.monotonic(), 0

    def evaluate(inputs):
        nonlocal evaluated
        observed = over_windows(reference)(inputs)
        with jax.enable_x64(new_val=True):
            exact = over_windows(truth)(widened(inputs))
            twin = over_windows(native_truth)(widened(inputs))
        for name, value in exact.items():
            assert_computes_the_oracle(twin[name], value, f"Unified float64 backward {name}",
                                       roundings=roundings)
        evaluated += 1
        if evaluated % 16 == 0:
            print(f"backward: {evaluated} windows in {time.monotonic() - started:.0f} s", flush=True)
        return observed, exact

    windowed = windows(lambda seed: drawn(seed)[0], BACKWARD_WINDOW)
    if seeds is not None:
        return record_shard("backward", windowed, evaluate, seeds), FIXTURE
    return record_case("backward", windowed, evaluate, store_inputs=False, summary=True), FIXTURE


ENVIRONMENT = "environment"
"""A backward shard's key for the environment that recorded it (`environment`), as JSON."""


def environment(device) -> dict:
    """What a recording's provenance names of the process that makes it: the device, the packages, and
    the generator's commit."""
    import torch
    from reference_source import git_head

    return {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "revision": git_head(ROOT),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("jax", "torch", "transformers", "numpy")}}


def record_site(site, output, device, *, shards=()):
    """`site`'s recording at `output`, with its provenance: the backward site's joined from `shards`
    (`merge_shards`) where they are given, its provenance the environment every shard was recorded in,
    which they must agree on."""
    import jax
    import numpy as np
    from recorded_references import merge_shards, provenance, record_case, save
    from reference_error import widened

    with tempfile.TemporaryDirectory() as directory:
        if site == "backward" and shards:
            held, environments = [], set()
            for path in shards:
                with np.load(path, allow_pickle=False) as shard:
                    held.append({key: value for key, value in shard.items() if key != ENVIRONMENT})
                    environments.add(str(shard[ENVIRONMENT]))
            if len(environments) != 1:
                raise ValueError(f"the shards were recorded in {len(environments)} environments: "
                                 f"{sorted(environments)}")
            made = json.loads(environments.pop())
            stored, source = merge_shards("backward", held), ROOT / RECORDINGS
        elif site == "backward":
            stored, source = backward_recording(device)
        else:
            reference, truth, draw, source = workload(site, Path(directory), device)

            def drawn(seed):
                arguments = draw(seed)
                return ({**arguments[0], "continuation": arguments[1]} if site == "cached"
                        else arguments[0])

            def evaluate(inputs):
                arguments = (({name: value for name, value in inputs.items() if name != "continuation"},
                              inputs["continuation"]) if site == "cached" else (inputs,))
                observed = reference(*arguments)
                with jax.enable_x64(new_val=True):
                    exact = truth(*widened(arguments))
                return observed, exact

            stored = record_case(site, drawn, evaluate, store_inputs=False)
        if not (site == "backward" and shards):
            made = environment(device)
        stored["precision"] = "ieee"
        stored["provenance"] = provenance(
            "transformers Gemma4UnifiedForConditionalGeneration", made["packages"]["transformers"],
            made["device"],
            (f"PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --device {device} "
             f"--shard START:STOP --output SHARD.npz, over seeds 0 to {len(stored['backward.input_digest'])} "
             f"in shards, then --merge SHARD.npz... --output {RECORDINGS}/backward.npz" if shards else
             f"PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py {site} --device {device} "
             f"--record --output {RECORDINGS}/{site}.npz"),
            generator_revision=made["revision"],
            model_sha256=hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
            packages=made["packages"])
        save(output, stored, reference="transformers Gemma4 Unified", license="Apache-2.0")


def main():
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    import numpy as np
    import torch
    from reference_error import FEWEST_SEEDS, MOST_SEEDS, as_given, seeds_for
    from reference_source import git_head

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=("forward", "long", "bfloat16", "backward", "cached",
                                         "text", "vision", "audio", "equal"))
    parser.add_argument("--start", type=int, default=FEWEST_SEEDS)
    parser.add_argument("--most", type=int, default=MOST_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--shard", help="the backward site's seeds START:STOP alone, recorded to --output "
                                        "for a --merge")
    parser.add_argument("--merge", type=Path, nargs="+", default=(),
                        help="the backward site's shards, joined into its recording at --output")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    if args.shard is not None:
        start, stop = (int(bound) for bound in args.shard.split(":"))
        shard, _ = backward_recording(args.device, range(start, stop))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **shard, **{ENVIRONMENT: np.asarray(json.dumps(environment(args.device)))})
        return
    if args.record or args.merge:
        record_site(args.site, args.output, args.device, shards=args.merge)
        return
    with tempfile.TemporaryDirectory() as directory:
        reference, truth, draw, source = workload(args.site, Path(directory), args.device)

        @functools.lru_cache(maxsize=1)
        def observed(seed):
            # The reference-vs-itself control asks for the same output twice in immediate succession.
            return reference(*draw(seed))

        count = seeds_for(as_given(lambda seed: observed(int(seed))),
                          lambda seed: (np.asarray(seed, np.int64),), args.site,
                          start=args.start, most=args.most, truth_fn=lambda seed: truth(*draw(int(seed))))
        weights = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
        config = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    record = {"site": args.site, "seeds": count, "revision": git_head(ROOT),
              "model_sha256": weights, "config_sha256": config,
              "method": "reference_vs_reference", "start": args.start, "most": args.most,
              "device": args.device,
              "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "precision": "ieee",
              "packages": {name: importlib.metadata.version(name)
                           for name in ("jax", "torch", "transformers", "numpy")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.site}: {count} declared seeds", flush=True)


if __name__ == "__main__":
    main()
