"""Size one live Unified reference workload without evaluating Dew's candidate arm.

    PYTHONPATH=src python tools/qualify_gemma4_unified.py forward --output qualification.json

Run in the project's reference environment. The tests own the workloads; reference_error owns
the sizing rule. --start and --most bound one reference-only sizing job.

The backward recording, BACKWARD_WINDOW draws a seed, is made in shards of seeds where one process is
too slow for it, each run as one process runs it, its pilots' shards (seeds from SIZING_FROM) holding Dew's
errors on their draws beside the reference's, then joined, the count sized paired on the pilots (its errors
are heavy-tailed alike in both arms, so the reference compared with itself never resolves):

    PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --shard 0:32 --output 0.npz
    PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --shard 1048576:1048608 \
        --output pilot-0.npz
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
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = "tests/fixtures/hf/gemma4-unified-native-tiny"
"""Where the Unified fixture and its sites' recordings are, from the checkout."""

type Arrays = dict[str, np.ndarray]
"""A site's arrays by name: a draw's inputs, a reference's or a truth's outputs."""

type Arguments = tuple[Arrays] | tuple[Arrays, np.ndarray]
"""A draw's arguments: its inputs, and the cached site's continuation after them."""


def workload(site, directory, device) -> tuple[Callable[..., Arrays], Callable[..., Arrays],
                                                Callable[[int], Arguments], Path]:
    """`site`'s reference and float64 truth, each called with a draw's arguments, its draw of a seed's
    arguments (the cached site's inputs and continuation, every other one input), and its source's
    directory."""
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
        reference, truth, drawn = functions[1:]
        return reference, truth, drawn, _directory(source[0].source)
    _, reference, truth = forward_functions(source)
    return reference, truth, lambda seed: (draw(seed),), _directory(source[0].source)


def _directory(source: Path | None) -> Path:
    """A loaded source's directory, which every site's source has."""
    if source is None:
        raise ValueError("the Unified fixture loads from no directory")
    return source


def backward_recording(device, seeds=None):
    """The backward site's summarized recording, BACKWARD_WINDOW draws a seed (`windows`), its count sized
    paired with Dew's arm run as check 1 runs it (`summed_as_run`): whole (`record_case(..., dew=)`, which
    runs Dew on its sizing draws), or the shard of `seeds` a merge joins (`record_shard`), a pilot shard's
    (seeds from SIZING_FROM) holding Dew's errors on its draws, which the merge sizes the count by and keeps
    none of."""
    import jax
    from recorded_references import record_case, record_shard
    from reference_error import (
        SIZING_FROM,
        assert_computes_the_oracle,
        chain_roundings,
        over_windows,
        summed_as_run,
        widened,
        windows,
    )
    from test_gemma4_unified import BACKWARD_WINDOW, FIXTURE, gradient_functions, source_pair

    from dew.interop import Pretrained

    source = source_pair(FIXTURE)
    source[1].to(device)
    source[2].to(device)
    ours, reference, truth, drawn = gradient_functions(source)
    with jax.enable_x64(new_val=True):
        native = Pretrained.load(FIXTURE, dtype="float64", param_dtype="float64")
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

    def dew(inputs):
        with summed_as_run():
            return over_windows(ours)(inputs)

    if seeds is not None:
        pilots = seeds.start >= SIZING_FROM
        return record_shard("backward", windowed, evaluate, seeds, dew=dew if pilots else None), FIXTURE
    return record_case("backward", windowed, evaluate, store_inputs=False, summary=True, dew=dew), FIXTURE


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


def recorded_command(site, device) -> str:
    """The command that records `site` whole on `device`, as its provenance names it."""
    return (f"PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py {site} --device {device} "
            f"--record --output {RECORDINGS}/{site}.npz")


def record_site(site, output, device, *, shards=()):
    """`site`'s recording at `output`, with its provenance: the backward site's joined from `shards`
    (`merge_shards`) where they are given, its provenance the environment every shard was recorded in,
    which they must agree on."""
    import jax
    import numpy as np
    from recorded_references import merge_shards, provenance, record_case, save
    from reference_error import SIZING_FROM, widened

    with tempfile.TemporaryDirectory() as directory:
        if site == "backward" and shards:
            held, environments = [], set[str]()
            for path in shards:
                with np.load(path, allow_pickle=False) as shard:
                    held.append({key: value for key, value in shard.items() if key != ENVIRONMENT})
                    environments.add(str(shard[ENVIRONMENT]))
            if len(environments) != 1:
                raise ValueError(f"the shards were recorded in {len(environments)} environments: "
                                 f"{sorted(environments)}")
            made = json.loads(environments.pop())
            stored, source = merge_shards("backward", held), ROOT / RECORDINGS
            # Pilots, where any shard held Dew's errors on their draws (`record_shard`'s `dew`).
            pilots = any(key.startswith("backward.dew_mse.") for shard in held for key in shard)
            command = (f"PYTHONPATH=src:tests:tools python tools/qualify_gemma4_unified.py backward --device "
                       f"{device} --shard START:STOP --output SHARD.npz, over seeds 0 to "
                       f"{len(stored['backward.input_digest'])} "
                       + (f"and pilots from {SIZING_FROM} (Dew's errors on their draws beside the "
                          "reference's) " if pilots else "")
                       + f"in shards, then --merge SHARD.npz... --output {RECORDINGS}/backward.npz")
        elif site == "backward":
            stored, source = backward_recording(device)
            made, command = environment(device), recorded_command(site, device)
        else:
            reference, truth, draw, source = workload(site, Path(directory), device)

            def drawn(seed):
                match draw(seed):
                    case (inputs, continuation):
                        return {**inputs, "continuation": continuation}
                    case (inputs,):
                        return inputs

            def evaluate(inputs):
                arguments = (({name: value for name, value in inputs.items() if name != "continuation"},
                              inputs["continuation"]) if site == "cached" else (inputs,))
                observed = reference(*arguments)
                with jax.enable_x64(new_val=True):
                    exact = truth(*widened(arguments))
                return observed, exact

            # Its evidence alone (`summary`): the test holds Dew's own float64 run to the recorded truth's
            # sketch at each draw (test_gemma4_unified.py's `rows_forward`, `cached_run`), where the whole
            # outputs took 94 MB of the fetched store over seven sites. The bfloat16 source's outputs stay
            # whole: Dew's float64 forward of its bfloat16 load is not the reference's float64 run.
            stored = record_case(site, drawn, evaluate, store_inputs=False, summary=site != "bfloat16")
            made, command = environment(device), recorded_command(site, device)
        stored["precision"] = np.asarray("ieee")
        stored["provenance"] = provenance(
            "transformers Gemma4UnifiedForConditionalGeneration", made["packages"]["transformers"],
            made["device"], command,
            generator_revision=made["revision"],
            model_sha256=hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
            packages=made["packages"])
        save(output, stored, reference="transformers Gemma4 Unified", license="Apache-2.0")


def main():
    sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
    import numpy as np
    import torch
    from reference_error import FEWEST_SEEDS, as_given, seeds_for
    from reference_source import git_head

    from dew.parity import MOST_SEEDS

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
                        help="the backward site's shards, its pilots' from SIZING_FROM among them, joined "
                             "into its recording at --output")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    if args.shard is not None:
        start, stop = (int(bound) for bound in args.shard.split(":"))
        shard, _ = backward_recording(args.device, range(start, stop))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, allow_pickle=False, **shard,
                 **{ENVIRONMENT: np.asarray(json.dumps(environment(args.device)))})
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
                          start=args.start, most=args.most,
                          truth_fn=as_given(lambda seed: truth(*draw(int(seed)))))
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
    # On the reference attention, the run's kernel choice (a model's field before row 11938), Dew
    # imported where it runs, as this tool imports it.
    from dew.nn.kernels import generation

    with generation.chosen(attention="reference"):
        main()
