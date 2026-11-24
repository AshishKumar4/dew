"""Records a reference implementation run in its own environment over drawn weights and inputs, for a
test of Dew against it: each seed's draw (the test's `draw`) run by the site's worker in float32 and
float64, the recording sized paired against Dew's float32 run (the test's `dew`) and kept as its evidence
and its truth's sketch (`record_case(..., summary=True)`), or, for a site whose test has no float64 path of
Dew's to recompute the truth by, its outputs whole (`outputs`).

    python tools/venv_reference.py SITE PYTHON [SOURCE] [--kept DIR] [--resume DIR]

PYTHON is the interpreter of a virtual environment holding the site's package (a distribution's version
checked against the site's); SOURCE a checkout put on the worker's path where the package is a
repository rather than a distribution. `--kept DIR` keeps each seed's evaluation there as it is made, and
`--resume DIR` reads an interrupted run's instead of making it again (`recorded_references._Kept`): a
recording whose task is recalled goes on from what it had (armada hands a recalled attempt's artifacts to
the next at $ARMADA_RESUME).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The tests' own path (pyproject's `pythonpath`: the repository root), their directory and this one.
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests"), str(ROOT)]

_WORKER = importlib.import_module("native_diffusion_worker")
NATIVE = {**{f"{task}-{group}": (task, group) for task in _WORKER.TASKS for group in _WORKER.groups(task)},
          **{f"grid-{name}": f"grid-{name}"
             for name in importlib.import_module("diffusers_scheduler_pipeline_reference").SCHEDULERS}}
"""tools/check_native_diffusion.py's recorded sites, each its `pipeline_arms`: each saved pipeline's groups
of leaves (tools/native_diffusion_worker.py's `groups`), and the SD pipeline's walk over each source
scheduler's grid."""

DIFFUSERS = importlib.import_module("reference_releases").tested("diffusers")
RELEASES = importlib.import_module("reference_releases")
PINS = RELEASES.RELEASES
PIPELINES = PINS["diffusers-pipelines"]
MODELOPT = PINS["modelopt"]["nvidia-modelopt"]
MAXTEXT = PINS["maxtext"]["maxtext"]
VLLM = PINS["vllm"]["vllm"]
TORCHAUDIO = PINS["torchaudio"]["torchaudio"]
NATIVE_PIPELINES = (f"huggingface/diffusers {PIPELINES['diffusers']} with huggingface/transformers "
                    f"{PIPELINES['transformers']}")
"""The diffusers-pipelines cohort (tools/reference_releases.py) the saved tiny pipelines' own calls run in,
as a recording names it."""
SITES = {"conv": {"test": "test_dreamerv3_conv", "worker": "dreamerv3_worker.py",
                  "reference": f"danijar/dreamerv3 {RELEASES.commit('dreamerv3', 'dreamerv3')}",
                  "license": "MIT"},
         **{site: {"test": f"test_maxtext_{site}", "worker": "maxtext_worker.py",
                   "reference": f"AI-Hypercomputer/maxtext {MAXTEXT}", "license": "Apache-2.0",
                   "distribution": ("maxtext", MAXTEXT)} for site in ("gpt3", "engram")},
         "llama-cpp": {"test": "test_llama_cpp_export", "worker": "llama_cpp_worker.py",
                       "reference": "ggml-org/llama.cpp 7fe450e19305b828c199d602c23a8337aaa1f03b (v0.5.0)",
                       "license": "MIT", "host": True, "outputs": True, "judged": True},
         "vllm-replay": {"test": "test_engine_replay", "worker": "vllm_replay_worker.py", "device": "cuda",
                         "reference": (f"vllm-project/vllm {VLLM} (its CUDA 12.9 build) with "
                                       "huggingface/transformers 5.16.1"),
                         "license": "Apache-2.0", "distribution": ("vllm", f"{VLLM}+cu129"), "host": True,
                         "outputs": True},
         "laya": {"test": "test_decision_laya", "worker": "laya_worker.py",
                  "reference": f"NandhaKishorM/laya {RELEASES.commit('laya', 'laya')}",
                  "license": "Apache-2.0", "distribution": ("transformers", PINS["laya"]["transformers"])},
         "kimi-k3": {"test": "test_kimi_k3", "worker": "kimi_worker.py", "device": "cuda",
                     "reference": "moonshotai/Kimi-K3 f831ab66814297da540d832a5235f8e904f29d06",
                     "license": "other (kimi-k3)", "distribution": ("fla-core", PINS["kimi-k3"]["fla-core"])},
         "kimi-k3-vision": {"test": "test_kimi_k3_vision", "worker": "kimi_k3_vision_worker.py",
                            "reference": "moonshotai/Kimi-K3 f831ab66814297da540d832a5235f8e904f29d06",
                            "license": "other (kimi-k3)",
                            "distribution": ("transformers", PINS["kimi-k3"]["transformers"])},
         "kimi-linear": {"test": "test_kimi_linear", "worker": "kimi_worker.py", "device": "cuda",
                         "reference": "moonshotai/Kimi-Linear-48B-A3B-Instruct "
                                      "e1df551a447157d4658b573f9a695d57658590e9",
                         "license": "MIT", "distribution": ("fla-core", PINS["kimi-linear"]["fla-core"])},
         **{f"uvit-{case}": {"test": "test_uvit_source", "arms": case, "worker": "uvit_worker.py",
                             "reference": "baofff/U-ViT ce551708dc9cde9818d2af7d84dfadfeb7bd9034",
                             "license": "MIT"} for case in ("published", "variant")},
         **{f"jepa-{case}": {"test": "test_jepa_source", "arms": case, "worker": "vjepa_worker.py",
                             "reference": "facebookresearch/jepa 51c59d518fc63c08464af6de585f78ac0c7ed4d5",
                             "license": "CC-BY-NC-4.0"} for case in ("published", "variant")},
         **{f"router-{case}": {"test": "test_router_source", "arms": case, "worker": "vllm_router_worker.py",
                               "reference": "vllm-project/vllm ced6857afa0ea7b2e3f0846a62e1394e90f15607",
                               "license": "Apache-2.0"}
            for case in ("kimi", "grouped", "softmax", "unbiased")},
         "repa-preprocessing": {"test": "test_alignment", "worker": "repa_worker.py",
                                "reference": "sihyun-yu/REPA 67f714503e3892f993844aab088ffc5791c92613 with "
                                             "End2End-Diffusion/iREPA "
                                             "99ad4ac234efe8de52ce157120f72856e836d09f",
                                "license": "MIT"},
         "mimo-mel": {"test": "test_mimo_audio", "arms": "mel", "worker": "torchaudio_mel_worker.py",
                      "reference": f"pytorch/audio {TORCHAUDIO} MelSpectrogram", "license": "BSD-2-Clause",
                      "distribution": ("torchaudio", f"{TORCHAUDIO}+cpu"), "host": True, "outputs": True},
         **{"flaxdiff-" + case.replace("/", "-"): {
             "test": "test_flaxdiff_family_source", "arms": case, "worker": "flaxdiff_family_worker.py",
             "reference": f"AshishKumar4/FlaxDiff {RELEASES.commit('flaxdiff-trainer', 'flaxdiff')}",
             "license": "MIT"}
            for case in importlib.import_module("flaxdiff_family_cases").CASES},
         **{f"multimodal-{family}": {
             "test": "test_multimodal_model", "arms": family, "worker": "multimodal_worker.py",
             "reference": "huggingface/transformers 5.16.1", "license": "Apache-2.0",
             "distribution": ("transformers", "5.16.1"),
             "weighed": family in importlib.import_module("multimodal_families").WEIGHED,
             # Its unweighted sizing never resolves (Qwen3.5's kept GPU records' reading, rows 5236 and
             # 5237; Gemma 3n's ladder to the cap in two GPU recordings, IG-MMGPU16 and 17): the cap
             # declared, its ladder not climbed (`record_case`'s `unweighted`).
             **({"unweighted": importlib.import_module("dew.parity").MOST_SEEDS}
                if family in ("qwen35", "gemma3n") else {})}
            for family in (importlib.import_module("multimodal_families").DRAWN
                           | importlib.import_module("multimodal_families").WEIGHED)},
         **{f"multimodal-{family}-text": {
             "test": "test_multimodal_model", "arms": f"{family}/text", "worker": "multimodal_worker.py",
             "reference": "huggingface/transformers 5.16.1", "license": "Apache-2.0",
             "distribution": ("transformers", "5.16.1"),
             "truths": importlib.import_module("multimodal_families").TEXT_TRUTHS}
            for family in importlib.import_module("multimodal_families").TEXT_DRAWN},
         **{f"codec-modelopt-{name}": {"test": "test_codec_checkpoints", "arms": f"modelopt/{name}",
                                     "worker": "modelopt_worker.py",
                                     "reference": f"NVIDIA/TensorRT-Model-Optimizer {MODELOPT} with "
                                                  "huggingface/transformers 5.16.1",
                                     "license": "Apache-2.0", "distribution": ("nvidia-modelopt", MODELOPT),
                                     "outputs": True} for name in ("tiny", "mixed")},
         **{f"codec-{family}-{kind}": {"test": "test_codec_checkpoints", "arms": f"{family}/{kind}",
                                       "worker": "codec_worker.py",
                                       "reference": "neuralmagic/compressed-tensors 0.17.1 with "
                                                    "huggingface/transformers 5.16.1",
                                       "license": "Apache-2.0",
                                       "distribution": ("compressed-tensors", "0.17.1")}
            for family, kinds in (("storage", ("fp8", "nvfp4")), ("nvfp4_qdq", ("unrounded", "e4m3")))
            for kind in kinds},
         **{f"beam-{name}": {"test": "test_beam_search", "arms": name, "worker": "beam_worker.py",
                             "reference": "huggingface/transformers 5.16.1 with pytorch/pytorch 2.14.0",
                             "license": "Apache-2.0", "distribution": ("transformers", "5.16.1"),
                             "host": True, "outputs": True}
            for name in ("llama-tiny", "qwen3-tiny")},
         "lm-steps": {"test": "test_reference_runs", "arms": "qwen3-tiny", "worker": "lm_steps_worker.py",
                      "reference": "huggingface/transformers 5.16.1 with pytorch/pytorch 2.14.0",
                      "license": "Apache-2.0", "distribution": ("transformers", "5.16.1"), "host": True,
                      "outputs": True},
         "qwen-image-export": {"test": "test_qwen_image_source", "arms": "export",
                               "worker": "qwen_image_export_reference.py",
                               "reference": "huggingface/diffusers "
                                            + RELEASES.commit("diffusers-qwen-image", "diffusers"),
                               "license": "Apache-2.0",
                               "distribution": ("transformers",
                                                PINS["diffusers-qwen-image"]["transformers"])},
         **{f"wan-{case}": {"test": "test_wan_source", "arms": case, "worker": "wan_worker.py",
                            "reference": f"huggingface/diffusers {DIFFUSERS}", "license": "Apache-2.0",
                            "distribution": ("diffusers", DIFFUSERS)} for case in ("published", "variant")},
         **{f"wan-bf16-{case}": {"test": "test_wan_source", "arms": f"bf16/{case}", "worker": "wan_worker.py",
                                 "reference": f"huggingface/diffusers {DIFFUSERS}", "license": "Apache-2.0",
                                 "distribution": ("diffusers", DIFFUSERS), "narrow": "bfloat16"}
            for case in ("published", "variant")},
         **{f"native-{name}": {"test": "check_native_diffusion", "arms": arms,
                               "worker": "native_diffusion_worker.py", "reference": NATIVE_PIPELINES,
                               "license": "Apache-2.0",
                               "distribution": ("transformers", PIPELINES["transformers"]),
                               "host": True, "outputs": True}
            for name, arms in NATIVE.items()}}
"""Each site's test module (its `FIXTURE`, `draw`, `dew` and `OUTPUTS`, or, with `arms`, its
`pipeline_arms(arms)`, which gives the four), worker and reference, the distribution and version its
environment must hold, where the reference is one, and the device its worker's float32 run takes, the CPU
unless named; `narrow` the precision the reference's narrow run and Dew's take where not float32; `host`
where its `dew` runs on the host as given (a pipeline's walk, its own programs inside), `outputs` where
the recording keeps the outputs whole, `judged` where the recording is the arm its test's check 1 judges
and `dew` that arm's reference (another library running what Dew exports for it), so its seeds are sized
with the two in those places, and `truths` where its float32 errors no count of draws resolves: the
recording keeps the reference's float64 truth alone at that many draws (`record_truths`), which its test
holds Dew's float64 run to (docs/design/testing.md); `weighed` where its test declares check 1's weighed
check: asked for a draw's rounding scale (the request's `<draw>.weighed`), its worker answers it alone
(`<draw>.scale.<leaf>`, the reference's own, `_dew_torch.rounding.rounding_scale`), which the recording
keeps (`record_case`'s `scale`), asked only of the draws the weighed check's stages read, the scale's taps
costing what the draw's runs do several times over."""


def site_arms(site: str):
    """`site`'s recording's fixture, draw, Dew run and leaves: its test's `FIXTURE` (None for a site only
    recorded, never saved), `draw`, `dew` and `OUTPUTS`, or its test's `pipeline_arms` of the site's
    `arms`."""
    known = SITES[site]
    test = importlib.import_module(known["test"])
    return (test.pipeline_arms(known["arms"]) if "arms" in known
            else (getattr(test, "FIXTURE", None), test.draw, test.dew, test.OUTPUTS))


def recording(site: str, python: str, source: str | None, given=None, *, kept_in: Path | None = None,
              resumed_from: Path | None = None) -> dict[str, np.ndarray]:
    """`site`'s recording: each seed's draw run by the site's worker under `python` (`session`), its float32
    run and its float64 truth, sized paired against Dew's float32 run (`record_case`), or its float64 truth
    alone at a few draws (`truths`, `record_truths`), over the site's arms (`site_arms`, or `given`, those
    already built)."""
    import jax
    import jax.numpy as jnp
    from recorded_references import record_case, record_truths
    from reference_worker import arms, session

    known = SITES[site]
    _, draw, run, outputs = given or site_arms(site)
    if "distribution" in known:
        name, wanted = known["distribution"]
        held = subprocess.run([python, "-c", f"import importlib.metadata as m; print(m.version({name!r}))"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if held != wanted:
            raise SystemExit(f"the recording is of {name} {wanted}; {python} holds {held}")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    # The worker's strings hashed alike at every run, so a set's order, and an operation order read from one,
    # repeats: a weighed scale's tap draws its signs in its operations' order, and a record that does not
    # repeat to the bit does not resume (`recorded_references._Kept`; Gemma 4's scale differed by hash seed).
    environment |= {"JAX_PLATFORMS": "cpu", "PYTHONHASHSEED": "0",
                    **({} if source is None else {"PYTHONPATH": source})}
    command = [python, str(ROOT / "tools" / known["worker"]), site]
    weighed = known.get("weighed", False)
    with session(command, env=environment) as worker:
        def evaluate(inputs):
            answer = worker.ask({f"case.{key}": value for key, value in inputs.items()})
            reference, truth = arms(answer, "case.")
            # The leaves the reference made that Dew replays (`record_case`), where the worker gives any.
            replayed = {leaf.removeprefix("case.replayed."): value for leaf, value in answer.items()
                        if leaf.startswith("case.replayed.")}
            return (*({leaf: arm[leaf] for leaf in outputs} for arm in (reference, truth)), replayed)

        def scale(inputs, seed: int) -> dict[str, np.ndarray]:
            answer = worker.ask({**{f"case.{key}": value for key, value in inputs.items()},
                                 "case.weighed": np.ones((), bool)})
            made = {leaf.removeprefix("case.scale."): value for leaf, value in answer.items()
                    if leaf.startswith("case.scale.")}
            missing = sorted(set(outputs) - set(made))
            if missing:
                raise ValueError(f"{site}'s worker answered draw {seed} with no rounding scale of "
                                 f"{missing}: a weighed site's worker answers `<draw>.scale.<leaf>` for "
                                 f"every leaf")
            return {leaf: made[leaf] for leaf in outputs}

        if "truths" in known:
            return record_truths(site, draw, evaluate, known["truths"])
        narrow = getattr(jnp, known.get("narrow", "float32"))
        dew = ((lambda inputs: run(narrow, inputs)) if known.get("host")
               else jax.jit(lambda inputs: run(narrow, inputs)))
        return record_case(site, draw, evaluate, store_inputs=False,
                           summary=not known.get("outputs"), dew=dew,
                           judged=known.get("judged", False), kept_in=kept_in, resumed_from=resumed_from,
                           scale=scale if weighed else None, unweighted=known.get("unweighted"))


DEVICE_WORKERS = frozenset({"multimodal_worker.py"})
"""The workers that run where DEW_REFERENCE_DEVICE names, a GPU's float64 units among them; every other
site's runs on its own device (`SITES`' `device`, the CPU by default)."""

def main() -> None:
    import argparse

    from recorded_references import provenance, save

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("site")
    parser.add_argument("python")
    parser.add_argument("source", nargs="?")
    parser.add_argument("--kept", type=Path, help="keep each seed's evaluation here as it is made")
    parser.add_argument("--resume", type=Path, help="an interrupted run's --kept, read and not made again")
    args = parser.parse_args()
    site, python, source = args.site, args.python, args.source
    known, given = SITES[site], site_arms(site)
    arrays = recording(site, python, source, given, kept_in=args.kept, resumed_from=args.resume)
    # The device the reference ran on: a site's own, or where its worker honors it, the one
    # DEW_REFERENCE_DEVICE names (`multimodal_worker.py`'s alone), which the command reproducing it states.
    override = os.environ.get("DEW_REFERENCE_DEVICE") if known["worker"] in DEVICE_WORKERS else None
    device = override or known.get("device", "cpu")
    prefix = "" if override is None else f"DEW_REFERENCE_DEVICE={override} "
    command = f"{prefix}python tools/venv_reference.py {site} PYTHON{' SOURCE' if source is not None else ''}"
    written = save(given[0], {**arrays, "provenance": provenance(known["reference"], "", device, command)},
                   reference=known["reference"], license=known["license"])
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
