"""Records a reference implementation run in its own environment over drawn weights and inputs, for a
test of Dew against it: each seed's draw (the test's `draw`) run by the site's worker in float32 and
float64, the recording sized paired against Dew's float32 run (the test's `dew`) and kept as its evidence
and its truth's sketch (`record_case(..., summary=True)`), or, for a site whose test has no float64 path of
Dew's to recompute the truth by, its outputs whole (`outputs`).

    python tools/venv_reference.py SITE PYTHON [SOURCE]

PYTHON is the interpreter of a virtual environment holding the site's package (a distribution's version
checked against the site's); SOURCE a checkout put on the worker's path where the package is a
repository rather than a distribution.
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

PIPELINES = importlib.import_module("reference_releases").RELEASES["diffusers-pipelines"]
NATIVE_PIPELINES = (f"huggingface/diffusers {PIPELINES['diffusers']} with huggingface/transformers "
                    f"{PIPELINES['transformers']}")
"""The diffusers-pipelines cohort (tools/reference_releases.py) the saved tiny pipelines' own calls run in,
as a recording names it."""
SITES = {"conv": {"test": "test_dreamerv3_conv", "worker": "dreamerv3_worker.py",
                  "reference": "danijar/dreamerv3 e01491fad6434b2245a3b8ca201dd7faedcc458c",
                  "license": "MIT"},
         **{site: {"test": f"test_maxtext_{site}", "worker": "maxtext_worker.py",
                   "reference": "AI-Hypercomputer/maxtext 0.2.5", "license": "Apache-2.0",
                   "distribution": ("maxtext", "0.2.5")} for site in ("gpt3", "engram")},
         "laya": {"test": "test_decision_laya", "worker": "laya_worker.py",
                  "reference": "NandhaKishorM/laya a4a8921afebfd852bba0000475cfb6ab737a124c",
                  "license": "Apache-2.0", "distribution": ("transformers", "5.16.1")},
         "kimi-k3": {"test": "test_kimi_k3", "worker": "kimi_worker.py", "device": "cuda",
                     "reference": "moonshotai/Kimi-K3 f831ab66814297da540d832a5235f8e904f29d06",
                     "license": "other (kimi-k3)", "distribution": ("fla-core", "0.5.2")},
         "kimi-k3-vision": {"test": "test_kimi_k3_vision", "worker": "kimi_k3_vision_worker.py",
                            "reference": "moonshotai/Kimi-K3 f831ab66814297da540d832a5235f8e904f29d06",
                            "license": "other (kimi-k3)", "distribution": ("transformers", "4.56.2")},
         "kimi-linear": {"test": "test_kimi_linear", "worker": "kimi_worker.py", "device": "cuda",
                         "reference": "moonshotai/Kimi-Linear-48B-A3B-Instruct "
                                      "e1df551a447157d4658b573f9a695d57658590e9",
                         "license": "MIT", "distribution": ("fla-core", "0.4.0")},
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
                      "reference": "pytorch/audio 2.11.0 MelSpectrogram", "license": "BSD-2-Clause",
                      "distribution": ("torchaudio", "2.11.0+cpu"), "host": True, "outputs": True},
         **{"flaxdiff-" + case.replace("/", "-"): {
             "test": "test_flaxdiff_family_source", "arms": case, "worker": "flaxdiff_family_worker.py",
             "reference": "AshishKumar4/FlaxDiff 15c55b001304604147a2a8002a76cf5d4c32092f", "license": "MIT"}
            for case in importlib.import_module("flaxdiff_family_cases").CASES},
         **{f"multimodal-{family}": {"test": "test_multimodal_model", "arms": family,
                                     "worker": "multimodal_worker.py",
                                     "reference": "huggingface/transformers 5.16.1",
                                     "license": "Apache-2.0", "distribution": ("transformers", "5.16.1")}
            for family in ("gemma4", "qwen35", "llama4", "gemma3n")},
         **{f"codec-{family}-{kind}": {"test": "test_codec_checkpoints", "arms": f"{family}/{kind}",
                                       "worker": "codec_worker.py",
                                       "reference": "neuralmagic/compressed-tensors 0.17.1 with "
                                                    "huggingface/transformers 5.16.1",
                                       "license": "Apache-2.0",
                                       "distribution": ("compressed-tensors", "0.17.1")}
            for family, kind in (("nvfp4_qdq", "unrounded"), ("nvfp4_qdq", "e4m3"), ("storage", "fp8"),
                                 ("storage", "nvfp4"))},
         **{f"native-{name}": {"test": "check_native_diffusion", "arms": arms,
                               "worker": "native_diffusion_worker.py", "reference": NATIVE_PIPELINES,
                               "license": "Apache-2.0",
                               "distribution": ("transformers", PIPELINES["transformers"]),
                               "host": True, "outputs": True}
            for name, arms in NATIVE.items()}}
"""Each site's test module (its `FIXTURE`, `draw`, `dew` and `OUTPUTS`, or, with `arms`, its
`pipeline_arms(arms)`, which gives the four), worker and reference, the distribution and version its
environment must hold, where the reference is one, and the device its worker's float32 run takes, the CPU
unless named; `host` where its `dew` runs on the host as given (a pipeline's walk, its own programs
inside), and `outputs` where the recording keeps the outputs whole."""


def site_arms(site: str):
    """`site`'s recording's fixture, draw, Dew run and leaves: its test's `FIXTURE` (None for a site only
    recorded, never saved), `draw`, `dew` and `OUTPUTS`, or its test's `pipeline_arms` of the site's
    `arms`."""
    known = SITES[site]
    test = importlib.import_module(known["test"])
    return (test.pipeline_arms(known["arms"]) if "arms" in known
            else (getattr(test, "FIXTURE", None), test.draw, test.dew, test.OUTPUTS))


def recording(site: str, python: str, source: str | None, given=None) -> dict[str, np.ndarray]:
    """`site`'s recording: each seed's draw run by the site's worker under `python` (`session`), its float32
    run and its float64 truth, sized paired against Dew's float32 run (`record_case`), over the site's arms
    (`site_arms`, or `given`, those already built)."""
    import jax
    import jax.numpy as jnp
    from recorded_references import record_case
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
    environment |= {"JAX_PLATFORMS": "cpu", **({} if source is None else {"PYTHONPATH": source})}
    with session([python, str(ROOT / "tools" / known["worker"]), site], env=environment) as worker:
        def evaluate(inputs):
            answer = worker.ask({f"case.{key}": value for key, value in inputs.items()})
            reference, truth = arms(answer, "case.")
            return tuple({leaf: arm[leaf] for leaf in outputs} for arm in (reference, truth))

        dew = ((lambda inputs: run(jnp.float32, inputs)) if known.get("host")
               else jax.jit(lambda inputs: run(jnp.float32, inputs)))
        return record_case(site, draw, evaluate, store_inputs=False, summary=not known.get("outputs"),
                           dew=dew)


def main() -> None:
    from recorded_references import provenance, save

    site, python = sys.argv[1], sys.argv[2]
    source = sys.argv[3] if len(sys.argv) > 3 else None
    known, given = SITES[site], site_arms(site)
    arrays = recording(site, python, source, given)
    written = save(given[0], {**arrays, "provenance": provenance(
        known["reference"], "", known.get("device", "cpu"),
        f"python tools/venv_reference.py {site} PYTHON{' SOURCE' if source is not None else ''}")},
        reference=known["reference"], license=known["license"])
    print(f"{written}: {sorted(arrays)}")


if __name__ == "__main__":
    main()
