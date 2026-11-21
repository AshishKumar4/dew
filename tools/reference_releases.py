"""The library releases Dew's reference fixtures were recorded against.

A fixture writer runs in an isolated environment and records what one release
computes; run on another, it would record something else under the same name.
So each writer names its cohort here and refuses any other release before it
records (`require`). Most fixtures share one cohort a library; the few recorded
against another release name their own, so the difference stays explicit: a
family a later transformers added, a release's remote code, diffusers' own
Transformers 4 pipelines.
"""

import re
import tomllib
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

from pins import pinned


def tested(distribution: str) -> str:
    """The release of `distribution` the reference suite pins in pyproject.toml's test extra, which
    the locks and CI install: the release most fixtures are recorded against."""
    extras = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    if (release := pinned(extras["project"]["optional-dependencies"]["test"], distribution)) is None:
        raise LookupError(f"pyproject.toml's test extra pins no {distribution} release")
    return release


MODELOPT = {"nvidia-modelopt": "0.47.0"}
"""ModelOpt's release, which its codec reader's cohort and the NVFP4 tools' share."""
RELEASES: Mapping[str, Mapping[str, str]] = {
    "transformers": {"transformers": tested("transformers")},
    # MiniMax-M2's rotary fix (transformers PR 48486) landed in 5.18.0.
    "minimax-m2": {"transformers": "5.18.0"},
    "qwen-image": {"transformers": "5.17.0", "torch": "2.14.0"},
    # Ouro's remote code imports names transformers 5 dropped.
    "ouro": {"transformers": "4.56.2"},
    "timm": {"timm": "1.0.29"},
    "torch": {"torch": "2.14.0"},
    # GPT-OSS's exchange gradients, recorded on torch's CPU build (gpt_oss_reference.write_exchange_cases).
    "gpt-oss-exchange": {"transformers": "5.16.1", "torch": "2.14.0"},
    # fla's kernels and Qwen3.5's fla path, in their own environment (joint_checkpoint_reference.on_fla): the
    # fleet's DEW_FLA_PYTHON (dew-ops' session.FLA, torch 2.14.1 past fla #640's Hopper miscompile), whose
    # 2.8.0 pin no node could record against (M-JS-92's refusal, 11-12).
    "fla": {"torch": "2.14.1", "transformers": "5.22.2", "fla-core": "0.5.2"},
    "diffusers": {"diffusers": tested("diffusers")},
    # The pipelines diffusers 0.34.0 builds read transformers 4's text towers.
    "diffusers-pipelines": {"diffusers": tested("diffusers"), "transformers": "4.49.0"},
    "diffusers-limits": {"diffusers": tested("diffusers"), "mpmath": "1.3.0"},
    "diffusers-0.40": {"diffusers": "0.40.0"},
    # The DC-AE fixture's first two variants (tools/diffusers_dc_ae_reference.py), as 0.34.0 recorded them.
    "diffusers-0.34": {"diffusers": "0.34.0"},
    # Wan 2.1's transformer and pipeline fixtures: the transformer, the pipeline's walk and its UMT5 text
    # tower, the test extra's releases (diffusers_wan_reference).
    "diffusers-wan": {"diffusers": tested("diffusers"), "transformers": tested("transformers")},
    "rae": {"diffusers": "0.40.0", "transformers": "4.57.1"},
    "gptqmodel": {"gptqmodel": "7.5.0"},
    "trl": {"trl": "1.12.0"},
    # The suite's locked PEFT, which tests/test_lora_peft.py and tests/test_lora.py run in the test.
    "peft": {"peft": "0.21.2", "transformers": tested("transformers"), "torch": "2.14.0"},
    "verl": {"verl": "0.9.0"},
    # verl at the revision its agentic tools require of their install (verl_interop_reference's REVISION).
    "verl-agentic": {
        "verl": "git+https://github.com/volcengine/verl@12ebe0cb4d300c58449fb6c675379e8700015c51"},
    # The serving engines Dew's serving bench runs in-process (tools/serving_vllm_engine.py,
    # tools/serving_sglang_engine.py), each in its own interpreter, as the fleet builds them
    # (tools/gateway_capture.py's PACKAGES).
    "vllm": {"vllm": "0.30.0"},
    "sglang": {"sglang": "0.5.20"},
    # ModelOpt's NVFP4 codec checkpoints' reader (tools/modelopt_worker.py, tools/venv_reference.py's sites).
    "modelopt": MODELOPT,
    # MiMo Audio's mel reference (tools/torchaudio_mel_worker.py, tools/venv_reference.py's mimo-mel site).
    "torchaudio": {"torchaudio": "2.11.0"},
    # The reference environments each tool's own notes install (row 7933: each type-checked in its own,
    # tools/check_tool_types.py). A release that is a repository at a commit (`git+...@<sha>`) is the
    # checkout its site runs, which no published release holds; `require` reads none of these.
    "tilelang": {"torch": "2.10.0", "tilelang": "0.1.8", "transformers": tested("transformers")},
    "gemma-diffusion": {
        "gemma": "git+https://github.com/google-deepmind/gemma@bf0b49901a428d13e9c2b2629f0eb9c153d3cbd3",
        "hackable-diffusion": "1.0.1", "kauldron": "1.4.4"},
    "dreamerv3": {
        "dreamerv3": "git+https://github.com/danijar/dreamerv3@e01491fad6434b2245a3b8ca201dd7faedcc458c",
        "jax": "0.4.33", "ninjax": "3.5.1", "elements": "3.19.1"},
    "flaxdiff": {
        "flaxdiff": "git+https://github.com/AshishKumar4/FlaxDiff@3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a",
        "jax": "0.5.3", "flax": "0.10.6"},
    # The hybrid DiT's FlaxDiff (flaxdiff_reference's HYBRID_COMMIT), its releases the simple model's.
    "flaxdiff-hybrid": {
        "flaxdiff": "git+https://github.com/AshishKumar4/FlaxDiff@94d2d21caeb79aba48a505fa79d9a1a974751234",
        "jax": "0.5.3", "flax": "0.10.6"},
    # FlaxDiff's source beside Dew's own JAX, for the tools that import Dew too (flaxdiff_pipeline_reference).
    "flaxdiff-source": {
        "flaxdiff": "git+https://github.com/AshishKumar4/FlaxDiff@3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a"},
    # FlaxDiff's trainer beside Dew's own JAX (reference_runs/diffusion.py, which records its checkout):
    # a commit holding the trainer API the run calls (get_diffusion_preset, DiffusionObjective), and the
    # one the schedule, solver and family references fetch their modules at.
    "flaxdiff-trainer": {
        "flaxdiff": "git+https://github.com/AshishKumar4/FlaxDiff@15c55b001304604147a2a8002a76cf5d4c32092f"},
    "laya": {
        "laya": "git+https://github.com/NandhaKishorM/laya@a4a8921afebfd852bba0000475cfb6ab737a124c",
        "transformers": "5.16.1"},
    "lm-engine": {
        "lm-engine":
            "git+https://github.com/open-lm-engine/lm-engine@45b6b57b0deea5b9ef6c9c6f637dc151f1383343",
        "torch": "2.10.0", "pydantic": "2.13.4", "transformers": "4.57.3"},
    "mamba-ssm": {
        "mamba-ssm": "git+https://github.com/state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa",
        "torch": "2.14.0"},
    "maxtext": {"maxtext": "0.2.5"},
    "modelopt-nvfp4": {**MODELOPT, "torch": "2.13.0", "triton": "3.7.1", "diffusers": "0.34.0"},
    "pytorch-fid": {"pytorch-fid": "0.3.0", "torch": "2.14.0"},
    "megatron": {"megatron-core": "0.19.2", "torch": "2.14.0"},
    "torch-qwen35": {"torch": "2.10.0", "transformers": "5.3.0", "triton": "3.6.0", "fla-core": "0.5.2",
                     "flash-linear-attention": "0.5.2", "liger-kernel": "0.8.0"},
    "kimi-k3": {"torch": "2.8.0", "triton": "3.4.0", "fla-core": "0.5.2", "transformers": "4.56.2",
                "compressed-tensors": "0.17.1"},
    "kimi-linear": {"torch": "2.8.0", "triton": "3.4.0", "fla-core": "0.4.0", "transformers": "4.57.1"},
    "vllm-dory": {"vllm": "git+https://github.com/shan18/vllm@55076950b6c155f384d6766d63563c3bc20a0e3f",
                  "torch": "2.14.0"},
    "diffusers-qwen-image": {
        "diffusers": "git+https://github.com/huggingface/diffusers@6256aa7666cedd47443adc8f82da9a10e110b09c",
        "transformers": "5.17.0", "torch": "2.14.0"},
}
"""Each cohort's pins: the distribution and the public release of it, a local
build label (`+cpu`) aside. The default cohorts' releases are the test extra's
(`tested`), so the suite and its fixtures' writers pin one release; a cohort
recorded against another names its own."""


def commit(cohort: str, distribution: str) -> str:
    """The commit cohort `cohort` pins `distribution` at, a repository pin's (`git+...@<sha>`): the one copy
    of it, which the tools that fetch or check that repository read."""
    pin = RELEASES[cohort][distribution]
    found = re.fullmatch(r"git\+https://\S+@([0-9a-f]{40})", pin)
    if found is None:
        raise ValueError(f"{cohort} pins {distribution} at {pin}, a release, not a repository's commit")
    return found[1]



def require(*cohorts: str) -> dict[str, str]:
    """Refuse any installed release but `cohorts`' pins, naming each one that
    differs, and return the pins, which a writer records beside its fixtures."""
    pins = {package: release for cohort in cohorts for package, release in RELEASES[cohort].items()}
    found = {package: version(package).split("+")[0] for package in pins}
    wrong = [f"{package} {found[package]} (pinned {release})" for package, release in pins.items()
             if found[package] != release]
    if wrong:
        raise RuntimeError(f"the {' and '.join(cohorts)} fixtures were recorded against other releases than "
                           f"{', '.join(wrong)}")
    return pins
