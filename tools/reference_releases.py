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


def tested(distribution: str) -> str:
    """The release of `distribution` the reference suite pins in pyproject.toml's test extra, which
    the locks and CI install: the release most fixtures are recorded against."""
    extras = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    for requirement in extras["project"]["optional-dependencies"]["test"]:
        pinned = re.fullmatch(rf"{re.escape(distribution)}==([^;\s]+)", requirement)
        if pinned:
            return pinned[1]
    raise LookupError(f"pyproject.toml's test extra pins no {distribution} release")


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
    # fla's kernels and Qwen3.5's fla path, in their own environment (joint_checkpoint_reference.on_fla).
    "fla": {"torch": "2.8.0", "transformers": "5.16.1", "fla-core": "0.5.2"},
    "diffusers": {"diffusers": tested("diffusers")},
    # The pipelines diffusers 0.34.0 builds read transformers 4's text towers.
    "diffusers-pipelines": {"diffusers": tested("diffusers"), "transformers": "4.49.0"},
    "diffusers-limits": {"diffusers": tested("diffusers"), "mpmath": "1.3.0"},
    "diffusers-0.40": {"diffusers": "0.40.0"},
    # Wan 2.1's transformer and pipeline fixtures, recorded against diffusers 0.34.0, whose UniPC schedule a later
    # release moves by a step (diffusers_wan_reference).
    "diffusers-0.34": {"diffusers": "0.34.0", "transformers": "5.16.1"},
    "rae": {"diffusers": "0.40.0", "transformers": "4.57.1"},
    "gptqmodel": {"gptqmodel": "7.5.0"},
    "trl": {"trl": "1.12.0"},
    "peft": {"peft": "0.20.0", "transformers": tested("transformers"), "torch": "2.14.0"},
    "verl": {"verl": "0.9.0"},
}
"""Each cohort's pins: the distribution and the public release of it, a local
build label (`+cpu`) aside. The default cohorts' releases are the test extra's
(`tested`), so the suite and its fixtures' writers pin one release; a cohort
recorded against another names its own."""


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
