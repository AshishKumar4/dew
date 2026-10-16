"""The library releases Dew's reference fixtures were recorded against.

A fixture writer runs in an isolated environment and records what one release
computes; run on another, it would record something else under the same name.
So each writer names its cohort here and refuses any other release before it
records (`require`). Most fixtures share one cohort a library; the few recorded
against another release name their own, so the difference stays explicit: a
family a later transformers added, a release's remote code, diffusers' own
Transformers 4 pipelines.
"""

from collections.abc import Mapping
from importlib.metadata import version

RELEASES: Mapping[str, Mapping[str, str]] = {
    "transformers": {"transformers": "5.16.1"},
    # MiniMax-M2's rotary fix (transformers PR 48486) landed in 5.18.0.
    "minimax-m2": {"transformers": "5.18.0"},
    "qwen-image": {"transformers": "5.17.0", "torch": "2.14.0"},
    # Ouro's remote code imports names transformers 5 dropped.
    "ouro": {"transformers": "4.56.2"},
    "timm": {"timm": "1.0.29"},
    "diffusers": {"diffusers": "0.34.0"},
    # The pipelines diffusers 0.34.0 builds read transformers 4's text towers.
    "diffusers-pipelines": {"diffusers": "0.34.0", "transformers": "4.49.0"},
    "diffusers-limits": {"diffusers": "0.34.0", "mpmath": "1.3.0"},
    "diffusers-0.40": {"diffusers": "0.40.0"},
    "rae": {"diffusers": "0.40.0", "transformers": "4.57.1"},
    "gptqmodel": {"gptqmodel": "7.5.0"},
    "trl": {"trl": "1.12.0"},
    "verl": {"verl": "0.9.0"},
}
"""Each cohort's pins: the distribution and the public release of it, a local
build label (`+cpu`) aside."""


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
