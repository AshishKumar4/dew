"""Dew's install metadata."""
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_accelerator_extras_install_what_the_pinned_jax_asks_for():
    """`dew-ml[cuda12]`, `[cuda13]` and `[tpu]` are how an install gets the
    pinned jax's accelerator build. pip can't satisfy PyPI's jax[cuda13]
    against the pinned jax's archive URL, so `pip install "dew-ml @ git+..."
    "jax[cuda13]"` failed with ResolutionImpossible. Each extra asks for the
    packages the installed jax's extra of the same name asks for, at the same
    versions, apart from jaxlib, which jax pins itself; a pin that moves
    without its extras fails here."""
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]

    def key(requirement: Requirement) -> tuple[str, frozenset[str], str]:
        return canonicalize_name(requirement.name), frozenset(requirement.extras), str(requirement.specifier)

    for extra in ("cuda12", "cuda13", "tpu"):
        wanted = {key(requirement) for requirement in map(Requirement, importlib.metadata.requires("jax") or ())
                  if requirement.marker is not None and requirement.marker.evaluate({"extra": extra})
                  and not requirement.marker.evaluate({"extra": ""}) and requirement.name != "jaxlib"}
        assert wanted, f"the installed jax has no {extra} extra"
        assert {key(Requirement(line)) for line in extras.get(extra, ())} == wanted, extra
