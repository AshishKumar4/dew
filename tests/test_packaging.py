"""Dew's install metadata."""
import importlib.metadata
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_accelerator_extras_install_what_the_installed_jax_asks_for():
    """`dewml[cuda12]`, `[cuda13]` and `[tpu]` are how one install gets jax's
    accelerator build. Each extra asks for the packages the installed jax's
    extra of the same name asks for, at the same versions, apart from jaxlib,
    which jax pins itself; a jax that moves without its extras fails here. A
    CUDA extra also brings Dew's own FlashAttention-2 wheel for its major."""
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    own = {"cuda12": "dew-flash-attn-cu12", "cuda13": "dew-flash-attn-cu13"}

    def key(requirement: Requirement) -> tuple[str, frozenset[str], str]:
        return canonicalize_name(requirement.name), frozenset(requirement.extras), str(requirement.specifier)

    for extra in ("cuda12", "cuda13", "tpu"):
        wanted = {
            key(requirement)
            for requirement in map(Requirement, importlib.metadata.requires("jax") or ())
            if requirement.marker is not None
            and requirement.marker.evaluate({"extra": extra})
            and not requirement.marker.evaluate({"extra": ""})
            and requirement.name != "jaxlib"
        }
        assert wanted, f"the installed jax has no {extra} extra"
        listed = [Requirement(line) for line in extras.get(extra, ())]
        jax_asks = {key(requirement) for requirement in listed if requirement.name != own.get(extra)}
        assert jax_asks == wanted, extra
        assert [requirement.name for requirement in listed if requirement.name == own.get(extra)] == \
            ([own[extra]] if extra in own else []), extra


def test_every_requirement_names_a_release():
    """PyPI refuses a distribution whose requirements name a URL, so every
    requirement and extra of the published dewml names a release; the jax
    build CI runs on comes through constraints.txt."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    lines = [*project["dependencies"], *(line for extra in project["optional-dependencies"].values()
                                         for line in extra)]
    assert [line for line in lines if Requirement(line).url is not None] == []
