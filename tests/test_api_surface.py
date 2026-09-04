"""The API is the one in docs/design/api.md, and only that one.

Section 5 of the design names what the new surface replaces. Dew is
unpublished and keeps no migrations (decision 7), so each of those names
reads zero across the tree: not deprecated, not aliased, gone. A name that
comes back is a second path, which is what the design exists to prevent.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The trees a user or a test can import from or read. The tutorials are parked
# for a rewrite on the new surface and stay out until then.
TREES = ("src", "recipes", "examples", "tools", "tests", "docs", "README.md")

# Design section 5, plus the wave-1 deletions and the names the reviews of
# 2026-09-03 found doing a job the design gives to something else.
SUPERSEDED = (
    "ObjectiveTrainer", "SimpleTrainer", "SimpleTrainState", "GeneralDiffusionTrainer",
    "RandomMarkovState", "MarkovState",
    "get_diffusion_preset",
    "canonicalize_architecture", "ARCHITECTURE_SUFFIX_FLAGS", "MODEL_REGISTRY",
    "map_config_strings", "apply_precision_policy", "build_model",
    "DiffusionInputConfig", "ConditionalInputConfig", "model_key_override",
    "serialize_model", "parse_config",
    "datasetMap", "onlineDatasetMap", "mediaDatasetMap",
    "FLAXDIFF_AUGMENT_MODE",
    "__encode__", "__decode__",
    "make_validation_step", "log_validation_artifacts", "validation_step_args",
    "training_steps_per_epoch",
    "use_hilbert", "use_zigzag", "+hilbert", "+zigzag",
    "convert_legacy_checkpoint",
)

# A name may appear in the two review documents and the design, which record
# what was replaced, and in this file.
RECORDS = {
    "docs/design/api.md",
    "docs/design/review-api-2026-09-03.md",
    "docs/design/review-multihost-2026-09-03.md",
    "docs/design/plan.md",
    "tests/test_api_surface.py",
}


def occurrences(name: str) -> list[str]:
    """`path:line` for every appearance of `name` outside the records."""
    result = subprocess.run(
        ["git", "grep", "-n", "-F", "--", name, "--", *TREES],
        cwd=ROOT, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines()
             if line.split(":", 1)[0] not in RECORDS
             and not re.search(r"\.(pyc|npz|safetensors|png)$", line.split(":", 1)[0])]
    return lines


@pytest.mark.parametrize("name", SUPERSEDED)
def test_a_superseded_name_is_gone(name):
    found = occurrences(name)
    assert not found, f"{name!r} survives in {len(found)} place(s):\n" + "\n".join(found[:12])


def test_training_imports_no_modality():
    """`dew.training` is the general trainer: it knows no diffusion, no input
    conventions, no sampler and no tracker backend (design rule 4 and the
    wave 2 acceptance)."""
    code = (
        "import sys, dew.training\n"
        "loaded = sorted(m for m in sys.modules if m.startswith(('dew.diffusion', 'dew.inputs', 'dew.sampling', 'wandb')))\n"
        "print(loaded)\n")
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "JAX_PLATFORMS": "cpu"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout
