"""User-facing code spells each concept through its owning type.

The free functions and string-keyed paths below were replaced by methods and
typed classes, with no alias left behind (docs/design: one concept, one
spelling). This scans what a user reads and copies, the README, the docs,
the examples, the recipes, the tutorials and the landing page, so a removed
spelling cannot come back through a snippet. Each entry names what replaced it.
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

USER_CODE = ("README.md", "CONTRIBUTING.md", "docs", "examples", "recipes", "tutorials", "site/snippets",
             "site/src", "site/live")
"""What a user reads; docs/research and docs/design are records of past decisions."""

EXCLUDED = (":!docs/research", ":!docs/design")

REMOVED = {
    r"(?<![\w.])load_pretrained\(|import[^\n]*\bload_pretrained\b": "Pretrained.load and its kinds",
    r"\bsave_pretrained_decoder\b": "PretrainedDecoder.from_model(...).save",
    r"(?<![\w.])push_to_hub\(|import[^\n]*\bpush_to_hub\b": "a bundle's push_to_hub method",
    r"\bbuild_mesh\b": "MeshSpec(...).build()",
    r"\bdata_partition\b": "DataPartition.of(mesh)",
    r"\bbuild_optimizer\b": "OptimConfig.build(steps)",
    r"\bapply_quantization\b": "Quantization(...).apply(model)",
    r"(?<![\w.])evaluate\(objective|from dew[\w.]* import [^\n]*\bevaluate\b": "Evaluation.run",
    r"(?<![\w.])scalar_loss\(|import[^\n]*\bscalar_loss\b": "objective.scalar_loss",
    r"\bmean_loss\b": "Ratio.mean()",
    r"\bmup_param_groups\b": "ParamGroup.mup",
    r"posthoc import reconstruct|(?<![\w.])reconstruct\(": "Checkpoints.posthoc_ema",
    r"\bdew\.profile\(|import[^\n]*\bprofile\b": "dew.Profiler",
    r"import [^\n]*\bwrite_tokens\b|data\.write_tokens|\bTokenMeta\b": "TokenCorpus.write / TokenCorpus.read",
    r"\bmulti_block_mask\b": "MultiBlockMask.for_grid",
    r"\bsample_trajectory\b": "FlowSDE(...).trajectory",
    r"\bhost_banked\b|\bstream_banked\b": "LayerBanks.place / SafetensorsBanks.stream",
    r"\bload_flaxdiff\b": "TextToImage.from_flaxdiff",
    r"\bpixel_field\b": "Field",
    r"\bclip_score_metric\b|(?<![\w.])(fid|clip_score)\(": "FID(...).score / CLIPScore(...).score",
    r"from dew[\w.]* import [^\n]*\b(perplexity|knn_probe|linear_probe|psnr|ssim|clip)\b": "metric classes",
    r"(?<![\w.])metrics\.\w+\(": "the metric classes",
    r"models\.build\(\s*[\"']": "the model's class, or ModelConfig for a record",
    r"from dew import [^\n]*\b(models|metrics|presets|samplers|datasets|encoders)\b": "dew.registry",
    r"\bdew\.(models|metrics|presets|samplers|datasets|encoders)\b": "dew.registry",
    r"(?<![\w.])sampler=|(?<![\w.])samplers\[|import [^\n]*\bsamplers\b":
        "solver= and dew.registry.solvers",
}


def tracked(pattern: str) -> list[str]:
    """The lines of user-facing files git tracks that match `pattern`."""
    found = subprocess.run(["git", "grep", "-nP", pattern, "--", *USER_CODE, *EXCLUDED], cwd=ROOT,
                           capture_output=True, text=True)
    if found.returncode not in (0, 1):
        pytest.skip(f"git grep is unavailable here: {found.stderr.strip()}")
    return found.stdout.splitlines()


@pytest.mark.parametrize("pattern", list(REMOVED))
def test_a_removed_spelling_does_not_come_back(pattern):
    hits = tracked(pattern)
    assert not hits, f"use {REMOVED[pattern]} instead:\n" + "\n".join(hits)
