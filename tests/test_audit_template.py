"""tools/audit_template.py reads the sampled turn off the audited template.

The tiny-tools fixture writes tool calls as `<tool_call id="...">name {json}`,
not Qwen's Hermes JSON, and has no reasoning markup. Its histories stay
append-only whenever the template re-renders the parsed call as it wrote it,
so the audit must report every case but the compact-JSON one as merging.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_template", ROOT / "tools" / "audit_template.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_non_hermes_template_audits_its_own_tool_call_syntax(audit):
    report = audit.audit_template(str(ROOT / "tests" / "fixtures" / "tokenizers" / "tiny-tools"), "<|im_end|>")
    held = {(case["observation_role"], case["reasoning_in_sampled_turn"], case["arguments_as_json_string"],
             case["sampled_compact_json"]): case["strict_prefix_holds"] for case in report["cases"]}
    assert held == {("tool", False, False, False): True, ("tool", True, False, False): True,
                    ("user", False, False, False): True, ("user", True, False, False): True,
                    ("tool", False, True, False): True, ("tool", False, False, True): False}
    assert [case["sampled_compact_json"] for case in report["lenient_merge_would_corrupt"]] == [True]
