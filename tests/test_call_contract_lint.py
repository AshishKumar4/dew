"""SLOP014: a layer's call reads what its block hands it, and a mixer's `reads` is what it reads."""

import textwrap
from pathlib import Path

import pytest
from tool_support import load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


INPUTS = 'MIXING = ("pairwise_mask", "position_bias", "admitted")\n'
CASES = {
    "a call argument taken and deleted": ("""
        class Layer:
            def __call__(self, x, decode=False, kv_store=None):
                del kv_store
                return x if decode else -x
        """, True),
    "every call argument read": ("""
        class Layer:
            def __call__(self, x, decode=False, kv_store=None):
                return x if decode or kv_store else -x
        """, False),
    "a mask read without being declared": ("""
        class Mixer:
            reads = frozenset()

            def __call__(self, x, call):
                return x if call.metadata.pairwise_mask is None else -x
        """, True),
    "a mask declared and never read": ("""
        class Mixer:
            reads = frozenset({"pairwise_mask"})

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "a mask declared and read in a helper of the module": ("""
        def masked(metadata):
            return metadata.pairwise_mask

        class Mixer:
            reads = frozenset({"pairwise_mask"})

            def __call__(self, x, call):
                return x if masked(call.metadata) is None else -x
        """, False),
    "a field no call sets": ("""
        class Mixer:
            reads = frozenset({"segments"})

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "reads spelled some other way": ("""
        class Mixer:
            reads = MASKS

            def __call__(self, x, call):
                return x if call.decode else -x
        """, True),
    "a tokenwise mixer that ignores its call": ("""
        class Mixer:
            reads = TOKENWISE

            def __call__(self, x, call):
                return 2 * x
        """, False),
    "a tokenwise mixer that reads a mask": ("""
        class Mixer:
            reads = TOKENWISE

            def __call__(self, x, call):
                return x if call.metadata.admitted is None else -x
        """, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_call_that_drops_its_arguments_or_misdeclares_its_reads_is_reported(lint, case, tmp_path):
    body, reported = CASES[case]
    (tmp_path / "src" / "dew" / "nn").mkdir(parents=True)
    (tmp_path / "src" / "dew" / "nn" / "inputs.py").write_text(INPUTS)
    (tmp_path / "src" / "dew" / "nn" / "layer.py").write_text(textwrap.dedent(body))
    [module] = [module for module in lint.collect(["src/dew/nn"], tmp_path, "dew")
                if module.path.stem == "layer"]
    assert any(True for _ in lint.calls(module)) is reported


@pytest.mark.parametrize("body, reported", [
    ("assert step._cache_size() == 1\n", True),
    ("with Compiles('step') as compiled:\n    step()\nassert compiled.count == 1\n", False),
], ids=["private cache size", "public compile count"])
def test_a_read_of_jaxs_private_cache_size_is_reported(lint, body, reported, tmp_path):
    """SLOP021: a jitted function's private `_cache_size` is read nowhere; its compiles are counted
    through the public compile event (`dew.telemetry.instrumentation.Compiles`)."""
    assert reported_in_tests(lint, tmp_path, body, "SLOP021") is reported


def reported_in_tests(lint, tmp_path: Path, body: str, code: str) -> bool:
    """Whether lint_slop reports `code` in a test file holding `body`."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_linted.py").write_text(body)
    [module] = lint.collect(["tests"], tmp_path, "dew")
    return any(finding.code == code for finding in lint.check(module))
