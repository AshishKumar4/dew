"""SLOP013: what importing a module may not do to the process, and what it may."""

import textwrap

import pytest
from tool_support import load


@pytest.fixture(scope="module")
def lint():
    return load("lint_slop")


WRITE = 'jax.config.update("jax_enable_x64", True)'
CASES = {
    "a config write at import": (WRITE, True),
    "an environment write": ('os.environ["XLA_FLAGS"] = ""', True),
    "a global listener": ("jax.monitoring.register_event_listener(print)", True),
    "inside a module-level if": (f"if True:\n    {WRITE}", True),
    "inside except and finally": (f"try:\n    pass\nexcept ValueError:\n    {WRITE}\n"
                                  f"finally:\n    {WRITE}", True),
    "in a class body": (f"class Configured:\n    {WRITE}", True),
    "in the script guard's else": (f"if __name__ == '__main__':\n    pass\nelse:\n    {WRITE}", True),
    "under a negated script guard": (f"if __name__ != '__main__':\n    {WRITE}", True),
    "inside a function": (f"def when_asked():\n    {WRITE}", False),
    "inside a method": (f"class Configured:\n    def when_asked(self):\n        {WRITE}", False),
    "under the script guard": (f"if __name__ == '__main__':\n    {WRITE}", False),
    "in a function's default": (f"def when_asked(enabled={WRITE}):\n    return enabled", True),
    "in a decorator": (f"@({WRITE} or (lambda function: function))\ndef decorated():\n    pass", True),
    "in a class's base": (f"class Configured({WRITE} or object):\n    pass", True),
    "in a class's keyword": (f"class Configured(object, metaclass={WRITE} or type):\n    pass", True),
    "inside a lambda": (f"later = lambda: {WRITE}", False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_import_time_writes_are_reported_and_deferred_ones_are_not(lint, case, tmp_path):
    body, reported = CASES[case]
    (tmp_path / "src" / "dew").mkdir(parents=True)
    (tmp_path / "src" / "dew" / "effects.py").write_text(textwrap.dedent("import os\nimport jax\n\n" + body))
    [module] = lint.collect(["src/dew"], tmp_path, "dew")
    assert any(True for _ in lint.imports(module)) is reported
