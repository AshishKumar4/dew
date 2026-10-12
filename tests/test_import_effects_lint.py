"""SLOP013: what importing a module may not do to the process, and what it may."""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "lint_slop.py"


@pytest.fixture(scope="module")
def lint():
    spec = importlib.util.spec_from_file_location("lint_slop_imports", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = """
    import logging
    import os
    import warnings

    import jax
    from absl import flags

    jax.config.update("jax_enable_x64", True)
    os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp"
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    try:
        logging.basicConfig()
    except ValueError:
        warnings.filterwarnings("ignore")
    jax.monitoring.register_event_listener(print)


    def when_asked():
        jax.config.update("jax_enable_x64", True)
        os.environ["XLA_FLAGS"] = ""
        jax.monitoring.register_event_listener(print)


    if __name__ == "__main__":
        logging.basicConfig()
"""


def test_import_time_writes_are_reported_and_called_ones_are_not(lint):
    source = textwrap.dedent(SOURCE)
    module = lint.Module(Path("src/dew/effects.py"), "src/dew/effects.py", source, ast.parse(source))
    lines = sorted(finding.line for finding in lint.imports(module))
    assert lines == [9, 10, 12, 14, 16, 17]
