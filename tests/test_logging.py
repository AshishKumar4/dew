"""Default Dew diagnostics reach stderr without configuring an application's root."""

import os
import subprocess
import sys

import pytest


def python(source):
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": "src"}, check=True, timeout=20)


def test_default_warning_reaches_stderr_once_and_a_pipe_is_plain():
    result = python("""
import logging
import dew
from dew.logging import configure
configure()
configure()
logging.getLogger('dew.data.sources').warning('missing rows [literal]')
logging.getLogger('dew.data.sources').info('hidden info')
assert len(logging.getLogger('dew').handlers) == 1
assert not logging.getLogger().handlers
""")
    assert result.stdout == ""
    assert result.stderr.count("missing rows [literal]") == 1
    assert "dew.data.sources" in result.stderr
    assert "WARNING" in result.stderr
    assert "hidden info" not in result.stderr
    assert "\x1b" not in result.stderr


def test_a_callers_existing_dew_handler_and_level_are_not_replaced():
    result = python("""
import logging
import sys
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('custom %(name)s %(message)s'))
logger = logging.getLogger('dew')
logger.addHandler(handler)
logger.setLevel(logging.INFO)
import dew
from dew.logging import configure
configure()
assert logger.handlers == [handler]
assert logger.level == logging.INFO
logging.getLogger('dew.data').info('visible info')
""")
    assert result.stdout == "custom dew.data visible info\n"
    assert result.stderr == ""


def test_a_callers_level_before_import_is_respected_without_a_custom_handler():
    result = python("""
import logging
logging.getLogger('dew').setLevel(logging.INFO)
import dew
logging.getLogger('dew.training').info('early verbosity')
""")
    assert "early verbosity" in result.stderr


def test_the_standard_logging_api_changes_the_public_level():
    result = python("""
import logging
import dew
logging.getLogger('dew').setLevel(logging.INFO)
logging.getLogger('dew.training').info('device count')
""")
    assert "device count" in result.stderr


@pytest.mark.parametrize("width", [30, 120])
def test_a_narrow_live_console_keeps_log_words_readable(width):
    result = python(f"""
import io
import logging
import dew
from rich.console import Console
from dew.logging import display_console
screen = io.StringIO()
with display_console(Console(file=screen, width={width}, force_terminal=True, color_system=None)):
    logging.getLogger('dew.training.test').warning('diagnostic above the live panel')
assert 'diagnostic' in screen.getvalue(), screen.getvalue()
""")
    assert result.stdout == result.stderr == ""


@pytest.mark.parametrize("exception", [False, True])
def test_the_live_console_is_restored_even_when_a_display_fails(exception):
    result = python(f"""
import io
import logging
import dew
from rich.console import Console
from dew.logging import display_console
screen = io.StringIO()
handler, = logging.getLogger('dew').handlers
original = handler.console
try:
    with display_console(Console(file=screen, force_terminal=True, color_system=None)):
        logging.getLogger('dew.training').warning('above the panel')
        if {exception!r}:
            raise ValueError('failed display')
except ValueError:
    pass
assert handler.console is original
assert 'above the panel' in screen.getvalue()
""")
    assert result.stdout == result.stderr == ""
