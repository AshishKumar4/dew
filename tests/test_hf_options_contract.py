"""HF option values and the optional dependency's real failure boundary."""

import os
import subprocess
import sys


def test_importing_hf_source_without_the_optional_package_does_not_load_it():
    # A fresh interpreter makes the optional package genuinely unavailable
    # at the import boundary, rather than replacing a Dew module or object.
    code = """
import builtins
original = builtins.__import__
def absent(name, *args, **kwargs):
    if name == 'datasets':
        raise ModuleNotFoundError('datasets is absent')
    return original(name, *args, **kwargs)
builtins.__import__ = absent
from dew.data.sources.hf import HFOptions, HFDatasetSource
options = HFOptions(config='named')
source = HFDatasetSource(name='acme/records', options=options)
assert source.options is options
try:
    options.load(path='acme/records', split='train', streaming=False)
except ImportError as error:
    assert isinstance(error.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('loading a source without its optional dependency succeeded')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": "src"}, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
