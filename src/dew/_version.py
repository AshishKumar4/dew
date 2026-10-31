"""The release's version, the one line a release changes. `dew.__version__` reads it when it is first used
(`dew._exports.lookup`), so `import dew` runs no module that holds it: every test file's key in CI's test
cache covers the top level of each module `import dew` executes, and a release's bump here re-keys none of
them. pyproject.toml reads it from this file without importing it (`[tool.setuptools.dynamic]`), and the
release workflow's tag check does the same."""

__version__ = "0.2.0"
