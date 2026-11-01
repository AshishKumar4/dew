"""MuJoCo Playground's environment registry."""

from collections.abc import Mapping

from ._src.mjx_env import MjxEnv

def load(env_name: str, config: object = ..., config_overrides: Mapping[str, str | int | float | bool] | None = ...) -> MjxEnv: ...
