"""HyperLux's camera outputs `dew.rl.hyperlux` asks for."""

from collections.abc import Sequence

class RGB:
    def __init__(self, layout: str = ...) -> None: ...

class CameraSpec:
    def __init__(self, name: str, outputs: Sequence[RGB]) -> None: ...
