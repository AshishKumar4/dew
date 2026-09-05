"""The surface of `video-reader-rs` that `dew.data.sources.av_utils` calls.

The package is optional and ships no type information, so this stub narrows
it to what dew reads: the reader over one file, its `get_info` mapping
(`frame_count`, `fps`, `width`, `height`, `duration`), and the two decoders,
`decode(start_frame, end_frame)` for a frame window and `decode_fast` for
the whole file at reduced fidelity. Both return uint8 frames `[T, H, W, 3]`.
"""

from typing import Any, Mapping

import numpy as np

class PyVideoReader:
    def __init__(self, filename: str, *args: Any, **kwargs: Any) -> None: ...
    def get_info(self) -> Mapping[str, Any]: ...
    def decode(self, start_frame: int | None = None, end_frame: int | None = None,
               compression_factor: float | None = None) -> np.ndarray: ...
    def decode_fast(self, start_frame: int | None = None, end_frame: int | None = None,
                    compression_factor: float | None = None) -> np.ndarray: ...
