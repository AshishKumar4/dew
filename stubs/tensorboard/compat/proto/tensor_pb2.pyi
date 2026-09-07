from collections.abc import Iterable

from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

class TensorProto:
    def __init__(self, *, dtype: int = ..., tensor_shape: TensorShapeProto = ...,
                 string_val: Iterable[bytes] = ...) -> None: ...
