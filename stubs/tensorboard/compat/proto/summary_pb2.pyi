from collections.abc import Iterable

from tensorboard.compat.proto.tensor_pb2 import TensorProto

class SummaryMetadata:
    class PluginData:
        def __init__(self, *, plugin_name: str = ...) -> None: ...
    def __init__(self, *, plugin_data: SummaryMetadata.PluginData = ...) -> None: ...

class HistogramProto:
    def __init__(self, *, min: float = ..., max: float = ..., num: float = ...,
                 sum: float = ..., sum_squares: float = ...,
                 bucket_limit: Iterable[float] = ...,
                 bucket: Iterable[float] = ...) -> None: ...

class Summary:
    class Image:
        def __init__(self, *, height: int = ..., width: int = ..., colorspace: int = ...,
                     encoded_image_string: bytes = ...) -> None: ...

    class Value:
        def __init__(self, *, tag: str = ..., metadata: SummaryMetadata = ...,
                     simple_value: float = ..., image: Summary.Image = ...,
                     histo: HistogramProto = ..., tensor: TensorProto = ...) -> None: ...

    def __init__(self, *, value: Iterable[Summary.Value] = ...) -> None: ...
