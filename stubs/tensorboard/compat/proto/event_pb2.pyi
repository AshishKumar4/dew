from tensorboard.compat.proto.summary_pb2 import Summary

class Event:
    def __init__(self, *, wall_time: float = ..., step: int = ...,
                 summary: Summary = ...) -> None: ...
