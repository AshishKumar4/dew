"""The tensorboard surface TensorBoardTracker reaches: the event-file writer
and the summary protos it fills.

tensorboard ships no py.typed, and its generated `*_pb2` modules build their
message classes at import time, so a checker sees an empty module. These
declarations follow tensorboard/compat/proto/*.proto and
tensorboard/summary/writer/event_file_writer.py, with the fields dew sets.
"""

from . import compat as compat
from . import summary as summary
