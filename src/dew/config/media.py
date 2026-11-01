"""The batch field a run over images or clips reads (`media_field`), which the diffusion and JEPA
runs share: a module of its own, so a configuration that reads no media reads no conditions."""

from dew._records import record
from dew.data.pipeline import Dataset
from dew.inputs import Field


def media_field(chain: Dataset, run: str) -> Field:
    """The batch field a run over images or clips reads, at the size its chain makes it
    (`Dataset.facts`): a clip's video, else an image. Raises ValueError, naming the `run`, for a chain
    of neither, or one whose size varies."""
    fields = record(chain.facts()["fields"], "the chain's fields")
    name = "video" if "video" in fields else "image"
    shape = fields.get(name)
    if not isinstance(shape, list | tuple) or not all(isinstance(size, int) for size in shape):
        raise ValueError(f"a {run} run trains on images or clips of one size (dew.data.presets.images or "
                         f"clips); this chain's {name} is {shape!r}")
    return Field(name, tuple(shape))
