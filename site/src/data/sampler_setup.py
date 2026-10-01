from functools import cache

from PIL import Image

from dew.artifacts import uint8_pixels
from dew.interop import load_pretrained
from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, Sampling, TextToImage

pipe = TextToImage.from_run("/opt/models/text-to-image")


@cache
def text_model(name):
    model = load_pretrained(f"/opt/models/{name}", dtype="float32", max_seq_len=256)
    return model.text_generation(sampling=Sampling(temperature=0))
