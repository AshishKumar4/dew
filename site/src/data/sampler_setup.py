from PIL import Image

from dew.artifacts import uint8_pixels
from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, TextToImage

pipe = TextToImage.from_run("/opt/models/text-to-image")
