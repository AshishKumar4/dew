from functools import cache

from dew.interop import load_pretrained
from dew.sampling import CFG, DPMSolverMultistep, EulerAncestral, Heun, Sampling, TextToImage


@cache
def from_pretrained(repo_id):
    return TextToImage.from_pretrained(repo_id)


@cache
def text_model(name):
    model = load_pretrained(f"/opt/models/{name}", dtype="float32", max_seq_len=256)
    return model.text_generation(sampling=Sampling(temperature=0))


pipe = from_pretrained("dewml/hybrid-dit-176m")
