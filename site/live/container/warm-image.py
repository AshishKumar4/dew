# A context's image sample through the shared model, without JAX, run by benchmark_gateway.py's warmed().
import model_client

model_client.install()
from dew.sampling import CFG, DPMSolverMultistep, TextToImage

pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')
result = pipe(['a lake beneath the northern lights'], key=3, steps=15, solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9)))
assert result.pil()[0].size == (256, 256)
import sys

assert 'jax' not in sys.modules
