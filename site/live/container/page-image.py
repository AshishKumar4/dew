# The page's image cell, run by smoke-shared.py through the relay.
from dew.sampling import CFG, DPMSolverMultistep, TextToImage

pipe = TextToImage.from_pretrained('dewml/hybrid-dit-176m')
pipe(['a lake beneath the northern lights'], key=3, steps=15, solver=DPMSolverMultistep(), guidance=CFG(6, interval=(0.15, 0.9))).pil()[0]
