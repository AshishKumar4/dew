pipe = from_pretrained("dewml/hybrid-dit-176m")
prompt = "a turquoise alpine lake surrounded by pine trees and rugged mountains"
result = pipe(
    [prompt],
    seed=0,
    steps=15,
    sampler=DPMSolverMultistep(),
    guidance=CFG(5.0),
)
result.pil()[0]
