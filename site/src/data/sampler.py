pipe = from_pretrained("dewml/hybrid-dit-176m", revision="434d7e8a940a906ef5d422b31d12b965944a5562")
prompt = "a turquoise alpine lake surrounded by pine trees and rugged mountains"
result = pipe(
    [prompt],
    key=0,
    steps=15,
    solver=DPMSolverMultistep(),
    guidance=CFG(5.0),
)
result.pil()[0]
