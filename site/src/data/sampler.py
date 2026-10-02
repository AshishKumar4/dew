pipe = from_pretrained("dewml/hybrid-dit-176m", revision="0964f57387afc938927b1047f19ed32b63fe0619")
prompt = "a turquoise alpine lake surrounded by pine trees and rugged mountains"
result = pipe(
    [prompt],
    key=0,
    steps=15,
    solver=DPMSolverMultistep(),
    guidance=CFG(5.0),
)
result.pil()[0]
