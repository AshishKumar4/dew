from model_client import CFG, DPMSolverMultistep, from_pretrained

pipe = from_pretrained("dewml/hybrid-dit-176m", revision="fc23e794b5f50fd53a619d1b2e0f3f6c7ccf51e5")
prompt = "the northern lights over a frozen lake at night, vivid colors, dramatic lighting, highly detailed"
negative = ("letterbox, white border, black border, frame, text, watermark, collage, blurry, lowres, "
            "low quality, dull colors, washed out, low contrast, grainy")
inputs = pipe.prepare([prompt], key=3, steps=15, unconditional=negative)
result = pipe(
    inputs,
    key=3,
    steps=15,
    solver=DPMSolverMultistep(),
    guidance=CFG(6.0, interval=(0.15, 0.9)),
)
result.pil()[0]
