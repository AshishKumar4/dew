from dew.sampling import CFG, DPMSolverMultistep, TextToImage

pipe = TextToImage.from_pretrained("dewml/hybrid-dit-176m", revision="73b7e9ed118eaa46fb7f199e688334bb37216cdd")
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
