prompt = "a red fox in a snowy forest"
result = pipe(
    [prompt],
    seed=0,
    steps=15,
    sampler=DPMSolverMultistep(),
    guidance=CFG(5.0),
)
images = uint8_pixels(result.host().images)
Image.fromarray(images[0])
