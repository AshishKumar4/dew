prompt = "a red fox in a snowy forest"
result = pipe([prompt], seed=0, steps=15, sampler=DPMSolverMultistep(), guidance=CFG(5.0))
Image.fromarray(uint8_pixels(result.host().images)[0])
