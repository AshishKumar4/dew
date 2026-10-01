model = "HuggingFaceTB/SmolLM2-135M-Instruct"
task = text_model(model)
print(task("The capital of France is", 24, seed=0).text[0])
