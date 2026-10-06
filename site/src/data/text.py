from model_client import text_model

model = "HuggingFaceTB/SmolLM2-135M-Instruct"
task = text_model(model)
print(task("The capital of France is", 24, key=0).text[0])
