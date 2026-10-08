from dew.interop import PretrainedDecoder
from dew.sampling import Sampling

model = PretrainedDecoder.load("HuggingFaceTB/SmolLM2-135M-Instruct",
                               dtype="float32", max_seq_len=256)
task = model.text_generation(sampling=Sampling(temperature=0))
print(task("The capital of France is", 24, key=0).text[0])
