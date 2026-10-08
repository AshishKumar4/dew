from dataclasses import replace

from dew import Trainer
from dew.config import OptimConfig
from dew.data import load
from dew.interop import PretrainedDecoder
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling

name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = PretrainedDecoder.load(name, dtype="float32", max_seq_len=256)
data = load("hf/winglian/tiny-shakespeare", batch=1, tokenizer=name, seq_len=64)
state = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-4), key=0).fit(
    data, steps=20, log_every=5)
before = model.text_generation(sampling=Sampling(temperature=0))
after = replace(model, variables=state.variables).text_generation(sampling=Sampling(temperature=0))
print("Before:", before("ROMEO:", 32, key=0).text[0])
print("After: ", after("ROMEO:", 32, key=0).text[0])
