# What a training context may and may not do, run in one after the OOM probe (shared_bridge.py) by
# benchmark_gateway.py's training(), which checks what it prints and what it left outside.
import urllib.request
from pathlib import Path
try:
    urllib.request.urlopen("https://huggingface.co", timeout=5)
except OSError:
    pass
else:
    raise AssertionError("a training context reached the network")
# The prepared cache is an overlay of the context's own (gateway_manager.py); training() checks
# from outside that this write stayed in it. The overlay holds no more than 64 MiB: files of
# 20 MiB, under the 32 MiB a file may be (guest_limits.py), fill it until a write finds no space.
import errno
Path("/opt/train/escape").write_text("x")
written = 0
try:
    for k in range(5):
        Path(f"/opt/train/fill{k}").write_bytes(bytes(20 * 1024 * 1024))
        written += 20
except OSError as error:
    assert error.errno == errno.ENOSPC, f"filling the overlay failed otherwise than for space: {error}"
else:
    raise AssertionError("a training context wrote 100 MiB into its 64 MiB overlay")
assert 40 <= written <= 64, f"the overlay took {written} MiB"
for k in range(5):
    Path(f"/opt/train/fill{k}").unlink(missing_ok=True)
for path in ("/opt/live/escape", "/work/../escape"):
    try:
        Path(path).write_text("x")
    except OSError:
        pass
    else:
        raise AssertionError(f"a training context wrote {path}")
Path("/tmp/inside").write_text("x")
try:
    bytearray(7 * 1024 * 1024 * 1024)
except MemoryError:
    pass
else:
    raise AssertionError("a training context exceeded its memory allowance")
from dew import Trainer
from dew.config import OptimConfig
from dew.data import Hub, presets
from dew.nn.backbones import CausalTransformer
from dew.objectives.lm import LMObjective
# The lm cell's corpus, which the preparation compiled (prepare-cells.py), read with the Hub offline.
data = Hub("winglian/tiny-shakespeare").pipe(presets.lm, "byte", seq_len=64, batch=64)
model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=1, num_heads=2,
                          mlp_features=64, max_seq_len=128)
state = Trainer(LMObjective(model, seq_len=64), OptimConfig(learning_rate=1e-3), key=0).fit(data, steps=1000)
assert int(state.step) == 20
from dew.interop import PretrainedDecoder
from dew.sampling import Sampling
model = PretrainedDecoder.load("HuggingFaceTB/SmolLM2-135M-Instruct", dtype="float32", max_seq_len=256)
assert model.text_generation(sampling=Sampling(temperature=0))("ROMEO:", 4, key=0).text[0]
