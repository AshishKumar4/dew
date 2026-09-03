"""Reproductions for data findings 4, 5, 28."""
import os, json, tempfile, io, contextlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax

def verdict(tag, ok, detail):
    print(f"[{tag}] {'CONFIRMED' if ok else 'NOT REPRODUCED'}: {detail}")

root = tempfile.mkdtemp()
tokens = np.arange(1, 1001, dtype=np.uint16)
tokens.tofile(os.path.join(root, "train.bin"))
json.dump({"tokenizer": "byte", "vocab_size": 256, "dtype": "<u2"}, open(os.path.join(root, "meta.json"), "w"))

# 4. Missing val.bin silently validates on train.bin.
from dew.config import DataConfig
from dew.data.dataloaders import load_data
data = load_data(DataConfig(dataset=root, sequence_length=8, batch_size=4, worker_count=0, val_steps_per_epoch=1))
val_batch = next(iter(data["val"]()))["text"]
first_train_window = tokens[:9]
verdict("4 val-fallback", data["val_len"] == data["train_len"] and np.array_equal(val_batch[0], first_train_window),
        f"val_len={data['val_len']} == train_len={data['train_len']}; first val row == first train window: {np.array_equal(val_batch[0], first_train_window)}")

# 5. local batch is floor-divided; global_batch_size still reports the request.
from dew.data.dataloaders import get_token_dataset_grain
real_count = jax.process_count
jax.process_count = lambda: 8
try:
    d65 = get_token_dataset_grain(os.path.join(root, "train.bin"), os.path.join(root, "train.bin"), batch_size=65, seq_len=8, worker_count=0)
    d7 = get_token_dataset_grain(os.path.join(root, "train.bin"), os.path.join(root, "train.bin"), batch_size=7, seq_len=8, worker_count=0)
finally:
    jax.process_count = real_count
verdict("5 local-batch", d65["local_batch_size"] == 8 and d65["global_batch_size"] == 65 and d7["local_batch_size"] == 0,
        f"8 processes: batch 65 -> local {d65['local_batch_size']} (actual global 64) while global_batch_size reports {d65['global_batch_size']}; batch 7 -> local {d7['local_batch_size']}")

# 28. shuffle(seed).repeat(n) replays one permutation; repeat(n).shuffle(seed) reshuffles per epoch.
import grain.python as pygrain
base = pygrain.MapDataset.range(10)
a = list(base.shuffle(0).repeat(2))
b = list(base.repeat(2).shuffle(0))
same_epochs_a = a[:10] == a[10:]
same_epochs_b = b[:10] == b[10:]
print(f"   shuffle.repeat epochs: {a[:10]} | {a[10:]}")
print(f"   repeat.shuffle epochs: {b[:10]} | {b[10:]}")
verdict("28 packed-reshuffle", same_epochs_a and not same_epochs_b,
        "the packed loader's shuffle(seed).repeat(epochs) order (dataloaders.py:823-825) replays the same permutation every epoch")
