import numpy as np

blocks = []
try:
    while True:
        blocks.append(np.ones(16 << 20))
except MemoryError:
    print(f"held {len(blocks) * 128} MiB")
