# The language-model head

## Result

Scoring the vocabulary in four chunks instead of one takes the small causal
transformer's step from 82.977 ms to 82.017 ms and its peak allocation from
5.803 GiB to 4.721 GiB: 1.2% faster, 18.6% leaner. The loss is unchanged and
`train/token_accuracy` keeps both its value and its key, computed from a
running argmax across the chunks rather than a pass over the whole logits
tensor. The accuracy now costs nothing measurable: 49.71 ms of head work with
it and 49.71 ms without.

The time saving is small because the head's three matmuls are the floor and
chunking does not remove them. What chunking removes is the traffic around
them. The full-vocabulary path held an 8,192 by 50,304 float32 logits tensor,
1.57 GiB, and its softmax gradient at the same time; the chunked path holds
four 0.39 GiB tiles and one gradient tile.

## Fixed point

Measured at commit `d19b242` of `wave/lm-head-perf`, against `main` at
`124a347`. One NVIDIA GeForce RTX 4080, 16 GiB, driver 595.84. Python 3.12.13,
JAX and jaxlib 0.11.1, Flax 0.12.9, Optax 0.2.8. Every run started with
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` showing only
`gnome-remote-desktop-daemon`. Both sides of every comparison ran through the
same harness, `tools/benchmark_step.py` from this branch, with `PYTHONPATH`
pointed at the library under test, so the only difference between a before row
and an after row is `src/dew`.

## The step

| | ms/step | samples/s | peak GiB | XLA GFLOP/step | util % |
|---|---:|---:|---:|---:|---:|
| main, full vocabulary | 82.977 | 192.8 | 5.803 | 1,320.1 | 16.3 |
| this branch, 4 chunks | 82.017 | 195.1 | 4.721 | 2,068.8 | 25.9 |

The GFLOP column moves without any arithmetic changing. Chunking replaces one
cuBLAS custom call with four dots the compiler can count, so
`cost_analysis()` sees 2.069 TFLOP where it saw 1.320 TFLOP. The analytic
figure for this model is 3.406 TFLOP either way
([benchmark-parity.md](benchmark-parity.md)). It is more evidence that the
logged utilization is a compiler-visibility number, not a hardware one.

## The head in isolation

8,192 tokens (batch 16, sequence 512), width 768, vocabulary 50,304, bf16
states, fp32 head, fp32 loss, forward and backward, 50 repeats. One process
per row, because `peak_bytes_in_use` is a process high-water mark and a shared
process reports the first row's peak for every later one.

| variant | forward ms | forward+backward ms | peak delta GiB |
|---|---:|---:|---:|
| full vocabulary with argmax accuracy | 20.55 | 50.72 | 3.238 |
| 4 chunks, tiles kept | 18.04 | **49.71** | 2.157 |
| 4 chunks, tiles kept, no accuracy | 18.01 | 49.71 | 2.157 |
| 8 chunks, tiles kept | 18.56 | 50.82 | 2.020 |
| 4 chunks, tiles recomputed (`jax.checkpoint`) | 18.12 | 67.19 | 1.834 |

Read three things off it. The accuracy term is free, so the frozen metric did
not have to move to validation. Eight chunks are slower than four by 1.11 ms
and save 0.14 GiB, so four ship. Recomputing the tiles in the backward pass
costs 17.48 ms, which is a fourth pass of the head matmul, and saves 0.32
GiB; on one card that is the wrong trade, so the tiles are stored. MaxText
recomputes them behind a `custom_vjp`, which is the right trade when the
vocabulary and the device count are much larger and the tiles carry sharding
constraints.

## Rejected variants

| variant | number | why it was rejected |
|---|---:|---|
| 8 chunks | 50.82 ms head, vs 49.71 for 4 | slower, for 0.14 GiB |
| `jax.checkpoint` per tile | 67.19 ms head, vs 49.71 | 17.48 ms slower, for 0.32 GiB |
| accuracy moved to validation | not measured | unnecessary: the chunk-wise argmax is free, and moving it would have migrated the frozen `train/token_accuracy` key |

## Does the value move

Three checks, from tightest to loosest.

**The head's own loss, same inputs.** Bitwise identical: relative difference
0.00e+00 between the full pass and 4 chunks at 8,192 tokens. Gradients agree
to 2.54e-05 on the head matrix, inside the 1e-4 the contribution rules ask
for. The state gradient agrees to 4.48e-03, which is one bfloat16 unit in the
last place: it is a bf16 tensor on both sides, and the same comparison in
float32 gives 9.12e-08. Parameter gradients through the real backbone are held
to 1e-4 by `tests/test_chunked_cross_entropy.py`.

**One training step.** Same seed, same fixed batch, small preset: step 0's
loss differs by 5.06e-07 relative, inside 1e-5.

**Twenty training steps.** The trajectories drift to 2.87e-04 by step 18. That
is not the chunking. The control is main against itself with only the matmul
precision changed, `precision=None` (which is TF32 on this executable) against
`precision='highest'`: the same code diverges by 1.42e-04 over the same twenty
steps from the same seed. Adam divides by a running gradient magnitude, so a
last-bit difference in one step becomes a visibly different parameter in the
next, and any reformulation that changes a matmul's shape changes those last
bits. Run under `precision='highest'` on both sides, branch against main still
drifts, to 2.22e-04. So twenty-step agreement at 1e-5 is below this
hardware's reproducibility floor, and the honest statement is the first two
checks plus this control, not a tolerance the machine cannot hold.

| comparison | step 0 | worst over 20 steps |
|---|---:|---:|
| branch against main, TF32 | 5.06e-07 | 2.87e-04 |
| branch against main, `precision='highest'` | 4.38e-06 | 2.22e-04 |
| main against main, TF32 against `'highest'` | 9.61e-06 | 1.42e-04 |

## Reproduction

```bash
cd ~/Desktop/dew/.worktrees/lm-head
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # only the desktop

# The step, before and after. The harness is this branch's in both runs.
PYTHONPATH=~/Desktop/dew/src ~/Desktop/dew/.venv/bin/python tools/benchmark_step.py \
  --preset small --architectures causal_transformer --steps 50 \
  --json-out /tmp/lm-head/before.json
PYTHONPATH=$PWD/src ~/Desktop/dew/.venv/bin/python tools/benchmark_step.py \
  --preset small --architectures causal_transformer --steps 50 \
  --json-out /tmp/lm-head/after.json

# The head in isolation, one process per variant.
for v in baseline stored4 stored8 remat4 stored4-noacc; do
  PYTHONPATH=$PWD/src ~/Desktop/dew/.venv/bin/python /tmp/lm-head/head_loss.py $v
done

# Twenty steps from one seed, each side and each precision.
PYTHONPATH=~/Desktop/dew/src ~/Desktop/dew/.venv/bin/python /tmp/lm-head/loss_parity.py \
  --steps 20 --out /tmp/lm-head/losses-main.json
PYTHONPATH=$PWD/src ~/Desktop/dew/.venv/bin/python /tmp/lm-head/loss_parity.py \
  --steps 20 --out /tmp/lm-head/losses-branch.json
PYTHONPATH=~/Desktop/dew/src ~/Desktop/dew/.venv/bin/python /tmp/lm-head/loss_parity.py \
  --steps 20 --precision highest --out /tmp/lm-head/losses-main-fp32.json
PYTHONPATH=$PWD/src ~/Desktop/dew/.venv/bin/python /tmp/lm-head/loss_parity.py \
  --steps 20 --precision highest --out /tmp/lm-head/losses-branch-fp32.json
```

The two harness scripts live in `/tmp/lm-head`. `head_loss.py` times one
variant per process and prints a JSON row; `loss_parity.py` runs the small
preset's language model for a fixed number of steps from seed 0 on the
benchmark's fixed batch and writes every step's loss and token accuracy.

## What is left

The head's three matmuls are 40.5 ms of the 82.0 ms step at the measured 49.5
TFLOP/s TF32 ceiling for this shape, and chunking cannot remove them. Two
things could. A bfloat16 head would run them at the 103.1 TFLOP/s bf16 ceiling,
which is the largest single number left on this model, and it changes the
loss, so it needs a convergence decision rather than a tolerance
([benchmark-parity.md](benchmark-parity.md) records 27.8% for the PyTorch
equivalent). cuDNN attention is worth 8.8% on this preset and is a
configuration default, not new code.
