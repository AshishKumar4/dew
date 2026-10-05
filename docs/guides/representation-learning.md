# JEPA

`JepaObjective` trains an image or video encoder using I-JEPA or V-JEPA (joint-embedding predictive architectures). A predictor uses the visible regions to estimate representations of masked regions. The targets come from an exponential moving average of the encoder. The loss compares representations; it does not reconstruct pixels or tokens.

## Example

This example runs three training steps on synthetic images to show the API. It does not learn useful representations.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Field, Trainer
from dew.data import Dataset
from dew.objectives.jepa import JepaEncoder, JepaObjective, JepaPredictor, MultiBlockMask

rng = np.random.default_rng(0)
images = rng.integers(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
data = Dataset(train=lambda partition: itertools.repeat({"image": images}),
               val=None, records=8, batch=8)
encoder = JepaEncoder(patch_size=4, emb_features=32, num_layers=2, num_heads=2)
predictor = JepaPredictor(grid=(4, 4), emb_features=32,
                          predictor_features=16, num_layers=1, num_heads=2)
mask = MultiBlockMask.for_grid((4, 4), num_targets=1, scale=(0.25, 0.25))
objective = JepaObjective(encoder, predictor, mask=mask,
                          sample=Field("image", (16, 16, 3)))
trainer = Trainer(objective, optax.adam(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
assert int(state.step) == 3
assert state.ema is not None
print("Completed three JEPA training steps.")
```

16×16 images with 4×4 patches give a 4×4 grid. Set the predictor's `grid` to match the encoder's patch grid. `MultiBlockMask.for_grid` selects target blocks on this grid. It raises an error if no block fits the requested `scale` and aspect ratio.

| Part | Role |
|---|---|
| Context encoder (`encoder`) | Encodes the visible patches; trained by the optimizer |
| Predictor (`predictor`) | Estimates the representations of the target blocks from the context and the target positions |
| Target encoder | The EMA copy of the context encoder (`state.ema`); encodes the whole image, with no gradient |

The target encoder is a second copy of the encoder's weights and counts toward memory.

## Metrics

Unrelated images can map to similar embeddings and still give a low prediction loss. To check for this collapse, the objective logs `repr_std`, the per-dimension standard deviation across the batch, and `repr_cov_offdiag`, the magnitude of the off-diagonal covariance.

To measure downstream performance, use held-out labeled examples and a probe. Run it as validation with `eval_every` and `metrics`. [Evaluation and tracking](evaluation.md) describes how metrics use evaluation artifacts.

## Changing the model and data

The dataset's images must match the declared sample shape. If you change the patch size, change the predictor grid too. The predictor takes the encoder's output width (`emb_features`) and has its own internal width (`predictor_features`).

For video, give `JepaObjective` a sample field with shape `(T, H, W, C)` and use `JepaVideoEncoder`. The video tensors, spatio-temporal patch geometry, masks and predictor must agree. The video path in `recipes/jepa/train.py` provides a complete configuration.

The [I-JEPA tutorial](../tutorials.md) trains an encoder on real images and probes it.
