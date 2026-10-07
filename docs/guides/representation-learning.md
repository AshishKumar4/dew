# JEPA

`JepaObjective` trains an image or video encoder with a joint-embedding predictive architecture (I-JEPA, V-JEPA): a predictor estimates the representations of masked regions from the visible ones, and the targets come from an exponential moving average of the encoder. The loss is measured in representation space, not on pixels or tokens.

## Example

Three training steps on synthetic images; this shows the API and does not learn useful representations.

```python
import itertools

import jax
import numpy as np

from dew import Field, Trainer
from dew.config import OptimConfig
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
trainer = Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
assert int(state.step) == 3
assert state.ema is not None
print("Completed three JEPA training steps.")
```

16×16 images with 4×4 patches give a 4×4 grid. The predictor's `grid` must match the encoder's patch grid, and `MultiBlockMask.for_grid` picks target blocks on that grid; a combination of `scale` and aspect ratio that leaves no block that fits raises an error.

| Part | Role |
|---|---|
| Context encoder (`encoder`) | Encodes the visible patches; trained by the optimizer |
| Predictor (`predictor`) | Estimates the representations of the target blocks from the context and the target positions |
| Target encoder | The EMA copy of the context encoder (`state.ema`); encodes the whole image, with no gradient |

The target encoder is a second copy of the encoder's weights and counts toward memory.

## Metrics

The loss is the prediction error in representation space. A low loss does not show that the representations are useful: collapsed representations, where unrelated images map to similar embeddings, can also have a low loss. The objective therefore logs two statistics of the representations: `repr_std`, the per-dimension standard deviation across the batch, and `repr_cov_offdiag`, the magnitude of the off-diagonal covariance.

A downstream result needs held-out labeled examples and a probe, run as validation with `eval_every` and `metrics`. [Evaluation and tracking](evaluation.md) describes how evaluation artifacts become metrics.

## Changing the model and data

The dataset's images must match the declared sample shape, and the patch size and predictor grid change together. The predictor reads the encoder's output width (`emb_features`) and has its own internal width (`predictor_features`).

`JepaObjective` trains on video when the sample field has shape `(T, H, W, C)`, with `JepaVideoEncoder` as the encoder. The video tensors, the spatio-temporal patch geometry, the masks and the predictor must agree; the video path of `recipes/jepa/train.py` is a complete configuration.

The [I-JEPA tutorial](../tutorials.md) trains an encoder on real images and probes it.
