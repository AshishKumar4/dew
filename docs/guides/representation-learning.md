# Train a representation model with JEPA

This guide assumes you have done the [first training run](../getting-started.md) and know what image patches are. JEPA trains an encoder by predicting the representations of hidden image regions from the visible ones. The training target is an encoded representation. It is not a pixel reconstruction or a next-token label.

## Run a small image example

This example trains for three steps on synthetic images. It shows how to build and optimize the objective; three steps will not learn useful representations.

```python
import itertools

import jax
import numpy as np
import optax

from dew import Field, Trainer, models
from dew.data import Dataset
from dew.objectives.jepa import JepaObjective, multi_block_mask

rng = np.random.default_rng(0)
images = rng.integers(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
data = Dataset(train=lambda: itertools.repeat({"image": images}),
               val=None, records=8, batch=8)
encoder = models.JepaEncoder(patch_size=4, emb_features=32, num_layers=2, num_heads=2)
predictor = models.JepaPredictor(grid=(4, 4), emb_features=32,
                                 predictor_features=16, num_layers=1, num_heads=2)
mask = multi_block_mask((4, 4), num_targets=1, scale=(0.25, 0.25))
objective = JepaObjective(encoder, predictor, mask=mask,
                          sample=Field("image", (16, 16, 3)))
trainer = Trainer(objective, optax.adam(0.001), key=jax.random.key(0))
state = trainer.fit(data, steps=3, log_every=1)
assert int(state.step) == 3
assert state.ema is not None
print("Completed three JEPA training steps.")
```

The 16×16 images and 4×4 patches give a 4×4 grid of patches. The predictor's `grid` must match the encoder's patch geometry. The mask picks target regions on that grid. Some combinations of area and aspect ratio leave no block that fits, and the mask raises an error for them.

The context encoder reads the visible patches. The predictor takes the context and the target positions and estimates the representations of the hidden regions. The target encoder is an exponential moving average of the context encoder's variables, and gradients do not flow through the targets. The target copy takes memory, so count it when you estimate memory use.

## Interpret the training signals

The training loss is the prediction error in representation space. A low loss does not mean the embeddings are useful: if the representations collapse, unrelated images get similar embeddings and the loss can still be low. To help you catch that, the objective reports representation statistics: `repr_std`, the per-dimension standard deviation across the batch, and `repr_cov_offdiag`, the size of the off-diagonal covariance.

To judge the representations on a downstream task, use held-out labeled examples and a probe. Supply validation data and set `eval_every`; passing metrics alone does not turn validation on. [Evaluation and tracking](evaluation.md) explains how artifacts are reduced to metrics and how trackers behave.

## Adapt the model and data

Replace the repeated synthetic batch with a dataset whose images match the declared sample shape. Change the patch size and the predictor grid together. When you change the encoder and predictor widths, remember that the predictor reads the encoder's output width and also has its own internal width.

`JepaObjective` also trains on video when the sample field has shape `(T, H, W, C)`; `models.JepaVideoEncoder` is the matching encoder. Video uses time as well as space, so the video tensors, patch geometry, masks and predictor all have to agree. Read the video path of `recipes/jepa/train.py` before you reuse an image configuration.

JEPA only learns representations. It does not give you a planner, a tool-use policy, a reward function or an agent runtime; those need their own training objectives and evaluation. Agentic post-training in general is still part of Dew's research and design work.
