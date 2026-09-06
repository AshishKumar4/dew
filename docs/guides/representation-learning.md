# Train a representation model with JEPA

This guide assumes the [first training run](../getting-started.md) and basic image patches. JEPA trains an encoder by predicting representations of hidden image regions from visible context. The target is an encoded representation, not a pixel reconstruction or a next-token label.

## Run a small image example

This complete example uses synthetic images and trains three steps on CPU. It demonstrates construction and optimization; it does not establish useful representations.

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

The 16×16 images and 4×4 patches produce a 4×4 patch grid. The predictor's `grid` and encoder patch geometry must agree. The mask selects target regions on that grid; invalid area/aspect combinations can have no realizable block and raise an error.

The context encoder processes visible patches. The predictor uses context and target positions to estimate the hidden-region representations. The target encoder uses an exponential moving average of selected encoder variables, with gradients stopped through those targets. Account for the target copy in memory estimates.

## Interpret the training signals

Training loss measures prediction error in representation space. A low loss alone does not imply useful embeddings: collapsed representations can be similar for unrelated images. The objective reports representation statistics such as standard deviation and covariance diagnostics to help detect that failure mode.

For a downstream assessment, use held-out labeled examples and an appropriate probe. Supply validation data and an explicit `eval_every` cadence; metrics alone do not enable validation. [Evaluation and tracking](evaluation.md) explains artifact reduction and tracker behavior.

## Adapt the model and data

Replace the repeated synthetic batch with a dataset whose images match the declared sample shape. Set patch size and predictor grid together. Change the encoder and predictor widths deliberately; the predictor consumes the encoder's output width and has its own internal width.

The video objective uses temporal as well as spatial structure and requires compatible video tensors, patch geometry, masks, and predictors. Inspect the selected video recipe before reusing an image configuration.

JEPA is a representation-learning method. It does not by itself supply a planner, tool-use policy, reward function, or agent runtime. Those behaviors require separate training objectives and evaluation. General agentic post-training remains part of Dew's research and design work.
