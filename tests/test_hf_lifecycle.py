"""Hugging Face sources through Dew's public lifecycle, one case per fixture.

An untrained source exports, transformers 5.16.1 reads the export to Dew's
logits, and Dew reloads it to the same model, tree and logits. A causal
source takes one public `Trainer` step of `LMObjective`, which lowers its
loss, moves every module and keeps the state beside the parameters. The trained
exports are tests/test_decoder_export.py's and
tests/test_masked_diffusion_export.py's.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from interop_support import assert_reloads, assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference, float64_forward

from dew.data.dataset import Loader
from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.sources import load_shards
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training import Trainer

FIXTURES = Path(__file__).parent / 'fixtures' / 'hf'


@dataclass(frozen=True)
class Export:
    """One untrained source's export.

    `derived` writes it through `PretrainedDecoder.from_model`, under the
    config derived from the model, and otherwise the source's own `save`
    writes it. `writes` is the family a derived export saves as when it is
    not the source's: a multi-head GPTBigCode computes GPT-2's model, so it
    saves as GPT-2, in GPT-2's tensors. `bound` holds a fixture that has no
    float64 logits to an absolute fp32 bound and every argmax. `masked`
    reads the export as a masked LM, which takes no cache.
    """

    fixture: str
    derived: bool = False
    writes: str | None = None
    bound: float | None = None
    masked: bool = False

    @property
    def label(self) -> str:
        return self.fixture if self.derived or self.fixture not in DERIVED else f'{self.fixture}-saved'


# The sources a derived export writes, which their own `save` writes too, as themselves.
DERIVED = ('gpt-bigcode-tiny', 'gpt-bigcode-mha-tiny', 'starcoder2-tiny', 'falcon-tiny', 'falcon-mha-tiny',
           'gptj-tiny', 'phi-tiny', 'phi3-tiny', 'mamba-tiny')


EXPORTS = (
    Export('gpt-bigcode-tiny', derived=True),
    Export('gpt-bigcode-mha-tiny', derived=True, writes='gpt2'),
    Export('starcoder2-tiny', derived=True),
    Export('falcon-tiny', derived=True),
    Export('falcon-mha-tiny', derived=True),
    Export('gptj-tiny', derived=True),
    Export('phi-tiny', derived=True),
    Export('phi3-tiny', derived=True),
    Export('bloom-tiny'),
    # These fixtures have no float64 logits, so the dense-family fp32 bound.
    Export('gpt2-tiny', bound=1e-4),
    Export('gpt-neox-tiny', bound=1e-4),
    Export('opt-tiny', bound=1e-4),
    Export('gpt-neo-tiny'),
    Export('mimo-v2-flash-tiny'),
    Export('mamba-tiny', derived=True),
    Export('modernbert-tiny', masked=True),
    *(Export(name) for name in DERIVED),
)


@pytest.mark.parametrize('case', EXPORTS, ids=lambda case: case.label)
def test_an_untrained_export_reads_in_transformers_and_reloads_bit_for_bit(case, tmp_path):
    """The export names its family and holds the source's tensors byte for
    byte, unless it writes another family's. Transformers reads it on eager
    attention and eager experts, and Dew's logits are as exact as its under
    the float64 rule. Dew reads it back to the model, tree and logits it was
    saved from."""
    import torch
    import transformers

    directory = FIXTURES / case.fixture
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    exported = PretrainedDecoder.from_model(loaded.model, loaded.variables) if case.derived else loaded
    exported.save(tmp_path)
    family = case.writes or json.loads((directory / 'config.json').read_text())['model_type']
    assert json.loads((tmp_path / 'config.json').read_text())['model_type'] == family
    auto = transformers.AutoModelForMaskedLM if case.masked else transformers.AutoModelForCausalLM
    reference = auto.from_pretrained(tmp_path, attn_implementation='eager',
                                     experts_implementation='eager').float().eval()
    ids = np.load(directory / 'input_ids.npy')
    with torch.no_grad():
        expected = reference(torch.from_numpy(ids).long(),
                             **({} if case.masked else {'use_cache': False})).logits.numpy()
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    if case.bound is None:
        _, terms = float64_forward(loaded.model, loaded.variables, ids)
        assert_as_exact_as_the_reference(actual, expected, np.load(directory / 'logits_f64.npy'),
                                         f'{case.fixture} export', terms=terms)
    else:
        np.testing.assert_allclose(actual, expected, atol=case.bound, rtol=0)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    if case.writes is None:
        assert_same_stored_tensors(load_shards(tmp_path), load_shards(directory))
    assert_reloads(loaded, tmp_path, ids)


@dataclass(frozen=True)
class Training:
    """One causal source's training step.

    `state` names the collections the source loads beside `params`, such as
    MiniMax-M2's routing bias, which a step without `balance` leaves
    where the checkpoint put it. `images` trains on the fixture's prompts with
    their images, one on the first row and two on the second, so the vision
    tower and the projector train with the language model.
    """

    fixture: str
    state: tuple[str, ...] = ()
    images: bool = False


TRAINING = (
    Training('gpt-neo-tiny'),
    Training('phi3-tiny'),
    Training('falcon-tiny'),
    Training('bloom-tiny'),
    Training('qwen2-moe-tiny'),
    Training('granitemoe-tiny'),
    Training('minimax-m2-tiny', state=('moe',)),
    Training('qwen35-moe-native-tiny', images=True),
)


def training_rows(case: Training, loaded: Pretrained, directory: Path) -> ModelInputs:
    if not case.images:
        return ModelInputs(jnp.asarray(np.load(directory / 'input_ids.npy'), jnp.int32))
    images = np.load(directory / 'raw_images.npy')
    return loaded.processor(json.loads((directory / 'prompts.json').read_text()),
                            images=[[images[0]], [images[1], images[2]]])


@pytest.mark.parametrize('case', TRAINING, ids=lambda case: case.fixture)
def test_a_public_training_step_lowers_the_loss_and_moves_every_module(case):
    """One SGD step through `Trainer`, on the fixture's rows repeated over the
    devices. The committed tree scores the rows' shifted, attention-masked
    targets below the checkpoint; every parameter is finite and every
    top-level module moved (each layer, the embedding, the head and a
    wrapper's vision tower and projector); every other collection is the
    checkpoint's to the bit. A single leaf can rightly stay put. Under
    BLOOM's ALiBi a key bias adds one score to every key a query sees, which
    the softmax cancels, so its gradient is zero."""
    directory = FIXTURES / case.fixture
    loaded = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    assert set(loaded.variables) == {'params', *case.state}
    inputs = training_rows(case, loaded, directory)
    count = inputs.tokens.shape[0]
    rows = inputs.take_rows(jnp.arange(count * jax.device_count()) % count)
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1, ema_decay=None)
    data = Loader(train=lambda partition: iter([{'text': rows}]), val=None,
                  records=rows.tokens.shape[0], batch=rows.tokens.shape[0])
    started = jax.device_get(loaded.variables)
    state = Trainer(objective, optax.sgd(1e-3), key=jax.random.key(0)).fit(data, steps=1, log_every=1)
    trained = jax.device_get(state.variables)

    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)
    loss = jax.jit(lambda variables: objective.scalar_loss(variables, {'text': inputs}, step)[0])
    assert float(loss(trained)) < float(loss(started))
    assert set(trained) == set(started)
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(trained['params']))
    for module, tree in started['params'].items():
        moved = jax.tree.map(lambda before, after: not np.array_equal(before, after),
                             tree, trained['params'][module])
        assert any(jax.tree.leaves(moved)), f'{module} did not move'
    for name in case.state:
        jax.tree.map(np.testing.assert_array_equal, trained[name], started[name])
