"""Every decoder fixture with padded rows against the reference that wrote it: tools/hf_reference.py,
tools/mamba_reference.py, tools/ouro_reference.py, tools/k2_horizon_reference.py,
tools/minimax_reference.py and tools/qwen4_exp_reference.py write each directory's ids, its left-padded
rows and mask, the reference's logits for both and its greedy continuation, and the remote-code ones
their logits over check 1's seeds. A family's own checks stay in its own file."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import fixture_ids, transformers_arms, transformers_model
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / 'fixtures' / 'hf'

FIXTURES = {
    'phi-tiny': {}, 'gptj-tiny': {}, 'gpt-neo-tiny': {}, 'falcon-tiny': {}, 'falcon-mha-tiny': {},
    'gpt-bigcode-tiny': {}, 'gpt-bigcode-mha-tiny': {}, 'starcoder2-tiny': {}, 'mimo-v2-flash-tiny': {},
    'mamba-tiny': {}, 'ouro-tiny': {}, 'k2-horizon-tiny': {}, 'k2-horizon-mova-tiny': {}, 'minimax-tiny': {},
    'qwen4-exp-tiny': {},
    # BLOOM's ALiBi config states no context length.
    'bloom-tiny': {'max_seq_len': 32},
}
"""Each fixture directory, with what loading it needs beyond float32 and the reference kernel."""

SEEDS = 8
"""Each transformers family's declared draws of ids against transformers' padded and unpadded logits
(check 1), sized by `seeds_for` paired with Dew's logits on the CPU."""

RECORDED = ('k2-horizon-tiny', 'k2-horizon-mova-tiny', 'minimax-tiny', 'ouro-tiny')
"""The fixtures whose reference is a release's remote code, which the test cannot run: each one's logits
recorded over check 1's seeds in its recorded_logits.npz (`--recording` and `--mova` of
tools/k2_horizon_reference.py, `--recording` of tools/minimax_reference.py and of tools/ouro_reference.py),
whose declared count it reads."""


def load(name: str) -> Pretrained:
    return Pretrained.load(ROOT / name, dtype='float32', attention_impl='reference', **FIXTURES[name])


CHECK1_SHARDS = {
    "test_padded_and_unpadded_logits_and_the_greedy_continuation_match_the_reference[qwen4-exp-tiny]":
        {"seeds": 8, "per_task": 8},
}
"""The cells whose check-1 seeds run as map tasks (`check1_shards`), by node id, a literal the CI plan reads
as it reads a mark's (tools/armada/ci.py): Qwen4-Exp's logits leave its first eight seeds undecided on every
host and draw its second stage's 32, 69 s alone cold, 31 s of it compiling (armada at c27860888, 4 of 4
runs), where CI ran the cell 130 s (row 7820): each stage's seeds a task of 8."""


def fixtures() -> list:
    """Each fixture a cell, marked where `CHECK1_SHARDS` names its node."""
    test = "test_padded_and_unpadded_logits_and_the_greedy_continuation_match_the_reference"
    return [pytest.param(name, id=name, marks=[pytest.mark.check1_shards(**held)] if (
        held := CHECK1_SHARDS.get(f"{test}[{name}]")) else []) for name in sorted(FIXTURES)]


@pytest.mark.parametrize('name', fixtures())
def test_padded_and_unpadded_logits_and_the_greedy_continuation_match_the_reference(name):
    """Left-padded rows under their mask and the unpadded rows, each on par with the reference from
    float64 over drawn ids (check 1), with the reference's argmax, and four ids continued six tokens
    greedily through Dew's own cached generation. A family transformers carries runs in the test from
    its float64 twin; a recorded one's truth is Dew's own float64 run held to the recording's sketch."""
    import torch

    directory, loaded = ROOT / name, load(name)
    mask = np.load(directory / 'attention_mask.npy')
    draw = fixture_ids(directory)

    def logits(model, variables):
        def run(ids):
            padded = model.apply(variables, jnp.where(mask, ids, 0), attention_mask=mask)
            return {'padded_logits': padded[mask], 'logits': model.apply(variables, ids)}
        return run

    ours = logits(loaded.model, loaded.variables)
    if name in RECORDED:
        path = directory / 'recorded_logits.npz'
        seeds = declared_seeds(path)['logits']
        wide = logits(loaded.model.clone(dtype=jnp.float64), widened(loaded.variables))
        recorded, reference, truth = indexed_recording(
            recording(path), 'logits', seeds, truth_fn=lambda inputs: wide(inputs['input_ids']),
            inputs_of=lambda seed: {'input_ids': draw(seed)[0]}, leaves=('logits', 'padded_logits'))
        assert_as_exact(lambda index, inputs: ours(inputs['input_ids']), reference, recorded,
                        f'{name} logits', seeds=seeds, truth_fn=truth)
        ids = np.load(directory / 'input_ids.npy')
        expected = {'padded_logits': np.load(directory / 'padded_logits.npy')[mask],
                    'logits': np.load(directory / 'logits.npy')}
    else:
        # The fields its recorder (tools/hf_reference.py `write_classic_tiny`) gave the padded rows:
        # BLOOM's ALiBi reads its positions off the mask.
        valid = torch.from_numpy(mask)
        fields = {'attention_mask': valid, **({} if name == 'bloom-tiny' else {
            'position_ids': torch.from_numpy(np.maximum(np.cumsum(mask, -1) - 1, 0))})}

        def read(model, ids):
            padded = model(torch.where(valid, ids, 0), **fields, use_cache=False).logits
            return {'padded_logits': padded[valid], 'logits': model(ids, use_cache=False).logits}

        reference, truth = transformers_arms(transformers_model(directory), read)
        # The ids as JAX arrays, so the gate traces Dew's arm alone: a draw holding NumPy arrays runs each arm
        # once eagerly (`arm_forms._refuse_numpy_arms`), 953 programs and 19 s of a cold run (row 7820).
        assert_as_exact(ours, reference, lambda seed: tuple(map(jnp.asarray, draw(seed))), f'{name} logits',
                        seeds=SEEDS, truth_fn=truth)
        (ids,) = draw(0)
        expected = reference(ids)
    for rows, actual in jax.jit(ours)(jnp.asarray(ids)).items():
        np.testing.assert_array_equal(np.asarray(actual).argmax(-1), expected[rows].argmax(-1))

    ids = np.load(directory / 'input_ids.npy')
    continued = generate(loaded.model, loaded.variables, ids[:, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(continued.tokens), np.load(directory / 'generated.npy'))
