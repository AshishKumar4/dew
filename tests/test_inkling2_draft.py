"""Inkling 2's multi-token-prediction draft (the release's `mtp_config` and `mtp.safetensors`, row 12945) as
Dew's Markov block drafter (`dew.nn.block_draft`), against vLLM's draft ported to torch
(tools/inkling2_draft_reference.py), which the suite cannot run, over transformers' own Inkling 2 decoder's
taps: tools/inkling2_fixture.py --draft writes the tiny draft beside tests/fixtures/hf/inkling2-tiny's decoder
and records both there (recorded_draft.npz)."""

import dataclasses
import json
import os
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import attending
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording, widened
from speculative_support import assert_replay_drafts_as_a_prefill, block_draw, greedy_walk

from dew.batch import coordinates as laid_out
from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config
from dew.nn.block_draft import BlockDrafter, Context, Draft, DraftAttention, MarkovBlockDraft, draft
from dew.nn.inputs import ModelInputs
from dew.nn.kv_cache import KVCache
from dew.sampling import Sample, Sampling, Speculative, generate

TINY = Path(__file__).parent / 'fixtures' / 'hf' / 'inkling2-tiny'
RECORDED = TINY / 'recorded_draft.npz'
RELEASE = json.loads((TINY / 'config.json').read_text())
SHAPE = (2, 40)
"""The ids' shape the recorder drew: the context, the anchor drawn after it."""
SPLIT = 24
"""Where the cached draft's context is cut: written in two calls, the second appending to the first."""
ACCEPTED = (5, 1)
"""Each row's tokens of the second call a verification accepted, the rest padding (a block's replay)."""


def draw(seed):
    """A recorded seed's inputs, as the recorder drew them (tools/inkling2_fixture.py `draft_draw`): ids, each
    row's anchor and the token fed back at each drafted place, uniform over the head's rows."""
    rng = np.random.default_rng(seed)
    vocabulary, block = RELEASE['text_config']['unpadded_vocab_size'], RELEASE['mtp_config']['block_size']
    return {'input_ids': rng.integers(0, vocabulary, SHAPE).astype(np.int32),
            'anchor': rng.integers(0, vocabulary, (SHAPE[0],)).astype(np.int32),
            'forced': rng.integers(0, vocabulary, (SHAPE[0], block - 1)).astype(np.int32)}


def context_of(model, variables, ids):
    """The draft's context over `ids`: the forward's taps (`draft_context`)."""
    _, state = model.apply(variables, ids, mutable=['prediction_inputs'])
    return model.apply(variables, state['prediction_inputs'], method='draft_context')


def forcing(forced):
    """A choice of each drafted place's token: the drawn one, fed back at the next place."""
    return lambda index, logits: forced[:, index]


def drafted(model, variables, inputs, valid=None):
    """The taps over the drawn ids and the draft's logits after them, uncached, over the context's real
    tokens (`valid`, every one where None)."""
    context = context_of(model, variables, inputs['input_ids'])
    coordinates = None if valid is None else laid_out(valid.shape, valid=valid)
    draft = model.apply(variables, context, inputs['anchor'], decode=False, choose=forcing(inputs['forced']),
                        coordinates=coordinates, method='draft')
    return {'context': context, 'logits': draft.logits}


def accepted():
    """The real tokens of a context whose `SPLIT` first are a prefill's and whose next are a replayed
    block's, each row's `ACCEPTED` of them, the rest padding."""
    kept = SPLIT + jnp.asarray(ACCEPTED)[:, None]
    return jnp.arange(SHAPE[1])[None, :] < kept


def cached(model, variables, inputs):
    """The draft's logits after its context written to its cache in two calls, the prefill's (`SPLIT`) then
    a block's whose rows a verification accepted apart (`accepted`), drafted from it."""
    context = context_of(model, variables, inputs['input_ids'])
    _, opened = model.apply(variables, SHAPE[0], method='init_draft_cache', mutable=['cache'])
    cache = opened['cache']
    valid = accepted()
    for part in (slice(0, SPLIT), slice(SPLIT, SHAPE[1])):
        _, written = model.apply({**variables, 'cache': cache}, context[:, part], None,
                                 coordinates=laid_out(valid[:, part].shape, valid=valid[:, part]),
                                 method='draft', mutable=['cache'])
        cache = written['cache']
    draft, _ = model.apply({**variables, 'cache': cache}, None, inputs['anchor'],
                           choose=forcing(inputs['forced']), method='draft', mutable=['cache'])
    return draft.logits


@pytest.fixture(scope='module')
@attending('reference')
def source():
    return Pretrained.load(TINY, dtype='float32')


@attending('reference')
@pytest.mark.parametrize('leaf', ['context', 'logits'])
@pytest.mark.parametrize(('site', 'dtype'), [('draft', 'float32'), ('draft_bf16', 'bfloat16')])
def test_its_taps_and_draft_are_on_par_with_the_releases(source, site, dtype, leaf):
    """The release-layout load's taps (each tapped visit's partial block, a looped layer's at its pass,
    and the final states: `context`) and its draft's logits after them, each place fed the drawn token
    before it (`logits`), in fp32 and in bf16 (the release's own dtype, where every rounding point of the
    draft's arithmetic shows: its fused adds and norms, its attention's scores and weights, its taps'
    weights, its logits' sum before their cap), on par with transformers' taps and vLLM's draft over them
    at that dtype from their float64 run's (check 1, the recording's), Dew's own float64 run the truth."""
    model, variables = source.model, source.variables
    narrow = source if dtype == 'float32' else Pretrained.load(TINY, dtype=dtype)
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    truth_of = jax.jit(lambda inputs: drafted(wide, held, inputs)[leaf])
    seeds = declared_seeds(RECORDED)[site]
    recorded, reference, truth = indexed_recording(
        recording(RECORDED), site, seeds, truth_fn=lambda inputs: {leaf: truth_of(inputs)},
        inputs_of=draw, leaves=(leaf,))
    run = jax.jit(lambda inputs: drafted(narrow.model, narrow.variables, inputs)[leaf])
    assert_as_exact(lambda index, inputs: {leaf: run(inputs)}, reference, recorded,
                    f'Inkling 2 draft ({dtype})', seeds=seeds, truth_fn=truth)


@attending('reference')
def test_a_cached_draft_after_a_partly_accepted_block_is_on_par_with_its_whole_context(source):
    """The draft after its context written to its cache in two calls, the second a replayed block each row
    of which a verification accepted apart (five tokens and one, the rest padding), each row's places
    counted on and its Markov head's taps its last accepted token's, on par with the draft over that whole
    context uncached (check 1, the uncached in float64 the truth)."""
    model, variables = source.model, source.variables
    wide, held = model.clone(dtype=jnp.float64), widened(variables)
    whole = jax.jit(lambda variables, inputs: drafted(model, variables, inputs, accepted())['logits'])
    truth = jax.jit(lambda _, inputs: drafted(wide, held, inputs, accepted())['logits'])
    assert_as_exact(jax.jit(lambda variables, inputs: cached(model, variables, inputs)), whole,
                    lambda seed: (variables, draw(seed)), 'Inkling 2 cached draft', seeds=8, truth_fn=truth)


@attending('reference')
def test_a_replayed_block_drafts_as_a_prefill_of_its_history(source):
    """A speculative block's verified stretch replayed into the cache saved before it and recorded into the
    Markov drafter's windows, its rows keeping four tokens and fewer, drafts its next block as a fresh
    prefill of each row's history does (row 13317; `speculative_support`)."""
    walk = greedy_walk(source.model, source.variables)

    def drawn(seed):
        return block_draw(walk, draw(seed)['input_ids'], 20, 4, np.random.default_rng([0x1331, seed]))

    assert_replay_drafts_as_a_prefill(source.model, source.variables, drawn,
                                      "Inkling 2's replayed block's draft", seeds=8)


def test_a_windows_ring_holds_each_slots_latest_place_below_its_count():
    """A local draft layer's ring of `window` slots (`block_draft.ring_places`) holds at each slot the latest
    place below its row's count that lands there, read off the count alone, and no place where the count
    reached none of the slot's: a row of none, one, a window and a half and two windows of a ring of 4."""
    from dew.nn.block_draft import ring_places

    held = ring_places(jnp.asarray([0, 1, 6, 8], jnp.int32), 4)
    np.testing.assert_array_equal(np.asarray(held) >= 0, [[False] * 4, [True, False, False, False],
                                                           [True] * 4, [True] * 4])
    np.testing.assert_array_equal(np.asarray(held)[1:], [[0, -3, -2, -1], [4, 5, 2, 3], [4, 5, 6, 7]])


def test_a_local_draft_layer_caches_its_window_alone(source):
    """A local draft layer's cached context holds its window's places alone (row 13541): its stores are
    `window` slots long where a global layer's hold the decoder's capacity, as the cached draft after a
    context longer than its window matches the uncached one (`cached`, 40 places over a window of 16)."""
    model, variables = source.model, source.variables
    _, opened = model.apply(variables, SHAPE[0], method='init_draft_cache', mutable=['cache'])
    drafter = opened['cache']['block_drafter']
    spec = RELEASE['mtp_config']
    for index in range(spec['num_hidden_layers']):
        held = drafter[f'layers_{index}']['self_attn']['held_context']
        lengths = {held[name].shape[1] for name in ('cached_key', 'cached_value')}
        expected = spec['sliding_window_size'] if index in spec['local_layer_ids'] else model.max_seq_len
        assert lengths == {expected}, (index, {name: leaf.shape for name, leaf in held.items()})


def test_a_local_draft_layers_window_is_its_own_under_a_paged_target(source):
    """A target whose cache is paged (pages of 64 slots) opens its draft's cache as the pages it names for
    its global layers and a window's ring unpaged for its local ones, of 16 slots no page divides (row
    13541): the cached draft is opened and drafts."""
    import dataclasses

    from dew.nn.kv_cache import KVCache

    model = source.model.clone(kv_cache=dataclasses.replace(source.model.kv_cache or KVCache(), page_size=64))
    _, opened = model.apply(source.variables, SHAPE[0], method='init_draft_cache', mutable=['cache'])
    spec = RELEASE['mtp_config']
    for index in spec['local_layer_ids']:
        held = opened['cache']['block_drafter'][f'layers_{index}']['self_attn']['held_context']
        assert held['cached_key'].shape[1] == spec['sliding_window_size']
    inputs = draw(0)
    context = context_of(model, source.variables, inputs['input_ids'])
    _, written = model.apply({**source.variables, 'cache': opened['cache']}, context, None, method='draft',
                             mutable=['cache'])
    drafted = model.apply({**source.variables, 'cache': written['cache']}, None, inputs['anchor'],
                          method='draft', mutable=['cache'])[0]
    assert np.all(np.isfinite(np.asarray(drafted.logits)))


def test_a_count_of_passes_is_refused_where_the_taps_are_recorded(source):
    """A forward recording the draft's taps over fewer passes than the loop's (`loop_passes`) is refused:
    its taps name visits in the loop's order, which a skipped pass would leave zeros in place of."""
    ids = jnp.asarray(draw(0)['input_ids'])
    source.model.apply(source.variables, ids, loop_passes=jnp.int32(2))
    with pytest.raises(ValueError):
        source.model.apply(source.variables, ids, loop_passes=jnp.int32(2), mutable=['prediction_inputs'])


def test_a_load_keeping_no_prediction_depths_reads_no_draft(tmp_path):
    """`prediction_depths=0` loads the decoder alone: no draft is built nor its `mtp.safetensors` read, so
    a copy of the release without that file loads, where the whole release builds the draft."""
    for name in ('config.json', 'model.safetensors'):
        shutil.copy(TINY / name, tmp_path / name)
    assert Pretrained.load(TINY, dtype='float32', weights=False).model.block_draft is not None
    assert Pretrained.load(tmp_path, dtype='float32', prediction_depths=0).model.block_draft is None


@attending('reference')
def test_the_markov_heads_packed_projection_writes_its_live_kernels(source):
    """The Markov head's `rnn_proj` an export writes packs its step's and its context's live kernels back
    (`Packed`, row 13538): one weight of each changed, the written tensor is the release's with exactly those
    two entries changed, at the step's columns (the first 2r) and the context's (after them), its bias the
    context's."""
    from safetensors.numpy import load_file

    # The loader's own layouts of the packed tensors (`decoder_bindings`), which an export writes.
    layouts = {layout.name: layout for layout in source.weight_layouts}
    head = source.variables['params']['block_drafter']['markov_head']
    rank = RELEASE['mtp_config']['markov_rank']
    step = {'kernel': head['rnn_step']['kernel'].at[1, 2].add(1.0)}
    context = {**head['rnn_context'], 'kernel': head['rnn_context']['kernel'].at[3, 4].add(1.0)}
    drafter = {**source.variables['params']['block_drafter'],
               'markov_head': {**head, 'rnn_step': step, 'rnn_context': context}}
    changed = {**source.variables, 'params': {**source.variables['params'], 'block_drafter': drafter}}
    given = load_file(TINY / 'mtp.safetensors')
    expected = np.asarray(given['markov_head.rnn_proj.weight'], np.float32).copy()
    expected[2, 1] += 1.0
    expected[4, 2 * rank + 3] += 1.0
    weight, bias = (layouts[f'mtp.markov_head.rnn_proj.{leaf}'] for leaf in ('weight', 'bias'))
    np.testing.assert_array_equal(np.asarray(weight.export(changed), np.float32), expected)
    np.testing.assert_array_equal(np.asarray(bias.export(changed), np.float32),
                                  np.asarray(given['markov_head.rnn_proj.bias'], np.float32))


def test_a_release_writes_each_sidecars_tensors_to_its_own_file(tmp_path):
    """A release whose family keeps more of its tensors beside its weights (`Placed.sidecars`, Inkling 2's
    `mtp.safetensors`) writes each sidecar's tensors to its own file under their own names, the prefix
    off, none of them in the root's shards, as a load reads them back (`sources.with_sidecars`) (row
    13538); the rest in the shards as before."""
    from safetensors.numpy import load_file

    from dew.interop.releases import Placed, ReleasePlan

    trunk, head = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(4, dtype=np.float32)
    tensors = {'model.weight': trunk, 'mtp.fuse_weight': head}
    plan = ReleasePlan(Placed({'': ({'model_type': 'tiny'}, tensors)}, sidecars={'mtp.safetensors': 'mtp.'}))
    plan.write(tmp_path)
    side = load_file(tmp_path / 'mtp.safetensors')
    assert list(side) == ['fuse_weight']
    np.testing.assert_array_equal(side['fuse_weight'], head)
    shards = {name: tensor for path in tmp_path.glob('model*.safetensors') for name, tensor in
              load_file(path).items()}
    assert list(shards) == ['model.weight']
    np.testing.assert_array_equal(shards['model.weight'], trunk)


def test_a_release_whose_trunk_fails_to_write_leaves_its_draft_as_it_was(tmp_path):
    """A save over a release that fails while it writes the trunk's shards leaves the release it would
    replace whole, its sidecar the earlier draft beside the earlier trunk, none of the failed save's
    files beside them (train 1349's review F3; row 13538)."""
    from safetensors.numpy import load_file

    from dew.interop.releases import Placed, ReleasePlan
    from dew.interop.safetensors_io import LazyTensors, tensor_specs

    def plan(tensors):
        return ReleasePlan(Placed({'': ({'model_type': 'tiny'}, tensors)},
                                  sidecars={'mtp.safetensors': 'mtp.'}))

    trunk, head = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(4, dtype=np.float32)
    plan({'model.weight': trunk, 'mtp.fuse_weight': head}).write(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())
    changed = {'model.weight': trunk + 1, 'mtp.fuse_weight': head + 1}

    def read(name):
        if name == 'model.weight':
            raise OSError("the trunk's storage went away")
        return changed[name]

    with pytest.raises(OSError):
        plan(LazyTensors(tensor_specs(changed), read)).write(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    np.testing.assert_array_equal(load_file(tmp_path / 'mtp.safetensors')['fuse_weight'], head)
    np.testing.assert_array_equal(load_file(tmp_path / 'model.safetensors')['model.weight'], trunk)


def test_a_release_replaces_only_entries_of_the_kind_it_writes(tmp_path):
    """A release replaces an earlier tree's entries only as the same kind of entry (train 1399's review):
    an empty folder named as a file it writes, or as a shard of its weights, is refused, kept; a FIFO where
    the earlier config was is a special entry, refused, never opened; and the draft an earlier Inkling 2
    release's config declared is the export's own to drop."""
    from dew.interop.releases import Placed, ReleasePlan

    trunk, head = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(4, dtype=np.float32)
    parted = ReleasePlan(Placed({'': ({'model_type': 'tiny'}, {'model.weight': trunk})}))
    # An empty folder named as a file the export writes, or as a shard of its weights, is no file of its
    # own: refused, kept (train 1399's review).
    for name in ('config.json', 'model.safetensors'):
        target = tmp_path / f'with-folder-{name}'
        parted.write(target)
        (target / name).unlink()
        (target / name).mkdir()
        with pytest.raises(FileExistsError):
            parted.write(target)
        assert (target / name).is_dir()
    # Its read waited on a writer under the publication's lock.
    target = tmp_path / 'with-fifo'
    parted.write(target)
    (target / 'config.json').unlink()
    os.mkfifo(target / 'config.json')
    (target / 'stale.safetensors').write_bytes(b'')
    with pytest.raises(FileExistsError):
        parted.write(target)
    # A release whose config declared Inkling 2's draft, replaced by one without it: the draft the earlier
    # release's config declared is the export's own to drop (train 1399's review).
    drafted_config = {'model_type': 'inkling2_loop', 'mtp_config': {'num_nextn_predict_layers': 1}}
    drafted = tmp_path / 'drafted'
    ReleasePlan(Placed({'': (drafted_config, {'model.weight': trunk, 'mtp.fuse_weight': head})},
                       sidecars={'mtp.safetensors': 'mtp.'})).write(drafted)
    assert (drafted / 'mtp.safetensors').is_file()
    ReleasePlan(Placed({'': ({'model_type': 'inkling2_loop'}, {'model.weight': trunk})})).write(drafted)
    assert not (drafted / 'mtp.safetensors').exists()


def test_a_release_is_one_generation_whatever_file_its_save_fails_at(tmp_path):
    """A release's save builds the whole release apart and puts it in place at once (row 13730): one
    that fails at its sidecar, as one that fails at its trunk, leaves the earlier release whole, byte for
    byte, and nothing of its own beside it; one into a directory not yet made, or emptied, is put in
    place whole; and a directory holding an entry no export writes, at its top or deeper (a file in a
    folder the export writes, weights in a folder it writes none of or of a stem it writes none of there, a
    link to a folder, an empty folder), or that is a file or a link itself, is refused, the release beside
    it as it was; an earlier export's release record, which a release stating none drops, is not."""
    from dew.interop.release_record import RELEASE_FILE
    from dew.interop.releases import Placed, ReleasePlan
    from dew.interop.safetensors_io import LazyTensors, tensor_specs

    def plan(tensors):
        return ReleasePlan(Placed({'': ({'model_type': 'tiny'}, tensors)},
                                  sidecars={'mtp.safetensors': 'mtp.'}))

    def snapshot(directory):
        return {path.name: path.read_bytes() for path in directory.iterdir()}

    trunk, head = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(4, dtype=np.float32)
    release = tmp_path / 'release'
    plan({'model.weight': trunk, 'mtp.fuse_weight': head}).write(release)
    before, around = snapshot(release), sorted(path.name for path in tmp_path.iterdir())
    changed = {'model.weight': trunk + 1, 'mtp.fuse_weight': head + 1}
    for failing in changed:
        def read(name, failing=failing):
            if name == failing:
                raise OSError(f"{failing}'s storage went away")
            return changed[name]

        with pytest.raises(OSError):
            plan(LazyTensors(tensor_specs(changed), read)).write(release)
        assert snapshot(release) == before, failing
        assert sorted(path.name for path in tmp_path.iterdir()) == around, failing
    fresh = tmp_path / 'fresh' / 'release'
    plan(changed).write(fresh)
    plan(changed).write(release)
    assert snapshot(fresh) == snapshot(release) != before
    (release / 'notes.txt').write_text('mine')
    after = snapshot(release)
    with pytest.raises(FileExistsError):
        plan({'model.weight': trunk, 'mtp.fuse_weight': head}).write(release)
    assert snapshot(release) == after
    # Deeper entries of its own: a file in a folder the next export writes too, weights in a folder it
    # writes none of, and a link to a folder, none followed: each refused, kept.
    vocabulary = tmp_path / 'vocab.txt'
    vocabulary.write_text('a b')
    parted = ReleasePlan(Placed({'': ({'model_type': 'tiny'}, {'model.weight': trunk}),
                                 'transformer': ({'model_type': 'tiny'}, {'model.weight': trunk})},
                                files={'transformer/tokenizer/vocab.txt': vocabulary}))
    for entry in ('transformer/notes.txt', 'backup/model.safetensors', 'linked',
                  'transformer/tokenizer/model.safetensors',
                  'transformer/diffusion_pytorch_model.safetensors'):
        target = tmp_path / entry.replace('/', '-')
        target.mkdir()
        parted.write(target)
        if entry == 'linked':
            (target / entry).symlink_to(tmp_path / 'fresh', target_is_directory=True)
        else:
            (target / entry).parent.mkdir(exist_ok=True)
            (target / entry).write_text('mine')
        with pytest.raises(FileExistsError):
            parted.write(target)
        assert (target / entry).exists(), entry
    # An empty folder no export writes is an entry of its own, refused, kept (train 1383's review: one was
    # dropped unlisted); the release record an earlier export wrote is the writer's own, dropped by a
    # release that states none (`Release.write`).
    target = tmp_path / 'with-backup'
    parted.write(target)
    (target / 'backup').mkdir()
    with pytest.raises(FileExistsError):
        parted.write(target)
    assert (target / 'backup').is_dir()
    (target / 'backup').rmdir()
    (target / RELEASE_FILE).write_text('{"step": 3}')
    parted.write(target)
    assert not (target / RELEASE_FILE).exists()
    # A destination that is a file of its own, or a link to a release: refused, kept as it was.
    (tmp_path / 'a-file').write_text('mine')
    (tmp_path / 'a-link').symlink_to(fresh, target_is_directory=True)
    for name in ('a-file', 'a-link'):
        with pytest.raises(FileExistsError):
            parted.write(tmp_path / name)
    assert (tmp_path / 'a-file').read_text() == 'mine' and (tmp_path / 'a-link').resolve() == fresh



def test_a_release_written_by_a_rank_that_publishes_none_leaves_its_sidecar_as_it_was(tmp_path):
    """A pool's rank that reads its share of a release and writes none of it (`reading_home(writes=False)`,
    beside the one that publishes) leaves the release's sidecar as the publisher wrote it: no file of its
    own is staged or published over it (train 1360's review of theme 61)."""
    from safetensors.numpy import load_file

    from dew._coordination import reading_home
    from dew.interop.releases import Placed, ReleasePlan

    def plan(tensors):
        return ReleasePlan(Placed({'': ({'model_type': 'tiny'}, tensors)},
                                  sidecars={'mtp.safetensors': 'mtp.'}))

    trunk, head = np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(4, dtype=np.float32)
    plan({'model.weight': trunk, 'mtp.fuse_weight': head}).write(tmp_path)
    with reading_home("a rank beside the publisher", writes=False):
        plan({'model.weight': trunk + 1, 'mtp.fuse_weight': head + 1}).write(tmp_path)
    np.testing.assert_array_equal(load_file(tmp_path / 'mtp.safetensors')['fuse_weight'], head)


def test_speculative_decoding_with_its_draft_emits_the_greedy_walk(source):
    """At zero temperature every rejected draft is replaced by the target's own token, so Speculative
    decoding with the draft's blocks emits the greedy walk, a left-padded row included."""
    ids = jnp.asarray(draw(0)['input_ids'])[:, :SPLIT]
    valid = jnp.ones(ids.shape, bool).at[1, :3].set(False)
    inputs = ModelInputs(jnp.where(valid, ids, 0), {'attention_mask': valid})
    walked, speculated = (generate(source.model, source.variables, inputs, 6, key=jax.random.key(0),
                                   sampling=Sampling(temperature=0, strategy=strategy))
                          for strategy in (Sample(), Speculative(block=4)))
    np.testing.assert_array_equal(np.asarray(speculated.tokens), np.asarray(walked.tokens))
    np.testing.assert_array_equal(np.asarray(speculated.lengths), 6)


def test_a_draft_layer_norms_its_stream_before_rounding_it():
    """A draft layer's add and norm (`added_norm`, FlashInfer's `fused_add_rmsnorm`, as vLLM's draft runs
    it) norms the stream's sum before it rounds: a bf16 stream `[1, 1]` plus `[2^-8, -2^-9]` rounds to `[1,
    1]`, whose norm is `[1, 1]`, where the unrounded sum's second part normalizes below one by more than a
    bf16 step's half."""
    from dew.nn.block_draft import added_norm
    from dew.nn.norms import RMSNorm

    norm = RMSNorm(epsilon=1e-6, dtype=jnp.bfloat16)
    residual = jnp.ones((1, 2), jnp.bfloat16)
    x = jnp.asarray([[2.0 ** -8, -2.0 ** -9]], jnp.bfloat16)
    params = norm.init(jax.random.key(0), residual)
    stream, normed = norm.apply(params, x, method=lambda module, x: added_norm(module, x, residual))
    np.testing.assert_array_equal(np.asarray(stream, np.float32), [[1.0, 1.0]])
    assert float(normed[0, 1]) == 1.0 - 2.0 ** -8 and float(normed[0, 0]) == 1.0, normed


TINY_DRAFT = MarkovBlockDraft(layers=1, num_heads=1, num_kv_heads=1, head_dim=2, mlp_features=2, block_size=3,
                              taps=(0, 1, 2), markov_rank=1, rope_theta=10000.0, norm_eps=1e-6)
"""A draft of one layer over two-wide states and three taps, small enough to set its weights by hand."""


def tiny_drafter() -> BlockDrafter:
    return BlockDrafter(TINY_DRAFT, emb_features=2, vocab_rows=4, kv_cache=KVCache(), max_seq_len=8,
                        dtype=jnp.bfloat16)


def test_a_drafts_taps_mix_by_their_weights_rounded():
    """The taps' mixing weights round to the draft's dtype after their softmax, as vLLM's
    (`refresh_derived_weights`): weights `softmax([0, 0, log 3])` over taps whose first coordinates are `[3,
    0, -1]` mix it to `3 (0.2) - 0.6`, zero in fp32 and `-2^-10` with the bf16 weights `0.2002` and
    `0.6016`, which the Markov head's carried taps keep, normed, clearly below zero."""
    drafter = tiny_drafter()
    taps = jnp.asarray([[[3.0, 1.0, 0.0, 1.0, -1.0, 1.0]]], jnp.bfloat16)
    params = drafter.init(jax.random.key(0), taps, None, 1, decode=False, method='contexts')['params']
    params = {**params, 'fuse_weight': jnp.zeros((2, 3)).at[:, 2].set(np.log(3.0))}
    _, carried = drafter.apply({'params': params}, taps, None, 1, decode=False, method='contexts')
    assert float(carried[0, 0]) < -1e-3, carried


def test_a_drafts_attention_weighs_its_values_unrounded():
    """A draft layer's attention weighs its values by its probabilities unrounded and rounds its output once,
    as vLLM's: three keys of equal scores (its queries zero) over values `1, 1, 1 + 2^-7` weigh to `1 +
    2^-7 / 3`, which rounds to 1 in bf16, where probabilities rounded to `0.334` weigh them to `1 + 2^-7`."""
    attention = DraftAttention(TINY_DRAFT, emb_features=2, local=False, kv_cache=KVCache(), max_seq_len=8,
                               dtype=jnp.bfloat16)
    one = 1.0 + 2.0 ** -7
    context = Context(jnp.zeros((1, 2, 1, 2), jnp.bfloat16),
                      jnp.asarray([[[[1.0, 1.0]], [[one, one]]]], jnp.bfloat16),
                      jnp.asarray([[0, 1]]), jnp.ones((1, 2), bool), jnp.asarray([2]))
    x = jnp.asarray([[[1.0, 0.0]]], jnp.bfloat16)
    params = attention.init(jax.random.key(0), x, context)['params']
    params = {**params, 'q_proj': {'kernel': jnp.zeros((2, 2))},
              'v_proj': {'kernel': jnp.asarray([[1.0, 1.0], [0.0, 0.0]])},
              'o_proj': {'kernel': jnp.eye(2)}}
    out = attention.apply({'params': params}, x, context)
    np.testing.assert_array_equal(np.asarray(out, np.float32), [[[1.0, 1.0]]])


def test_a_drafts_head_scores_its_states_and_its_correction_apart():
    """A drafted place's logits are the head's scores of its states and of the Markov head's correction,
    each rounded to the draft's dtype, summed and rounded again (vLLM's `markov_logits`), never the scores
    of their sum: every score the head gives (`scores_of`) is of a place's states or a correction alone,
    and each place's logits are its two scores' rounded sum."""
    drafter = tiny_drafter()
    taps = jnp.ones((1, 4, 6), jnp.bfloat16)
    anchor = jnp.zeros((1,), jnp.int32)
    embedded = jnp.ones((1, 2), jnp.bfloat16)
    row = jnp.asarray([[1.0], [1.0]], jnp.bfloat16)
    scored = []

    def scores_of(x):
        out = (x @ row).astype(jnp.float32)
        scored.append((x, out))
        return out

    def drafting(module):
        return draft(module, taps, None, anchor, lambda tokens: embedded, scores_of, None, decode=False)

    variables = drafter.init(jax.random.key(0), method=drafting)
    scored.clear()
    drafted_block = drafter.apply(variables, method=drafting)
    assert isinstance(drafted_block, Draft)
    logits = drafted_block.logits
    (states, base), *corrections = scored
    places = TINY_DRAFT.block_size - 1
    assert states.shape == (1, places, 2) and len(corrections) == places

    def rounded(x):
        return x.astype(jnp.bfloat16).astype(jnp.float32)

    for index, (_, correction) in enumerate(corrections):
        np.testing.assert_array_equal(np.asarray(logits[:, index]),
                                      np.asarray(rounded(rounded(base[:, index]) + rounded(correction))))


def test_a_markov_draft_through_a_head_not_linear_in_its_states_is_refused():
    """A Markov block drafter sums the head's scores of a place's states and of its correction, which a head
    bias would count twice: a decoder with one is refused as it is built; the fixture's builds."""
    model = translate_config(RELEASE).value
    tokens = jnp.zeros((1, 4), jnp.int32)
    jax.eval_shape(lambda: model.init(jax.random.key(0), tokens))
    with pytest.raises(ValueError):
        jax.eval_shape(lambda: dataclasses.replace(model, head_bias=True).init(jax.random.key(0), tokens))


def built(taps):
    """The fixture's decoder with its draft tapping `taps`, built abstractly (its setup's checks run)."""
    draft = {**RELEASE['mtp_config'], 'target_layer_ids': taps}
    model = translate_config({**RELEASE, 'mtp_config': draft}).value
    return jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))


@pytest.mark.parametrize('taps', [[1, 7, 13, 15], [1, 7, 13, 17]], ids=['short', 'past'])
def test_taps_past_the_targets_visits_are_refused(taps):
    """A draft whose last tap is not its target's count of visits (a looped layer's each pass: 16 for the
    fixture's 2 + 4 x 3 + 2), which stands for its final states, is refused as the model is built; the
    fixture's own taps build."""
    built(RELEASE['mtp_config']['target_layer_ids'])
    with pytest.raises(ValueError):
        built(taps)


def test_a_resharded_release_keeps_its_weights_private(tmp_path):
    """A release whose weights a user made private (`0600`), re-exported in shards: each shard takes the
    access the weights it replaces allowed, the earlier single file's (train 1407's review: the shards
    took the umask's, readable by others)."""
    import stat

    from dew.interop.releases import Placed, ReleasePlan

    weight = np.arange(4096, dtype=np.float32).reshape(64, 64)
    plan = ReleasePlan(Placed({'': ({'model_type': 'tiny'}, {'model.a': weight, 'model.b': weight + 1})}))
    target = tmp_path / 'release'
    plan.write(target)
    (target / 'model.safetensors').chmod(0o600)
    plan.write(target, max_shard_size=20000)
    shards = sorted(target.glob('model-*.safetensors'))
    assert len(shards) > 1
    for shard in shards:
        assert stat.S_IMODE(shard.stat().st_mode) == 0o600, shard


def _scaled(scale):
    """A tiny release of two 64 x 64 weights, both moved by `scale`."""
    from dew.interop.releases import Placed, ReleasePlan

    weight = np.arange(4096, dtype=np.float32).reshape(64, 64)
    return ReleasePlan(Placed({'': ({'model_type': 'tiny'}, {'model.a': weight * scale,
                                                             'model.b': weight + scale})}))


def test_two_first_saves_written_together_publish_one_under_its_names(tmp_path):
    """Two saves to a destination not yet made, written together, both name their shards as their library
    does: the one that publishes first keeps them, and the other is refused rather than put in its place
    under the same names a reader holding the first's index opens (train 1411's review)."""
    import os

    from dew._files import swapped
    from dew.interop.releases import ConcurrentRelease, refused_entries

    target = tmp_path / 'release'
    with pytest.raises(ConcurrentRelease), swapped(target, refused_entries) as staging:
        # The same release's files, named while the destination holds nothing (its staging written in
        # place), then the other save publishes, then this one commits.
        _scaled(2).write(staging, max_shard_size=20000, fresh=True)
        _scaled(1).write(target, max_shard_size=20000)
    assert not os.path.lexists(staging)
    shards = sorted(path.name for path in target.glob('model-*.safetensors'))
    assert len(shards) > 1 and all(name.startswith('model-0') for name in shards), shards


def test_a_first_save_a_third_replaced_between_is_refused_its_names(tmp_path):
    """A save begun while the destination held nothing, which names its shards as a first save does, is
    refused where another save published there and a third replaced that one while it was written: a reader
    holding the first published save's index opens none of its shards under the names it reads, but finds
    them gone, the third save's values published (train 1414's review: the shards were compared with the
    third save's names alone, and the reader silently read this save's)."""
    import json
    import os

    from safetensors.numpy import load_file

    from dew._files import swapped
    from dew.interop.releases import ConcurrentRelease, refused_entries

    weight = np.arange(4096, dtype=np.float32).reshape(64, 64)
    target = tmp_path / 'release'

    with pytest.raises(ConcurrentRelease), swapped(target, refused_entries) as staging:
        _scaled(3).write(staging, max_shard_size=20000, fresh=True)
        _scaled(1).write(target, max_shard_size=20000)
        # A reader opens the first published save's index, then a third save replaces that save.
        first = json.loads((target / 'model.safetensors.index.json').read_text())['weight_map']
        _scaled(2).write(target, max_shard_size=20000)
    assert not os.path.lexists(staging)
    assert not any((target / name).exists() for name in set(first.values()))
    index = json.loads((target / 'model.safetensors.index.json').read_text())['weight_map']
    published = {name: value for file in set(index.values())
                 for name, value in load_file(target / file).items()}
    np.testing.assert_array_equal(published['model.a'], weight * 2)
    np.testing.assert_array_equal(published['model.b'], weight + 2)


def test_a_release_whose_index_alone_is_left_saves_again_in_shards(tmp_path):
    """A release whose shards are gone but whose index is left saves again in shards, named apart from what
    that index names, rather than refused as another save's on every attempt (train 1414's review: an
    index alone was weights to the commit but none to the naming)."""
    import json

    from safetensors.numpy import load_file

    weight = np.arange(4096, dtype=np.float32).reshape(64, 64)
    target = tmp_path / 'release'

    _scaled(1).write(target, max_shard_size=20000)
    named = set(json.loads((target / 'model.safetensors.index.json').read_text())['weight_map'].values())
    for name in named:
        (target / name).unlink()
    _scaled(2).write(target, max_shard_size=20000)
    index = json.loads((target / 'model.safetensors.index.json').read_text())['weight_map']
    assert not named & set(index.values())
    published = {name: value for file in set(index.values())
                 for name, value in load_file(target / file).items()}
    np.testing.assert_array_equal(published['model.a'], weight * 2)
