"""The prompt, preference, chat-canvas and clip presets, each the public steps written out."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from test_text_presets import same_batches

from dew.data import Loading, Memory, presets
from dew.data.compile import compile
from dew.data.dataset import DataPartition

CHAT = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
LOADING = Loading(workers=0, threads=1, read_buffer=2)


def test_the_preferences_preset_lays_out_pairs_as_its_step_written_out():
    """Chosen and rejected token ids with their completion masks, laid out in six tokens padded
    with 7, read by index: the batches of the pair step written out."""
    from dew.data.preferences import PreferencePair

    pairs = Memory([{"chosen": [1, 2, 3, 4][: 2 + index % 3], "rejected": [1, 2, 5][: 2 + index % 2],
                     "chosen_mask": [0, 0, 1, 1][: 2 + index % 3],
                     "rejected_mask": [0, 0, 1][: 2 + index % 2]}
                    for index in range(12)])
    written = compile(pairs.shuffle(0).repeat().map(PreferencePair(6, pad_id=7)).batch(4), read="indexed",
                      loading=LOADING)
    chain = pairs.shuffle(0).repeat().pipe(presets.preferences, seq_len=6, batch=4, pad_id=7)
    same_batches(written, compile(chain, read="indexed", loading=LOADING), 3)


def test_the_canvas_preset_lays_out_turns_as_its_steps_written_out():
    """Conversations rendered by a chat template, each assistant turn one row of a prompt of 24
    tokens and two canvases of 8, ended by the tokenizer's EOS, read by index: the batches of the
    rendering and the canvas step written out."""
    from dew.data import chat
    from dew.data.steps.chat import ResponseCanvases

    conversations = Memory([{"talk": [{"role": "user", "content": "say " + "more " * index},
                                      {"role": "assistant", "content": "done " * (1 + index % 5)}]}
                            for index in range(12)])
    renderer = chat.template(str(CHAT), messages="talk")
    eos = renderer.vocabulary.eos_id
    assert eos is not None
    steps = conversations.shuffle(0).repeat().map(renderer).flat_map(ResponseCanvases(24, 8, 2, eos, 0))
    written = compile(steps.batch(2), read="indexed", loading=LOADING)
    chain = conversations.shuffle(0).repeat().pipe(presets.canvas, str(CHAT), prompt_length=24, canvas_size=8,
                                                   batch=2, canvases=2, messages="talk")
    same_batches(written, compile(chain, read="indexed", loading=LOADING), 3)


def test_a_canvas_needs_a_tokenizer_that_ends_a_response():
    """The chat tokenizer without its EOS token cannot end a response's canvas, as the recipe says;
    with it, the same conversation lays out."""
    from transformers import AutoTokenizer

    conversations = Memory([{"messages": [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "ok"}]}])
    rows = list(conversations.pipe(presets.canvas, str(CHAT), prompt_length=8, canvas_size=4, batch=1))
    assert len(rows) == 1
    endless = AutoTokenizer.from_pretrained(str(CHAT))
    endless.eos_token = None
    with pytest.raises(ValueError):
        conversations.pipe(presets.canvas, endless, prompt_length=8, canvas_size=4, batch=1)


def test_the_video_preset_reads_a_directorys_media_and_passes_over_the_rest(tmp_path):
    """A directory of clips and a note beside them, listed and read through the video preset: an
    epoch of the six clips, the note passed over, and facts saying the frames and the sound's
    length; read once to validate on, every clip."""
    from test_data_av import _mux

    from dew.data import Files
    from dew.data.compile import held
    from dew.data.sources.av_utils import FPS

    (tmp_path / "train").mkdir()
    for index in range(6):
        _mux(tmp_path / "train" / f"{index}.avi")
    (tmp_path / "train" / "notes.txt").write_text("not a clip")
    listing = Files(str(tmp_path / "train"), format="files")
    settings = {"frames": 4, "fps": FPS, "size": 16, "batch": 2, "caption": "a speaker", "sample_rate": 16000,
                "margin": 1 / FPS, "context": 1 / FPS, "missing": "reject", "rejects": False}
    chain = listing.shuffle(0).repeat().pipe(presets.video, 4 / FPS, **settings)
    facts = chain.facts()
    assert facts["fields"]["video"] == [4, 16, 16, 3]
    assert facts["audio_seconds"] == 6 / FPS
    built = compile(chain, read="indexed", loading=LOADING)
    # The listing is cut to its media: six clips, two a batch, an epoch of three batches.
    assert built.records == 6
    assert len(_epoch(built, 3)) == 6
    once = listing.pipe(presets.video, 4 / FPS, **settings)
    assert sum(len(batch["video"]) for batch in held(once)(DataPartition())) == 6


@pytest.mark.parametrize("missing", ["silence", "reject"])
def test_clips_fill_or_drop_media_without_sound_and_drop_or_stop_on_media_too_short(tmp_path, missing):
    """A clip with no sound beside one with sound: filled with silence marked invalid, or dropped and
    counted; a clip shorter than the window is dropped and counted, or stops the run with
    `rejects=False`."""
    from test_data_av import _mux

    from dew.data.sources.av_utils import FPS

    voiced, silent = _mux(tmp_path / "voiced.avi"), _mux(tmp_path / "silent.avi", sound=0)
    short = _mux(tmp_path / "short.avi", frames=2, sound=2)
    media = Memory([{"mp4": voiced}, {"mp4": silent}, {"mp4": short}])
    rows = list(media.pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=1, sample_rate=16000,
                           missing=missing))
    valid = [bool(np.asarray(row["audio_valid"]).any()) for row in rows]
    assert valid == ([True, False] if missing == "silence" else [True])
    with pytest.raises(ValueError):
        list(media.pipe(presets.clips, 4 / FPS, frames=4, fps=FPS, size=16, batch=1, rejects=False))


def _prompts(count: int, start: int = 0, *, widest_first: bool = False) -> Memory:
    """Prompt rows whose ground truth grows with their index, so the widest is the last, or
    shrinks with it."""
    sizes = [1 + index for index in range(start, start + count)]
    return Memory([{"prompt": "hi " * (1 + index % 3), "ground_truth": "x" * size, "data_source": "rule"}
                   for index, size in enumerate(sizes[::-1] if widest_first else sizes)])


def _widths(loaded, count: int) -> set[int]:
    """The widths the reward columns of `count` training batches are padded to."""
    from dew.data.prompts import TRUTH_KEY

    reader = loaded.train(DataPartition())
    try:
        return {np.asarray(next(reader)[TRUTH_KEY]).shape[-1] for _ in range(count)}
    finally:
        reader.close()


def test_the_prompts_preset_measures_its_reward_width_as_the_width_given_outright():
    """Prompt rows rendered to eight left-padded ids, long ones dropped, their reward columns as wide
    as the widest the store holds, measured as a memory read reads it: an epoch of the eight rows
    kept, ordered after rendering, and the batches of the width given outright."""
    rows = _prompts(12)

    def loaded(**options):
        rendered = rows.pipe(presets.prompts, "byte", max_prompt_len=8, batch=None, long="drop", **options)
        return compile(rendered.shuffle(0).repeat().batch(2), read="memory", loading=LOADING)

    built = loaded()
    # Eight of the twelve prompts fit in eight bytes: an epoch is four batches.
    assert built.records == 8 and len(set(_epoch(built, 4))) == 8
    assert _widths(built, 3) == {12}
    same_batches(built, loaded(reward_bytes=12), 3)


def _epoch(loaded, batches: int) -> list[tuple[bytes, ...]]:
    """The rows of a loader's first `batches` training batches, each its fields' bytes, sorted."""
    reader = loaded.train(DataPartition())
    try:
        rows = []
        for _ in range(batches):
            batch = next(reader)
            names = sorted(batch)
            rows += [tuple(np.asarray(batch[name])[row].tobytes() for name in names)
                     for row in range(len(np.asarray(batch[names[0]])))]
        return sorted(rows)
    finally:
        reader.close()


def test_a_measured_reward_width_is_the_whole_stores_in_both_parts_of_a_holdout():
    """A store held out in part, its widest row among the held: the training part and the held part
    bind the width of the widest row of the whole store, so both pad their reward columns alike."""
    from dew.data.prompts import TRUTH_KEY

    train, scored = _prompts(12, widest_first=True).holdout(records=3)

    def rendered(part):
        return part.pipe(presets.prompts, "byte", max_prompt_len=16, batch=None)

    loaded = compile(rendered(train).shuffle(0).repeat().batch(1), validation=rendered(scored).batch(1),
                     read="memory", loading=LOADING)
    assert _widths(loaded, 4) == {12}
    held = loaded.val(DataPartition())
    try:
        assert {np.asarray(batch[TRUTH_KEY]).shape[-1] for batch in held} == {12}
    finally:
        held.close()


def test_a_measured_reward_width_through_a_generation_is_the_whole_stores_in_both_parts(tmp_path):
    """The same holdout read through a generation published from the whole store: both parts bind
    the width of the widest row of the whole store, as uncached, so a generation standing in for
    the store changes no padded shape."""
    from dew.data.cache import Cache
    from dew.data.compile import publish
    from dew.data.prompts import TRUTH_KEY

    store = _prompts(12, widest_first=True)

    def kept(rows):
        return rows.map(lambda row: row, identity="kept@1")

    generation = publish(kept(store), Cache(str(tmp_path / "cache")), read="memory", loading=LOADING)
    train, scored = kept(store).cache(generation).holdout(records=3)

    def rendered(part):
        return part.pipe(presets.prompts, "byte", max_prompt_len=16, batch=None)

    loaded = compile(rendered(train).shuffle(0).repeat().batch(1), validation=rendered(scored).batch(1),
                     read="memory", loading=LOADING)
    assert _widths(loaded, 4) == {12}
    held = loaded.val(DataPartition())
    try:
        assert {np.asarray(batch[TRUTH_KEY]).shape[-1] for batch in held} == {12}
    finally:
        held.close()


def test_a_mixtures_prompt_step_binds_the_widest_of_its_stores():
    """The prompts preset over a mixture of two stores, one with wider reward strings: the step the
    mixture runs binds the wider store's width, so every row of either fits."""
    from dew.data import mix

    narrow, wide = _prompts(4), _prompts(4, start=20)
    chain = mix([narrow.repeat(), wide.repeat()]).pipe(presets.prompts, "byte", max_prompt_len=16, batch=2)
    assert _widths(compile(chain, read="memory", loading=LOADING), 4) == {24}


def test_a_prompt_width_neither_given_nor_stated_is_refused_where_the_read_cannot_measure_it():
    """An indexed read holds no store's records to measure: the preset is refused until the width
    is given, and reads with it."""
    rows = _prompts(4)
    with pytest.raises(ValueError):
        compile(rows.repeat().pipe(presets.prompts, "byte", max_prompt_len=16, batch=2), read="indexed",
                loading=LOADING)
    given = rows.repeat().pipe(presets.prompts, "byte", max_prompt_len=16, batch=2, reward_bytes=4)
    assert _widths(compile(given, read="indexed", loading=LOADING), 2) == {4}


def test_a_measured_reward_width_reads_the_rows_the_steps_before_the_preset_make():
    """Rows whose prompt and answer fields are renamed before the preset: the measured width is the
    widest answer the renamed rows hold, which every row then fits."""
    rows = Memory([{"q": "hi", "a": "an answer " * (1 + index % 2)} for index in range(4)])
    renamed = rows.rename_columns({"q": "prompt", "a": "ground_truth"})
    chain = renamed.repeat().pipe(presets.prompts, "byte", max_prompt_len=8, batch=2)
    assert _widths(compile(chain, read="memory", loading=LOADING), 2) == {len("an answer " * 2)}


@dataclasses.dataclass(frozen=True)
class Shouted:
    """A random step of the caller's: its rows are known only as the run reads them."""

    def random_map(self, row, rng):
        return {**row, "ground_truth": str(row["ground_truth"]).upper()}


def test_a_reward_width_after_a_random_step_is_refused_until_given():
    """A random step before the preset makes rows no measure before the run can read: the width is
    refused unmeasured, and reads given."""
    import grain.python as grain

    class Upper(Shouted, grain.RandomMapTransform):
        pass

    shouted = _prompts(4).random_map(Upper())
    with pytest.raises(ValueError):
        compile(shouted.repeat().pipe(presets.prompts, "byte", max_prompt_len=16, batch=2), read="memory",
                loading=LOADING)
    given = shouted.repeat().pipe(presets.prompts, "byte", max_prompt_len=16, batch=2, reward_bytes=8)
    assert _widths(compile(given, read="memory", loading=LOADING), 2) == {8}
